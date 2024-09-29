import numpy as np
import random
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import pybullet as p

class RRTStar:
    def __init__(self, robot, gamma_rrt_star=1.0, eta=0.05, max_iterations=10000, goal_threshold=0.1, goal_bias=0.5, with_visualization=False, visualize_interval=100):
        self.robot = robot
        self.tree = []
        self.gamma_rrt_star = gamma_rrt_star  # scaling parameter
        self.eta = eta  # step size
        self.max_iterations = max_iterations
        self.goal = None
        self.node_index = 0
        self.goal_threshold = goal_threshold
        self.goal_bias = goal_bias
        self.with_visualization = with_visualization
        self.visualize_interval = visualize_interval  # visualize every 'visualize_interval' iterations

        if with_visualization:
            plt.ion()
            self.fig, self.ax = plt.subplots()
            self.ax.set_xlim(-1, 1)
            self.ax.set_ylim(-1, 1)
            self.plot_initialized = False

        # Initialize KDTree for efficient nearest neighbor search
        self.tree_configs = []
        self.tree_kdtree = None

    def plan(self, start_pos, goal_pos):
        self.goal = goal_pos
        goal_config = self.get_goal_config(goal_pos)
        if(goal_config is None):
            return None
        
        start_node = {'id': self.node_index, 'config': self.robot.arm_joints_pos(), 'ee_pos': start_pos, 'cost': 0, 'parent_index': None}
        self.node_index += 1
        V = [start_node]
        E = []

        self.tree_configs.append(start_node['config'])
        self.tree_kdtree = KDTree(self.tree_configs)

        for i in range(self.max_iterations):
            print('iteration ', i)
            if random.random() < self.goal_bias:
                xrand = goal_config
            else:
                xrand = self.random_sample()

            xnearest_index = self.nearest_neighbor(V, xrand)
            xnearest = V[xnearest_index]
            xnew_config = self.steer(xnearest['config'], xrand)

            self.robot.reset_arm_joints(xnew_config)

            # Rewiring the tree
            if not self.robot.in_collision():
                xnew_pos = self.robot.end_effector_pose()  # Get the 4x4 transformation matrix
                if xnew_pos.shape == (4, 4):
                    xnew_pos = xnew_pos[:3, 3]  # Extract the 3D position from the transformation matrix

                xnew_cost = xnearest['cost'] + np.linalg.norm(xnew_config - xnearest['config'])
                xnew = {'id': self.node_index, 'config': xnew_config, 'ee_pos': xnew_pos, 'cost': xnew_cost, 'parent_index': xnearest['id']}
                self.node_index += 1

                # Find near neighbors and rewire
                Xnear_indices = self.near_neighbors(xnew_config)
                V.append(xnew)
                self.tree_configs.append(xnew['config'])
                self.tree_kdtree = KDTree(self.tree_configs)  # Update KDTree with new node

                xmin = xnearest
                cmin = xnew_cost

                for xnear_index in Xnear_indices:
                    xnear = V[xnear_index]
                    new_cost = xnear['cost'] + np.linalg.norm(xnear['config'] - xnew['config'])
                    if new_cost < cmin and not self.robot.in_collision():
                        xmin = xnear
                        cmin = new_cost

                xnew['parent_index'] = xmin['id']
                E.append((xmin['id'], xnew['id']))

                if self.with_visualization and i % self.visualize_interval == 0:
                    self.visualize_tree(V, E, goal_pos)

                if self.is_goal_reached(xnew['config']):
                    print("Goal reached!")
                    self.tree = V
                    return self.reconstruct_path(xnew)

                # Rewiring for lower cost paths
                for xnear_index in Xnear_indices:
                    xnear = V[xnear_index]
                    new_cost = xnew['cost'] + np.linalg.norm(xnew['config'] - xnear['config'])
                    if new_cost < xnear['cost'] and not self.robot.in_collision():
                        xparent_id = xnear['parent_index']
                        E = [(parent, child) for parent, child in E if not (parent == xparent_id and child == xnear['id'])]
                        E.append((xnew['id'], xnear['id']))
                        xnear['parent_index'] = xnew['id']
                        xnear['cost'] = new_cost

        self.tree = V
        return None

    def get_goal_config(self, goal_pos):
        for _ in range(10000):
            goal_config = self.robot.inverse_kinematics(goal_pos)
            if goal_config is not None:
                self.robot.reset_arm_joints(goal_config)
                if not self.robot.in_collision():
                    return goal_config
        return None

    def is_goal_reached(self, config):
        self.robot.reset_arm_joints(config)
        ee_pos = self.robot.end_effector_pose()[:3, 3]  # Extract 3D position
        return np.linalg.norm(ee_pos - self.goal) < self.goal_threshold

    def steer(self, start_config, target_config):
        direction = target_config - start_config
        distance = np.linalg.norm(direction)
        # if distance <= self.eta:
            # return target_config
        # else:
        direction = (direction / distance) * self.eta 
        return start_config + direction

    def nearest_neighbor(self, V, target_config):
        _, nearest_index = self.tree_kdtree.query(target_config)
        return nearest_index

    def near_neighbors(self, target_config):
        card_V = len(self.tree_configs)
        dimension = len(target_config)
        radius = min(self.gamma_rrt_star * (np.log(card_V) / card_V) ** (1 / dimension), self.eta)
        return self.tree_kdtree.query_ball_point(target_config, radius)

    def random_sample(self):
        lower_limits, upper_limits = self.robot.arm_joint_limits().T
        return np.random.uniform(lower_limits, upper_limits)

    def reconstruct_path(self, goal_node):
        path = []
        current_node = goal_node

        while current_node is not None:
            path.insert(0, current_node)
            parent_index = current_node['parent_index']
            current_node = next((node for node in self.tree if node['id'] == parent_index), None)

        if self.with_visualization:
            self.visualize_tree(self.tree, self.tree_edges(self.tree), self.goal, path)
        
        return path

    def tree_edges(self, V):
        E = []
        for node in V:
            if node['parent_index'] is not None:
                parent = next((n for n in V if n['id'] == node['parent_index']), None)
                if parent:
                    E.append((parent['id'], node['id']))
        return E

    def visualize_tree(self, V, E, goal_pos, path=None):
        self.ax.clear()

        for node in V:
            print(f"ee_pos shape: {node['ee_pos'].shape}, ee_pos: {node['ee_pos']}")

        # Assuming ee_pos is a 3D position vector (x, y, z)
        x_coords = []
        y_coords = []

        for node in V:
            ee_pos = node['ee_pos']

            # Check if ee_pos is a transformation matrix (4x4)
            if ee_pos.shape == (4, 4):
                # Extract the translation part (last column, first 3 elements)
                ee_pos = ee_pos[:3, 3]
            
            # Now ee_pos should be a 3D position
            x_coords.append(ee_pos[0])
            y_coords.append(ee_pos[1])

        # Include goal position coordinates
        x_coords.append(float(goal_pos[0]))
        y_coords.append(float(goal_pos[1]))

        margin = 0.1  # Margin for better visualization

        # Set plot limits based on the extents of the tree and goal position
        x_min, x_max = min(x_coords) - margin, max(x_coords) + margin
        y_min, y_max = min(y_coords) - margin, max(y_coords) + margin
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)

        # Plot edges between nodes
        for (parent_id, child_id) in E:
            parent = next(node for node in V if node['id'] == parent_id)
            child = next(node for node in V if node['id'] == child_id)
            self.ax.plot([float(parent['ee_pos'][0]), float(child['ee_pos'][0])],
                        [float(parent['ee_pos'][1]), float(child['ee_pos'][1])], 'k-')

        # Plot nodes as blue dots
        self.ax.plot([float(node['ee_pos'][0]) for node in V], [float(node['ee_pos'][1]) for node in V], 'bo')

        # Plot goal position as a red dot
        self.ax.plot(float(goal_pos[0]), float(goal_pos[1]), 'ro')

        # Plot path if available (yellow line)
        if path:
            for i in range(len(path) - 1):
                self.ax.plot([float(path[i]['ee_pos'][0]), float(path[i + 1]['ee_pos'][0])],
                            [float(path[i]['ee_pos'][1]), float(path[i + 1]['ee_pos'][1])], 'y-', linewidth=2)

        # Set plot labels and title
        self.ax.set_title('RRT* Tree')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        plt.draw()
        plt.pause(0.001)




