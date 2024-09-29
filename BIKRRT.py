import numpy as np
import pybullet as p
import pybullet_data
import random
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import time 

# Bidirectional Inverse Kinematics RRT
class BIKRRT:
    def __init__(self, robot, goal_direction_probability=0.5, with_visualization=False):
        self.robot = robot
        self.start_tree = []
        self.goal_tree = []
        self.goal_direction_probability = goal_direction_probability
        self.goal = None  # goal is a numpy array [x, y, z] of the goal position
        self.with_visualization = with_visualization

        if with_visualization:
            # Initialize plot
            plt.ion()
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111, projection='3d')
            self.ax.set_xlim(-1, 1)
            self.ax.set_ylim(-1, 1)
            self.ax.set_zlim(0, 1)
            self.plot_initialized = False

    def plan(self, start_pos, goal_pos):
        self.goal = goal_pos

        # Initialize both trees with start and goal configurations
        self.start_tree.append({'config': self.robot.get_joint_pos(), 'ee_pos': start_pos, 'parent_index': None})


        # goal_config = self.robot.inverse_kinematics(goal_pos,[0.2, 0, 0])
        # self.goal_tree.append({'config': goal_config, 'ee_pos': goal_pos, 'parent_index': None})

        # Ensure the initial configuration for the goal tree is collision-free
        found_collision_free = False
        for _ in range(10000): 
            # Generate a random orientation
            random_orientation = [random.uniform(-np.pi, np.pi), 0, 0]
            goal_config = self.robot.inverse_kinematics(goal_pos, random_orientation)
            self.robot.reset_joint_pos(goal_config)
            if not self.robot.in_collision():
                self.goal_tree.append({'config': goal_config, 'ee_pos': goal_pos, 'parent_index': None})
                print({'config': goal_config, 'ee_pos': goal_pos, 'parent_index': None})
                print("Found collision-free goal configuration:", goal_config)
                found_collision_free = True
                break
        
        if not found_collision_free:
            raise RuntimeError("Failed to find a collision-free initial configuration for the goal tree.")



        i = 0
        while True:
            i += 1
            print('Iteration %d' % i)

            # Grow the start tree
            if random.random() < self.goal_direction_probability:
                success = self.extend_tree(self.start_tree, self.goal_tree[-1]['ee_pos'])
            else:
                success = self.random_sample(self.start_tree) is not None

            if success and self.check_connection():
                break

            # Grow the goal tree
            if random.random() < self.goal_direction_probability:
                success = self.extend_tree(self.goal_tree, self.start_tree[-1]['ee_pos'])
            else:
                success = self.random_sample(self.goal_tree) is not None

            if success and self.check_connection():
                break

            if self.with_visualization:
                self.visualize_trees()  # Update visualization after each iteration

        return self.reconstruct_path()

    def extend_tree(self, tree, target_pos, step_size=0.1):
        nearest_index = self.nearest_neighbor(tree, target_pos)
        nearest_node = tree[nearest_index]

        new_config = self.step_towards(nearest_node['config'], target_pos, step_size)
        self.robot.reset_joint_pos(new_config)

        if not self.robot.in_collision():
            new_ee_pos = self.robot.ee_position()
            node = {'config': new_config, 'ee_pos': new_ee_pos, 'parent_index': nearest_index}
            tree.append(node)
            return True

        return False

    def nearest_neighbor(self, tree, target_ee_pos):
        """Find the nearest node in the tree to target_ee_pos."""
        closest_distance = np.inf
        closest_index = None
        for i, node in enumerate(tree):
            distance = np.linalg.norm(node['ee_pos'] - target_ee_pos)
            if distance < closest_distance:
                closest_distance = distance
                closest_index = i
        return closest_index

    def step_towards(self, q_near, target_pos, step_size=0.05):
        """Take a small step from q_near towards target_pos."""
        direction = target_pos - self.robot.ee_position()
        distance = np.linalg.norm(direction)  # Euclidean distance
        if distance <= step_size:
            return self.robot.inverse_kinematics(target_pos)
        else:
            direction = (direction / distance) * step_size
            target_pos = self.robot.ee_position() + direction
            return self.robot.inverse_kinematics(target_pos)

    def random_sample(self, tree, attempts=100):
        for _ in range(attempts):
            # Generate a random end-effector position
            x = np.random.uniform(-1, 1)
            y = np.random.uniform(-1, 1)
            z = np.random.uniform(0, 1)  # z-axis for 3D space
            target_ee_pos = np.array([x, y, z])

            try:
                q_rand = self.robot.inverse_kinematics(target_ee_pos)
            except:
                continue

            self.robot.reset_joint_pos(q_rand)

            if not self.robot.in_collision():
                ee_pos = self.robot.ee_position()

                # Find the nearest node in the tree to the new end-effector position
                nearest_index = self.nearest_neighbor(tree, ee_pos)
                q_near = tree[nearest_index]['config']
                q_new = self.step_towards(q_near, target_ee_pos)
                new_ee_pos = self.robot.ee_position()  # Get the new end-effector position after moving

                if q_new is not None and not self.robot.in_collision():
                    node = {'config': q_new, 'ee_pos': new_ee_pos, 'parent_index': nearest_index}
                    tree.append(node)
                    return True  # Indicate success

        return False

    def check_connection(self):
        """Check if the start and goal trees are connected."""
        min_distance = np.inf
        start_connect_node = None
        goal_connect_node = None

        for start_node in self.start_tree:
            for goal_node in self.goal_tree:
                distance = np.linalg.norm(start_node['ee_pos'] - goal_node['ee_pos'])
                if distance < min_distance:
                    min_distance = distance
                    start_connect_node = start_node
                    goal_connect_node = goal_node

        if min_distance < 0.05 and self.check_path(start_connect_node['config'], goal_connect_node['config']):
            self.connection = (start_connect_node, goal_connect_node)
            print("Trees are connected")

            return True
        return False

    def check_path(self, start_config, goal_config, steps=10):
        """Check if a direct path between two configurations is collision-free."""
        for step in range(steps + 1):
            alpha = step / steps
            intermediate_config = (1 - alpha) * np.array(start_config) + alpha * np.array(goal_config)
            self.robot.reset_joint_pos(intermediate_config)
            if self.robot.in_collision():
                return False
        return True

    def reconstruct_path(self):
        """Reconstruct the path from start to goal."""
        if not self.start_tree or not self.goal_tree:
            return []  # Return an empty list if either tree is empty

        path = []

        # Traverse from the connection point back to the start
        node = self.connection[0]
        while node is not None:
            path.insert(0, node)

            parent_index = node['parent_index']
            node = self.start_tree[parent_index] if parent_index is not None else None

        # Traverse from the connection point to the goal
        node = self.connection[1]
        while node is not None:
            path.append(node)
            parent_index = node['parent_index']
            node = self.goal_tree[parent_index] if parent_index is not None else None

        if self.with_visualization:
            self.visualize_trees(path)  # Visualize the final path
        
        return path

    def visualize_trees(self, path=None):
        self.ax.clear()
        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)
        self.ax.set_zlim(0, 1)

        for tree, color in [(self.start_tree, 'b'), (self.goal_tree, 'r')]:
            for node in tree:
                if node['parent_index'] is not None:
                    parent_node = tree[node['parent_index']]
                    self.ax.plot([node['ee_pos'][0], parent_node['ee_pos'][0]], 
                                 [node['ee_pos'][1], parent_node['ee_pos'][1]], 
                                 [node['ee_pos'][2], parent_node['ee_pos'][2]], color + '-')
                    self.ax.plot([node['ee_pos'][0]], [node['ee_pos'][1]], [node['ee_pos'][2]], color + 'o')

        # Plot the start and goal positions
        self.ax.plot([self.start_tree[0]['ee_pos'][0]], [self.start_tree[0]['ee_pos'][1]], [self.start_tree[0]['ee_pos'][2]], 'yo', markersize=10)  # Start in yellow
        self.ax.plot([self.goal_tree[0]['ee_pos'][0]], [self.goal_tree[0]['ee_pos'][1]], [self.goal_tree[0]['ee_pos'][2]], 'go', markersize=10)  # Goal in green

        # Draw the final path in orange
        if path:
            for i in range(len(path) - 1):
                self.ax.plot([path[i]['ee_pos'][0], path[i + 1]['ee_pos'][0]],
                             [path[i]['ee_pos'][1], path[i + 1]['ee_pos'][1]],
                             [path[i]['ee_pos'][2], path[i + 1]['ee_pos'][2]], 'orange', linewidth=2)

        self.ax.set_title('BIKRRT Tree')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        plt.draw()
        plt.pause(0.01)
