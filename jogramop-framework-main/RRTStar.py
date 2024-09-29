import numpy as np
import random
from scipy.spatial import KDTree

class RRTStar:
    def __init__(self, robot, gamma_rrt_star=1.0, eta=0.05, max_iterations=10000, goal_threshold=0.1, goal_bias=0.9):
        self.robot = robot
        self.gamma_rrt_star = gamma_rrt_star
        self.eta = eta
        self.max_iterations = max_iterations
        self.goal_threshold = goal_threshold
        self.goal_bias = goal_bias
        self.node_index = 0
        
        # Tree structure
        self.tree_configs = []  # Store configurations of nodes
        self.tree_kdtree = None  # KDTree for nearest neighbor search
        self.tree = []  # The tree itself with vertices (V) and edges (E)

    def get_goal_config(self, goal_pos):
        for _ in range(10000):
            goal_config = self.robot.inverse_kinematics(goal_pos)
            if goal_config is not None:
                self.robot.reset_arm_joints(goal_config)
                if not self.robot.in_collision():
                    return goal_config
        return None
    
    def plan(self, start_config, goal_pos):
        # Initialize the tree with start node
        T = self.initialize_tree(start_config)
        goal_config = self.get_goal_config(goal_pos)
        if goal_config is None:
            return None  # Exit if goal is not reachable

        for i in range(self.max_iterations):
            # Step 1: Sample a random configuration
            zrand = self.sample(i, goal_config)

            # Step 2: Find nearest node in the tree to the sampled point
            znearest = self.nearest(T, zrand)

            # Step 3: Steer towards the sampled point
            znew = self.steer(znearest, zrand)  # Corrected here: no need for Tnew

            # Step 4: Check if the new configuration is collision-free
            if self.obstacle_free(znew):
                # Step 5: Find near neighbors to rewire
                Znear = self.near(T, znew)

                # Step 6: Choose the best parent (the node with the lowest cost)
                zmin = self.choose_parent(Znear, znearest, znew)

                # Step 7: Insert the new node into the tree
                self.insert_node(zmin, znew, T)

                # Step 8: Rewire the tree by checking neighbors
                self.rewire(T, Znear, zmin, znew)

                # Step 9: Check if the goal is reached
                if self.is_goal_reached(znew, goal_config):
                    return self.reconstruct_path(znew)

        return None

    def initialize_tree(self, start_config):
        start_node = {'id': self.node_index, 'config': start_config, 'cost': 0, 'parent': None}
        self.node_index += 1
        self.tree.append(start_node)
        self.tree_configs.append(start_config)
        self.tree_kdtree = KDTree(self.tree_configs)
        return self.tree

    def sample(self, i, goal_config):
        if random.random() < self.goal_bias:
            return goal_config
        return np.random.uniform(self.robot.arm_joint_limits()[0], self.robot.arm_joint_limits()[1])

    def nearest(self, T, zrand):
        _, nearest_index = self.tree_kdtree.query(zrand)
        return T[nearest_index]

    def steer(self, znearest, zrand):
        direction = zrand - znearest['config']
        distance = np.linalg.norm(direction)
        step = (direction / distance) * min(self.eta, distance)  # Move by a small step size or the entire distance
        znew_config = znearest['config'] + step
        return znew_config  # No need to return T here

    def obstacle_free(self, znew_config):
        self.robot.reset_arm_joints(znew_config)
        return not self.robot.in_collision()

    def near(self, T, znew_config):
        radius = min(self.gamma_rrt_star * (np.log(len(T)) / len(T)) ** (1 / len(znew_config)), self.eta)
        return self.tree_kdtree.query_ball_point(znew_config, radius)

    def choose_parent(self, Znear, znearest, znew_config):
        zmin = znearest
        cmin = znearest['cost'] + np.linalg.norm(znew_config - znearest['config'])

        for znear_index in Znear:
            znear = self.tree[znear_index]
            zsteer, _ = self.steer(znear, znew_config)
            if self.obstacle_free(zsteer):
                cnew = znear['cost'] + np.linalg.norm(zsteer - znear['config'])
                if cnew < cmin:
                    zmin = znear
                    cmin = cnew

        return zmin

    def insert_node(self, zmin, znew_config, T):
        znew = {'id': self.node_index, 'config': znew_config, 'cost': zmin['cost'] + np.linalg.norm(znew_config - zmin['config']), 'parent': zmin['id']}
        self.node_index += 1
        self.tree.append(znew)
        self.tree_configs.append(znew_config)
        self.tree_kdtree = KDTree(self.tree_configs)  # Update KDTree

    def rewire(self, T, Znear, zmin, znew_config):
        for znear_index in Znear:
            znear = self.tree[znear_index]
            zsteer, _ = self.steer(znew_config, znear['config'])
            if self.obstacle_free(zsteer) and zmin['cost'] + np.linalg.norm(zsteer - znew_config) < znear['cost']:
                znear['parent'] = zmin['id']
                znear['cost'] = zmin['cost'] + np.linalg.norm(zsteer - znew_config)

    def is_goal_reached(self, znew_config, goal_config):
        return np.linalg.norm(znew_config - goal_config) < self.goal_threshold

    def reconstruct_path(self, znew):
        path = []
        current_node = znew
        while current_node is not None:
            path.insert(0, current_node)
            current_node = next((node for node in self.tree if node['id'] == current_node['parent']), None)
        return path
