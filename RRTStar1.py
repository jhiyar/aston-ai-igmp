import numpy as np
import pybullet as p
import pybullet_data
import random
from scipy.spatial import KDTree

class RRTStar:
    def __init__(self, robot, goal_direction_probability=0.05, step_size=0.1, search_radius=0.05):
        self.robot = robot
        self.tree = []
        self.goal_direction_probability = goal_direction_probability
        self.step_size = step_size
        self.search_radius = search_radius
        self.goal = None  # goal is a numpy array [x, y, z] of the goal position

    def plan(self, start_pos, goal_pos):
        self.goal = goal_pos
        self.tree.append({'config': self.robot.get_joint_pos(), 'ee_pos': start_pos, 'cost': 0, 'parent_index': None})

        while not self.is_goal_reached():
            if random.random() < self.goal_direction_probability:
                success = self.extend_tree(self.goal)
            else:
                success = self.extend_tree(self.random_sample())
            
            
            print(len(self.tree))

            if success:
                self.rewire_tree()

        return self.reconstruct_path()

    def extend_tree(self, target_pos):
        nearest_index = self.nearest_neighbor(target_pos)
        nearest_node = self.tree[nearest_index]

        new_config = self.step_towards(nearest_node['config'], target_pos)
        self.robot.reset_joint_pos(new_config)

        if not self.robot.in_collision():
            new_ee_pos = self.robot.ee_position()
            cost = nearest_node['cost'] + np.linalg.norm(new_ee_pos - nearest_node['ee_pos'])
            node = {'config': new_config, 'ee_pos': new_ee_pos, 'cost': cost, 'parent_index': nearest_index}
            self.tree.append(node)
            print("Added node to tree " + str(node))
            return True
        else:
            print("Collision detected.")

        return False

    def nearest_neighbor(self, target_pos):
        tree_positions = [node['ee_pos'] for node in self.tree]
        tree_kdtree = KDTree(tree_positions)
        _, nearest_index = tree_kdtree.query(target_pos)
        return nearest_index

    def near_neighbors(self, target_pos):
        tree_positions = [node['ee_pos'] for node in self.tree]
        tree_kdtree = KDTree(tree_positions)
        indices = tree_kdtree.query_ball_point(target_pos, self.search_radius)
        return indices

    def step_towards(self, q_near, target_pos):
        direction = target_pos - self.robot.ee_position()
        distance = np.linalg.norm(direction)
        if distance <= self.step_size:
            return self.robot.inverse_kinematics(target_pos)
        else:
            direction = (direction / distance) * self.step_size
            target_pos = self.robot.ee_position() + direction
            return self.robot.inverse_kinematics(target_pos)

    def random_sample(self):
        x = np.random.uniform(-1, 1)
        y = np.random.uniform(-1, 1)
        z = np.random.uniform(0, 1)  # Added z-axis for 3D space
        return np.array([x, y, z])

    def rewire_tree(self):
        for node_index, node in enumerate(self.tree):
            print("rewireing")
            near_indices = self.near_neighbors(node['ee_pos'])
            for near_index in near_indices:
                print("rewireing " + str(near_index))

                neighbor_node = self.tree[near_index]
                new_cost = node['cost'] + np.linalg.norm(neighbor_node['ee_pos'] - node['ee_pos'])
                if new_cost < neighbor_node['cost']:
                    self.robot.reset_joint_pos(node['config'])
                    if not self.robot.in_collision():
                        self.tree[near_index]['parent_index'] = node_index
                        self.tree[near_index]['cost'] = new_cost

    def is_goal_reached(self):
        current_ee_pos = self.tree[-1]['ee_pos']
        goal_pos = self.goal
        distance_to_goal = np.linalg.norm(current_ee_pos - goal_pos)
        threshold = 0.15  # Meters
        print('is_goal_reached  called' + str(distance_to_goal))
        return distance_to_goal <= threshold

    def reconstruct_path(self):
        if not self.tree:
            return []  # Return an empty list if the tree is empty

        path = []
        current_node_index = len(self.tree) - 1
        current_node = self.tree[current_node_index]

        print("reconstruct_path" + str(current_node))

        while current_node is not None:
            path.insert(0, current_node)
            parent_index = current_node['parent_index']
            current_node = self.tree[parent_index] if parent_index is not None else None

        return path