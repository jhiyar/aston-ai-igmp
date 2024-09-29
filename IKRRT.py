import numpy as np
import pybullet as p
import pybullet_data
import random

class IKRRT:
    def __init__(self, robot, goal_direction_probability=0.05):
        self.robot = robot
        self.tree = []
        self.goal_direction_probability = goal_direction_probability
        self.goal = None  # goal is a numpy array [x, y, z] of the goal position

    def plan(self, start_pos, goal_pos):
        self.goal = goal_pos
        # Start position is now assumed to be set directly in the robot,
        # so we start planning from the robot's current state
        self.tree.append({'config': self.robot.get_joint_pos(), 'ee_pos': self.robot.ee_position(), 'parent_index': None})

        while not self.is_goal_reached():
            if random.random() < self.goal_direction_probability:
                # Try to move towards the goal and update the robot's state
                success = self.move_towards_goal()
            else:
                # Sample a new position and update the robot's state
                success = self.random_sample() is not None

            # After updating the robot's state, check for collisions without passing new_pos
            if not success or self.robot.in_collision():
                print("Collision detected, searching for another point...")
                continue

        return self.reconstruct_path()

    def nearest_neighbor(self, target_ee_pos):
        """Find the nearest node in the tree to q_rand."""
        closest_distance = np.inf
        closest_index = None
        for i, node in enumerate(self.tree):
            distance = np.linalg.norm(node['ee_pos'] - target_ee_pos)
            if distance < closest_distance:
                closest_distance = distance
                closest_index = i
        return closest_index

    def step_towards(self, q_near, q_rand, step_size=0.1):
        """Take a small step from q_near towards q_rand."""
        direction = q_rand - q_near
        distance = np.linalg.norm(direction)  # Euclidean distance
        if distance <= step_size:
            return q_rand
        else:
            q_new = q_near + (direction / distance) * step_size  # a new position that is exactly one step size closer towards q_rand, without exceeding the step size limit.
            return q_new

    def random_sample(self, attempts=100):
        for _ in range(attempts):
            # Generate a random end-effector position
            x = np.random.uniform(-1, 1)
            y = np.random.uniform(-1, 1)
            z = np.random.uniform(0, 1)  # Added z-axis for 3D space
            target_ee_pos = np.array([x, y, z])

            try:
                q_rand = self.robot.inverse_kinematics(target_ee_pos)
            except:
                continue

            self.robot.reset_joint_pos(q_rand)

            if not self.robot.in_collision():
                ee_pos = self.robot.ee_position()

                # Find the nearest node in the tree to the new end-effector position
                nearest_index = self.nearest_neighbor(ee_pos) if self.tree else None
                if nearest_index is not None:
                    q_near = self.tree[nearest_index]['config']
                    q_new = self.step_towards(q_near, q_rand)
                else:
                    q_new = q_rand  # If tree is empty, start from the random position

                new_ee_pos = self.robot.ee_position()  # Get the new end-effector position after moving
                if q_new is not None and not self.robot.in_collision():
                    node = {'config': q_new, 'ee_pos': new_ee_pos, 'parent_index': nearest_index}
                    self.tree.append(node)
                    return True  # Indicate success

        return False

    def move_towards_goal(self):
        current_ee_pos = self.robot.ee_position()
        goal_pos = self.goal

        direction_vector = goal_pos - current_ee_pos
        direction_vector /= np.linalg.norm(direction_vector)

        step_size = 0.01
        desired_ee_velocity = direction_vector * step_size

        J = self.robot.get_jacobian()
        J_pseudo_inverse = np.linalg.pinv(J)

        joint_velocities = J_pseudo_inverse.dot(desired_ee_velocity)
        current_joint_positions = self.robot.get_joint_pos()
        new_joint_positions = current_joint_positions + joint_velocities

        lower_limits, upper_limits = self.robot.joint_limits()
        new_joint_positions = np.clip(new_joint_positions, lower_limits, upper_limits)

        self.robot.reset_joint_pos(new_joint_positions)
        if self.robot.in_collision():
            return False  # Move results in a collision, abort this step
        else:
            parent_index = len(self.tree) - 1 if self.tree else None
            node = {'config': new_joint_positions, 'ee_pos': self.robot.ee_position(), 'parent_index': parent_index}
            self.tree.append(node)
            return True

    def is_goal_reached(self):
        current_ee_pos = self.robot.ee_position()
        goal_pos = self.goal
        distance_to_goal = np.linalg.norm(current_ee_pos - goal_pos)
        threshold = 0.05  # Meters
        return distance_to_goal <= threshold

    def reconstruct_path(self):
        if not self.tree:
            return []  # Return an empty list if the tree is empty

        path = []
        current_node_index = len(self.tree) - 1
        current_node = self.tree[current_node_index]

        while current_node is not None:
            path.insert(0, current_node)
            parent_index = current_node['parent_index']
            current_node = self.tree[parent_index] if parent_index is not None else None

        return path
