

# Note: compare to an optimization based algorith vs ( Sample based algorithm)
# calculate the time that is required to achieve each goal, execute number of iterations and average and/or report
# success rate ()
#

import numpy as np
import random 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class JPlusRRT:
    def __init__(self, robot, goal_direction_probability=0.5,with_visualization=False):
        self.robot = robot
        self.tree = []
        self.goal_direction_probability = goal_direction_probability
        self.goal = None #goal is a numpy array [x, y, z] of the goal position
        self.with_visualization = with_visualization

        if with_visualization:
            # Initialize plot
            plt.ion()
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111, projection='3d')
            self.ax.set_xlim(-1, 1)
            self.ax.set_ylim(-1, 1)
            self.ax.set_zlim(0, 1)

    

    def plan(self, start_pos, goal_pos):
        self.goal = goal_pos
        self.start_pos = start_pos

        # Start position is now assumed to be set directly in the robot, 
        # so we start planning from the robot's current state

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

            if self.with_visualization:
                self.visualize_tree()
            
        return self.reconstruct_path()


    def nearest_neighbor(self, target_ee_pos):
        """Find the nearest node in the tree to q_rand."""
        closest_distance = np.inf
        closest_index = None
        for i, node in enumerate(self.tree):
            # distance = np.linalg.norm(node['config'] - q_rand)

            # change: we dont need to calculate distance each time, because q_near is same 
            distance = np.linalg.norm(node['ee_pos'] - target_ee_pos)
            if distance < closest_distance:
                closest_distance = distance
                closest_index = i
        return closest_index
    
    # def nearest_neighbor(self, q_rand):
    #     """Find the nearest node in the tree to q_rand."""
    #     closest_distance = np.inf
    #     closest_index = None
    #     for i, node in enumerate(self.tree):
    #         distance = np.linalg.norm(node['config'] - q_rand)
    #         if distance < closest_distance:
    #             closest_distance = distance
    #             closest_index = i
    #     return closest_index
    
    def step_towards(self, q_near, q_rand, step_size=0.05):
        """Take a small step from q_near towards q_rand."""
        direction = q_rand - q_near
        distance = np.linalg.norm(direction) # Euclidean distance 
        if distance <= step_size:
            return q_rand
        else:
            q_new = q_near + (direction / distance) * step_size #  a new position that is exactly one step size closer towards q_rand, without exceeding the step size limit.
            return q_new
    

    def step_towards_with_jacobian(self, q_near, goal_ee_pos, step_size=0.01):
        """
        Takes a step towards the goal using the Jacobian matrix from the current configuration q_near.
        
        :param q_near: The current joint configuration from which to start the step.
        :param goal_ee_pos: The target end-effector position in task space.
        :param step_size: The maximum step size in joint space (rad or meters).
        :return: The new joint configuration after taking the step, or None if the move isn't feasible.
        """
        # Set the robot's joints to q_near to compute the current end-effector position and Jacobian
        self.robot.reset_joint_pos(q_near)
        current_ee_pos = self.robot.ee_position()
        J = self.robot.get_jacobian()

        # Calculate the task-space error (direction vector)
        direction_vector = np.array(goal_ee_pos) - current_ee_pos
        if np.linalg.norm(direction_vector) < step_size:
            return q_near  # If already close enough, return the current configuration

        # Normalize the direction vector and scale by step size
        direction_vector = (direction_vector / np.linalg.norm(direction_vector)) * step_size

        # Calculate the desired joint velocities using the pseudoinverse of the Jacobian
        pseudo_inverse_J = np.linalg.pinv(J)
        joint_velocities = np.dot(pseudo_inverse_J, direction_vector)

        # Compute the new joint positions
        q_new = np.array(q_near) + joint_velocities

        # Apply joint limits
        lower_limits, upper_limits = self.robot.joint_limits()
        q_new = np.clip(q_new, lower_limits, upper_limits)

        # Optionally, you could check for collisions here and return None if a collision is detected
        self.robot.reset_joint_pos(q_new)
        if self.robot.in_collision():
            return None  # Collision detected, abort this step

        return q_new


    # def random_sample(self, attempts=100):
    #         lower_limits, upper_limits = self.robot.joint_limits()
    #         for _ in range(attempts):
    #             q_rand = np.random.uniform(lower_limits, upper_limits)
    #             if self.tree:
    #                 nearest_index = self.nearest_neighbor(q_rand)
    #                 q_near = self.tree[nearest_index]['config']
    #                 q_new = self.step_towards(q_near, q_rand)
    #                 self.robot.reset_joint_pos(q_new)
    #                 if not self.robot.in_collision():
    #                     node = {'config': q_new, 'ee_pos': self.robot.ee_position(), 'parent_index': nearest_index}
    #                     self.tree.append(node)
    #                     return True  # Indicate success
    #             else:
    #                 # If the tree is empty, initialize it with the start position
    #                 self.robot.reset_joint_pos(q_rand)
    #                 if not self.robot.in_collision():
    #                     node = {'config': q_rand, 'ee_pos': self.robot.ee_position(), 'parent_index': None}
    #                     self.tree.append(node)
    #                     return True
    #         return False

    def random_sample(self, attempts=100):
        lower_limits, upper_limits = self.robot.joint_limits()
        for _ in range(attempts):
            # Generate a random joint configuration
            q_rand = np.random.uniform(lower_limits, upper_limits)

            self.robot.reset_joint_pos(q_rand)

            if not self.robot.in_collision():
                ee_pos = self.robot.ee_position() 

                # Find the nearest node in the tree to the new end-effector position
                nearest_index = self.nearest_neighbor(ee_pos) if self.tree else None # needs to change: use configuration for nearest neighbor
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

        # change : find the closes end-effector position to the goal
        current_ee_pos = self.robot.ee_position()
        goal_pos = self.goal

        direction_vector = goal_pos - current_ee_pos
        direction_vector /= np.linalg.norm(direction_vector)

        step_size = 0.05
        desired_ee_velocity = direction_vector * step_size

        J = self.robot.get_jacobian()
        J_pseudo_inverse = np.linalg.pinv(J)

        joint_velocities = J_pseudo_inverse.dot(desired_ee_velocity)
        current_joint_positions = self.robot.get_joint_pos()
        new_joint_positions = current_joint_positions + joint_velocities

        lower_limits, upper_limits = self.robot.joint_limits()
        new_joint_positions = np.clip(new_joint_positions, lower_limits, upper_limits)

        # Temporarily set the robot to the new positions to check for collisions
        self.robot.reset_joint_pos(new_joint_positions)
        if self.robot.in_collision():
            return False  # Move results in a collision, revert changes
        else:
            # Successful move towards goal without collision, update the tree
            parent_index = len(self.tree) - 1 if self.tree else None
            # node = {'config': new_joint_positions, 'parent_index': parent_index}
            node = {'config': new_joint_positions, 'ee_pos': self.robot.ee_position(), 'parent_index': parent_index}

            self.tree.append(node)
            return True

    def is_goal_reached(self):
        """
        Checks if the current end effector position is sufficiently close to the goal.
        
        Returns:
            bool: True if the end effector is close to the goal, False otherwise.
        """
        # Get the current position of the end effector
        current_ee_pos = self.robot.ee_position()
        
        # Assuming self.goal is a numpy array [x, y, z] representing the goal position
        goal_pos = self.goal
        
        # Calculate the Euclidean distance between the current end effector position and the goal
        distance_to_goal = np.linalg.norm(current_ee_pos - goal_pos)
        
        # Define a threshold for how close the end effector needs to be to the goal to consider it reached
        threshold = 0.05  # Meters
        
        # Check if the distance to the goal is less than or equal to the threshold
        if distance_to_goal <= threshold:
            return True  # The goal is considered reached
        else:
            return False  # The goal is not reached

    def reconstruct_path(self):
        """
        Reconstructs the path from the goal node back to the start node.
        
        Returns:
            list: The sequence of configurations forming the path from start to goal.
        """
        if not self.tree:
            return []  # Return an empty list if the tree is empty

        path = []
        # Start from the last added node which is assumed to be the goal or closest to the goal
        current_node_index = len(self.tree) - 1
        current_node = self.tree[current_node_index]

        while current_node is not None:
            # Prepend the configuration to the path
            path.insert(0, current_node)
            # Move to the parent node
            parent_index = current_node['parent_index']
            current_node = self.tree[parent_index] if parent_index is not None else None

        if self.with_visualization:
            self.visualize_tree(final=True, path=path) 
        
        return path
    

    def visualize_tree(self, final=False, path=None):

        if not self.with_visualization:
            return
            
        self.ax.clear()
        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)
        self.ax.set_zlim(0, 1)

        # Plot the start and goal positions
        self.ax.scatter(self.start_pos[0], self.start_pos[1], self.start_pos[2], c='yellow', marker='o', s=100)
        self.ax.scatter(self.goal[0], self.goal[1], self.goal[2], c='green', marker='o', s=100)

        for node in self.tree:
            if node['parent_index'] is not None:
                parent_node = self.tree[node['parent_index']]
                self.ax.plot([node['ee_pos'][0], parent_node['ee_pos'][0]], 
                             [node['ee_pos'][1], parent_node['ee_pos'][1]], 
                             [node['ee_pos'][2], parent_node['ee_pos'][2]], 'b-')
                self.ax.scatter([node['ee_pos'][0]], [node['ee_pos'][1]], [node['ee_pos'][2]], c='blue', marker='o')

        if final and path:
            for i in range(len(path) - 1):
                self.ax.plot([path[i]['ee_pos'][0], path[i + 1]['ee_pos'][0]],
                             [path[i]['ee_pos'][1], path[i + 1]['ee_pos'][1]],
                             [path[i]['ee_pos'][2], path[i + 1]['ee_pos'][2]], 'orange', linewidth=2)

        plt.draw()
        plt.pause(0.01)