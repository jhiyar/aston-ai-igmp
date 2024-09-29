import numpy as np
import random 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class JPlusRRT:
    def __init__(self, robot, goal_direction_probability=0.5,step_size=0.1, with_visualization=False):
        self.robot = robot
        self.tree = []
        self.goal_direction_probability = goal_direction_probability
        self.goal = None  
        self.step_size = step_size  
        self.with_visualization = with_visualization
        self.closest_node_index = None  # Track the closest node to the goal


        if with_visualization:
            # Initialize plot
            plt.ion()
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111, projection='3d')
            self.ax.set_xlim(-1, 1)
            self.ax.set_ylim(-1, 1)
            self.ax.set_zlim(0, 1)

    def plan(self, start_config, goal_pos):
        self.goal = goal_pos

        self.start_config = start_config

        self.robot.reset_arm_joints(start_config)

        # Add the initial configuration as the first node in the tree
        full_pose = self.robot.end_effector_pose()
        start_ee_pos = full_pose[:3, 3]
        
        initial_node = {'config': start_config, 'ee_pos': start_ee_pos, 'parent_index': None}
        self.tree.append(initial_node)

        self.closest_node_index = 0

        while not self.is_goal_reached():
            if random.random() <= self.goal_direction_probability:
                # Try to move towards the goal and update the robot's state
                print("Moving towards goal")
                success = self.move_towards_goal()
            else:
                # Sample a new position and update the robot's state
                print("Sample a new position")
                success = self.random_sample() is not None

            # After updating the robot's state, check for collisions
            if not success:
                print("Collision detected, searching for another point...")
                continue

            if self.with_visualization:
                self.visualize_tree()
            
        return self.reconstruct_path()

    def nearest_neighbor(self, target_config):
        """Find the nearest node in the tree to the given configuration."""
        closest_distance = np.inf
        closest_index = None
        for i, node in enumerate(self.tree):
            distance = np.linalg.norm(node['config'] - target_config)  # Compare configurations, not ee_pos
            if distance < closest_distance:
                closest_distance = distance
                closest_index = i
        return closest_index

    # def step_towards(self, q_near, q_rand, step_size=0.05):
    #     """Take a small step from q_near towards q_rand."""
    #     direction = q_rand - q_near
    #     distance = np.linalg.norm(direction)  # Euclidean distance 
    #     if distance <= step_size:
    #         return q_rand
    #     else:
    #         q_new = q_near + (direction / distance) * step_size  # a new position that is exactly one step size closer towards q_rand, without exceeding the step size limit.
    #         return q_new

    def step_towards(self, q_near, q_rand):
        """Take a small step from q_near towards q_rand, ensuring EE does not move more than step_size."""
        direction = q_rand - q_near
        distance = np.linalg.norm(direction)
        if distance <= self.step_size:
            return q_rand  # If within step size, return the target position

        # Take a step in joint space
        q_new = q_near + (direction / distance) * self.step_size
        self.robot.reset_arm_joints(q_new)

        # Check the EE movement
        full_pose_near = self.robot.end_effector_pose()
        ee_pos_near = full_pose_near[:3, 3]
        
        self.robot.reset_arm_joints(q_new)
        full_pose_new = self.robot.end_effector_pose()
        ee_pos_new = full_pose_new[:3, 3]

        ee_movement = np.linalg.norm(ee_pos_new - ee_pos_near)

        if ee_movement > self.step_size:
            # Scale down the joint movement to ensure the EE doesn't move more than the step size
            q_new = q_near + (direction / distance) * (self.step_size / ee_movement)

        return q_new

    def random_sample(self, attempts=100):
        lower_limits, upper_limits = self.robot.arm_joint_limits().T
        for _ in range(attempts):
            q_rand = np.random.uniform(lower_limits, upper_limits)
            self.robot.reset_arm_joints(q_rand)

            if not self.robot.in_collision():
                nearest_index = self.nearest_neighbor(q_rand) if self.tree else None
                if nearest_index is not None:
                    q_near = self.tree[nearest_index]['config']
                    q_new = self.step_towards(q_near, q_rand)
                else:
                    q_new = q_rand

                if q_new is not None and not self.robot.in_collision():
                    full_pose = self.robot.end_effector_pose()
                    new_ee_pos = full_pose[:3, 3] 
                    node = {'config': q_new, 'ee_pos': new_ee_pos, 'parent_index': nearest_index}
                    self.tree.append(node)

                    # Update the closest node to the goal if this one is closer
                    if self.closest_node_index is None or np.linalg.norm(new_ee_pos - self.goal) < np.linalg.norm(self.tree[self.closest_node_index]['ee_pos'] - self.goal):
                        self.closest_node_index = len(self.tree) - 1

                    return True

        return False
    
    def move_towards_goal(self):
        if self.closest_node_index is None:
            print("No valid starting node")
            return False  # No valid node to start from

        # Use the configuration from the closest node to the goal
        closest_node = self.tree[self.closest_node_index]
        closest_node = self.tree[-1]
        self.robot.reset_arm_joints(closest_node['config'])

        full_pose = self.robot.end_effector_pose()
        current_ee_pos = full_pose[:3, 3]

        goal_pos = self.goal
        direction_vector = goal_pos - current_ee_pos
        distance_to_goal = np.linalg.norm(direction_vector)

        print("Current EE Position:", current_ee_pos)
        print("Goal Position:", goal_pos)
        print("Direction Vector:", direction_vector)
        print("Distance to Goal:", distance_to_goal)

        if distance_to_goal < self.step_size:
            step_distance = distance_to_goal  # Move directly to the goal if within step size
        else:
            step_distance = self.step_size

        adjustment = np.array([4, 0, 0])  # Adjust this vector based on your robot's coordinate system
        direction_vector = direction_vector + adjustment

        # Normalize the direction vector and scale it to the desired step distance
        unit_direction_vector = direction_vector / distance_to_goal
        desired_ee_velocity = unit_direction_vector * step_distance

        print("Desired EE Velocity:", desired_ee_velocity)

        J = self.robot.get_jacobian()
        print("Jacobian Matrix:", J)

        # Compute the pseudo-inverse of the Jacobian
        try:
            J_pseudo_inverse = np.linalg.pinv(J)
        except np.linalg.LinAlgError as e:
            print("Jacobian pseudo-inverse computation failed:", e)
            return False

        print("Jacobian Pseudo-Inverse:", J_pseudo_inverse)

        # Calculate the required joint velocities
        joint_velocities = J_pseudo_inverse @ desired_ee_velocity
        print("Joint Velocities:", joint_velocities)

        # Limit the joint velocities to avoid instability
        max_joint_velocity = 1  # Maximum allowable joint velocity
        joint_velocity_norm = np.linalg.norm(joint_velocities)
        if joint_velocity_norm > max_joint_velocity:
            joint_velocities = joint_velocities / joint_velocity_norm * max_joint_velocity
            print("Scaled Joint Velocities:", joint_velocities)

        current_joint_positions = closest_node['config']
        new_joint_positions = current_joint_positions + joint_velocities

        print("New Joint Positions:", new_joint_positions)

        # Temporarily set the robot to the new positions to check for collisions
        self.robot.reset_arm_joints(new_joint_positions)
        if self.robot.in_collision():
            print("Collision detected, skipping this node.")
            return False  # Move results in a collision, revert changes
        else:
            # Successful move towards goal without collision, update the tree
            parent_index = self.closest_node_index
            full_pose = self.robot.end_effector_pose()
            new_ee_pos = full_pose[:3, 3]
            node = {'config': new_joint_positions, 'ee_pos': new_ee_pos, 'parent_index': parent_index}

            print("New EE Position:", new_ee_pos)
            print("New node found, adding to the tree", node)

            self.tree.append(node)

            # Update the closest node to the goal if this one is closer
            if np.linalg.norm(new_ee_pos - self.goal) < np.linalg.norm(self.tree[self.closest_node_index]['ee_pos'] - self.goal):
                self.closest_node_index = len(self.tree) - 1

            return True

    # using inverse_kinematics to move towards the goal
    # def move_towards_goal(self):
    #     if self.closest_node_index is None:
    #         print("No valid starting node")
    #         return False  # No valid node to start from

    #     # Use the configuration from the closest node to the goal
    #     closest_node = self.tree[self.closest_node_index]
    #     closest_node = self.tree[-1]
    #     self.robot.reset_arm_joints(closest_node['config'])

    #     full_pose = self.robot.end_effector_pose()
    #     current_ee_pos = full_pose[:3, 3]

    #     goal_pos = self.goal
    #     direction_vector = goal_pos - current_ee_pos
    #     distance_to_goal = np.linalg.norm(direction_vector)

    #     print("Current EE Position:", current_ee_pos)
    #     print("Goal Position:", goal_pos)
    #     print("Direction Vector:", direction_vector)
    #     print("Distance to Goal:", distance_to_goal)

    #     if distance_to_goal < self.step_size:
    #         step_distance = distance_to_goal  # Move directly to the goal if within step size
    #     else:
    #         step_distance = self.step_size

    #     # Normalize the direction vector and scale it to the desired step distance
    #     unit_direction_vector = direction_vector / distance_to_goal
    #     desired_ee_pos = current_ee_pos + unit_direction_vector * step_distance

    #     print("Desired EE Position:", desired_ee_pos)

    #     # Calculate the inverse kinematics to find the joint positions for the desired EE position
    #     desired_ee_orientation = None  # we don't have a specific orientation goal yet
    #     new_joint_positions = self.robot.inverse_kinematics(desired_ee_pos, desired_ee_orientation)

    #     if new_joint_positions is None:
    #         print("Inverse Kinematics failed to find a solution")
    #         return False

    #     print("New Joint Positions:", new_joint_positions)

    #     # Temporarily set the robot to the new positions to check for collisions
    #     self.robot.reset_arm_joints(new_joint_positions)
    #     if self.robot.in_collision():
    #         print("Collision detected, skipping this node.")
    #         return False  # Move results in a collision, revert changes
    #     else:
    #         # Successful move towards goal without collision, update the tree
    #         parent_index = self.closest_node_index
    #         full_pose = self.robot.end_effector_pose()
    #         new_ee_pos = full_pose[:3, 3]
    #         node = {'config': new_joint_positions, 'ee_pos': new_ee_pos, 'parent_index': parent_index}

    #         print("New EE Position:", new_ee_pos)
    #         print("New node found, adding to the tree", node)

    #         self.tree.append(node)

    #         # Update the closest node to the goal if this one is closer
    #         if np.linalg.norm(new_ee_pos - self.goal) < np.linalg.norm(self.tree[self.closest_node_index]['ee_pos'] - self.goal):
    #             self.closest_node_index = len(self.tree) - 1

    #         return True



    def is_goal_reached(self):
        """
        Checks if the current end effector position is sufficiently close to the goal.
        
        Returns:
            bool: True if the end effector is close to the goal, False otherwise.
        """
        full_pose = self.robot.end_effector_pose()
        current_ee_pos = full_pose[:3, 3] 
        goal_pos = self.goal
        distance_to_goal = np.linalg.norm(current_ee_pos - goal_pos)
        threshold = 0.1  # Meters

        return distance_to_goal <= threshold

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
            path.insert(0, current_node)
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
        start_ee_pos = self.tree[0]['ee_pos']  # Use the end-effector position of the first node in the tree
        self.ax.scatter(start_ee_pos[0], start_ee_pos[1], start_ee_pos[2], c='yellow', marker='o', s=100)
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

