import numpy as np
import pybullet as p
import pybullet_data

from goal import Goal


class Robot:
    def __init__(self, with_gui=False):
        """
        Initializes the simulation environment and the robot.
        The robot class offers methods to control the robot, such as resetting the joint positions, using forward
        kinematics, and checking collisions.
        :param with_gui: bool, if True will show the GUI.
        """
        # init simulation
        p.connect(p.GUI if with_gui else p.DIRECT)
        if with_gui:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            look_at = [0, 0, 0.2]
            yaw = 45  # left/right degree
            pitch = -45  # up/down degree
            distance = 1.5  # [m]
            p.resetDebugVisualizerCamera(distance, yaw, pitch, look_at)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        # load objects and robot
        self.plane_id = p.loadURDF('plane.urdf')
        self.robot_id = p.loadURDF('franka_panda/panda.urdf', [0, 0, 0.01], useFixedBase=True)
        collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName='assets/table.obj', meshScale=[0.8]*3)
        self.obstacle_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision_shape_id,
                                             basePosition=[1.2, -0.8, 0],
                                             baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi/2]))

        # robot properties
        self.end_effector_link_id = 11
        self.arm_joint_ids = [0, 1, 2, 3, 4, 5, 6]
        self.home_conf = [-0.017792060227770554, -0.7601235411041661, 0.019782607023391807, -2.342050140544315,
                          0.029840531355804868, 1.5411935298621688, 0.7534486589746342]

        # determine joint limits
        limits = []
        for i in self.arm_joint_ids:
            limits.append(p.getJointInfo(self.robot_id, i)[8:10])
        self._joint_limits = np.asarray(limits)

        # set initial robot configuration
        self.reset_joint_pos(self.home_conf)
        for finger_id in [9, 10]:
            p.resetJointState(self.robot_id, finger_id, 0.04)

    def joint_limits(self):
        """
        Returns the joint limits of the robot.
        :return: tuple of ndarrays, (7,) lower and upper joint limits.
        """
        lower, upper = self._joint_limits[:, 0], self._joint_limits[:, 1]
        return lower, upper

    def reset_joint_pos(self, joint_pos):
        """
        Resets the joint positions of the robot.
        :param joint_pos: list/array of floats, joint positions.
        """
        assert len(joint_pos) == len(self.arm_joint_ids), 'Invalid joint position list.'
        for i, pos in zip(self.arm_joint_ids, joint_pos):
            p.resetJointState(self.robot_id, i, pos)

    def get_joint_pos(self):
        """
        Returns the current joint positions of the robot.
        :return: list of floats, joint positions.
        """
        return np.asarray([p.getJointState(self.robot_id, i)[0] for i in self.arm_joint_ids])

    def ee_position(self):
        """
        Computes the end effector position using forward kinematics.
        :return: ndarray (3,), position of the end effector.
        """
        pos, *_ = p.getLinkState(self.robot_id, self.end_effector_link_id, computeForwardKinematics=True)
        return np.asarray(pos)

    def get_jacobian(self):
        """
        Computes the translational Jacobian matrix for the end effector link based on the robot's current joint config.
        :return: (3, 7) ndarray, translational Jacobian matrix.
        """
        # pybullet also needs the finger joints for calculating the Jacobian, so actually gives a (3, 9) matrix.
        # however, the finger joints do not affect end-effector position, so we do not consider them and only provide
        # the Jacobian for the arm joints, as a (3, 7) matrix.
        zero_vec = [0.0] * (len(self.arm_joint_ids) + 2)
        local_pos = [0.0, 0.0, 0.0]
        joint_pos = list(self.get_joint_pos()) + [0.04, 0.04]
        jac_t, _ = p.calculateJacobian(self.robot_id, self.end_effector_link_id, local_pos, joint_pos,
                                       zero_vec, zero_vec)
        jac_t = np.asarray(jac_t)
        return jac_t[:, :7]

    def in_collision(self):
        """
        Checks if the robot is currently in collision with the environment or itself.
        :return: bool, True if the robot is in collision, False otherwise.
        """
        # check obstacle
        if len(p.getClosestPoints(bodyA=self.robot_id, bodyB=self.obstacle_id, distance=0.0)) > 0:
            # print('in collision with plane')
            return True

        # check plane
        if len(p.getClosestPoints(bodyA=self.robot_id, bodyB=self.plane_id, distance=0.0)) > 0:
            # print('in collision with plane')
            return True

        # check self-collision
        ignore_links = [7, 11]  # they do not have a collision shape
        first_links = [0, 1, 2, 3, 4, 5]  # 6 cannot collide with the fingers due to kinematics

        for first_link in first_links:
            # skip links that are next to each other (supposed to be in contact) plus all the ignore links
            check_links = [link for link in np.arange(first_link + 2, self.end_effector_link_id + 1) if
                           link not in ignore_links]
            for check_link in check_links:
                collision = len(p.getClosestPoints(bodyA=self.robot_id, bodyB=self.robot_id, distance=0.0,
                                                   linkIndexA=first_link, linkIndexB=check_link)) > 0
                if collision:
                    # print(f'collision between link {first_link} and link {check_link}')
                    return True
        return False

    def set_goal(self, goal):
        """
        displays a goal in the visualization
        :param goal: Goal object
        """
        visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=[1.0, 1.0, 0.0, 1.0])
        goal_body_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual_shape, basePosition=list(goal.pos))

    def disconnect(self):
        p.disconnect()
    
    # def inverse_kinematics(self, target_pos, target_orientation=None):
    #     """
    #     Computes the joint positions needed to reach a given target position (and optionally orientation) with the end effector.
        
    #     :param target_pos: A list or ndarray of 3 floats specifying the target position of the end effector in the world frame.
    #     :param target_orientation: An optional list or ndarray of 4 floats (quaternion [x, y, z, w]) specifying the target orientation of the end effector. If None, orientation is ignored in the IK calculation.
    #     :return: ndarray of float values representing the joint positions required to achieve the target end effector pose.
    #     """
    #     # The ikSolver argument is optional and specifies the IK algorithm. ikSolver=0 uses the default Damped Least Squares method.
    #     if target_orientation is None:
    #         # If no orientation is specified, use a neutral orientation for the calculation
    #         target_orientation = p.getQuaternionFromEuler([0, 0, 0])
        
    #     joint_positions = p.calculateInverseKinematics(
    #         self.robot_id,
    #         self.end_effector_link_id,
    #         targetPosition=target_pos,
    #         targetOrientation=target_orientation,
    #         # jointDamping=[0.01] * len(self.arm_joint_ids),  # Optional damping for the IK calculation
    #         lowerLimits=self.joint_limits()[0],
    #         upperLimits=self.joint_limits()[1],
    #         # jointRanges=[2*np.pi]*len(self.arm_joint_ids),  # Assuming all joints can rotate fully for simplicity
    #         restPoses=self.home_conf  # Optional: IK calculation can consider these as preferred positions
    #     )
        
    #     # The returned joint positions include positions for all joints, so extract only the relevant arm joint positions
    #     return np.array(joint_positions[:len(self.arm_joint_ids)])

    def inverse_kinematics(self, target_pos, target_orientation=None):
        """
        Computes the joint positions needed to reach a given target position (and optionally orientation) with the end effector.
        
        :param target_pos: A list or ndarray of 3 floats specifying the target position of the end effector in the world frame.
        :param target_orientation: An optional list or ndarray of 4 floats (quaternion [x, y, z, w]) specifying the target orientation of the end effector. If None, orientation is ignored in the IK calculation.
        :return: ndarray of float values representing the joint positions required to achieve the target end effector pose.
        """

        # orientations = [
        #     p.getQuaternionFromEuler([np.pi, 0, 0]),   # Facing upwards
        #     p.getQuaternionFromEuler([0, 0, 0]),       # Facing downwards
        #     p.getQuaternionFromEuler([np.pi/2, 0, 0]), # Facing front
        #     p.getQuaternionFromEuler([-np.pi/2, 0, 0]) # Facing back
        # ]


        # The ikSolver argument is optional and specifies the IK algorithm. ikSolver=0 uses the default Damped Least Squares method.
        if target_orientation is None:
            # If no orientation is specified, use a neutral orientation for the calculation
            target_orientation = p.getQuaternionFromEuler([0, 0, 0])
        else:
            #check : check the documentation for getQuaternionFromEuler
            target_orientation = p.getQuaternionFromEuler(target_orientation)
        
        
        joint_positions = p.calculateInverseKinematics(
            self.robot_id,
            self.end_effector_link_id,
            targetPosition=target_pos,
            targetOrientation=target_orientation,
            lowerLimits=self.joint_limits()[0],
            upperLimits=self.joint_limits()[1],
            # restPoses=self.home_conf  # Optional: IK calculation can consider these as preferred positions
        )
        
        # The returned joint positions include positions for all joints, so extract only the relevant arm joint positions
        return np.array(joint_positions[:len(self.arm_joint_ids)])
