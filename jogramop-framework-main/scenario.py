import copy
import os
import numpy as np
import open3d as o3d
import burg_toolkit as burg

import simulation
from simulation import GraspingSimulator, FrankaRobot
import util


class Scenario:
    def __init__(self, scenario_id, robot_pose=None, with_platform=False):
        self.id = scenario_id
        self.scene_fn = os.path.join(util.SCENARIO_DIR, f'{scenario_id:03d}', 'scene.yaml')
        self.scene, lib, _ = burg.Scene.from_yaml(self.scene_fn)
        print(f'Loaded scene with {len(self.scene.objects)} objects and {len(self.scene.bg_objects)} obstacles.')

        grasps_fn = os.path.join(util.SCENARIO_DIR, f'{scenario_id:03d}', 'grasps.npy')
        grasps = np.load(grasps_fn, allow_pickle=True)
        grasps = grasps @ FrankaRobot.tf_grasp2ee
        self.gs = burg.GraspSet.from_poses(grasps)
        print(f'Loaded {len(self.gs)} grasps')
        self.select_indices = None

        self.robot_pose = robot_pose or self.default_robot_pose()
        self.with_platform = with_platform

    @property
    def grasp_poses(self):
        if self.select_indices is None:
            return self.gs.poses
        return self.gs.poses[self.select_indices]

    def select_n_grasps(self, n=None, seed=None):
        if n is None:
            print('Erasing grasp selection. Whole grasp set available again.')
            self.select_indices = None
            return

        if n > len(self.gs):
            raise ValueError('Cannot select more grasps than available. n must be smaller than number of grasps')

        rng = np.random.default_rng(seed)
        self.select_indices = rng.choice(len(self.gs), n, replace=False)

    def get_robot_and_sim(self, with_gui=False) -> tuple[simulation.FrankaRobot, simulation.GraspingSimulator]:
        sim = GraspingSimulator(verbose=with_gui)
        sim.add_scene(self.scene)
        robot = FrankaRobot(sim, self.robot_pose, with_platform=self.with_platform)

        return robot, sim

    @staticmethod
    def default_robot_pose():
        pose = np.array([
            [0.0, -1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.05],
            [0.0, 0.0, 0.0, 1.0],
        ])
        return pose

    def get_ik_solutions(self):
        robot, sim = self.get_robot_and_sim()
        joint_limits = robot.arm_joint_limits()

        def in_joint_limits(joint_conf):
            return np.all(joint_limits[:, 0] <= joint_conf) and np.all(joint_conf <= joint_limits[:, 1])

        ik_solutions = []
        count = util.Timer()
        print('Calculating IK solutions for all grasps')
        for g in range(len(self.gs)):
            count.start('calculate IK')
            target_pose = self.gs.poses[g]
            pos, orn = burg.util.position_and_quaternion_from_tf(target_pose, convention='pybullet')

            target_conf = robot.inverse_kinematics(pos, orn, null_space_control=True)
            count.stop('calculate IK')
            count.start('perform checks')
            if target_conf is not None:
                robot.reset_arm_joints(target_conf)

                if not in_joint_limits(target_conf):
                    target_conf = None
                    count.count('IK solution not in joint limits')
                elif robot.in_self_collision():
                    target_conf = None
                    count.count('IK solution in self-collision')
                elif robot.in_collision():
                    target_conf = None
                    count.count('IK solution in collision')
                else:
                    count.count('IK solution is OK')
            else:
                count.count('IK solution too far away from goal')
            count.stop('perform checks')

            if target_conf is not None:
                ik_solutions.append(target_conf)

        # count.print()
        ik_solutions = np.array(ik_solutions)
        return ik_solutions
