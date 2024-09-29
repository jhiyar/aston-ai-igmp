from robot import Robot
from JPlusRRT11 import JPlusRRT
from IKRRT import IKRRT
from BIKRRT import BIKRRT
from RRTStar2 import RRTStar
from BIKRRTOptimized import BIKRRTOptimized

import numpy as np
import time  # For adding delays between movements
from goal import Goal
from util import move_to_joint_pos,move_to_ee_pose



if __name__ == '__main__':
    robot = Robot(with_gui=False)
    goal_position = np.array([0.7, 0.0, 0.6])  
    # goal_position = np.array([0.7, 0.3, 0.6]) 
    # goal_position = np.array([0.7, 0.0, 0.2]) #under the table

    for i in range(6):
        goal = Goal(i)
        robot.set_goal(goal)

        
    goal_direction_probability = 0.9


    planner = JPlusRRT(robot, goal_direction_probability=0.9,with_visualization=False)
    # planner = RRTStar(robot, goal_direction_probability=0.5,with_visualization=False)
    # planner = IKRRT(robot, goal_direction_probability=0.9)
    # planner = BIKRRT(robot, goal_direction_probability=0.1,with_visualization=False)
    
    
    # start_position = robot.get_joint_pos()

    start_position = np.array(robot.ee_position())

    
    path = planner.plan(start_position, goal_position)
    
    robot.disconnect()

    if path:
        robot_for_visualization = Robot(with_gui=True)
        for i in range(6):
                goal = Goal(i)
                robot_for_visualization.set_goal(goal)

        # while True:
            
        print("Moving the robot along the found path...")
        for node in path:
            if 'config' in node:  # Ensure 'config' key exists
                joint_positions = node['config']
                # print("Found joint position : " , joint_positions)
                # move_to_joint_pos(robot_for_visualization.robot_id, joint_positions)
                # robot_for_visualization.reset_joint_pos(joint_positions)  # Move the robot to each position in the path
                time.sleep(1)  # Wait a bit to see the movement
        print("Path execution completed. Press Enter to finish.")
        input() 
        
        robot_for_visualization.disconnect()
        
    else:
        print("No path found.")

