from util import SCENARIO_IDS
from scenario import Scenario
from RRTStar import RRTStar  
import numpy as np
import time

def main():
    scenario_id = 21
    print(f'********** SCENARIO {scenario_id:03d} **********')
    
    # Load the scenario
    s = Scenario(scenario_id)
    
    # Select the grasp poses
    s.select_n_grasps(60)
    
    # Get the robot and simulation environment
    robot, sim = s.get_robot_and_sim(with_gui=True)
    
    # Get the initial configuration of the robot's joints
    start_config = robot.arm_joints_pos()
    
    # Define the goal position (using the first grasp pose)
    goal_pos = s.grasp_poses[0][:3, 3]

    print(s.grasp_poses[0])
    
    # # Create the RRTStar planner
    # planner = RRTStar(robot)
    
    # # Run the planner to find a path
    # print("Running RRT* planner...")
    # path = planner.plan(start_config, goal_pos)
    
    # # If a path is found, execute it
    # if path:
    #     print("Path found! Executing path ... path length:" , len(path))
    #     for node in path:
    #         # Move the robot to each configuration along the path
    #         robot.move_to(node['config'])
    #         time.sleep(0.2)  # Delay to simulate the robot's motion
        
    #     print("Path execution complete.")
    # else:
    #     print("No path found.")
    
    # # Wait for the user to continue/exit
    # input('Press Enter to exit')

if __name__ == '__main__':
    main()
