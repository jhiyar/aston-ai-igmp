from util import SCENARIO_IDS
from scenario import Scenario
from RRTStar import RRTStar
import numpy as np
import time

def run_scenario(scenario_id, n_runs=100):
    total_time = 0
    total_path_length = 0
    success_count = 0

    for i in range(n_runs):
        print(f'********** SCENARIO {scenario_id:03d} | Run {i + 1} **********')

        # Load the scenario
        s = Scenario(scenario_id)

        # Select the grasp poses
        s.select_n_grasps(60)

        # Get the robot and simulation environment
        robot, sim = s.get_robot_and_sim(with_gui=False)  # No GUI for performance

        # Get the initial configuration of the robot's joints
        start_config = robot.arm_joints_pos()

        # Define the goal position (using the first grasp pose)
        goal_pos = s.grasp_poses[0][:3, 3]

        # Create the RRTStar planner
        planner = RRTStar(robot, goal_bias=0.1, with_visualization=False)

        # Time the planning process
        start_time = time.time()
        path = planner.plan(start_config, goal_pos)
        end_time = time.time()

        # Calculate time taken
        elapsed_time = end_time - start_time
        total_time += elapsed_time

        if path:
            # Success, count this run
            success_count += 1
            total_path_length += len(path)

            print(f"Path found! Path length: {len(path)}")
        else:
            # Path not found, continue to the next run
            print("No path found.")

    # Calculate averages
    avg_time = total_time / n_runs
    avg_path_length = total_path_length / success_count if success_count > 0 else 0
    success_rate = (success_count / n_runs) * 100

    # Output results
    print(f"\nScenario {scenario_id:03d} results after {n_runs} runs:")
    print(f"Average Time: {avg_time:.4f} seconds")
    print(f"Average Path Length: {avg_path_length:.2f}")
    print(f"Success Rate: {success_rate:.2f}%\n")

    log_results_to_file(scenario_id, n_runs, avg_time, avg_path_length, success_rate)


def log_results_to_file(scenario_id, n_runs, avg_time, avg_path_length, success_rate, filename="scenario_results.txt"):
    with open(filename, "a") as f:  # Open the file in append mode
        f.write(f"\nScenario {scenario_id:03d} results after {n_runs} runs:\n")
        f.write(f"Average Time: {avg_time:.4f} seconds\n")
        f.write(f"Average Path Length: {avg_path_length:.2f}\n")
        f.write(f"Success Rate: {success_rate:.2f}%\n\n")

def run_all_scenarios(n_runs=10):
    for scenario_id in SCENARIO_IDS:
        run_scenario(scenario_id, n_runs)

if __name__ == '__main__':
    run_all_scenarios()
