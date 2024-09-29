import time
import numpy as np
import matplotlib.pyplot as plt
from robot import Robot
from goal import Goal

from JPlusRRT11 import JPlusRRT
from IKRRT import IKRRT
from BIKRRT import BIKRRT
from RRTStar2 import RRTStar

# change: add all goals for all three algorithms

def compare_algorithms(algorithms, start_pos, goal_pos, num_runs=10):
    results = {name: {'path_length': [], 'planning_time': [], 'success': []} for name in algorithms.keys()}
    
    for name, algorithm in algorithms.items():
        for run in range(num_runs):
            print(f"Running {name} - Iteration {run + 1}")
            start_time = time.time()
            path = algorithm.plan(start_pos, goal_pos)
            end_time = time.time()
            
            if path:
                path_length = sum(np.linalg.norm(np.array(path[i]['ee_pos']) - np.array(path[i + 1]['ee_pos'])) for i in range(len(path) - 1))
                results[name]['path_length'].append(path_length)
                results[name]['planning_time'].append(end_time - start_time)
                results[name]['success'].append(1)
            else:
                results[name]['success'].append(0)
    
    return results

def plot_results(results):
    names = list(results.keys())
    path_lengths = [np.mean(results[name]['path_length']) for name in names]
    planning_times = [np.mean(results[name]['planning_time']) for name in names]
    success_rates = [np.mean(results[name]['success']) * 100 for name in names]

    x = np.arange(len(names))

    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('Algorithms')
    ax1.set_ylabel('Path Length', color=color)
    ax1.bar(x - 0.2, path_lengths, 0.4, label='Path Length', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:green'
    ax2.set_ylabel('Planning Time (s)', color=color)
    ax2.bar(x + 0.2, planning_times, 0.4, label='Planning Time', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    ax1.set_xticks(x)
    ax1.set_xticklabels(names)
    fig.tight_layout()
    plt.title('Algorithm Comparison')
    plt.show()

    fig, ax = plt.subplots()
    ax.bar(x, success_rates, 0.4, label='Success Rate')
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_xlabel('Algorithms')
    ax.set_ylabel('Success Rate (%)')
    plt.title('Algorithm Success Rate')
    plt.show()
    input() 

if __name__ == '__main__':

    robot = Robot(with_gui=False)
    start_position = np.array(robot.ee_position())
    goal_position = np.array([0.7, 0.0, 0.6]) 

    for i in range(6):
        goal = Goal(i)
        robot.set_goal(goal)
    

    robot = Robot()  


    algorithms = {
        'JPlusRRT': JPlusRRT(robot, goal_direction_probability=0.9),
        'RRTStar': RRTStar(robot, goal_direction_probability=0.9),
        'IKRRT': IKRRT(robot,goal_direction_probability=0.9),
        'BIKRRT': BIKRRT(robot,goal_direction_probability=0.9)
    }

    results = compare_algorithms(algorithms, start_position, goal_position)

    print(results)
    plot_results(results)