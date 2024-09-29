import matplotlib.pyplot as plt

# Data for planning time and path length for each algorithm
results = {
    'JPlusRRT': {
        'path_length': [0.3442, 0.3465, 0.3430, 0.3487, 0.3425, 0.3470, 0.3441, 0.3456, 0.3432, 0.3475], 
        'planning_time': [0.0040, 0.0042, 0.0041, 0.0039, 0.0043, 0.0040, 0.0038, 0.0042, 0.0037, 0.0041]
    },
    'RRTStar': {
        'path_length': [0.3125, 0.2621, 0.2910, 0.2684, 0.2580, 0.2845, 0.2700, 0.2647, 0.2599, 0.2523], 
        'planning_time': [0.0062, 0.0020, 0.0010, 0.0012, 0.0030, 0.0011, 0.0020, 0.0021, 0.0020, 0.0023]
    },
    'IKRRT': {
        'path_length': [0.5852, 0.5704, 0.5589, 0.5783, 0.5635, 0.5810, 0.5608, 0.5745, 0.5667, 0.5769], 
        'planning_time': [0.4741, 0.4582, 0.4427, 0.4621, 0.4538, 0.4693, 0.4489, 0.4610, 0.4563, 0.4660]
    },
    'BIKRRT': {
        'path_length': [2.6474, 2.6395, 2.6568, 2.6412, 2.6531, 2.6497, 2.6444, 2.6510, 2.6488, 2.6550], 
        'planning_time': [0.0099, 0.0532, 0.0490, 0.0080, 0.0782, 0.0201, 0.0977, 0.0070, 0.0376, 0.1001]
    }
}

# Prepare data for box plot
planning_times = [results['JPlusRRT']['planning_time'], results['RRTStar']['planning_time'], 
                  results['IKRRT']['planning_time'], results['BIKRRT']['planning_time']]

path_lengths = [results['JPlusRRT']['path_length'], results['RRTStar']['path_length'], 
                results['IKRRT']['path_length'], results['BIKRRT']['path_length']]

# Labels
labels = ['JPlusRRT', 'RRTStar', 'IKRRT', 'BIKRRT']

# Plot Planning Time Distribution
plt.figure(figsize=(10, 5))
plt.boxplot(planning_times, labels=labels, patch_artist=True, boxprops=dict(facecolor='lightblue'))
plt.title('Planning Time Distribution Across Algorithms')
plt.ylabel('Planning Time (s)')
plt.grid(True)
plt.show()

# Plot Path Length Distribution
plt.figure(figsize=(10, 5))
plt.boxplot(path_lengths, labels=labels, patch_artist=True, boxprops=dict(facecolor='lightgreen'))
plt.title('Path Length Distribution Across Algorithms')
plt.ylabel('Path Length')
plt.grid(True)

plt.show()
