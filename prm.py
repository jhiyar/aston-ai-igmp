import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

# Parameters
num_samples = 1000  # Number of samples
k_neighbors = 5    # Number of nearest neighbors for each sample
space_size = 100    # Size of the space

# Generate random samples in a 2D space
samples = np.random.rand(num_samples, 2) * space_size

# Create a KD-Tree for fast nearest neighbor search
tree = KDTree(samples)

# Function to check if a connection between two points is valid (simple straight-line test)
def is_valid_connection(p1, p2):
    # In a more complex scenario, this function could test for obstacles
    return True

# Build the roadmap by connecting each point to its nearest neighbors
edges = []
for i, sample in enumerate(samples):
    distances, indices = tree.query(sample, k=k_neighbors+1)  # Nearest neighbors
    for j in range(1, len(indices)):  # Skip the sample itself
        neighbor = samples[indices[j]]
        if is_valid_connection(sample, neighbor):
            edges.append((sample, neighbor))

# Visualize the PRM
plt.figure(figsize=(8, 8))
plt.scatter(samples[:, 0], samples[:, 1], c='blue', label='Nodes')

# Draw edges
for edge in edges:
    p1, p2 = edge
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'green')

# Add start and goal points
start = np.array([10, 10])
goal = np.array([90, 90])
plt.scatter([start[0]], [start[1]], c='red', label='Start')
plt.scatter([goal[0]], [goal[1]], c='purple', label='Goal')  # Changed the goal color to purple

plt.title('PRM with 1000 nodes')
plt.legend()
plt.xlim(0, space_size)
plt.ylim(0, space_size)
plt.show()  # Removed the grid with no plt.grid(False) since default is no grid
