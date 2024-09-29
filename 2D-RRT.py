import numpy as np
import matplotlib.pyplot as plt

class RRT:
    def __init__(self, start, goal, obstacles, x_lim, y_lim, step_size=0.5, max_iter=1000):
        self.start = start
        self.goal = goal
        self.obstacles = obstacles
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.step_size = step_size
        self.max_iter = max_iter
        # Nodes stored as tuples of (position, parent_index)
        self.nodes = [(start, -1)]  # Initialize with start node having no parent

    def distance(self, p1, p2):
        return np.linalg.norm(np.array(p1)-np.array(p2))

    def is_collision(self, p1, p2):
        for (ox, oy, size) in self.obstacles:
            d = np.abs(np.linalg.norm(np.array([ox, oy]) - np.array(p1)) + np.linalg.norm(np.array([ox, oy]) - np.array(p2)) - np.linalg.norm(np.array(p2) - np.array(p1)))
            if d < size:  # Approximate collision check
                return True
        return False

    def find_nearest(self, point):
        distances = [self.distance(point, node[0]) for node in self.nodes]
        nearest_index = distances.index(min(distances))
        return nearest_index

    def steer(self, from_node, to_node):
        if self.distance(from_node, to_node) < self.step_size:
            return to_node
        else:
            theta = np.arctan2(to_node[1] - from_node[1], to_node[0] - from_node[0])
            return from_node[0] + self.step_size * np.cos(theta), from_node[1] + self.step_size * np.sin(theta)

    def generate_random_point(self):
        return np.random.uniform(self.x_lim[0], self.x_lim[1]), np.random.uniform(self.y_lim[0], self.y_lim[1])

    def build(self):
        for i in range(self.max_iter):
            rnd_point = self.generate_random_point() if np.random.rand() > 0.1 else self.goal
            nearest_index = self.find_nearest(rnd_point)
            new_point = self.steer(self.nodes[nearest_index][0], rnd_point)

            if not self.is_collision(self.nodes[nearest_index][0], new_point):
                self.nodes.append((new_point, nearest_index))
                if self.distance(new_point, self.goal) <= self.step_size:
                    return True, i + 1  # Return True with iteration count when goal is reached
        return False, None

    def get_path(self):
        path = []
        current_index = len(self.nodes) - 1
        while current_index != -1:
            path.append(self.nodes[current_index][0])
            current_index = self.nodes[current_index][1]
        return path[::-1]  # Return reversed path

    def plot(self, path=None):
        plt.figure()
        for (ox, oy, size) in self.obstacles:
            circle = plt.Circle((ox, oy), size, color='r')
            plt.gca().add_patch(circle)

        plt.plot(self.start[0], self.start[1], "go")
        plt.plot(self.goal[0], self.goal[1], "bx")

        if path is not None:
            plt.plot([p[0] for p in path], [p[1] for p in path], color="green")

        for point in self.nodes:
            plt.plot(point[0][0], point[0][1], "yo")

        plt.xlim(self.x_lim)
        plt.ylim(self.y_lim)
        plt.show()

# Example usage
start = (2, 2)
goal = (8, 8)
obstacles = [(5, 5, 0.4), (6, 6, 0.2) , (4, 4, 0.3) , (3, 2.5, 0.5), (3, 2, 0.5)]
x_lim = (0, 10)
y_lim = (0, 10)

rrt = RRT(start, goal, obstacles, x_lim, y_lim)
found, iterations = rrt.build()
if found:
    print(f"Path found in {iterations} iterations!")
    path = rrt.get_path()
    rrt.plot(path)
else:
    print("Path not found.")
    rrt.plot()
