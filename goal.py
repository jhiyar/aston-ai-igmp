import numpy as np


class Goal:
    def __init__(self, goal_id=0):
        goal_poses = {
            0: [0.7, 0.0, 0.6],  # above table
            1: [0.7, 0.0, 0.2],  # under table
            2: [0.7, 0.3, 0.6],
            3: [0.7, 0.3, 0.2],
            4: [0.7, -0.3, 0.6],
            5: [0.7, -0.3, 0.2],
        }

        
        if goal_id not in goal_poses.keys():
            raise ValueError('invalid goal id')

        self._pos = np.asarray(goal_poses[goal_id])

        self.distance_threshold = 0.05

    @property
    def pos(self):
        return self._pos

    def distance(self, query_pos):
        pos = np.asarray(query_pos)
        return np.linalg.norm(pos - self.pos)

    def reached(self, query_pos):
        return self.distance(query_pos) < self.distance_threshold
