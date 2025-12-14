import numpy as np

class Point:

    def __init__(self, point_id: int, position, t: float = 0.0):
        self.id = point_id
        self.t = t
        self.x = np.array(position, dtype=float)

    def set_state(self, new_position, new_time: float):
        self.x = np.array(new_position, dtype=float)
        self.t = new_time

    def get_state(self):
        return self.x, self.t

    def __repr__(self):
        return f"Point(id={self.id}, x={self.x}, t={self.t})"
