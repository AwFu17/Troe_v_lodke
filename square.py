import numpy as np

class Square:

    def __init__(self, center_x, center_y):
        self.center = np.array([center_x, center_y], dtype=float)
        self.side = 1.0
        self.half = self.side / 2.0  # 0.5

    def contains(self, x, y):
        cx, cy = self.center
        return (cx - self.half <= x <= cx + self.half) and \
               (cy - self.half <= y <= cy + self.half)