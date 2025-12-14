import numpy as np

class Circle:
    def __init__(self, center_x, center_y):
        self.center = np.array([center_x, center_y], dtype=float)
        self.radius = 1.0

    def contains(self, x, y):  #проверяем принадлежит ли точка кругу
        dx = x - self.center[0]
        dy = y - self.center[1]
        return dx*dx + dy*dy <= self.radius**2