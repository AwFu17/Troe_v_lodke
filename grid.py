import numpy as np
from points import Point


class Grid:
    def __init__(self, step, x1_min, x1_max, x2_min, x2_max, t_0):
        self.step = step
        self.x1_min = x1_min
        self.x1_max = x1_max
        self.x2_min = x2_min
        self.x2_max = x2_max
        self._build_grid()
        self.t_0 = t_0
        self.points = self._create_points(t_0)


    def _build_grid(self):
        X1, X2 = np.meshgrid(np.arange(self.x1_min, self.x1_max+self.step, self.step), np.arange(self.x2_min, self.x2_max+self.step, self.step))
        self.p = np.column_stack((X1.ravel(), X2.ravel()))


    def _create_points(self, t0):
        points = []
        for i, coord in enumerate(self.p):
            points.append(Point(i, coord[0], coord[1], t0))
        return points

    def repr(self):
        return (
            f"Grid(step={self.step}, max_coord={self.max_coord}, "
            f"points={len(self.p)})"
        )