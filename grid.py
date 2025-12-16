import numpy as np
from points import Point

class Grid:
    def init(self, step: float = 0.01,min_coord: float = -3.0, max_coord: float = 3.0):
        self.step = step
        self.max_coord = max_coord
        self.min_coord = min_coord
        self._build_grid()

    def __build_grid__(self):
        coords = np.arange(self.min_coord, self.max_coord + self.step, self.step)
        self.X1, self.X2 = np.meshgrid(coords, coords)
        self.points = np.column_stack((self.X1.ravel(), self.X2.ravel()))

    def create_points(self, t0: float = 0.0):
        points = []
        for i, coord in enumerate(self.points):
            points.append(Point(i, coord, t0))
        return points

    def select_points_inside_shape(self, shape):
        mask = np.array([shape.contains(x, y) for x, y in self.points])
        return self.points[mask]

    def repr(self):
        return (
            f"Grid(step={self.step}, max_coord={self.max_coord}, "
            f"points={len(self.points)})"
        )