import numpy as np
from point import Point


class Grid:

    def __init__(self, step: float = 0.01, max_coord: float = 3.0):
        self.step = step
        self.max_coord = max_coord
        self._build_grid()


    def _build_grid(self):
        # Формируем одномерный массив координат и создаем двумерную сетку координат
        coords = np.arange(self.step, self.max_coord + self.step, self.step)
        self.X1, self.X2 = np.meshgrid(coords, coords)
        #Преобразуем двумерную сетку в массив
        self.points = np.column_stack((self.X1.ravel(), self.X2.ravel()))

    def create_points(self, t0: float = 0.0):
        points = []

        for i, coord in enumerate(self.points):

            points.append(Point(i, coord, t0))

        return points

    # Выбираем точки, лежащие внутри заданной фигуры
    def select_points_inside_shape(self, shape):
        mask = np.array([
            shape.contains(x, y) for x, y in self.points
        ])
        return self.points[mask]  # True — точка внутри фигуры


    # Определяет строковое представление
    def __repr__(self):
        return (
            f"Grid(step={self.step}, max_coord={self.max_coord}, "
            f"points={len(self.points)})"
        )
