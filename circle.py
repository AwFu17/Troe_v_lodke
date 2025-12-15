import numpy as np

class Circle:
    def __init__(self, center_x, center_y):
        self.center = np.array([center_x, center_y], dtype=float)
        self.radius = 1.0

    def contains(self, x, y):  #проверяем принадлежит ли точка кругу
        dx = x - self.center[0]
        dy = y - self.center[1]
        return dx*dx + dy*dy <= self.radius**2
    @classmethod
    def from_user_input(cls):
        cx = float(input("Введите x-координату центра круга: "))
        cy = float(input("Введите y-координату центра круга: "))
        assert cx < 0 and cy > 0, "Центр должен быть во 2-й четверти (x < 0, y > 0)" 
        return cls(cx, cy)