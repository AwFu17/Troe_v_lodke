import numpy as np
import matplotlib.pyplot as plt
from ulianeno import Point
from ulianeno import Grid

class Circle:
    def __init__(self, center_x, center_y):
        self.center = Point(center_x, center_y)
        self.points = [Point(center_x + np.cos(t), center_y + np.sin(t)) for t in np.linspace(0, 2*np.pi, 200)]
    
    def plot(self):
        xs = [p.x for p in self.points]
        ys = [p.y for p in self.points]
        plt.plot(xs, ys)
        plt.axis('equal')
        plt.show()

    @classmethod
    def user_input(cls):
        while True:  # бесконечный цикл — "начинай заново", пока не будет правильно
            cx = float(input("x (должен быть < -1): "))
            cy = float(input("y (должен быть > 1): "))
            if cx < -1 and cy > 1:
                return cls(cx, cy)  # всё хорошо — выходим и создаём окружность
            else:
                print("Неправильно! Центр должен быть во II четверти и не пересекать оси. Попробуйте снова.")

class square:
    def __init__(self, niz_x, niz_y):
        self.center = Point(niz_x, niz_y)
        self.points = [Point(niz_x , niz_y)]
    

# Использование
Circle = Circle.user_input()
Circle.plot()