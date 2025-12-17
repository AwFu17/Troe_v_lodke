import matplotlib.pyplot as plt
import settings as s
import numpy as np
from integrator import RK4
import matplotlib.patches as patches

class Circle:
    def __init__(self, center_x1, center_x2, radius = 1):
        if (center_x1 >= -radius) or (center_x2 <= radius):
            print("КРУГ ДОЛЖЕН БЫТЬ ВО ВТОРОЙ ЧЕТВЕРТИ!!! КООРДИНАТЫ ЦЕНТРА БЫЛИ ВЫСТАВЛЕНЫ АВТОМАТИЧЕСКИ НА (-1.1, 1.1)")
            center_x1 = -1.1
            center_x2 = 1.1
        self.c_x1 = center_x1
        self.c_x2 = center_x2
        self.radius = radius

    def deformations(self,t_end = s.t_end, t_0 = s.t_0):
        c1 = self.c_x1
        c2 = self.c_x2
        R = self.radius
        x11, x21 = RK4(s.f1, s.f2, c1, c2+R, t_0, t_end, s.h)
        x12, x22 = RK4(s.f1, s.f2, c1, c2-R, t_0, t_end, s.h)
        x13, x23 = RK4(s.f1, s.f2, c1-R, c2, t_0, t_end, s.h)
        x14, x24 = RK4(s.f1, s.f2, c1+R, c2, t_0, t_end, s.h)
        xc1, xc2 = RK4(s.f1, s.f2, c1, c2, t_0, t_end, s.h)

        d_x1 = abs(x14[-1] - x13[-1])
        d_x2 = abs(x21[-1] - x22[-1])


        print(x11[0], x21[0])
        print(x12[0], x22[0])
        print(x13[0], x23[0])
        print(x14[0], x24[0])
        print(xc1[0], xc2[0])

        print(x11[-1], x21[-1])
        print(x12[-1], x22[-1])
        print(x13[-1], x23[-1])
        print(x14[-1], x24[-1])
        print(xc1[-1], xc2[-1])

        print(f"диаметр по x1 = {d_x1}, деформация по x1 составляет {(d_x1 - 2*R)/(2*R)}")
        print(f"диаметр по x2 = {d_x2}, деформация по x2 составляет {(d_x2 - 2*R)/(2*R)}")

        fig, ax = plt.subplots()


        square = patches.Ellipse((c1, c2), R*2, R*2, facecolor='none', edgecolor='red', linewidth=2)
        ax.add_patch(square)

        square = patches.Ellipse((xc1[-1], xc2[-1]), d_x1, d_x2, facecolor='none', edgecolor='blue', linewidth=2)
        ax.add_patch(square)

        plt.plot(x11, x21)


        plt.plot(x12, x22)


        plt.plot(x13, x23)


        plt.plot(x14, x24)


        plt.tight_layout()

        plt.grid(True)
        fig.suptitle(f'Деформация круга при t={t_end:.2f}', fontsize=10, y=0.03)

        plt.show()

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