import matplotlib.pyplot as plt
import settings as s
import numpy as np
from integrator import Integrator
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
        I = Integrator(t_0, s.h)
        x11, x21 = I.RK4(s.f1, s.f2, c1, c2+R, t_end)
        x12, x22 = I.RK4(s.f1, s.f2, c1, c2-R, t_end)
        x13, x23 = I.RK4(s.f1, s.f2, c1-R, c2, t_end)
        x14, x24 = I.RK4(s.f1, s.f2, c1+R, c2, t_end)
        xc1, xc2 = I.RK4(s.f1, s.f2, c1, c2, t_end)

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

        ax.axis('equal')

        plt.tight_layout()

        plt.grid(True)
        fig.suptitle(f'Деформация круга при t={t_end:.2f}', fontsize=10, y=0.03)

        plt.show()

    @classmethod
    def from_user_input(cls):
        cx1 = float(input("Введите x-координату центра квадрата: "))
        cx2 = float(input("Введите y-координату центра квадрата: "))

        return cls(cx1, cx2)
