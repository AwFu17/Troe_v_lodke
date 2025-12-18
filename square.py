import matplotlib.pyplot as plt
import settings as s
from integrator import Integrator
import matplotlib.patches as patches
import numpy as np


class Square:

    def __init__(self, center_x1, center_x2, side = 1.0):
        if  (center_x1 >= -side/2) or (center_x2 <= side/2):
            print("КВАДРАТ ДОЛЖЕН БЫТЬ ВО ВТОРОЙ ЧЕТВЕРТИ!!! КООРДИНАТЫ ЦЕНТРА БЫЛИ ВЫСТАВЛЕНЫ АВТОМАТИЧЕСКИ НА (-0.6, 0.6)")
            center_x1 = -0.6
            center_x2 = 0.6
        self.c_x1 = center_x1
        self.c_x2 = center_x2
        self.side = side
        self.half = self.side / 2.0  # 0.5

    def deformations(self, t_end = s.t_end, t_0 = s.t_0):
        I = Integrator(t_0, s.h)
        c1 = self.c_x1
        c2 = self.c_x2
        a = self.half
        x11, x21 = I.RK4(s.f1, s.f2, c1+a, c2+a, t_end)
        x12, x22 = I.RK4(s.f1, s.f2, c1+a, c2-a, t_end)
        x13, x23 = I.RK4(s.f1, s.f2, c1-a, c2-a, t_end)
        x14, x24 = I.RK4(s.f1, s.f2, c1-a, c2+a, t_end)
        l_x1 = abs(x11[-1] - x13[-1])
        l_x2 = abs(x21[-1] - x22[-1])


        print(x11[0], x21[0])
        print(x12[0], x22[0])
        print(x13[0], x23[0])
        print(x14[0], x24[0])

        print(x11[-1], x21[-1])
        print(x12[-1], x22[-1])
        print(x13[-1], x23[-1])
        print(x14[-1], x24[-1])

        print(f"длина по x1 = {l_x1}, деформация по x1 составляет {(l_x1 - 2*a)/(2*a)}")
        print(f"длина по x2 = {l_x2}, деформация по x2 составляет {(l_x2 - 2*a)/(2*a)}")

        fig, ax = plt.subplots()


        square = patches.Rectangle((min(x11[0], x12[0], x13[0], x14[0]), min(x21[0], x22[0], x23[0], x24[0])), a*2,a*2, facecolor='none', edgecolor='red', linewidth=2)
        ax.add_patch(square)

        square = patches.Rectangle((min(x11[-1], x12[-1], x13[-1], x14[-1]), min(x21[-1], x22[-1], x23[-1], x24[-1])), l_x1,
                                   l_x2, facecolor='none', edgecolor='blue', linewidth=2)
        ax.add_patch(square)

        plt.plot(x11, x21)


        plt.plot(x12, x22)


        plt.plot(x13, x23)


        plt.plot(x14, x24)
        plt.grid(True)
        fig.suptitle(f'Деформация квадрата при t={t_end:.2f}', fontsize=10, y=0.03)

        plt.tight_layout()
        plt.show()


    def contains(self, x, y):
        cx1, cx2 = self.c_x1, self.c_x2
        return (cx1 - self.half <= x <= cx1 + self.half) and (cx2 - self.half <= y <= cx2 + self.half)