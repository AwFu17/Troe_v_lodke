from integrator import RK4
import numpy as np
import matplotlib.pyplot as plt

def f1(x1, t):
    return -x1 * np.log(t)

def f2(x2, t):
    return -x2 * np.exp(t)

if __name__ == "__main__":
    t_0 = 0.01
    t_end = 10
    h = 0.0001

    x11, x21 = RK4(f1, f2, -5, 1, t_0, t_end, h)
    x12, x22 = RK4(f1, f2, -2, 1, t_0, t_end, h)
    x13, x23 = RK4(f1, f2, -5, 10, t_0, t_end, h)
    x14, x24 = RK4(f1, f2, -2, 10, t_0, t_end, h)

    plt.plot(x11, x21)
    plt.grid(True)

    plt.plot(x12, x22)
    plt.grid(True)

    plt.plot(x13, x23)
    plt.grid(True)

    plt.plot(x14, x24)
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    plt.plot([[0, 0], [1, 1], [2, 2]])
    plt.grid(True)