import numpy as np
import matplotlib.pyplot as plt
from integrator import RK4
from grid import Grid


class LiniiTokov():
    def __init__(self, grid: Grid, t):
        self.grid = grid
        self.t = t
        self.x2 = np.linspace(self.grid.x2_min, self.grid.x2_max, self.grid.n)
        self.x1 = np.array([])
        for i in range(self.grid.n):
            self.grid.points[i].trajectory_x1 = np.append(self.x1, self.grid.points[i].C * self.x2[i] ** (np.log(self.t)/np.exp(self.t)))
            plt.scatter(self.x1[i], self.x2[i])
        plt.show()



