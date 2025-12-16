from integrator import RK4
import numpy as np
import matplotlib.pyplot as plt
from square import Square
from circle import Circle



if __name__ == "__main__":
    circle = Circle(-2, 0)
    circle.deformations(0.01, 1)



