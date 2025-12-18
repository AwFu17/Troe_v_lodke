import numpy as np
from settings import *

class Point:
    def __init__(self, id_of_point, x1_0, x2_0, t_0):
        self.id = id_of_point
        self.t_0 = t_0
        self.x1_0 = x1_0
        self.x2_0 = x2_0

    def get_velocity(self, t):
        v1 = f1(x)
        return [v1, v2]


    # Функция для вывода, чтобы был красивый вид
    def __repr__(self):
        return f"Point(id={self.id}, x1_0={self.x1_0}, x2_0={self.x2_0}, t_0={self.t_0})"
