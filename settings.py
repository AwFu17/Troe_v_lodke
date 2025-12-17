import numpy as np

def f1(x1, t):
    return -x1 * np.log(t)

def f2(x2, t):
    return -x2 * np.exp(t)

t_0 = 0.0001
t_end = 1
h = 0.0001

