from square import Square
from circle import Circle
from visualisation import Visualisation



if __name__ == "__main__":
    v = Visualisation(15, -10, -1, 1, 10)
    v.visualisate(1)
    v.visualisate(2)
    v.visualisate(5)

    circle = Circle(-2, 2, 1)
    circle.deformations(t_end=1)
    circle.deformations(t_end=2)
    circle.deformations(t_end = 5)

    square = Square(-2, 2, 1)
    square.deformations(t_end=1)
    square.deformations(t_end=2)
    square.deformations(t_end=5)




