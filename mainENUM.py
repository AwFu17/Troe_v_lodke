from square import Square
from circle import Circle
from visualisation import Visualisation
from wwclassikenum import Body, create_body

if __name__ == "__main__":
    v = Visualisation(15, -10, -1, 1, 10)
    v.visualisate(1)
    v.visualisate(2)
    v.visualisate(5)

circle =create_body(Body.CIRCLE, -2, 2, 1)
square = create_body(Body.SQUARE, -2, 2, 1)

for t in [1, 2, 5]:
        circle.deformations(t_end=t)
        square.deformations(t_end=t)



