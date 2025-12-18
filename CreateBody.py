from square import Square
from circle import Circle
from wwclassikenum import Body

class BodyCreator():
   def __init__(self, type):
        self.type = type

   def create_body(body_type: Body, cx1: float, cx2: float, size: float = 1):
       if body_type == Body.CIRCLE:
           return Circle(cx1, cx2, size)
       elif body_type == Body.SQUARE:
           return Square(cx1, cx2, size)
       else:
           raise ValueError("Неизвестный тип тела")
