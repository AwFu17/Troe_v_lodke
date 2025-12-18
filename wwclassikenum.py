from enum import Enum
from circle import Circle
from square import Square

class Body(Enum):
    CIRCLE = 1
    SQUARE = 2

    @classmethod
    def create_body(cls, body_type, cx1: float, cx2: float, size: float = 1):
        if body_type == Body.CIRCLE:
            return Circle(cx1, cx2, size)
        elif body_type == Body.SQUARE:
            return Square(cx1, cx2, size)
        else:
           raise ValueError("Неизвестный тип тела")