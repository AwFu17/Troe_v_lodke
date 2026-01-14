from square import Square
from circle import Circle
from wwclassikenum import Body
from visualisation import Visualisation

if __name__ == "__main__":
    v = Visualisation(15, -10, -1, 1, 10)
    v.visualisate(1)
    #v.visualisate(2)
    #v.visualisate(5)

    print("Выберите фигуру:")
    print("1 — Круг")
    print("2 — Квадрат")
    choice = input("Выберите (1/2): ")

    if choice == "1":
        body = Circle.from_user_input()
    elif choice == "2":
        body = Square.from_user_input()
    else:
        print("Неверный выбор. Программа завершена.")
        exit()
    
body.deformations(t_end=1)
body.deformations(t_end=2)
body.deformations(t_end=5)