class MaterialPoint:
    def __init__(self, id, t, x_0, y_0, v_x0 = 0, v_y0 = 0, a_x0 = 0, a_y0 = 0):
        self.id = id
        self.t = t
        self.x = x_0
        self.y = y_0
        self.v_x = v_x0
        self.v_y = v_y0
        self.a_x = a_x0
        self.a_y = a_y0

    def get_position(self):
        print(self.x, "  ", self.y)


