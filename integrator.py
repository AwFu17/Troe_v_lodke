class Integrator:
    def __init__(self, t_0, h):
     self.t_0 = t_0
     self.h = h

    def RK4(self, f1, f2, x1_0, x2_0, t_end):
        x1 = [x1_0]
        x2 = [x2_0]
        t = [self.t_0]
        n = 1

        while t[n-1] <= t_end:
            k1_1 = f1(x1[n - 1], t[n - 1])
            k1_2 = f2(x2[n - 1], t[n - 1])
            k2_1 = f1(x1[n - 1] + k1_1 * self.h, t[n - 1] + self.h)
            k2_2 = f2(x2[n - 1] + k1_2 * self.h, t[n - 1] + self.h)
            x1.append(x1[n - 1] + (k1_1 + k2_1)/2 * self.h)
            x2.append(x2[n - 1] + (k1_2 + k2_2) / 2 * self.h)
            t.append(t[n - 1] + self.h)
            n += 1

        return x1, x2