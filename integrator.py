def RK4(f1, f2, x1_0, x2_0, t_0, t_end, h):
    x1 = [x1_0]
    x2 = [x2_0]
    # x = [[x1_0, x2_0]]
    t = [t_0]
    n = 1

    while t[n-1] <= t_end:
        k1_1 = f1(x1[n - 1], t[n - 1])
        k1_2 = f2(x2[n - 1], t[n - 1])
        k2_1 = f1(x1[n - 1] + k1_1 * h, t[n - 1] + h)
        k2_2 = f2(x2[n - 1] + k1_2 * h, t[n - 1] + h)
        x1.append(x1[n - 1] + (k1_1 + k2_1)/2 * h)
        x2.append(x2[n - 1] + (k1_2 + k2_2) / 2 * h)
        # k1_1 = f1(x[n - 1][0], t[n - 1])
        # k1_2 = f2(x[n - 1][1], t[n - 1])
        # k2_1 = f1(x[n - 1][0] + k1_1 * h, t[n - 1] + h)
        # k2_2 = f2(x[n - 1][1] + k1_2 * h, t[n - 1] + h)
        # x += [[(x[n - 1][0] + (k1_1 + k2_1) / 2 * h), (x[n - 1][1] + (k1_2 + k2_2) / 2 * h)]]
        # print(x)
        t.append(t[n - 1] + h)
        n += 1

    return x1, x2


