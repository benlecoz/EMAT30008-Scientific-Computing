import matplotlib.pyplot as plt
import math
import numpy as np


def f(x, t, *args):
    return np.array([x, t], *args)


def euler_step(f, x0, t0, h, *args):

    func = f(x0, t0, *args)
    x1 = x0 + h * func[0]
    t1 = t0 + h

    return x1, t1


def RK4_step(f, x0, t0, h, *args):

    k1 = f(x0, t0, *args)
    k2 = f(x0 + h * 0.5 * k1[0], t0 + 0.5 * h, *args)
    k3 = f(x0 + h * 0.5 * k2[0], t0 + 0.5 * h, *args)
    k4 = f(x0 + h * k3[0], t0 + h, *args)

    k = 1/6 * h * (k1 + 2 * k2 + 2 * k3 + k4)

    x1 = x0 + k[0]
    t1 = t0 + h

    return x1, t1


def solve_to(f, x1, t1, t2, step_type, deltat_max, *args):

    min_number_steps = math.floor((t2 - t1) / deltat_max)

    if step_type == 'euler':

        for i in range(min_number_steps):
            x1, t1 = euler_step(f, x1, t1, deltat_max, *args)

        if t1 != t2:
            x1, t1 = euler_step(f, x1, t1, t2 - t1)

    if step_type == 'RK4':

        for i in range(min_number_steps):

            x1, t1 = RK4_step(f, x1, t1, deltat_max, *args)

        if t1 != t2:
            x1, t1 = RK4_step(f, x1, t1, t2 - t1)

    return x1


def solve_ode(f, x0, t, step_type, deltat_max, *args):

    min_number_steps = math.ceil((t[-1] - t[0]) / deltat_max)

    X = np.zeros(min_number_steps + 1)
    X[0] = x0

    for i in range(min_number_steps):

        X[i + 1] = solve_to(f, X[i], t[i], t[i+1], step_type, deltat_max, *args)

    return X, t


X_values, t_values = solve_ode(f, 1, [0, 0.35, 0.7, 1], 'euler', 0.35)
print(X_values, t_values)

