import matplotlib.pyplot as plt
import math
import numpy as np


def f(x, t):
    return np.array(x, t)


def euler_step(f, x0, t0, h):
    x1 = x0 + h * f(x0, t0)
    t1 = t0 + h
    return x1, t1


def RK4_step(f, x0, t0, h):
    k1 = f(x0, t0)
    k2 = f(x0 + h * 0.5 * k1, t0 + 0.5 * h)
    k3 = f(x0 + h * 0.5 * k2, t0 + 0.5 * h)
    k4 = f(x0 + h * k3, t0 + h)

    x1 = x0 + 1/6 * h * (k1 + 2 * k2 + 2 * k3 + k4)
    t1 = t0 + h

    return x1, t1


def solve_to(f, x1, t1, t2, step, deltat_max):

    min_number_steps = math.ceil((t2 - t1) / deltat_max) + 1
    X = np.zeros(min_number_steps)
    T = np.zeros(min_number_steps)
    X[0] = x1
    T[0] = t1

    if step == 'euler':

        for i in range(min_number_steps-1):
            X[i+1], T[i+1] = euler_step(f, X[i], T[i], deltat_max)

        X[-1], T[-1] = euler_step(f, X[-2], T[-2], t2 - T[-2])

    if step == 'RK4':

        for i in range(min_number_steps - 1):
            X[i + 1], T[i + 1] = RK4_step(f, X[i], T[i], deltat_max)

        X[-1], T[-1] = RK4_step(f, X[-2], T[-2], t2 - T[-2])

    return X, T


# def solve_ode(f, x0, t, deltat_max):


yes, nope_sire = solve_to(f, 1, 0, 1, 'RK4', 0.1)
print(yes)




















#
# def euler_error(x0, deltat_max):
#     deltat = [1/6, 1/5, 1/4, 1/3, 1/2, 1]
#     i = 0
#     h = deltat[i]
#     error_list = []
#     deltat_used = []
#     while h < deltat_max:
#         sum_h = h
#         x1 = x0
#         while sum_h < 1:
#             x2 = euler_step(x1, h)
#             x1 = x2
#             sum_h += h
#         error = math.exp(1) - x2
#         error_list.append(error)
#         deltat_used.append(h)
#         i += 1
#         h = deltat[i]
#     return deltat_used, error_list
#
#
# deltat_used, error_list = euler_error(1, 1)
# plt.loglog(deltat_used, error_list)
# plt.xlabel('Timestep 'r'$\Delta t$')
# plt.ylabel("Error")
# plt.show()


