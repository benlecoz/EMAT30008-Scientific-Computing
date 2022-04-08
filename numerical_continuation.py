from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import numpy as np
import math


def cubic(x, args):
    c = args
    eq = x ** 3 - x + c
    return eq


def nat_param_continuation(ODE, u0, param, step, solver):

    args = param[0]
    first_sol = solver(ODE, u0, args)

    number_steps = math.ceil(abs((param[1] - param[0])) / step + 1)

    sol = np.zeros((number_steps, len(u0)))
    param_list = np.zeros(number_steps)
    sol[0] = first_sol
    param_list[0] = param[0]

    for i in range(number_steps - 1):

        param_list[i+1] = param_list[i] + step

        if param_list[i+1] <= param[-1]:
            sol[i+1] = solver(ODE, sol[i], args = param_list[i+1])
        else:
            param_list[-1] = param[1]
            sol[i + 1] = solver(ODE, sol[i], args = param_list[-1])

    return sol, param_list


c_interval = np.array([-2, 2])
cubic_sol, cubic_param_list = nat_param_continuation(cubic, np.array([1]), c_interval, 0.015, fsolve)

print(cubic_sol)

plt.plot(cubic_param_list, cubic_sol)
plt.show()





