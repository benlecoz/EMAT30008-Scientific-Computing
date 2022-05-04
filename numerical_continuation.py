from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import numpy as np
import math
from numerical_shooting import shooting, phase_condition
import warnings


def cubic(x, t, args):
    c = args
    eq = x ** 3 - x + c
    return eq


def Hopf_bif(U, t, args):
    beta = args
    u1, u2 = U

    du1dt = beta * u1 - u2 - u1 * (u1 ** 2 + u2 ** 2)
    du2dt = u1 + beta * u2 - u2 * (u1 ** 2 + u2 ** 2)
    dudt = np.array([du1dt, du2dt])

    return dudt


def mod_Hopf_bif(U, t, args):
    beta = args
    u1, u2 = U

    du1dt = beta * u1 - u2 + u1 * (u1 ** 2 + u2 ** 2) - u1 * (u1 ** 2 + u2 ** 2) ** 2
    du2dt = u1 + beta * u2 + u2 * (u1 ** 2 + u2 ** 2) - u2 * (u1 ** 2 + u2 ** 2) ** 2
    dudt = np.array([du1dt, du2dt])

    return dudt


def nat_param_continuation(ODE, u0, param_range, param_number, solver, discretisation, pc, system):
    warnings.filterwarnings('ignore')

    if system:
        args = param_range[1]
        param_list = np.linspace(param_range[1], param_range[0], param_number)

    else:
        args = param_range[0]
        param_list = np.linspace(param_range[0], param_range[1], param_number)

    first_sol = solver(discretisation(ODE), u0, (pc, args))

    sol = np.zeros((param_number, len(u0)))
    sol[0] = first_sol

    for i in range(param_number - 1):
        sol[i + 1] = solver(discretisation(ODE), np.round(sol[i], 5), (pc, param_list[i + 1]))

    return sol, param_list


c_interval = np.array([-2, 2])
u0 = np.array([1])
pc = phase_condition
cubic_sol, cubic_param_list = nat_param_continuation(cubic, u0, c_interval, 10000, fsolve, lambda x: x, pc, False)

plt.plot(cubic_param_list, cubic_sol)
plt.show()

# beta_interval = np.array([-1, 2])
# u0 = np.array([1.2, 1.2, 6.4])
# hopf_sol, hopf_param_list = nat_param_continuation(Hopf_bif, u0, beta_interval, 50, fsolve, shooting, phase_condition, True)
#
# plt.plot(hopf_param_list, hopf_sol[:, 0])
# plt.show()

# beta_interval = np.array([-1, 2])
# u0 = np.array([1, 1, 6])
# mod_hopf_sol, mod_hopf_param_list = nat_param_continuation(mod_Hopf_bif, u0, beta_interval, 50, fsolve, shooting, phase_condition)
#
# plt.plot(mod_hopf_param_list, mod_hopf_sol[:, 0])
# plt.show()
