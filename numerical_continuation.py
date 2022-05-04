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


def pseudo_arclength_continuation(ODE, u0, pars, max_pars, vary_par, param_number, discretisation, solver, pc):

    param_list = np.linspace(pars[vary_par], max_pars, param_number)

    v0 = solver(discretisation(ODE), u0, (pc, pars))

    pars[vary_par] = param_list[1]

    v1 = solver(discretisation(ODE), np.round(v0, 5), (pc, pars))

    def update_par(pars, vary_par, predicted_p):
        pars[vary_par] = predicted_p

        return pars

    solution = [v0, v1]
    par_list = [param_list[0], param_list[1]]

    i = 0

    while i < 40:

        delta_x = solution[-1] - solution[-2]
        delta_p = par_list[-1] - par_list[-2]

        predicted_x = solution[-1] + delta_x
        predicted_p = par_list[-1] + delta_p

        predicted_state = np.append(predicted_x, predicted_p)

        pars[vary_par] = predicted_state[-1]

        pseudo_sol = solver(lambda yeah: np.append(discretisation(ODE)(yeah[:-1], pc, update_par(pars, vary_par, yeah[-1])), np.dot(yeah[:-1] - predicted_x, delta_x) + np.dot(yeah[-1] - predicted_p, delta_p)), predicted_state)

        solution.append(pseudo_sol[:-1])
        par_list.append(pseudo_sol[-1])

        i += 1

    return solution, par_list


c_interval = np.array([-2, 2])
u0 = np.array([1])
pc = phase_condition
cubic_sol, cubic_param_list = nat_param_continuation(cubic, u0, c_interval, 10000, fsolve, lambda x: x, pc, False)

sol, par = pseudo_arclength_continuation(cubic, u0, [-2], 2, 0, 50, lambda x: x, fsolve, pc)

plt.plot(par, sol)
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
