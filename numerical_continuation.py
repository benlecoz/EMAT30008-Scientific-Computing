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


def pseudo_arclength_continuation(ODE, u0, pars, max_pars, vary_par, param_number, discretisation, solver, pc, system):

    warnings.filterwarnings('ignore')

    param_list = np.linspace(pars[vary_par], max_pars, param_number)

    def pars_ind(pars, vary_par):

        if system:
            return pars[vary_par]
        else:
            return pars

    if system:
        v0 = solver(discretisation(ODE), u0, (pc, param_list[-1]))
        pars[vary_par] = param_list[-2]
        par_list = [param_list[-1], param_list[-2]]
    else:
        v0 = solver(discretisation(ODE), u0, (pc, param_list[0]))
        pars[vary_par] = param_list[1]
        par_list = [param_list[0], param_list[1]]

    v1 = solver(discretisation(ODE), np.round(v0, 2), (pc, pars_ind(pars, vary_par)))
    print('v1', v1)

    def update_par(pars, vary_par, predicted_p):

        pars[vary_par] = predicted_p
        # print('pred', pars[vary_par])

        return pars_ind(pars, vary_par)

    solution = [v0, v1]

    i = 0

    while i < 40:

        delta_x = solution[-1] - solution[-2]
        delta_p = par_list[-1] - par_list[-2]

        predicted_x = solution[-1] + delta_x
        predicted_p = par_list[-1] + delta_p

        # print('pred_x', predicted_x, 'pred_p', predicted_p)

        predicted_state = np.append(predicted_x, predicted_p)
        # print('full', predicted_state)

        pars[vary_par] = predicted_state[-1]

        pseudo_sol = solver(lambda state: np.append(discretisation(ODE)(state[:-1], pc, update_par(pars, vary_par, state[-1])), np.dot(state[:-1] - predicted_x, delta_x) + np.dot(state[-1] - predicted_p, delta_p)), predicted_state)

        # print(pseudo_sol[:-1])
        solution.append(pseudo_sol[:-1])
        par_list.append(pseudo_sol[-1])

        i += 1

    return solution, par_list


# c_interval = np.array([-2, 2])
# u0 = np.array([1])
# pc = phase_condition
# cubic_sol, cubic_param_list = nat_param_continuation(cubic, u0, c_interval, 10000, fsolve, lambda x: x, pc, False)
#
# sol, par = pseudo_arclength_continuation(cubic, u0, [-2], 2, 0, 50, lambda x: x, fsolve, pc, False)
#
# plt.plot(cubic_param_list, cubic_sol, label = 'Natural Parameter')
# plt.plot(par, sol, label = 'Pseudo Arc Length')
# plt.legend()
# plt.show()

beta_interval = np.array([-1, 2])
u0 = np.array([1.2, 1.2, 6.4])

hopf_sol, hopf_param_list = nat_param_continuation(Hopf_bif, u0, beta_interval, 50, fsolve, shooting, phase_condition, True)
pseudo_hopf_sol, pseudo_hopf_param = pseudo_arclength_continuation(Hopf_bif, u0, [-1], 2, 0, 50, shooting, fsolve, phase_condition, True)

sol_sol = []
for i in range(len(pseudo_hopf_sol)):
    sol_sol.append(pseudo_hopf_sol[i][0])
print(sol_sol)

plt.plot(hopf_param_list, hopf_sol[:, 0])
plt.plot(pseudo_hopf_param, sol_sol)

plt.show()

# beta_interval = np.array([-1, 2])
# u0 = np.array([1, 1, 6])
# mod_hopf_sol, mod_hopf_param_list = nat_param_continuation(mod_Hopf_bif, u0, beta_interval, 50, fsolve, shooting, phase_condition, True)
# pseudo_mod_hopf_sol, pseudo_mod_hopf_param = pseudo_arclength_continuation(mod_Hopf_bif, u0, [-1], 2, 0, 50, shooting, fsolve, phase_condition, True)
#
# print(pseudo_mod_hopf_sol)

# sol_sol = []
# for i in range(len(pseudo_mod_hopf_sol)):
#     sol_sol.append(pseudo_mod_hopf_sol[i][0])
# print(sol_sol)

# plt.plot(mod_hopf_param_list, mod_hopf_sol[:, 0])
# plt.plot(pseudo_mod_hopf_param, sol_sol)
# plt.show()
