import matplotlib.pyplot as plt
import numpy as np
from ODE_solver import solve_ode, SO_plot
from scipy.optimize import fsolve
from math import sqrt, cos, sin


def predator_prey(X, t, args):

    x, y = X
    a, b, d = args[0], args[1], args[2]

    dxdt = x * (1 - x) - (a * x * y) / (d + x)
    dydt = b * y * (1 - y / x)

    dXdt = np.array([dxdt, dydt])

    return dXdt


def Hopf_bif(U, t, args):

    beta = args[0]
    sigma = args[1]

    u1, u2 = U
    du1dt = beta * u1 - u2 + sigma * u1 * (u1 ** 2 + u2 ** 2)
    du2dt = u1 + beta * u2 + sigma * u2 * (u1 ** 2 + u2 ** 2)
    dudt = np.array([du1dt, du2dt])

    return dudt


def Hopf_bif_true_sol(t, args):

    beta = args[0]
    phase_con = 0
    u1 = np.zeros(len(t))
    u2 = np.zeros(len(t))

    for i in range(len(t)):
        u1[i] = sqrt(beta) * cos(t[i] + phase_con)
        u2[i] = sqrt(beta) * sin(t[i] + phase_con)

    return u1, u2


def Hopf_ext(U, t, args):

    beta = args[0]
    sigma = args[1]

    u1, u2, u3 = U
    du1dt = beta * u1 - u2 + sigma * u1 * (u1 ** 2 + u2 ** 2)
    du2dt = u1 + beta * u2 + sigma * u2 * (u1 ** 2 + u2 ** 2)
    du3dt = - u3
    dudt = np.array([du1dt, du2dt, du3dt])

    return dudt


def phase_condition(ODE, u0, *args):

    x0, t = u0[:-1], u0[-1]
    phase_con = ODE(x0, t, *args)[0]

    return x0, t, phase_con


def shooting(ODE):

    def conds(u0, pc, *args):
        x0, t, phase_con = pc(ODE, u0, *args)

        sol, sol_time = solve_ode(ODE, x0, 0, t, 'rungekutta', 0.01, True, *args)

        period_con = []

        for i in range(len(x0)):
            period_con.append(x0[i] - sol[-1, i])

        full_conds = np.r_[np.array(period_con), phase_con]

        return full_conds

    return conds


def shooting_cycle(ODE, ODE_sol, solution, error, *args):

    x0, t = solution[:-1], solution[-1]

    X, T = SO_plot(ODE, x0, 0, t, *args)

    if error == 'yes':

        u1, u2 = ODE_sol(T, *args)

        error1 = np.zeros(len(T))
        error2 = np.zeros(len(T))

        for i in range(len(T)):
            error1[i] = abs(u1[i] - X[i, 0])
            error2[i] = abs(u2[i] - X[i, 1])

        plt.plot(T, error1, label = 'S1 error')
        plt.plot(T, error2, label = 'S2 error')
        plt.legend()
        plt.show()


def shooting_orbit(ODE, solution, *args):

    x0, t = solution[:-1], solution[-1]
    print(t)
    sol, sol_time = solve_ode(ODE, x0, 0, t, 'RK4', 0.01, *args)

    plt.plot(sol[:, 0], sol[:, 1])
    plt.show()


def main():
    # args = [1, 0.16, 0.1]
    # shooting_solution = shooting(predator_prey, [0.2, 0.2, 21], args)
    #
    # shooting_cycle(predator_prey, shooting_solution, args)
    # shooting_orbit(predator_prey, shooting_solution, args)

    hopf_args = [1, -1]
    u0 = [1.2, 1.2, 8]
    pc = phase_condition

    real_sol = fsolve(shooting(Hopf_bif), u0, (pc, hopf_args), full_output=True)
    print(real_sol[0])

    # shooting_cycle(Hopf_bif, Hopf_bif_true_sol, shooting_solution, 'yes', args)
    # shooting_orbit(Hopf_bif, shooting_solution, args)

    # args = [1, -1]
    # shooting_solution = shooting(Hopf_ext, [1, 1, 1, 8], phase_condition, args)


if __name__ == '__main__':
    main()

