import matplotlib.pyplot as plt
import numpy as np
from ODE_solver import solve_ode, SO_ode_plot
from scipy.optimize import fsolve
from math import sqrt, cos, sin


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

        sol, sol_time = solve_ode(ODE, x0, 0, t, 'RK4', 0.01, True, *args)

        period_con = []

        for i in range(len(x0)):
            period_con.append(x0[i] - sol[-1, i])

        full_conds = np.r_[np.array(period_con), phase_con]

        return full_conds

    return conds


def shooting_orbit(ODE, u0, pc, system, *args):

    shooting_solution = fsolve(shooting(ODE), u0, (pc, *args), full_output=True)

    convergence = shooting_solution[3]

    if convergence == 'The solution converged.':
        print(convergence + '\n')
        x0, t = shooting_solution[0][:-1], shooting_solution[0][-1]
    else:
        raise ValueError(f"The shooting algorithm could not converge to a solution, please try again with different values.")

    sol, sol_time = solve_ode(ODE, x0, 0, t, 'RK4', 0.01, True, *args)

    for i in range(sol.shape[1]):
        plt.plot(sol_time, sol[:, i], label = 'S' + str(i))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()

    if system:
        plt.figure()
        plt.plot(sol[:, 0], sol[:, 1])

    plt.show()


def main():

    def predator_prey(X, t, args):
        """
        Function for predator-prey equation
            Parameters:
                X:      initial conditions
                t:      t value
                args:   any additional arguments that ODE expects

            Returns:
                Array of dxdt, dydt
        """
        x, y = X
        a, b, d = args[0], args[1], args[2]

        dxdt = x * (1 - x) - (a * x * y) / (d + x)
        dydt = b * y * (1 - y / x)

        dXdt = np.array([dxdt, dydt])

        return dXdt

    """
        We simulate the predator-prey equations for a = 1, d = 0.1, and choosing two b values on either side of 0.26
    """

    b1 = np.round(np.random.uniform(0.1, 0.25), 2)
    b2 = np.round(np.random.uniform(0.27, 0.5), 2)

    pred_prey_sol1, pred_prey_time1 = solve_ode(predator_prey, [0.2, 0.2], 0, 120, 'RK4', 0.01, True, [1, b1, 0.1])
    pred_prey_sol2, pred_prey_time2 = solve_ode(predator_prey, [0.2, 0.2], 0, 120, 'RK4', 0.01, True, [1, b2, 0.1])

    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.plot(pred_prey_time1, pred_prey_sol1, label = 'Predator Prey equation with b = ' + str(b1))
    ax1.legend()
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')

    ax2.plot(pred_prey_time2, pred_prey_sol2, label = 'Predator Prey equation with b = ' + str(b2))
    ax2.legend()
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')

    plt.show()

    """
    Plotting this long term, we see that when b < 0.26, the predator prey solutions start 
    oscillating periodically, whereas when b > 0.26, the solutions converge.
    """

    args = [1, 0.16, 0.1]
    pc = phase_condition
    u0 = np.array([1.2, 1.2, 6])

    shooting_orbit(predator_prey, u0, pc, True, args)

    # hopf_args = [1, -1]
    # u0 = [1.2, 1.2, 8]
    # pc = phase_condition
    #
    # real_sol = fsolve(shooting(Hopf_bif), u0, (pc, hopf_args), full_output=True)
    # print(real_sol[0])

    # shooting_cycle(Hopf_bif, Hopf_bif_true_sol, shooting_solution, 'yes', args)
    # shooting_orbit(Hopf_bif, shooting_solution, args)

    # args = [1, -1]
    # shooting_solution = shooting(Hopf_ext, [1, 1, 1, 8], phase_condition, args)


if __name__ == '__main__':
    main()

