import matplotlib.pyplot as plt
import numpy as np
from ODE_solver import solve_ode, SO_plot
from scipy.optimize import fsolve


def SO_f2(X, t, *args):

    x, y = X
    dxdt = x * (1 - x) - (1 * x * y) / (0.1 + x)
    dydt = 0.16 * y * (1 - y / x)
    dXdt = np.array([dxdt, dydt, *args])

    return dXdt


def Hopf_bif(U, t, *args):

    beta = 1
    sigma = -1

    u1, u2 = U
    du1dt = beta * u1 - u2 + sigma * u1 * (u1 ** 2 + u2 ** 2)
    du2dt = u1 + beta * u2 + sigma * u2 * (u1 ** 2 + u2 ** 2)
    dudt = np.array([du1dt, du2dt, *args])

    return dudt


def conds(u0):
    x0, t = u0[:-1], u0[-1:]
    sol, sol_time = solve_ode(Hopf_bif, x0, 0, t)
    phase_con = Hopf_bif(x0, 0, t)[0]
    period_con1 = x0[0] - sol[-1, 0]
    period_con2 = x0[1] - sol[-1, 1]
    period_con = np.array([period_con1, period_con2])

    print('x0 is:', x0,'sol is:',  sol,'t is:', t)

    return np.r_[phase_con, period_con]


def shooting(x0, t):

    real_sol = fsolve(conds, np.r_[x0, t])

    return real_sol


def shooting_cycle(f, solution):

    x0, t = solution[:-1], solution[-1]

    SO_plot(f, x0, 0, t)


def shooting_orbit(f, solution):

    x0, t = solution[:-1], solution[-1]

    sol, sol_time = solve_ode(f, x0, 0, t)

    plt.plot(sol[:, 0], sol[:, 1])
    plt.show()


x0 = np.array([1, 1])
t = np.array([8])

solution = shooting(x0, t)
print(solution)

shooting_cycle(Hopf_bif, solution)
shooting_orbit(Hopf_bif, solution)



