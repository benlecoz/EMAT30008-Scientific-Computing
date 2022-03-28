import matplotlib.pyplot as plt
import numpy as np
from ODE_solver import solve_ode
from scipy.optimize import fsolve


def SO_f2(X, t, *args):

    x, y = X
    dxdt = x * (1 - x) - (1 * x * y) / (0.1 + x)
    dydt = 0.16 * y * (1 - y / x)
    dXdt = np.array([dxdt, dydt, *args])

    return dXdt


def conds(u0):
    x0, t = u0[:-1], u0[-1:]
    sol, sol_time = solve_ode(SO_f2, x0, 0, t)
    phase_con = SO_f2(x0, 0, t)[0]
    period_con1 = x0[0] - sol[-1, 0]
    period_con2 = x0[1] - sol[-1, 1]
    period_con = np.array([period_con1, period_con2])

    return np.r_[phase_con, period_con]


def shooting(x0, t):

    real_sol = fsolve(conds, np.r_[x0, t])

    return real_sol


def shooting_orbit(solution):

    x0, t = solution[:-1], solution[-1]

    sol, sol_time = solve_ode(SO_f2, x0, 0, t)
    plt.plot(sol[:, 0], sol[:, 1])
    plt.show()


x0 = np.array([0.25, 0.3])
t = np.array([23])

solution = shooting(x0, t)
print(solution)

shooting_orbit(solution)



