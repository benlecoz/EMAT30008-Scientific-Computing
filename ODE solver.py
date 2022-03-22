import matplotlib.pyplot as plt
import math
import numpy as np
import time


def f(x, t, *args):
    return np.array([x], *args)


def SO_f(X, t, *args):
    x, y = X
    dxdt = y
    dydt = -x
    dXdt = np.array([dxdt, dydt, *args])
    return dXdt


def true_solution(t):
    x = math.exp(t)

    return x


def euler_step(f, x0, t0, h, *args):

    x1 = x0 + h * f(x0, t0, *args)
    t1 = t0 + h

    return x1, t1


def RK4_step(f, x0, t0, h, *args):
    k1 = f(x0, t0, *args)
    k2 = f(x0 + h * 0.5 * k1, t0 + 0.5 * h, *args)
    k3 = f(x0 + h * 0.5 * k2, t0 + 0.5 * h, *args)
    k4 = f(x0 + h * k3, t0 + h, *args)

    k = 1 / 6 * h * (k1 + 2 * k2 + 2 * k3 + k4)

    x1 = x0 + k
    t1 = t0 + h

    return x1, t1


def solve_to(f, x1, t1, t2, step_type, deltat_max, *args):
    min_number_steps = math.floor((t2 - t1) / deltat_max)

    if step_type == 'euler':

        for i in range(min_number_steps):
            x1, t1 = euler_step(f, x1, t1, deltat_max, *args)

        if t1 != t2:
            x1, t1 = euler_step(f, x1, t1, t2 - t1, *args)

    if step_type == 'RK4':

        for i in range(min_number_steps):
            x1, t1 = RK4_step(f, x1, t1, deltat_max, *args)

        if t1 != t2:
            x1, t1 = RK4_step(f, x1, t1, t2 - t1, *args)

    return x1


def solve_ode(f, x0, t, step_type, deltat_max, *args):
    min_number_steps = math.ceil((t[1] - t[0]) / deltat_max)

    X = np.zeros((min_number_steps + 1, 2))
    T = np.zeros(min_number_steps + 1)
    X[0] = x0
    T[0] = t[0]

    for i in range(min_number_steps):

        if T[i] + deltat_max < t[1]:
            T[i + 1] = T[i] + deltat_max

        else:
            T[i + 1] = t[1]
        X[i + 1] = solve_to(f, X[i], T[i], T[i + 1], step_type, deltat_max, *args)

    return X, T


def error_plot(f, x0, t):
    timesteps = np.logspace(-4, 0, 100)

    euler_error = np.zeros(len(timesteps))
    RK4_error = np.zeros(len(timesteps))

    for i in range(len(timesteps)):
        true_sol = true_solution(t[1])
        euler_sol, euler_time = solve_ode(f, x0, t, 'euler', timesteps[i])
        RK4_sol, RK4_time = solve_ode(f, x0, t, 'RK4', timesteps[i])

        euler_error[i] = abs(euler_sol[-1] - true_sol)
        RK4_error[i] = abs(RK4_sol[-1] - true_sol)

    ax = plt.gca()
    ax.scatter(timesteps, euler_error)
    ax.scatter(timesteps, RK4_error)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('Timestep size')
    ax.set_ylabel('Absolute Error')
    ax.legend(('Euler error', 'RK4 error'))
    plt.show()

    return timesteps, euler_error, RK4_error


def SO_plot(x0, t, step_type, delta_max):

    X, T = solve_ode(SO_f, x0, t, step_type, delta_max)

    plt.plot(X[:, 0], T)
    plt.plot(X[:, 1], T)
    plt.show()


def time_difference(f, x0, t, RK4_timestep, euler_timestep):

    euler_start_time = time.time()
    euler_sol, euler_time = solve_ode(f, x0, t, 'euler', euler_timestep)
    euler_error = abs(euler_sol[-1] - true_solution(t[1]))
    print('The Euler error value is', euler_error, 'achieved in', time.time() - euler_start_time, 'seconds, for timestep',
          euler_timestep)

    RK4_start_time = time.time()
    RK4_sol, RK4_time = solve_ode(f, x0, t, 'RK4', RK4_timestep)
    RK4_error = abs(RK4_sol[-1] - true_solution(t[1]))
    print('The RK4 error value is', RK4_error,'achieved in', time.time() - RK4_start_time, 'seconds, for timestep', RK4_timestep)


# time_difference(f, 1, [0, 2], 0.3, 9.7 * 10 ** (-5), 1)
# one, two, three = error_plot(f, 1, [0, 2], 1)

SO_plot([0, 2], [0, 50], 'euler', 0.1)
