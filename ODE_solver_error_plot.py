from ODE_solver import solve_ode, f, true_solution
import numpy as np
import matplotlib.pyplot as plt


def error_plot(ODE, x0, t0, t1, *args):
    timesteps = np.logspace(-4, 0, 100)

    euler_error = np.zeros(len(timesteps))
    RK4_error = np.zeros(len(timesteps))

    for i in range(len(timesteps)):
        true_sol = true_solution(t1)
        euler_sol, euler_time = solve_ode(ODE, x0, t0, t1, 'euler', timesteps[i], *args)
        RK4_sol, RK4_time = solve_ode(ODE, x0, t0, t1, 'RK4', timesteps[i], *args)
        euler_error[i] = abs(euler_sol[-1, -1] - true_sol)
        RK4_error[i] = abs(RK4_sol[-1, -1] - true_sol)

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


def main():
    error_plot(f, 1, 0, 1)


if __name__ == '__main__':
    main()