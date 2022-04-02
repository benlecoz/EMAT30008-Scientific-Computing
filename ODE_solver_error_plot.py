from ODE_solver import solve_ode, f, true_solution
import numpy as np
import matplotlib.pyplot as plt


def error(ODE, ODE_sol, u0, num, plot, *args):
    """
    Plots double logarithmic graph of both the Euler and RK4 errors, for different timesteps.

        Parameters:
            ODE (function):     the ODE function that we want to solve
            ODE_sol (function): the true solution to the ODE function that we want to solve
            u0 (list):          initial conditions
            num (int):          number of timesteps
            plot (bool):        plots the Euler and RK4 errors against the timesteps if True
            *args (array):      any additional arguments that ODE expects

        Returns:
            Arrays of the Euler and RK4 errors at each timestep
    """

    def log_time_step(start, stop):
        """
        Returns numbers spaced evenly on a logarithmic scale

            Parameters:
                start (float):  initial timestep at 10**start
                stop (float):   final timestep at 10**stop

            Returns:
                Array of all the evenly spaced timesteps
        """

        log_timesteps = np.logspace(start, stop, num)

        return log_timesteps

    def error_calculation_and_plot(time):
        """
        Calculate the Euler and RK4 errors at every timestep

            Parameters:
                time (ndarray):  all the evenly spaced timesteps

            Returns:
                Euler and RK4 error lists at every timestep
        """

        # Create empty Euler and RK4 error lists of same dimension as the number of timesteps
        euler_error = np.zeros(num)
        RK4_error = np.zeros(num)

        # Unpack initial conditions
        x0, t0, t1 = u0[:-2], u0[-2], u0[-1]

        for i in range(num):
            # Find the true solution of the ODE at timestep i
            true_sol = ODE_sol(t1)

            # Compute the Euler and RK4 values using solve_ode at timestep i
            euler_sol, euler_time = solve_ode(ODE, x0, t0, t1, 'euler', time[i], *args)
            RK4_sol, RK4_time = solve_ode(ODE, x0, t0, t1, 'RK4', time[i], *args)

            # Calculate the absolute difference between the true solution and the Euler and RK4 approximations
            euler_error[i] = abs(euler_sol[-1, -1] - true_sol)
            RK4_error[i] = abs(RK4_sol[-1, -1] - true_sol)

        return euler_error, RK4_error

    def error_plot(time, euler_error, RK4_error):
        """
        Plot the Euler and RK4 errors against the timesteps

            Parameters:
                time (ndarray):         all the evenly spaced timesteps
                euler_error (ndarray):  array of Euler errors
                RK4_error (ndarray):    array of RK4 errors

        """

        ax = plt.gca()
        ax.scatter(time, euler_error)
        ax.scatter(time, RK4_error)
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlabel('Timestep size')
        ax.set_ylabel('Absolute Error')
        ax.legend(('Euler error', 'RK4 error'))
        plt.show()

    timesteps = log_time_step(-4, 0)
    euler, RK4 = error_calculation_and_plot(timesteps)

    if plot:
        error_plot(timesteps, euler, RK4)

    return euler, RK4


def main():
    u0 = [1, 0, 1]
    error(f, true_solution, u0, 100, False)


if __name__ == '__main__':
    main()
