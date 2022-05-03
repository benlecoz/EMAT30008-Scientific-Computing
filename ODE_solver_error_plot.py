from ODE_solver import solve_ode
import numpy as np
import matplotlib.pyplot as plt
from math import exp, sin, cos, pi
import random
import time


def error(ODE, ODE_sol, u0, num, plot, minmax, timing, system, *args):
    """
    Plots double logarithmic graph of both the Euler and RK4 errors, for different timesteps.

        Parameters:
            ODE (function):     the ODE function that we want to solve
            ODE_sol (function): the true solution to the ODE function that we want to solve
            u0 (list):          initial conditions x0, t0 and t1
            num (int):          number of timesteps
            plot (bool):        plots the Euler and RK4 errors against the timesteps if True
            minmax (bool):      visualise overlapping Euler and RK4 errors if True
            timing (bool):      displays points at timesteps where Euler and RK4 have almost identical errors if True
            system (bool):      boolean value that is True if the ODE is a system of equations, False if single ODE
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

    def error_calculation(times):
        """
        Calculate the Euler and RK4 errors at every timestep

            Parameters:
                times (ndarray):  all the evenly spaced timesteps

            Returns:
                Euler and RK4 error lists at every timestep
        """

        # Create empty Euler and RK4 error lists of same dimension as the number of timesteps
        euler_error = np.zeros(num)
        RK4_error = np.zeros(num)

        # Unpack initial conditions
        if system:
            x0, t0, t1 = u0[:-2], u0[-2], u0[-1]
        else:
            x0, t0, t1 = u0[-3], u0[-2], u0[-1]

        # Find the true solution of the ODE at timestep i
        true_sol = ODE_sol(t1)

        for i in range(num):

            # Compute the Euler and RK4 values using solve_ode at timestep i
            euler_sol, euler_time = solve_ode(ODE, x0, t0, t1, 'euler', times[i], system, *args)
            RK4_sol, RK4_time = solve_ode(ODE, x0, t0, t1, 'RK4', times[i], system, *args)

            # Calculate the absolute difference between the true solution and the Euler and RK4 approximations
            euler_error[i] = abs(euler_sol[-1, -1] - true_sol)
            RK4_error[i] = abs(RK4_sol[-1, -1] - true_sol)

        return euler_error, RK4_error

    def error_plot(times, euler_error, RK4_error):
        """
        Plot the Euler and RK4 errors against the timesteps

            Parameters:
                times (ndarray):        all the evenly spaced timesteps
                euler_error (ndarray):  array of Euler errors
                RK4_error (ndarray):    array of RK4 errors
        """

        def time_difference():
            """
            Calculate time it takes for Euler and RK4 methods to run, with respective timesteps that produce similar error results

                Returns:
                    Coordinates of arbitrary Euler and RK4 points that have similar error values
            """

            def euler_RK4_equal_error():
                """
                Returns Euler and RK4 coordinates that have similar errors
                """

                # limit the Euler and RK4 error arrays to arrays where the error values overlap
                RK4_over_euler_min = RK4_error[RK4_error > min(euler_error)]
                euler_under_RK4_max = euler_error[euler_error < max(RK4_error)]

                # choose an arbitrary RK4 error value and find its respective timestep
                rand_RK4_idx = random.randrange(len(RK4_over_euler_min))
                RK4_idx = rand_RK4_idx + len(RK4_error) - len(RK4_over_euler_min)
                rand_RK4_err = RK4_error[RK4_idx]
                RK4_time = times[RK4_idx]

                # find the closest Euler error to the RK4 error above, and find its respective timestep
                euler_idx = (np.abs(euler_under_RK4_max - RK4_error[RK4_idx])).argmin()
                euler_near_rand_RK4 = euler_error[euler_idx]
                euler_time = times[euler_idx]

                return euler_time, euler_near_rand_RK4, RK4_time, rand_RK4_err

            x0, t0, t1 = u0[:-2], u0[-2], u0[-1]

            euler_t, euler_err, RK4_t, RK4_err = euler_RK4_equal_error()

            # time how long it takes to run the Euler method with the timestep found previously
            euler_start_time = time.time()
            solve_ode(ODE, x0, t0, t1, 'euler', euler_t, system, *args)
            print('The Euler error value is', euler_err, 'achieved in', time.time() - euler_start_time,
                  'seconds, for timestep', euler_t)

            # repeat for the RK4 method
            RK4_start_time = time.time()
            solve_ode(ODE, x0, t0, t1, 'RK4', RK4_t, system, *args)
            print('The RK4 error value is', RK4_err, 'achieved in', time.time() - RK4_start_time,
                  'seconds, for timestep',
                  RK4_t)

            return [euler_t, euler_err, RK4_t, RK4_err]

        def general_plot():
            """
            Set a general plot format, which scatters the Euler and RK4 error values against all the timesteps
            """

            ax = plt.gca()
            ax.scatter(times, euler_error)
            ax.scatter(times, RK4_error)
            ax.set_yscale('log')
            ax.set_xscale('log')
            ax.set_xlabel('Timestep size')
            ax.set_ylabel('Absolute Error')

            return ax

        """
        Plot lines at the minimum of the Euler errors, and at the maximum of the RK4 errors, to show the overlap
        """

        if minmax:
            ax = general_plot()
            euler_min = np.full((1, len(times)), min(euler_error))[0]
            RK4_max = np.full((1, len(times)), max(RK4_error))[0]
            plt.plot(times, euler_min, '--', alpha=0.4)
            plt.plot(times, RK4_max, '--', alpha=0.4)

            """
            Plot the arbitrarily chosen points which has similar Euler and RK4 errors
            """

            if timing:
                idx = time_difference()
                ax.scatter(idx[0], idx[1], c='black')
                ax.scatter(idx[2], idx[3], c='black')
                ax.legend(('Min Euler error', 'Max RK4 error', 'Euler error', 'RK4 error', 'Equivalent Euler and RK4 errors'))

            else:
                ax.legend(('Min Euler error', 'Max RK4 error', 'Euler error', 'RK4 error'))

        else:
            ax = general_plot()
            ax.legend(('Min Euler error', 'Max RK4 error'))

        plt.show()

    timesteps = log_time_step(-4, 0)
    euler, RK4 = error_calculation(timesteps)

    if plot:
        error_plot(timesteps, euler, RK4)

    return euler, RK4


def main():

    """
        Plot a double logarithmic scale of the error of Euler and RK4, depending on the timestep
    """

    def FO_f(x, t, *args):
        """
        Function for first order ODE dxdt = x
            Parameters:
                x (int):    x value
                t (int):    t value
                *args:      any additional arguments that ODE expects

            Returns:
                Array of dxdt at (x,t)
        """

        dxdt = np.array([x], *args)

        return dxdt

    def FO_true_solution(t):
        """
        True solution to the first order ODE dxdt = x defined above
            Parameters:
                t (int):    t value

            Returns:
                Result of x = e^(t)
        """
        x = exp(t)

        return x

    def SO_f(u, t, *args):
        """
        Second order ODE function for d2xdt2 = -x
            Parameters:
                u (list):   initial x values
                t (int):    initial t value
                *args:      any additional arguments that ODE expects

            Returns:
                Array of dXdt at (x,t)
        """
        x, y = u
        dxdt = y
        dydt = -x
        dXdt = np.array([dxdt, dydt, *args])
        return dXdt

    def SO_true_solution(t):
        """
        True solution to the second order ODE d2xdt2 = -x defined above
            Parameters:
                t (int):    t value

            Returns:
                Array of solutions
        """
        x = cos(t) + sin(t)
        y = cos(t) - sin(t)

        u = [x, y]

        return u


if __name__ == '__main__':
    main()
