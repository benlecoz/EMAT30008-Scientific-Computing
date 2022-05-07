from ODE_solver import solve_ode
import numpy as np
import matplotlib.pyplot as plt
from math import exp, pi
import random
import warnings
import time
from ODE_solver import input_test, test_init_conds, test_func_output


def error(ODE, ODE_sol, u0, num, plot, minmax, timing, system, *args):
    """
    Plots double logarithmic graph of both the Euler and RK4 errors, for different timesteps.

        Parameters:
            ODE (function):     the ODE function that we want to solve
            ODE_sol (function): the true solution to the ODE function that we want to solve
            u0:                 initial conditions x0, t0 and t1
            num (int):          number of timesteps
            plot (bool):        plots the Euler and RK4 errors against the timesteps if True
            minmax (bool):      visualise overlapping Euler and RK4 errors if True
            timing (bool):      displays points at timesteps where Euler and RK4 have almost identical errors if True
            system (bool):      boolean value that is True if the ODE is a system of equations, False if single ODE
            *args (array):      any additional arguments that ODE expects

        Returns:
            Arrays of the Euler and RK4 errors at each timestep
    """

    """
    Test all the inputs of the error function
    """

    input_test(ODE, 'ODE', 'function')
    input_test(ODE_sol, 'ODE_sol', 'function')

    input_test(system, 'system', 'boolean')
    input_test(plot, 'plot', 'boolean')
    input_test(minmax, 'minmax', 'boolean')
    input_test(timing, 'timing', 'boolean')

    test_init_conds(u0, True)

    input_test(num, 'num', 'int_or_float')
    test_func_output(ODE, u0[:-2], u0[-2], system)

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
            x0, t0, t1 = u0[0], u0[1], u0[2]

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

            if system:
                x0, t0, t1 = u0[:-2], u0[-2], u0[-1]
            else:
                x0, t0, t1 = u0[0], u0[1], u0[2]

            euler_t, euler_err, RK4_t, RK4_err = euler_RK4_equal_error()

            print('Results from Timing the two methods for equivalent error values:\n')

            # time how long it takes to run the Euler method with the timestep found previously
            euler_start_time = time.time()
            solve_ode(ODE, x0, t0, t1, 'euler', euler_t, system, *args)
            print('The Euler error value is', euler_err, 'achieved in', time.time() -
                  euler_start_time, 'seconds, for timestep', euler_t)

            # repeat for the RK4 method
            RK4_start_time = time.time()
            solve_ode(ODE, x0, t0, t1, 'RK4', RK4_t, system, *args)
            print('The RK4 error value is', RK4_err, 'achieved in', time.time() - RK4_start_time,
                  'seconds, for timestep',
                  RK4_t, '\n')

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
                ax.legend(('Euler error', 'RK4 error', 'Min overlapping Euler error', 'Max overlapping RK4 error', 'Equivalent Euler and RK4 errors'))

            else:
                ax.legend(('Euler error', 'RK4 error', 'Min overlapping Euler error', 'Max overlapping RK4 error'))

        else:
            ax = general_plot()
            ax.legend(('Euler error', 'RK4 error'))

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

    def FO_f(x, t):
        """
        Function for first order ODE dxdt = x
            Parameters:
                x (int):    x value
                t (int):    t value

            Returns:
                Array of dxdt at (x,t)
        """

        dxdt = np.array([x])

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

    # Plot the logarithmic plot of how the error depends on the timestep
    FO_u0 = [1, 0, 1]
    error(FO_f, FO_true_solution, FO_u0, 100, True, False, False, False)

    # Plot logarithmic plot of error against timestep, highlighting timesteps that produce similar Euler and RK4 errors
    # Print running time of Euler and RK4 algorithms for respective timesteps that return similar errors
    error(FO_f, FO_true_solution, FO_u0, 100, True, True, True, False)

    """
    Depending on the error value that the algorithm chooses, the Euler method runs either instantly (0.0 seconds), or 
    take a fraction of a second to run.
    On the other hand, the RK4 method always runs instantly (0.0 seconds). This shows that the RK4 method is either 
    quicker or equal to the Euler method, when the two produce similar error values.
    """

    def SO_f(u, t):
        """
        Second order ODE function for d2xdt2 = -x
            Parameters:
                u (list):   initial x values
                t (int):    initial t value

            Returns:
                Array of dXdt at (x,t)
        """
        x, y = u
        dxdt = y
        dydt = -x
        dXdt = np.array([dxdt, dydt])
        return dXdt

    def SO_true_solution(t):
        """
        True solution to the second order ODE d2xdt2 = -x defined above
            Parameters:
                t (ndarray):    t value

            Returns:
                Array of solutions
        """
        x = np.cos(t) + np.sin(t)
        y = np.cos(t) - np.sin(t)

        u = np.array([x, y])

        return u

    # Suppress warnings
    warnings.simplefilter("ignore", category=RuntimeWarning)

    """
    Plotting the numerical solutions over a large range of t
    """

    # Run the Euler and RK4 method over large range of t and a large timestep

    SO_euler = solve_ode(SO_f, [1, 1], 0, 20, 'euler', 0.5, True)
    SO_RK4 = solve_ode(SO_f, [1, 1], 0, 100, 'RK4', 1, True)

    # We use slightly different range and timestep values for the two methods, as the respective values evidence the
    # results better for each respective method

    # Run the true solution code
    SO_true = SO_true_solution(np.linspace(0, 2 * pi, 100))

    fig, (ax1, ax2) = plt.subplots(1, 2)

    # Show the Euler results and the true solutions, plotting x against dx/dt
    ax1.plot(SO_true[0], SO_true[1], label = 'True')
    ax1.plot(SO_euler[0][:, 0], SO_euler[0][:, 1], label='Euler')
    ax1.set_xlabel('x')
    ax1.set_ylabel('dx/dt')
    ax1.legend()

    # Show the RK4 results and the true solutions, plotting x against dx/dt
    ax2.plot(SO_true[0], SO_true[1], label='True')
    ax2.plot(SO_RK4[0][:, 0], SO_RK4[0][:, 1], label='RK4')
    ax2.legend()
    ax2.set_xlabel('x')
    ax2.set_ylabel('dx/dt')
    plt.show()

    """
    We find that for the Euler and the RK4 method, there are quite significant errors when running the methods with a 
    large range of t and a high timestep. 
    In the case of the Euler method, the solutions seem to diverge outward, in an infinitely large spiral.
    On the other hand, for the RK4 method, the solutions are converging inward 
    For both methods, the results jump around quite a lot, given the large timestep.  
    """


if __name__ == '__main__':
    main()
