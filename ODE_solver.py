import matplotlib.pyplot as plt
import math
import numpy as np
from input_output_tests import input_test, test_init_conds, test_func_output, ODE_close_to_true


def euler_step(ODE, x0, t0, h, *args):
    """
    Performs a single step of the Euler method, at (x0, t0) for step size h.

        Parameters:
            ODE (function):     the ODE function that we want to solve
            x0 (float/list):    initial x value(s)
            t0 (float):         initial t value
            h (float):          step size
            *args (ndarray):    any additional arguments that ODE expects

        Returns:
            New values (x1, t1) of the ODE after one Euler step
    """

    x1 = x0 + h * ODE(x0, t0, *args)
    t1 = t0 + h

    return x1, t1


def RK4_step(ODE, x0, t0, h, *args):
    """
    Performs a single step of the 4th Runge-Kutta method, at (x0, t0) for step size h.

        Parameters:
            ODE (function):     the ODE function that we want to solve
            x0 (float/ndarray):    initial x value(s)
            t0 (float):         initial t value
            h (float):          step size
            *args (ndarray):      any additional arguments that ODE expects

        Returns:
            New values (x1, t1) of the ODE after one Runge Kutta step
    """

    k1 = ODE(x0, t0, *args)
    k2 = ODE(x0 + h * 0.5 * k1, t0 + 0.5 * h, *args)
    k3 = ODE(x0 + h * 0.5 * k2, t0 + 0.5 * h, *args)
    k4 = ODE(x0 + h * k3, t0 + h, *args)

    k = 1 / 6 * h * (k1 + 2 * k2 + 2 * k3 + k4)

    x1 = x0 + k
    t1 = t0 + h

    return x1, t1


def solve_to(ODE, x1, t1, t2, method, deltat_max, *args):
    """
    Solves the ODE for x1 between t1 and t2 using a specific method, in steps no bigger than delta_tmax

        Parameters:
            ODE (function):     the ODE function that we want to solve
            x1 (ndarray):       initial x value to solve for
            t1 (float):         initial time value
            t2 (float):         final time value
            method (function):  name of the function to use as the method, either 'euler' or 'RK4'
            deltat_max (float): maximum step size to use
            *args (ndarray):    any additional arguments that ODE expects

        Returns:
            Solution to the ODE found at t2, using method and with step size no bigger than delta_tmax
    """

    min_number_steps = math.floor((t2 - t1) / deltat_max)

    for i in range(min_number_steps):
        x1, t1 = method(ODE, x1, t1, deltat_max, *args)

    if t1 != t2:
        x1, t1 = method(ODE, x1, t1, t2 - t1, *args)

    return x1


def solve_ode(ODE, x0, t0, t1, method_name, deltat_max, system, *args):
    """
    Solves the ODE for x1 between t1 and t2 using a specific method, in steps no bigger than delta_tmax

        Parameters:
            ODE (function):     the ODE function that we want to solve
            x0:                 initial x value to solve for
            t0 (float):         initial time value
            t1 (float):         final time value
            method_name (str):  name of the method to use, either 'euler' or 'RK4'
            deltat_max (float): maximum step size to use
            system (bool):      boolean value that is True if the ODE is a system of equations, False if single ODE
            *args (ndarray):    any additional arguments that ODE expects

        Returns:
            Solution to the ODE found at t2, using method and with step size no bigger than delta_tmax
    """

    """
    Test all the inputs of the solve_ode function are the right type
    """

    # tests that the inputted ODE to solve is a function
    input_test(ODE, 'ODE', 'function')

    # tests that the inputted system parameter is a boolean
    input_test(system, 'system', 'boolean')

    # tests inputs for the time values and the maximum timestep
    input_test(t0, 't0', 'int_or_float')
    input_test(t1, 't1', 'int_or_float')
    input_test(deltat_max, 'deltat_max', 'int_or_float')

    # test inputs for the initial conditions
    # code for this is quite long and will be repeated in other files so define these tests in a separate function
    test_init_conds(x0, system)

    # tests that the inputted method is a string
    input_test(method_name, 'method', 'string')

    # test the output of the ODE is the right type and has the right dimensions
    test_func_output(ODE, x0, t0, system, *args)

    """
    Test to see if the inputted method has the right name
    """

    if method_name == 'euler':
        method = euler_step
    elif method_name == 'RK4':
        method = RK4_step
    else:
        raise ValueError(
            f"The method '{method_name}' is not accepted, please try 'euler' or 'RK4'")

    """
    Start the solve_ode code
    """

    # Calculate the amount of steps necessary to complete the algorithm
    number_steps = math.ceil(abs(t1 - t0) / deltat_max)

    # this ensures that the right number of columns depending on if the ODE is single or system
    if system:
        X = np.zeros((number_steps + 1, len(x0)))
    else:
        X = np.zeros((number_steps + 1, 1))

    T = np.zeros(number_steps + 1)
    X[0] = x0
    T[0] = t0

    for i in range(number_steps):

        # Calculate the next time value, with difference deltat_max to the previous one
        # On the condition that this new time value does not exceed the final time value
        if T[i] + deltat_max < t1:
            T[i + 1] = T[i] + deltat_max

        # Make sure the last time value in the array is the final time value defined at the beginning
        else:
            T[i + 1] = t1

        # Solve the ODE with the previous time value, as well as the one just defined
        X[i + 1] = solve_to(ODE, X[i], T[i], T[i + 1], method, deltat_max, *args)

    return X, T


def main():

    """
        Plot the solutions to the first order ODE, dx/dt = x
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
                t (ndarray):    t value

            Returns:
                Result of x = e^(t)
        """
        x = np.exp(t)

        return x

    # Solve the ODE using the Euler equation and plot the result
    method = 'euler'
    FO_euler, FO_time = solve_ode(FO_f, 1, 0, 1, method, 0.01, False)
    accuracy = ODE_close_to_true(solve_ode, FO_f, FO_true_solution, [1, 1], method, False)
    print(f"The accuracy of the {method} method to calculate the solution of {FO_f.__name__} is in the order {accuracy}.")
    plt.plot(FO_time, FO_euler, label='Euler')

    # Solve the ODE using the RK4 equation and plot the result
    method = 'RK4'
    FO_RK4, FO_time = solve_ode(FO_f, 1, 0, 1, method, 0.01, False)
    accuracy = ODE_close_to_true(solve_ode, FO_f, FO_true_solution, [1, 1], method, False)
    print(f"The accuracy of the {method} method to calculate the solution of {FO_f.__name__} is in the order {accuracy}.")

    plt.plot(FO_time, FO_RK4, label='RK4')

    # Plot the true solution to the ODE
    plt.plot(FO_time, FO_true_solution(FO_time), label='True')

    plt.xlabel('t')
    plt.ylabel('x')
    plt.legend()
    plt.show()

    """
    Plot the solutions to the second order ODE, d2xdt2 = -x
    """

    def SO_f(u, t):
        """
        Second order ODE function for d2xdt2 = -x, also expressed as dx/dt = y, dy/dt = -x
            Parameters:
                u (list):           initial x values
                t (int):            initial t value

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
                t (ndarray):  t values

            Returns:
                Array of solutions
        """
        x = np.cos(t) + np.sin(t)
        y = np.cos(t) - np.sin(t)

        u = [x, y]

        return u

    # Solve the second order ODE using the Euler equation
    method = 'euler'
    SO_euler, SO_time = solve_ode(SO_f, [1, 1], 0, 10, method, 0.01, True)
    accuracy = ODE_close_to_true(solve_ode, SO_f, SO_true_solution, [1, 1, 10], method, True)
    print(f"The accuracy of the {method} method to calculate the solution of {SO_f.__name__} is in the order {accuracy}.")

    # Solve the second order ODE using the RK4 equation
    method = 'RK4'
    SO_RK4, SO_time = solve_ode(SO_f, [1, 1], 0, 10, method, 0.01, True)
    accuracy = ODE_close_to_true(solve_ode, SO_f, SO_true_solution, [1, 1, 10], method, True)
    print(f"The accuracy of the {method} method to calculate the solution of {SO_f.__name__} is in the order {accuracy}.")


    # Plot the Euler, RK4 and True solutions to the first initial condition
    plt.subplot(2, 1, 1)
    plt.plot(SO_time, SO_euler[:, 0], label='Euler')
    plt.plot(SO_time, SO_RK4[:, 0], label='RK4')
    plt.plot(SO_time, SO_true_solution(SO_time)[0], label='True')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.legend()

    # Plot the Euler, RK4 and True solutions to the second initial condition
    plt.subplot(2, 1, 2)
    plt.plot(SO_time, SO_euler[:, 1], label='Euler')
    plt.plot(SO_time, SO_RK4[:, 1], label='RK4')
    plt.plot(SO_time, SO_true_solution(SO_time)[1], label='True')

    plt.xlabel('t')
    plt.ylabel('x')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
