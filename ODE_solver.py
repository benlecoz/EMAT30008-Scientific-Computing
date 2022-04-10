import matplotlib.pyplot as plt
import math
import numpy as np


def euler_step(ODE, x0, t0, h, *args):
    """
    Performs a single step of the Euler, at (x0, t0) for step size h.

        Parameters:
            ODE (function):     the ODE function that we want to solve
            x0 (float/list):    initial x value(s)
            t0 (float):         initial t value
            h (float):          step size
            *args (array):      any additional arguments that ODE expects

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


def input_test(test, test_name, test_type):

    def int_or_float():
        if not isinstance(test, (float, int)):
            raise TypeError(f"The argument passed for {test_name} is not a float or an integer, but a {type(test)}. Please input an integer or a float")

    if test_type == 'int_or_float':
        int_or_float()

    def function():
        if not callable(test):
            raise TypeError(f"The argument passed for {test_name} is not a function, but a {type(test)}. Please input a function")

    if test_type == 'function':
        function()

    def string():
        if not isinstance(test, str):
            raise TypeError(f"The argument passed for {test_name} is not a string, but a {type(test)}. Please input a string")

    if test_type == 'string':
        string()


def solve_to(ODE, x1, t1, t2, method, deltat_max, *args):
    """
    Solves the ODE for x1 between t1 and t2 using a specific method, in steps no bigger than delta_tmax

        Parameters:
            ODE (function):     the ODE function that we want to solve
            x1 (ndarray):       initial x value to solve for
            t1 (float):         initial time value
            t2 (float):         final time value
            method (str):       name of the method to use, either 'euler' or 'rungekutta'
            deltat_max (float): maximum step size to use
            *args (ndarray):    any additional arguments that ODE expects

        Returns:
            Solution to the ODE found at t2, using method and with step size no bigger than delta_tmax
    """

    min_number_steps = math.floor((t2 - t1) / deltat_max)

    if method == 'euler':
        use_method = euler_step
    elif method == 'rungekutta':
        use_method = RK4_step
    else:
        raise TypeError(
            f"The method '{method}' is not accepted, please use 'euler' or 'rungekutta'")

    for i in range(min_number_steps):
        x1, t1 = use_method(ODE, x1, t1, deltat_max, *args)

    if t1 != t2:
        x1, t1 = use_method(ODE, x1, t1, t2 - t1, *args)

    return x1


def solve_ode(ODE, x0, t0, t1, method, deltat_max, *args):
    """
    Solves the ODE for x1 between t1 and t2 using a specific method, in steps no bigger than delta_tmax

        Parameters:
            ODE (function):     the ODE function that we want to solve
            x0 (ndarray):       initial x value to solve for
            t0 (float):         initial time value
            t1 (float):         final time value
            method (str):       name of the method to use, either 'euler' or 'rungekutta'
            deltat_max (float): maximum step size to use
            *args (ndarray):    any additional arguments that ODE expects

        Returns:
            Solution to the ODE found at t2, using method and with step size no bigger than delta_tmax
    """

    def input_check():

        for x in range(len(x0)):
            input_test(x, 'x0', 'int_or_float')

        input_test(t0, 't0', 'int_or_float')
        input_test(t1, 't1', 'int_or_float')
        input_test(deltat_max, 'deltat_max', 'int_or_float')

        input_test(ODE, 'ODE', 'function')

        input_test(method, 'method', 'string')

    input_check()

    min_number_steps = math.ceil((t1 - t0) / deltat_max)
    X = np.zeros((min_number_steps + 1, len(x0)))
    T = np.zeros(min_number_steps + 1)
    X[0] = x0
    T[0] = t0

    for i in range(min_number_steps):

        if T[i] + deltat_max < t1:
            T[i + 1] = T[i] + deltat_max

        else:
            T[i + 1] = t1
        X[i + 1] = solve_to(ODE, X[i], T[i], T[i + 1], method, deltat_max, *args)

    return X, T


def SO_plot(ODE, x0, t0, t1, *args):
    X, T = solve_ode(ODE, x0, t0, t1, 'rungekutta', 0.01, *args)

    plt.plot(T, X[:, 0], label='S1')
    plt.plot(T, X[:, 1], label='S2')
    plt.legend()

    plt.show()

    return X, T


def main():
    def FO_f(x, t, *args):
        """
        Function for first Order Differential Equation (DE) dxdt = x
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
        True solution to the first ODE dxdt = x
            Parameters:
                t (int):    t value

            Returns:
                Result of x = e^(t)
        """
        x = math.exp(t)

        return x

    def SO_f(u, t, *args):
        """
        Second Order DE function for d2xdt2 = -x
            Parameters:
                u (list):    x and t values
                *args:      any additional arguments that ODE expects

            Returns:
                Array of dXdt at (x,t)
        """
        x, y = u
        dxdt = y
        dydt = -x
        dXdt = np.array([dxdt, dydt, *args])
        return dXdt

    # SO_plot(SO_f, [0, 2], 0, 50)
    solve_ode(FO_f, [1], 0, 2, 'rungekutta', 0.01)


if __name__ == "__main__":
    main()
