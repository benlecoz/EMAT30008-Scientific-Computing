import matplotlib.pyplot as plt
import math
import numpy as np


def euler_step(ODE, x0, t0, h, *args):
    x1 = x0 + h * ODE(x0, t0, *args)
    t1 = t0 + h

    return x1, t1


def RK4_step(ODE, x0, t0, h, *args):
    k1 = ODE(x0, t0, *args)
    k2 = ODE(x0 + h * 0.5 * k1, t0 + 0.5 * h, *args)
    k3 = ODE(x0 + h * 0.5 * k2, t0 + 0.5 * h, *args)
    k4 = ODE(x0 + h * k3, t0 + h, *args)

    k = 1 / 6 * h * (k1 + 2 * k2 + 2 * k3 + k4)

    x1 = x0 + k
    t1 = t0 + h

    return x1, t1


def solve_to(ODE, x1, t1, t2, method, deltat_max, *args):
    min_number_steps = math.floor((t2 - t1) / deltat_max)

    if method == 'euler':

        for i in range(min_number_steps):
            x1, t1 = euler_step(ODE, x1, t1, deltat_max, *args)

        if t1 != t2:
            x1, t1 = euler_step(ODE, x1, t1, t2 - t1, *args)

    if method == 'RK4':

        for i in range(min_number_steps):
            x1, t1 = RK4_step(ODE, x1, t1, deltat_max, *args)

        if t1 != t2:
            x1, t1 = RK4_step(ODE, x1, t1, t2 - t1, *args)

    return x1


def solve_ode(ODE, x0, t0, t1, method, deltat_max, *args):
    min_number_steps = math.ceil((t1 - t0) / deltat_max)
    X = np.zeros((min_number_steps + 1, 2))
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

    X, T = solve_ode(ODE, x0, t0, t1, 'RK4', 0.01, *args)

    plt.plot(T, X[:, 0], label = 'S1')
    plt.plot(T, X[:, 1], label = 'S2')
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

    SO_plot(SO_f, [0, 2], 0, 50)


if __name__ == "__main__":
    main()



