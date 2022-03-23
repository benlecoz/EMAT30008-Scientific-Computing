import numpy as np


def SO_f(X, t, *args):
    x, y = X
    dxdt = x * (1 - x) - (1 * x * y) / (0.1 + x)
    dydt = 0.25 * y * (1 - y/x)
    dXdt = np.array([dxdt, dydt, *args])
    return dXdt