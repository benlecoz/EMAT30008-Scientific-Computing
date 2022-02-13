import matplotlib.pyplot as plt
import math


def euler_step(f, x0, t0, h):
    x1 = x0 + h * f(t0, x0)
    t1 = t0 + h
    return x1, t1


def euler_error(x0, deltat_max):
    deltat = [1/6, 1/5, 1/4, 1/3, 1/2, 1]
    i = 0
    h = deltat[i]
    error_list = []
    deltat_used = []
    while h < deltat_max:
        sum_h = h
        x1 = x0
        while sum_h < 1:
            x2 = euler_step(x1, h)
            x1 = x2
            sum_h += h
        error = math.exp(1) - x2
        error_list.append(error)
        deltat_used.append(h)
        i += 1
        h = deltat[i]
    return deltat_used, error_list


deltat_used, error_list = euler_error(1, 1)
plt.loglog(deltat_used, error_list)
plt.xlabel('Timestep 'r'$\Delta t$')
plt.ylabel("Error")
plt.show()


