import numpy as np
import pylab as pl
from math import pi
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# Set problem parameters/functions
kappa = 1.0  # diffusion constant
L = 1.0  # length of spatial domain
T = 0.5  # total time to solve for


def u_I(x):
    # initial temperature distribution
    y = np.sin(pi * x / L)
    return y


def new_u_I(x, p):
    y = np.sin(pi * x) ** p
    return y


def u_exact(x, t):
    # the exact solution
    y = np.exp(-kappa * (pi ** 2 / L ** 2) * t) * np.sin(pi * x / L)
    return y


def left_boundary():
    return 0


def right_boundary():
    return 0


def matrix_form(method, lmbda, mx):
    if method == 'FE':
        diag = [[lmbda] * (mx - 1), [1 - 2 * lmbda] * mx, [lmbda] * (mx - 1)]
        FE_matrix = (diags(diag, [-1, 0, 1])).toarray()

        return FE_matrix

    if method == 'BE':
        diag = [[- lmbda] * (mx - 1), [1 + 2 * lmbda] * mx, [- lmbda] * (mx - 1)]
        BE_matrix = diags(diag, [-1, 0, 1])

        return BE_matrix

    if method == 'CN':
        diag = [[-lmbda / 2] * (mx - 1), [1 + lmbda] * mx, [-lmbda / 2] * (mx - 1)]
        CN_matrix1 = diags(diag, [-1, 0, 1])
        diag = [[lmbda / 2] * (mx - 1), [1 - lmbda] * mx, [lmbda / 2] * (mx - 1)]
        CN_matrix2 = diags(diag, [-1, 0, 1])

        return CN_matrix1, CN_matrix2


def numerical_initialisation(boundary, initial_cond, p):
    # Set numerical parameters
    mx = 20  # number of gridpoints in space
    mt = 1000  # number of gridpoints in time

    # Set up the numerical environment variables
    if boundary == 'periodic':
        x = np.linspace(0, L, mx)  # mesh points in space
    else:
        x = np.linspace(0, L, mx + 1)
    t = np.linspace(0, T, mt + 1)  # mesh points in time
    deltax = x[1] - x[0]  # gridspacing in x
    deltat = t[1] - t[0]  # gridspacing in t
    lmbda = kappa * deltat / (deltax ** 2)  # mesh fourier number
    # print("deltax=", deltax)
    # print("deltat=", deltat)
    # print("lambda=", lmbda)

    # Set up the solution variables
    u_j = np.zeros(x.size)  # u at current time step
    u_jp1 = np.zeros(x.size)  # u at next time step

    # Set initial condition
    if boundary == 'periodic':
        for i in range(0, mx):
            if initial_cond == new_u_I:
                u_j[i] = initial_cond(x[i], p)
            else:
                u_j[i] = initial_cond(x[i])
    else:
        for i in range(0, mx + 1):
            if initial_cond == new_u_I:
                u_j[i] = initial_cond(x[i], p)
            else:
                u_j[i] = initial_cond(x[i])

    return x, mx, mt, lmbda, u_j, u_jp1


def Forward_Euler(boundary, initial_cond, p):

    if boundary == 'dirichlet':

        x, mx, mt, lmbda, u_j, u_jp1 = numerical_initialisation('not periodic', initial_cond, p)

        AFE = matrix_form('FE', lmbda, mx - 1)
        add_vec = np.zeros(mx - 1)

        for j in range(0, mt):
            add_vec[0] = left_boundary()
            add_vec[-1] = right_boundary()

            u_jp1 = np.dot(AFE, u_j[1:mx]) + add_vec * lmbda

            u_j[0] = left_boundary()
            u_j[1:mx] = u_jp1
            u_j[mx] = right_boundary()

    elif boundary == 'periodic':

        x, mx, mt, lmbda, u_j, u_jp1 = numerical_initialisation('periodic', initial_cond, p)

        AFE = matrix_form('FE', lmbda, mx)
        AFE[0, mx - 1] = lmbda
        AFE[mx - 1, 0] = lmbda

        for j in range(0, mt):

            u_jp1 = np.dot(AFE, u_j[:mx])

            u_j[:mx] = u_jp1

    elif boundary == 'neumann':

        x, mx, mt, lmbda, u_j, u_jp1 = numerical_initialisation('not periodic', initial_cond, p)

        AFE = matrix_form('FE', lmbda, mx + 1)
        AFE[0, 1] = 2 * lmbda
        AFE[-1, -2] = 2 * lmbda
        add_vec = np.zeros(mx + 1)

        for j in range(0, mt):
            add_vec[0] = - left_boundary()
            add_vec[-1] = right_boundary()

            u_j = np.dot(AFE, u_j) + add_vec * lmbda * (x[1] - x[0])

    else:
        raise ValueError(f"boundary value should be 'dirichlet' or 'periodic' but was '{boundary}' instead. Please change this.")

    return x, u_j


def Backwards_Euler(initial_cond, p):
    x, mx, mt, lmbda, u_j, u_jp1 = numerical_initialisation('not periodic', initial_cond, p)

    ABE = matrix_form('BE', lmbda, mx - 1)

    add_vec = np.zeros(mx - 1)

    for j in range(0, mt):
        add_vec[0] = left_boundary()
        add_vec[-1] = right_boundary()

        u_jp1 = spsolve(ABE, u_j[1:mx]) + add_vec * lmbda

        u_j[0] = left_boundary()
        u_j[1:mx] = u_jp1
        u_j[mx] = right_boundary()

    return x, u_j


def Crank_Nicholson(initial_cond, p):
    x, mx, mt, lmbda, u_j, u_jp1 = numerical_initialisation('not periodic', initial_cond, p)

    ACN, BCN = matrix_form('CN', lmbda, mx - 1)

    add_vec = np.zeros(mx - 1)

    for j in range(0, mt):

        add_vec[0] = left_boundary()
        add_vec[-1] = right_boundary()

        u_jp1 = spsolve(ACN, BCN * u_j[1:mx]) + add_vec * lmbda

        u_j[0] = left_boundary()
        u_j[1:mx] = u_jp1
        u_j[mx] = right_boundary()

    return x, u_j


def plot():
    xx = np.linspace(0, L, 250)
    pl.plot(xx, u_exact(xx, T), 'b-', label='exact')
    pl.xlabel('x')
    pl.ylabel('u(x,0.5)')
    pl.legend(loc='upper right')


def main():
    """
    We run the code using our new initial condition function, and see what happens when we vary the p parameter
    """
    for i in range(10):
        FE_x, FE_u_j = Forward_Euler('dirichlet', new_u_I, i)
        pl.plot(FE_x, FE_u_j, label=f'p = {i}')

    plot()
    pl.show()

    """
    As shown by the above plot, when the p value is raised, the curve becomes less and less similar to the true solution
    """

    """
    Next, we plot the three different methods: Forward Euler, Backwards Euler and Crank Nicholson to compare the three
    """

    FE_x, FE_u_j = Forward_Euler('dirichlet', u_I, None)
    BE_x, BE_u_j = Backwards_Euler(u_I, None)
    CN_x, CN_u_j = Crank_Nicholson(u_I, None)

    pl.plot(FE_x, FE_u_j, 'o', label='Forward Euler')
    pl.plot(BE_x, BE_u_j, 'o', label='Backwards Euler')
    pl.plot(CN_x, CN_u_j, 'o', label='Crank Nicholson')

    plot()
    pl.show()

    """
    Finally, the code allows three different boundary conditions to be run: 'dirichlet', 'periodic' and 'neumann'
    It seems that the results found in the plots below are inaccurate, but it is unclear why this is, as the boundary 
    conditions are thought to have been well implemented. 
    """

    FE_x, FE_u_j = Forward_Euler('neumann', u_I, None)

    pl.plot(FE_x, FE_u_j, 'ro', label='num')

    plot()
    pl.show()

    FE_x, FE_u_j = Forward_Euler('periodic', u_I, None)

    pl.plot(FE_x, FE_u_j, 'ro', label='num')

    plot()
    pl.show()


if __name__ == "__main__":
    main()
