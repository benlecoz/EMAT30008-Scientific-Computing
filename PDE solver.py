# simple forward Euler solver for the 1D heat equation
#   u_t = kappa u_xx  0<x<L, 0<t<T
# with zero-temperature boundary conditions
#   u=0 at x=0,L, t>0
# and prescribed initial temperature
#   u=u_I(x) 0<=x<=L,t=0

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


def numerical_initialisation():
    # Set numerical parameters
    mx = 10  # number of gridpoints in space
    mt = 1000  # number of gridpoints in time

    # Set up the numerical environment variables
    x = np.linspace(0, L, mx + 1)  # mesh points in space
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
    for i in range(0, mx + 1):
        u_j[i] = u_I(x[i])
        # u_j[i] = new_u_I(x[i], 1/2)

    return x, mx, mt, lmbda, u_j, u_jp1


def Forward_Euler():

    x, mx, mt, lmbda, u_j, u_jp1 = numerical_initialisation()

    AFE = matrix_form('FE', lmbda, mx - 1)

    add_vec = np.zeros(mx - 1)

    for j in range(0, mt):

        add_vec[0] = 1
        add_vec[-1] = 1

        u_jp1 = np.dot(AFE, u_j[1:mx])

        # Boundary conditions
        # u_jp1[0] = 0
        # u_jp1[mx] = 0
        #
        # # Save u_j at time t[j+1]
        # u_j[:] = u_jp1[:]

        u_j[0] = 0
        u_j[1:mx] = u_jp1
        u_j[mx] = 0

    return x, u_j


def Backwards_Euler():

    x, mx, mt, lmbda, u_j, u_jp1 = numerical_initialisation()

    # Solve the PDE: loop over all time points
    for j in range(0, mt):
        # Forward Euler timestep at inner mesh points
        # PDE discretised at position x[i], time t[j]

        BFE = matrix_form('BE', lmbda, mx - 1)

        u_jp1 = spsolve(BFE, u_j[1:mx])

        # Boundary conditions
        u_j[0] = 0
        u_j[1:mx] = u_jp1
        u_j[mx] = 0

        # Save u_j at time t[j+1]
        # u_j[:] = u_jp1[:]

    return x, u_j


# Plot the final result and exact solution
FE_x, FE_u_j = Forward_Euler()
BE_x, BE_u_j = Backwards_Euler()

# pl.plot(FE_x, FE_u_j, 'ro', label='num')
pl.plot(BE_x, BE_u_j, 'ro', label='num')
xx = np.linspace(0, L, 250)
pl.plot(xx, u_exact(xx, T), 'b-', label='exact')
pl.xlabel('x')
pl.ylabel('u(x,0.5)')
pl.legend(loc='upper right')
pl.show()
