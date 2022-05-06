import time
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import numpy as np
from numerical_shooting import shooting, phase_condition
import warnings
from ODE_solver import input_test, test_init_conds


def nat_param_continuation(ODE, u0, param_range, vary_par, param_number, solver, discretisation, pc, system):
    """
        Perform natural parameter continuation: increment the parameter by a set value and use the previous solution as
        the initial guess for the next parameter.

            Parameters:
                ODE (function):         the ODE whos root we want to find
                u0 (ndarray):           list of initial x0 and t values
                param_range (ndarray):  range of parameters to run the code on
                vary_par (int):         index of the parameter in param_range to start the code on
                param_number (int):     number of equally spaced parameters to test for within the param_range
                solver:                 solver used
                discretisation:         discretisation method used
                pc (function):          phase condition function
                system (boolean):       True if the ODE is a system of equations, False otherwise

            Returns:
                Array of all the parameter values tested for, as well as the solutions to the equations for each parameter value
        """

    """
    Test all the inputs of the shooting_orbit function are the right type
    """

    input_test(ODE, 'ODE', 'function')
    input_test(discretisation, 'discretisation', 'function')
    input_test(pc, 'pc', 'function')

    input_test(system, 'system', 'boolean')

    test_init_conds(u0, system)

    input_test(param_range, 'param_range', 'list_or_array')

    input_test(vary_par, 'vary_par', 'int_or_float')
    input_test(param_number, 'param_number', 'int_or_float')

    """
    Start the natural parameter continuation
    """

    print(f'Running the natural parameter continuation for the {ODE.__name__} function.')
    start_time = time.time()

    warnings.filterwarnings('ignore')

    first_args = param_range[vary_par]

    last_args = param_range[abs(vary_par - 1)]  # this ensures that if first_args = param_range[0], then last_args = param_range[1], and vice versa

    param_list = np.linspace(first_args, last_args, param_number)

    # solve for the first solution using the first parameter
    first_sol = solver(discretisation(ODE), u0, (pc, first_args))

    sol = np.zeros((param_number, len(u0)))
    sol[0] = first_sol

    # solve for all the remaining solutions, using the previous solution as the initial guess of the next solution
    for i in range(param_number - 1):
        sol[i + 1] = solver(discretisation(ODE), np.round(sol[i], 5), (pc, param_list[i + 1]))

    print(f"Completed in {time.time() - start_time} seconds.\n")

    return sol, param_list


def pseudo_arclength_continuation(ODE, u0, pars, max_pars, vary_par, param_number, discretisation, solver, pc, system):
    """
         Perform natural parameter continuation: increment the parameter by a set value and use the previous solution as
         the initial guess for the next parameter.

             Parameters:
                 ODE (function):         the ODE whos root we want to find
                 u0 (ndarray):           list of initial x0 and t values
                 pars (list):            list of the intial parameter to test for
                 max_pars (int):         value of the maximum parameter to test for
                 vary_par (int):         index of the parameter in param_range to start the code on
                 param_number (int):     number of equally spaced parameters to test for within the param_range
                 solver:                 solver used
                 discretisation:         discretisation method used
                 pc (function):          phase condition function
                 system (boolean):       True if the ODE is a system of equations, False otherwise

             Returns:
                 Array of all the parameter values tested for, as well as the solutions to the equations for each parameter value
         """

    """
    Test all the inputs of the shooting_orbit function are the right type
    """

    input_test(ODE, 'ODE', 'function')
    input_test(discretisation, 'discretisation', 'function')
    input_test(pc, 'pc', 'function')

    input_test(system, 'system', 'boolean')

    test_init_conds(u0, system)

    input_test(pars, 'pars', 'list_or_array')

    input_test(vary_par, 'vary_par', 'int_or_float')
    input_test(max_pars, 'max_pars', 'int_or_float')
    input_test(param_number, 'param_number', 'int_or_float')

    """
    Start the pseudo arclength continuation code
    """

    print(f'Running the pseudo arc length continuation for the {ODE.__name__} function.')

    start_time = time.time()

    warnings.filterwarnings('ignore')

    param_list = np.linspace(pars[vary_par], max_pars, param_number)

    # calculate the first solution, and find the second parameter to solve using
    if system:
        v0 = solver(discretisation(ODE), u0, (pc, param_list[-1]))
        pars[vary_par] = param_list[-2]
        par_list = [param_list[-1], param_list[-2]]
    else:
        v0 = solver(discretisation(ODE), u0, (pc, param_list[0]))
        pars[vary_par] = param_list[1]
        par_list = [param_list[0], param_list[1]]

    # calculate the second solution
    v1 = solver(discretisation(ODE), np.round(v0, 2), (pc, pars[vary_par]))

    def update_par(pars, vary_par, p):
        """
        Function that updates the pars value depending on the p value
        """

        pars[vary_par] = p

        # if the ODE is a system, the update_pars function needs to return the pars indexed, otherwise the code doesn't run properly
        if system:
            return pars[vary_par]
        else:
            return pars

    solution = [v0, v1]

    i = 0

    # run the pseudo code. Number of times ran is defined by the param_number input
    while i < param_number:

        delta_x = solution[-1] - solution[-2]
        delta_p = par_list[-1] - par_list[-2]

        predicted_x = solution[-1] + delta_x
        predicted_p = par_list[-1] + delta_p

        predicted_state = np.append(predicted_x, predicted_p)

        pars[vary_par] = predicted_state[-1]

        pseudo_sol = solver(lambda state: np.append(discretisation(ODE)(state[:-1], pc, update_par(pars, vary_par, state[-1])), np.dot(state[:-1] - predicted_x, delta_x) + np.dot(state[-1] - predicted_p, delta_p)), predicted_state)

        solution.append(np.round(pseudo_sol[:-1], 3))
        par_list.append(pseudo_sol[-1])

        i += 1

    # if the ODE is a system, the solution values need to be unpacked before being able to be plotted
    if system:
        sol_list = []
        for i in range(len(solution)):
            sol_list.append(solution[i][0])
        solution = sol_list

    print(f"Completed in {time.time() - start_time} seconds.\n")

    return solution, par_list


def main():

    def cubic(x, t, args):
        """
        Function for cubic equation
            Parameters:
                x:      initial condition
                t:      t value
                args:   any additional arguments that ODE expects

            Returns:
                The cubic equation
        """
        c = args
        eq = x ** 3 - x + c
        return eq

    """
    We start by calculating the natural parameter and pseudo arclength continuation for the cubic function above.
    """

    # We define the following conditions to run for both methods
    c_interval = np.array([-2, 2])
    u0 = np.array([1])
    pc = phase_condition

    # Running both the natural parameter and the pseudo arclength continuation code
    cubic_sol, cubic_param_list = nat_param_continuation(cubic, u0, c_interval, 0, 10000, fsolve, lambda x: x, pc, False)
    sol, par = pseudo_arclength_continuation(cubic, u0, [-2], 2, 0, 50, lambda x: x, fsolve, pc, False)

    plt.plot(cubic_param_list, cubic_sol, label = 'Natural Parameter')
    plt.plot(par, sol, label = 'Pseudo Arc Length')
    plt.xlabel('c')
    plt.legend()
    plt.show()

    """
    The plot shows us that at approximately c = 0.38 there is a fold, which the natural parameter continuation cannot 
    overcome, yet the pseudo arc length code can, evidenced by how the curve folds 
    """

    """
    Next, we solve the Hopf bifurcation function with both continuation methods 
    """

    def Hopf_bif(U, t, args):
        """
        Function for Hopf bifurcation normal form equation
            Parameters:
                U:      initial conditions
                t:      t value
                args:   any additional arguments that ODE expects

            Returns:
                An array of the du1/dt and du2/dt
        """
        beta = args
        u1, u2 = U

        du1dt = beta * u1 - u2 - u1 * (u1 ** 2 + u2 ** 2)
        du2dt = u1 + beta * u2 - u2 * (u1 ** 2 + u2 ** 2)
        dudt = np.array([du1dt, du2dt])

        return dudt

    # We define the following conditions
    beta_interval = np.array([-1, 2])
    u0 = np.array([1.2, 1.2, 6.4])
    pc = phase_condition

    hopf_sol, hopf_param_list = nat_param_continuation(Hopf_bif, u0, beta_interval, 1, 50, fsolve, shooting, pc, True)
    pseudo_hopf_sol, pseudo_hopf_param = pseudo_arclength_continuation(Hopf_bif, u0, [-1], 2, 0, 50, shooting, fsolve, pc, True)

    plt.plot(hopf_param_list, hopf_sol[:, 0], label='Natural Parameter')
    plt.plot(pseudo_hopf_param, pseudo_hopf_sol, label='Pseudo Arc Length')
    plt.xlabel('c')
    plt.legend()
    plt.show()

    """
    The plot shows us that at c = 0 there is a fold, which the natural parameter continuation cannot 
    overcome, yet the pseudo arc length code can, evidenced by how the curve folds 
    """

    """
    Next, we repeat the same process for the modified Hopf bifurcation equations
    """

    def mod_Hopf_bif(U, t, args):
        beta = args
        u1, u2 = U

        du1dt = beta * u1 - u2 + u1 * (u1 ** 2 + u2 ** 2) - u1 * (u1 ** 2 + u2 ** 2) ** 2
        du2dt = u1 + beta * u2 + u2 * (u1 ** 2 + u2 ** 2) - u2 * (u1 ** 2 + u2 ** 2) ** 2
        dudt = np.array([du1dt, du2dt])

        return dudt

    beta_interval = np.array([-1, 2])
    u0 = np.array([1.4, 0, 6.3])
    pc = phase_condition

    mod_hopf_sol, mod_hopf_param_list = nat_param_continuation(mod_Hopf_bif, u0, beta_interval, 1, 50, fsolve, shooting, pc, True)
    pseudo_mod_hopf_sol, pseudo_mod_hopf_param = pseudo_arclength_continuation(mod_Hopf_bif, u0, [-1], 2, 0, 50, shooting, fsolve, pc, True)

    plt.plot(mod_hopf_param_list, mod_hopf_sol[:, 0], label='Natural Parameter')
    plt.plot(pseudo_mod_hopf_param, pseudo_mod_hopf_sol, label='Pseudo Arc Length')
    plt.xlabel('c')
    plt.legend()
    plt.show()

    """
    The plot shows us that at c = 0 there is a fold, which the natural parameter continuation cannot 
    overcome, yet the pseudo arc length code can, evidenced by how the curve folds 
    """


if __name__ == '__main__':
    main()
