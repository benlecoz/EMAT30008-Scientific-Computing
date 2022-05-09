from numerical_continuation import continuation, phase_condition
import numpy as np
from scipy.optimize import fsolve


def continuation_tests():

    """
    Run tests on the inputs and outputs of the numerical shooting code
    """

    """
    Define correct values for all parameters, to be used as standard practice while testing other inputs
    """

    # Correct ODE to solve the continuation problem for
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

    param_range = np.array([-2, 2])  # Correct parameter range
    vary_par = 0  # Correct index of parameter to vary
    u0 = np.array([1])  # Correct initial conditions x0 and t
    param_number = 50  # Correct number of parameters to test for
    pc = phase_condition  # Correct phase condition
    solver = fsolve  # Correct solver to use
    discretisation = lambda x: x  # Correct discretisation
    system = False

    # create lists for both failed and passed tests
    failed_tests = []
    passed_tests = []

    # wrong method type
    try:
        continuation(1, cubic, u0, param_range, vary_par, param_number, solver, discretisation, pc, False)
        failed_tests.append('method wrong type')
    except TypeError:
        passed_tests.append('method wrong type')

    # wrong method value
    try:
        continuation('natural', cubic, u0, param_range, vary_par, param_number, solver, discretisation, pc, False)
        failed_tests.append('method wrong value')
    except ValueError:
        passed_tests.append('method wrong value')

    # wrong ODE type
    try:
        continuation('nat', 'not a function', u0, param_range, vary_par, param_number, solver, discretisation, pc, False)
        failed_tests.append('method wrong type')
    except TypeError:
        passed_tests.append('method wrong type')

    # wrong ODE output type
    try:
        continuation('nat', lambda x, t: 'wrong output type', u0, param_range, vary_par, param_number, solver, discretisation, pc, False)
        failed_tests.append('ODE wrong output type')
    except TypeError:
        passed_tests.append('ODE wrong output type')

    # wrong ODE output length
    try:
        continuation('nat', lambda x, t: np.array([x, t, x]), u0, param_range, vary_par, param_number, solver, discretisation, pc, False)
        failed_tests.append('ODE wrong output length')
    except TypeError:
        passed_tests.append('ODE wrong output length')

    # ODE with too many positional arguments
    try:
        continuation('nat', lambda x, y, z, a: np.array([x, x]), u0, param_range, vary_par, param_number, solver,
                     discretisation, pc, False)
        failed_tests.append('ODE too many positional arguments')
    except IndexError:
        passed_tests.append('ODE too many positional arguments')

    # ODE with too many positional arguments
    try:
        continuation('nat', lambda x: np.array([x, x]), u0, param_range, vary_par, param_number, solver,
                     discretisation, pc, False)
        failed_tests.append('ODE not enough positional arguments')
    except IndexError:
        passed_tests.append('ODE not enough positional arguments')

    # wrong u0 type
    try:
        continuation('nat', cubic, 'initial conditions', param_range, vary_par, param_number, solver, discretisation, pc, False)
        failed_tests.append('u0 wrong type')
    except TypeError:
        passed_tests.append('u0 wrong type')

    # wrong param_range type
    try:
        continuation('nat', cubic, u0, 'param_range', vary_par, param_number, solver, discretisation, pc, False)
        failed_tests.append('param_range wrong type')
    except TypeError:
        passed_tests.append('param_range wrong type')

    # wrong vary_par type
    try:
        continuation('nat', cubic, u0, param_range, 'vary_par', param_number, solver, discretisation, pc, False)
        failed_tests.append('vary_par wrong type')
    except TypeError:
        passed_tests.append('vary_par wrong type')

    # wrong vary_par value
    try:
        continuation('nat', cubic, u0, param_range, 2, param_number, solver, discretisation, pc, False)
        failed_tests.append('vary_par wrong value')
    except ValueError:
        passed_tests.append('vary_par wrong value')

    # wrong param_number type
    try:
        continuation('nat', cubic, u0, param_range, vary_par, 'param_number', solver, discretisation, pc, False)
        failed_tests.append('param_number wrong type')
    except TypeError:
        passed_tests.append('param_number wrong type')

    # wrong discretisation type
    try:
        continuation('nat', cubic, u0, param_range, vary_par, param_number, solver, 'discretisation', pc, False)
        failed_tests.append('discretisation wrong type')
    except TypeError:
        passed_tests.append('discretisation wrong type')

    # wrong pc type
    try:
        continuation('nat', cubic, u0, param_range, vary_par, param_number, solver, discretisation, 'pc', False)
        failed_tests.append('pc wrong type')
    except TypeError:
        passed_tests.append('pc wrong type')

    # wrong system type
    try:
        continuation('nat', cubic, u0, param_range, vary_par, param_number, solver, discretisation, pc, 'False')
        failed_tests.append('system wrong type')
    except TypeError:
        passed_tests.append('system wrong type')

    number_of_tests = len(failed_tests) + len(passed_tests)

    if len(passed_tests) != number_of_tests:
        print(f"Not all the tests were passed. {number_of_tests - len(passed_tests)} test(s) was/were failed. \n")
        print('This is the name of the failed test(s).\n')
        for i in failed_tests:
            print(failed_tests)

    else:
        print(f'Congratulations, you have passed all the {number_of_tests} tests for the numerical continuation!')


if __name__ == "__main__":
    continuation_tests()