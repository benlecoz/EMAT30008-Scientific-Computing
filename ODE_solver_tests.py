from ODE_solver import solve_ode
import numpy as np
from input_output_tests import ODE_close_to_true

# One of the error trap tests will raise a Visible Deprecation Warning, so this will suppress it
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


def ODE_solver_input_output_tests():
    """
    Run tests on the inputs and outputs of the ODE solver
    """

    """
    Define correct values for all parameters, to be used as standard practice while testing other inputs
    """

    # Correct single ODE
    def FO_f(x, t):
        dxdt = np.array([x])

        return dxdt

    # Correct system of ODE
    def SO_f(u, t):
        x, y = u
        dXdt = np.array([y, -x])

        return dXdt

    single_x0 = [1]  # Correct single x0
    system_x0 = [1, 1]  # Correct system of x0

    t0 = 0  # Correct t0
    t1 = 1  # Correct t1

    method_name = 'euler'  # Correct method name

    deltat_max = 0.01  # Correct deltat max

    """
    Start running the tests with intentionally bad inputs/outputs to make sure it can graciously handle wrong types and values 
    """

    # Create a failed test list that will collect the names of all the failed tests, and same for the passed tests
    failed_tests = []
    passed_tests = []

    # ODE is not a function
    try:
        solve_ode('not a function', single_x0, t0, t1, method_name, deltat_max, False)
        failed_tests.append('ODE is not a function')
    except TypeError:
        passed_tests.append('ODE is not a function')

    # wrong type of x0
    try:
        solve_ode(FO_f, 'not an initial condition', t0, t1, method_name, deltat_max, False)
        failed_tests.append('x0 is not a int/float')
    except TypeError:
        passed_tests.append('x0 is not a int/float')

    # x0 where one condition is not the right type
    try:
        solve_ode(SO_f, [1, 'int'], t0, t1, method_name, deltat_max, True)
        failed_tests.append('one value of x0 is not a int/float')
    except TypeError:
        passed_tests.append('one value of x0 is not a int/float')

    # single x0 when ODE is a system of equations
    try:
        solve_ode(SO_f, single_x0, t0, t1, method_name, deltat_max, True)
        failed_tests.append('single x0 for system ODE')
    except ValueError:
        passed_tests.append('single x0 for system ODE')

    # multiple x0 when ODE is single equation
    try:
        solve_ode(FO_f, system_x0, t0, t1, method_name, deltat_max, False)
        failed_tests.append('system x0 for single ODE')
    except ValueError:
        passed_tests.append('system x0 for single ODE')

    # empty list for x0 when system is False
    try:
        solve_ode(FO_f, [], t0, t1, method_name, deltat_max, False)
        failed_tests.append('empty x0 for single')
    except IndexError:
        passed_tests.append('empty x0 for single')

    # empty list for x0 when system is True
    try:
        solve_ode(FO_f, [], t0, t1, method_name, deltat_max, True)
        failed_tests.append('empty x0 for system')
    except IndexError:
        passed_tests.append('empty x0 for system')

    # wrong type for t0
    try:
        solve_ode(FO_f, single_x0, 'not a int/float', t1, method_name, deltat_max, False)
        failed_tests.append('t0 is not an int/float')
    except TypeError:
        passed_tests.append('t0 is not an int/float')

    # wrong type for t1
    try:
        solve_ode(FO_f, single_x0, t0, 'not a int/float', method_name, deltat_max, False)
        failed_tests.append('t1 is not an int/float')
    except TypeError:
        passed_tests.append('t1 is not an int/float')

    # wrong type for method_name
    try:
        solve_ode(FO_f, single_x0, t0, t1, 1, deltat_max, False)
        failed_tests.append('method_name is not a string')
    except TypeError:
        passed_tests.append('method_name is not a string')

    # wrong value for method_name
    try:
        solve_ode(FO_f, single_x0, t0, t1, 'not euler or RK4', deltat_max, False)
        failed_tests.append('method_name has wrong name')
    except ValueError:
        passed_tests.append('method_name has wrong name')

    # wrong type for deltat_max
    try:
        solve_ode(FO_f, single_x0, t0, t1, method_name, 'not deltat_max', False)
        failed_tests.append('deltat_max is not a int/float')
    except TypeError:
        passed_tests.append('deltat_max is not a int/float')

    # wrong type for system
    try:
        solve_ode(FO_f, single_x0, t0, t1, method_name, deltat_max, 1)
        failed_tests.append('system is not a boolean')
    except TypeError:
        passed_tests.append('system is not a boolean')

    # function that outputs the wrong type
    try:
        solve_ode(lambda x, t: 'not the right output type', single_x0, t0, t1, method_name, deltat_max, False)
        failed_tests.append('function has wrong output type')
    except TypeError:
        passed_tests.append('function has wrong output type')

    # function with single x0 but system output
    try:
        solve_ode(lambda x, t: np.array([x, t]), single_x0, t0, t1, method_name, deltat_max, False)
        failed_tests.append('single x0 but system function output')
    except ValueError:
        passed_tests.append('single x0 but system function output')

    # function with system x0 but single output
    try:
        solve_ode(lambda x, y: np.array([x]), system_x0, t0, t1, method_name, deltat_max, True)
        failed_tests.append('system x0 but single function output')
    except ValueError:
        passed_tests.append('system x0 but single function output')

    # function with not enough positional arguments
    try:
        solve_ode(lambda x: np.array([x]), single_x0, t0, t1, method_name, deltat_max, False)
        failed_tests.append('function with not enough positional arguments')
    except IndexError:
        passed_tests.append('function with not enough positional arguments')

    # function with too many positional arguments
    try:
        solve_ode(lambda x, t, args, pc: np.array([x]), single_x0, t0, t1, method_name, deltat_max, False)
        failed_tests.append('function with too many enough positional arguments')
    except IndexError:
        passed_tests.append('function with too many positional arguments')

    number_of_tests = len(failed_tests) + len(passed_tests)

    if len(passed_tests) != number_of_tests:
        print(f"Not all the tests were passed. {number_of_tests - len(passed_tests)} test(s) was/were failed. \n")
        print('This is the name of the failed test(s).\n')
        for i in failed_tests:
            print(failed_tests)

    else:
        print(f'Congratulations, you have passed all the {number_of_tests} input/output tests for the ODE solver!')


def ODE_solver_value_tests():
    """
    Run tests on the values of the output of the ODE solver compared to the true solutions
    """

    failed_tests = []
    passed_tests = []

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

    def not_FO_true_solution(t):
        """
        Not the True solution to the first order ODE dxdt = x defined above

        """
        x = np.exp(t) * np.exp(t)

        return x

    try:
        ODE_close_to_true(solve_ode, FO_f, not_FO_true_solution, [1, 1], 'euler', False)
        failed_tests.append('euler value testing for wrong function')
    except ValueError:
        passed_tests.append('euler value testing for wrong function')

    try:
        ODE_close_to_true(solve_ode, FO_f, not_FO_true_solution, [1, 1], 'euler', False)
        failed_tests.append('RK4 value testing for wrong function')
    except ValueError:
        passed_tests.append('RK4 value testing for wrong function')

    try:
        ODE_close_to_true(solve_ode, FO_f, FO_true_solution, [1, 1], 'euler', False)
        passed_tests.append('euler value testing for right function')
    except ValueError:
        failed_tests.append('euler value testing for right function')

    try:
        ODE_close_to_true(solve_ode, FO_f, FO_true_solution, [1, 1], 'RK4', False)
        passed_tests.append('RK4 value testing for right function')
    except ValueError:
        failed_tests.append('RK4 value testing for right function')

    number_tests = len(passed_tests) + len(failed_tests)

    if len(passed_tests) == number_tests:
        print(f'The ODE solver has passed all the {number_tests} Value tests!')
    else:
        print('The ODE solver has failed some of the value tests:')
        print(failed_tests)


if __name__ == "__main__":
    ODE_solver_input_output_tests()
    ODE_solver_value_tests()

