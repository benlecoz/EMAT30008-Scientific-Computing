from numerical_shooting import shooting_orbit, phase_condition
import numpy as np
from input_output_tests import shooting_close_to_true

# One of the error trap tests will raise a Visible Deprecation Warning, so this will suppress it
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


def numerical_shooting_tests():
    """
    Run tests on the inputs and outputs of the numerical shooting code
    """

    """
    Define correct values for all parameters, to be used as standard practice while testing other inputs
    """

    # Correct ODE to solve the shooting root finding problem for
    def predator_prey(X, t, args):
        x, y = X
        a, b, d = args[0], args[1], args[2]

        dxdt = x * (1 - x) - (a * x * y) / (d + x)
        dydt = b * y * (1 - y / x)

        dXdt = np.array([dxdt, dydt])

        return dXdt

    u0 = np.array([1.2, 1.2, 6])  # Correct initial conditions x0 and t to use
    pc = phase_condition  # Correct phase condition
    system = True  # Correct system boolean
    args = [1, 0.1, 0.1]  # correct set of arguments for the ODE

    # create lists for both failed and passed tests
    failed_tests = []
    passed_tests = []

    # ODE is wrong type
    try:
        shooting_orbit('not a function', u0, pc, system, args)
        failed_tests.append('ODE is wrong type')
    except TypeError:
        passed_tests.append('ODE is wrong type')

    # system is wrong type
    try:
        shooting_orbit(predator_prey, u0, pc, 'not a boolean', args)
        failed_tests.append('system is wrong type')
    except TypeError:
        passed_tests.append('system is wrong type')

    # u0 is wrong type
    try:
        shooting_orbit(predator_prey, 'not initial conditions', pc, system, args)
        failed_tests.append('u0 is wrong type')
    except TypeError:
        passed_tests.append('u0 is wrong type')

    # one element of u0 is wrong type
    try:
        shooting_orbit(predator_prey, np.array([1, 1, 'int']), pc, system, args)
        failed_tests.append('one element of u0 is wrong type')
    except TypeError:
        passed_tests.append('one element of u0 is wrong type')

    # u0 is has only 1 x0 condition when ODE is a system
    try:
        shooting_orbit(predator_prey, u0[:-1], pc, system, args)
        failed_tests.append('u0 is missing only initial condition')
    except ValueError:
        passed_tests.append('u0 is missing only initial condition')

    # u0 is empty
    try:
        shooting_orbit(predator_prey, [], pc, system, args)
        failed_tests.append('u0 is empty')
    except IndexError:
        passed_tests.append('u0 is empty')

    # u0 has multiple initial conditions for x0 when system is False
    try:
        shooting_orbit(predator_prey, u0, pc, False, args)
        failed_tests.append('u0 does not match single ODE dimension')
    except ValueError:
        passed_tests.append('u0 does not match single ODE dimension')

    # pc is wrong type
    try:
        shooting_orbit(predator_prey, u0, 'not a function', system, args)
        failed_tests.append('pc is wrong type')
    except TypeError:
        passed_tests.append('pc is wrong type')

    # ODE output is wrong type
    try:
        shooting_orbit(lambda x, t: 'wrong output type', u0, pc, True, args)
        failed_tests.append('ODE with wrong output type')
    except TypeError:
        passed_tests.append('ODE with wrong output type')

    # ODE output is wrong length
    try:
        shooting_orbit(lambda x, t, arg: np.array([x, t, arg]), u0, pc, True, args)
        failed_tests.append('ODE with wrong output length')
    except ValueError:
        passed_tests.append('ODE with wrong output length')

    # ODE takes not enough positional arguments
    try:
        shooting_orbit(lambda x: [x, x, x], u0, pc, True, args)
        failed_tests.append('ODE takes not enough positional arguments')
    except IndexError:
        passed_tests.append('ODE takes not enough positional arguments')

    # ODE takes too many positional arguments
    try:
        shooting_orbit(lambda x, y, z, a: [x, x, x], u0, pc, True, args)
        failed_tests.append('ODE takes too many positional arguments')
    except IndexError:
        passed_tests.append('ODE takes too many positional arguments')

    # pc takes not enough positional arguments
    try:
        shooting_orbit(predator_prey, u0, lambda x: [x, x, x], True, args)
        failed_tests.append('pc takes not enough positional arguments')
    except IndexError:
        passed_tests.append('pc takes not enough positional arguments')

    # pc takes too many positional arguments
    try:
        shooting_orbit(predator_prey, u0, lambda x, y, z, a: [x, x, x], True, args)
        failed_tests.append('pc takes too many positional arguments')
    except IndexError:
        passed_tests.append('pc takes too many positional arguments')

    # pc wrong output type
    try:
        shooting_orbit(predator_prey, u0, lambda x, y, z: 1, True, args)
        failed_tests.append('pc with wrong output type')
    except TypeError:
        passed_tests.append('pc with wrong output type')

    # pc wrong number of outputs
    try:
        shooting_orbit(predator_prey, u0, lambda x, y, z: np.array([y, y, y, y]), True, args)
        failed_tests.append('pc with wrong output type')
    except ValueError:
        passed_tests.append('pc with wrong output type')

    number_of_tests = len(failed_tests) + len(passed_tests)

    print(passed_tests)

    if len(passed_tests) != number_of_tests:
        print(f"Not all the tests were passed. {number_of_tests - len(passed_tests)} test(s) was/were failed. \n")
        print('This is the name of the failed test(s).\n')
        for i in failed_tests:
            print(failed_tests)

    else:
        print(f'Congratulations, you have passed all the {number_of_tests} tests for the numerical shooting!')


def value():

    def Hopf_bif(U, t, args):
        beta = args[0]
        sigma = args[1]

        u1, u2 = U
        du1dt = beta * u1 - u2 + sigma * u1 * (u1 ** 2 + u2 ** 2)
        du2dt = u1 + beta * u2 + sigma * u2 * (u1 ** 2 + u2 ** 2)
        dudt = np.array([du1dt, du2dt])

        return dudt

    def true_Hopf_bif(t, args):
        beta = args[0]
        phase = args[1]

        u1 = np.sqrt(beta) * np.cos(t + phase)
        u2 = np.sqrt(beta) * np.sin(t + phase)

        return np.array([u1, u2])

    shooting_close_to_true(shooting_orbit, Hopf_bif, true_Hopf_bif, [1.2, 1.2, 8], phase_condition, True, [1, -1])


if __name__ == "__main__":
    # numerical_shooting_tests()
    value()