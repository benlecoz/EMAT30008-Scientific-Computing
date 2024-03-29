import numpy as np
from positional_arguments_checker import count_positional_args_required


def input_test(test, test_name, test_type):
    """
    Tests the type of a specific parameter

        Parameters:
            test (Any):         parameter tested
            test_name (str):    name of parameter tested
            test_type (str):    type that the parameter should be, either 'int_or_float', 'string', 'function', 'boolean', or 'list_or_array'

        Returns:
            An error with a description of the issue with the type of the parameter if the test is failed
    """

    def int_or_float():
        if not isinstance(test, (float, int)) and type(test) != np.int_ and type(test) != np.float_:
            raise TypeError(f"The argument passed for {test_name} is not a float or an integer, but a {type(test)}. Please input an integer or a float")

    def function():
        if not callable(test):
            raise TypeError(f"The argument passed for {test_name} is not a function, but a {type(test)}. Please input a function")

    def string():
        if not isinstance(test, str):
            raise TypeError(f"The argument passed for {test_name} is not a string, but a {type(test)}. Please input a string")

    def boolean():
        if not isinstance(test, bool):
            raise TypeError(f"The argument passed for {test_name} is not a boolean, but a {type(test)}. Please input a boolean")

    def list_or_array():
        if not isinstance(test, (list, np.ndarray)):
            raise TypeError(f"The argument passed for {test_name} is not a list or an array, but a {type(test)}. Please input a list or an array")

    if test_type == 'int_or_float':
        int_or_float()

    elif test_type == 'function':
        function()

    elif test_type == 'boolean':
        boolean()

    elif test_type == 'string':
        string()

    elif test_type == 'list_or_array':
        list_or_array()

    else:
        raise ValueError(f"Please input a valid test name. Test options are 'int_or_float', 'string', 'function', 'boolean', or 'list_or_array', while the input was {test_name}.")


def test_init_conds(x0, system):
    """
    Tests the initial conditions

        Parameters:
            x0:             initial conditions to be tested
            system (bool):  True if the ODE is a system of equations, False otherwise

        Returns:
            Raises an error if there is anything wrong with the initial conditions (wrong type/shape)
    """

    # test inputs for the initial x conditions if ODE is a system
    if system:
        # test to make sure x0 is either a list or a numpy array
        input_test(x0, 'x0', 'list_or_array')

        # test to make sure x0 is not empty
        if len(x0) == 0:
            raise IndexError(f"Please do not input an empty x0")

        # test to make sure that there are multiple initial conditions since system is defined as True
        if len(x0) == 1:
            raise ValueError(f"system is defined as True, but there is {len(x0)} initial condition. Please change system to False or input more initial conditions")

        # cycle through all the values defined in x0 and test if there are the right type
        for x in range(len(x0)):
            input_test(x0[x], 'x0', 'int_or_float')

    # test inputs for the initial x conditions if ODE is NOT a system
    else:
        # if x0 is a list or a np.array, only allow a length of 1
        if isinstance(x0, (list, np.ndarray)):

            if len(x0) > 1:
                raise ValueError(
                    f"system is defined as False, but there is {len(x0)} initial conditions. Please change system to True or input only one initial condition")

            elif len(x0) == 0:
                raise IndexError(f"Please do not input an empty x0")

            elif len(x0) == 1:
                input_test(x0[0], 'x0', 'int_or_float')

        # if x0 is a single input, check if that it is the right type
        else:
            input_test(x0, 'x0', 'int_or_float')


def test_func_output(ODE, x0, t, system, *args):
    """
    Tests the output of a function

        Parameters:
            ODE (function): the ODE we want to test
            x0:             initial conditions to be tested
            t:              time value(s)
            system (bool):  True if the ODE is a system of equations, False otherwise
            *args:          any additional arguments that the ODE function defined above expects

        Returns:
            Raises an error if there is anything wrong with the output of the function (wrong type/shape)
    """

    pos_args = count_positional_args_required(ODE)

    if pos_args != 2 and pos_args != 3:
        raise IndexError(f"pc function needs to allow either 2 or 3 positional arguments: x0, t and args (optional). Yet, this pc function allowed {pos_args} positional argument(s).")

    test_output = ODE(x0, t, *args)

    if not isinstance(test_output, (int, np.ndarray, float)):
        raise TypeError(f"Output of the ODE has to be int/ndarray/float, but was {type(test_output)} instead.")

    if system:
        if len(test_output) != len(x0):
            raise ValueError(f"Output of the ODE has length {len(test_output)}, while the initial conditions has length {len(x0)}.")
    else:
        if not isinstance(test_output, (float, int)) and type(test_output) != np.int_ and type(test_output) != np.float_:
            if len(test_output) > 1:
                raise ValueError(f"Output of the ODE has length {len(test_output)}, while the initial conditions has length 1.")


def test_pc_output(ODE, u0, pc, *args):
    """
    Tests the output of a phase condition

        Parameters:
            ODE (function): the ODE that the phase condition is defined on
            u0:             initial conditions x0 and t
            pc (function):  phase condition we want to test
            system (bool):  True if the ODE is a system of equations, False otherwise
            *args:          any additional arguments that the ODE function defined above expects

        Returns:
            Raises an error if there is anything wrong with the output of the phase condition (wrong type/shape)
    """

    pos_args = count_positional_args_required(pc)

    if pos_args != 2 and pos_args != 3:
        raise IndexError(f"pc function needs to allow either 2 or 3 positional arguments: ODE, x0 and args (optional). Yet, this pc function allowed {pos_args} positional argument(s).")

    # assign the output of the pc function to a test variable
    test_output = pc(ODE, u0, *args)

    # ensure that the output of the pc function is not a single int/float, else the len() function wont work
    if not isinstance(test_output, (float, int)) and type(test_output) != np.int_ and type(test_output) != np.float_:

        # test to see if the output is 3 objects (x0, t and phase_con)
        if len(test_output) != 3:
            raise ValueError(f"The phase condition function needs to have 3 outputs: x0, t and phase_con, while {ODE.__name__} has {len(test_output)}.")

    else:
        raise TypeError(f"Output of the phase condition function needs to be multiple objects, not an int/float")


def ODE_close_to_true(ODE_solver, ODE, ODE_true, u0, method_name, system, *args):
    """
    Finds the accuracy of a ODE solver solution compared to the true solution

        Parameters:
            ODE_solver (function):  this is solve_ode, but to avoid circular imports, the solve_ode function could not be imported into this code
            ODE (function):         the ODE to test the solutions on
            ODE_true (function):    the true solution function of the ODE
            u0:                     initial conditions x0 and t1
            method_name (str):      name of the method to use, either 'euler' or 'RK4'
            system (bool):          True if the ODE is a system of equations, False otherwise
            *args:                  any additional arguments that the ODE function defined above expects

        Returns:
            The accuracy of the true solution, as a power of 10.
    """
    empirical_sol = ODE_solver(ODE, u0[:-1], 0, u0[-1], method_name, 0.01, system, *args)
    true_sol = ODE_true(u0[-1], *args)

    tolerance_values = [10**x for x in list(range(-10, 1, 1))]

    proximity = False
    i = 0

    while not proximity:
        if system:
            proximity = np.allclose(empirical_sol[0][-1, :], true_sol, rtol=tolerance_values[i])

        else:
            proximity = np.isclose(empirical_sol[0][-1], true_sol, rtol=tolerance_values[i])

        i += 1

        if i == 11:
            raise ValueError(f"The empirical solution was never close to the true solution, to a tolerance of 1. Please try again with a different ODE or check your functions.")

    return tolerance_values[i - 1]


def shooting_close_to_true(shooting_code, ODE, true_ODE, u0, pc, system, *args):
    """
    Finds the accuracy of a shooting solution compared to the true solution

        Parameters:
            shooting_code (function):   this is the shooting function, but to avoid circular imports, the function could not be imported into this code
            ODE (function):             the ODE to test the solutions on
            true_ODE (function):        the true solution function of the ODE
            u0:                         initial conditions x0 and t
            pc (function):              the phase condition
            system (bool):              True if the ODE is a system of equations, False otherwise
            *args:                      any additional arguments that the ODE function defined above expects

        Returns:
            The accuracy of the true solution, as a power of 10.
    """
    empirical_sol = shooting_code(ODE, u0, pc, system, False, *args)
    true_sol = true_ODE(u0[-1], *args)

    tolerance_values = [10**x for x in list(range(-10, 1, 1))]

    proximity = False
    i = 0

    while not proximity:
        if system:
            proximity = np.allclose(empirical_sol[0][-1, :], true_sol, rtol=tolerance_values[i])

        else:
            proximity = np.isclose(empirical_sol[0][-1], true_sol, rtol=tolerance_values[i])

        i += 1

        if i == 11:
            raise ValueError(f"The empirical solution was never close to the true solution, to a tolerance of 1. Please try again with a different ODE or check your functions.")

    return tolerance_values[i - 1]




