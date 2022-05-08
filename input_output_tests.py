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
