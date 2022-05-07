import inspect


# This function was not created by me, but was instead copied from a stackoverflow post
# The link to such post is: https://stackoverflow.com/questions/70313550/is-it-possible-to-get-how-many-positional-arguments-does-a-function-need-in-pyth
# I take no credit for the creation of this function, but I thought the concept of positional argument testing was necessary enough to my code to include this function
def count_positional_args_required(func):
    """
        Counts the number of positional arguments that a given function takes (including the optional *args)

        Parameters:
            func (function):    function who's positional arguments we want to count
    """

    signature = inspect.signature(func)
    empty = inspect.Parameter.empty
    total = 0

    for param in signature.parameters.values():
        if param.default is empty:
            total += 1

    return total
