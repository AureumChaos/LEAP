#!/usr/bin/env python3
"""
    Defines miscellaneous utility functions.

    print_list : for pretty printing a list when pprint isn't sufficient.
"""


def inc_generation(context, callbacks=[]):
    """ This tracks the current generation

    The `context` is used to report the current generation, though that
    can also be given by inc_generation.generation().

    This will optionally call all the given callback functions whenever the
    generation is incremented. The registered callback functions should have a signature f(int),
    where the int is the new generation.

    TODO Should we make core.context the default?

    >>> import core
    >>> my_inc_generation = inc_generation(core.context)

    :param context: will set ['leap']['generation'] to the incremented generation
    :param callbacks: optional list of callback function to call when a generation changes
    generation is incremented
    :return:
    """
    curr_generation  = 0
    context = context
    context['leap']['generation'] = 0
    callbacks = callbacks

    def generation():
        return curr_generation

    def do_increment():
        nonlocal curr_generation
        nonlocal context
        nonlocal callbacks
        curr_generation += 1

        # Update the context
        context['leap']['generation'] = curr_generation

        # Now echo the new generation to all the registered callbacks.
        # TODO There is probably a more pythonic way to do this
        [f(curr_generation) for f in callbacks]

        return curr_generation

    do_increment.generation = generation

    return do_increment


def print_list(l):
    """
    Return a string representation of a list.

    This uses __str__() to resolve the elements of the list:

    >>> from leap.core import Individual
    >>> l = [Individual([0, 1, 2]), Individual([3, 4, 5])]
    >>> print_list(l)
    [[0, 1, 2], [3, 4, 5]]

    As opposed to the standard printing mechanism, which calls __repr__() on the elements to produce

    >>> print(l)
    [Individual([0, 1, 2], None, None), Individual([3, 4, 5], None, None)]

    :param l:
    :return:
    """
    print('[' + ', '.join([x.__str__() for x in l]) + ']')


if __name__ == '__main__':
    pass
