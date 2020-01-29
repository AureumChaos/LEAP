#!/usr/bin/env python3
"""
    Defines miscellaneous utility functions.

    print_list : for pretty printing a list when pprint isn't sufficient.
"""
import collections
import itertools
import inspect

# ##############################
# Function is_sequence
# ##############################
def is_sequence(obj):
    """ :return: True if obj is a sequence

        Cribbed from https://stackoverflow.com/questions/2937114/python-check-if-an-object-is-a-sequence?lq=1

        E.g., used to determine if gaussian mutation has a single specified
        standard deviation, or a vector of standard deviations.

        >>> is_sequence(0.5)
        False

        >>> is_sequence([0.1, 0.2, 0.3])
        True
    """
    return isinstance(obj, collections.abc.Sequence)


def is_iterable(obj):
    """
    :param obj: that we want to determine is a generator
    :return: True if obj can use next(obj)
    """
    return inspect.isgenerator(obj) or inspect.isgeneratorfunction(obj)


# ##############################
# Function inc_generation
# ##############################
def inc_generation(context, callbacks=()):
    """ This tracks the current generation

    The `context` is used to report the current generation, though that
    can also be given by inc_generation.generation().

    This will optionally call all the given callback functions whenever the
    generation is incremented. The registered callback functions should have a signature f(int),
    where the int is the new generation.

    TODO Should we make core.context the default?

    >>> from leap import core
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


# ##############################
# Function print_list
# ##############################
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


# ##############################
# Function birth_brander
# ##############################
def birth_brander():
    """ This pipeline operator will add or update a "birth" attribute for
    passing individuals.

    If the individual already has a birth, just let it float by with the
    original value.  If it doesn't, assign the individual the current birth
    ID, and then increment the global, stored birth count.

    We don't increment a birth ID in the ctor because that overall birth
    count will bloat due to clone operations.  Inserting this operator into
    the pipeline will ensure that each individual that passes through is
    properly "branded" with a unique birth ID.  However, care must be made to
    ensure that the initial population is similarly branded.

    :param next_thing: preceding individual in the pipeline
    :return: branded individual
    """
    # incremented with each birth
    num_births = itertools.count()

    # sometimes next_thing is a population, so we need this to track that
    # the next individual in the population
    iterator = None

    def do_birth_branding(next_thing):
        """

        :param next_thing: either the next individual in the pipeline or a population of individuals to be branded
        :return: branded individual
        """
        nonlocal num_births
        nonlocal iterator

        while True:
            if is_iterable(next_thing):
                # We're being passed in a single individual in a pipeline
                next_thing = next(next_thing)
            else:
                # We're being passed a sequence/population
                if iterator is None:
                    iterator = iter(next_thing)
                next_thing = next(iterator)

            if not hasattr(next_thing, "birth"):
                # Only assign a birth ID if they don't already have one
                next_thing.birth = next(num_births)

            yield next_thing

    return do_birth_branding


if __name__ == '__main__':
    pass
