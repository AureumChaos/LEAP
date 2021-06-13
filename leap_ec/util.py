#!/usr/bin/env python3
"""
    Defines miscellaneous utility functions.

    print_list : for pretty printing a list when pprint isn't sufficient.
"""
import collections
import itertools
import inspect

from leap_ec.global_vars import context


###############################
# Function print_population
###############################
def print_population(population, generation):
    """ Convenience function for pretty printing a population that's
    associated with a given generation

    :param population:
    :param generation:
    :return: None
    """
    for individual in population:
        print(generation, individual.genome, individual.fitness)


###############################
# Function is_sequence
###############################
def is_sequence(obj):
    """ :return: True if obj is a test_sequence

        Cribbed from https://stackoverflow.com/questions/2937114/python-check-if-an-object-is-a-sequence?lq=1

        E.g., used to determine if gaussian mutation has a single specified
        standard deviation, or a vector of standard deviations.

        >>> is_sequence(0.5)
        False

        >>> is_sequence([0.1, 0.2, 0.3])
        True
    """
    return isinstance(obj, collections.abc.Sequence)


##############################
# Function is_flat
##############################
def is_flat(obj):
    """
    :return: True if obj is a flat collection (as opposed to, say, a hierarchical list of lists).

    >>> is_flat((0, 1))
    True

    >>> is_flat(1)
    False

    >>> is_flat([(0, 1), (0, 1)])
    False
    """
    if not is_sequence(obj):
        return False

    for e in obj:
        if is_sequence(e):
            return False

    return True


###############################
# Function is_iterable
###############################
def is_iterable(obj):
    """
    :param obj: that we want to determine is a generator
    :return: True if obj can use next(obj)
    """
    return inspect.isgenerator(obj) or inspect.isgeneratorfunction(obj)


###############################
# Function inc_generation
###############################
def inc_generation(context=context, callbacks=()):
    """ This tracks the current generation

    The `context` is used to report the current generation, though that
    can also be given by inc_generation.generation().

    This will optionally call all the given callback functions whenever the
    generation is incremented. The registered callback functions should have
    a signature f(int), where the int is the new generation.

    >>> from leap_ec.global_vars import context
    >>> my_inc_generation = inc_generation(context)

    :param context: will set ['leap']['generation'] to the incremented
        generation
    :param callbacks: optional list of callback function to call when a
        generation is incremented
    :return: function for incrementing generations
    """
    curr_generation = 0
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


###############################
# Function inc_births
###############################
def inc_births(context=context, start=0, callbacks=()):
    """ This tracks the current number of births

    The `context` is used to report the current births, though that
    can also be given by inc_births.generation().

    This will optionally call all the given callback functions whenever the
    generation is incremented. The registered callback functions should have
    a signature f(int), where the int is the new birth.

    >>> from leap_ec.global_vars import context
    >>> my_inc_births = inc_births(context)

    Each time we call the object, the birth count is incremented and returned:

    >>> my_inc_births()
    1

    >>> my_inc_births()
    2

    >>> my_inc_births()
    3

    The count can be viewed without changing it like so:

    >>> my_inc_births.births()
    3

    And decremented like so:

    >>> my_inc_births.do_decrement()
    2

    :param context: will set ['leap']['births'] to the incremented births
    :param start: if we want to start counter at a higher value; e.g., take
        into consideration births of an initial population
    :param callbacks: optional list of callback function to call when a
            birth numberis incremented
    :return: function for incrementing births
    """
    curr_births = start
    context = context
    context['leap']['births'] = start
    callbacks = callbacks

    def births():
        return curr_births

    def do_increment(size=1):
        nonlocal curr_births
        nonlocal context
        nonlocal callbacks
        curr_births += size

        # Update the context
        context['leap']['births'] += size

        # Now echo the new generation to all the registered callbacks.
        # TODO There is probably a more pythonic way to do this
        [f(curr_births) for f in callbacks]

        return curr_births

    def do_decrement():
        # Sometimes we want to decrement, as is the case for compensating
        # for non-viable individuals
        nonlocal curr_births
        nonlocal context
        nonlocal callbacks
        curr_births -= 1

        # Update the context
        context['leap']['births'] -= 1

        return curr_births

    do_increment.births = births
    do_increment.do_decrement = do_decrement

    return do_increment


###############################
# Function print_list
###############################
def print_list(l):
    """
    Return a string representation of a list.

    This uses __str__() to resolve the elements of the list:

    >>> from leap_ec.individual import Individual
    >>> l = [Individual([0, 1, 2]), Individual([3, 4, 5])]
    >>> print_list(l)
    [[0, 1, 2] None, [3, 4, 5] None]

    As opposed to the standard printing mechanism, which calls __repr__() on
    the elements to produce

    >>> print(l)
    [Individual([0, 1, 2], IdentityDecoder(), None), Individual([3, 4, 5], IdentityDecoder(), None)]

    :param l:
    :return:
    """
    print('[' + ', '.join([x.__str__() for x in l]) + ']')


###############################
# Function birth_brander
###############################
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

    Provides:

    * brand_population() to brand an entire population all at once,
        which is useful for branding initial populations.
    * brand() for explicitly branding a single individual

    :param next_thing: preceding individual in the pipeline
    :return: branded individual
    """
    # incremented with each birth
    num_births = itertools.count()

    # sometimes next_thing is a population, so we need this to track that
    # the next individual in the population
    iterator = None

    def brand(individual):
        """ brand the given individual
        :param individual: to be branded
        :return: branded individual
        """
        if not hasattr(individual, "birth"):
            # Only assign a birth ID if they don't already have one
            individual.birth = next(num_births)
        return individual

    def brand_population(population):
        """ We want to brand an entire population in one go

        Usually used to brand an initial population is one shot.

        :param population: to be branded
        :return: branded population
        """
        return [brand(i) for i in population]

    def do_birth_branding(next_thing):
        """ This has the flexibility of being inserted in a pipeline such that
        the preceding pipeline is a population or a generator that provides
        an individual.  It'll flexibly handle either situation.

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
                # We're being passed a test_sequence/population
                if iterator is None:
                    iterator = iter(next_thing)
                next_thing = next(iterator)

            next_thing = brand(next_thing)

            yield next_thing

    do_birth_branding.brand_population = brand_population

    return do_birth_branding


if __name__ == '__main__':
    pass
