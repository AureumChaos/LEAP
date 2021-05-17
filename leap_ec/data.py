"""A module for synthetic data that we use in test and examples."""
from leap_ec.individual import Individual
from leap_ec.binary_rep.problems import MaxOnes
from leap_ec.decoder import IdentityDecoder


##############################
# test_population
##############################
def _build_test_pop():
    """Construct a synthetic population for illustrating example operations."""
    pop = [Individual([1, 0, 1, 1, 0], IdentityDecoder(),
                           MaxOnes()),
           Individual([0, 0, 1, 0, 0], IdentityDecoder(),
                           MaxOnes()),
           Individual([0, 1, 1, 1, 1], IdentityDecoder(),
                           MaxOnes()),
           Individual([1, 0, 0, 0, 1], IdentityDecoder(),
                           MaxOnes())]
    pop = Individual.evaluate_population(pop)

    # Assign distinct values to an attribute on each individual
    attrs = [('foo', ['GREEN', 15, 'BLUE', 72.81]),
             ('bar', ['Colorless', 'green', 'ideas', 'sleep']),
             ('baz', [['a', 'b', 'c'], [1, 2, 3], [None, None, None],
                      [0.1, 0.2, 0.3]])]

    for attr, vals in attrs:
        for (ind, val) in zip(pop, vals):
            ind.__dict__[attr] = val

    return pop


"""A synthetic population for illustrating example operations"""
test_population = _build_test_pop()
