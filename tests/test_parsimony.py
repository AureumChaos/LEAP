"""
    For testing the parsimony pressure functions that can be used in selection
    operators.

    The existing doctest do a good job for testing for maximization problems.
    However, we also need to check how well they work for minimization
    problems, which will be the focus of the tests here.
"""
from leap_ec.individual import Individual
from leap_ec.decoder import IdentityDecoder
from leap_ec.real_rep.problems import SpheroidProblem
import leap_ec.ops as ops
from leap_ec.parsimony import koza_parsimony, lexical_parsimony

decoder = IdentityDecoder()

def test_koza_minimization():
    """
        Tests the koza_parsimony() function
    """
    problem = SpheroidProblem(maximize=False)

    pop = []

    pop.append(Individual([0,0,0,0,0,0,0,0], problem=problem, decoder=decoder))
    pop.append(Individual([1,1,1], problem=problem, decoder=decoder))
    pop.append(Individual([2,2,2], problem=problem, decoder=decoder))

    pop = Individual.evaluate_population(pop)
    pop.sort() # sort by fitness

    # Now truncate down to the "best" which should be the one with all zeroes.
    best = ops.truncation_selection(pop, size=1)

    assert best[0].genome == [0,0,0,0,0,0,0,0]

    pop.sort(key=koza_parsimony(penalty=1))

    # Ok, now we want to turn on parsimony pressure, which should knock the
    # really really really long genome out of the running for "best"
    best = ops.truncation_selection(pop, size=2, key=koza_parsimony(penalty=1))

    assert best[0].genome == [0,0,0,0,0,0,0,0]
