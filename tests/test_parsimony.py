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
from leap_ec.binary_rep.problems import MaxOnes
import leap_ec.ops as ops
from leap_ec.parsimony import koza_parsimony, lexical_parsimony

decoder = IdentityDecoder()


def test_koza_maximization():
    """
        Tests the koza_parsimony() function for maximization problems
    """
    problem = SpheroidProblem(maximize=True)

    pop = []

    # We set up three individuals in ascending order of fitness of
    # [0, 1, 2]
    pop.append(Individual([0], problem=problem, decoder=decoder))
    pop.append(Individual([1], problem=problem, decoder=decoder))
    pop.append(Individual([0,0,1,1], problem=problem, decoder=decoder))

    pop = Individual.evaluate_population(pop)

    # Now truncate down to the "best" that should be the third one
    best = ops.truncation_selection(pop, size=1)
    assert best[0].genome == [0,0,1,1]

    # This is just to look at the influence of the parsimony pressure on
    # the order of the individual.  You should observe that the order is now
    # ([0,0,1,1], [0], [1]) because now their biased fitnesses are respectively
    # (-2, -1, 0)
    pop.sort(key=koza_parsimony(penalty=1))

    # Ok, now we want to turn on parsimony pressure, which should knock the
    # really really really long genome out of the running for "best"
    best = ops.truncation_selection(pop, size=1, key=koza_parsimony(penalty=1))

    assert best[0].genome == [1]


def test_koza_minimization():
    """
        Tests the koza_parsimony() function for _minimization_ problems.
    """
    problem = SpheroidProblem(maximize=False)

    pop = []

    # First individual has a fitness of three but len(genome) of 4
    pop.append(Individual([0,1,1,1], problem=problem, decoder=decoder))

    # Second has a fitness of 4, but len(genome) of 1
    pop.append(Individual([2], problem=problem, decoder=decoder))

    pop = Individual.evaluate_population(pop)

    best = ops.truncation_selection(pop, size=1, key=koza_parsimony(penalty=1))

    assert best[0].genome == [2]


def test_lexical_maximization():
    """
        Tests the lexical_parsimony() for maximization problems
    """
    problem = MaxOnes()

    # fitness=3, len(genome)=6
    pop = [Individual([0, 0, 0, 1, 1, 1], problem=problem, decoder=decoder)]

    # fitness=2, len(genome)=2
    pop.append(Individual([1, 1], problem=problem, decoder=decoder))

    # fitness=3, len(genome)=3
    pop.append(Individual([1, 1, 1], decoder=decoder, problem=problem))

    pop = Individual.evaluate_population(pop)

    best = ops.truncation_selection(pop, size=1, key=lexical_parsimony)

    # prefers the shorter of the 3 genomes
    assert best[0].genome == [1,1,1]


def test_lexical_minimization():
    """
        Tests lexical_parsimony() for minimization problems
    """
    problem = SpheroidProblem(maximize=False)

    pop = []

    # fitness=4, len(genome)=1
    pop.append(Individual([2], problem=problem, decoder=decoder))

    # fitness=4, len(genome)=4
    pop.append(Individual([1,1,1,1], problem=problem, decoder=decoder))

    pop = Individual.evaluate_population(pop)

    best = ops.truncation_selection(pop, size=1, key=lexical_parsimony)

    # prefers the shorter of the genomes with equivalent fitness
    assert best[0].genome == [2]
