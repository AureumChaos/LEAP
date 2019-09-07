import numpy as np
from toolz import curry

from leap.individual import Individual


def evaluate(population):
    """
    Evaluate all the individuals in the given population

    :param population: to be evaluated
    :return: the evaluated individuals
    """
    for individual in population:
        individual.evaluate()

    return population


def cloning(population):
    result = []
    for ind in population:
        result.append(ind.clone())
    return result


@curry
def mutate_bitflip(population, prob):
    """
    >>> population = [Individual(genome=[1, 0, 1, 1, 0])]
    >>> always = mutate_bitflip(prob=1.0)
    >>> list(always(population))
    [[0, 1, 0, 0, 1]]

    Individuals are modified in place:

    >>> population
    [[0, 1, 0, 0, 1]]

    >>> population = [Individual(genome=[1, 0, 1, 1, 0])]
    >>> never = mutate_bitflip(prob=0.0)
    >>> list(never(population))
    [[1, 0, 1, 1, 0]]
    """
    def flip(x):
        if np.random.uniform() < prob:
            return 0 if x == 1 else 1
        else:
            return x

    result = []
    for ind in population:
        ind.genome = [flip(x) for x in ind.genome]
        ind.fitness = None
        result.append(ind)
    return result


@curry
def mutate_gaussian(population, prob, std):
    def add_gauss(x):
        if np.random.uniform() < prob:
            return x + np.random.normal()*std
        else:
            return x

    result = []
    for ind in population:
        ind.genome = [add_gauss(x) for x in ind.genome]
        ind.fitness = None
        result.append(ind)
    return result


@curry
def truncation(population, mu):
    """
    Returns the `mu` individuals with the best fitness.

    For example, say we have a population of 10 individuals with the following fitnesses:

    >>> fitnesses = [0.12473057, 0.74763715, 0.6497458 , 0.36178902, 0.41318757, 0.69130493, 0.67464942, 0.14895497, 0.15406642, 0.31307095]
    >>> population = [Individual([i]) for i in range(10)]
    >>> for (ind, f) in zip(population, fitnesses):
    ...     ind.fitness = f

    The three highest-fitness individuals are are the indices 1, 5, and 6:

    >>> list(truncation(population, 3))
    [[1], [5], [6]]
    """
    inds = list(sorted(list(population), key=lambda x: x.fitness, reverse=True))
    for ind in inds[0:mu]:
        yield ind


@curry
def tournament(population, n, num_competitors=2):
    """
    Select `n` individuals form a population via tournament selection.
    :param list population: A list of :py:class:`Individual`s
    :param int n: The number of individuals to select
    :param int num_competitors: The number of individuals that compete in each tournament
    :return: A generator that produces `n` individuals
    >>> pop = [Individual(genome=[1, 0, 1, 1, 0]), \
               Individual(genome=[0, 0, 1, 0, 0]), \
               Individual(genome=[0, 1, 1, 1, 1]), \
               Individual(genome=[1, 0, 0, 0, 1])]
    >>> for (ind, f) in zip(pop, [3, 1, 4, 2]):
    ...     ind.fitness = f
    >>> result = tournament(pop, 3)
    >>> result # doctest:+ELLIPSIS
    <generator object tournament at ...>

    >>> print(*list(result), sep='\\n') # doctest:+ELLIPSIS
    [...]
    [...]
    [...]
    """
    for i in range(n):
        competitors = np.random.choice(population, num_competitors)
        yield max(competitors, key=lambda x: x.fitness)


def best(population):
    """
    >>> from leap import decode, binary
    >>> pop = [Individual([1, 0, 1, 1, 0], decode.IdentityDecoder(), binary.MaxOnes()), \
               Individual([0, 0, 1, 0, 0], decode.IdentityDecoder(), binary.MaxOnes()), \
               Individual([0, 1, 1, 1, 1], decode.IdentityDecoder(), binary.MaxOnes()), \
               Individual([1, 0, 0, 0, 1], decode.IdentityDecoder(), binary.MaxOnes())]
    >>> pop = evaluate(pop)
    >>> best(pop)
    [0, 1, 1, 1, 1]
    """
    assert(len(population) > 0)
    return max(population)


@curry
def cloning(population, offspring_per_ind=1):
    """
    >>> pop = [Individual([1, 2]),
    ...        Individual([3, 4]),
    ...        Individual([5, 6])]
    >>> new_pop = list(cloning(pop))
    >>> new_pop
    [[1, 2], [3, 4], [5, 6]]

    If we edit individuals in the original, new_pop shouldn't change:

    >>> pop[0].genome[1] = 7
    >>> pop[2].genome[0] = 0
    >>> pop
    [[1, 7], [3, 4], [0, 6]]

    >>> new_pop
    [[1, 2], [3, 4], [5, 6]]

    If we set `offspring_per_ind`, we can create bigger populations:

    >>> pop = [Individual([1, 2]),
    ...        Individual([3, 4]),
    ...        Individual([5, 6])]
    >>> new_pop = list(cloning(pop, offspring_per_ind=3))
    >>> new_pop
    [[1, 2], [1, 2], [1, 2], [3, 4], [3, 4], [3, 4], [5, 6], [5, 6], [5, 6]]
    """
    assert(population is not None)
    assert(offspring_per_ind > 0)

    for ind in population:
        for i in range(offspring_per_ind):
            yield ind.clone()
