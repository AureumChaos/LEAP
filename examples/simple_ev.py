#!/usr/bin/env python3
"""
    EV is a simple ES-like EA invented by Ken De Jong for educational purposes.

    Essentially, an EV uses a real-valued representation, and a pre-defined,
    static standard deviation applied to the Gaussian mutation operator.

    Note that there are sibling examples the demonstrate more true
    evolutionary strategy-like approaches. """
from toolz import pipe

from leap import core
from leap import ops
from leap import real_problems
from leap import util


def print_population(population, generation):
    """ Convenience function for pretty printing a population that's
    associated with a given generation

    :param population:
    :param generation:
    :return: None
    """
    for individual in population:
        print(generation, individual.genome, individual.fitness)


BROOD_SIZE = 3  # how many offspring each parent will reproduce
POPULATION_SIZE = 10
MAX_GENERATIONS = 5

if __name__ == '__main__':
    # Define the real value bounds for initializing the population. In this
    # case, we define a genome of four bounds.

    # the (-5.12,5.12) was what was originally used for this problem in
    # Ken De Jong's 1975 dissertation, so was used for historical reasons.
    bounds = [(-5.12, 5.12), (-5.12, 5.12), (-5.12, 5.12), (-5.12, 5.12)]
    parents = core.Individual.create_population(POPULATION_SIZE,
                                                initialize=core.create_real_vector(
                                                    bounds),
                                                decoder=core.IdentityDecoder(),
                                                problem=real_problems.SpheroidProblem(maximize=False))

    # Evaluate initial population
    parents = core.Individual.evaluate_population(parents)

    # print initial, random population
    print_population(parents, generation=0)

    max_generation = MAX_GENERATIONS

    # We use the provided core.context, but we could roll our own if we
    # wanted to keep separate contexts.  E.g., island models may want to have
    # their own contexts.
    generation_counter = util.inc_generation(context=core.context)

    while generation_counter.generation() < max_generation:
        offspring = pipe(parents,
                         ops.random_selection,
                         ops.clone,
                         ops.mutate_gaussian(std=.1),
                         ops.evaluate,
                         ops.pool(
                             size=len(parents) * BROOD_SIZE),
                         # create the brood
                         ops.insertion_selection(parents=parents))

        parents = offspring

        generation_counter()  # increment to the next generation

        # Just to demonstrate that we can also get the current generation from
        # the context
        print_population(parents, core.context['leap']['generation'])
