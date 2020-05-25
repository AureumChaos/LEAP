#!/usr/bin/env python3
"""
    This implements a simple Evolutionary Programming (EP) system, but it
    does not evolve state machines as done with the original EP approach.

    TODO convert to a state machines problem

"""
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

if __name__ == '__main__':
    # Define the real value bounds for initializing the population. In this case,
    # we define a genome of four bounds.

    # the (-5.12,5.12) was what was originally used for this problem in
    # Ken De Jong's 1975 dissertation, so was used for historical reasons.
    bounds = [(-5.12, 5.12), (-5.12, 5.12), (-5.12, 5.12), (-5.12, 5.12)]
    parents = core.Individual.create_population(5,
                                                initialize=core.create_real_vector(
                                                    bounds),
                                                decoder=core.IdentityDecoder(),
                                                problem=real_problems.SpheroidProblem(maximize=False))

    # Evaluate initial population
    parents = core.Individual.evaluate_population(parents)

    # print initial, random population
    print_population(parents, generation=0)

    max_generation = 100

    # We use the provided core.context, but we could roll our own if we
    # wanted to keep separate contexts.  E.g., island models may want to have
    # their own contexts.
    generation_counter = util.inc_generation(context=core.context)

    while generation_counter.generation() < max_generation:
        offspring = pipe(parents,
                         ops.cyclic_selection,
                         ops.clone,
                         ops.mutate_gaussian(std=.1),
                         ops.evaluate,
                         ops.pool(
                             size=len(parents) * BROOD_SIZE),
                         # create the brood
                         ops.truncate(size=len(parents), parents=parents))  # mu + lambda

        parents = offspring

        generation_counter()  # increment to the next generation

        # Just to demonstrate that we can also get the current generation from
        # the context
        print_population(parents, core.context['leap']['generation'])
