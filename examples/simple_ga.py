#!/usr/bin/env python3
"""
    Simple GA-like example using a MAX ONES problem

    Canonical GAs don't use truncation selection, but we used that here for
    didactic purposes. """
from toolz import pipe

from leap import core
from leap import ops
from leap import binary_problems
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


if __name__ == '__main__':
    parents = core.Individual.create_population(5,
                                                initialize=core.create_binary_sequence(
                                                    4),
                                                decoder=core.IdentityDecoder(), problem=binary_problems.MaxOnes())

    # Evaluate initial population
    parents = core.Individual.evaluate_population(parents)

    # print initial, random population
    print_population(parents, generation=0)

    max_generation = 6

    # We use the provided core.context, but we could roll our own if we
    # wanted to keep separate contexts.  E.g., island models may want to have
    # their own contexts.
    generation_counter = util.inc_generation(context=core.context)

    while generation_counter.generation() < max_generation:
        offspring = pipe(parents,
                         ops.tournament,
                         ops.clone,
                         ops.mutate_bitflip,
                         ops.uniform_crossover,
                         ops.evaluate,
                         ops.pool(size=len(parents)))  # accumulate offspring

        parents = offspring

        generation_counter()  # increment to the next generation

        # Just to demonstrate that we can also get the current generation from
        # the context
        print_population(parents, core.context['leap']['generation'])
