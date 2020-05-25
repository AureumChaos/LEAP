#!/usr/bin/env python3
"""
    This implements a simple Evolutionary Strategy (ES) system, and implements
    a very crude self-adaptive step-size mechanism to show how to use contexts.

"""
from toolz import pipe

from leap import core
from leap import ops
from leap import real_problems
from leap import util


BROOD_SIZE = 3  # how many offspring each parent will reproduce
MAX_GENERATIONS = 10


def print_population(population, generation):
    """ Convenience function for pretty printing a population that's
    associated with a given generation

    :param population:
    :param generation:
    :return: None
    """
    for individual in population:
        print(
            generation,
            core.context['leap']['std'],
            individual.genome,
            individual.fitness)


def anneal_std(generation):
    """ Cool down the step-size each generation.

    This tickles core.context['leap']['std']

    :param generation: current generation, which the callback always gives
    :return: new std
    """
    # .85 was an original annealing step size used by Hans-Paul Schwefel, though
    # this was in the context of the 1/5 success rule, which we've not
    # implemented here.
    # Handbook of EC, B1.3:2
    core.context['leap']['std'] *= .85


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

    core.context['leap']['std'] = 2

    # We use the provided core.context, but we could roll our own if we
    # wanted to keep separate contexts.  E.g., island models may want to have
    # their own contexts.
    generation_counter = util.inc_generation(
        context=core.context, callbacks=(anneal_std,))

    # print initial, random population
    print_population(parents, generation=0)

    while generation_counter.generation() < MAX_GENERATIONS:
        offspring = pipe(parents,
                         ops.random_selection,
                         ops.clone,
                         ops.mutate_gaussian(std=core.context['leap']['std']),
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
