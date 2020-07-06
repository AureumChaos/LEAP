#!/usr/bin/env python
""" Simple example of using leap_ec.distributed.synchronous

"""
import toolz
from dask.distributed import Client

from leap_ec import core
from leap_ec import ops
from leap_ec import binary_problems
from leap_ec.distributed import synchronous

if __name__ == '__main__':

    with Client() as client:
        # create an initial population of 5 parents of 4 bits each for the
        # MAX ONES problem
        parents = core.Individual.create_population(5,
                                                    initialize=core.create_binary_sequence(4),
                                                    decoder=core.IdentityDecoder(),
                                                    problem=binary_problems.MaxOnes())

        # Scatter the initial parents to dask workers for evaluation
        parents = synchronous.eval_population(parents, client=client)

        for current_generation in range(5):
            offspring = toolz.pipe(parents,
                                   ops.tournament,
                                   ops.clone,
                                   ops.mutate_bitflip,
                                   ops.uniform_crossover,
                                   # Scatter offspring to be evaluated
                                   synchronous.eval_pool(client=client,
                                                         size=len(parents)))

            print('generation:', current_generation)
            [print(x.genome, x.fitness) for x in offspring]

            parents = offspring

    print('Final population:')
    [print(x.genome, x.fitness) for x in parents]
