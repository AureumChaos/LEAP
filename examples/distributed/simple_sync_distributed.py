#!/usr/bin/env python
""" Simple example of using leap_ec.distrib.synchronous

"""
import os

import multiprocessing.popen_spawn_posix  # Python 3.9 workaround for Dask.  See https://github.com/dask/distributed/issues/4168
from distributed import Client
import toolz

from leap_ec import test_env_var
from leap_ec import ops
from leap_ec.decoder import IdentityDecoder
from leap_ec.binary_rep.initializers import create_binary_sequence
from leap_ec.binary_rep.ops import mutate_bitflip
from leap_ec.binary_rep.problems import MaxOnes
from leap_ec.distrib import DistributedIndividual
from leap_ec.distrib import synchronous

if __name__ == '__main__':

    with Client() as client:
        # create an initial population of 5 parents of 4 bits each for the
        # MAX ONES problem
        parents = DistributedIndividual.create_population(5,
                                                          initialize=create_binary_sequence(
                                                              4),
                                                          decoder=IdentityDecoder(),
                                                          problem=MaxOnes())

        # Scatter the initial parents to dask workers for evaluation
        parents = synchronous.eval_population(parents, client=client)


        # When running the test harness, just run for two generations
        # (we use this to quickly ensure our examples don't get bitrot)
        if os.environ.get(test_env_var, False) == 'True':
            generations = 2
        else:
            generations = 5

        for current_generation in range(generations):
            offspring = toolz.pipe(parents,
                                   ops.tournament_selection,
                                   ops.clone,
                                   mutate_bitflip(expected_num_mutations=1),
                                   ops.uniform_crossover,
                                   # Scatter offspring to be evaluated
                                   synchronous.eval_pool(client=client,
                                                         size=len(parents)),
                                   ops.elitist_survival(parents=parents))

            print('generation:', current_generation)
            [print(x.genome, x.fitness) for x in offspring]

            parents = offspring

    print('Final population:')
    [print(x.genome, x.fitness) for x in parents]
