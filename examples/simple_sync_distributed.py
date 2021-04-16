#!/usr/bin/env python
""" Simple example of using leap_ec.distrib.synchronous

"""
import multiprocessing.popen_spawn_posix  # Python 3.9 workaround for Dask.  See https://github.com/dask/distributed/issues/4168
from dask.distributed import Client
import toolz

from leap_ec.decoder import IdentityDecoder
import leap_ec.ops as ops

from leap_ec.binary_rep.problems import MaxOnes
from leap_ec.binary_rep.initializers import create_binary_sequence
from leap_ec.binary_rep.ops import mutate_bitflip

from leap_ec.distrib.individual import DistributedIndividual
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

        for current_generation in range(5):
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
