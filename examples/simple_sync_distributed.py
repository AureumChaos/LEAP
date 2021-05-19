#!/usr/bin/env python
""" Simple example of using leap_ec.distrib.synchronous

"""
import multiprocessing.popen_spawn_posix  # Python 3.9 workaround for Dask.  See https://github.com/dask/distributed/issues/4168
from distributed import Client
import toolz

from leap_ec.decoder import IdentityDecoder
import leap_ec.ops as ops
from leap_ec.probe import AttributesCSVProbe
from leap_ec.context import context

from leap_ec.binary_rep.problems import MaxOnes
from leap_ec.binary_rep.initializers import create_binary_sequence
from leap_ec.binary_rep.ops import mutate_bitflip

from leap_ec.distrib.individual import DistributedIndividual
from leap_ec.distrib import synchronous

if __name__ == '__main__':

    # We've added some additional state to the probe for DistributedIndividual,
    # so we want to capture that.
    probe = AttributesCSVProbe(attributes=['hostname',
                                           'pid',
                                           'uuid',
                                           'birth_id',
                                           'start_eval_time',
                                           'stop_eval_time'],
                               do_fitness=True,
                               do_genome=True,
                               stream=open('simple_sync_distributed.csv', 'w'))

    # Just to demonstrate multiple outputs, we'll have a separate probe that
    # will take snapshots of the offspring before culling.  That way we can
    # compare the before and after to see what specific individuals were culled.
    offspring_probe = AttributesCSVProbe(attributes=['hostname',
                                           'pid',
                                           'uuid',
                                           'birth_id',
                                           'start_eval_time',
                                           'stop_eval_time'],
                               do_fitness=True,
                               stream=open('simple_sync_distributed_offspring.csv', 'w'))

    with Client() as client:
        # create an initial population of 5 parents of 4 bits each for the
        # MAX ONES problem
        parents = DistributedIndividual.create_population(5, # make five individuals
                                                          initialize=create_binary_sequence(
                                                              4), # with four bits
                                                          decoder=IdentityDecoder(),
                                                          problem=MaxOnes())

        # Scatter the initial parents to dask workers for evaluation
        parents = synchronous.eval_population(parents, client=client)

        # probes rely on this information for printing CSV 'step' column
        context['leap']['generation'] = 0

        probe(parents) # generation 0 is initial population
        offspring_probe(parents) # generation 0 is initial population

        for current_generation in range(5):
            context['leap']['generation'] += 1

            offspring = toolz.pipe(parents,
                                   ops.tournament_selection,
                                   ops.clone,
                                   mutate_bitflip(expected_num_mutations=1),
                                   ops.uniform_crossover,
                                   # Scatter offspring to be evaluated
                                   synchronous.eval_pool(client=client,
                                                         size=len(parents)),
                                   offspring_probe, # snapshot before culling
                                   ops.elitist_survival(parents=parents),
                                   # snapshot of population after culling
                                   # in separate CSV file
                                   probe)

            print('generation:', current_generation)
            [print(x.genome, x.fitness) for x in offspring]

            parents = offspring

    print('Final population:')
    [print(x.genome, x.fitness) for x in parents]
