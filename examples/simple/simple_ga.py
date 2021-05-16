#!/usr/bin/env python3
"""
    Simple GA-like example using a MAX ONES problem

    Canonical GAs don't use truncation selection, but we used that here for
    didactic purposes. """
from toolz import pipe

from leap_ec.individual import Individual
from leap_ec.decoder import IdentityDecoder

import leap_ec.ops as ops
from leap_ec.binary_rep.problems import MaxOnes
from leap_ec.binary_rep.initializers import create_binary_sequence
from leap_ec.binary_rep.ops import mutate_bitflip
from leap_ec import util
from leap_ec import probe



if __name__ == '__main__':
    parents = Individual.create_population(5,
                                           initialize=create_binary_sequence(
                                               4),
                                           decoder=IdentityDecoder(),
                                           problem=MaxOnes())

    # Evaluate initial population
    parents = Individual.evaluate_population(parents)

    # print initial, random population
    util.print_population(parents, generation=0)

    max_generation = 6

    # Set up a generation counter using the default global context variable
    generation_counter = util.inc_generation()

    while generation_counter.generation() < max_generation:
        offspring = pipe(parents,
                         ops.tournament_selection,
                         ops.clone,
                         # these are optional probes to demonstrate their use
                         probe.print_individual(prefix='before mutation: '),
                         mutate_bitflip(expected_num_mutations=1),
                         probe.print_individual(prefix='after mutation: '),
                         ops.uniform_crossover,
                         ops.evaluate,
                         ops.pool(size=len(parents)),  # accumulate offspring
                         ops.elitist_survival(parents=parents)) # keep best
                                                                # parent to
                                                                # compete with
                                                                # offspring

        parents = offspring

        generation_counter()  # increment to the next generation

        # Just to demonstrate that we can also get the current generation from
        # the context
        util.print_population(parents, context['leap']['generation'])