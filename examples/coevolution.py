"""
    Provides an example of a co-evolutionary system.
"""
from leap_ec.individual import Individual
from leap_ec.decoder import IdentityDecoder
from leap_ec.representation import Representation
from leap_ec.algorithm import multi_population_ea
from leap_ec.context import context

import leap_ec.ops as ops

from leap_ec.binary_rep.problems import MaxOnes
from leap_ec.binary_rep.initializers import create_binary_sequence
from leap_ec.binary_rep.ops import mutate_bitflip

if __name__ == '__main__':
    pop_size = 5

    with open('./coop_stats.csv', 'w') as log_stream:
        ea = multi_population_ea(generations=1000, pop_size=pop_size,
                                 num_populations=9,
                                 problem=MaxOnes(),
                                 # Fitness function

                                 init_evaluate=ops.const_evaluate(value=-100),

                                 representation=Representation(
                                     individual_cls=Individual,
                                     initialize=create_binary_sequence(
                                         length=1),
                                     decoder=IdentityDecoder()
                                 ),

                                 # Operator pipeline
                                 shared_pipeline=[
                                     ops.tournament_selection,
                                     ops.clone,
                                     mutate_bitflip(expected_num_mutations=1),
                                     ops.CooperativeEvaluate(
                                         context=context,
                                         num_trials=1,
                                         collaborator_selector=ops.random_selection,
                                         log_stream=log_stream),
                                     ops.pool(size=pop_size)
                                 ])

        print('generation, subpop_bsf')
        for g, x in ea:
            print(f"{g}, {[ind.fitness for ind in x]}")
