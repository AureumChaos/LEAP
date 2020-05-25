"""
    Provides an example of a co-evolutionary system.
"""


import numpy as np


from leap import core, ops, binary_problems
from leap.algorithm import multi_population_ea

import networkx as nx

if __name__ == '__main__':
    pop_size = 5

    with open('./coop_stats.csv', 'w') as log_stream:
        ea = multi_population_ea(generations=1000, pop_size=pop_size, num_populations=9,
                                 problem=binary_problems.MaxOnes(),  # Fitness function

                                 init_evaluate=ops.const_evaluate(value=-100),

                                 representation=core.Representation(
                                     individual_cls=core.Individual,
                                     initialize=core.create_binary_sequence(
                                         length=1),
                                     decoder=core.IdentityDecoder()
                                 ),

                                 # Operator pipeline
                                 shared_pipeline=[
                                     ops.tournament,
                                     ops.clone,
                                     ops.mutate_bitflip(expected=1),
                                     ops.coop_evaluate(context=core.context,
                                                       num_trials=1,
                                                       collaborator_selector=ops.random_selection,
                                                       log_stream=log_stream),
                                     ops.pool(size=pop_size)
                                 ])

        print('generation, subpop_bsf')
        for g, x in ea:
            print(f"{g}, {[ind.fitness for ind in x]}")
