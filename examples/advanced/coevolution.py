"""
    Provides an example of a co-evolutionary system.
"""
import os

from leap_ec import Individual, Representation, test_env_var
from leap_ec import ops
from leap_ec.algorithm import multi_population_ea
from leap_ec.binary_rep.problems import MaxOnes
from leap_ec.binary_rep.initializers import create_binary_sequence
from leap_ec.binary_rep.ops import mutate_bitflip


##############################
# Entry point
##############################
if __name__ == '__main__':
    pop_size = 5

    # When running the test harness, just run for two generations
    # (we use this to quickly ensure our examples don't get bitrot)
    if os.environ.get(test_env_var, False) == 'True':
        generations = 2
    else:
        generations = 1000

    with open('./coop_stats.csv', 'w') as log_stream:
        ea = multi_population_ea(max_generations=generations, pop_size=pop_size,
                                 num_populations=9,
                                 problem=MaxOnes(),
                                 # Fitness function

                                 init_evaluate=ops.const_evaluate(value=-100),

                                 representation=Representation(
                                     individual_cls=Individual,
                                     initialize=create_binary_sequence(
                                         length=1)
                                 ),

                                 # Operator pipeline
                                 shared_pipeline=[
                                     ops.tournament_selection,
                                     ops.clone,
                                     mutate_bitflip(expected_num_mutations=1),
                                     ops.CooperativeEvaluate(
                                         num_trials=1,
                                         collaborator_selector=ops.random_selection,
                                         log_stream=log_stream),
                                     ops.pool(size=pop_size)
                                 ])

        print('generation, subpop_bsf')
        for g, x in ea:
            print(f"{g}, {[ind.fitness for ind in x]}")
