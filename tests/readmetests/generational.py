#!/usr/bin/env python3
"""
    Ensures that the second code snippet in top-level README.md works.
"""

from leap_ec.algorithm import generational_ea
from leap_ec import ops, decoder, probe, representation
from leap_ec.binary_rep import initializers
from leap_ec.binary_rep import problems
from leap_ec.binary_rep.ops import mutate_bitflip

pop_size = 5
final_pop = generational_ea(max_generations=10, pop_size=pop_size,

                            # Solve a MaxOnes Boolean optimization problem
                            problem=problems.MaxOnes(),

                            representation=representation.Representation(
                                # Genotype and phenotype are the same for this task
                                decoder=decoder.IdentityDecoder(),
                                # Initial genomes are random binary sequences
                                initialize=initializers.create_binary_sequence(length=10)
                            ),

                            # The operator pipeline
                            pipeline=[
                                    # Select parents via tournament_selection selection
                                    ops.tournament_selection,
                                    ops.clone,  # Copy them (just to be safe)
                                    # Basic mutation with a 1/L mutation rate
                                    mutate_bitflip(expected_num_mutations=1),
                                    # Crossover with a 40% chance of swapping each gene
                                    ops.UniformCrossover(p_swap=0.4),
                                    ops.evaluate,  # Evaluate fitness
                                    # Collect offspring into a new population
                                    ops.pool(size=pop_size),
                                    probe.BestSoFarProbe()  # Print the BSF
                                    ])

