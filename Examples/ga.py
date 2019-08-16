#!/usr/bin/env python
"""
    A simple genetic algorithm example that uses the very basic One Max problem.

"""

import problem  # TODO Need to fix this to be part of the LEAP space
import decoder  # TODO ditto
import ea
import selection
import operators

if __name__ == '__main__':

    pop_size = 20
    max_generation = 10
    genome_size = 10

    problem = problem.FunctionOptimization(problem.oneMaxFunction)

    decoder = decoder.BinaryDecoder(problem, genome_size)

    ga = ea.GenerationalEA(decoder,
                           operators.BitFlipMutation(
                               operators.CloneOperator(
                                   selection.ProportionalSelection()),
                               1.0/genome_size),
                               pop_size)

    ga.run(maxGeneration=max_generation)





