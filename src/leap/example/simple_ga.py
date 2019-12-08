#!/usr/bin/env python3
"""
    Simple GA-like example using a MAX ONES problem

    Canonical GAs don't use truncation selection, but we used that here for didactic purposes.
"""
from toolz import pipe

from leap import core
from leap import ops
from leap import binary_problems


if __name__ == '__main__':
    parents = core.Individual.create_population(5, initialize=core.create_binary_sequence,
                                                decoder=core.IdentityDecoder(), problem=binary_problems.MaxOnes())

    # Evaluate initial population
    parents = core.Individual.evaluate_population(parents)

    max_generation = 5

    for generation in range(max_generation):
        survivors, args, kwargs = pipe(parents,
                         ops.tournament,
                         ops.clone,
                         ops.mutate_bitflip,
                         ops.pool(size=10),  # 10 offspring
                         ops.truncate(size=5))  # (mu + lambda)

        parents = list(offspring)
