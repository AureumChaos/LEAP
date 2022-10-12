#!/usr/bin/env python3
"""
    Implementation of Non-dominated sorted genetic algorithm II (NSGA-II).

    - Deb, Kalyanmoy, Amrit Pratap, Sameer Agarwal, and T. A. M. T. Meyarivan.
      "A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II." IEEE
      transactions on evolutionary computation 6, no. 2 (2002): 182-197.

"""
from toolz import pipe

from leap_ec import ops, util
from leap_ec.global_vars import context
from leap_ec.individual import Individual
from leap_ec.multiobjective.ops import fast_nondominated_sort, \
    crowding_distance_calc

def nsga_2(max_generations: int, pop_size: int, problem, representation,
           pipeline,
           stop=lambda x: False,
           init_evaluate=Individual.evaluate_population,
           start_generation: int = 0,
           context=context):
    """
    :param int max_generations: The max number of generations to run the algorithm for.
        Can pass in float('Inf') to run forever or until the `stop` condition is reached.
    :param int pop_size: Size of the initial population
    :param int stop: A function that accepts a population and
        returns True iff it's time to stop evolving.
    :param `Problem` problem: the Problem that should be used to evaluate
        individuals' fitness
    :param representation: How the problem is represented in individuals
    :param list pipeline: a list of operators that are applied (in order) to
        create the offspring population at each generation
    :param init_evaluate: a function used to evaluate the initial population,
        before the main pipeline is run.  The default of
        `Individual.evaluate_population` is suitable for many cases, but you
        may wish to pass a different operator in for distributed evaluation
        or other purposes.
    :param start_generation: index of the first generation to count from (defaults to 0).
        You might want to change this, for example, in experiments that involve stopping
        and restarting an algorithm.

    :return: a generator of `(int, individual_cls)` pairs representing the
        best individual at each generation.
    """
    # Ensure that we're dealing with a multi-objective Problem.
    assert isinstance(problem, MultiObjectiveProblem)

    # Initialize a population of pop_size individuals of the same type as
    # individual_cls
    parents = representation.create_population(pop_size, problem=problem)

    # Set up a generation counter that records the current generation to
    # context
    generation_counter = util.inc_generation(start_generation=start_generation,
                                             context=context)

    # Evaluate initial population
    parents = init_evaluate(parents)

    # Output the best individual in the initial population
    # bsf = max(other_population)
    # yield (0, bsf)

    # FIXME just trial scaffolding code to get a better feel for NSGA-II
    # needs.



    while (generation_counter.generation() < max_generations) and not stop(
            parents):
        # Execute the operators to create a new offspring population
        offspring = pipe(parents, *pipeline,
                         ops.elitist_survival(parents=parents,
                                              k=k_elites))

        # if max(offspring) > bsf:  # Update the best-so-far individual
        #     bsf = max(offspring)

        parents = offspring  # Replace other_population with offspring
        generation_counter()  # Increment to the next generation

        # Output the best-so-far individual for each generation
        # yield (generation_counter.generation(), bsf)