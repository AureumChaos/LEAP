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
from leap_ec.multiobjective.ops import rank_ordinal_sort, \
    crowding_distance_calc, sort_by_dominance
from leap_ec.multiobjective.problems import MultiObjectiveProblem

def generalized_nsga_2(max_generations: int,
                       pop_size: int,
                       problem: MultiObjectiveProblem,
                       representation,
                       pipeline,
                       rank_func=rank_ordinal_sort,
                       stop=lambda x: False,
                       init_evaluate=Individual.evaluate_population,
                       start_generation: int = 0,
                       context=context):
    """ NSGA-II multi-objective evolutionary algorithm.

        - Deb, Kalyanmoy, Amrit Pratap, Sameer Agarwal, and T. A. M. T. Meyarivan.
            "A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II." IEEE
            transactions on evolutionary computation 6, no. 2 (2002): 182-197.

        - Bogdan Burlacu. 2022. Rank-based Non-dominated Sorting. arXiv.
            DOI:https://doi.org/10.48550/ARXIV.2203.13654

    This classic algorithm relies on the idea of "non-dominated sorting" and de-crowding
    to evolve a diverse Pareto front.  The "generalized" NSGA-II we implement here differs
    slightly from the canonical algorithm, in that we default to a faster sorting
    algorithm devised by Burlacu (2022).

    If you wish the algorithm to use the original NSGA-II behavior instead (which runs much
    slower), you can select the original operator by passing in `rank_func=fast_nondominated_sort`.

    >>> from leap_ec.representation import Representation
    >>> from leap_ec.ops import random_selection, clone, evaluate, pool
    >>> from leap_ec.real_rep.initializers import create_real_vector
    >>> from leap_ec.real_rep.ops import mutate_gaussian
    >>> from leap_ec.multiobjective.nsga2 import generalized_nsga_2
    >>> from leap_ec.multiobjective.problems import SCHProblem
    >>> pop_size = 10
    >>> max_generations = 5
    >>> final_pop = generalized_nsga_2(
    ...     max_generations=max_generations, pop_size=pop_size,
    ...
    ...     problem=SCHProblem(),
    ...
    ...     representation=Representation(
    ...         initialize=create_real_vector(bounds=[(-10, 10)])
    ...     ),
    ...
    ...     pipeline=[
    ...         random_selection,
    ...         clone,
    ...         mutate_gaussian(std=0.5, expected_num_mutations=1),
    ...         evaluate,
    ...         pool(size=pop_size),
    ...     ]
    ... )

    [Individual(...), Individual(...), Individual(...), ... Individual(...)]

    :note: You will need a selection as first operator in `pipeline`. This will use
        Deb's multiobjective criteria for comparing individuals as dictated in
        `MultiobjectiveProblem`.

    :param int max_generations: The max number of generations to run the algorithm for.
        Can pass in float('Inf') to run forever or until the `stop` condition is reached.
    :param int pop_size: Size of the initial population
    :param rank_func: the function used to calculate non-domination rankings for the
        individuals of the population.
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

    :return: a list of the final population
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
    while (generation_counter.generation() < max_generations) and \
            not stop(parents):
        offspring = pipe(parents,
                         # pipeline for user defined selection, cloning,
                         # mutation, and maybe crossover
                         *pipeline,
                         rank_func(parents=parents),
                         crowding_distance_calc,
                         # sort_by_dominance,
                         # truncation_selection w/ key should do this implicitly
                         ops.truncation_selection(size=len(parents),
                                                  key=lambda x: (-x.rank,
                                                                 x.distance)))

        parents = offspring  # Replace other_population with offspring

        generation_counter()  # Increment to the next generation

    return parents
