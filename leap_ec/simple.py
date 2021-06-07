"""
    Provides a very high-level convenience function for a very general EA,
    ea_solve().
"""
from matplotlib import pyplot as plt

from leap_ec import Individual, Representation
from leap_ec import ops, probe
from leap_ec.algorithm import generational_ea
from leap_ec.problem import FunctionProblem
from leap_ec.real_rep import create_real_vector
from leap_ec.real_rep.ops import mutate_gaussian


##############################
# Function ea_solve()
##############################
def ea_solve(function, bounds, generations=100, pop_size=2,
             mutation_std=1.0, maximize=False, viz=False, viz_ylim=(0, 1),
             hard_bounds=True):
    """Provides a simple, top-level interfact that optimizes a real-valued
    function using a simple generational EA.

    :param function: the function to optimize; should take lists of real
        numbers as input and return a float fitness value

    :param [(float, float)] bounds: a list of (min, max) bounds to define the
        search space

    :param int generations: the number of generations to run for
    :param int pop_size: the population size
    :param float mutation_std: the width of the mutation distribution
    :param bool maximize: whether to maximize the function (else minimize)
    :param bool viz: whether to display a live best-of-generation plot
    :param bool hard_bounds: if True, bounds are enforced at all times during
        evolution; otherwise they are only used to initialize the population.

    :param (float, float) viz_ylim: initial bounds to use of the plots
        vertical axis

    The basic call includes instrumentation that prints the best-so-far fitness
    value of each generation to stdout:

    >>> from leap_ec.simple import ea_solve
    >>> ea_solve(sum, bounds=[(0, 1)]*5) # doctest:+ELLIPSIS
    generation, bsf
    0, ...
    1, ...
    ...
    100, ...
    [..., ..., ..., ..., ...]

    When `viz=True`, a live BSF plot will also display:

    >>> ea_solve(sum, bounds=[(0, 1)]*5, viz=True) # doctest:+ELLIPSIS
    generation, bsf
    0, ...
    1, ...
    ...
    100, ...
    [..., ..., ..., ..., ...]

    .. plot::

        from leap_ec.simple import ea_solve
        ea_solve(sum, bounds=[(0, 1)]*5, viz=True)

    """

    if hard_bounds:
        mutation_op = mutate_gaussian(std=mutation_std, hard_bounds=bounds,
                        expected_num_mutations='isotropic')
    else:
        mutation_op = mutate_gaussian(std=mutation_std,
                        expected_num_mutations='isotropic')

    pipeline = [
        ops.tournament_selection,
        ops.clone,
        mutation_op,
        ops.uniform_crossover(p_swap=0.4),
        ops.evaluate,
        ops.pool(size=pop_size)
    ]

    if viz:
        plot_probe = probe.FitnessPlotProbe(ylim=viz_ylim, ax=plt.gca())
        pipeline.append(plot_probe)

    ea = generational_ea(max_generations=generations,
                         pop_size=pop_size,
                         problem=FunctionProblem(function, maximize),

                         representation=Representation(
                             individual_cls=Individual,
                             initialize=create_real_vector(bounds=bounds)
                         ),

                         pipeline=pipeline)

    best_genome = None
    print('generation, bsf')
    for g, ind in ea:
        print(f"{g}, {ind.fitness}")
        best_genome = ind.genome

    return best_genome
