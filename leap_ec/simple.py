"""
    Provides a very high-level convenience function for a very general EA,
    ea_solve().
"""
from matplotlib import pyplot as plt

from leap_ec import ops, probe
from leap_ec.context import context
from leap_ec.algorithm import generational_ea
from leap_ec.real_rep.ops import mutate_gaussian
from leap_ec.representation import Representation
from leap_ec.individual import Individual
from leap_ec.problem import FunctionProblem
from leap_ec.decoder import IdentityDecoder
from leap_ec.real_rep.initializers import create_real_vector

def ea_solve(function, bounds, generations=100, pop_size=2,
             mutation_std=1.0, maximize=False, viz=False, viz_ylim=(0, 1)):
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

    :param (float, float) viz_ylim: initial bounds to use of the plots
        vertical axis

    >>> from leap_ec import simple
    >>> ea_solve(sum, bounds=[(0, 1)]*5) # doctest:+ELLIPSIS
    generation, bsf
    0, ...
    1, ...
    ...
    100, ...
    [..., ..., ..., ..., ...]
    """

    pipeline = [
        ops.tournament_selection,
        ops.clone,
        mutate_gaussian(std=mutation_std, expected_num_mutations='isotropic'),
        ops.uniform_crossover(p_swap=0.4),
        ops.evaluate,
        ops.pool(size=pop_size)
    ]

    if viz:
        plot_probe = probe.PopulationPlotProbe(ylim=viz_ylim, ax=plt.gca())
        pipeline.append(plot_probe)

    ea = generational_ea(generations=generations, pop_size=pop_size,
                         problem=FunctionProblem(function, maximize),

                         representation=Representation(
                             individual_cls=Individual,
                             decoder=IdentityDecoder(),
                             initialize=create_real_vector(bounds=bounds)
                         ),

                         pipeline=pipeline)

    best_genome = None
    print('generation, bsf')
    for g, ind in ea:
        print(f"{g}, {ind.fitness}")
        best_genome = ind.genome

    return best_genome
