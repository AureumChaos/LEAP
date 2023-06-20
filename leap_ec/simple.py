"""
    Provides a very high-level convenience function for a very general EA,
    ea_solve().
"""
import sys

from matplotlib import pyplot as plt
from multiprocessing import cpu_count

from leap_ec import Representation
from leap_ec import ops, probe
from leap_ec.algorithm import generational_ea
from leap_ec.distrib import DistributedIndividual
from leap_ec.distrib import synchronous
from leap_ec.probe import BestSoFarProbe
from leap_ec.problem import FunctionProblem
from leap_ec.real_rep import create_real_vector
from leap_ec.real_rep.ops import mutate_gaussian


##############################
# Function ea_solve()
##############################
def ea_solve(function, bounds, generations=100, pop_size=cpu_count(),
             mutation_std=1.0, maximize=False, viz=False, viz_ylim=(0, 1),
             hard_bounds=True, dask_client=None, stream=sys.stdout):
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
    :param dask_client: is optional dask Client to enable parallel evaluations
    :param stream: a stream to write best-so-far values to (defaults to stdout)

    The basic call includes instrumentation that prints the best-so-far fitness
    value of each generation to stdout:

    >>> import io
    >>> from leap_ec.simple import ea_solve
    >>> stream = io.StringIO()
    >>> ea_solve(sum, bounds=[(0, 1)]*5, stream=stream) # doctest:+ELLIPSIS
    array([..., ..., ..., ..., ...])

    The stream captures the best-so-far individual at each iteration of the algorithm:
    >>> print(stream.getvalue())  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    step,bsf
    0,...
    1,...
    2,...
    ...
    99,...
    <BLANKLINE>

    When `viz=True`, a live BSF plot will also display:

    >>> ea_solve(sum, bounds=[(0, 1)]*5, viz=True) # doctest:+ELLIPSIS
    array([..., ..., ..., ..., ...])

    .. plot::

        from leap_ec.simple import ea_solve
        ea_solve(sum, bounds=[(0, 1)]*5, viz=True)

    """

    ########## Parallelization ##########
    # If a dask client is given, then use the synchronous (map/reduce) parallel
    # evaluation of individuals; else, revert to serial evaluations.
    if dask_client:
        eval_ops = [ synchronous.eval_pool(client=dask_client, size=pop_size) ]
    else:
        eval_ops = [ops.evaluate,
                    ops.pool(size=pop_size)]

    ########## Operators ##########
    # Isotropic Gaussian mutation operator
    if hard_bounds:
        mutation_op = mutate_gaussian(std=mutation_std, bounds=bounds,
                                      expected_num_mutations='isotropic')
    else:
        mutation_op = mutate_gaussian(std=mutation_std,
                                      expected_num_mutations='isotropic')

    bsf_probe = BestSoFarProbe(stream=stream)  # Bind this probe to a varaible so we can access it after the run
    pipeline = [
        ops.tournament_selection,
        ops.clone,
        mutation_op,
        ops.UniformCrossover(p_swap=0.2),
        *eval_ops,
        bsf_probe
    ]

    ########## Visualization ##########
    if viz:
        plot_probe = probe.FitnessPlotProbe(ylim=viz_ylim, ax=plt.gca())
        pipeline.append(plot_probe)


    ########## Run! ##########
    final_pop = generational_ea(max_generations=generations,
                         pop_size=pop_size,
                         problem=FunctionProblem(function, maximize),

                         representation=Representation(
                             individual_cls=DistributedIndividual,
                             initialize=create_real_vector(bounds=bounds)
                         ),

                         pipeline=pipeline)

    # Return the genome of the best-found individual
    return bsf_probe.bsf.genome
