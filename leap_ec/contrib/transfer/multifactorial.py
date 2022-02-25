"""
An an example of an island model with a heterogenous configuration: each island
holds a separate fitness function.
"""
import math
import sys

from matplotlib import pyplot as plt
import networkx as nx
from toolz import curry

from leap_ec.individual import Individual
from leap_ec.decoder import IdentityDecoder
from leap_ec.representation import Representation
from leap_ec.algorithm import multi_population_ea
from leap_ec.global_vars import context

import leap_ec.ops as ops
from leap_ec import probe
from leap_ec.algorithm import multi_population_ea

from leap_ec.real_rep. problems import ScaledProblem, TranslatedProblem, SpheroidProblem, RastriginProblem, AckleyProblem
from leap_ec.real_rep.ops import mutate_gaussian
from leap_ec.real_rep.initializers import create_real_vector


##############################
# viz_plots function
##############################
def viz_plots(problems, modulo):
    """A convenience method that creates a figure with grid of subplots for
    visualizing the population genotypes and best-of-gen fitness for a number
    of different problems.

    :return: two lists of probe operators (for the phenotypes and fitness,
    respectively). Insert these into your algorithms to plot measurements to
    the respective subplots. """

    num_rows = min(4, len(problems))
    num_columns = math.ceil(len(problems) / num_rows)
    true_rows = len(problems) / num_columns
    fig = plt.figure(figsize=(6 * num_columns, 2 * true_rows))
    fig.tight_layout()
    genotype_probes = []
    fitness_probes = []
    for i, p in enumerate(problems):
        plt.subplot(true_rows, num_columns * 2, 2 * i + 1)
        tp = probe.PlotTrajectoryProbe(
            contours=p,
            xlim=p.bounds,
            ylim=p.bounds,
            modulo=modulo,
            ax=plt.gca())
        genotype_probes.append(tp)

        plt.subplot(true_rows, num_columns * 2, 2 * i + 2)
        fp = probe.PopulationPlotProbe(ylim=(0, 1), modulo=modulo, ax=plt.gca())
        fitness_probes.append(fp)

    plt.subplots_adjust(
        left=0.05,
        bottom=0.05,
        right=0.95,
        top=0.95,
        wspace=0.2,
        hspace=0.2)

    return genotype_probes, fitness_probes


##############################
# function problem_stamp()
##############################
def problem_stamp(problems):
    """This closure returns a callback that stamps individuals with the current
    island's problem as they go past.  This catches immigrants as they arrive,
    and notifies them that future fitness evaluations should be conducted on
    a new island's function instead of their home fitness function."""
    def stamp(ind, subpop_id):
        ind.problem = problems[subpop_id]
        ind.evaluate()
        return ind

    return stamp
    
def evaluate_and_assign_skill_factors(population, problems):
    for ind in population:
        phenome = ind.decode()
        factorial_costs = [ p.evaluate(phenome) for p in problems ]
        # TODO Convert this into factorial ranks, and then skill factors


##############################
# main
##############################
if __name__ == '__main__':
    # file_probe = probe.AttributesCSVProbe(context, stream=sys.stdout, do_fitness=True, do_genome=True)

    l = 2
    bounds = (0, 1)

    def transform(problem):
        return TranslatedProblem.random(ScaledProblem(problem, new_bounds=bounds), (-0.5, 0.5), l)

    # Island-specific fitness functions
    problems = [ transform(SpheroidProblem(maximize=False)),
                 transform(RastriginProblem(maximize=False)),
                 transform(AckleyProblem(maximize=False))
               ]
    
    # Probes and visualization
    genotype_probes, fitness_probes = viz_plots(problems, modulo=10)
    subpop_probes = list(zip(genotype_probes, fitness_probes))

    parents = Individual.create_population(pop_size,
                                           initialize=create_real_vector(
                                                bounds=[bounds] * l),
                                           decoder=IdentityDecoder(),
                                           problem=None)
                                                



    def get_island(context):
        """Closure that returns a callback for retrieving the current island
        ID during logging."""
        return lambda _: context['leap']['current_subpopulation']

    pop_size = 10
    ea = multi_population_ea(generations=100,
                             num_populations=topology.number_of_nodes(),
                             pop_size=pop_size,
                             problem=problems,  # Fitness function

                             # Representation
                             representation=Representation(
                                 individual_cls=Individual,
                                 initialize=create_real_vector(
                                     bounds=[bounds] * l),
                                 decoder=IdentityDecoder()
                             ),

                             # Operator pipeline
                             shared_pipeline=[
                                 ops.tournament_selection,
                                 ops.clone,
                                 mutate_gaussian(std=0.03, expected_num_mutations=1, hard_bounds=bounds),
                                 ops.evaluate,
                                 ops.pool(size=pop_size),
                                 ops.migrate(topology=topology,
                                             emigrant_selector=ops.tournament_selection,
                                             replacement_selector=ops.random_selection,
                                             migration_gap=5,
                                             customs_stamp=problem_stamp(problems)),
                                 probe.FitnessStatsCSVProbe(stream=sys.stdout,
                                        computed_columns={ 'island': get_island(context) })
                             ],
                             subpop_pipelines=subpop_probes)

    list(ea)
