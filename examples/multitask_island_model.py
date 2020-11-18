"""
An an example of an island model with a heterogenous configuration: each island
holds a separate fitness function.
"""
import math
import sys

from matplotlib import pyplot as plt
import networkx as nx

from leap_ec.individual import Individual
from leap_ec.decoder import IdentityDecoder
from leap_ec.representation import Representation
from leap_ec.algorithm import multi_population_ea
from leap_ec.context import context

import leap_ec.ops as ops
from leap_ec import probe
from leap_ec.algorithm import multi_population_ea

from leap_ec.real_rep import problems
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
            context,
            contours=p,
            xlim=p.bounds,
            ylim=p.bounds,
            modulo=modulo,
            ax=plt.gca())
        genotype_probes.append(tp)

        plt.subplot(true_rows, num_columns * 2, 2 * i + 2)
        fp = probe.PopulationPlotProbe(
            context, ylim=(
                0, 1), modulo=modulo, ax=plt.gca())
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
# main
##############################
if __name__ == '__main__':
    # file_probe = probe.AttributesCSVProbe(context, stream=sys.stdout, do_fitness=True, do_genome=True)

    topology = nx.complete_graph(3)
    nx.draw(topology)

    bounds = (0, 1)

    # Island-specific fitness functions
    topology.nodes[0]['problem'] = problems.ScaledProblem(problems.SpheroidProblem(), new_bounds=(0, 1))
    topology.nodes[1]['problem'] = problems.ScaledProblem(problems.RastriginProblem(), new_bounds=(0, 1))
    topology.nodes[2]['problem'] = problems.ScaledProblem(problems.AckleyProblem(), new_bounds=(0, 1))
    
    # Probes and visualization
    problems = [ n['problem'] for n in topology.nodes ]
    genotype_probes, fitness_probes = viz_plots(problems, modulo=10)
    subpop_probes = list(zip(genotype_probes, fitness_probes))

    l = 2
    pop_size = 10
    ea = multi_population_ea(generations=1000,
                             num_populations=topology.number_of_nodes(),
                             pop_size=pop_size,
                             problem=problem,  # Fitness function

                             # Representation
                             representation=Representation(
                                 individual_cls=Individual,
                                 initialize=create_real_vector(
                                     bounds=bounds * l),
                                 decoder=IdentityDecoder()
                             ),

                             # Operator pipeline
                             shared_pipeline=[
                                 ops.tournament_selection,
                                 ops.clone,
                                 mutate_gaussian(
                                     std=30, hard_bounds=bounds),
                                 ops.evaluate,
                                 ops.pool(size=pop_size),
                                 ops.migrate(context,
                                             topology=topology,
                                             emigrant_selector=ops.tournament_selection,
                                             replacement_selector=ops.random_selection,
                                             migration_gap=50),
                                 probe.FitnessStatsCSVProbe(
                                     context, stream=sys.stdout)
                             ],
                             subpop_pipelines=subpop_probes)

    list(ea)
    plt.show()
