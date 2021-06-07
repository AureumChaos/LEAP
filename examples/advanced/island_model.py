"""
    Provides an island model example.
"""
import math
import os
import sys

from matplotlib import pyplot as plt
import networkx as nx

from leap_ec import Individual, Representation, context, test_env_var
from leap_ec import ops, probe
from leap_ec.algorithm import multi_population_ea
from leap_ec.real_rep.problems import SchwefelProblem
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
    fig = plt.figure(figsize=(6 * num_columns, 2.5 * true_rows))
    fig.tight_layout()
    genotype_probes = []
    fitness_probes = []
    for i, p in enumerate(problems):
        plt.subplot(true_rows, num_columns * 2, 2 * i + 1)
        tp = probe.CartesianPhenotypePlotProbe(
            contours=p,
            xlim=p.bounds,
            ylim=p.bounds,
            modulo=modulo,
            ax=plt.gca())
        genotype_probes.append(tp)

        plt.subplot(true_rows, num_columns * 2, 2 * i + 2)
        fp = probe.FitnessPlotProbe(ylim=(0, 1), modulo=modulo, ax=plt.gca())
        fitness_probes.append(fp)

    plt.subplots_adjust(
        left=0.05,
        bottom=0.05,
        right=0.95,
        top=0.95,
        wspace=0.2,
        hspace=0.3)

    return genotype_probes, fitness_probes


##############################
# Entry point
##############################
if __name__ == '__main__':
    # Set up up the network of connections between islands
    topology = nx.complete_graph(3)
    nx.draw(topology)
    problem = SchwefelProblem(maximize=False)

    genotype_probes, fitness_probes = viz_plots(
        [problem] * topology.number_of_nodes(), modulo=10)
    subpop_probes = list(zip(genotype_probes, fitness_probes))

    def get_island(context):
        """Closure that returns a callback for retrieving the current island
        ID during logging."""
        return lambda _: context['leap']['current_subpopulation']
    
    # When running the test harness, just run for two generations
    # (we use this to quickly ensure our examples don't get bitrot)
    if os.environ.get(test_env_var, False) == 'True':
        generations = 2
    else:
        generations = 1000

    
    l = 2
    pop_size = 10
    ea = multi_population_ea(max_generations=generations,
                             num_populations=topology.number_of_nodes(),
                             pop_size=pop_size,
                             problem=problem,  # Fitness function

                             # Representation
                             representation=Representation(
                                 individual_cls=Individual,
                                 initialize=create_real_vector(
                                     bounds=[problem.bounds] * l)
                             ),

                             # Operator pipeline
                             shared_pipeline=[
                                 ops.tournament_selection,
                                 ops.clone,
                                 mutate_gaussian(
                                     std=30,
                                     expected_num_mutations=1,
                                     hard_bounds=problem.bounds),
                                 ops.evaluate,
                                 ops.pool(size=pop_size),
                                 ops.migrate(topology=topology,
                                             emigrant_selector=ops.tournament_selection,
                                             replacement_selector=ops.random_selection,
                                             migration_gap=50),
                                 probe.FitnessStatsCSVProbe(stream=sys.stdout,
                                        extra_metrics={ 'island': get_island(context) })
                             ],
                             subpop_pipelines=subpop_probes)

    list(ea)
