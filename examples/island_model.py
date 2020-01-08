import sys

from matplotlib import pyplot as plt
import networkx as nx

from leap import core, ops, probe, real_problems
from leap.algorithm import multi_population_ea


def viz_plots(problems, modulo):
    """Create a figure with subplots for visualizing the population genotypes and best-of-gen
    fitness for each problem.

    :return: two lists of probe operators (for the phenotypes and fitness, respectively).
        Insert these into your algorithms to plot measurements to the respective subplots."""
    plt.figure(figsize=(4, 0.75 * len(problems)))
    genotype_probes = []
    fitness_probes = []
    for i, p in enumerate(problems):
        plt.subplot(len(problems), 2, 2 * i + 1)
        tp = probe.PlotTrajectoryProbe(core.context, contours=p, xlim=p.bounds, ylim=p.bounds, modulo=modulo, ax=plt.gca())
        genotype_probes.append(tp)

        plt.subplot(len(problems), 2, 2 * i + 2)
        fp = probe.PopulationPlotProbe(core.context, ylim=(0, 1), modulo=modulo, ax=plt.gca())
        fitness_probes.append(fp)

    return genotype_probes, fitness_probes


if __name__ == '__main__':
    #file_probe = probe.AttributesCSVProbe(core.context, stream=sys.stdout, do_fitness=True, do_genome=True)

    topology = nx.complete_graph(12)
    nx.draw(topology)
    problem = real_problems.SchwefelProblem(maximize=False)

    genotype_probes, fitness_probes = viz_plots([problem]*topology.number_of_nodes(), modulo=10)
    subpop_probes = list(zip(genotype_probes, fitness_probes))

    l = 2
    pop_size = 10
    ea = multi_population_ea(generations=1000, num_populations=topology.number_of_nodes(), pop_size=pop_size,
                             individual_cls=core.Individual,

                             decoder=core.IdentityDecoder(),
                             problem=problem,
                             initialize=core.create_real_vector(bounds=[problem.bounds] * l),

                             pipeline=[
                                 ops.tournament,
                                 ops.clone,
                                 ops.mutate_gaussian(std=30, hard_bounds=problem.bounds),
                                 ops.evaluate,
                                 ops.pool(size=pop_size),
                                 ops.migrate(core.context,
                                             topology=topology,
                                             emigrant_selector=ops.tournament,
                                             replacement_selector=ops.random_selection,
                                             migration_gap=20),
                                 probe.FitnessStatsCSVProbe(core.context, stream=sys.stdout)
                             ],
                             subpop_pipelines=subpop_probes)

    list(ea)
    plt.show()
