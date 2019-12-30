from matplotlib import pyplot as plt

from leap.example.simple_ea import simple_ea
from leap import core, real, ops
from leap.probe import PlotTrajectoryProbe, PlotProbe


def initialize_seeded(initialize, seed_pop):
    """A population initializer that injects a fixed list of seed individuals
    into the population, and fills the remaining space with newly generated individuals.

    >>> from leap import real_problems
    >>> random_init = real_problems.initialize_vectors_uniform(bounds=[[0, 0]] * 2)
    >>> init = initialize_seeded(random_init, [[5.0, 5.0], [4.5, -6]])
    >>> init(5)
    [[5.0, 5.0], [4.5, -6], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]

    """
    assert (initialize is not None)
    assert (seed_pop is not None)

    def f(pop_size):
        assert (pop_size >= len(seed_pop))
        n_new = pop_size - len(seed_pop)
        return seed_pop + initialize(n_new)

    return f


def ea(problem, initialize, probes=[], step_notify=[], evals=200, l=2):
    pop_size=5
    mutate_prob=1/l
    mutate_std=0.5

    ea = simple_ea(evals=evals, pop_size=pop_size,
                   individual_cls=core.Individual,  # Use the standard Individual as the prototype for the population.
                   decoder=core.IdentityDecoder(),  # Genotype and phenotype are the same for this task.
                   problem=problem,
                   evaluate=ops.evaluate,  # Evaluate fitness with the basic evaluation operator.

                   initialize=initialize,

                   step_notify_list=step_notify,

                   # The operator pipeline.
                   pipeline=probes + [
                       # Select mu parents via tournament selection.
                       ops.tournament(n=pop_size),
                       # Clone them to create offspring.
                       ops.cloning,
                       # Apply Gaussian mutation to each gene with a certain probability.
                       ops.mutate_gaussian(prob=mutate_prob, std=mutate_std, hard_bounds=(-5.12, 5.12))
                   ])
    return ea


def quad_probes(modulo):
    plt.figure(figsize=(8, 6))  # Setup a figure to plot to
    plt.subplot(221)
    trajectory_probe_1 = PlotTrajectoryProbe(contours=problem1, xlim=(-5.12, 5.12), ylim=(-5.12, 5.12), granularity=0.1, modulo=modulo, ax=plt.gca())
    plt.subplot(222)
    bsf_probe_1 = PlotProbe(ylim=(0, 1), modulo=modulo, ax=plt.gca())
    plt.subplot(223)
    trajectory_probe_2 = PlotTrajectoryProbe(contours=problem2, xlim=(-5.12, 5.12), ylim=(-5.12, 5.12), granularity=0.1, modulo=modulo, ax=plt.gca())
    plt.subplot(224)
    bsf_probe_2 = PlotProbe(ylim=(0, 1), modulo=modulo, ax=plt.gca())
    return trajectory_probe_1, bsf_probe_1, trajectory_probe_2, bsf_probe_2


if __name__ == '__main__':
    random_init = real_problems.initialize_vectors_uniform(bounds=[[-0.5, 0.5]] * 2)

    problem1 = real_problems.Spheroid()
    problem2 = real_problems.Rastrigin(a=10)

    trajectory_probe_1, bsf_probe_1, trajectory_probe_2, bsf_probe_2 = quad_probes(50)
    plt.draw()

    def do_sequential_transfer(ea1, ea2):
        result = list(ea1())
        best_ind = list(result)[-1][1]
        print("Best ind on source task: " + str(best_ind))
        list(ea2(initialize_seeded(random_init, [best_ind.genome])))


    def ea1():
        return ea(problem1, random_init, [], [], evals=500)


    def ea2(initialize):
        return ea(problem2, initialize, [trajectory_probe_2, bsf_probe_2],
                  [trajectory_probe_2.set_step, bsf_probe_2.set_step], evals=500)


    do_sequential_transfer(ea1, ea2)
