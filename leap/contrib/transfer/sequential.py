import abc


class Repertoire(abc.ABC):

    @abc.abstractmethod
    def build_repertoire(self, problems, initialize, algorithm):
        pass

    @abc.abstractmethod
    def apply(self, problem, algorithm):
        pass


class PopulationSeedingRepertoire:
    def __init__(self, problems, initialize, algorithm, problem_kwargs):
        assert(problems is not None)
        assert(len(problems) >= 0)
        assert(algorithm is not None)
        self.repertoire = []
        self.problems = problems
        self.initialize = initialize
        self.algorithm = algorithm
        self.problem_kwargs = problem_kwargs

    def build_repertoire(self):
        results = [self.algorithm(p, self.initialize, **self.problem_kwargs[i]) for i, p in enumerate(self.problems)]
        results = list(zip(*results))  # Execute the algorithm concurrently on all the source tasks
        results = zip(*results)  # Unzip them into individual BSF trajectories
        #assert(len(results) == len(self.problems))
        for r in results:
            last_step, last_ind = r[-1]
            self.repertoire.append(last_ind.genome)

    def apply(self, problem, **kwargs):
        repertoire_init = initialize_seeded(self.initialize, self.repertoire)
        return self.algorithm(problem, repertoire_init, **kwargs)


def initialize_seeded(initialize, seed_pop):
    """A population initializer that injects a fixed list of seed individuals
    into the population, and fills the remaining space with newly generated individuals.

    >>> from leap import core
    >>> random_init = core.create_real_vector(bounds=[[0, 0]] * 2)
    >>> init = initialize_seeded(random_init, [[5.0, 5.0], [4.5, -6]])
    >>> [init() for _ in range(5)]
    [[5.0, 5.0], [4.5, -6], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]

    """
    assert (initialize is not None)
    assert (seed_pop is not None)

    i = 0

    def create():
        nonlocal i
        if i < len(seed_pop):
            ind = seed_pop[i]
            i += 1
            return ind
        else:
            return initialize()

    return create



