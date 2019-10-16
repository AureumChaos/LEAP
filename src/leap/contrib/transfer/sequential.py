import abc


class Repertoire(abc.ABC):

    @abc.abstractmethod
    def build_repertoire(self, problems, initialize, algorithm):
        pass

    @abc.abstractmethod
    def apply(self, problem, algorithm):
        pass


class PopulationSeedingRepertoire(Repertoire):
    def __init__(self):
        self.repertoire = []

    def build_repertoire(self, problems, initialize, algorithm):
        assert(problems is not None)
        assert(len(problems) >= 0)
        assert(algorithm is not None)
        results = [algorithm(p, initialize) for p in problems]
        results = list(zip(results))  # Execute the algorithm concurrently on all the source tasks
        results = zip(*results)  # Unzip them into individual BSF trajectories
        assert(len(results) == len(problems))
        for r in results:
            last_step, last_ind = r[-1]
            self.repertoire.append(last_ind.genome)

    def apply(self, problem, initialize, algorithm):
        repertoire_init = initialize_seeded(initialize, self.repertoire)
        return algorithm(problem, repertoire_init)


def initialize_seeded(initialize, seed_pop):
    """A population initializer that injects a fixed list of seed individuals
    into the population, and fills the remaining space with newly generated individuals.

    >>> from leap import real
    >>> random_init = real.initialize_vectors_uniform(bounds=[[0, 0]] * 2)
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



