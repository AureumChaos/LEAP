from abc import ABC, abstractmethod


##############################
# Class Problem
##############################
class Problem(ABC):
    """
        Abstract Base Class used to define problem definitions.

        A `Problem` is in charge of two major parts of an EA's behavior:

         1. Fitness evaluation (the `evaluate()` method)

         2. Fitness comparision (the `worse_than()` and `same_as()` methods
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def evaluate(self, individual):
        """
        Decode and evaluate the given individual based on its genome.

        Practicitioners *must* over-ride this member function.

        Note that by default the individual comparison operators assume a
        maximization problem; if this is a minimization problem, then just
        negate the value when returning the fitness.

        :param individual:
        :return: fitness
        """
        raise NotImplementedError

    @abstractmethod
    def worse_than(self, first, second):
        raise NotImplementedError

    @abstractmethod
    def same_as(self, first, second):
        raise NotImplementedError


##############################
# Class ScalarProblem
##############################
class ScalarProblem(Problem):
    def __init__(self, maximize=True):
        self.maximize = maximize

    def worse_than(self, first, second):
        """
            Used in Individual.__lt__().

            By default returnss first.fitness < second.fitness.  Please
            over-ride if this does not hold for your problem.

            :return: true if the first individual is less fit than the second
        """
        if self.maximize:
            return first.fitness < second.fitness
        else:
            return first.fitness > second.fitness

    def same_as(self, first, second):
        """
            Used in Individual.__eq__().

            By default returns first.fitness== second.fitness.  Please
            over-ride if this does not hold for your problem.

            :return: true if the first individual is equal to the second
        """
        return first.fitness == second.fitness


##############################
# Class FunctionProblem
##############################
class FunctionProblem(Problem):

    def __init__(self, fitness_function):
        self.fitness_function = fitness_function

    def evaluate(self, individual):
        phenome = individual.decode()
        return self.fitness_function(phenome)