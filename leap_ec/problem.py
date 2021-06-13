"""
Defines the abstract-base classes Problem, ScalarProblem,
and FunctionProblem.

"""
from abc import ABC, abstractmethod
from itertools import islice
import logging
from math import nan, floor, isclose, isnan
import random
from subprocess import Popen, PIPE, STDOUT

import numpy as np

from leap_ec import leap_logger_name
from leap_ec.global_vars import context


# Set up a logger using LEAP's global logger name
logger = logging.getLogger(leap_logger_name)


##############################
# Class Problem
##############################
class Problem(ABC):
    """
        Abstract Base Class used to define problem definitions.

        A `Problem` is in charge of two major parts of an EA's behavior:

         1. Fitness evaluation (the `evaluate()` method)

         2. Fitness comparision (the `worse_than()` and `equivalent()` methods)
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def evaluate(self, phenome, *args, **kwargs):
        """
        Evaluate the given individual based on its decoded phenome.

        Practitioners *must* over-ride this member function.

        Note that by default the individual comparison operators assume a
        maximization problem; if this is a minimization problem, then just
        negate the value when returning the fitness.

        :param phenome:
        :return: fitness
        """
        raise NotImplementedError

    def evaluate_multiple(self, phenomes):
        """Evaluate multiple individuals all at once, returning a list of fitness
        values.
        
        By default this just calls `self.evaluate()` multiple times.  Override this
        if you need to, say, send a group of individuals off to parallel """
        return [ self.evaluate(p) for p in phenomes ]

    @abstractmethod
    def worse_than(self, first_fitness, second_fitness):
        raise NotImplementedError

    @abstractmethod
    def equivalent(self, first_fitness, second_fitness):
        raise NotImplementedError


##############################
# Class ScalarProblem
##############################
class ScalarProblem(Problem):
    def __init__(self, maximize):
        super().__init__()
        self.maximize = maximize

    def worse_than(self, first_fitness, second_fitness):
        """
            Used in Individual.__lt__().

            By default returns first_fitness < second_fitness if a maximization
            problem, else first_fitness > second_fitness if a minimization
            problem.  Please over-ride if this does not hold for your problem.

            :return: true if the first individual is less fit than the second
        """
        # NaN is assigned if the individual is non-viable, which can happen if
        # an exception is thrown during evaluation. We consider NaN fitnesses to
        # always be the worse possible with regards to ordering.
        if isnan(first_fitness):
            if isnan(second_fitness):
                # both are nan, so to reduce bias flip a coin to arbitrarily
                # select one that is worst.
                return random.choice([True, False])
            # Doesn't matter how awful second_fitness is, nan will already be
            # considered worse.
            return True
        elif isnan(second_fitness):
            # No matter how awful the first_fitness is, if it's not a NaN the
            # NaN will always be worse
            return False

        # TODO If we accidentally pass an Individual in as first_ or second_fitness,
        # TODO then this can result in an infinite loop.  Add some error
        # handling for this.
        if self.maximize:
            return first_fitness < second_fitness
        else:
            return first_fitness > second_fitness

    def equivalent(self, first_fitness, second_fitness):
        """
            Used in Individual.__eq__().

            By default returns first.fitness== second.fitness.  Please
            over-ride if this does not hold for your problem.

            :return: true if the first individual is equal to the second
        """

        # Since we're comparing two real values, we need to be a little
        # smarter about that.  This will return true if the difference
        # between the two is within a small tolerance. This also handles
        # NaNs, inf, and -inf.
        if type(first_fitness) == float and type(second_fitness) == float:
            return isclose(first_fitness, second_fitness)
        else: # fallback if one or more are not floats
            return first_fitness == second_fitness


##############################
# Class FunctionProblem
##############################
class FunctionProblem(ScalarProblem):

    def __init__(self, fitness_function, maximize):
        super().__init__(maximize)
        self.fitness_function = fitness_function

    def evaluate(self, phenome, *args, **kwargs):
        return self.fitness_function(phenome, *args, **kwargs)


##############################
# Class ConstantProblem
##############################
class ConstantProblem(ScalarProblem):
    """A flat landscape, where all phenotypes have the same fitness.

    This is sometimes useful for sanity checks or as a control in certain
    kinds of research.

    .. math::

       f(\\vec{x}) = c

    :param float c: the fitness value to return for any input.

    .. plot::
       :include-source:

       from leap_ec.problem import ConstantProblem
       from leap_ec.real_rep.problems import plot_2d_problem
       bounds = ConstantProblem.bounds
       plot_2d_problem(ConstantProblem(), xlim=bounds, ylim=bounds, granularity=0.025)

    """

    """Default bounds."""
    bounds = (-1.0, 1.0)

    def __init__(self, maximize=False, c=1.0):
        super().__init__(maximize)
        self.c = c

    def evaluate(self, phenome, *args, **kwargs):
        """
        Return a contant value for any input phenome:

        >>> phenome = [0.5, 0.8, 1.5]
        >>> ConstantProblem().evaluate(phenome)
        1.0

        >>> ConstantProblem(c=500.0).evaluate('foo bar')
        500.0

        :param phenome: real-valued vector to be evaluated
        :return: 1.0, or the constant defined in the constructor
        """
        return self.c

    def __str__(self):
        return ConstantProblem.__name__


################################
# Class ExternalProcessProblem
################################
class ExternalProcessProblem(ScalarProblem):
    """
    Evaluate individuals by launching an external program, writing phenomes to its stdin
    as CSV rows, and reading back fitness values from its stdout.

    Assumes that individuals are represented with list phenomes with elements that can
    be cast to strings.
    """
    def __init__(self, command: str, maximize: bool, args: list = None, ):
        super().__init__(maximize=maximize)
        self.command = command
        self.args = args[:] if args else []
        
    def evaluate(self, phenome):
        fitnesses = self.evaluate_multiple([ phenome ])
        assert(len(fitnesses) == 1)
        return fitnesses[0]
    
    def evaluate_multiple(self, phenomes):
        # Convert the phenomes into one big string
        def phenome_to_str(p):
            return ','.join([ str(x) for x in p ])
        phenome_bytes = '\n'.join([ phenome_to_str(p) for p in phenomes ]).encode()
        
        logger.debug(f"Input: {phenome_bytes}")

        # Start the external process and send the phenomes to its stdin
        p = Popen([self.command] + self.args, stdout=PIPE, stdin=PIPE, stderr=PIPE)
        outs, errs = p.communicate(input=phenome_bytes)

        # Receive output back from the external process
        logger.debug(f"Simulation-stdout: {outs}")
        logger.debug(f"Simulation-stderr: {errs}")

        if p.returncode != 0:
            raise RuntimeError(f"Error in the external simulation during fitness evaluation.")
        
        out_strs = outs.split(b'\n')[:-1]  # Ignoring  trailing newline
        fitnesses = [ float(o) for o in out_strs]
        
        if len(fitnesses) != len(phenomes):
            raise RuntimeError(f"Expected to receive {len(phenomes)} fitness values back from external simulation, but actually received {len(fitnesses)}.")

        logger.debug(f"Fitnesses: {fitnesses}\n")
            
        return fitnesses


########################
# class AverageFitnessProblem
########################
class AverageFitnessProblem(Problem):
    """Problem wrapper that copies each genome n times, evaluates them, and averages the
    results back together to produce a mean-fitness estimate.

    This is a common strategy for approaching noisy fitness functions, to make it easier 
    for an optimization algorithm to follow a gradient.
    
    >>> from leap_ec.real_rep.problems import NoisyQuarticProblem
    >>> p = AverageFitnessProblem(
    ...                 wrapped_problem = NoisyQuarticProblem(),
    ...                 n = 20)
    >>> x = [ 1, 1, 1, 1 ]
    >>> y = p.evaluate(x)
    >>> print(f"Fitness: {y}")  # The mean of this will be approximately 10
    Fitness: ...

    """
    def __init__(self, wrapped_problem, n: int):
        assert(wrapped_problem is not None)
        assert(n > 0)
        assert(hasattr(wrapped_problem, 'evaluate'))
        self.wrapped_problem = wrapped_problem
        self.n = n

    def evaluate(self, phenome):
        """Evaluates the wrapped function n times sequentially and returns the mean."""
        fitnesses = [ self.wrapped_problem.evaluate(phenome) for _ in range(self.n) ]
        return np.mean(fitnesses)

    def multiple_evaluate(self, phenomes: list):
        """
        Evaluate a collections of phenomes by creating n jobs for each phenome,
        sending all the jobs to the wrapped multiple_evaluate() function, and then
        averaging the n results for each phenome into a list of results.
        """
        def mean_by_chunk(l):
            """Take n elements at a time from an iterator and average them."""
            means = []
            while l != []:
                chunk, l = l[:self.n], l[self.n:]
                means.append(np.mean(chunk))
            return means
            
        # Copy each phenome n times, because we're going to evaluate each one n times
        expanded_phenomes = [ p for p in phenomes for _ in range(self.n) ]

        # Evaluate them
        fitnesses = self.wrapped_problem.multiple_evaluate(expanded_phenomes)

        # Average the copies back together
        contracted_phenomes = mean_by_chunk(fitnesses)

        assert(len(contracted_phenomes) == len(phenomes))
        return contracted_phenomes

    def worse_than(self, first_fitness, second_fitness):
        return self.wrapped_problem.worse_than(first_fitness, second_fitness)

    def equivalent(self, first_fitness, second_fitness):
        return self.wrapped_problem.equivalent(first_fitness, second_fitness)
        

########################
# Class AlternatingProblem
########################
class AlternatingProblem(Problem):
    def __init__(self, problems, modulo, context=context):
        assert(len(problems) > 0)
        assert(modulo > 0)
        assert(context is not None)
        self.problems = problems
        self.modulo = modulo
        self.context = context
        self.current_problem_idx = 0

    def _get_current_problem(self):
        assert('leap' in self.context)
        assert('generation' in self.context['leap'])
        step = self.context['leap']['generation']

        i = floor(step / self.modulo) % len(self.problems)

        return self.problems[i]

    def evaluate(self, phenome):
        return self._get_current_problem().evaluate(phenome)

    def worse_than(self, first_fitness, second_fitness):
        return self._get_current_problem().worse_than(first_fitness, second_fitness)

    def equivalent(self, first_fitness, second_fitness):
        return self._get_current_problem().equivalent(first_fitness, second_fitness)
