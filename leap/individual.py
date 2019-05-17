#!/usr/bin/env python3
"""
    Classes related to individuals that represent posed solutions.


    TODO Need to decide if the logic in __eq__ and __lt__ is overly complex.
    I like that this reduces the dependency on Individuals on a Problem
    (because sometimes you have a super simple situation that doesn't require
    explicitly couching your problem in a Problem subclass.)
"""
from functools import total_ordering
import itertools
from datetime import datetime


from uuid import uuid4
from copy import deepcopy


@total_ordering
class Individual:
    """
        A single single individual that represents a solution instance.

        Presumes fitness is a comparable value because this class relies on
        @total_ordering.  This means care must be given to the meaning of
        fitness with regards to maximization vs. minimization problems since
        the comparable functions are heavily used in selection.  By default,
        this assumes a maximization problem;
        if not, then problem.evaluate() needs to flip the fitness value,
        possibly via negation; or the __lt__() function needs to be adjusted.

        problem: is the corresponding problem for which this individual
                 represents a posed solution
        encoding: how the individual represents its values
        fitness: denotes an associated quality
        uuid: is a UUID unique to each created individual
        birth: is the unique birth number for each created individual
        start_eval: is the time stamp when the individual has _started_ being evaluated
        end_eval: is the time stamp when the individual has _finished_ being evaluated
    """
    birth_counter = itertools.count(0) # for counting each newly created individual

    def __init__(self, problem=None, encoding=None):
        """
        :param problem: is the problem associated with this individual
        """
        self.problem = problem

        # deep copy is necessary because otherwise *all* individuals would share the same state.
        # TODO maybe we should pass in a factory function for creating unique encodings for each individual as this
        # seems kludgy.
        if encoding is not None:
            # There may be unusual situations whereby someone may actually create
            # an individual with no encoding.
            # TODO Should this be an exception, instead?
            self.encoding = deepcopy(encoding)
            self.encoding.random() # set a random, initial value (TODO do we want to always do this?)

        self.fitness = None  # TODO should fitness be a property?

        # Generate unique integer ID; not that this is also set in self.clone()
        self.birth = next(Individual.birth_counter)

        # Generate a unique identifier for this individual
        self.uuid = uuid4()

        # start and end evaluation times stamps; set in evaluate()
        self.start_eval = None
        self.end_eval = None



    def clone(self):
        """ Returns an almost identical copy of this individual

        Almost because self.birth and self.uuid will be unique for the newly created individual;
        and those values are set via the __deepcopy__ and __init__ member functions.

        :return: a copy of this individual
        """
        cloned = deepcopy(self)

        # Since we've technically created a new individual, let's update the global individual counter
        cloned.birth = next(Individual.birth_counter)

        # We don't want to reuse the UUID from whom we copied; that wouldn't be so unique, now would it?  ;)
        cloned.uuid = uuid4()

        cloned.fitness = None

        return cloned



    def evaluate(self):
        """ Evaluates this individual

        :return: None
        """
        self.start_eval = datetime.now()

        # Always create an end_val time stamp even if there is an exception
        try:
            self.fitness = self.problem.evaluate(self)
        finally:
            self.end_eval = datetime.now()


    def is_viable(self):
        """ Within this context viable means having been evaluated, so fitness will not be None.

        The intent is for a subclass to over-ride this to represent scenarios where the individual *has* been
        evaluated, but does not represent a proper solution. For example, the individual may represent a deep-
        learner configuration that is malformed in some way, so it will have been evaluated and thus have a non-None
        fitness, but yet not represent a proper solution.

        :return: True if this individual represents a viable solution
        """
        return self.fitness is not None


    def decode(self):
        """
        :return: the decoded value for this individual
        """
        return self.encoding.decode()


    def __iter__(self):
        """
        :raises: exception if self.encoding is None
        :return: the encapsulated encoding's iterator
        """
        return self.encoding.__iter__()


    def __eq__(self, other):
        """
        Note that the associated problem knows best how to compare associated
        individuals

        :param other: to which to compare
        :return: if this Individual is less fit than another
        """
        return self.problem.same_as(self, other)


    def __lt__(self, other):
        """
        Note that the associated problem knows best how to compare associated
        individuals

        :param other: to which to compare
        :return: if this Individual has the same fitness as another even if
                 they have different genomes
        """
        return self.problem.worse_than(self, other)


    @classmethod
    def create_population(cls, size, problem=None, encoding=None):
        """ Create a population of individuals

        The individuals are bound to problem and use encoding to create
        initial values.

        :param size: dictates how many individuals are to be created
        :param problem: binds the associated problem with the individual
        :param encoding: how the individual represents its values
        :return a collection of randomly generated individuals
        """
        return [cls(problem, encoding) for _ in range(size)]



