#! /usr/bin/env python

##############################################################################
#
#   LEAP - Library for Evolutionary Algorithms in Python
#   Copyright (C) 2004  Jeffrey K. Bassett
#
#   This program is free software; you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation; either version 2 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program; if not, write to the Free Software
#   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
#
##############################################################################

# Python 2 & 3 compatibility
from __future__ import print_function

import copy    # for Clone
import random
import math
from queue import Queue

#from gene import *
#from exceptions import *
import LEAP

DEBUG=False

#############################################################################
#
# setSeqVal
#
#############################################################################
def setSeqVal(seq, index, val):
    """
    A generic way of setting a value in a sequence (list, str or tuple).

    @param seq: The sequence to modify.
    @param index: The index of the location in the sequence to modify.
    @param val: The new value to be assigned to the location in the sequence.
    """
    if isinstance(seq, list):
        seq[index] = val
    elif isinstance(seq, str):
        seq = seq[:index] + val + seq[index+1:]
    elif isinstance(seq, tuple):
        seq = seq[:index] + (val,) + seq[index+1:]
    else:
        raise TypeError("The genome must be a sequence.")
    return seq


#############################################################################
#
# class Operator
#
#############################################################################
class Operator:
    """
    A base class for any type of operator: genetic, selection, survival and
    others.  This class is responsible for maintaining the operator's cache,
    which stores extra children if more than one is created.

    This class should only be subclassed, and never instantiated.
    """
    def __init__(self):
        "Initialize the cache."
        self.cache = []   # The cache is a queue
        #self.cache = Queue.Queue()   # unlimited size


    def reinitialize(self, population):
        """
        Clean out the cache.
        In general, this method should be used to perform any special
        operations when a new generation is started.  Therefore, subclasses
        should always call their parent class' reinitialize method.

        @param population: This parameter is not used in this class, but
                           selection and pipeline subclasses will use it.
        """
        del self.cache[:]  # Clean out the cache
        #while not self.cache.empty():
        #    self.cache.get_nowait()


    def apply(self, individuals):
        """
        This method accepts the appropriate number of individuals to perform
        a single operation, and then outputs a list containing the results of
        that operation.  For example, a mutation would take a list of 1 and
        return a list of 1, whereas a recombination might take a list of 2
        and return a list of 2.

        @param individuals: A list of the individuals on which an operation
                            will be performed.
        @return: A list of the individuals which resulted from the operation.
                 Typically these are the same individuals which came in, but
                 with slight modifications.
        """
        raise NotImplementedError     # Subclasses should redefine this


    def addToCache(self, individuals):
        """
        Subclasses may occasionally want to add items to the cache.  This
        method should be used.
        """
        assert(type(individuals) == type([]))
        assert(individuals != [])

        self.cache += individuals
        #for ind in individuals:
        #    self.cache.put_nowait(ind)


    def pull(self):
        """
        Pull an individual down the pipeline, and apply all operators to that
        individual along the way.

        @return: An individual (child).
        """
        if self.cache == []:
            self.applyAndCache()
        return self.cache.pop(0)

        #if self.cache.empty():
        #    self.applyAndCache()
        #return self.cache.get_nowait()

        #result = self.cache.pop(0)
        #print("pull()", len(self.cache), self)
        #return result


    def isAnythingCached(self):
        """
        Checks to see if there is anything cached in this operator.

        @return: A Boolean indicating whether or not there is something in the
                 cache.
        """
        return self.cache != []


    def finishUp(self):
        """
        """
        # XXX The idea was to call this at the end of a generation.  Currently
        # I use the reinitialize method.  I thought this might make things a
        # little clearer.  It's not really implemented yet though.
        pass


#############################################################################
#
# class PipelineOperator
#
# Originally the provider was a single pipeline operator, but I decided to
# allow for multiple providers.  That way operators that require more that
# one parent can use different selection operators for each.  This means that
# one can create a tree instead of just a pipeline.
#
# In order to maintain backward compatability though, I still allow the
# provider to be a single operator, or a list.
#
#############################################################################
class PipelineOperator(Operator):
    """
    This type of operator can attach to another one, thus allowing us to
    create a pipeline.  This acts as a baseclass for just about everything
    except selection operators.
    """
    parentsNeeded = 1   # Subclasses should set this appropriately

    def __init__(self, provider, pApply = 1.0):
        """
        Initialize the PipelineOperator.

        @param provider: The Operator immediately preceding this one in the
                         pipeline.  This can also be a list of Operators.
                         If parentsNeeded > 1, each provider in the list will
                         be used successively to draw an individual.
        """
        Operator.__init__(self)
        self.pApply = pApply
        self.noopPullsLeft = 0
        if isinstance(provider, list):
            self.provider = provider
        else:
            self.provider = [provider]
        self.prevFitnesses = []  # Part of an undo mechanism to improve perf.


    def reinitialize(self, population):
        """
        Reinitialize this operator, and pass the population up the pipeline.
        It is important to call the superclass version of this method when
        overriding this method.

        @param population: The population from which individuals will be
                           drawn.  In an EA, this is usually the current
                           population.  PipelineOperator passes the population
                           up the pipeline until it reaches a
                           SelectionOperator.
        """
        for p in self.provider:
            if p != None:
                p.reinitialize(population)  # pass up pipeline
        Operator.reinitialize(self, population)
        self.noopPullsLeft = 0


    def applyAndCache(self):
        """
        Assembles a list containing the appropriate number of individuals, and
        then calls apply.  The results are then placed in the cache.
        """
        # When not performing an operation, I pass individuals along as soon
        # as I get them.  This way the family probes will only associate
        # individuals with one parent instead of many.

        if self.noopPullsLeft > 0:
            prov = (self.parentsNeeded - self.noopPullsLeft)
            ind = self.provider[prov % len(self.provider)].pull()
            self.addToCache([ind])
            self.noopPullsLeft -= 1
        elif random.random() < self.pApply:  # Apply operator?
            individuals = [self.provider[i % len(self.provider)].pull()
                           for i in range(self.parentsNeeded)]
            self.resetFitnesses(individuals)
            children = self.apply(individuals)
            self.addToCache(children)
        else:                                # Don't apply operator
            self.addToCache([self.provider[0].pull()])
            self.noopPullsLeft = self.parentsNeeded - 1
            


    def resetFitnesses(self, individuals):
        """
        This is part of a mechanism to reduce the number of evaluations
        necessary for an individual.  If this operator modifies the genome
        of an individual, then it also resets the fitness of that
        individual to None so that its fitness must be recalculated in
        Individual.evaluate().  If this operator does not change the genome
        then it leaves the fitness alone, thereby reducing the number of
        evaluations.

        By default we assume that the operator has not changed the genome,
        and so this function does nothing.

        param individuals: The individuals that are about to be operated on.
        """
        pass  # By default, assume this operator does not change the genome.


    def isAnythingCached(self, after = None):
        """
        Checks to see if there is anything cached anywhere in the pipeline
        prior to (and including) this operator.

        @param after: This is an optional parameter that references an operator
                      somewhere in the pipeline.  The query will cease if this
                      operator is reached.
        @return: A Boolean indicating whether or not there is something in the
                 cache.
        """
        if (after == self):
            return False

        if Operator.isAnythingCached(self):
            return True
        else:
            return any([p.isAnythingCached(after) for p in self.provider])



#############################################################################
#
# class GeneticOperator
#
# Genetic operators alter individuals coming down the pipeline, providing
# the variation part of Darwin's reproduction-with-variation.  They attach
# to a selection operator or another genetic operator.  The operator they
# attach to is called the provider.
#
#############################################################################
class GeneticOperator(PipelineOperator):
    "Base class for genetic operators"

    def resetFitnesses(self, individuals):
        """
        For GeneticOperators we will assume that the genome will be changed by
        default.  If it turns out the operator does not change the genome,
        extra evaluations can be avoided by calling
        Individual.genomeUnchanged().

        @param individuals: The individuals (children) that are about to be
                            operated on.
        """
        for ind in individuals:
            ind.resetFitness()


#############################################################################
#
# class WrapperOperator
#
#############################################################################
class WrapperOperator(GeneticOperator):
    """
    Wraps one or more other GeneticOperators.  Can be used for creating
    operators which alternatively activate (i.e. crossover OR mutation).
    It may also be useful to subclass from this operator to create probes for
    measure operator effects.

    Note: The operators that are wrapped will never pull anything down the
          pipeline themselves.  Instead, this class will call their apply()
          functions.  Be cautious of any side-effects that this might cause.
    """
    parentsNeeded = 1

    def __init__(self, provider, wrappedOps, opProbs):
        GeneticOperator.__init__(self, provider)

        # Check for errors
        epsilon = 0.001   # We may want a smaller epsilon here
        sp = sum(opProbs)
        assert(sp > 1-epsilon and sp < 1+epsilon)

        self.wrappedOps = wrappedOps

        # Build a roulette wheel (I should probably combine this code with the
        #                         roulette wheel code in selection.py)
        wheel = opProbs
        runningsum = 0.0
        for i,v in zip(range(len(wheel)), wheel):
            runningsum += v
            wheel[i] = runningsum
        wheel[-1] = 1.0   # just to be safe
        self.wheel = wheel


    def reinitialize(self, population):
        """
        Reinitialize this operator, and the wrapped ones.
        """
        for o in self.wrappedOps:
            o.reinitialize(population)
        GeneticOperator.reinitialize(self, population)


    def applyAndCache(self):
        """
        Assembles a list containing the appropriate number of individuals, and
        then calls apply.  The results are then placed in the cache.
        """
        self.pickActiveOperator()
        GeneticOperator.applyAndCache(self)


    def pickActiveOperator(self):
        """
        Randomly chooses a new wrapped operator to call.
        """
        # Spin the roulette wheel to pick an operator
        r = random.random()
        self.opInd = [r <= i for i in self.wheel].index(True)
        op = self.wrappedOps[self.opInd]
        self.parentsNeeded = op.parentsNeeded


    def apply(self, individuals):
        """
        @param individuals: A list of the individuals to pass on to the
                            appropriate wrapped operator.
        @return: The same list of individuals.
        ASSUMES: pickActiveOperator() was called just prior to this function
                 being called.  This sets self.opInd, which indicates which
                 operator in self.wrappedOps[] to use.  This will be done
                 automatically if this operator is used within a pipeline.

                 It has to be done this way so that self.parentsNeeded is set
                 properly before applyAndCache is called.  This will assure
                 that the number of individuals passed into apply is correct.

        The opInd stuff I descriped above is a bit of a kludge, but I
        wanted to keep the signature for apply() the same in all wrappedOps.
        It does break one of my design principles for operators though --
        this operator will not quite function properly if just apply() is
        called on it.  If one uses this outside of a pipeline,
        pickActiveOperator() should be called before apply() is.

        On the other hand, this approach will make it easy to make subclasses
        for the purpose of creating probes.
        """
        individuals = self.wrappedOps[self.opInd].apply(individuals)
        return individuals



#############################################################################
#
# class DummyOperator
#
#############################################################################
class DummyOperator(GeneticOperator):
    """
    This operator does nothing.  In general it is not necessary since
    operators can just be left out of the pipeline, but it might be useful as
    as placeholder on occasion.
    """
    parentsNeeded = 1

    def __init__(self, provider, parentsNeeded = 1):
        GeneticOperator.__init__(self, provider)
        self.parentsNeeded = parentsNeeded

    def apply(self, individuals):
        """
        @param individuals: A list of the individuals to be cloned.

                            Despite the fact that parentsNeeded = 1, this
                            function can handle any number of individuals here.
        @return: A list of the resulting clones.
        """
        for individual in individuals:
            individual.genomeUnchanged()
            individual.previous.append([individual])
        return individuals



#############################################################################
#
# class CloneOperator
#
# I added the clone operator so that the other genetic operators could
# operate directly on individuals coming down the pipeline instead of
# making copies every time.  As I recall, object creation is one of the
# more time consuming operations in Python.
#
# I'm willing to reconsider this decision though.  The clone operator may
# just confuse the issue for newcomers.  Perhaps we should require that
# generic operators make their own copies, or at least the selection
# operators.
#
#############################################################################
class CloneOperator(GeneticOperator):
    """
    Make a copy of the individual coming down the pipeline, and pass it
    along instead of the original.  This can be thought of as the moment
    when a parent gives birth to a child.  Every pipeline should have a
    CloneOperator (usually directly after the SelectionOperator) or else
    the individuals in the parent population will be modified.  This could
    cause a great deal of problems, especially if a parent is selected more
    than once.
    """
    parentsNeeded = 1

    def apply(self, individuals):
        """
        @param individuals: A list of the individuals to be cloned.

                            Despite the fact that parentsNeeded = 1, this
                            function can handle any number of individuals here.
        @return: A list of the resulting clones.
        """
        children = []
        for ind in individuals:
            clone = ind.clone()
            clone.popIndex = ind.popIndex  # Inherit. Selection will change.
            clone.genomeUnchanged()
#            clone.resetFitness()
            clone.parents = [ind]
            clone.previous = []
            children.append(clone)

            # Sever links to great-...-great-grandparents.
            # We don't want to keep the entire family tree in memory.
            for p in ind.parents:
                del p.parents[:]
                del p.previous[:]
#XXX                for gp in p.parents:
#XXX                    for ggp in gp.parents:
#XXX                        for gggp in ggp.parents:
#XXX                            gggp.parents = []
        return children


#############################################################################
#
# class RandomSearchOperator
#
#############################################################################
class RandomSearchOperator(GeneticOperator):
    """
    Creates a new random genome.
    """
    parentsNeeded = 1

    def apply(self, individuals):
        """
        @param individuals: A list of the individuals to be fixed.
        @return: The same list of individuals, with their genomes randomized.
        """
        for i in individuals:
            i.genome = i.decoder.randomGenome()
            i.previous.append([i])
        return individuals


#############################################################################
#
# class FixupOperator
#
#############################################################################
class FixupOperator(GeneticOperator):
    """
    Fixes a genome to make sure all its values are legal.
    """
    parentsNeeded = 1

    def apply(self, individuals):
        """
        @param individuals: A list of the individuals to be fixed.
        @return: The same list of individuals, with their genomes fixed.
        """
        for i in individuals:
            i.genome = i.decoder.fixupGenome(i.genome)
            # It would be nice if we could tell if the genome has changed.
        return individuals


#############################################################################
#
# class MutationOperator
#
#############################################################################
class MutationOperator(GeneticOperator):
    """
    Base class for mutation operators.
    """
    parentsNeeded = 1
    numMut = 0  # Track how many changes were made to a genome.

    def __init__(self, provider, pMutate, linear = True):
        """
        @param provider: The operator which immediately precedes this one in
                         the pipeline.
        @param pMutate: The probability that a gene will be mutated.
                        A value between 0 and 1.
        @param linear: Indicates whether the genome is flat (True) or
                       contains sublists (False).  Default = True.
        """
        GeneticOperator.__init__(self, provider)
        self.pMutate = pMutate  # Probability that a gene will be mutated
        self.linear = linear  # Descend into sub-lists if we encounter them


    def nextMutation(self):
        """
        Calculates how many genes to skip until the next mutation.
        """
        if self.pMutate == 1.0:
            return 0
        else:
            return int((math.log(random.random()) /
                        math.log(1.0 - self.pMutate)))


    def mutateGene(self, gene):
        """
        Mutates a single gene.  Subclasses should redefine this.

        @param gene: The gene to mutate.
        @return: A modified or new version of C{gene}.
        """
        raise NotImplementedError  # Subclasses should redefine this
        return gene


    def linearMutate(self, genome):
        """
        Mutate a linear or flat genome.  This allows it to be more optimized
        than heirarchicalMutate.

        @param genome: The genome to be mutated.
        @return: The mutated version of the genome.
        """
        if self.pMutate == 0.0:
            return

        locus = self.nextMutation();
        while locus < len(genome):
            self.numMut += 1
            #genome[locus] = self.mutateGene(genome[locus])
            genome = setSeqVal(genome, locus, self.mutateGene(genome[locus]))
            locus += self.nextMutation() + 1

        return genome


    def hierarchicalMutate(self, genome, relLocus):
        """
        Mutate a genome which contains sub-lists using recursion.

        @param genome: The genome to be mutated.
        @param relLocus: The number of nodes left to traverse before we reach
                         the gene we want to mutate.
        @return: The mutated version of the genome.
        """
        if self.pMutate == 0.0:
            return

        for i in range(len(genome)):
            g = genome[i]
            if isinstance(g, list) or isinstance(g, tuple) \
               or (isinstance(g, str) and len(g) > 1):
                newPiece, relLocus = self.hierarchicalMutate(genome[i],relLocus)
                genome[i] = newPiece
                genome = setSeqVal(genome, i, newPiece)
            elif relLocus == 0:
                self.numMut += 1
                #genome[i] = self.mutateGene(genome[i])
                genome = setSeqVal(genome, i, self.mutateGene(genome[i]))
                relLocus = self.nextMutation()
            else:
                relLocus -= 1
        return genome, relLocus


    def mutateIndividual(self, individual):
        """
        Mutate an individual's genome.  Calls either linearMutate or
        heirarchicalMutate.

        @param individual: The individual to be mutated.
        """
        #print("before =", individual.genome)
        self.numMut = 0
        if self.linear:
            genome = self.linearMutate(individual.genome)
        else:
            genome, dummy = self.hierarchicalMutate(individual.genome,
                                                    self.nextMutation())
        individual.genome = genome
        individual.previous.append([individual])
        individual.numMut = self.numMut
        if self.numMut == 0:
            individual.genomeUnchanged()
        #print("after =", individual.genome)


    def apply(self, individuals):
        """
        @param individuals: A list of the individuals to be fixed.
        @return: The same list of individuals, with their genomes fixed.
        """
        for ind in individuals:
            self.mutateIndividual(ind)
        return individuals


#############################################################################
#
# class UniformMutation
#
#############################################################################
class UniformMutation(MutationOperator):
    """
    Mutate an discrete valued gene by uniformly selecting a new allele from
    a list of possible values.
    @note: Every mutation is guaranteed to make a change.  In other words,
           a gene that is mutated will not end up with the same value it
           started with.
    """
    def __init__(self, provider, pMutate, alleles, linear = True):
        """
        @param provider: The operator which immediately precedes this one in
                         the pipeline.
        @param pMutate: The probability that a gene will be mutated.
                        A value between 0 and 1.
        @param alleles: A list of legal values for genes.
        @param linear: Indicates whether the genome is flat (True) or
                       contains sublists (False).  Default = True.
        """
        MutationOperator.__init__(self, provider, pMutate, linear)
        self.alleles = alleles

    def mutateGene(self, gene):
        """
        Mutate a single gene by selecting (without replacement) a new
        value from C{alleles}.

        @param gene: The gene to mutate.
        @return: A modified or new version of C{gene}.
        """
        woReplacement = self.alleles[:]
        woReplacement.remove(gene)
        g = random.choice(woReplacement)
        return g


#############################################################################
#
# class BitFlipMutation
#
#############################################################################
class BitFlipMutation(UniformMutation):
    "Mutate a binary gene by flipping 0's to 1's and vice versa."

    def __init__(self, provider, pMutate, alleles = ['0', '1'], linear = True):
        """
        @param provider: The operator which immediately precedes this one in
                         the pipeline.
        @param pMutate: The probability that a gene will be mutated.
                        A value between 0 and 1.
        @param alleles: A list of legal values for genes.
                        Default = ['0', '1'].
        @param linear: Indicates whether the genome is flat (True) or
                       contains sublists (False).  Default = True.
        """
        UniformMutation.__init__(self, provider, pMutate, alleles, linear)


#############################################################################
#
# class BoundedMutationOperator
#
#############################################################################
class BoundedMutationOperator(MutationOperator):
    """
    Base class for mutation operators.
    """
    parentsNeeded = 1

    def __init__(self, provider, pMutate, linear = True, bounds = None):
        """
        @param provider: The operator which immediately precedes this one in
                         the pipeline.
        @param pMutate: The probability that a gene will be mutated.
                        A value between 0 and 1.
        @param linear: Indicates whether the genome is flat (True) or
                       contains sublists (False).  Default = True.
        @param bounds: Defines bounds for the gene values.  Only works when
                       linear == True.  Takes the following form:
                       [(low1,high1), (low2, high2), ..., (lowN, highN)] where
                       N is the number of genes in the genome.  Any of these
                       values can be None, indicating there is no bound.
        """
        MutationOperator.__init__(self, provider, pMutate, linear=linear)
        self.bounds = bounds


    def withinBounds(self, newVal, locus):
        """
        Returns true if a genome value falls within the appropriate bounds.
        """
        if not self.bounds:
            return True

        # It wouldn't make sense to use Genes with bounds, but just in case...
        if isinstance(newVal, LEAP.Gene):
            v = newVal.data
        else:
            v = newVal

        if v > self.bounds[locus][1]:
            return False
        elif v > self.bounds[locus][0]:
            return True
        else:
            return False


    def linearMutate(self, genome):
        """
        Mutate a linear or flat genome.  This allows it to be more optimized
        than heirarchicalMutate.

        @param genome: The genome to be mutated.
        @return: The mutated version of the genome.
        """
        if self.pMutate == 0.0:
            return genome

        if self.bounds:
            assert(len(self.bounds) == len(genome))

        locus = self.nextMutation();
        while locus < len(genome):
            self.numMut += 1
            newVal = self.mutateGene(genome[locus])
            while not self.withinBounds(newVal, locus):  # Timeout?
                newVal = self.mutateGene(genome[locus])
            genome = setSeqVal(genome, locus, newVal)
            locus += self.nextMutation() + 1

        return genome



#############################################################################
#
# class GaussianMutation
#
#############################################################################
class GaussianMutation(BoundedMutationOperator):
    """
    Mutation operator for real valued genes which adds a value drawn from a
    gaussian distribution.

    @warning: This operator does nothing to enforce bounds on gene values.
              If this is important to you, you should either use this in
              conjunction with the L{FixupOperator}, or use a decoder which
              enforces the bounds, like L{BoundedRealDecoder}.

    """
    def __init__(self, provider, sigma, pMutate = 1.0, linear = True, \
                 bounds = None):
        """
        @param provider: The operator which immediately precedes this one in
                         the pipeline.
        @param sigma: The standard deviation of the Gaussian distribution.
        @param pMutate: The probability that a gene will be mutated.
                        A value between 0 and 1.  Default = 1.0.
        @param linear: Indicates whether the genome is flat (True) or
                       contains sublists (False).  Default = True.
        """
        BoundedMutationOperator.__init__(self, provider, pMutate, linear, \
                                         bounds)
        self.sigma = sigma

    def mutateGene(self, gene):
        """
        Mutate a single gene by adding a random value drawn from a Gaussian
        distribution.

        @param gene: The gene to mutate.
        @type gene: float or L{Gene}
        @return: A modified or new version of C{gene}.
        """
        if isinstance(gene, LEAP.Gene):
            gene.data += random.gauss(0.0, self.sigma)
        else:
            gene += random.gauss(0.0, self.sigma)
        return gene


#############################################################################
#
# class ExponentialMutation
#
#############################################################################
class ExponentialMutation(BoundedMutationOperator):
    """
    Mutation operator for real valued genes which adds a value drawn from an
    exponential distribution.  The idea was to create a real valued mutation
    operator that was more similar to bitflip mutation when applied to binary
    encoded real values.

    The mutation value is calculated as
        m = (+/-)epsilon * 2**x,
    where x is chosen randomly in the range [minExp, maxExp).

    @warning: This operator does nothing to enforce bounds on gene values.
              If this is important to you, you should either use this in
              conjunction with the L{FixupOperator}, or use a decoder which
              enforces the bounds, like L{BoundedRealDecoder}.

    """
    def __init__(self, provider, minExp, maxExp, epsilon = 1.0, pMutate = 1.0,
                 linear = True, bounds = None):
        """
        @param provider: The operator which immediately precedes this one in
                         the pipeline.
        @param minExp: The minimum value of the exponent.
        @param maxExp: The maximum value of the exponent.
        @param epsilon: The smallest possible change in a gene value.
        @param pMutate: The probability that a gene will be mutated.
                        A value between 0 and 1.  Default = 1.0.
        @param linear: Indicates whether the genome is flat (True) or
                       contains sublists (False).  Default = True.
        """
        BoundedMutationOperator.__init__(self, provider, pMutate, linear, \
                                         bounds)
        self.minExp = minExp
        self.maxExp = maxExp
        self.epsilon = epsilon

    def mutateGene(self, gene):
        """
        Mutate a single gene by adding a random value drawn from an
        exponential distribution (sort of).

        @param gene: The gene to mutate.
        @type gene: float or L{Gene}
        @return: A modified or new version of C{gene}.
        """
        sign = random.choice([1.0, -1.0])
        m = sign * self.epsilon * 2**random.uniform(self.minExp, self.maxExp) 

        if isinstance(gene, LEAP.Gene):
            gene.data += m
        else:
            gene += m
        return gene


#############################################################################
#
# class CreepMutation
#
# I recently learned that some might consider Gaussian mutation to be creep
# mutation.  I may need to reconsider the name.
#
#############################################################################
class CreepMutation(BoundedMutationOperator):
    """
    Mutation operator for real valued genes which adds or subtracts a
    constant delta.
    """
    def __init__(self, provider, pMutate, delta, linear = True, bounds = None):
        """
        @param provider: The operator which immediately precedes this one in
                         the pipeline.
        @param pMutate: The probability that a gene will be mutated.
                        A value between 0 and 1.
        @param delta: The fixed value which is added to or subtracted from a
                      gene.
        @param linear: Indicates whether the genome is flat (True) or
                       contains sublists (False).  Default = True.
        """
        BoundedMutationOperator.__init__(self, provider, pMutate, linear, \
                                         bounds)
        self.delta = delta

    def mutateGene(self, gene):
        """
        Mutate a single gene by adding or subtracting delta from it.

        @param gene: The gene to mutate.
        @type gene: float or L{Gene}
        @return: A modified or new version of C{gene}.
        """
        thisDelta = self.delta
        if (random.random() < 0.5):
            thisDelta = -self.delta

        if isinstance(gene, LEAP.Gene):
            gene.data += thisDelta
        else:
            gene += thisDelta

        return gene


#############################################################################
#
# class ProxyMutation
#
#############################################################################
class ProxyMutation(MutationOperator):
    """
    Mutation operator for genomes containing mixed data types.  The genome
    must be made up of genes of type L{Gene}, but the C{Gene.data} field can
    contain any type of data.  Each gene must also have a mutation operator
    associated with it.  When a given gene is mutated, its associated mutation
    operator will be called.
    """
    def __init__(self, provider):
        """
        @param provider: The operator which immediately precedes this one in
                         the pipeline.
        @param linear: Indicates whether the genome is flat (True) or
                       contains sublists (False).  Default = True.
        """
        # Set the mutation rate to 1.0.  Let the proxied mutation operators
        # decide how often they want to do mutations.
        # Set linear to True.  Things are not currently set up to handle
        # proxies with hierarchical genomes.
        MutationOperator.__init__(self, provider, pMutate=1.0, linear=True)


    def mutateGene(self, gene):
        """
        Mutate a single gene by calling that genes C{mutate()} function.

        @param gene: The gene to mutate.
        @type gene: L{Gene}
        @return: A modified or new version of C{gene}.
        """
        return gene.mutate()


    def linearMutate(self, genome):
        """
        Mutate a linear or flat genome.  This allows it to be more optimized
        than heirarchicalMutate.

        @param genome: The genome to be mutated.
        @return: The mutated version of the genome.
        """
        if self.pMutate == 0.0:
            return genome

        for i in range(len(genome)):
            l = genome[i].mutationOperator.nextMutation() 
            if l == 0:
                self.numMut += 1
                #genome[i] = self.mutateGene(genome[i])
                genome = setSeqVal(genome, i, self.mutateGene(genome[i]))

        return genome



#############################################################################
#
# class AdaptiveMutation
#
#############################################################################
class AdaptiveMutation(MutationOperator):
    """
    Gaussian mutation operator for real valued genes.  It uses the ES style
    adaptive mutation mechanism described in T. Back and H.-P. Schwefel,
    "An Overview of Evolutionary Algorithms for Parameter Optimization",
    Evolutionary Computation, 1(1):1-23, The MIT Press, 1993.  A standard
    deviation (sigma) is associated with each gene, and these are adapted from
    one generation to the next.
    """
    tau = None
    tauPrime = None
    tauPrimeTerm = None
    sigmaBounds = None

    def __init__(self, provider, sigmaBounds, linear = True):
        """
        @param provider: The operator that immediately precedes this one in
                         the pipeline.
        @param sigmaBounds: A tuple containing the minimum and maximum sigma
                            values allowed.  Sigma values which go beyond
                            these bounds will be clipped.
        @param linear: Indicates whether the genome is flat (True) or
                       contains sublists (False).  Default = True.
        """
        MutationOperator.__init__(self, provider, 1.0, linear)
        self.sigmaBounds = sigmaBounds

    def reinitialize(self, population):
        "Update adaptive mutation parameters each generation."
        if self.tau == None:
            aParent = population[0]
            self.tau = 1/math.sqrt(2 * math.sqrt(len(aParent.genome)))
            self.tauPrime = 1/math.sqrt(2 * len(aParent.genome))
        MutationOperator.reinitialize(self, population)

    def mutateIndividual(self, individual):
        "Update adaptive mutation parameters for each individual."
        self.tauPrimeTerm = self.tauPrime * random.gauss(0,1)
        MutationOperator.mutateIndividual(self, individual)

    def mutateGene(self, gene):
        """
        Mutate a single gene by adding a random value drawn from a Gaussian
        distribution.

        @param gene: The gene to mutate.
        @type gene: L{AdaptiveRealGene}
        @return: A modified or new version of C{gene}.
        """
        assert(isinstance(gene, LEAP.AdaptiveRealGene))

        gene.sigma *= math.exp(self.tauPrimeTerm + self.tau * random.gauss(0,1))
        gene.sigma = min(max(gene.sigma, self.sigmaBounds[0]),
                         self.sigmaBounds[1])
        gene.data += gene.sigma * random.gauss(0,1)
        return gene



#############################################################################
#
# class CrossoverOperator
#
#############################################################################
class CrossoverOperator(GeneticOperator):
    """
    Base class for crossover operators.
    """
    parentsNeeded = 2

    def __init__(self, provider, pCross, numChildren = 2):
        """
        @param provider: The operator which immediately precedes this one in
                         the pipeline.
        @param pCross: The probability that a pair of individuals will be
                       recombined.  A value between 0 and 1.
        @param numChildren: The number of children that will be produced
                            (1 or 2).  Default is 2.
        """
        GeneticOperator.__init__(self, provider, pApply = pCross)
        assert (numChildren == 1 or numChildren == 2)
        self.numChildren = numChildren

    def apply(self, individuals):
        """
        @param individuals: A list of the individuals to be recombined.
        @return: The same list of individuals, with their genomes recombined.
        """
        children = []   # Assume individuals are already cloned
        # Loop through all the individuals, even if there are more than 2
        while len(individuals) >= 2:
            child1 = individuals.pop(0)
            child2 = individuals.pop(0)

            # Modify parent lists.
            parents1 = child1.parents + child2.parents
            child2.parents += child1.parents
            child1.parents = parents1

            # Modify previous lists.
            child1.previous.append([child1, child2])
            child2.previous.append([child2, child1])

            (child1, child2) = self.recombine(child1, child2)
            #if random.random() < self.pCross:   # Perform the crossover?
            #    (child1, child2) = self.recombine(child1, child2)
            #else:
            #    child1.genomeUnchanged()
            #    child2.genomeUnchanged()

            if self.numChildren == 1:
                children += [random.choice([child1, child2])]
            else:
                children += [child1, child2]  # whether modified or not

        children += individuals   # add remaining odd parent (if exists)
        return children


    def recombine(self, child1, child2):
        raise NotImplementedError



#############################################################################
#
# class NPointCrossover
#
#############################################################################
class NPointCrossover(CrossoverOperator):
    """
    Standard crossover operator with a parameterized number of cross points.
    """
    parentsNeeded = 2

    def __init__(self, provider, pCross, numPoints = 1, numChildren = 2,
                 allowXoverAt0 = False):
        """
        @param provider: The operator which immediately precedes this one in
                         the pipeline.
        @param pCross: The probability that a pair of individuals will be
                       recombined.  A value between 0 and 1.
        @param numPoints: The number of crossover points.  Default = 1.
        @param numChildren: The number of children that will be produced
                            (1 or 2).  Default is 2.
        @param allowXoverAt0: This allows crossover points to be selected
                              right at the beginning of the string.  For
                              1 point crossover this would mean that no
                              crossover occurs.  When using more crossover
                              points though, this can have a significant
                              effect, reducing genetic drift.  See De Jong,
                              Evolutionary Computation: A Unified Approach,
                              p. 145.
        """
        CrossoverOperator.__init__(self, provider, pCross, numChildren)
        self.numPoints = numPoints
        self.allowXoverAt0 = allowXoverAt0


    def pickCrossoverPoints(self, numPoints, genomeSize):
        """
        Randomly choose (without replacement) crossover points.
        """
        if self.allowXoverAt0:
            pp = list(range(0,genomeSize))  # See De Jong, EC, pg 145
        else:
            pp = list(range(1,genomeSize))  # Possible xover points
        xpts = [pp.pop(random.randrange(len(pp))) for i in range(numPoints)]
        xpts.sort()
        xpts = [0] + xpts + [genomeSize]  # Add start and end
        return xpts


    def recombine(self, child1, child2):
        """
        @param child1: The first individual to be recombined.
        @param child2: The second individual to be recombined.
        @return: (child1, child2)
        """
        # Check for errors.
        assert(len(child1.genome) == len(child2.genome))
        assert(len(child1.genome) >= self.numPoints + 1-int(self.allowXoverAt0))

        children = [child1, child2]
        genome1 = child1.genome[0:0]  # empty sequence - maintain type
        genome2 = child2.genome[0:0]
        src1, src2 = 0, 1

        # Pick crossover points
        xpts = self.pickCrossoverPoints(self.numPoints, len(child1.genome))

        # Perform the crossover
        for i in range(len(xpts)-1):  # swap odd segments
            genome1 += children[src1].genome[xpts[i]:xpts[i+1]]
            genome2 += children[src2].genome[xpts[i]:xpts[i+1]]
            src1, src2 = src2, src1

        child1.genome = genome1
        child2.genome = genome2

        # Gather some statistics
        child1.numSwaps = child2.numSwaps = sum([xpts[i+1] - xpts[i] for i \
                                                 in range(1,len(xpts)-1,2)])

        return (child1, child2)



#############################################################################
#
# class UniformCrossover
#
#############################################################################
class UniformCrossover(CrossoverOperator):
    """
    Genes are exchanged between individuals depending on a coin flip.
    """

    def __init__(self, provider, pCross, pSwap = 0.5, numChildren = 2):
        """
        @param provider: The operator which immediately precedes this one in
                         the pipeline.
        @param pCross: The probability that a pair of individuals will be
                       recombined.  A value between 0 and 1.
        @param pSwap: The probability that two corresponding genes will be
                      swapped.  Default = 0.5.
        @param numChildren: The number of children that will be produced
                            (1 or 2).  Default is 2.
        """
        CrossoverOperator.__init__(self, provider, pCross, numChildren)
        self.pSwap = pSwap


    def recombine(self, child1, child2):
        """
        @param child1: The first individual to be recombined.
        @param child2: The second individual to be recombined.
        @return: (child1, child2)
        """

        # Check for errors.
        assert(len(child1.genome) == len(child2.genome))

        #print("recombining")
        # Perform the crossover
        numSwaps = 0
        for i in range(len(child1.genome)):
            if random.random() < self.pSwap:
                temp = child1.genome[i]
                #child1.genome[i] = child2.genome[i]
                child1.genome = setSeqVal(child1.genome, i, child2.genome[i])
                #child2.genome[i] = temp
                child2.genome = setSeqVal(child2.genome, i, temp)
                numSwaps += 1

        # Gather some statistics
        child1.numSwaps = child2.numSwaps = numSwaps

        return (child1, child2)



#############################################################################
#
# unit_test
#
#############################################################################
#from selection import *
#from survival import *
def unit_test():

#    class MyIndividual:
#        def __init__(self, genome):
#            self.genome = genome
#
#    m = MutationOperator(None, 1.0)
#    print("nextMutation =", m.nextMutation())
#
#    x = NPointCrossover(None, 1.0, numPoints = 1, numChildren = 2)
#    a = MyIndividual([1,2])
#    b = MyIndividual([3,4])
#    c,d = x.recombine(a,b)
#    assert(c.genome == [1,4] and d.genome == [3,2])

    a = Operator()
    b = PipelineOperator(None)
    c = GeneticOperator(None)
    e = FixupOperator(None)
    f = MutationOperator(None, 0)

    length = 15

    print("CloneOperator")
    clone = CloneOperator(None)
    parent = LEAP.Individual(None, list(range(length)))
    child = clone.apply([parent])[0]
    print(parent.genome)
    print(child.genome)
    assert(parent.genome == child.genome)
    assert(parent.genome is not child.genome)
    print()

    print("UniformMutation")
    alleles = ['a', 'b', 'c']
    unifMut = UniformMutation(None, 1.0, alleles)
    genome = [random.choice(alleles) for i in list(range(length))]
    parent = LEAP.Individual(None, genome)
    child = clone.apply([parent])[0]
    child = unifMut.apply([child])[0]
    print(parent.genome)
    print(child.genome)
    assert(parent.genome != child.genome)
    assert(all([g in alleles for g in child.genome]))
    print()

    print("BitFlipMutation")
    alleles = ['0', '1']
    bitflip = BitFlipMutation(None, 1.0)
    genome = "".join([random.choice(alleles) for i in list(range(length))])
    parent = LEAP.Individual(None, genome)
    child = clone.apply([parent])[0]
    child = bitflip.apply([child])[0]
    print(parent.genome)
    print(child.genome)
    assert(parent.genome != child.genome)
    assert(all([g in alleles for g in child.genome]))
    print()

    print("GaussianMutation")
    numGenes = 3
    gauss = GaussianMutation(None, 1.0)
    parent = LEAP.Individual(None, [0.0]*numGenes)
    child = clone.apply([parent])[0]
    child = gauss.apply([child])[0]
    print(parent.genome)
    print(child.genome)
    assert(parent.genome != child.genome)
    print()

    print("GaussianMutation w/ bounds")
    numGenes = 3
    lower = 0.0
    upper = 0.5
    gauss = GaussianMutation(None, 1.0, bounds=[(lower, upper)]*numGenes)
    parent = LEAP.Individual(None, [0.0]*numGenes)
    child = clone.apply([parent])[0]
    child = gauss.apply([child])[0]
    print(parent.genome)
    print(child.genome)
    assert(parent.genome != child.genome)
    for g in child.genome:
        assert(g >= lower and g <= upper)
    print()

    creep = CreepMutation(None, 0, 0)
    proxy = ProxyMutation(None)
    adapt = AdaptiveMutation(None, [0])

    import string

    uppercase = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    lowercase = uppercase.lower()
    length = 2
    print("NPointCrossover")
    p2xover = NPointCrossover(None, 1.0, numPoints=2, allowXoverAt0=True)
    mom = LEAP.Individual(None, uppercase[0:length])
    dad = LEAP.Individual(None, lowercase[0:length])
    sis, bro = clone.apply([mom, dad])
    sis, bro = p2xover.apply([sis,bro])
    print(mom.genome + "  " + dad.genome)
    print(sis.genome + "  " + bro.genome)
    assert(sis.genome == "aB")
    assert(bro.genome == "Ab")

    length = 10
    p2xover = NPointCrossover(None, 1.0, numPoints=2, allowXoverAt0=False)
    mom = LEAP.Individual(None, uppercase[0:length])
    dad = LEAP.Individual(None, lowercase[0:length])
    sis, bro = clone.apply([mom, dad])
    sis, bro = p2xover.apply([sis,bro])
    print(mom.genome + "  " + dad.genome)
    print(sis.genome + "  " + bro.genome)
    assert(sorted(mom.genome + dad.genome) == sorted(sis.genome + bro.genome))
    assert(mom.genome != sis.genome and dad.genome != bro.genome)
    print()

    print("UniformCrossover")
    length = 10
    unifXover = UniformCrossover(None, pCross=1.0, pSwap=0.5)
    mom = LEAP.Individual(None, uppercase[0:length])
    dad = LEAP.Individual(None, lowercase[0:length])
    sis, bro = clone.apply([mom, dad])
    sis, bro = unifXover.apply([sis,bro])
    print(mom.genome + "  " + dad.genome)
    print(sis.genome + "  " + bro.genome)
    assert(sorted(mom.genome + dad.genome) == sorted(sis.genome + bro.genome))
    assert(mom.genome != sis.genome and dad.genome != bro.genome)
    print()

    # Should test the pipeline too

    print("Passed?")


if __name__ == '__main__':
    unit_test()

