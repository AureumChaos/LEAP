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

import sys
import random
import string

import LEAP
#import scipy.stats

from math import *
from numpy import *
#from Numeric import *
#from numarray import *

import functools
import copy

#############################################################################
#
# E
#
# Returns the expected value (mean) of all the items in the list X
#
#############################################################################
def E(X):
    return (sum(X) + 0.0) / len(X)


#############################################################################
#
# Cov
#
# Returns the covariance of the elements of lists X and Y.
#
#############################################################################
def Cov(X, Y):
    Xbar = E(X)
    Ybar = E(Y)
    return E( [(xi - Xbar) * (yi - Ybar) for (xi,yi) in zip(X,Y)] )


#############################################################################
#
# Var
#
# Returns the variance of the elements of list X.
#
#############################################################################
def Var(X):
    return Cov(X, X)


#############################################################################
#
# mystr
#
#   Used for doing customized string conversions, especially on lists
# and arrays.
#
#############################################################################
def mystr(x):
    if isinstance(x, list):
        s = "["
        for i in x:
            s += mystr(i) + ","
        if s[-1] == ",":
            s = s[:-1]
        s += "]"
        return s
    elif type(x) == type(array([])):    # Couldn't get isinstance to work
        s = string.replace(repr(x), ' ', '')
        s = string.replace(s, 'array(', '')
        s = string.replace(s, ')', '')
        return s
        #return str(x[1])
    elif isinstance(x, float):
        s = "%.12f" % x
        return s[0:11]
    else:
        return str(x)


#############################################################################
#
# class RandomSearchOperator
#
#############################################################################
class RandomSearchOperator(LEAP.CloneOperator):
    """
    Generates a completely random child.
    """
    parentsNeeded = 1

    def apply(self, individuals):
        """
        @param individuals: A list of "parents".

                            Despite the fact that parentsNeeded = 1, this
                            function can handle any number of individuals here.
        @return: A list of the resulting individuals.
        """
        for ind in individuals:
            ind.genome = ind.decoder.randomGenome()

        return individuals


#############################################################################
#
# printPopulation
#
#############################################################################
def printPopulation(population, generation = None):
    if len(population) == 0:
        return
    qBar = sum([i.q[-1] for i in population]) / len(population)
    for i, ind in enumerate(population):
        if generation != None:
            print("Gen:", generation, end=' ')
        print("Ind:", i, "", ind, end=' ')

        #qdiff = population[i].z * (population[i].q[-1] - qBar)
        #print(" z(q-qBar): %16.10f" % qdiff, end=' ')

        print(" deltaQx:", end=' ')
        for parent in ind.parents:
            deltaq = ind.q[0] - parent.q[-1]
            #print("%16.10f" % deltaq,  end=' ')
            print(mystr(deltaq),  end=' ')
        print()



def repPrev(previous):
    if isinstance(previous, PriceIndividual):
        return previous.popIndex
    elif isinstance(previous, list):
        return [repPrev(i) for i in previous]


#############################################################################
#
# class PriceIndividual
#
#############################################################################
class PriceIndividual(LEAP.Individual):
    """
    Keep some statistics that are related to Price's equation
    """
    numSwaps = None      # number of genes swapped during crossover
    numMut = None        # number of genes mutated

    def __repr__(self):
        rep = ""
        rep = rep + "genome: " + mystr(self.genome)
        rep = rep + "  parents: " + mystr([p.popIndex for p in self.parents])
        rep = rep + "  numSwaps: " + str(self.numSwaps)
        rep = rep + "  numMut: " + str(self.numMut)
        rep = rep + "  fitness: " + str(self.getFitness()) #"%16.10f" % self.getFitness()
        rep = rep + "  q: " + mystr(self.q)
        rep = rep + "  z: " + str(self.z)
        rep = rep + "  previous: " + mystr(repPrev(self.previous))
        return rep

    def clone(self):
        #clone = LEAP.Individual.clone(self)
        if isinstance(self.genome, str):
            clone = PriceIndividual(self.decoder, self.genome)
        elif isinstance(self.genome[0], float) \
             or isinstance(self.genome[0], int):
            clone = PriceIndividual(self.decoder, self.genome[:])
        else:  # Gene class
            clone = PriceIndividual(self.decoder,
                                    [copy.copy(i) for i in self.genome])
        clone.rawFitness = self.rawFitness
        clone.fitness = self.fitness
        clone.z = 0
        clone.q = self.q
        return clone


#############################################################################
#
# class LexPriceIndividual
#
#############################################################################
class LexPriceIndividual(LEAP.Individual):

    def cmp(self, other):
        """
        Returns 1 if self is better than other
                0 if self = other
               -1 if self is worse than other
        Better than means '>' if maximizing or '<' if minimizing.

        Implements lexicographic parsimony pressure.
        """
        if self == None or other == None:
            return cmp(self, other)

        result = self.decoder.cmpFitness(self.getFitness(), other.getFitness())
        if result == 0:
            return -cmp(size(self.genome), size(other.genome))
        else:
            return result




#############################################################################
#
# PriceInitOperator
#
#############################################################################
class PriceInitOperator(LEAP.PipelineOperator):
    """
    Initialize individuals so that information can be stored in them later.
    """
    parentsNeeded = 1

    def apply(self, children):
        for child in children:
            child.q = []
            child.z = 0
        return children


#############################################################################
#
# PriceMeasureOperator
#
#############################################################################
class PriceMeasureOperator(LEAP.PipelineOperator):
    """
    Gather statistics for Price's equation and store them in the individuals.
    """
    parentsNeeded = 1
    measureFunction = None

    def __init__(self, provider, measureFunction):
        LEAP.PipelineOperator.__init__(self, provider)
        self.measureFunction = measureFunction

    def apply(self, children):
        for child in children:
            child.q.append(self.measureFunction(child))
        return children


#############################################################################
#
# CollectPopulationOperator
#
#############################################################################
class CollectPopulationOperator(LEAP.PipelineOperator):
    """
    Gather up a population as it comes down the pipeline.
    Useful after a selection operator.
    """
    parentsNeeded = 1

    def __init__(self, provider):
        LEAP.PipelineOperator.__init__(self, provider)
        self.collectedPop = []

    def reinitialize(self, population):
        LEAP.PipelineOperator.reinitialize(self, population)
        self.collectedPop = []

    def apply(self, children):
        self.collectedPop += children
        return children

    def getPopulation(self):
        return self.collectedPop


#############################################################################
#
# AverageMeasureOperator
#
#############################################################################
class AverageMeasureOperator(CollectPopulationOperator):
    """
    Calculates the average measurement for a population.
    This is how you might do things if you weren't using Price's equation.
    """
    parentsNeeded = 1

    def __init__(self, provider, tag):
        LEAP.PipelineOperator.__init__(self, provider)
        if tag:
            self.tag = tag
        else:
            self.tag = "Qbar"
        self.generation = 0
        self.qBar = 0.0
        self.samples = []
        self.indices = []
        AverageMeasureOperator.prevqBar = 0.0

    def reinitialize(self, population):
        LEAP.PipelineOperator.reinitialize(self, population)

        if len(self.samples) > 0:
            self.qBar = E(self.samples)
        print("Gen:", self.generation, end=' ')
        print("%10s" % self.tag + ": %16.10f" % self.qBar, end=' ')
        print(" len(samples):", len(self.samples), end=' ')
        print(" delta: %16.10f" % (self.qBar - self.prevqBar), end=' ')
#        print(" samples:", self.samples)
        print(" indices:", self.indices)

        AverageMeasureOperator.prevqBar = self.qBar
        self.generation += 1
        self.qBar = 0.0
        self.numSamples = 0
        self.samples = []
        self.indices = []

    def apply(self, children):
        for child in children:
            self.numSamples += 1
            #self.qBar = runningAvg(self.qBar, child.q[-1], self.numSamples)
            self.samples += [child.q[-1]]
            self.indices += [child.popIndex]
        return children



#############################################################################
#
# PriceCalcOperator
#
#############################################################################
class PriceCalcOperator(CollectPopulationOperator):
    """
    Calculates the terms in Price's equation.
    """
    def __init__(self, provider, zero=0, tag=None):
        CollectPopulationOperator.__init__(self, provider)
        self.P1 = []
        self.P2 = []
        self.zero = zero
        self.tag = tag
        self.generation = 1

    def reinitialize(self, population):
        self.P1 = self.P2
        self.P2 = self.getPopulation()[:]
        #if not self.P2:
        #    self.P2 = population  # This may not be a good idea
        self.calcPrice()
        CollectPopulationOperator.reinitialize(self, population)
        self.generation += 1


    def calcPrice(self):
        #printPopulation(self.P1, self.generation-1)
        if self.P1 and self.P2:
            print("Gen:", self.generation - 1, end=' ')
            if self.tag:
                print("Tag:", self.tag, end=' ')
            calcPriceFunc(self.P1, self.P2, self.zero)



#############################################################################
#
# VarianceCalcOperator
#
#############################################################################
class VarianceCalcOperator(CollectPopulationOperator):
    """
    Calculates correlation (and other things) between parent (directly prior
    to operator) and offspring (directly after operator).
    """
    def __init__(self, provider, zero=0):
        CollectPopulationOperator.__init__(self, provider)
        self.P1 = []
        self.P2 = []
        self.zero = zero
        self.generation = 1

    def reinitialize(self, population):
        self.P1 = self.P2
        self.P2 = self.getPopulation()[:]
        #if not self.P2:
        #    self.P2 = population  # This may not be a good idea
        self.calcVariance()
        CollectPopulationOperator.reinitialize(self, population)
        self.generation += 1


    def calcVariance(self):
        #printPopulation(self.P1, self.generation-1)
        if self.P1 and self.P2:
            print("Gen:", self.generation - 1, end=' ')
            calcVarianceFunc(self.P1, self.P2, self.zero)


#############################################################################
#
# PriceRankOperator
#
#############################################################################
class PriceRankOperator(LEAP.BaseMuLambdaSurvival):
    """
    When using rankMeasure() as you measurement function, this should be
    placed directly before PriceCalcOperator in the pipeline.
    NOTE: Check to see if this really works!!
          My recollection was that it's not finished yet.
    """
    def __init__(self, provider, popSize):
        LEAP.BaseMuLambdaSurvival.__init__(self, provider, popSize, popSize,
                LEAP.DeterministicSelection())

    def combinePopulations(self, parents, children):
        """
        All we really want here is access to both the parent and child
        populatations while in the pipeline.  This is why I've used the
        BaseMuLambdaSurvival operator.  We will process what we need and
        then just pass the child population along.
        """
        allRanks = [i.q[-1] for i in parents]
        for j in range(len(parents[0].q)):
            allRanks += [i.q[j] for i in children]

        allRanks.sort()

        for i in range(len(allRanks)):
            allRanks[i][1] = i+1
            if i !=0 and allRanks[i][0] == allRanks[i-1][0]:
                allRanks[i][1] = allRanks[i-1][1]

        return children


#############################################################################
#
# Price Measurement Functions
#
#############################################################################
def fitnessMeasure(ind = None):
    if ind == None:
        return 0.0   # define a zero measurement
    ind.fitness = None    # Make sure a new evaluation is performed(?)
    return ind.evaluate()


def locationMeasure(ind = None):
    numVars = 30  # This is a hack!  I shouldn't have to define this here!
    if ind == None:
        return zeros(numVars)   # define a zero measurement
    return array(ind.decoder.decodeGenome(ind.genome))


def rankMeasure(ind = None):
    """
    The goal here is to rank all individuals in the parent population, child
    population and any intermediates populations based on their fitnesses.
    Of course we cannot do this until all the individuals have been measured,
    which means we're going to have to do some post-processing.  To facilitate
    this I return two values in a list.  The first is the fitness, and the
    second will be the rank.  The rank will have to be calculated later using
    the PriceRankOperator.
    """
    if ind == None:
        return array([0.0, 0])   # define a zero measurement
    return array([ind.evaluate(), 0])



#############################################################################
#
# Some useful utility functions
#
#############################################################################
def add(a,b):
    return a+b

def norm(a):
    return sqrt(sum(a**2))

def runningAvg(avg, sample, numSamples):
    return (avg * (numSamples - 1) + sample) / numSamples


#############################################################################
#
# unique
#
# This could be easily implemented using sets now.
#
#############################################################################
def unique(l):
    """
    Returns a copy of the list l, with all duplicates removed.
    """
    l2 = l[:]
    for i in range(len(l2)-1, -1, -1):
        if l2.count(l2[i]) > 1:
            del l2[i]
    return l2


#############################################################################
#
# getAncestors
#
#############################################################################
def getAncestors(ind, generation=1):
    """
    Returns a list of all the ancestors from a specific generation.
    For example, if generation = 1 all the parents will be returned.
    If generation = 2, all the grandparents will be returned.
    If generation = 5, all the great-great-great grandparents will be
    returned.
    """
    ancestors = [ind]
    for gen in range(generation):
        inds = ancestors
        ancestors = []
        for i in inds:
            ancestors += i.parents

    return ancestors


#############################################################################
#
# calcPriceFunc
#
# Price's equation:
# 
# qPrimeBar - qBar = Cov(z, q) / zBar + sum(z_i * delta_q_i) / (N * zBar)
#
##############################################################################
def calcPriceFunc(P1, P2, zero, genStep=1):
    """
    P1 - Parent population
    P2 - Child population
    zero - The zero value for q.  Depends on the measurement function.
    genStep - The number of generations between P1 and P2.
    """
    P1 = unique(P1)

    # Set z (number of children) to zero in each parent
    for parent in P1:
        parent.z = 0

    opMatchList = []  # contains N * zBar instances of parent/child tuples,
                      #   one for each unique path through the operators from
                      #   parent to child.
    
    # Compute z in each parent
    for child in P2:
        for parent in getAncestors(child, genStep):
            parent.z += 1
            opMatchList.append( (parent, child) )

    # Compute some of the variables in Price's equation
    N = float(len(P1))
    zBar = E([i.z for i in P1])
    qBar = E([i.q[-1] for i in P1])       # Avg of parent final measurements
    qPrimeBar = E([i.q[-1] for i in P2])  # Avg of child final measurements

    # For debugging
    #print()
    #print("N =", N)
    #print("sum(z) =", sum([i.z for i in P1]))
    #print("zBar =", zBar)
    #print("qBar =", qBar)
    #print("qPrimeBar =", qPrimeBar)

    # Calculate the selection term
    T1 = Cov([i[0].z for i in opMatchList], [i[0].q[-1] for i in opMatchList])

    T1measure = sum([i.z * (i.q[-1] - qBar) / (N * zBar) for i in P1])
    S1 = sqrt(sum([i.z * (i.q[-1] - qBar)**2 / (N * zBar) for i in P1]))
    T1min = min([i.q[-1] - qBar for i in P1])
    T1max = max([i.q[-1] - qBar for i in P1])

    # Calculate the operator terms (T2 - Tn) and standard deviations (S2 - Sn)
    k = len(P2[0].q)   # number of measurements made (i.e. operators in pipe)
    T_ops = [None] * k
    S_ops = [None] * k
    T_opsMins = [None] * k
    T_opsMaxs = [None] * k
    op_qBars = [None] * k
    op_qPrimeBars = [None] * k
    op_qStdevs = [None] * k
    op_qPrimeStdevs = [None] * k
    op_Covs = [None] * k   # Cov(q,q') but different from Price's equation in
                           # that all q's are specified instead of being
                           # averaged.

    for j in range(k):
        op_qs = []        # List of qs of all parents passed to op j
        op_qPrimes = []   # List of qPrimes of all children passed out of op j
        deltaList = []

# Will this work?
#        if j == 0:
#            op_qs = [i[0].q[-1] for i in opMatchList]
#        else:
#            op_qs = [i[1].q[j-1] for i in opMatchList]
#        op_qPrimes = [i[1].q[j] for i in opMatchList]
#        deltaList = [qPrime - q for (qPrime, q) in zip(op_qPrimes, op_qs)]

        for child in P2:
            # Calculate all the deltaQ_ij values.  This does not work
            # quite the same way Price's equation works.  I collect every
            # change in fitness caused by operator j into something called
            # deltaList.  With Price's equation these are averaged together
            # for each parent.
            # XXX The previous array seems kind of kludgy.
            #     I'd like to improve the design.
            numPrev = len(child.previous[j])
            for i in range(len(child.parents)):
                if j == 0:
                    q_ij = child.parents[i%numPrev].q[-1]
                    qPrime_ij = child.q[0]
                else:
                    q_ij = child.previous[j][i%numPrev].q[j-1]
                    qPrime_ij = child.q[j]
                delta = qPrime_ij - q_ij
                deltaList.append(delta)
                op_qs.append(q_ij)
                op_qPrimes.append(qPrime_ij)

        # Try to catch some obvious errors.
        if len(deltaList) != int(N * zBar):
            print()
            print("len(deltaList) =", len(deltaList))
            print("N =", N)
            print("zBar =", zBar)
            print("N * zBar =", N * zBar)
            raise(RuntimeError, "Mismatch in number of operations")

        T_opsMins[j] = min(deltaList)
        T_opsMaxs[j] = max(deltaList)

        T_ops[j] = functools.reduce(add,deltaList) / float(len(deltaList))
        deltaSqList = [x * x for x in deltaList]
        operatorVariance = functools.reduce(add,deltaSqList) / (N * zBar) - T_ops[j] ** 2

        # Try to make sure we're not taking sqrt() of a negative number
        if operatorVariance > zero:  # XXX This may not work for some measures,
                                     #     vectors in particular.
            S_ops[j] = sqrt(operatorVariance)
        else:
            S_ops[j] = zero

        # Here I calculate Cov(q,q'), but q' doesn't mean quite what it means
        # in Price's equation.  Instead of being the average of a parents z
        # chilren, we consider every childs q value separately.  The
        # covariance just wouldn't make sense if we used averages the way
        # Price does because sometimes a child can have 2 parents (crossover)
        # or just one parent (clone).
        NzBar = int(N * zBar)
        #print("NzBar:", NzBar, end=' ')
        op_qBars[j] = sum(op_qs) / NzBar
        op_qPrimeBars[j] = sum(op_qPrimes) / NzBar

        op_qStdevs[j] = sqrt(sum([(op_qs[i] - op_qBars[j]) ** 2 \
                                  for i in range(NzBar)]) / NzBar)
        op_qPrimeStdevs[j] = sqrt(sum([(op_qPrimes[i] - op_qPrimeBars[j])**2 \
                                  for i in range(NzBar)]) / NzBar)

        op_Covs[j] = sum([(op_qs[i] - op_qBars[j]) * 
                            (op_qPrimes[i] - op_qPrimeBars[j])
                        for i in range(NzBar)]) / NzBar

    DeltaQmeasure = qPrimeBar - qBar
    DeltaQcalc = T1 + functools.reduce(add,T_ops)  # As I recall, I used reduce() instead
                                         # of sum() here because sum() wasn't
                                         # working well when q was a vector.


    # Print out the results
    # XXX I don't like the fact that I'm printing information from here.
    #     I would prefer to print the results from within
    #     PriceEA.printStats().
    print("best:", LEAP.fittest(P1).fitness, end=' ')
    print("qPrimeBar:", mystr(qPrimeBar), end=' ')
    print("DeltaQmeasure:", mystr(DeltaQmeasure), end=' ')
    print("DeltaQcalc:", mystr(DeltaQcalc), end=' ')
    print("T1measure:", mystr(T1measure), end=' ')
    print("T1:", mystr(T1), end=' ')
    for j in range(k):
        print("T" + str(j+2) + ":", mystr(T_ops[j]), end=' ')
    print("S1:", mystr(S1), end=' ')
    for j in range(k):
        print("S" + str(j+2) + ":", mystr(S_ops[j]), end=' ')

    print("T1min:", mystr(T1min), end=' ')
    print("T1max:", mystr(T1max), end=' ')
    for j in range(k):
        print("T" + str(j+2) + "min:", mystr(T_opsMins[j]), end=' ')
        print("T" + str(j+2) + "max:", mystr(T_opsMaxs[j]), end=' ')

    #print("deltaList:", mystr(deltaList)

    # Temporary?
    # Print out the "op_" statistics gathered about the operators.
    # I'm looking for a relationship between parent and child, hopefully
    # covariance with maybe a couple of these other factors thrown in.
    for j in range(k):
        print("op_qBar" + str(j+2) + ":", mystr(op_qBars[j]), end=' ')
        print("op_qStdev" + str(j+2) + ":", mystr(op_qStdevs[j]), end=' ')
        print("op_qPrimeBar" + str(j+2) + ":", mystr(op_qPrimeBars[j]), end=' ')
        print("op_qPrimeStdev" + str(j+2) + ":", mystr(op_qPrimeStdevs[j]),
              end=' ')
        print("op_Covs" + str(j+2) + ":", mystr(op_Covs[j]), end=' ')

    print()


#############################################################################
#
# calcPriceFunc2
#
# This is probably old and could be removed.
#
##############################################################################
def calcPriceFunc2(P1, P2, zero, genStep=1):
    """
    P1 - Parent population
    P2 - Child population
    zero - The zero value for q.  Depends on the measurement function.
    genStep - The number of generations between P1 and P2.
    """
    P1 = unique(P1)

    # Set z to zero in each parent
    for parent in P1:
        parent.z = 0

    # Compute z in each parent
    for child in P2:
        for parent in getAncestors(child, genStep):
            parent.z += 1

    # Compute some of the variables in Price's equation
    N = float(len(P1))
    zBar = E([i.z for i in P1])
    qBar = E([i.q[-1] for i in P1])
    qPrimeBar = E([i.q[-1] for i in P2])

    # For debugging
    #print()
    #print("N =", N)
    #print("sum(z) =", sum([i.z for i in P1]))
    #print("zBar =", zBar)
    #print("qBar =", qBar)
    #print("qPrimeBar =", qPrimeBar)

    # Calculate the selection term
    covList = [(i.z - zBar) * (i.q[-1] - qBar) for i in P1]
    T1 = sum(covList) / (N * zBar)

    T1measure = sum([i.z * (i.q[-1] - qBar) / (N * zBar) for i in P1])
    S1 = sqrt(sum([i.z * (i.q[-1] - qBar)**2 / (N * zBar) for i in P1]))
    T1min = min([i.q[-1] - qBar for i in P1])
    T1max = max([i.q[-1] - qBar for i in P1])

    # Calculate the operator terms (T2 - Tn) and standard deviations (S2 - Sn)
    k = len(P2[0].q)   # number of measurements made (i.e. operators in pipe)
    T_ops = [None] * k
    S_ops = [None] * k
    T_opsMins = [None] * k
    T_opsMaxs = [None] * k
    op_qBars = [None] * k
    op_qPrimeBars = [None] * k
    op_qStdevs = [None] * k
    op_qPrimeStdevs = [None] * k
    op_Covs = [None] * k   # Cov(q,q') assuming all q's are specified instead
                           # of being averaged.
    for j in range(k):
        op_qs = []        # List of qs of all parents passed to op j
        op_qPrimes = []   # List of qPrimes of all children passed out of op j
        deltaList = []
        for child in P2:
            # Calculate all the deltaQ_ij values.  This does not work
            # quite the same way Price's equation works.  I collect every
            # change in fitness caused by operator j into something called
            # deltaList.  With Price's equation these are averaged together
            # for each parent.
            # XXX The previous array seems kind of kludgy.
            #     I'd like to improve the design.
            numPrev = len(child.previous[j])
            for i in range(len(child.parents)):
                if j == 0:
                    q_ij = child.parents[i%numPrev].q[-1]
                    qPrime_ij = child.q[0]
                else:
                    q_ij = child.previous[j][i%numPrev].q[j-1]
                    qPrime_ij = child.q[j]
                delta = qPrime_ij - q_ij
                deltaList.append(delta)
                op_qs.append(q_ij)
                op_qPrimes.append(qPrime_ij)

        # Try to catch some obvious errors.
        if len(deltaList) != int(N * zBar):
            print()
            print("len(deltaList) =", len(deltaList))
            print("N =", N)
            print("zBar =", zBar)
            print("N * zBar =", N * zBar)
            raise(RuntimeError, "Mismatch in number of operations")

        T_opsMins[j] = min(deltaList)
        T_opsMaxs[j] = max(deltaList)

        T_ops[j] = functools.reduce(add,deltaList) / float(len(deltaList))
        deltaSqList = [x * x for x in deltaList]
        operatorVariance = functools.reduce(add,deltaSqList) / (N * zBar) - T_ops[j] ** 2

        # Try to make sure we're not taking sqrt(-x)
        if operatorVariance > zero:  # XXX This may not work for some measures
            S_ops[j] = sqrt(operatorVariance)
        else:
            S_ops[j] = zero

        # Here I calculate Cov(q,q'), but q' doesn't mean quite what it means
        # in Price's equation.  Instead of being the average of a parents z
        # chilren, we consider every childs q value separately.  The
        # covariance just wouldn't make sense if we used averages the way
        # Price does.
        NzBar = int(N * zBar)
        #print("NzBar:", NzBar, end=' ')
        op_qBars[j] = sum(op_qs) / NzBar
        op_qPrimeBars[j] = sum(op_qPrimes) / NzBar

        op_qStdevs[j] = sqrt(sum([(op_qs[i] - op_qBars[j]) ** 2 \
                                  for i in range(NzBar)]) / NzBar)
        op_qPrimeStdevs[j] = sqrt(sum([(op_qPrimes[i] - op_qPrimeBars[j])**2 \
                                  for i in range(NzBar)]) / NzBar)

        op_Covs[j] = sum([(op_qs[i] - op_qBars[j]) * 
                            (op_qPrimes[i] - op_qPrimeBars[j])
                        for i in range(NzBar)]) / NzBar

    DeltaQmeasure = qPrimeBar - qBar
    DeltaQcalc = T1 + functools.reduce(add,T_ops)


    # Print out the results
    # XXX I don't like the fact that I'm printing information from here.
    #     I would prefer to print the results from within
    #     PriceEA.printStats().
    print("best:", LEAP.fittest(P1).fitness, end=' ')
    print("qPrimeBar:", mystr(qPrimeBar), end=' ')
    print("DeltaQmeasure:", mystr(DeltaQmeasure), end=' ')
    print("DeltaQcalc:", mystr(DeltaQcalc), end=' ')
    print("T1measure:", mystr(T1measure), end=' ')
    print("T1:", mystr(T1), end=' ')
    for j in range(k):
        print("T" + str(j+2) + ":", mystr(T_ops[j]), end=' ')
    print("S1:", mystr(S1), end=' ')
    for j in range(k):
        print("S" + str(j+2) + ":", mystr(S_ops[j]), end=' ')

    print("T1min:", mystr(T1min), end=' ')
    print("T1max:", mystr(T1max), end=' ')
    for j in range(k):
        print("T" + str(j+2) + "min:", mystr(T_opsMins[j]), end=' ')
        print("T" + str(j+2) + "max:", mystr(T_opsMaxs[j]), end=' ')

    #print("deltaList:", mystr(deltaList)

    # Temporary?
    # Print out the "op_" statistics gathered about the operators.
    # I'm looking for a relationship between parent and child, hopefully
    # covariance with maybe a couple of these other factors thrown in.
    for j in range(k):
        print("op_qBar" + str(j+2) + ":", mystr(op_qBars[j]), end=' ')
        print("op_qStdev" + str(j+2) + ":", mystr(op_qStdevs[j]), end=' ')
        print("op_qPrimeBar" + str(j+2) + ":", mystr(op_qPrimeBars[j]), end=' ')
        print("op_qPrimeStdev" + str(j+2) + ":", mystr(op_qPrimeStdevs[j]),
              end=' ')
        print("op_Covs" + str(j+2) + ":", mystr(op_Covs[j]), end=' ')

    print()


#############################################################################
#
# leastCommonMultiple
#
##############################################################################
def leastCommonMultiple(l):
    l.sort()
    multiple = functools.reduce(lambda x,y: x*y, l)
    # This is not done
    

#############################################################################
#
# calcVarianceFunc
#
##############################################################################
def calcVarianceFunc(P1, P2, zero, opStep=1):
    """
    """

    k = len(P2[0].q)   # number of measurements made (i.e. operators in pipe,
                       # not including selection)

    # Biologists typically assume that all individuals will have the same
    # number of parents.  In EC we cannot make this assumption.  To make sure
    # our calculations come out right, we need to add extra q values for
    # individuals that have fewer parents.  Its really about adding fractions
    # properly.
    numParents = unique([len(child.parents) for child in P2])
    lcm = functools.reduce(lambda x,y: x*y, numParents)  # Least common multiple

    # Create the q lists, one for each op including selection
    qs = [ [] ] * k+1
    for child in P2:
        # Take care of selection first
        for parent in child.parents:
            qs[0] += [parent.q[-1]] * lcm/len(child.parents)

        # Now the other operators
        for op in range(k):
            for prevList in child.previous:
                for prev in prevList:
                    qs[op+1] += [prev.q[op]] * lcm/len(child.parents)

    #------------- rewrite everything below

    # Calculate means
    print("qs[0], qPrimes[0]:", qs[0], qPrimes[0])
    N = len(qs)
    qBar = sum(qs) / N
    qPrimeBar = sum(qPrimes) / N
    deltaqBar = sum([qPrime - q for (q, qPrime) in zip(qs, qPrimes)]) / N

    # Calculate variances
    qVar = sum([q - qBar for q in qs]) / N
    qPrimeVar = sum([qPrime - qPrimeBar for qPrime in qPrimes]) / N
    deltaqVar = sum([qPrime - q - deltaqBar for (q, qPrime) in 
                          zip(qs, qPrimes) ]) / N

    # Calculate standard deviations
    # Try to avoid sqrt(-x)
    qSigma = zero
    if qVar > zero:  # XXX This may not work for some measures
        qSigma = sqrt(qVar)

    qPrimeSigma = zero
    if qPrimeVar > zero:  # XXX This may not work for some measures
        qPrimeSigma = sqrt(qPrimeVar)

    deltaqSigma = zero
    if deltaqVar > zero:  # XXX This may not work for some measures
        deltaqSigma = sqrt(deltaqVar)


    # Calculate parent-offspring correlation
    corr = sum([ (q - qBar) * (qPrime - qPrimeBar) for (q, qPrime) in
                          zip(qs, qPrimes) ]) / N

    # Print out the results
    # XXX I don't like the fact that I'm printing information from here.
    #     I would prefer to print the results from within PriceEA.printStats().
    print("op:", mystr(len(P2[0].q)), end=' ')

    print("qBar:", mystr(qBar), end=' ')
    print("qPrimeBar:", mystr(qPrimeBar), end=' ')
    print("deltaqBar:", mystr(deltaqBar), end=' ')

    print("qVar:", mystr(qVar), end=' ')
    print("qPrimeVar:", mystr(qPrimeVar), end=' ')
    print("deltaqVar:", mystr(deltaqVar), end=' ')

    print("deltaqVar/qVar:", mystr(deltaqVar/qVar), end=' ')
    print("corr(q,qPrime):", mystr(corr), end=' ')
    #print("qPrimeSigma/qSigma:", mystr(qPrimeSigma/qSigma), end=' ')

    print()



#############################################################################
#
# PriceEA
#
##############################################################################
class PriceEA(LEAP.GenerationalEA):
    """
    A generational EA which calculates the coefficients of Price's equation.

    Note, the initPipeline must be defined.  It should probably look
    something like this:
        initPipeline = DeterministicSelection()
        initPipeline = PriceInitOperator(initPipeline)
        initPipeline = PriceMeasureOperator(initPipeline, <measureFunc>)
    """
    def __init__(self, decoder, pipeline, popSize, zero, initPipeline, \
                 indClass=PriceIndividual, initPopsize=None):
        LEAP.GenerationalEA.__init__(self, decoder, pipeline, popSize, \
                                     initPipeline = initPipeline, \
                                     indClass = indClass,
                                     initPopsize = initPopsize)
        self.prevPop = None
        self.zero = zero


    def step(self):
        self.prevPop = self.population
        LEAP.GenerationalEA.step(self)


    def printStats(self):
        #if self.generation > 0:
        #    printPopulation(self.prevPop, self.generation-1)
        LEAP.GenerationalEA.printStats(self)



#############################################################################
#
# The main part of the script
#
#############################################################################

if __name__ == '__main__':
    # Some parameters
    #popSize = 500
    #maxGeneration = 200
    popSize = 10
    maxGeneration = 10

    # Setup the problem
    function = LEAP.schwefelFunction
    bounds = LEAP.schwefelBounds
    maximize = LEAP.schwefelMaximize
    numVars = len(bounds)

    problem = LEAP.FunctionOptimization(function, maximize = maximize)

    # ...for binary genes
    #bitsPerReal = 16
    #genomeSize = bitsPerReal * numVars
    #decoder = LEAP.BinaryRealDecoder(problem, [bitsPerReal] * numVars, bounds)

    # ...for float genes
    #decoder = LEAP.FloatDecoder(problem, bounds, bounds)

    # ...for adaptive real genes
    sigmaBounds = (0.0, bounds[0][1] - bounds[0][0])
    initSigmas = [(bounds[0][1] - bounds[0][0]) / sqrt(numVars)] * numVars
    decoder = LEAP.AdaptiveRealDecoder(problem, bounds, bounds, initSigmas)

    measure = fitnessMeasure
    #measure = rankMeasure

    # Setup the reproduction pipeline
    pipeline = LEAP.TournamentSelection(2)
    #pipeline = LEAP.ProportionalSelection()
    #pipeline = LEAP.RankSelection()
    #pipeline = LEAP.DeterministicSelection()
    pipeline = PriceCalcOperator(pipeline, zero=measure(), tag="SurvivalSel")
    pipeline = LEAP.CloneOperator(pipeline)
    pipeline = PriceInitOperator(pipeline)
    #pipeline = LEAP.Shuffle2PointCrossover(pipeline, 0.8, 2)
    pipeline = LEAP.NPointCrossover(pipeline, 0.8, 2)
    #pipeline = LEAP.UniformCrossover(pipeline, 0.8, 0.5)
    #pipeline = price1 = PriceMeasureOperator(pipeline, measure)
    #pipeline = LEAP.ProxyMutation(pipeline)
    #pipeline = LEAP.BitFlipMutation(pipeline, 1.0/genomeSize)
    #pipeline = LEAP.UniformMutation(pipeline, 1.0/genomeSize, alleles)
    pipeline = LEAP.AdaptiveMutation(pipeline, sigmaBounds)
    #pipeline = LEAP.GaussianMutation(pipeline, sigma = 1.0,
    #                                 pMutate = 1.0)
    #pipeline = LEAP.FixupOperator(pipeline)
    pipeline = price2 = PriceMeasureOperator(pipeline, measure)
    #pipeline = LEAP.ElitismSurvival(pipeline, 2)
    #pipeline = PriceRankOperator(pipeline, popSize)
    #pipeline = PriceCalcOperator(pipeline, zero=measure(),
    #                             tag="ParentSel")
    pipeline = PriceCalcOperator(pipeline, zero=measure(), tag="ParentSel")
    #pipeline = VarianceCalcOperator(pipeline, zero=measure()) 
    #pipeline = LEAP.MuCommaLambdaSurvival(pipeline, popSize, popSize*10)
    

    initPipe = LEAP.DeterministicSelection()
    initPipe = PriceInitOperator(initPipe)
    initPipe = PriceMeasureOperator(initPipe, measure)

    print("popSize =", popSize)
    ea = PriceEA(decoder, pipeline, popSize, measure(), initPipe)
    ea.run(maxGeneration)

#    import profile
#    profile.run('ea(params)', 'eaprof')
#
#    import pstats
#    p = pstats.Stats('eaprof')
#    p.sort_stats('time').print_stats(20)


