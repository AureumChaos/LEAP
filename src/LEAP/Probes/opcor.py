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
import copy

from price import *
from LEAP.operators import WrapperOperator

from math import *
from numpy import *


#############################################################################
#
# OperatorCorrelationProbe
#
#############################################################################
class OperatorCorrelationProbe(WrapperOperator):
    """
    Measures the "parent-offspring" correlation of a single operator.
    In this context parents are the individuals coming down the
    pipeline, and the offspring are the result of the wrapped operator.
    """
    def __init__(self, provider, wrappedOps, opProbs, measureFunc,
                 tag="opcor", measureFile = None):
        WrapperOperator.__init__(self, provider, wrappedOps, opProbs)

        self.measureFile = measureFile
        self.firstCall = True

        self.measureFunc = measureFunc
        self.tag = tag
        self.zero = measureFunc()
        self.setGeneration(0)


    def setGeneration(self, newGen):
        self.generation = newGen

        # Create a list of empty lists.  This approach may look like overkill,
        # but I have to make sure each element is unique.  
        self.preMeasures = [[] for i in range(len(self.wrappedOps))]
        self.postMeasures = [[] for i in range(len(self.wrappedOps))]

        # Doing things this way is a bit of a hack
        self.preLengths = [[] for i in range(len(self.wrappedOps))]
        self.postLengths = [[] for i in range(len(self.wrappedOps))]


    def reinitialize(self, population):
        WrapperOperator.reinitialize(self, population)

        # On the first call to reinitilize, not data is available
        if self.preMeasures != [[]] * len(self.wrappedOps):
            ratios = [len(opPre) for opPre in self.preMeasures]
            total = sum(ratios)
            ratios = [float(i) / total for i in ratios]

            # We will get errors if pre and post contain any empty lists
            # For now I will just place a singe zero in any empty list
            for i in range(len(self.preMeasures)):
                if self.preMeasures[i] == []:
                    self.preMeasures[i] = self.postMeasures[i] = [0.0]

            deltas = [[postval-preval for preval,postval in zip(opPre,opPost)]\
                 for opPre,opPost in zip(self.preMeasures, self.postMeasures)]

            # population means
            prebar = [E(opPre) for opPre in self.preMeasures]
            postbar = [E(opPost) for opPost in self.postMeasures]
            deltabar = [E(opDelta) for opDelta in deltas]

            # population variances
            varpre = [Var(opPre) for opPre in self.preMeasures]
            varpost = [Var(opPost) for opPost in self.postMeasures]
            vardelta = [Var(opDelta) for opDelta in deltas]

            # heretability and correlation
            covs = [Cov(opPre, opPost) for opPre,opPost in
                    zip(self.preMeasures, self.postMeasures)]

            # Since these calculations can potentially produce non-results
            # because of divide-by-zero errors, I'll comment them out and
            # leave them to be performed later, using R for example.

            #heretabilities = covs[:]
            #for i in range(len(covs)):  # avoid divide by zeros
            #    if varpost[i] == 0:
            #        heretabilities[i] = None
            #    else:
            #        heretabilities[i] = covs[i] / varpre[i]

            #correlations = covs[:]
            #for i in range(len(covs)):  # avoid divide by zeros
            #    if varpost[i] == 0 or varpre[i] == 0:
            #        correlations[i] = None
            #    else:
            #        correlations[i] = covs[i] / math.sqrt(varpre[i]*varpost[i])

            # print the results
            print("Gen:", self.generation, end=' ')
            print("Tag:", self.tag, end=' ')
            for i in range(len(self.wrappedOps)):
                print("r"+str(i+1)+":", ratios[i], end=' ')

            for i in range(len(self.wrappedOps)):
                print("DeltaBar"+str(i+1)+":", deltabar[i], end=' ')

            for i in range(len(self.wrappedOps)):
                print("VarPre"+str(i+1)+":", varpre[i], end=' ')

            for i in range(len(self.wrappedOps)):
                print("VarPost"+str(i+1)+":", varpost[i], end=' ')

            for i in range(len(self.wrappedOps)):
                print("VarDelta"+str(i+1)+":", vardelta[i], end=' ')

            for i in range(len(self.wrappedOps)):
                print("Cov"+str(i+1)+":", covs[i], end=' ')
            print()

            # print the results
            if self.measureFile:
                if self.firstCall:
                    s = '"Gen", "OpNum", "Tag", "PreMeasure", ' + \
                        '"PostMeasure", "PreLen", "PostLen"\n'
                    self.measureFile.write(s)
                    self.firstCall = False
                for opInd in range(len(self.wrappedOps)):
                    for pre,post,prelen,postlen in zip(self.preMeasures[opInd],\
                                                     self.postMeasures[opInd],
                                                     self.preLengths[opInd],
                                                     self.postLengths[opInd]):
                        s = "%d, %d, %s, %g, %g, %d, %d\n" % (self.generation,\
                                   opInd, self.tag, pre, post, prelen, postlen)
                        self.measureFile.write(s)
                        
            self.generation += 1

        # Empty out the measurements to make room for next generation.
        self.preMeasures = [[] for i in range(len(self.wrappedOps))]
        self.postMeasures = [[] for i in range(len(self.wrappedOps))]

        self.preLengths = [[] for i in range(len(self.wrappedOps))]
        self.postLengths = [[] for i in range(len(self.wrappedOps))]


    def apply(self, individuals):
        # I've written other operators so that they can deal with getting more
        # individuals than expected.  I cannot allow that here because I need
        # to know the relationships between parents and offspring.
        assert(len(individuals) == self.parentsNeeded)

        # Measure before op
        preMeasures = [self.measureFunc(i) for i in individuals]
        preLengths = [len(i.genome) for i in individuals]

        # Perform op
        individuals = WrapperOperator.apply(self, individuals)

        # Measure after op
        postMeasures = [self.measureFunc(i) for i in individuals]
        postLengths = [len(i.genome) for i in individuals]

        # Make sure all parents and offspring are associated with each other.
        # This may mean putting duplicates measurements in the list.
        for pre,prelen in zip(preMeasures, preLengths):
            for post,postlen in zip(postMeasures, postLengths):
                self.preMeasures[self.opInd].append(pre)
                self.preLengths[self.opInd].append(prelen)
                self.postMeasures[self.opInd].append(post)
                self.postLengths[self.opInd].append(postlen)

        return individuals
        


#############################################################################
#
# unit test
#
#############################################################################

if __name__ == '__main__':
    from LEAP.problem import schwefelFunction
    from LEAP.problem import schwefelBounds
    from LEAP.problem import schwefelMaximize
    from LEAP.problem import FunctionOptimization
    from LEAP.encoding import *
    from LEAP.selection import *
    from LEAP.operators import *
    from LEAP.survival import *
    from LEAP.ea import *

    # Some parameters
    #popSize = 500
    #maxGeneration = 200
    popSize = 10
    maxGeneration = 10

    # Setup the problem
    function = schwefelFunction
    bounds = schwefelBounds
    maximize = schwefelMaximize
    numVars = len(bounds)

    problem = FunctionOptimization(function, maximize = maximize)

    # ...for binary genes
    #bitsPerReal = 16
    #genomeSize = bitsPerReal * numVars
    #encoding = BinaryRealEncoding(problem, [bitsPerReal] * numVars, bounds)

    # ...for float genes
    #encoding = FloatEncoding(problem, bounds, bounds)

    # ...for adaptive real genes
    sigmaBounds = (0.0, bounds[0][1] - bounds[0][0])
    initSigmas = [(bounds[0][1] - bounds[0][0]) / sqrt(numVars)] * numVars
    encoding = AdaptiveRealEncoding(problem, bounds, bounds, initSigmas)

    measure = fitnessMeasure
    measureFile = open("unit_test.measure", "w")
    #measure = rankMeasure

    # Setup the reproduction pipeline
    pipeline = TournamentSelection(2)
    #pipeline = ProportionalSelection()
    #pipeline = RankSelection()
    #pipeline = DeterministicSelection()
#    pipeline = PriceCalcOperator(pipeline, zero=measure(), tag="SurvivalSel")
    pipeline = CloneOperator(pipeline)
#    pipeline = PriceInitOperator(pipeline)
    #pipeline = Shuffle2PointCrossover(pipeline, 0.8, 2)
    #pipeline = NPointCrossover(pipeline, 0.8, 2)
    op1 = NPointCrossover(None, 1.0, 2)
    op2 = DummyOperator(None, 2)
    pipeline = OperatorCorrelationProbe(pipeline, [op1, op2], [0.75, 0.25],
                                        measure, tag="crossover")
    #pipeline = UniformCrossover(pipeline, 0.8, 0.5)
    #pipeline = price1 = PriceMeasureOperator(pipeline, measure)
    #pipeline = ProxyMutation(pipeline)
    #pipeline = BitFlipMutation(pipeline, 1.0/genomeSize)
    #pipeline = UniformMutation(pipeline, 1.0/genomeSize, alleles)
    #pipeline = AdaptiveMutation(pipeline, sigmaBounds)
    #op1 = GaussianMutation(pipeline, sigma = 1.0, pMutate = 1.0)
    op1 = AdaptiveMutation(pipeline, sigmaBounds)
    pipeline = OperatorCorrelationProbe(pipeline, [op1], [1.0], measure,
                                        tag="mutation", measureFile=measureFile)
    #pipeline = GaussianMutation(pipeline, sigma = 1.0,
    #                                 pMutate = 1.0)
    #pipeline = FixupOperator(pipeline)
#    pipeline = price2 = PriceMeasureOperator(pipeline, measure)
    #pipeline = ElitismSurvival(pipeline, 2)
    #pipeline = PriceRankOperator(pipeline, popSize)
    #pipeline = PriceCalcOperator(pipeline, zero=measure(),
    #                             tag="ParentSel")
#    pipeline = PriceCalcOperator(pipeline, zero=measure(), tag="ParentSel")
    #pipeline = VarianceCalcOperator(pipeline, zero=measure()) 
    #pipeline = MuCommaLambdaSurvival(pipeline, popSize, popSize*10)
    

#    initPipe = DeterministicSelection()
#    initPipe = PriceInitOperator(initPipe)
#    initPipe = PriceMeasureOperator(initPipe, measure)

    ea = GenerationalEA(encoding, pipeline, popSize)
    ea.run(maxGeneration)

#    import profile
#    profile.run('ea(params)', 'eaprof')
#
#    import pstats
#    p = pstats.Stats('eaprof')
#    p.sort_stats('time').print_stats(20)


