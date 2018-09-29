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
import math

#from individual import *
#from selection import *
#from halt import *
import LEAP



#############################################################################
#
# printPopulation
#
#############################################################################
def printPopulation(population, generation = None):
    for index, individual in enumerate(population):
        if generation != None:
            print("Gen:", generation, end='')
        print("Ind:", index, "", individual)



#############################################################################
#
# ea
#
##############################################################################
class GenerationalEA:
    """
    A generational Evolutionary Algorithm.

    decoder - The class which can encode/decode the genome.  Also contains a
              reference to the problem.
    pipeline - The operator pipeline (e.g. selection, crossover, mutation)
    popSize - Size of the population
    initPipeline - An optional operator pipeline which will only be applied
                   to the first generation.  None by default.
    indClass - Optional class which defines an individual.
               LEAP.Individual is used by default.
    """
    def __init__(self, decoder, pipeline, popSize, \
                 initPipeline=LEAP.DeterministicSelection(), \
                 indClass=LEAP.Individual,
                 initPopsize=None, halt=None):
        self.decoder = decoder
        self.pipeline = pipeline
        self.popSize = popSize
        self.initPipeline = initPipeline
        self.indClass = indClass
        self.halt = halt

        if initPopsize == None:
            self.initPopsize = popSize
        else:
            self.initPopsize = initPopsize

        #self.startup()


    def startup(self):
        """
        Resets some internal variables and creates an initial population.

        Probably best if this is only called from within the class.
        """
        self.population = []
        self.bestOfGen = None
        self.bestSoFar = None
        self.generation = 0

        # Create initial population
        randomPopulation = []
        for i in range(self.initPopsize):
            ind = self.indClass(self.decoder)
            ind.evaluate()
            randomPopulation.append(ind)

        # Apply init pipeline
        self.initPipeline.reinitialize(randomPopulation)
        for i in range(self.popSize):
            self.population.append(self.initPipeline.pull())

        self.calcStats()
        self.printStats()


    def run(self, maxGeneration = None):
        # I would remove maxGeneration in favor of halt, but I've decided to
        # keep it for backward compatability

        # Make sure either halt or maxGeneration is set (xor)
        assert(bool(maxGeneration == None) != bool(self.halt == None))

        if self.halt == None:
            self.halt = LEAP.HaltAfterGeneration(maxGeneration)

        # Reset the EA, just in case it was already run before
        self.startup()
        self.halt.reset()

        while not self.halt.shouldHaltNow(self.population):
            self.step()
            self.calcStats()
            self.printStats()

        #print(mystr(self.bestSoFar.genome))
        #print(self.decoder.decodeGenome(self.bestSoFar.genome))
        return self.bestSoFar


    def step(self):
        children = []
        self.generation += 1
        self.pipeline.reinitialize(self.population)
        for i in range(self.popSize):
            child = self.pipeline.pull()
            child.evaluate()
            children.append(child)

        self.population = children


    def calcStats(self):
        # I use copy instead of clone so that the parent list is retained.
        #    Do I even use the parents list anymore?
        #self.bestOfGen = copy.deepcopy(LEAP.fittest(self.population))
        self.bestOfGen = LEAP.fittest(self.population).clone()
        self.bestSoFar = LEAP.fittest(self.bestSoFar, self.bestOfGen)


    def printStats(self):
        #pass
        #printPopulation(self.population, self.generation)

        print("Gen:", self.generation, " Ind: BOG ", self.bestOfGen)
        print("Gen:", self.generation, " Ind: BSF ", self.bestSoFar)



#############################################################################
#
# Main
#
#############################################################################
if __name__ == '__main__':
    # Parameters
    popSize = 100 # 100
    maxGeneration = 200

    # Setup the problem
    #function = LEAP.schwefelFunction
    #bounds = LEAP.schwefelBounds
    #maximize = LEAP.schwefelMaximize
    #numVars = len(bounds)

#    function = LEAP.sphereFunction
#    #bounds = LEAP.sphereBounds
#    #bounds = [(0.0,1.0)] * 6
#    bounds = [LEAP.sphereBounds[0]] * 10
#    maximize = LEAP.sphereMaximize
#    numVars = len(bounds)

    numVars = 10
    function = LEAP.valleyFunctor([1.0] + [0.0] * (numVars-1))
    bounds = [LEAP.valley2Bounds[0]] * numVars
    maximize = LEAP.valleyMaximize

    problem = LEAP.FunctionOptimization(function, maximize = maximize)

    # ...for binary genes
#    bitsPerReal = 16
#    genomeSize = bitsPerReal * numVars
#    decoder = LEAP.BinaryRealDecoder(problem, [bitsPerReal] * numVars, bounds)

    # ...for float genes
    #decoder = LEAP.FloatDecoder(problem, bounds, bounds)

    # ...for adaptive real genes
    sigmaBounds = (0.0, bounds[0][1] - bounds[0][0])
    initSigmas = [(bounds[0][1] - bounds[0][0]) / math.sqrt(numVars)] * numVars
    decoder = LEAP.AdaptiveRealDecoder(problem, bounds, bounds, initSigmas)


    #pipe2 = LEAP.UniformSelection()
    #pipe2 = LEAP.CloneOperator(pipe2)

    # Setup the reproduction pipeline
    #pipeline = LEAP.TournamentSelection(2)
    #pipeline = LEAP.ProportionalSelection()
    pipeline = LEAP.TruncationSelection(popSize//2)
    #pipeline = LEAP.RankSelection()
    #pipeline = LEAP.DeterministicSelection()
    pipeline = LEAP.CloneOperator(pipeline)
    #pipeline = LEAP.Shuffle2PointCrossover(pipeline, 0.8, 2)
    #pipeline = LEAP.NPointCrossover(pipeline, 0.8, 2)
    #pipeline = LEAP.NPointCrossover([pipeline, pipe2], 0.8, 2)
    #pipeline = LEAP.UniformCrossover(pipeline, 0.8, 0.5)
    #pipeline = LEAP.ProxyMutation(pipeline)
    #pipeline = LEAP.BitFlipMutation(pipeline, 1.0/genomeSize)
    #pipeline = LEAP.UniformMutation(pipeline, 1.0/genomeSize, alleles)
    pipeline = LEAP.AdaptiveMutation(pipeline, sigmaBounds)
    #pipeline = LEAP.GaussianMutation(pipeline, sigma = 0.04, pMutate = 1.0)
    #pipeline = LEAP.FixupOperator(pipeline)
    #pipeline = LEAP.ElitismSurvival(pipeline, 2)
    #pipeline = LEAP.MuPlusLambdaSurvival(pipeline, popSize, popSize*2)
    #pipeline = LEAP.MuCommaLambdaSurvival(pipeline, popSize, popSize*10)
    #pipeline = LEAP.MuCommaLambdaSurvival(pipeline, popSize, popSize,
    #                                      LEAP.RankSelection())

    ea = LEAP.GenerationalEA(decoder, pipeline, popSize)
    #ea = LEAP.GenerationalEA(decoder, pipeline, popSize, \
    #                         indClass=Price.PriceIndividual)
    ea.run(maxGeneration)

#    import profile
#    profile.run('ea(params)', 'eaprof')
#
#    import pstats
#    p = pstats.Stats('eaprof')
#    p.sort_stats('time').print_stats(20)


