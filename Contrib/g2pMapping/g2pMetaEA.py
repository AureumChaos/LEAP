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
class g2pMetaEA(LEAP.GenerationalEA):
    def __init__(self, decoder, pipeline, popSize, validationProblem, \
                 initPipeline=LEAP.DeterministicSelection(), \
                 indClass=LEAP.Individual, \
                 initPopsize=None, halt=None, validationFrequency=20):
        LEAP.GenerationalEA.__init__(self, decoder, pipeline, popSize, \
                 initPipeline=initPipeline, indClass=indClass, \
                 initPopsize=initPopsize, halt=halt)
        self.validationProblem = validationProblem
        self.validationFrequency = validationFrequency
        self.validationBOG = None  # Best of all validations for a generation
        self.validationBSF = None
        self.bogValidation = None  # Validation of usual best-of-generation


    def validate(self):
        """
        Perform the validation step.  Evaluate the individuals on a different
        set of problems that have never been used for fitness calculations
        """
        bestIndex = None
        bestVal = None
        for i in range(len(self.population)):
            ind = self.population[i]
            # Evaluating the individual won't work, so I'll do this
            val = self.validationProblem.evaluate(self.decoder.decodeGenome(ind.genome))

            if bestVal == None or self.decoder.cmpFitness(bestVal, val) == -1:
                bestVal = val
                bestIndex = i

        return (bestVal, self.population[bestIndex])


    def startup(self):
        self.validationBOG = (None, [])  # Best of all validations for a generation
        self.validationBSF = (None, [])
        self.bogValidation = None  # Validation of usual best-of-generation
        LEAP.GenerationalEA.startup(self)


    def calcStats(self):
        LEAP.GenerationalEA.calcStats(self)

        if self.generation % self.validationFrequency == 0:
            tempValid = self.validate()
            print(tempValid)
            self.validationBOG = (tempValid[0], tempValid[1])
            if self.validationProblem.cmpFitness(self.validationBSF[0],
                                                 self.validationBOG[0]) == -1:
                self.validationBSF = self.validationBOG

        # Run validation on the bestOfGen individual
        self.bogValidation = self.validationProblem.evaluate(
                               self.decoder.decodeGenome(self.bestOfGen.genome))


    def printStats(self):
        LEAP.GenerationalEA.printStats(self)
        print("Gen:", self.generation, " Ind: BOGV  Val:", self.bogValidation)
        if self.generation % self.validationFrequency == 0:
            print("Gen:", self.generation, " Ind: VBOG ", self.validationBOG[1], \
                  "Val:", self.validationBOG[0])




#############################################################################
#
# Main
#
#############################################################################
if __name__ == '__main__':
    import LEAP.Contrib.g2pMapping as g2p
    import LEAP.Domains.Translated as tp

    numDimensions = 2
    numVectorsPerDimension = 10
    numVectors = numDimensions * numVectorsPerDimension

    # ----- SubEA -----
    subPopSize = 50
    subPmutate = 1.0 / numVectors
    numTrainingExamples = 5
    numValidationExamples = 5
    subGensWithoutImprovement = 40
    subHalt = LEAP.HaltWhenNoChange(subGensWithoutImprovement)
    #subHalt = LEAP.HaltAfterGeneration(40)

    # subProblem
    valleyDirection = [1.0] * (numDimensions)
    valleyFunc = LEAP.valleyFunctor(valleyDirection)
    #bounds = [LEAP.valley2Bounds[0]] * numDimensions
    valleyMax = LEAP.valleyMaximize
    subProblem = LEAP.FunctionOptimization(valleyFunc, maximize = valleyMax)

    # training set
    # Translate within [-15, 15] along each dimension.
    trainingExamples = []
    for i in range(numTrainingExamples):
        trans = [random.uniform(-15, 15) for i in range(numDimensions)]
        trainingExample = tp.TranslatedProblem(subProblem, trans)
        trainingExamples += [trainingExample]

    # validation set
    validationExamples = []
    for i in range(numValidationExamples):
        trans = [random.uniform(-15, 15) for i in range(numDimensions)]
        validationExample = tp.TranslatedProblem(subProblem, trans)
        validationExamples += [validationExample]

    # subPipeline
    subPipeline = LEAP.TournamentSelection(2)
    subPipeline = LEAP.CloneOperator(subPipeline)
    subPipeline = LEAP.NPointCrossover(subPipeline, 1.0, 2)
    #subPipeline = LEAP.UniformCrossover(subPipeline, 1.0, 0.5)
    subPipeline = LEAP.BitFlipMutation(subPipeline, subPmutate)


    subDecoder = None  # Will be set by the metaEA
    subEA = LEAP.GenerationalEA(subDecoder, subPipeline, subPopSize,\
                                halt=subHalt)
    #subEA = LEAP.GenerationalEA(subDecoder, subPipeline, subPopSize, \
    #                             indClass=Price.PriceIndividual)

    # ----- MetaEA -----
    metaPopSize = 5
    metaGenerations = 1
    magInitRange = (-4, 4)   # the magnitude is an exponent
    vectorInitRange = (-1.0, 1.0)
    magSigma = 1.0
    vectorSigma = 1.0  # From Siggy's paper
    metaPmutate = 1.0 / numVectors
    validationFrequency = 20

    # metaProblem
    metaProblem = g2p.g2pMappingProblem(trainingExamples, subEA)
    validationProblem = g2p.g2pMappingProblem(validationExamples, subEA)

    # metaDecoder
    initRanges = [magInitRange] + [vectorInitRange] * numDimensions
    #bounds = initRanges   # I'm not sure if these really work
    bounds = None
    metaDecoder = g2p.g2pMappingDecoder(metaProblem, numVectors,
                                        initRanges, bounds)

    # metaPipeline
    # Parent Selection (necessary)
    metaPipeline = LEAP.TournamentSelection(2)
    #metaPipeline = LEAP.ProportionalSelection()
    #metaPipeline = LEAP.TruncationSelection(popSize/2)
    #metaPipeline = LEAP.RankSelection()
    #metaPipeline = LEAP.DeterministicSelection()

    # Clone (necessary)
    metaPipeline = LEAP.CloneOperator(metaPipeline)

    # Crossover (not strictly necessary)
    metaPipeline = LEAP.NPointCrossover(metaPipeline, 1.0, 2)
    #metaPipeline = LEAP.UniformCrossover(metaPipeline, 1.0, 0.5)

    # Mutation (not strictly necessary, but you'll almost certainly want it)
    metaPipeline = g2p.g2pMappingGaussianMutation(metaPipeline, vectorSigma,
                                                  metaPmutate, bounds)
    #metaPipeline = g2p.g2pMappingMagnitudeGaussianMutation(metaPipeline,
    #                                      magSigma, metaPmutate, bounds)
    #metaPipeline = g2p.g2pMappingVectorGaussianMutation(metaPipeline,
    #                                      vectorSigma, metaPmutate,
    #                                      bounds)

    # Survival selection (not necessary)
    # If you do use this, you probably want DeterministicSelection above.
    #metaPipeline = LEAP.ElitismSurvival(metaPipeline, 2)
    #metaPipeline = LEAP.MuPlusLambdaSurvival(metaPipeline, popSize, popSize*2)
    #metaPipeline = LEAP.MuCommaLambdaSurvival(metaPipeline, popSize, popSize*10)

    metaEA = g2p.g2pMetaEA(metaDecoder, metaPipeline, metaPopSize, \
                           validationProblem=validationProblem, \
                           validationFrequency=validationFrequency)

    profiling = False
    if profiling:
        import profile
        #profile.run('ea(params)', 'eaprof')
        profile.run('metaEA.run(metaGenerations)', 'eaprof')
    
        import pstats
        p = pstats.Stats('eaprof')
        p.sort_stats('time').print_stats(20)
    else:
        metaEA.run(metaGenerations)

