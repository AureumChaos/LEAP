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
import string
import copy
import random
import math

from LEAP.encoding import Encoding
from LEAP.encoding import FloatEncoding
from LEAP.problem import Problem
from LEAP.individual import Individual
from LEAP.selection import DeterministicSelection
from LEAP.ea import GenerationalEA

#import pyximport; pyximport.install()
#from g2pEncodingCy import *
from LEAP.Contrib.g2pMapping.sub import g2pSubEncoding
from LEAP.Contrib.g2pMapping.meta_ops import g2pMetaGaussianMutation
from LEAP.Contrib.g2pMapping.meta_ops import g2pMetaMagnitudeGaussianMutation
from LEAP.Contrib.g2pMapping.meta_ops import g2pMetaVectorGaussianMutation




#############################################################################
#
# g2pMetaEncoding
#
#############################################################################
class g2pMetaEncoding(Encoding):
    """
    This class defines the genetic encoding for a genotype to phenotype map.
    The phenotype it generates is a mapping, that can then be used to create a
    g2pEncoding and then convert a bit string into a real value.
    
    The idea is that this mapping will be evolved in a meta-EA.  To evaluate a
    mapping (a meta-EA individual), a sub-EA will be launched and will try to
    solve the problem.

    The parameters initRanges and bounds should both be the same length
    (although bounds is an optional parameter).  That length is the number of
    dimensions in the problem +1 for the magnitude value of the vector (which
    is the first).  The magnitude is an exponent, as in 2**mag.
    """
    def __init__(self, problem, numVectors, initRanges, bounds=None):
        Encoding.__init__(self, problem)

        self.numDimensions = len(initRanges) - 1
        self.numVectors = numVectors
        self.vectorEncoding = FloatEncoding(problem, initRanges, bounds)

    def decodeGenome(self, genome):
        # The phenotype is a encoding that uses this mapping
        #return g2pEncoding(self.problem, genome)

        return genome


    def randomGenome(self):
        newGenome = []
        for i in range(self.numVectors):
            newGene = self.vectorEncoding.randomGenome()
            newGenome.append(newGene)

        return newGenome


#############################################################################
#
# g2pMetaProblem
#
#############################################################################
class g2pMetaProblem(Problem):
    """
    This problem defines a fitness function that is evaluated by running a
    sub-EA on a problem given a certain genotype to phenotype mapping.
    """
    def __init__(self, subProblems, subEA):
        self.subProblems = subProblems
        self.subEA = subEA


    def evaluate(self, phenome):
        """
        Evaluates (calculates the fitness of) a genotype to phenotype
        mapping that is given in the form of an encoding.
        """
        fitness = 0
        for subProblem in self.subProblems:
            encoding = g2pSubEncoding(subProblem, phenome)
            #encoding = g2pSubEncodingCy(subProblem, phenome)
            self.subEA.encoding = encoding
            bsf = self.subEA.run()
            fitness += bsf.getFitness() / len(self.subProblems)

        return fitness


    def cmpFitness(self, fitness1, fitness2):
        """
        Compare two fitnesses and determine if C{fitness1} is "better than"
        "equal to" or "worse than" C{fitness2}.
        Better than means '>' if maximizing or '<' if minimizing.

        @param fitness1: The first fitness value to be compared.
        @param fitness2: The second fitness value to be compared.

        @return:
            1 if fitness1 is "better than" fitness2
            0 if fitness1 = fitness2
            -1 if fitness1 is worse than fitness2
        """
        # I will assume that all the subProblems are roughtly similar in how
        # comparisons are performed.  If not, some strange things could
        # happen.
        return self.subProblems[0].cmpFitness(fitness1, fitness2)


#############################################################################
#
# g2pMetaEA
#
##############################################################################
class g2pMetaEA(GenerationalEA):
    def __init__(self, encoding, pipeline, popSize, validationProblem, \
                 initPipeline=DeterministicSelection(), \
                 indClass=Individual, \
                 initPopsize=None, halt=None, validationFrequency=20):
        GenerationalEA.__init__(self, encoding, pipeline, popSize, \
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
            val = self.validationProblem.evaluate(
                                       self.encoding.decodeGenome(ind.genome) )

            if bestVal == None or self.encoding.cmpFitness(bestVal, val) == -1:
                bestVal = val
                bestIndex = i

        return (bestVal, self.population[bestIndex])


    def startup(self):
        self.validationBOG = (None, [])  # Best of all validations per gen
        self.validationBSF = (None, [])
        self.bogValidation = None  # Validation of usual best-of-generation
        GenerationalEA.startup(self)


    def calcStats(self):
        GenerationalEA.calcStats(self)

        if self.generation % self.validationFrequency == 0:
            tempValid = self.validate()
            print(tempValid)
            self.validationBOG = (tempValid[0], tempValid[1])
            if self.validationProblem.cmpFitness(self.validationBSF[0],
                                                 self.validationBOG[0]) == -1:
                self.validationBSF = self.validationBOG

        # Run validation on the bestOfGen individual
        self.bogValidation = self.validationProblem.evaluate(
                              self.encoding.decodeGenome(self.bestOfGen.genome))


    def printStats(self):
        GenerationalEA.printStats(self)
        print("Gen:", self.generation, " Ind: BOGV  Val:", self.bogValidation)
        if self.generation % self.validationFrequency == 0:
            print("Gen:", self.generation, " Ind: VBOG ", self.validationBOG[1],
                  "Val:", self.validationBOG[0])




#############################################################################
#
# unit_test_g2pMetaEncoding
#
#############################################################################
#def myFunction(phenome):
#   return(sum(abs(phenome)))

def myFunction(phenome):
   return(sum([abs(p) for p in phenome]))


def unit_test_g2pMetaEncoding():
    """
    Test the mapping encoding.
    """
    from LEAP.Contrib.g2pMapping.sub import g2pSubEncoding 
    from LEAP.problem import FunctionOptimization

    numDimensions = 2
    initRanges = [(-5, 2)] + [(0.5, 1.0)] * numDimensions
    bounds = None
    numVectors = 20

    problem = None
    encoding = g2pMetaEncoding(problem, numVectors, initRanges, bounds)
    genome = encoding.randomGenome()

    assert(len(genome) == numVectors)
    assert(len(genome[0]) == numDimensions + 1)

    genome = [[3.0, 1.0, 0.0],
              [2.0, 1.0, 0.0],
              [1.0, 1.0, 0.0],
              [0.0, 1.0, 0.0],
              [3.0, 0.0, 1.0],
              [2.0, 0.0, 1.0],
              [1.0, 0.0, 1.0],
              [0.0, 0.0, 1.0]]
    subProblem = FunctionOptimization(myFunction, maximize = False)
    subEncoding = g2pSubEncoding(subProblem, encoding.decodeGenome(genome))
    subGenome = '10000101'
    subPhenome = subEncoding.decodeGenome(subGenome)
    print("subPhenome =", subPhenome)
    assert(subPhenome == [8.0, 5.0])


    print("Passed")


#############################################################################
#
# unit_test_g2pMetaProblem
#
#############################################################################
def unit_test_g2pMetaProblem():
    """
    Test the meta EA problem class
    """
    print("No unit test for g2pMetaProblem")


#############################################################################
#
# Main
#
#############################################################################
def test_metaEA():
    from LEAP.problem import valleyFunctor
    from LEAP.problem import valleyMaximize
    from LEAP.problem import valleyBounds
    from LEAP.problem import FunctionOptimization
    from LEAP.selection import TournamentSelection
    from LEAP.operators import CloneOperator
    from LEAP.operators import BitFlipMutation
    from LEAP.operators import NPointCrossover
    #from LEAP.survival import ElitismSurvival
    from LEAP.halt import HaltWhenNoChange
    from LEAP.halt import HaltAfterGeneration

    from LEAP.Domains.Translated.translatedProblem import TranslatedProblem

    from LEAP.Contrib.g2pMapping.meta import g2pMetaProblem
    from LEAP.Contrib.g2pMapping.meta import g2pMetaEncoding
    from LEAP.Contrib.g2pMapping.meta_ops import g2pMetaGaussianMutation
    from LEAP.Contrib.g2pMapping.meta_ops import g2pMetaMagnitudeGaussianMutation
    from LEAP.Contrib.g2pMapping.meta_ops import g2pMetaVectorGaussianMutation

    numDimensions = 2
    numVectorsPerDimension = 10
    numVectors = numDimensions * numVectorsPerDimension

    # ----- SubEA -----
    subPopSize = 50
    subPmutate = 1.0 / numVectors
    numTrainingExamples = 5
    numValidationExamples = 5
    subGensWithoutImprovement = 40
    subHalt = HaltWhenNoChange(subGensWithoutImprovement)
    #subHalt = HaltAfterGeneration(40)

    # subProblem
    valleyDirection = [1.0] * (numDimensions)
    valleyFunc = valleyFunctor(valleyDirection)
    #bounds = [valley2Bounds[0]] * numDimensions
    valleyMax = valleyMaximize
    subProblem = FunctionOptimization(valleyFunc, maximize = valleyMax)

    # training set
    # Translate within [-15, 15] along each dimension.
    trainingExamples = []
    for i in range(numTrainingExamples):
        trans = [random.uniform(-15, 15) for i in range(numDimensions)]
        trainingExample = TranslatedProblem(subProblem, trans)
        trainingExamples += [trainingExample]

    # validation set
    validationExamples = []
    for i in range(numValidationExamples):
        trans = [random.uniform(-15, 15) for i in range(numDimensions)]
        validationExample = TranslatedProblem(subProblem, trans)
        validationExamples += [validationExample]

    # subPipeline
    subPipeline = TournamentSelection(2)
    subPipeline = CloneOperator(subPipeline)
    subPipeline = NPointCrossover(subPipeline, 1.0, 2)
    #subPipeline = UniformCrossover(subPipeline, 1.0, 0.5)
    subPipeline = BitFlipMutation(subPipeline, subPmutate)


    subEncoding = None  # Will be set by the metaEA
    subEA = GenerationalEA(subEncoding, subPipeline, subPopSize,\
                                halt=subHalt)
    #subEA = GenerationalEA(subEncoding, subPipeline, subPopSize, \
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
    metaProblem = g2pMetaProblem(trainingExamples, subEA)
    validationProblem = g2pMetaProblem(validationExamples, subEA)

    # metaEncoding
    initRanges = [magInitRange] + [vectorInitRange] * numDimensions
    #bounds = initRanges   # I'm not sure if these really work
    bounds = None
    metaEncoding = g2pMetaEncoding(metaProblem, numVectors,
                                        initRanges, bounds)

    # metaPipeline
    # Parent Selection (necessary)
    metaPipeline = TournamentSelection(2)
    #metaPipeline = ProportionalSelection()
    #metaPipeline = TruncationSelection(popSize/2)
    #metaPipeline = RankSelection()
    #metaPipeline = DeterministicSelection()

    # Clone (necessary)
    metaPipeline = CloneOperator(metaPipeline)

    # Crossover (not strictly necessary)
    metaPipeline = NPointCrossover(metaPipeline, 1.0, 2)
    #metaPipeline = UniformCrossover(metaPipeline, 1.0, 0.5)

    # Mutation (not strictly necessary, but you'll almost certainly want it)
    metaPipeline = g2pMetaGaussianMutation(metaPipeline, vectorSigma,
                                                  metaPmutate, bounds)
    #metaPipeline = g2pMetaMagnitudeGaussianMutation(metaPipeline,
    #                                      magSigma, metaPmutate, bounds)
    #metaPipeline = g2pMetaVectorGaussianMutation(metaPipeline,
    #                                      vectorSigma, metaPmutate,
    #                                      bounds)

    # Survival selection (not necessary)
    # If you do use this, you probably want DeterministicSelection above.
    #metaPipeline = ElitismSurvival(metaPipeline, 2)
    #metaPipeline = MuPlusLambdaSurvival(metaPipeline, popSize, popSize*2)
    #metaPipeline = MuCommaLambdaSurvival(metaPipeline, popSize, popSize*10)

    metaEA = g2pMetaEA(metaEncoding, metaPipeline, metaPopSize, \
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


if __name__ == '__main__':
    #unit_test_g2pMetaEncoding()
    test_metaEA()


