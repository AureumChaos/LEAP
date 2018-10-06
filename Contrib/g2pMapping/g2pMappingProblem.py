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

import LEAP
#import pyximport; pyximport.install()
from g2pDecoder import *
#from g2pDecoderCy import *
from g2pMappingDecoder import *
from g2pMappingGaussianMutation import *
from g2pMappingMagnitudeGaussianMutation import *
from g2pMappingVectorGaussianMutation import *



#############################################################################
#
# g2pMappingProblem
#
#############################################################################
class g2pMappingProblem(LEAP.Problem):
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
        mapping that is given in the form of an decoder.
        """
        fitness = 0
        for subProblem in self.subProblems:
            decoder = g2pDecoder(subProblem, phenome)
            #decoder = g2pDecoderCy(subProblem, phenome)
            self.subEA.decoder = decoder
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
# unit_test
#
#############################################################################
def myFunction(phenome):
   return(sum([abs(p) for p in phenome]))
                    

def unit_test():
    """
    Test mutation operator
    """
    # Define the subEA
    numDimensions = 2
    subProblem = LEAP.FunctionOptimization(myFunction, maximize = False)
    subProblems = [subProblem]

    subDecoder = None   # Will be set by metaEA

    genomeSize = 8 * numDimensions  # Same for both sub and meta

    subPipeline = LEAP.TournamentSelection(2)
    subPipeline = LEAP.CloneOperator(subPipeline)
    subPipeline = LEAP.UniformCrossover(subPipeline, pCross = 0.8, pSwap = 0.5)
    subPipeline = LEAP.BitFlipMutation(subPipeline, 1.0 / genomeSize)

    subPopSize = 50
    subHalt = LEAP.HaltWhenNoChange(10)

    subEA = LEAP.GenerationalEA(subDecoder, subPipeline, subPopSize, \
                                halt=subHalt)


    # Define the metaEA
    metaProblem = g2pMappingProblem(subProblems, subEA)

    metaInitRanges = [(-5, 2)] + [(0.5, 1.0)] * numDimensions
    metaDecoder = g2pMappingDecoder(metaProblem, genomeSize, metaInitRanges)

    metaPipeline = LEAP.TournamentSelection(2)
    metaPipeline = LEAP.CloneOperator(metaPipeline)
    metaPipeline = LEAP.UniformCrossover(metaPipeline, pCross = 0.8, pSwap = 0.5)
    metaPipeline = g2pMappingGaussianMutation(metaPipeline, sigma=0.1, \
                                              pMutate=1.0/genomeSize)
    #metaPipeline = g2pMappingMagnitudeGaussianMutation(metaPipeline, sigma=0.1,\
    #                                          pMutate=1.0/genomeSize)
    #metaPipeline = g2pMappingVectorGaussianMutation(metaPipeline, sigma=0.1, \
    #                                          pMutate=1.0/genomeSize)

    metaPopSize = 50
    #metaHalt = LEAP.HaltAfterGeneration(50)
    metaHalt = LEAP.HaltWhenNoChange(10)

    metaEA = LEAP.GenerationalEA(metaDecoder, metaPipeline, metaPopSize, \
                                 halt=metaHalt)

    # Run the metaEA
    metaEA.run()

    #print("Passed")



if __name__ == '__main__':
    unit_test()

