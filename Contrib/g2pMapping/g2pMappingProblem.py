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


from LEAP.problem import Problem

#import pyximport; pyximport.install()
#from g2pEncodingCy import *
from LEAP.Contrib.g2pMapping.g2pEncoding import g2pEncoding
from LEAP.Contrib.g2pMapping.g2pMappingEncoding import g2pMappingEncoding
from LEAP.Contrib.g2pMapping.g2pMappingGaussianMutation import g2pMappingGaussianMutation
from LEAP.Contrib.g2pMapping.g2pMappingMagnitudeGaussianMutation import g2pMappingMagnitudeGaussianMutation
from LEAP.Contrib.g2pMapping.g2pMappingVectorGaussianMutation import g2pMappingVectorGaussianMutation



#############################################################################
#
# g2pMappingProblem
#
#############################################################################
class g2pMappingProblem(Problem):
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
            encoding = g2pEncoding(subProblem, phenome)
            #encoding = g2pEncodingCy(subProblem, phenome)
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
# unit_test
#
#############################################################################
def myFunction(phenome):
   return(sum([abs(p) for p in phenome]))
                    

def unit_test():
    from LEAP.problem import FunctionOptimization
    from LEAP.selection import TournamentSelection
    from LEAP.operators import CloneOperator
    from LEAP.operators import UniformCrossover
    from LEAP.operators import BitFlipMutation
    from LEAP.halt import HaltWhenNoChange
    from LEAP.ea import GenerationalEA

    """
    Test mutation operator
    """
    # Define the subEA
    numDimensions = 2
    subProblem = FunctionOptimization(myFunction, maximize = False)
    subProblems = [subProblem]

    subEncoding = None   # Will be set by metaEA

    genomeSize = 8 * numDimensions  # Same for both sub and meta

    subPipeline = TournamentSelection(2)
    subPipeline = CloneOperator(subPipeline)
    subPipeline = UniformCrossover(subPipeline, pCross = 0.8, pSwap = 0.5)
    subPipeline = BitFlipMutation(subPipeline, 1.0 / genomeSize)

    subPopSize = 50
    subHalt = HaltWhenNoChange(10)

    subEA = GenerationalEA(subEncoding, subPipeline, subPopSize, \
                                halt=subHalt)


    # Define the metaEA
    metaProblem = g2pMappingProblem(subProblems, subEA)

    metaInitRanges = [(-5, 2)] + [(0.5, 1.0)] * numDimensions
    metaEncoding = g2pMappingEncoding(metaProblem, genomeSize, metaInitRanges)

    metaPipeline = TournamentSelection(2)
    metaPipeline = CloneOperator(metaPipeline)
    metaPipeline = UniformCrossover(metaPipeline, pCross = 0.8, pSwap = 0.5)
    metaPipeline = g2pMappingGaussianMutation(metaPipeline, sigma=0.1, \
                                              pMutate=1.0/genomeSize)
    #metaPipeline = g2pMappingMagnitudeGaussianMutation(metaPipeline, sigma=0.1,\
    #                                          pMutate=1.0/genomeSize)
    #metaPipeline = g2pMappingVectorGaussianMutation(metaPipeline, sigma=0.1, \
    #                                          pMutate=1.0/genomeSize)

    metaPopSize = 50
    #metaHalt = HaltAfterGeneration(50)
    metaHalt = HaltWhenNoChange(10)

    metaEA = GenerationalEA(metaEncoding, metaPipeline, metaPopSize, \
                                 halt=metaHalt)

    # Run the metaEA
    metaEA.run()

    #print("Passed")



if __name__ == '__main__':
    unit_test()

