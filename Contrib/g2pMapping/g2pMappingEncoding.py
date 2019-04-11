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



#############################################################################
#
# g2pMappingEncoding
#
#############################################################################
class g2pMappingEncoding(Encoding):
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
# unit_test
#
#############################################################################
def myFunction(phenome):
   return(sum(abs(phenome)))


def unit_test():
    """
    Test the mapping encoding.
    """
    #from g2pEncoding import g2pEncoding
    from LEAP.Contrib.g2pMapping.g2pEncoding import g2pEncoding 
    from LEAP.problem import FunctionOptimization

    numDimensions = 2
    initRanges = [(-5, 2)] + [(0.5, 1.0)] * numDimensions
    bounds = None
    numVectors = 20

    problem = None
    encoding = g2pMappingEncoding(problem, numVectors, initRanges, bounds)
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
    subEncoding = g2pEncoding(subProblem, encoding.decodeGenome(genome))
    subGenome = '10000101'
    subPhenome = subEncoding.decodeGenome(subGenome)
    print("subPhenome =", subPhenome)
    assert(subPhenome == [8.0, 5.0])


    print("Passed")



if __name__ == '__main__':
    unit_test()

