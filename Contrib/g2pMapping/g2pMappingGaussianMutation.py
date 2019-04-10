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

from LEAP.operators import MutationOperator
from LEAP.operators import GaussianMutation



#############################################################################
#
# g2pMappingGaussianMutation
#
#############################################################################
class g2pMappingGaussianMutation(MutationOperator):
    """
    Given that each gene in a mapping is a vector (magitude + set of
    components), this operator picks a set of genes, and then performs
    Gaussian mutation on all the element of those genes.
    """
    def __init__(self, provider, sigma, pMutate, bounds = None):
        """
        @param provider: The operator which immediately precedes this one in
                         the pipeline.
        @param sigma: The standard deviation of the Gaussian distribution.
        @param pMutate: The probability that a gene will be mutated.
                        A value between 0 and 1.  Default = 1.0.
        @param linear: Indicates whether the genome is flat (True) or
                       contains sublists (False).  Default = True.
        """
        # XXX Should the magnitude have a separate sigma?
        # XXX For that matter, should each dimension have a separate sigma?
        MutationOperator.__init__(self, provider, pMutate, True)
        self.gaussianOp = GaussianMutation(provider = None, \
              sigma = sigma, pMutate = 1.0, linear = True, bounds = bounds)

    def mutateGene(self, gene):
        """
        Mutate a single gene by performing Gaussian mutation on all it's
        components.

        @param gene: The gene to mutate.
        @return: A modified or new version of C{gene}.
        """
        newGene = self.gaussianOp.linearMutate(gene)
        return newGene



#############################################################################
#
# unit_test
#
#############################################################################
def myFunction(phenome):
   return(sum(abs(phenome)))
                    

def unit_test():
    """
    Test mutation operator
    """
    from LEAP.problem import FunctionOptimization
    from LEAP.individual import Individual
    from LEAP.Contrib.g2pMapping.g2pDecoder import g2pEncoding
    from LEAP.Contrib.g2pMapping.g2pMappingDecoder import g2pMappingEncoding

    numDimensions = 2
    initRanges = [(-5, 2)] + [(0.5, 1.0)] * numDimensions
    bounds = None
    numVectors = 10
        
    problem = FunctionOptimization(myFunction, maximize = False)
    decoder = g2pMappingEncoding(problem, numVectors, initRanges, bounds)
    genome = decoder.randomGenome()
    oldGenome = [i[:] for i in genome]
    ind = Individual(decoder, genome)
    
    assert(len(genome) == numVectors)
    assert(len(genome[0]) == numDimensions + 1)
            
    mutator = g2pMappingGaussianMutation(provider = None, sigma = 0.1, 
                                         pMutate = 0.5) #1.0 / numVectors)
    [newInd] = mutator.apply([ind])
    
    # Did something change?
    newGenome = newInd.genome
    changed = not all([all([f1 == f2 for f1,f2 in zip(g1, g2)]) \
                         for g1,g2 in zip(oldGenome, newGenome)])

    print(oldGenome)
    print(newGenome)
    print("Genome changed:", changed)

    if changed:
        print("Passed")
    else:
        print("Failed")



if __name__ == '__main__':
    unit_test()

