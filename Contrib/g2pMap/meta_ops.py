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
# g2pMetaGaussianMutation
#
#############################################################################
class g2pMetaGaussianMutation(MutationOperator):
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
# g2pMetaMagnitudeGaussianMutation
#
#############################################################################
class g2pMetaMagnitudeGaussianMutation(g2pMetaGaussianMutation):
    """
    Given that each gene in a mapping is a vector (magitude + set of
    components), this operator picks a set of genes, and then performs
    Gaussian mutation on only the magnitude element of those genes.
    """
    def __init__(self, provider, sigma, pMutate, bounds = None):
        """
        @param provider: The operator which immediately precedes this one in
                         the pipeline.
        @param sigma: The standard deviation of the Gaussian distribution.
        @param pMutate: The probability that a gene will be mutated.
                        A value between 0 and 1.  Default = 1.0.
        @param bounds: Enforced bounds that the operator will not cross.
                       Define magnitude only.  Default = None.
        """
        g2pMetaGaussianMutation.__init__(self, provider, sigma, pMutate,
                                             bounds)

    def mutateGene(self, gene):
        """
        Mutate a single gene by performing Gaussian mutation on all it's
        components.

        @param gene: The gene to mutate.
        @return: A modified or new version of C{gene}.
        """
        gene[:1] = self.gaussianOp.linearMutate(gene[:1])
        return gene



#############################################################################
#
# g2pMetaVectorGaussianMutation
#
#############################################################################
class g2pMetaVectorGaussianMutation(g2pMetaGaussianMutation):
    """
    Given that each gene in a mapping is a vector (magitude + set of
    components), this operator picks a set of genes, and then performs
    Gaussian mutation on everything but the magnitude element of those genes.
    """
    def __init__(self, provider, sigma, pMutate, bounds = None):
        """
        @param provider: The operator which immediately precedes this one in
                         the pipeline.
        @param sigma: The standard deviation of the Gaussian distribution.
        @param pMutate: The probability that a gene will be mutated.
                        A value between 0 and 1.  Default = 1.0.
        @param bounds: Enforced bounds that the operator will not cross.
                       Define magnitude only.  Default = None.
        """
        g2pMetaGaussianMutation.__init__(self, provider, sigma, pMutate,
                                             bounds)

    def mutateGene(self, gene):
        """
        Mutate a single gene by performing Gaussian mutation on all it's
        components.

        @param gene: The gene to mutate.
        @return: A modified or new version of C{gene}.
        """
        gene[1:] = self.gaussianOp.linearMutate(gene[1:])
        return gene



#############################################################################
#
# unit_test_all
#
#############################################################################
def myFunction(phenome):
   return(sum(abs(phenome)))
                    

def unit_test_all():
    """
    Test mutation operator
    """
    from LEAP.problem import FunctionOptimization
    from LEAP.individual import Individual
    from LEAP.Contrib.g2pMapping.sub import g2pSubEncoding
    from LEAP.Contrib.g2pMapping.meta import g2pMetaEncoding

    numDimensions = 2
    initRanges = [(-5, 2)] + [(0.5, 1.0)] * numDimensions
    bounds = None
    numVectors = 10
        
    problem = FunctionOptimization(myFunction, maximize = False)
    encoding = g2pMetaEncoding(problem, numVectors, initRanges, bounds)
    genome = encoding.randomGenome()
    oldGenome = [i[:] for i in genome]
    ind = Individual(encoding, genome)
    
    assert(len(genome) == numVectors)
    assert(len(genome[0]) == numDimensions + 1)
            
    mutator = g2pMetaGaussianMutation(provider = None, sigma = 0.1, 
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
        return True
    else:
        print("Failed")
        return False



#############################################################################
#
# unit_test_mag
#
#############################################################################
def unit_test_mag():
    """
    Test magnitude mutation operator
    """
    from LEAP.problem import FunctionOptimization
    from LEAP.individual import Individual
    from LEAP.Contrib.g2pMapping.meta import g2pMetaEncoding

    numDimensions = 2
    initRanges = [(-5, 2)] + [(0.5, 1.0)] * numDimensions
    bounds = None
    numVectors = 10
        
    problem = FunctionOptimization(myFunction, maximize = False)
    encoding = g2pMetaEncoding(problem, numVectors, initRanges, bounds)
    genome = encoding.randomGenome()
    oldGenome = [i[:] for i in genome]
    ind = Individual(encoding, genome)
    
    assert(len(genome) == numVectors)
    assert(len(genome[0]) == numDimensions + 1)
            
    mutator = g2pMetaMagnitudeGaussianMutation(provider = None, sigma = 0.1, 
                                         pMutate = 0.5) #1.0 / numVectors)
    [newInd] = mutator.apply([ind])
    
    # What change?
    newGenome = newInd.genome
    magChanged = not all([g1[0] == g2[0] for g1,g2 in zip(oldGenome, newGenome)])
    vecChanged = not all([all([f1 == f2 for f1,f2 in zip(g1[1:], g2[1:])]) \
                         for g1,g2 in zip(oldGenome, newGenome)])

    print(oldGenome)
    print(newGenome)
    print("Magnitudes changed:", magChanged)
    print("Vectors changed:", vecChanged)

    if magChanged and not vecChanged:
        print("Passed")
        return True
    else:
        print("Failed")
        return False


#############################################################################
#
# unit_test_vec
#
#############################################################################
def unit_test_vec():
    """
    Test vector mutation operator
    """
    from LEAP.problem import FunctionOptimization
    from LEAP.individual import Individual
    from LEAP.Contrib.g2pMapping.meta import g2pMetaEncoding

    numDimensions = 2
    initRanges = [(-5, 2)] + [(0.5, 1.0)] * numDimensions
    bounds = None
    numVectors = 10
        
    problem = FunctionOptimization(myFunction, maximize = False)
    encoding = g2pMetaEncoding(problem, numVectors, initRanges, bounds)
    genome = encoding.randomGenome()
    oldGenome = [i[:] for i in genome]
    ind = Individual(encoding, genome)
    
    assert(len(genome) == numVectors)
    assert(len(genome[0]) == numDimensions + 1)
            
    mutator = g2pMetaVectorGaussianMutation(provider = None, sigma = 0.1, 
                                         pMutate = 0.5) #1.0 / numVectors)
    [newInd] = mutator.apply([ind])
    
    # What change?
    newGenome = newInd.genome
    magChanged = not all([g1[0] == g2[0] for g1,g2 in zip(oldGenome, newGenome)])
    vecChanged = not all([all([f1 == f2 for f1,f2 in zip(g1[1:], g2[1:])]) \
                         for g1,g2 in zip(oldGenome, newGenome)])

    print(oldGenome)
    print(newGenome)
    print("Magnitudes changed:", magChanged)
    print("Vectors changed:", vecChanged)

    if not magChanged and vecChanged:
        print("Passed")
        return True
    else:
        print("Failed")
        return False


#############################################################################
#
# unit_tests
#
#############################################################################
def unit_tests():
    """
    Test each mutation operator
    """
    test_all = unit_test_all()
    test_mag = unit_test_mag()
    test_vec = unit_test_vec()

    if all([test_all, test_mag, test_vec]):
        print("Passed all tests")
    else:
        print("Failed at least one test")


if __name__ == '__main__':
    unit_tests()

