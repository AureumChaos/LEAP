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

"""
conditionSpaceCrossover.py

Specialized crossover for nearest-neighbor pitt approach systems.  Genes are
swapped based on their location in feature space instead their location on the
genome.
"""

# Python 2 & 3 compatibility
from __future__ import print_function

from math import *
import random

from LEAP.operators import CrossoverOperator



#############################################################################
#
# class ConditionSpaceCrossover
#
#############################################################################
class ConditionSpaceCrossover(CrossoverOperator):
    """
    This is an experimental crossover operator for Pitt approach rule systems
    that use nearest-neighbor generalization.  This crossover operator does
    not pick locations on the genome as crossover points.  Instead, it
    randomly picks a hyper-plane in feature space as the crossover 'point'.
    When nearest-neighbor generalization is used, the condition section of
    each rule determines a point in feature space.  This operator will
    transfer a rule to one child or the other depending on which side of the
    hyper-plane its condition lies.

    The motiviation for this approach is simple.  In most EAs which use a
    canonical representation, the population tends to converge to a small area
    near the solution in problem space (e.g. phenotype space).  Standard Pitt
    approach EAs do not share this feature due to their unusual crossover
    operators.  For example in a Pitt system where both parents contain the
    same crucial rule, it is possible that one of the offspring will inherit
    neither rule.  This means that the population as a whole cannot converge
    on a solution because defective offspring are always being created.  These
    defective offspring are often very far away from their parents in problem
    space.

    This ConditionSpaceCrossover operator should not suffer from problem
    described above.  It also handles variable length genomes easily, and it
    does not require some sort of complicated matching process (required by
    most homologous crossover operators) in order to inherit rules
    effectively.
    """
    parentsNeeded = 2

    def __init__(self, provider, pCross, conditionBounds, numChildren = 2):
        """
        @param provider: The operator which immediately precedes this one in
                         the pipeline.
        @param pCross: The probability that a pair of individuals will be
                       recombined.  A value between 0 and 1.
        @param conditionBounds: A list of tuples.  Each tuple defines the min
                                and max values for one of the rule conditions
                                (i.e. an input).  These obviously need to be
                                defined in the same order as the conditions in
                                a rule.  None values are not allowed.
        @param numChildren: The number of children that will be produced
                            (1 or 2).  Default is 2.
        """
        CrossoverOperator.__init__(self, provider, pCross, numChildren)

        # XXX We could, and perhaps should, check conditionBounds for errors.
        self.conditionBounds = conditionBounds


    def pickCrossoverPlane(self, conditionBounds):
        numConditions = len(conditionBounds)

        # Pick the normal to plane (distorted to match bounds)
        normalToPlane = [random.gauss(0,1) * (b[1]-b[0]) \
                           for b in conditionBounds]
        length = sqrt(sum([i*i for i in normalToPlane]))
        normalToPlane = [i/length for i in normalToPlane]

        # Find the farthest extents of the boundaries.  In other words, if a
        # plane defined by normalToPlane were to move along its normal vector,
        # What would be the first conditionBounds point/corner (minCorner) and
        # last point/corner (maxCorner) it would travel through.
        maxCorner = [int(i >= 0) for i in normalToPlane]
        # Calculate the intersection between the normal vector (at the
        # origin) and the plane defined by the normal and the corner.
        temp1 = sum([i*j for i,j in zip(normalToPlane, maxCorner)]) # N dot c
        temp2 = sum([i*i for i in normalToPlane])                   # N dot N
        maxDist = temp1/temp2

        minCorner = [int(i < 0) for i in normalToPlane]
        temp1 = sum([i*j for i,j in zip(normalToPlane, minCorner)]) # N dot c
        temp2 = sum([i*i for i in normalToPlane])                   # N dot N
        minDist = temp1/temp2

        # Randomly pick distance to pointOnPlane
        dist = random.random()*(maxDist - minDist) + minDist

        # Calculate pointOnPlane
        pointOnPlane = [dist * i for i in normalToPlane]

        return pointOnPlane, normalToPlane


    def recombine(self, child1, child2):
        """
        @param child1: The first individual to be recombined.
        @param child2: The second individual to be recombined.
        @return: (child1, child2)
        """
        # Check for errors.
        #if len(child1.genome) < self.numPoints + 1:
        #    raise OperatorError("Not enough available crossover locations.")

        child1.numSwaps = child2.numSwaps = 0
        genome1 = child1.genome[0:0]  # empty sequence - maintain type
        genome2 = child2.genome[0:0]

        # Pick the crossover plane
        pointOnPlane, normalToPlane = self.pickCrossoverPlane(
                                                         self.conditionBounds)

        # Perform the crossover
        for rule in child1.genome:
            diff = [r - p for r,p in zip(rule, pointOnPlane)]
            dot = sum([d * n for d,n in zip(diff, normalToPlane)])
            
            if dot >= 0.0:
                genome1.append(rule)
            else:
                genome2.append(rule)
                child1.numSwaps += 1

        for rule in child2.genome:
            diff = [r - p for r,p in zip(rule, pointOnPlane)]
            dot = sum([d * n for d,n in zip(diff, normalToPlane)])
            
            if dot >= 0.0:
                genome2.append(rule)
            else:
                genome1.append(rule)
                child2.numSwaps += 1

        child1.genome = genome1
        child2.genome = genome2

        return (child1, child2)



SpatialCrossover = ConditionSpaceCrossover  # For backward compatibility


#############################################################################
#
# unit_test
#
#############################################################################
def unit_test():

    import copy

    class MyIndividual:
        def __init__(self, genome):
            self.genome = genome

    numInputs = 2
    numOutputs = 1
    minB = 0.0
    maxB = 1.0
    conditionBounds = [(minB, maxB)] * numInputs

    #x = NPointCrossover(None, 1.0, 2)
    x = ConditionSpaceCrossover(None, 1.0, conditionBounds)

    # Test to make sure two identical genomes produce identical offspring
    genome1 = [[minB, minB, 0.0], [minB, maxB, 0.0], \
               [maxB, minB, 0.0], [maxB, maxB, 0.0]]
    genome2 = [[minB, minB, 0.0], [minB, maxB, 0.0], \
               [maxB, minB, 0.0], [maxB, maxB, 0.0]]
    a = MyIndividual(genome1)
    b = MyIndividual(genome2)

    c,d = x.recombine(a,b)
    c.genome.sort() 
    d.genome.sort() 
    print(c.genome)
    print(d.genome)
    assert(c.genome == d.genome)

    numChanges = 0.0
    total = 1000
    for i in range(total):
        # Test to make sure two different genomes produce different offspring
        # Note: This is not guaranteed, but is very likely (maybe 86%?).
        #       If the assert below fails, try running it again.
        genome1 = [[random.random()+minB for i in range(3)] for j in range(4)]
        genome2 = [[random.random()+maxB for i in range(3)] for j in range(4)]
        genome1.sort()
        genome2.sort()
        a = MyIndividual(genome1[:])
        b = MyIndividual(genome2[:])
        c,d = x.recombine(a,b)
        c.genome.sort() 
        d.genome.sort() 

        #print()
        #print(genome1)
        #print(genome2)
        #print(c.genome)
        #print(d.genome)
        #print()
        #assert(genome1 != c.genome)
        #assert(genome2 != d.genome)
        if genome1 != c.genome and genome2 != d.genome:
            numChanges += 1

    changeRatio = numChanges/total
    print("change ratio =", changeRatio)
    assert(changeRatio > .8 and changeRatio < .9)

    # Test to see if it works when there is only one dimension/condition
    numInputs = 1
    numOutputs = 1
    minB = 0.0
    maxB = 1.0
    conditionBounds = [(minB, maxB)] * numInputs

    #x = NPointCrossover(None, 1.0, 2)
    x = ConditionSpaceCrossover(None, pCross=1.0, \
                                conditionBounds=conditionBounds)
    genome1 = [[minB, 0.0], [minB, 0.0], [minB, 0.0], \
               [maxB, 0.0]]
    genome2 = [[minB, 1.0], [minB, 1.0], \
               [maxB, 1.0], [maxB, 1.0]]

    a = MyIndividual(genome1)
    b = MyIndividual(genome2)

    c,d = x.recombine(a,b)
    c.genome.sort() 
    d.genome.sort() 
    print()
    print(c.genome)
    print(d.genome)
    assert((len(c.genome) == 3 and len(d.genome) == 5) or
           (len(c.genome) == 5 and len(d.genome) == 3))


    # Can it handle a single variable?
    conditionBounds = [(minB, maxB)]
    x = ConditionSpaceCrossover(None, 1.0, conditionBounds)

    genome1 = [[0.0, 'A'], [0.2, 'A'], [0.4, 'A'],
               [0.6, 'A'], [0.8, 'A'], [1.0, 'A']]
    genome2 = [[0.0, 'B'], [0.2, 'B'], [0.4, 'B'],
               [0.6, 'B'], [0.8, 'B'], [1.0, 'B']]

    a = MyIndividual(genome1)
    b = MyIndividual(genome2)

    c,d = x.recombine(a,b)
    c.genome.sort() 
    d.genome.sort() 
    print()
    print(c.genome)
    print(d.genome)
#    assert((len(c.genome) == 3 and len(d.genome) == 5) or
#           (len(c.genome) == 5 and len(d.genome) == 3))

    print("Passed?")


if __name__ == '__main__':
    unit_test()

#    import profile
#    profile.run('unit_test()', 'profile')
#
#    import pstats
#    p = pstats.Stats('profile')
#    p.sort_stats('time').print_stats(20)

