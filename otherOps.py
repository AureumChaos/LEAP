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

import LEAP
import random


#############################################################################
#
# class Shuffle2PointCrossover
#
#############################################################################
class Shuffle2PointCrossover(LEAP.CrossoverOperator):
    """
    A 2 Point Crossover operator which shuffles gene locations.  In other
    words, the sections of the genomes which are swapped will always be the
    same size, but they may not come from the same place on the genomes.
    """
    parentsNeeded = 2

    def __init__(self, provider, pCross, numChildren = 2):
        """
        @param provider: The operator which immediately precedes this one in
                         the pipeline.
        @param pCross: The probability that a pair of individuals will be
                       recombined.  A value between 0 and 1.
        @param numChildren: The number of children that will be produced
                            (1 or 2).  Default is 2.
        """
        LEAP.CrossoverOperator.__init__(self, provider, pCross, numChildren)


    def pickCrossoverPoints(self, numPoints, genomeSize):
        """
        Randomly choose (without replacement) crossover points.
        """
        pp = list(range(1,genomeSize))  # Possible points
        xpts = [pp.pop(random.randrange(len(pp))) for i in range(numPoints)]
        xpts.sort()
        xpts = [0] + xpts + [genomeSize]  # Add start and end
        return xpts


    def pickOtherCrossoverPoints(self, swapSize, genomeSize):
        """
        Select crossover points such that the areas swapped will be the same
        size, but the location is not important.

        Assumes 2 crossover points.
        """
        first = random.randint(1, genomeSize - swapSize - 1)
        xpts = [0, first, first + swapSize, genomeSize]
        return xpts


    def recombine(self, child1, child2):
        """
        @param child1: The first individual to be recombined.
        @param child2: The second individual to be recombined.
        @return: (child1, child2)
        """
        # Check for errors.
        if len(child1.genome) != len(child2.genome):
            raise OperatorError("Cannot recombine genomes of different size.")
        if len(child1.genome) < 3:
            raise OperatorError("Not enough available crossover locations.")

        children = [child1, child2]
        genome1 = child1.genome[0:0]  # empty sequence - maintain type
        genome2 = child2.genome[0:0]
        src1, src2 = 0, 1

        # Pick crossover points
        xpts1 = self.pickCrossoverPoints(2, len(child1.genome))
        xpts2 = self.pickOtherCrossoverPoints(xpts1[2] - xpts1[1],
                                              len(child1.genome))
        xpts = [xpts1, xpts2]

        # Perform the crossover
        for i in range(len(xpts[0])-1):  # swap odd segments
            genome1 += children[src1].genome[xpts[src1][i]:xpts[src1][i+1]]
            genome2 += children[src2].genome[xpts[src2][i]:xpts[src2][i+1]]
            src1, src2 = src2, src1

        child1.genome = genome1
        child2.genome = genome2

        # Gather some statistics
        child1.numSwaps = child2.numSwaps = sum([xpts[0][i+1] - xpts[0][i] \
                                             for i in range(1,len(xpts)-1,2)])

        return (child1, child2)



#############################################################################
#
# class Shuffle2PointGeneOffsetCrossover
#
#############################################################################
class Shuffle2PointGeneOffsetCrossover(Shuffle2PointCrossover):
    """
    """

    def pickCrossoverPoints(self, numPoints, genomeSize):
        """
        Randomly choose (without replacement) crossover points.
        """
        pp = list(range(0,genomeSize))  # Possible points
        xpts = [pp.pop(random.randrange(len(pp))) for i in range(numPoints)]
        xpts.sort()
        xpts = [0] + xpts + [genomeSize]  # Add start and end
        return xpts


    def pickOtherCrossoverPoints(self, swapSize, genomeSize):
        """
        Select crossover points such that the areas swapped will be the same
        size, but the location is not important.

        Assumes 2 crossover points.
        """
        first = random.randint(0, genomeSize - swapSize - 1)
        xpts = [0, first, first + swapSize, genomeSize]
        return xpts


    def recombine(self, child1, child2):
        # Check for errors.
        if len(child1.genome) != len(child2.genome):
            raise OperatorError("Cannot recombine genomes of different size.")
        if len(child1.genome) < 3:
            raise OperatorError("Not enough available crossover locations.")

        children = [child1, child2]

        # Pick crossover points
        xpts1 = self.pickCrossoverPoints(2, len(child1.genome))
        xpts2 = self.pickOtherCrossoverPoints(xpts1[2] - xpts1[1],
                                              len(child1.genome))
        xpts = [xpts1, xpts2]

        # Pick rule offsets
        ruleLen = len(child1.genome[0])
        offsets = [random.randrange(ruleLen) for i in range(2)]
        offsets = [0] + offsets + [0]

        # Perform the crossover
        src1, src2 = 0, 1
        genome1 = children[src1].genome[:xpts[src1][1]]
        genome2 = children[src2].genome[:xpts[src2][1]]
        for i in range(1,len(xpts[0])-1):
            genome1.append(children[src1].genome[xpts[src1][i]][:offsets[i]] +\
                           children[src2].genome[xpts[src2][i]][offsets[i]:])
            genome2.append(children[src2].genome[xpts[src2][i]][:offsets[i]] +\
                           children[src1].genome[xpts[src1][i]][offsets[i]:])
            src1, src2 = src2, src1

            genome1 += children[src1].genome[xpts[src1][i]+1:xpts[src1][i+1]]
            genome2 += children[src2].genome[xpts[src2][i]+1:xpts[src2][i+1]]
            
        child1.genome = genome1
        child2.genome = genome2

        # Gather some statistics
        #child1.numSwaps = ...
        #child2.numSwaps = ...

        return (child1, child2)




#############################################################################
#
# unit_test
#
#############################################################################
def unit_test():

    class MyIndividual:
        def __init__(self, genome):
            self.genome = genome

    x = Shuffle2PointCrossover(None, 1.0, numChildren = 2)
    a = MyIndividual(list(range(10)))
    b = MyIndividual(list(range(10)))
    c,d = x.recombine(a,b)
    print(c.genome)
    print(d.genome)

    x = Shuffle2PointGeneOffsetCrossover(None, 1.0, numChildren = 2)
    a = MyIndividual(['abc','def','ghi','jkl','mno'])
    b = MyIndividual(['ABC','DEF','GHI','JKL','MNO'])
    c,d = x.recombine(a,b)
    print(c.genome)
    print(d.genome)

    print("Passed?")


if __name__ == '__main__':
    unit_test()

