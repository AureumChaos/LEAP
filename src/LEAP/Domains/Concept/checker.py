#! /usr/bin/env python

# concept.py
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

import math

from LEAP.individual import Individual
from LEAP.encoding import FloatEncoding
from concept import *


#############################################################################
#
# CheckerBoardProblem
#
#############################################################################
class CheckerBoardProblem(BinaryConceptLearning):
    """
    """
    def __init__(self, lowerleft = (0.0, 0.0), upperright = (1.0, 1.0),
                 checksize = 0.2):
        BinaryConceptLearning.__init__(self)
        self.top, self.right = upperright
        self.bottom, self.left = lowerleft
        self.checksize = checksize


    def getMaxExamples(self):
        """
        Returns the maximum number of examples available or possible.
        If there is no upper limit, None is returned.
        """
        return None


    def generateExamples(self, numExamples):
        """
        Returns a list of examples for use in either training or testing.
        """
        examples = []
        for i in range(numExamples):
            x = random.random() * (self.right - self.left) + self.left
            y = random.random() * (self.top - self.bottom) + self.bottom
            xb = ((x/self.checksize + 1.0) % 2.0 - 1.0) >= 0.0
            yb = ((y/self.checksize + 1.0) % 2.0 - 1.0) >= 0.0
            classVal = not (xb ^ yb)

            examples.append([[x,y], [classVal]])

        return examples



#############################################################################
#
# unit_test
#
# Creates a data file named checker.dat containing only positive examples for
# a 5x5 checker pattern.  The file is text format, and each line contains an x
# and y value separated by a space.
#
# The file can be viewed in R using the following commands:
#
#     examples <- read.table("checker.dat")
#     plot(examples, pch=".")
#
#############################################################################
def unit_test():
    from LEAP.Exec.Pitt.PittEncoding import PittNearestNeighborEncoding

    checker = CheckerBoardProblem((0.0,0.0),(1.0,1.0),0.2)
    checker.generateExampleGroups(200, 3)
    checker.selectTestSetGroup(0)

    bounds = [(0.0, 0.1)]*3
    ruleEncoding = FloatEncoding(None, bounds, bounds)

    encoding = PittNearestNeighborEncoding(checker, ruleEncoding, 10, 10, 2, 1)

    perfectGenome = \
    [[0.1, 0.9, 1],[0.3, 0.9, 0],[0.5, 0.9, 1],[0.7, 0.9, 0],[0.9, 0.9, 1],\
     [0.1, 0.7, 0],[0.3, 0.7, 1],[0.5, 0.7, 0],[0.7, 0.7, 1],[0.9, 0.7, 0],\
     [0.1, 0.5, 1],[0.3, 0.5, 0],[0.5, 0.5, 1],[0.7, 0.5, 0],[0.9, 0.5, 1],\
     [0.1, 0.3, 0],[0.3, 0.3, 1],[0.5, 0.3, 0],[0.7, 0.3, 1],[0.9, 0.3, 0],\
     [0.1, 0.1, 1],[0.3, 0.1, 0],[0.5, 0.1, 1],[0.7, 0.1, 0],[0.9, 0.1, 1]]
    perfectIndividual = Individual(encoding, perfectGenome)

    wrongGenome = [[r[0], r[1], int(not r[2])] for r in perfectGenome]
    wrongIndividual = Individual(encoding, wrongGenome)

    perfectFitness = perfectIndividual.evaluate()
    print("perfectFitness =", perfectFitness)
    assert(perfectFitness == 1.0)

    wrongFitness = wrongIndividual.evaluate()
    print("wrongFitness =", wrongFitness)
    assert(wrongFitness == 0.0)


    f = open("checker.dat", mode="w")
    examples = checker.generateExamples(100000)
    print(examples[0])
    lines = [str(e[0][0]) + " " + str(e[0][1]) + "\n" for e in examples
             if e[1] == [1]]
    f.writelines(lines)
        
    print("Passed")


if __name__ == '__main__':
    unit_test()

