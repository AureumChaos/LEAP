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

import copy    # for Clone
import random
import math
from queue import Queue

#from problem import *
#from decoder import *
import LEAP

#############################################################################
#
# class HaltingCriteria
#
#############################################################################
class HaltingCriteria:
    """
    Defines a halting criteria that will end the EA run.
    """
    def reset(self):
        """
        Used to reset the state for a new run.
        """
        raise NotImplementedError  # Subclasses should redefine this


    def shouldHaltNow(self, population):
        """
        This will be called at the end of each generation.
        """
        raise NotImplementedError  # Subclasses should redefine this
        return True


#############################################################################
#
# class HaltAfterGeneration
#
#############################################################################
class HaltAfterGeneration(HaltingCriteria):
    """
    Halt after a certain number of generations.
    """
    def __init__(self, maxGeneration):
        self.maxGeneration = maxGeneration
        self.reset()


    def reset(self):
        self.currentGen = 0


    def shouldHaltNow(self, population):
        """
        This will be called at the end of each generation.
        """

        self.currentGen += 1
        haltNow = self.currentGen >= self.maxGeneration + 1
        return haltNow



#############################################################################
#
# class HaltWhenNoChange
#
#############################################################################
class HaltWhenNoChange(HaltingCriteria):
    """
    Halt when the best so far hasn't changed for a certain number of
    generations.
    """
    def __init__(self, maxGenWithoutChange):
        self.maxGenWithoutChange = maxGenWithoutChange
        self.reset()


    def reset(self):
        self.lastChange = 0
        self.bestSoFar = None


    def shouldHaltNow(self, population):
        """
        This will be called at the end of each generation.
        """
        bestOfGen = copy.copy(LEAP.fittest(population))
        if LEAP.fittest(self.bestSoFar, bestOfGen) == self.bestSoFar:
            self.lastChange += 1
        else:
            self.bestSoFar = bestOfGen
            self.lastChange = 0

        return self.lastChange >= self.maxGenWithoutChange



#############################################################################
#
# unit_test
#
#############################################################################
def testFunction(phenome):
    return(phenome[0])

def unit_test():

    print("HaltAfterGeneration")
    pop = []
    halt = HaltAfterGeneration(10)
    generation = 0
    for i in range(1, 11):   # 1 through 10
        print(i)
        assert(halt.shouldHaltNow(pop) == False)

    assert(halt.shouldHaltNow(pop) == True)
    print("Passed")


    print()
    print("HaltWhenNoChange")

    testProblem = LEAP.FunctionOptimization(testFunction)
    testDecoder = LEAP.Decoder(testProblem)
    halt = HaltWhenNoChange(10)
    for i in range(10):
        pop = [LEAP.Individual(testDecoder, [i])]
        assert(halt.shouldHaltNow(pop) == False)

    for i in range(9):
        assert(halt.shouldHaltNow(pop) == False)
    assert(halt.shouldHaltNow(pop) == True)

    print("Passed")
    


if __name__ == '__main__':
    unit_test()

