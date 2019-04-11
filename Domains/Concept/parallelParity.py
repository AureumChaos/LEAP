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

import random
import copy
import math

from LEAP.Exec.Pitt.ruleInterp import pyRuleInterp
from LEAP.problem import Problem
from LEAP.encoding import int2bin


def int2base(value, digits, strlen=None):
    "Convert an integer into a string of another base."
    bstr = ''
    base = len(digits)
    if strlen == None:
        if value == 0:
            order = 1
        else:
            order = int(math.log(value) / math.log(base)) + 1
    else:
        order = strlen

    for i in range(order-1,-1,-1):
        dval = int(value / base ** i)
        bstr += str(digits[dval])
        value -= dval * base ** i
    return bstr


#############################################################################
#
# ParallelParity
#
# This code is very old, and I don't even know how it works anymore.
# It should probably be removed.
#
#############################################################################
class ParallelParity(Problem):
    """
    Given a binary string of size n, the rule-set should determine the parity
    of the string.
    """
    def __init__(self, numBits, numTests = 10, digits=[0,1]):
        #Problem.__init__(self)
        self.digits = digits
        self.numBits = numBits
        self.numInputs = numBits
        self.numOutputs = 1
        self.initMemory = []
        self.numMemory = len(self.initMemory)
        self.numConditions = self.numInputs + self.numMemory      # 2 + 1 = 3
        self.numActions = self.numOutputs + self.numMemory        # 1 + 1 = 2
        self.ruleLength = self.numConditions + self.numActions    # 3 + 2 = 5

        self.numTests = 2 ** self.numInputs # numTests
        self.testvals = []
        self.generateTests()


    def generateTests(self):
        self.testvals = []
        for i in range(self.numTests):
            #self.testvals.append(random.randrange(2**15))
            teststr = int2bin(i, self.numInputs)
            self.testvals.append([int(c) for c in teststr])
    

    def episode(self, phenome, value):
        parity = 1 - value.count(1) % 2   # even parity

        # Initialize the rule interpreter
        interp = pyRuleInterp(phenome, self.numInputs,
                                             self.numOutputs, self.initMemory)
        
	# XXX Make it handle hex
        fitness = 0
        #for i in range (maxlen-1, -1, -1):
        #    input = [int(base_num1[i]), int(base_num2[i])]
        #    output = interp.executeStep(input)
        #    fitness += abs(output[0] - int(base_num3[i]))

        output = interp.executeStep(value)

        return int(output[0] == parity)


    def evaluate(self, phenome):
        fitness = 0
        for i in range(self.numTests):
            fitness += self.episode(phenome, self.testvals[i])
        #fitness = float(fitness) / self.numTests
        return fitness


    def testGeneralization(self, phenome):
        "Test the phenome against inputs it has not encountered."
        numTests = 100
        successes = 0
        for i in range(numTests):
            testnum = random.randrange(2**15)
            teststr = int2bin(testnum, self.numBits)
            testinput = [int(c) for c in teststr]
            successes += self.episode(phenome, testinput)

        return float(successes) / numTests


    def cmpFitness(self, fitness1, fitness2):
        """
        Returns 1 if fitness1 is better than fitness2
                0 if fitness1 = fitness2
               -1 if fitness1 is worse than fitness2
        Better than means '>' if maximizing or '<' if minimizing.
        """
        return cmp(fitness1, fitness2)   # Maximize



#############################################################################
#
# test
#
#############################################################################
def test():
    from LEAP.individual import Individual
    from LEAP.encoding import BinaryEncoding

    problem = ParallelParity(3)
    encoding = BinaryEncoding(problem, 7)
    genome = [[0,1, 0,1, 0,1, 0]]

    ind = Individual(encoding, genome)
    #ind2 = Individual(encoding)
    fitness = ind.evaluate()
    print("fitness =", fitness)

    if fitness == 4:
        print("Passed")
    else:
        print("FAILED")


if __name__ == '__main__':
    test()


