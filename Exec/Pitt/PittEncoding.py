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
#sys.path.insert(0,'..')
import string
import copy
import random
import math

from LEAP.problem import Problem
from LEAP.encoding import Encoding
from LEAP.encoding import FloatEncoding
from LEAP.Exec.Pitt.ruleInterp import RuleInterp
from LEAP.Exec.Pitt.ruleInterp import pyRuleInterp
#from LEAP.Exec.Pitt.ruleInterp import cRuleInterp

# Define the default rule interpreter version
def getDefaultRuleInterp():
    #return cRuleInterp
    return pyRuleInterp

def getDefaultPriorityMetric():
    return RuleInterp.PERIMETER
    


#############################################################################
#
# PittEncoding
#
#############################################################################
class PittEncoding(Encoding):
    """
    A base class encoding for Pitt approach style rule sets.
    The randomGenome() should be defined by sub-classes.

    The PERIMETER priority metric is used as the default for conflict
    resolution in order to maintain backwards compatibility.  GENERALITY
    is the preferred choice among users though.  As of this writing, the
    GENERALITY metric doesn't work very well with binary representations.
    Check the RuleInterp class to see if this is still the case.
    """
    def __init__(self, problem, minRules, maxRules, numInputs, numOutputs, \
                 initMem = [], \
                 priorityMetric = None, \
                 ruleInterpClass = None):
        Encoding.__init__(self, problem)

        self.minRules = minRules
        self.maxRules = maxRules

        self.numInputs = numInputs
        self.numOutputs = numOutputs
        self.initMem = initMem

        if priorityMetric is None:
            self.priorityMetric = getDefaultPriorityMetric()
        else:
            self.priorityMetric = priorityMetric

        if ruleInterpClass is None:
            self.ruleInterpClass = getDefaultRuleInterp()
        else:
            self.ruleInterpClass = ruleInterpClass

        # XXX Do I need these?
        self.numConditions = numInputs + len(initMem)
        self.numActions = numOutputs + len(initMem)


    def decodeGenome(self, genome):
        floatGenome = []
        for rule in genome:
            newrule = [float(i) for i in rule]
            floatGenome.append(newrule)
        return self.ruleInterpClass(floatGenome, self.numInputs, \
                                    self.numOutputs, self.initMem, \
                                    self.priorityMetric)


    def randomGenome(self):
        raise(NotImplementedError)



#############################################################################
#
# PittRuleEncoding
#
#############################################################################
class PittRuleEncoding(PittEncoding):
    """
    Uses another encoding for decoding rules and generating random rules.
    """
    def __init__(self, problem, ruleEncoding, minRules, maxRules, \
                 numInputs, numOutputs, initMem=[], \
                 priorityMetric = None, \
                 ruleInterpClass = None):
        PittEncoding.__init__(self, problem, minRules, maxRules, numInputs,
                             numOutputs, initMem, priorityMetric,
                             ruleInterpClass)
        self.ruleEncoding = ruleEncoding


    def decodeGenome(self, genome):
        floatGenome = [self.ruleEncoding.decodeGenome(rule) for rule in genome]
        return self.ruleInterpClass(floatGenome, self.numInputs, \
                                    self.numOutputs, self.initMem, \
                                    self.priorityMetric)


    def randomGenome(self):
        numRules = random.randrange(self.maxRules - self.minRules + 1) \
                   + self.minRules
        genome = [self.ruleEncoding.randomGenome() for i in range(numRules)]
        return genome


    def fixupGenome(self, genome):
        genome = [self.ruleEncoding.fixupGenome(rule) for rule in genome]
        return genome



#############################################################################
#
# PittFixedEncoding
#
#############################################################################
class PittFixedEncoding(PittEncoding):
    """
    Uses another encoding to create a fixed lenth string for the genome.
    When decoding the genome, the fixed length string is transformed into
    a list of rules so that a rule interpreter can be created.
    """
    def __init__(self, problem, fixedEncoding, ruleSize, \
                 numInputs, numOutputs, initMem=[], \
                 priorityMetric = None, \
                 ruleInterpClass = None):
        PittEncoding.__init__(self, problem, 0, 0, numInputs, numOutputs, \
                             initMem, priorityMetric, ruleInterpClass)
        self.fixedEncoding = fixedEncoding
        self.ruleSize = ruleSize


    def decodeGenome(self, genome):
        g2 = self.fixedEncoding.decodeGenome(genome)
        ruleGenome = [g2[i:i+self.ruleSize] \
                      for i in range(0,len(g2),self.ruleSize)]
        return self.ruleInterpClass(ruleGenome, self.numInputs, \
                                    self.numOutputs,  self.initMem, \
                                    self.priorityMetric)


    def randomGenome(self):
        return self.fixedEncoding.randomGenome()


    def fixupGenome(self, genome):
        return self.fixedEncoding.fixupGenome()



#############################################################################
#
# PittNearestNeighborEncoding
#
#############################################################################
class PittNearestNeighborEncoding(PittRuleEncoding):
    """
    Uses a single point instead of a hyper-rectangle to represent a rule.  The
    standard rule interpreter should work fine as long as both corners are the
    same for all hyper-rectangles.
    """
    def pointRule2boxRule(self, rule):
        """
        Converts a rule where the condition is defined as a point to one where
        the condition is defined as a box.
        """
        newRule = rule[0:0]  # maintain type
        for i in range(0, self.numInputs):
            newRule += rule[i:i+1]   # maintain type
            newRule += rule[i:i+1]
        newRule += rule[self.numInputs:]
        return newRule


    def decodeGenome(self, genome):
        floatGenome = \
                  [self.pointRule2boxRule(self.ruleEncoding.decodeGenome(rule)) 
                   for rule in genome]
        return self.ruleInterpClass(floatGenome, self.numInputs, \
                                    self.numOutputs, self.initMem, \
                                    self.priorityMetric)



#############################################################################
#
# unit_test
#
#############################################################################
class MyProblem(Problem):
    """
    Essentially a classification problem.  There are two inputs, each ranging
    from 0.0 to 1.0.  Any input in the lower left half of the space will have
    a classification of 0.0.  The upper right half has a classification of
    1.0.
    
    The phenome is evaluated using a number of input points.  The fitness is
    the ratio of the number of correct responses to the total set of inputs.
    """
    def evaluate(self, phenome):
        fitness = 0
        total = 0
        for input1 in [i*0.1 for i in range(11)]:   # loop 0.0 to 1.0 by 0.1
            for input2 in [i*0.1+.05 for i in range(10)]:  # Avoid x+y=1
                answer = [((input1 + input2) >= 1.0) * 1.0]
                output = phenome.execute([input1, input2])
                total += 1
                if answer == output:
                    fitness += 1

        fitness = 1.0 * fitness / total
        return fitness



def unit_test():
    """
    Test the rule interpreter.
    """
    initRanges = [(0.0, 1.0)] * 3
    bounds = None
    ruleEncoding = FloatEncoding(None, initRanges, bounds)

    encoding = PittRuleEncoding(None, ruleEncoding, 10, 10, 1, 1)
    genome = encoding.randomGenome()

    assert(len(genome) == 10)
    assert(len(genome[0]) == 3)

    # Test nearest-neighbor pitt encoding
    numInputs = 2
    numOutputs = 1
    bounds = [(0.0, 1.0)] * (numInputs + numOutputs)
    initRanges = bounds
    ruleEncoding = FloatEncoding(None, initRanges, bounds)

    encoding = PittNearestNeighborEncoding(MyProblem, ruleEncoding, 2, 2,
                                         numInputs, numOutputs)
    genome = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
    phenome = encoding.decodeGenome(genome)

    myProblem = MyProblem()
    fitness = myProblem.evaluate(phenome);
    print("fitness =", fitness)
    assert(fitness == 1.0)

    print("Passed")



if __name__ == '__main__':
    unit_test()

