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

from LEAP.Exec.executableObject import ExecutableObject

# TODO: Do the C stuff in Cython now.
#import cRuleFuncs


#############################################################################
#
# RuleInterp
#
# Base class for rule interpreters.  There will be Python and C versions
# built on top of this also.  This class will handle the memory registers.
#
# ruleset:
#    [ [ c1 c1'  c2 c2' ... cn cn'  a1 a2 a3 ] [ rule2 ] ... [ ruleN ] ]
#
#############################################################################
class RuleInterp(ExecutableObject):
    """
    Rule interpreter for Pitt approach style rule learning.  
    Rules take the form: input_pairs, memory_pairs, output, memory
    """

    numPriorityMetrics = 3
    GENERALITY, PERIMETER, RULE_ORDER = range(numPriorityMetrics)

    def __init__(self, ruleset, numInputs, numOutputs, initMem=[],
                 priorityMetric = PERIMETER):
        raise(NotImplementedError)


    def execute(self, input):
        raise(NotImplementedError)



#############################################################################
#
# pyRuleInterp
#
#############################################################################
class pyRuleInterp(RuleInterp):
    """
    A python implementation of RuleInterp.
    """

    def __init__(self, ruleset, numInputs, numOutputs, initMem=[],
                 tapes=[], tapeHeads=[], priorityMetric=RuleInterp.PERIMETER):
        self.numInputs = numInputs
        self.numOutputs = numOutputs
        self.memRegs = initMem
        self.numMemory = len(initMem)

#        if len(tapes) != len(tapeHeads):
#            raise(ValueError, "Number of tapes and tape heads do not match.")
#
#        if tapes != []:
#            if tapeHeads == []:   # Need to set default values for tape heads
#                tapeHeads = [0] * len(tapes)
#            else:                 # Need to check validity of tape heads
#                for i in len(tapeHeads): 
#                    if tapeHeads[i] >= len(tapes[i]):
#                        raise(ValueError, "Invalid tape head value.")
#
#        self.tapes = tapes
#        self.tapeHeads = tapeHeads
#        self.numTapes = len(tapes)

        self.numConditions = self.numInputs + self.numMemory
        self.numActions = self.numOutputs + self.numMemory
        self.ruleset = ruleset
        self.rulePriorities = self.calcRulePriorities(ruleset,
                                                      metric=priorityMetric)

        #print("ruleset =", self.ruleset)
        #print("rulePriorities =", self.rulePriorities)


    def calcRulePriorities(self, ruleset, metric=RuleInterp.PERIMETER):
        """
        Calculates the amount of priority of each rule, allowing us to
        do conflict resolution.  The current metrics available are
        PERIMETER, GENERALITY, and RULE_ORDER.  Currently PERIMETER is
        the default just to maintain backwards compatability with some
        previous experiments.  GENERALITY would be a better choice for the
        default since this is the metric most commonly used.
        """
        if metric not in range(RuleInterp.numPriorityMetrics):
            raise(ValueError, "Illegal value for priorityMetric")

        priorities = []
        for r in range(len(ruleset)):
            rule = ruleset[r]
            if metric == RuleInterp.RULE_ORDER:
                priority = r
            elif metric == RuleInterp.GENERALITY:
                priority = 1
                for c in range(self.numConditions):
                    priority *= (rule[c*2] - rule[c*2+1])
                priority = abs(priority)
            elif metric == RuleInterp.PERIMETER:
                priority = 0
                for c in range(self.numConditions):
                    priority += abs(rule[c*2] - rule[c*2+1])
            priorities.append(priority)
        return priorities

        # I tried to optimize this function below, but it didn't make much
        # difference.
        #return [sum( [abs(rule[c] - rule[c+1])
        #              for c in range(0, self.numConditions * 2, 2)])
        #        for rule in ruleset]


    def execute(self, input):
        """
        Selects the appropriate rule and fires it.  The output is returned.
        Python version.
        """
        #print("input =", input)
        #print("memRegs =", self.memRegs)
        assert len(input) == self.numInputs
        allInput = input + self.memRegs
        bestMatchScore = -1
        matchList = []       # indices of rules

        # Build match list.  Find all rules that match the input.
        for r in range(len(self.ruleset)):
            rule = self.ruleset[r]
            matchScore = 0
            #print("rule #", r, "=", rule)
            for c in range(len(allInput)):   # just conditions
                #print("condition #", c, "=", (rule[c*2], rule[c*2+1]))
                #print("input #", c ,"=", allInput[c])
                diff1 = rule[c*2] - allInput[c]   # I should normalize this,
                diff2 = rule[c*2+1] - allInput[c] # especially if the possible
                                                  # ranges are very different.
                if diff1 * diff2 <= 0:      # Check sign
                    diff = 0                # Within the range
                else:
                    diff = min(abs(diff1), abs(diff2)) 
                matchScore += diff * diff  # Distance w/o sqrt
                #print("diff1, diff2, diff, matchScore =", \
                #       diff1, diff2, diff, matchScore)

            #print("matchScore =", matchScore)

            if matchList == [] or matchScore < bestMatchScore:
                bestMatchScore = matchScore
                matchList = [r]
            elif matchScore == bestMatchScore:
                matchList.append(r)

        #print("matchList =", matchList)

        # Conflict resolution
        # For exact matches, choose the rule(s) with the lowest generality.
        if bestMatchScore == 0:
            highestPriority = self.rulePriorities[matchList[0]]
            for i in matchList[1:]:
                highestPriority = min(highestPriority, self.rulePriorities[i])

            #print("highestPriority =", highestPriority)
            # Cull the matchList based on priority.
            i = 0
            while i < len(matchList):
                if self.rulePriorities[matchList[i]] > highestPriority:
                    del matchList[i]
                else:
                    i += 1

        #print("matchList =", matchList)

        # More conflict resolution
        # A common approach is to select the output which has the most
        # rules advocating it (i.e. vote).
        # A simpler approach is to just pick a rule randomly.
        # For now we'll just pick randomly.
        winner = random.choice(matchList)

        # "Fire" the rule.
        #print("Firing:", winner, self.ruleset[winner])
        if self.numMemory == 0:
            self.memRegs = []
            output = self.ruleset[winner][-self.numOutputs:]
        else:
            self.memRegs = self.ruleset[winner][-self.numMemory:]
            output = self.ruleset[winner][-self.numOutputs - self.numMemory : 
                                          -self.numMemory]

        # XXX Major hack here.  I did this to get the Pitt stuff working on
        # the two spirals problem, using adaptive real genes.  This is very
        # problem specific, and really shouldn't be here.  It might be more
        # appropriate to use a mixed genome with the ProxyMutation.  There
        # are still some issues there too though.  I need to figure out how
        # to properly address this problem.
        if output[0] >= 0.5:
            output = [1]
        else:
            output = [0]

        #print("output =", output)
        return output
        #return output, self.memRegs



#############################################################################
#
# cRuleInterp
#
#############################################################################
class cRuleInterp(RuleInterp):
    """
    A C implementation of RuleInterp.  Optimized for speed.
    """

    def __init__(self, ruleset, numInputs, numOutputs, initMem=[],
                 priorityMetric=RuleInterp.PERIMETER):
        if priorityMetric not in range(RuleInterp.numPriorityMetrics):
            raise(ValueError, "Illegal value for priorityMetric")

        #print(ruleset)
        # XXX Fix this once we work out the cython stuff
        #self.addr = cRuleFuncs.cInit(ruleset, numInputs, numOutputs, initMem,
        #                             priorityMetric)

        raise(NotImplementedError)


    def __del__(self):
        # XXX Fix this once we work out the cython stuff
        #cRuleFuncs.cDel(self.addr)
        return None


    def execute(self, input):
        # XXX Fix this once we work out the cython stuff
        #output = cRuleFuncs.cExecute(self.addr, input)
        #return output

        raise(NotImplementedError)



#############################################################################
#
# makeMap
#
#############################################################################
def makeMap(interp):
    f = open("map.dat", "w")
    res = 0.0025
    y = 0.0
    while y <= 1.0:
        x = 0.0
        while x <= 1.0:
            if interp.execute([x, y]) == [1]:
                f.write(str(x) + " " + str(y) + "\n")
            x += res
        y += res
    f.close()



#############################################################################
#
# test
#
#############################################################################
def test():
    """
    Test the rule interpreter.
    """
    ruleset = [[0,0, 0,0, 0, 0],
               [0,0, 1,1, 1, 1],
               [1,1, 0,0, 1, 1],
               [1,1, 1,1, 0, 0]]

    numInputs = 1
    numOutputs = 1
    initMem = [0]

    # Test the python version
    pyInterp = pyRuleInterp(ruleset, numInputs, numOutputs, initMem)

    output = pyInterp.execute([1])
    output += pyInterp.execute([1])
    output += pyInterp.execute([0])
    output += pyInterp.execute([1])

    print(" output =", output)
    print(" memRegs =", pyInterp.memRegs)
    assert(output == [1, 0, 0, 1])
    assert(pyInterp.memRegs == [1])
    passed = True

    # Test the C version
    # XXX Fix this once the cython stuff is worked out
    comment = """
    pyOutput = output
    cInterp = cRuleInterp(ruleset, numInputs, numOutputs, initMem)

    output = cInterp.execute([1])
    output += cInterp.execute([1])
    output += cInterp.execute([0])
    output += cInterp.execute([1])

    print(" output =", output)
    assert(output == pyOutput)


    #ruleset = [[0.0,0.6, 0.0,0.4, 0],
    #           [0.4,1.0, 0.6,1.0, 1]]
    ruleset = [[0.0,0.0, 0.0,0.0, 0.0],
               [1.0,1.0, 1.0,1.0, 1.0]]

    numInputs = 2
    numOutputs = 1

    interp = cRuleInterp(ruleset, numInputs, numOutputs)
    #assert(interp.execute([0.61, 0.0]) == [0])
    #assert(interp.execute([0.5, 0.6]) == [1])

    print("Writing map file...")
    makeMap(interp)
    """

    if passed:
        print("Passed")
    else:
        print("Failed!")


def test2():
    "I use this to check for memory leaks."
    ruleset = [[0,0, 0],
               [1,1, 1]]

    numInputs = 1
    numOutputs = 1
    initMem = []

    i = 0;
    max = 10000000
    modval = 100000 #max/100
    while i < max:
        if i % modval == 0:
            print(i,"   ", str(100*i/max)+"%")
        cInterp = cRuleInterp(ruleset, numInputs, numOutputs, initMem)
        output = cInterp.execute([1])
        del cInterp
        i += 1


if __name__ == '__main__':
    test()
    #test2()

