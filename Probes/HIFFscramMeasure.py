#! /usr/bin/env python

import sys
import random
import string
import copy

import LEAP
#import scipy.stats

from math import *
from Numeric import *
#from numarray import *


#############################################################################
#
# HIFFlimit
#
#############################################################################
def HIFFlimit(phenome):
    """
    Richard Watson's Heirarchical if-and-only-if function.  This function was
    designed to be easy to solve using NPointCrossover.
    """
    l = len(phenome)
    if l == 1:
        return 1
    val = phenome.count(phenome[0])
    if not val == l:
        val = 0
    if val > 4:
        val = 0
    return val + HIFFfunction(phenome[:l/2]) + HIFFfunction(phenome[l/2:])



#############################################################################
#
# HIFFscrambleMeasure
#
#############################################################################
def HIFFscrambleMeasure(map, block = None):
    """
    Measure the "scrambledness" of an encoding of Richard Watson's
    Heirarchical if-and-only-if function.
    """
    l = len(map)

    if block == None:
        block = map
    b = len(block)
    if b == 1:
        return 0

    #print "block =", block

    # Calculate defining length as if the genome were circular
    sortedBlock = block[:]
    sortedBlock.sort()
    maxGap = sortedBlock[0] - sortedBlock[-1] + l
    #print "gap (",sortedBlock[-1],",",sortedBlock[0],") = ", maxGap
    for i in range(b-1):
        gap = sortedBlock[i+1] - sortedBlock[i]
        maxGap = max(maxGap, gap)
        #print "gap (",sortedBlock[i],",",sortedBlock[i+1],") = ", gap
    defLen = l-maxGap
    extra = defLen - b + 1

    #print "extra = ", extra

    return extra + HIFFscrambleMeasure(map, block[:b/2]) \
                 + HIFFscrambleMeasure(map, block[b/2:])



#############################################################################
#
# findScrambles
#
#############################################################################
def findScrambles(desiredMeasure, numGenomes = 1):
    length = 64
    HIFFproblem = LEAP.FunctionOptimization(LEAP.HIFFfunction)
    binEncoder = LEAP.BinaryEncoder(HIFFproblem, length)

    measure = -1
    numFound = 0
    while numFound < numGenomes:
        scrambledEncoder = LEAP.ScramblerEncoder(binEncoder)
        #scrambledEncoder = LEAP.PerturbingEncoder(binEncoder, 0.2, 32) # 1600
        #scrambledEncoder = LEAP.PerturbingEncoder(binEncoder, 0.2, 8) # 800
        #scrambledEncoder = LEAP.PerturbingEncoder(binEncoder, 0.2, 5) # 400
        #scrambledEncoder = LEAP.PerturbingEncoder(binEncoder, 0.2, 3) # 200
        #scrambledEncoder = LEAP.PerturbingEncoder(binEncoder, 0.2, 2) # 100

        genome = scrambledEncoder.randomGenome()
        map = scrambledEncoder.map
        measure = HIFFscrambleMeasure(map)
        #if measure > desiredMeasure:
        #    print measure,
        print measure,

        if measure == desiredMeasure:
            print
            print "map =", map
            print "measure = ", HIFFscrambleMeasure(map)
            numFound += 1



#############################################################################
#
# test_measure
#
#############################################################################
def test_measure(map, val):
    print "map =", map
    measure = HIFFscrambleMeasure(map)
    print "measure ==", measure, "==", val
    print
    return measure == val


#############################################################################
#
# unit_test
#
#############################################################################
def unit_test():
    passed = True

    map = range(8)
    passed = passed and test_measure(map, 0)

    map.reverse()
    passed = passed and test_measure(map, 0)

    map = [4,5,6,7,0,1,2,3]
    passed = passed and test_measure(map, 0)

    map = [0,2,1,3,4,5,6,7]
    passed = passed and test_measure(map, 2)

    map = [0,1,2,4,3,5,6,7]
    passed = passed and test_measure(map, 4)

    map = [1, 2, 7, 6, 3, 5, 0, 4]
    passed = passed and test_measure(map, 7)

    map = [0, 5, 2, 7, 4, 1, 6, 3]
    passed = passed and test_measure(map, 12)

    map = [6, 2, 0, 5, 4, 1, 3, 7]
    passed = passed and test_measure(map, 14)

    if passed:
        print "Passed"
    else:
        print "FAILED"


if __name__ == '__main__':
    #unit_test()
    findScrambles(1600)


