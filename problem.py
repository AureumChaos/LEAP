#! /usr/bin/env python

# problem.py
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
def cmp(a,b):
    return int(a>b) - int(a<b)

import sys
import string
import random
import math


#############################################################################
#
# class Problem
#
#############################################################################
class Problem:
    "Base class for problem landscapes"
    def evaluate(self, phenome):
        "Evaluates (calculates the fitness of) a phenome."
        raise NotImplementedError


    # I chose not to override __cmp__ because I thought that might get
    # confusing.  For example, for a problem where one is minimizing, you
    # would need to use a > b to find out if a has a lower fitness than b.
    def cmpFitness(self, fitness1, fitness2):
        """
        Compare two fitnesses and determine if C{fitness1} is "better than"
        "equal to" or "worse than" C{fitness2}.
        Better than means '>' if maximizing or '<' if minimizing.

        @param fitness1: The first fitness value to be compared.
        @param fitness2: The second fitness value to be compared.

        @return:
            1 if fitness1 is "better than" fitness2
            0 if fitness1 = fitness2
            -1 if fitness1 is worse than fitness2
        """
        return cmp(fitness1, fitness2)   # Maximize by default


#############################################################################
#
# FunctionOptimization
#
#############################################################################
class FunctionOptimization(Problem):
    """
    Real valued function optimization problem.  The class is given a pointer
    to a real valued function which is to be optimized.
    """
    def __init__(self, objectiveFunction, maximize = True):
        """
        @param objectiveFunction: A pointer to a real valued function.  The function
                          takes a list of numbers as its only parameter.
        @param maximize: Should the function be maximized or not?
                         Default = True.
        """
        self.objectiveFunction = objectiveFunction
        self.maximize = maximize

    def cmpFitness(self, fitness1, fitness2):
        """
        Determine which fitness is better depending on whether we are
        maximizing or minimizing.

        @param fitness1: The first fitness value to be compared.
        @param fitness2: The second fitness value to be compared.

        @return:
            1 if fitness1 is "better than" fitness2
            0 if fitness1 = fitness2
            -1 if fitness1 is worse than fitness2
        """
        if self.maximize:
            return cmp(fitness1, fitness2)
        else:
            return -cmp(fitness1, fitness2)

    def evaluate(self, phenome):
        """
        @param phenome: The input parameters to the function.
        @return: The fitness.
        """
        fitness = self.objectiveFunction(phenome)
        return fitness


#############################################################################
#
# Some classic test problems.
#
#############################################################################
def product(list):
    return reduce(lambda x,y:x*y, list)

sphereMaximize = False
sphereBounds = [(-5.12, 5.12)] * 3
def sphereFunction(phenome):
    return sum([x**2 for x in phenome])

invSphereMaximize = True
invSphereBounds = [(-5.12, 5.12)] * 3
def invSphereFunction(phenome):
    fit = 0.0
    for i in range(len(phenome)):
        fit += invSphereBounds[0][0]**2 - phenome[i]**2
    return fit

# Only 2-dimensional now.  Can it be generalized to higher dimensionality?
rosenbrockMaximize = False
rosenbrockBounds = [(-2.048, 2.048)] * 2
def rosenbrockFunction(phenome):
    return 100 * (phenome[0] - phenome[1])**2 + (1 - phenome[1])**2

schwefelMaximize = False
schwefelBounds = [(-500.0, 500.0)] * 20
def schwefelFunction(phenome):
    return sum([x * math.sin(math.sqrt(abs(x))) for x in phenome])

rastriginMaximize = False
rastriginBounds = [(-5.12, 5.12)] * 20
def rastriginFunction(phenome):
    return sum([x**2 - 10*math.cos(2 * math.pi * x) + 10 for x in phenome])

griewangkMaximize = False
griewangkBounds = [(-600.0, 600.0)] * 10
def griewangkFunction(phenome):
    return sum([x**2/4000 for x in phenome]) + \
           product([math.cos(x)/math.sqrt(i) for x in phenome]) + 1

ackleyMaximize = False
ackleyBounds = [(-30.0, 30.0)] * 30
def ackleyFunction(phenome):
    return -20 * exp(-0.2 * math.sqrt(1/len(phenome) * \
                                      sum([x**2 for x in phenome]))) - \
           exp(1/len(phenome) * sum([math.cos(2*math.pi*x) for x in phenome])) \
           + 20 + math.e

# From Box (1996)
# Minima is 0 at x*=(1,10,1)  and  x*=(10,1,-1)
# See (Schwefel, 1995) EVOLUTION AND OPTIMUM SEEKING, p. 332
boxMaximize = False
boxBounds = [(0,20)]*3
def boxFunction(phenome):
    total = 0.0
    for j in range(1,11):
        total = total + \
                  ( math.exp(-0.1*j*phenome[0]) - \
                    math.exp(-0.1*j*phenome[1]) - \
                    phenome[2] * (math.exp(-0.1*j) - math.exp(-j)) )**2
    return(total)


import numpy

axisValleyMaximize = False
axisValleyBounds = [(0,20)]*2
def axisValleyFunction(phenome):
    return abs(phenome[0]) + 10 * math.sqrt(sum([x * x for x in phenome[1:]]))

# A valley that is diagonal between axes 1 and 2
diagValleyMaximize = False
diagValleyBounds = [(0,20)]*2
def diagValleyFunction(phenome):
    #a = abs((phenome[0] + phenome[1]) / 2)
    #b = (phenome[0] - phenome[1])**2 / 2)
    return abs((phenome[0] + phenome[1]) / 2) + \
               10 * math.sqrt( ((phenome[0] - phenome[1])**2 / 2) + \
                                sum([x * x for x in phenome[2:]]) )

valleyMaximize = False
valleyBounds = [(0,20)]*2
def valleyFunction(phenome, direction = None):   # This should be a functor
    if direction != None:
        l = math.sqrt(sum([float(i*i) for i in direction]))
        valleyFunction.direction = numpy.array(direction) / l

    if not hasattr(valleyFunction, "direction"):
        # set a default direction of [1, 1, ... 1]
        d = [1.0] * len(phenome)
        l = math.sqrt(sum([i*i for i in d]))
        valleyFunction.direction = numpy.array(d) / l

    V = valleyFunction.direction
    X = numpy.array(phenome)
 
    # Calculate distance from the origin to the point in the valley that is
    # closest to p.
    r = numpy.dot(X,V) / numpy.dot(V,V);  # distance from origin along valley

    # Calculate distance from point X to line V
    B = X - r * V;
    b = numpy.sqrt(numpy.sum(numpy.dot(B, B))) # dist to bottom of valley

    return abs(r) + b * 10


class valleyFunctor:
    def __init__(self, direction):
        # Normalize the direction vector
        l = math.sqrt(sum([i*i for i in direction]))
        self.direction = numpy.array(direction) / l
        self.VdotV = numpy.dot(self.direction, self.direction)

    def __call__(self, phenome):
        V = self.direction
        X = numpy.array(phenome)
 
        # Calculate distance from the origin to the point in the valley that
        # is closest to p.
        r = numpy.dot(X,V) / self.VdotV;  # distance from origin along valley

        # Calculate distance from point X to line V
        B = X - r * V;
        b = numpy.sqrt(numpy.sum(numpy.dot(B, B))) # dist to bottom of valley

        return abs(r) + b * 10


# I'm not sure what valley2 is.  It looks like I created a valley that is
# quadratic instead of linear.  This would make the bottom of the valley
# smoother, and perhaps easier to navigate.
valley2Maximize = False
valley2Bounds = [(-5.12,5.12)]*2
class valley2Functor(valleyFunctor):  # Not sure why I inherit
    def __init__(self, direction, valleyFactor=0.1, wallFactor=1):
        self.direction = numpy.array(direction)
        self.valleyFactor = valleyFactor
        self.wallFactor = wallFactor  # 
        self.VdotV = numpy.dot(self.direction, self.direction)

    def __call__(self, phenome):
        V = self.direction
        X = numpy.array(phenome)
 
        # Calculate distance from the origin to the point in the valley that
        # is closest to p.
        r = numpy.dot(X,V) / self.VdotV;  # distance from origin along
                                              # valley

        # Calculate distance from point X to line V
        B = X - r * V;
        bSquared = numpy.sum(numpy.dot(B, B))  # dist to bottom of valley

        return self.valleyFactor * r*r + self.wallFactor * bSquared


GaussianMaximize = True
GaussianBounds = [(-3.0,3.0)]*2
class GaussianFunctor(valleyFunctor):
    def __init__(self, sigmas=[1.0, 1.0], mus=None):
        """
        A multivariate gaussian fitness function.
        sigmas = standard deviations
        mus = means (i.e. the coordinate of the optimimum)
        """
        self.numDimensions = len(sigmas)
        self.sigmas = sigmas
        if mus == None:
            self.mus = [0.0] * self.numDimensions
        else:
            assert(len(mus) == len(sigmas))
            self.mus = mus


    def __call__(self, phenome):
        V = self.direction
        X = numpy.array(phenome)
 
        # Calculate distance from the origin to the point in the valley that
        # is closest to p.
        r = numpy.dot(X,V) / numpy.dot(V,V);  # distance from origin along
                                              # valley

        # Calculate distance from point X to line V
        B = X - r * V;
        b = numpy.sqrt(numpy.sum(numpy.dot(B, B))) # dist to bottom of valley

        return self.valleyFactor * r*r + self.wallFactor * b*b


#############################################################################
#
# OneMax
#
#############################################################################
def oneMaxFunction (phenome):
    """
    The standard max-ones function.  The genome is assumed to be either
    a string or list of '1's and '0's.  The fitness is equal to the number
    of ones in the genome.
    """
    print(phenome)
    return phenome.count('1')



#############################################################################
#
# HIFF
#
#############################################################################
def HIFFfunction (phenome):
    """
    Richard Watson's Heirarchical if-and-only-if function.  This function was
    designed to be easy to solve using NPointCrossover.
    """
    l = len(phenome)
    if l == 1:
        return 1
    val = phenome.count(phenome[0])
    val = val * int(val == l)
    return float(val + HIFFfunction(phenome[:l/2]) \
                     + HIFFfunction(phenome[l/2:]))



#############################################################################
#
# unit_test
#
#############################################################################
def landscape(phenome):
    return sum(phenome)

def unit_test():
    #HIFF = FunctionOptimization(HIFFfunction)
    #print(HIFF.evaluate("1"*64))

    valleyFunction([1.0, 1.0, 1.0], [1.0, 1.0, 1.0])
    valleyFunction([0.0, 0.0, 0.0])

    valleyFunction([1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    valleyFunction([1.0, 1.0, 0.0])

    passed = True

    val1a = valleyFunctor([1.0, 1.0, 1.0])
    val1b = valleyFunctor([2.0, 2.0, 2.0])
    fit1 = val1a([1.0, 1.0, 1.0])
    fit2 = val1b([1.0, 1.0, 1.0])

    print("val1a([1,1,1]) == val1b([1,1,1]) : ", fit1, "==", fit2)
    passed = passed and (fit1 == fit2)

    oneMax = FunctionOptimization(oneMaxFunction)
    zeroPhenome = "0000000000"
    fourPhenome = "1000100110"
    tenPhenome = "1111111111"

    zeroEval = oneMax.evaluate(zeroPhenome)
    fourEval = oneMax.evaluate(fourPhenome)
    tenEval = oneMax.evaluate(tenPhenome)

    print("oneMax.evaluate('" + zeroPhenome +"') =", zeroEval, "== 0")
    print("oneMax.evaluate('" + fourPhenome +"') =", fourEval, "== 4")
    print("oneMax.evaluate('" + tenPhenome +"') =", tenEval, "== 10")

    passed = passed and (zeroEval == 0) and (fourEval == 4) and (tenEval == 10)

    funcOpt = FunctionOptimization(landscape)
    phenome1 = [0, 0]
    phenome2 = [10, 10]
    fit1 = funcOpt.evaluate(phenome1)
    fit2 = funcOpt.evaluate(phenome2)

    print("fit1 =", fit1, "== 0")
    print("fit2 =", fit2, "== 20")

    passed = passed and (fit1 == 0) and (fit2 == 20)

    if passed:
        print("Passed")
    else:
        print("FAILED")


if __name__ == '__main__':
    unit_test()

