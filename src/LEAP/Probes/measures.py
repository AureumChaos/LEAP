#! /usr/bin/env python

"""
"""
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
import random
import string
import copy

#from LEAP.operators import PipelineOperator
#from LEAP.selection import DeterministicSelection
#from LEAP.survival import BaseMuLambdaSurvival

from math import *
from numpy import *


# It's possible some of these functions could be useful for something,
# although I kind of doubt it now.  I'll keep them around until I have a
# chance to decide for sure.

#############################################################################
#
# InitMeasureProbe
#
#############################################################################
#class InitMeasureProbe(PipelineOperator):
#    """
#    Initialize individuals so that information can be stored in them later.
#    """
#    parentsNeeded = 1
#
#    def apply(self, children):
#        for child in children:
#            child.q = []
#            child.z = 0
#        return children


#############################################################################
#
# PerformMeasureProbe
#
#############################################################################
#class PerformMeasureProbe(PipelineOperator):
#    """
#    Gather statistics for Price's equation and store them in the individuals.
#    """
#    parentsNeeded = 1
#    measureFunction = None
#
#    def __init__(self, provider, measureFunction):
#        PipelineOperator.__init__(self, provider)
#        self.measureFunction = measureFunction
#
#    def apply(self, children):
#        for child in children:
#            child.q.append(self.measureFunction(child))
#        return children


#############################################################################
#
# PriceRankOperator
#
#############################################################################
#class PriceRankOperator(BaseMuLambdaSurvival):
#    """
#    When using rankMeasure() as you measurement function, this should be
#    placed directly before PriceCalcOperator in the pipeline.
#    NOTE: Check to see if this really works!!
#          My recollection was that it's not finished yet.
#    """
#    def __init__(self, provider, popSize):
#        BaseMuLambdaSurvival.__init__(self, provider, popSize, popSize,
#                DeterministicSelection())
#
#    def combinePopulations(self, parents, children):
#        """
#        All we really want here is access to both the parent and child
#        populatations while in the pipeline.  This is why I've used the
#        BaseMuLambdaSurvival operator.  We will process what we need and
#        then just pass the child population along.
#        """
#        allRanks = [i.q[-1] for i in parents]
#        for j in range(len(parents[0].q)):
#            allRanks += [i.q[j] for i in children]
#
#        allRanks.sort()
#
#        for i in range(len(allRanks)):
#            allRanks[i][1] = i+1
#            if i !=0 and allRanks[i][0] == allRanks[i-1][0]:
#                allRanks[i][1] = allRanks[i-1][1]
#
#        return children


#############################################################################
#
# class BaseMeasure
#
#############################################################################
class BaseMeasure:
    def __init__(self, zeroVal):
        self.zeroVal = zeroVal

    def zero(self):
        return self.zeroVal

    def measure(self, ind):
        raise(NotImplementedError)

    def __call__(self, ind = None):
        """
        This method is deprecated.  I keep it here for backward compatibility
        only.  Don't use it.
        """
        if ind == None:
            sys.stderr.write("Measure(None) deprecated.  Use Measure.zero()\n")
            return self.zero()
        return self.measure(ind)



#############################################################################
#
# class FitnessMeasure
#
#############################################################################
class FitnessMeasure(BaseMeasure):
    def __init__(self):
        BaseMeasure.__init__(self, 0.0)  # Assume fitness is a single float

    def measure(self, ind):
        ind.fitness = None    # Make sure a new evaluation is performed(?)
        return ind.evaluate()



#############################################################################
#
# class ParameterMeasure
#
#############################################################################
class ParameterMeasure(BaseMeasure):
    """
    Measures the parameters (i.e. phenotype) for function optimzation type
    problems.

    This function needs to have access to a encoding when performing
    measurements in order to know how many parameters there are.  All
    individuals have a reference to a encoding, but when None is passed in to
    get the zero value, it has no such reference.  To solve this problem, I've
    added an internal state variable called 'encoding' which can be used in
    these situations.  But this means that phenotypeMeasure.encoding needs to
    be set before calling this function with a None.  There are two ways of
    doing this.  1) Call this function with a real individual, and the
    encoding will be saved, or 2) set phenotypeMeasure.encoding = <some
    encoding>.
    """
    def __init__(self, encoding):
        self.encoding = encoding
        phenome = encoding.decodeGenome(encoding.randomGenome())
        BaseMeasure.__init__(self, zeros(len(phenome)))

    def measure(self, ind):
        return array(self.encoding.decodeGenome(ind.genome))



#############################################################################
#
# class RankMeasure
#
#############################################################################
class RankMeasure(BaseMeasure):
    """
    The goal here is to rank all individuals in the parent population, child
    population and any intermediates populations based on their fitnesses.
    Of course we cannot do this until all the individuals have been measured,
    which means we're going to have to do some post-processing.  To facilitate
    this I return two values in a list.  The first is the fitness, and the
    second will be the rank.  The rank will have to be calculated later using
    the PriceRankOperator.
    """
    def __init__(self):
        BaseMeasure.__init__(self, array([0.0, 0]))

    def measure(self, ind):
        # This doesn't look done.
        return array([ind.evaluate(), 0])



#############################################################################
#
# class ExObjSampleMeasure
#
#############################################################################
class ExObjSampleMeasure(BaseMeasure):
    """
    Measure an ExecutableObjects responses to a series of sample inputs.

    IMPORTANT: For now I assume that the executable object in question gives
               only a single output.

    This function needs access to a set of samples in the following form:
       samples = [[input1, input2, ... inputN],
                  [input1, input2, ... inputN],
                   ...
                  [input1, input2, ... inputN]]

    They should be set in the following way, before any calls to this function
    are made:
       executableObjectSampleMeasure.sample = <your samples>
    """
    def __init__(self, samples):
        self.samples = samples
        BaseMeasure.__init__(self, zeros(len(self.samples)))

    def measure(self, ind):
        exObj = ind.encoding.decodeGenome(ind.genome)
        #return array([exObj.execute(sample) for sample in self.samples])
        return array([exObj.execute(sample)[0] for sample in self.samples])



#############################################################################
#
# unit_test()
#
#############################################################################
def unit_test():
    from LEAP.problem import sphereBounds
    from LEAP.problem import sphereFunction
    from LEAP.problem import sphereMaximize
    from LEAP.problem import FunctionOptimization
    from LEAP.encoding import FloatEncoding
    from LEAP.individual import Individual
    from LEAP.Domains.Concept.funcApprox import FunctionApproximation
    from LEAP.Exec.Pitt.PittEncoding import PittRuleEncoding
    from LEAP.Exec.Pitt.interpolatingRuleInterp import pyInterpolatingRuleInterp

    # Test FitnessMeasure
    bounds = sphereBounds
    function = sphereFunction
    problem = FunctionOptimization(function, sphereMaximize)
    encoding = FloatEncoding(problem, bounds, bounds)

    fitMeasure = FitnessMeasure()
    zero = fitMeasure.zero()
    print("fitMeasure.zero() =", zero)
    assert(zero == 0.0)

    genome = [1.0, 2.0, 3.0]
    ind = Individual(encoding, genome)
    measure = fitMeasure(ind)
    print("fitMeasure.measure(ind) =", measure)
    assert(measure == function(genome))
    print()


    # Test ParameterMeasure
    paramMeasure = ParameterMeasure(encoding)
    zero = paramMeasure.zero()
    print("paramMeasure.zero() =", zero)
    assert(len(zero) == len(bounds))
    assert(all(zero == array([0.0] * len(bounds))))

    measure = paramMeasure(ind)
    print("paramMeasure.measure(ind) =", measure)
    assert(len(measure) == len(genome))
    assert(all(measure == genome))
    print()


    # Test RankMeasure (doesn't work yet).


    # Test ExObjSampleMeasure
    condBounds = [ (0.0, 10.0) ]
    actBounds = [ (-1.0, 1.0) ] 
    allBounds = condBounds + actBounds
    targetFunc = lambda x:sin(x[0])
    problem = FunctionApproximation(targetFunc, condBounds)
    #numGroups = 2
    #numExamples = 20
    #problem.generateExampleGroups(numExamples, numGroups)
    #problem.selectTestSetGroup(0)

    minRules = 2
    maxRules = 10
    ruleEncoding = FloatEncoding(None, allBounds, allBounds)
    pittEncoding = PittRuleEncoding(problem, ruleEncoding, \
                    minRules, maxRules, len(condBounds), 1, \
                    ruleInterpClass = pyInterpolatingRuleInterp)

    numSamples = 10
    b = condBounds[0]
    samples = [[i * (b[1]-b[0]) / numSamples + b[0]] \
                   for i in range(numSamples+1)]
    print("samples =", samples)

    sampleMeasure = ExObjSampleMeasure(samples)
    zero = sampleMeasure.zero()
    print("sampleMeasure.zero() =", zero)
    assert(len(zero) == len(samples))
    assert(all(zero == array([0.0] * len(samples))))

    genome = [ [1.0, 1.0, 1.0], [9.0, 9.0, 9.0] ]
    ind = Individual(pittEncoding, genome)
    answer = [1,1,2,3,4,5,6,7,8,9,9]
    measure = sampleMeasure.measure(ind)
    print("sampleMeasure.measure(ind) =", measure)
    assert(len(measure) == len(answer))
    assert(all(measure == answer))


    print("Passed")
    return



if __name__ == '__main__':
    unit_test()


