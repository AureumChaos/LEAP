#! /usr/bin/env python

# translatedEncoder.py
#
# When I started, this seemed really clever.  Now it's become really
# complicated.  It may be worthwhile to reconsider the design.


# Python 2 & 3 compatibility
from __future__ import print_function

import random
import math
import numpy

from LEAP.problem import Problem



##############################################################################
#
# class TranslatedProblem
#
#############################################################################
class TranslatedProblem(Problem):
    """
    This class acts as a wrapper around another problem, and it assumes that
    the phenome is a list of real valued numbers, such as for a function
    optimization problem.  The translationVector is subtracted from the
    phenome whenever evaluate is called.  This has the effect of translating
    the subProblem by the translationVector.
    """
    def __init__(self, subProblem, translationVector):
        self.subProblem = subProblem
        self.translationVector = translationVector

    def evaluate(self, phenome):
        assert len(phenome) == len(self.translationVector)
        transPhenome = [p - t for p, t in zip(phenome, self.translationVector)]
        return(self.subProblem.evaluate(transPhenome))

    def cmpFitness(self, fitness1, fitness2):
        return self.subProblem.cmpFitness(fitness1, fitness2)


############################################################################
#
# test
#
#############################################################################
def unit_test():
    from LEAP.problem import FunctionOptimization
    from LEAP.problem import sphereFunction
    from LEAP.problem import sphereMaximize
    from LEAP.problem import sphereBounds

    passed = True
    epsilon = 0.001

    sphereFunc = sphereFunction
    maximize = sphereMaximize
    bounds = sphereBounds
    numVars = len(bounds)
    sphereProb = FunctionOptimization(sphereFunc, maximize)
    
    transVec = [1.0] * numVars
    transProb = TranslatedProblem(sphereProb, transVec)

    phenome = [0.0] * numVars
    sphereFit = sphereProb.evaluate(phenome)
    transFit = transProb.evaluate(phenome)

    print("sphere(", phenome, ") = ", sphereFit)
    print("trans(", phenome, ") = ", transFit)
    assert(sphereFit != transFit)

    transFit = transProb.evaluate(transVec)
    print("trans(", transVec, ") = ", transFit)
    assert(sphereFit == transFit)

    if passed:
        print("Passed")
    else:
        print("FAILED")



if __name__ == '__main__':
    unit_test()


