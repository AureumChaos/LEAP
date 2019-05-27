#! /usr/bin/env python

# survival.py
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
import copy    # for Clone
import random

#from individual import *
#from operators import *
#from selection import *
import LEAP


#############################################################################
#
# class SurvivalOperator
#
# A survival operator stands at the back of the pipeline, and attaches
# to a genetic operator.
#
#############################################################################
class SurvivalOperator(LEAP.PipelineOperator):
    "Base class for survival selection operators"

    def reinitialize(self, population):
        self.originalPopulation = population
        LEAP.PipelineOperator.reinitialize(self, population)
        
    def isAnythingCached(self, after = None):
        sys.stderr.write("Warning: Calls to isAnythingCached() on a " +
                         "SurvivalOperator may not be useful.\n")
        return LEAP.PipelineOperator.isAnythingCached(self, after)
        

#############################################################################
#
# class ElitismSurvival
#
#############################################################################
class ElitismSurvival(SurvivalOperator):
    "Base class for survival selection operators"
    parentsNeeded = 1

    def __init__(self, provider, numElite = 1):
        SurvivalOperator.__init__(self, provider)
        self.numElite = numElite
        self.cloner = LEAP.CloneOperator(None)

    def reinitialize(self, population):
        # Currently I am sorting the population.  This can be very expensive.
        # I should probably use a more efficient approach.
        parents = population[:]  # just copy the pointers
        parents.sort(key=functools.cmp_to_key(cmpInd))
        elite = self.cloner.apply(parents[-self.numElite:])
        SurvivalOperator.reinitialize(self, population)
        self.addToCache(elite)   # Dump the elite into the cache

    def apply(self, individuals):
        return individuals


#############################################################################
#
# class BaseMuLambdaSurvival
#
#############################################################################
class BaseMuLambdaSurvival(SurvivalOperator):
    "Base class for ES (mu,lambda) and (mu+lambda)"
    def __init__(self, provider, mu, Lambda, selectionOp = None):
        SurvivalOperator.__init__(self, provider)
        self.mu = mu
        self.parentsNeeded = Lambda
        self.selectionOp = selectionOp
        if selectionOp == None:
            self.selectionOp = LEAP.TruncationSelection(mu)

    def combinePopulations(self, parents, children):
        raise NotImplementedError  # Subclasses should redefine this

    def apply(self, children):
        for child in children:
            child.evaluate()

        all = self.combinePopulations(self.originalPopulation, children)
        self.selectionOp.reinitialize(all)

        selected = []
        for i in range(self.mu):
            selected.append(self.selectionOp.pull())
        return selected

    def reinitialize(self, population):   # Is this needed?
        SurvivalOperator.reinitialize(self, population)



#############################################################################
#
# class MuCommaLambdaSurvival
#
#############################################################################
class MuCommaLambdaSurvival(BaseMuLambdaSurvival):
    "ES style (mu , lambda) survival selection."
    name = "MuCommaLambdaSurvival"

    def combinePopulations(self, parents, children):
        return children



#############################################################################
#
# class MuPlusLambdaSurvival
#
#############################################################################
class MuPlusLambdaSurvival(BaseMuLambdaSurvival):
    "ES style (mu + lambda) survival selection."
    name = "MuPlusLambdaSurvival"
    cloner = None

    def __init__(self, provider, mu, Lambda):
        self.cloner = LEAP.CloneOperator(None)
        BaseMuLambdaSurvival.__init__(self, provider, mu, Lambda)

    def combinePopulations(self, parents, children):
        # Parents must be cloned (I think?)
        clonedParents = self.cloner.apply(parents)
        return clonedParents + children



#############################################################################
#
# unit_test
#
#############################################################################
def unit_test():
    passed = False

    a = MuCommaLambdaSurvival(None,1,10)
    print("There are no tests currently")
    #if passed:
    #    print("Passed")
    #else:
    #    print("FAILED")


if __name__ == '__main__':
    unit_test()

