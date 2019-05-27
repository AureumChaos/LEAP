#! /usr/bin/env python

# selection.py
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
import functools

#from individual import *
#from operators import *
import LEAP


#############################################################################
#
# class SelectionOperator
#
#############################################################################
class SelectionOperator(LEAP.Operator):
    """
    Base class for selection operators.  A selection operator stands at the
    front of the pipeline, and attaches to a parent population.
    """
    parentPopulation = []

    def __init__(self, parentPopulation = []):
        LEAP.Operator.__init__(self)
        if parentPopulation != []:
            self.reinitialize(parentPopulation)

    def reinitialize(self, parentPopulation):
        self.parentPopulation = parentPopulation
        for i in range(len(parentPopulation)):
            parentPopulation[i].popIndex = i
        LEAP.Operator.reinitialize(self, parentPopulation)
        
    def applyAndCache(self):
        """
        Assembles the appropriate list of "parents" and calls apply.
        For selection, the entire parent population is appropriate.
        """
        parents = self.apply(self.parentPopulation)
        self.addToCache(parents)

    def isAnythingCached(self, after = None):
        """
        This is the base case, so we always return False, whether a selection
        operator uses the cache or not.
        """
        if after:
           sys.stderr.write(
                       "Warning: Cache query reached the selection operator\n")
        return False


#############################################################################
#
# class RouletteWheelSelection
#
#############################################################################
class RouletteWheelSelection(SelectionOperator):
    """
    Select individuals from the population based on a probability that is
    assigned to each.  These probabilities are usually related to the fitness
    of individuals somehow, but the details will be defined by sub-classes.

    The roulette wheel approach is usually associated with
    fitness-proportional selection, but this is not the only form of selection
    that can use a roulette wheel.  If one can assign a probability to
    selecting and individual then the roulette wheel can be used.

    This operator also implements Baker's Stochastic Uniform Sampling (SUS)
    approach, but in order to do this it must be told how many individuals
    will be selected when the pipeline is created.  Otherwise it will just
    use the standard roulette wheel approach.
    """
    def __init__(self, parentPopulation = [], SUS_numSelections = None):
        self.rouletteSum = 0.0
        self.SUS_numSelections = SUS_numSelections
        SelectionOperator.__init__(self, parentPopulation)


    def buildRouletteWheel(self, parentPopulation):
        """
        Create list containing roulette wheel values corresponding to
        individual in the parentPopulation.  Each wheel value defines
        the size of a single individuals portion of the wheel.
        """
        raise NotImplementedError  # Must be implemented in subclasses.


    def SUSselectAll(self, parentPopulation, numSelections, cumWheel):
        """
        Select the appropriate number of individuals and place them in the
        cache so that the pull() mechanism will retrieve them automatically.
        """
        increment = float(cumWheel[-1]) / numSelections
        wheelPos = random.random() * increment
        selections = []

        for ind,val in zip(parentPopulation, cumWheel):
            while val > wheelPos:
                selections.append(ind)
                wheelPos += increment

        # It's important to perform all the SUS selections up front so that
        # they can be shuffles.  Otherwise consecutive selections will often
        # return the same individual.
        random.shuffle(selections)

        return selections


    def reinitialize(self, parentPopulation):
        SelectionOperator.reinitialize(self, parentPopulation)
        wheel = self.buildRouletteWheel(parentPopulation)

        self.cumWheel = []
        currentSum = 0.0
        for val in wheel:
            currentSum += val
            self.cumWheel.append(currentSum)
        self.noSelectionsYet = True

        
    def apply(self, parents):
        """
        Select using the roulette wheel.
        """
        if self.SUS_numSelections:
            if self.noSelectionsYet:
                self.noSelectionsYet = False
            else:
                # If we reach this point then the SUS_numSelections parameter
                # underestimated how many selections there really would be.
                # There is the potential for introducing a lot more drift now.
                print("Warning: SUS_numSelections is too small")

            return self.SUSselectAll(parents, self.SUS_numSelections,
                                     self.cumWheel)

        wheelPos = random.random() * self.cumWheel[-1]
        for ind,val in zip(parents, self.cumWheel):
            if val > wheelPos:
                return [ind]

        raise NameError("Fell off the end of the roulette wheel")


#############################################################################
#
# class ProportionalSelection
#
#############################################################################
class ProportionalSelection(RouletteWheelSelection):
    """
    Individuals with higher fitness have a higher probability of being
    of being selected.  Roulette-wheel method is used.
    """
    def buildRouletteWheel(self, parentPopulation):
        """
        Create list containing roulette wheel values corresponding to
        individual in the parentPopulation.  Each wheel value defines
        the size of a single individuals portion of the wheel.
        """
        return [p.getFitness() for p in parentPopulation]



#############################################################################
#
# class RankSelection
#
#############################################################################
class RankSelection(RouletteWheelSelection):
    "Proportional selection using rank instead of fitness"

    def __init__(self, parentPopulation = [], SUS_numSelections = None):
        RouletteWheelSelection.__init__(self, parentPopulation,
                                        SUS_numSelections = SUS_numSelections)
        # Currently this constructor is not needed, but I should add a
        # parameter so that the user can change the selection pressure by
        # changing the slope of the line that defines the probabilities.


    def buildRouletteWheel(self, parentPopulation):
        """
        Create list containing roulette wheel values corresponding to
        individual in the parentPopulation.  Each wheel value defines
        the size of a single individuals portion of the wheel.
        """
        # We don't want to sort the original list.  Note that only the list is
        # copied though.  No new individuals are created.
        #print("Calling sort")
        sortedPop = parentPopulation[:]
        sortedPop.sort(key=functools.cmp_to_key(LEAP.cmpInd))

        m = float(len(parentPopulation))
        epsilon = 1 / m**2      # Same as tournament selection

        sortedPop[0].rank = epsilon
        firstInstance = 0
        for i in range(1, len(sortedPop)):
            # I got this equation from Ken's book
            sortedPop[i].rank = (2/m - epsilon) - (2/m - 2.0 * epsilon) * \
                                ((m-i-1) / (m-1))
            if LEAP.cmpInd(sortedPop[i], sortedPop[i-1]) == 1:  # fitness improved
                avgRank = sum([x.rank for x in sortedPop[firstInstance:i]]) / \
                          (i - firstInstance)
                for j in range(firstInstance, i):
                    sortedPop[j].rank = avgRank
                firstInstance = i

        return [p.rank for p in parentPopulation]
        


#############################################################################
#
# class UniformSelection
#
#############################################################################
class UniformSelection(SelectionOperator):
    """
    Each parent is selected with equal probability from the parent population.
    Useful for ES style EAs, where survival selection is used.
    """
    # This could be implemented using the RouletteWheelSelection operator,
    # but it seems a little wasteful.
    def apply(self, parents):
        selected = random.choice(parents)
        return [selected]


#############################################################################
#
# class TournamentSelection
#
#############################################################################
class TournamentSelection(SelectionOperator):
    "Picks the best individual from a pool of n individuals"
    tournament_size = 2
    compareFunc = LEAP.cmpInd
    name = "TournamentSelection"

    def __init__(self, tournament_size = 2, parentPopulation = [],
                 compareFunc = LEAP.cmpInd):
        SelectionOperator.__init__(self, parentPopulation)
        self.tournament_size = tournament_size
        self.compareFunc = compareFunc

    def apply(self, parents):
        tournament = []
        for i in range(self.tournament_size):
            tournament.append(random.choice(parents))
        winner = LEAP.fittest(tournament)
        #print([ind.getFitness() for ind in tournament], "-->",)
        #print(winner.getFitness())
        return [winner]


#############################################################################
#
# class DeterministicSelection
#
#############################################################################
class DeterministicSelection(SelectionOperator):
    """
    Each parent is selected in sequence from the parent population.
    Useful for ES style EAs, where survival selection is used.
    """
    popPosition = 0

    def reinitialize(self, parentPopulation):
        self.popPosition = 0

        # The population may be sorted by fitness, so mix things up a
        # bit before grinding through each parent.

        random.shuffle(parentPopulation)

        SelectionOperator.reinitialize(self, parentPopulation)
        
    def apply(self, parents):
        selected = parents[self.popPosition]

        self.popPosition += 1

        if self.popPosition == len(parents) :

            # if we exhaust the current list of parents, reshuffle
            # them before restarting the selection sequence to ensure
            # we get a different set of pairings.
            random.shuffle(parents)

            # We've exhausted the current set of parents so reset to
            # the first one
            self.popPosition = 0

        return [selected]


#############################################################################
#
# class TruncationSelection
#
#############################################################################
class TruncationSelection(DeterministicSelection):
#class TruncationSelection(UniformSelection):
    """
    The parent population is sorted, and only the top N (numToKeep) are
    returned.  The rest are "truncated" or thrown away.
    Note: Because this inherits from DeterministicSelection, if more than N
          individuals a requested, the top N individuals are returned again
          in a different sequence than they were the first time.
    """
    def __init__(self, numToKeep, parentPopulation = []):
        DeterministicSelection.__init__(self, [])
        #UniformSelection.__init__(self, parentPopulation)
        self.numToKeep = numToKeep
        self.reinitialize(parentPopulation)

    def reinitialize(self, parentPopulation):
        truncatedParentPop = parentPopulation[:]  # just copy the pointers
        truncatedParentPop.sort(key=functools.cmp_to_key(LEAP.cmpInd))
        truncatedParentPop = truncatedParentPop[-self.numToKeep:]
        #UniformSelection.reinitialize(self, trunctatedParentPop)
        DeterministicSelection.reinitialize(self, truncatedParentPop)
        

#############################################################################
#
# unit_test
#
#############################################################################
#from problem import *
#from decoder import *

def rampLandscape(phenome):
    return(phenome[0])
    #return(phenome[0]**2)


def unit_test():
    t_passed = True
    p_passed = True
    r_passed = True
    bounds = [(0,100)]
    problem = LEAP.FunctionOptimization(rampLandscape)
    decoder = LEAP.FloatDecoder(problem, bounds, bounds)
    popsize = 10
    population = []
    for i in range(1, popsize*2, 2):
#    m = 4.0
#    for i in [1/m**2, 1/m, 1/m, 2/m - 1/m**2]:
        ind = LEAP.Individual(decoder, [float(i)])
        ind.evaluate()
        population.append(ind)

    outer = 15
    inner = 20000
    numSel = outer * inner

    tourn = TournamentSelection(2, population)
    prop = ProportionalSelection(population, SUS_numSelections = numSel)
    rank = RankSelection(population, SUS_numSelections = numSel)
    ta = [0] * popsize
    pa = [0] * popsize
    ra = [0] * popsize

    print("Tournament selection:")
    for i in range(outer,0,-1):
        print(str(i) + "..", end="")
        sys.stdout.flush()
        for j in range(inner):
            ta[tourn.pull().popIndex] += 1

    #print("\nProportional selection:")
    #for i in range(outer,0,-1):
    #    print(str(i) + "..", end="")
    #    sys.stdout.flush()
    #    for j in range(inner):
    #        pa[prop.pull().popIndex] += 1

    #print("\nRanked selection:")
    #for i in range(outer,0,-1):
    #    print(str(i) + "..", end="")
    #    sys.stdout.flush()
    #    for j in range(inner):
    #        ra[rank.pull().popIndex] += 1

    total = outer * inner
    ta = [round(x*1000.0/total)/10.0 for x in ta]
    pa = [round(x*1000.0/total)/10.0 for x in pa]
    ra = [round(x*1000.0/total)/10.0 for x in ra]

    print()
    print("tournament:", ta, "sum=", sum(ta))
    print("proportion:", pa, "sum=", sum(pa))
    print("ranked:    ", ra, "sum=", sum(ra))

    TA = [i*2+1.0 for i in range(10)]
    inc = 100.0 / sum(range(1,11))
    PA = [inc * i for i in range(1,11)]

    tolerance = 0.2
    for i in range(10):
        if abs(ta[i] - TA[i]) > tolerance:
            t_passed = False
        if abs(pa[i] - PA[i]) > tolerance:
            p_passed = False
        if abs(ra[i] - PA[i]) > tolerance:
            r_passed = False

    #passed = t_passed & p_passed & r_passed
    passed = t_passed

    print()
    print("Tournament passed: ", t_passed)
    print("Proportional passed: Not tested") #, p_passed)
    print("Ranked passed: Not tested") #, r_passed)

    print()
    if passed:
        print("Passed")
    else:
        print("FAILED")


if __name__ == '__main__':
    unit_test()

#    import profile
#    profile.run('unit_test()', 'Selection.profile')
#
#    import pstats
#    p = pstats.Stats('Selection.profile')
#    p.sort_stats('time').print_stats(20)


