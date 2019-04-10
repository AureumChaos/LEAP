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
def cmp(a,b):
    return int(a>b) - int(a<b)

import sys
import string
import copy    # for Clone
import functools


#############################################################################
#
# mystr
#
#   Used for doing customized string conversions, especially on lists
# and arrays.
#
#############################################################################
def mystr(x):
    #if (getattr(x, '__iter__', False)):
    #    return "[" + ",".join([mystr(i) for i in x]) + "]"
    #else:
        return str(x)

#def mystr(x):
#    if isinstance(x, list):
#        s = "["
#        for i in x:
#            s += mystr(i) + ","
#        if s[-1] == ",":
#            s = s[:-1]
#        s += "]"
#        return s
##    elif type(x) == type(array([])):
##        s = string.replace(repr(x), ' ', '')
##        s = string.replace(s, 'array(', '')
##        s = string.replace(s, ')', '')
##        return s
##        #return str(x[1])
#    elif isinstance(x, float):
#        s = "%.12f" % x
#        return s[0:11]
#    elif isinstance(x, array):
#        mystr(list(x))
#    else:
#        return str(x)



#############################################################################
#
# cmpInd
#
#############################################################################
def cmpInd(ind1, ind2):
    """
    Similar to the cmp() function, but for individuals.  Useful for sorting
    and tournament selection.  Returns 1, 0, or -1.  1 is returned if ind1
    is "better than" ind2.  The definition of "better than" will be different
    if we are minimizing or maximizing.
    """
    return ind1.cmp(ind2)


#############################################################################
#
# fittest
#
#############################################################################
def fittest(ind1, ind2 = None):
    """
    Returns the fittest individual given 2 individuals.  If no second
    parameter is given, the the first is assumed to be a sequence and the
    fittest in the sequence is returned.  Similar to max().
    """
    # This is all kind of hacked, but I'm not sure what would be a better
    # way of doing this.
    if ind1 == None:
        return ind2

    if ind2 == None and (isinstance(ind1, list)):
        return functools.reduce(fittest, ind1)
    else:
        if cmpInd(ind1, ind2) >= 0:
            return ind1
        else:
            return ind2


#############################################################################
#
# size
#
#############################################################################
def size(sequence):
    """
    Recursive function which returns the number of elements in a list or
    tree (list of lists).  A single item has a size of 1, and an empty list
    has a size of 0.
    """
    #if isinstance(sequence, list) or isinstance(sequence, tuple):
    #    theSize = 0
    #    for i in sequence:
    #        theSize += size(i)
    #elif isinstance(sequence, str):
    #    theSize = len(sequence)
    #else:
    #    theSize = 1
    #return theSize
    theSize = len(sequence)
    theSize += sum([len(string)-1 for string in sequence
                    if isinstance(string, str)])
    theSize += sum([size(i)-1 for i in sequence 
                    if isinstance(i, list) or isinstance(i,tuple)])
    return theSize


#############################################################################
#
# class Individual
#
#############################################################################
class Individual:
    "Base class for individuals"

    def __init__(self, encoding, genome = None):
        self.encoding = encoding
        if genome == None:
            self.genome = self.encoding.randomGenome()
        else:
            self.genome = genome

        self.rawFitness = None
        self.fitness = None
        self.prevFitness = None
        self.parents = []  # Backpointers to parents
        self.previous = []  # Difficult to describe.  For Price's equ.
        self.popIndex = None


    def __repr__(self):
        return ("genome: " + mystr(self.genome) +
                "  fitness: " + mystr(self.getFitness()))


    def cmp(self, other):
        """
        Returns 1 if self is better than other
                0 if self = other
               -1 if self is worse than other
        Better than means '>' if maximizing or '<' if minimizing.

        This may seem like overkill, having 3 cmp routines (two here and
        one in the Problem class) but there is a good reason for it.
        Problem.cmpFitness is used to define whether fitness is minimized
        or maximized.  This function can be used to deal with other factors,
        such as parsimony pressure.  The cmpInd function can be used as a
        comparator function and passed to things like sort routines.
        """
        # I have a feeling I shouldn't be doing this
        if self == None and other == None:
            return 0

        if self == None:
            return -1

        if other == None:
            return 1

        return self.encoding.cmpFitness(self.getFitness(), other.getFitness())


    def evaluate(self):
        "Calculate the fitness of the individual"
        if self.fitness == None:
            self.rawFitness = self.encoding.evaluate(self.genome)
            self.fitness = self.rawFitness
#            self.prevFitness = self.fitness
        return self.fitness

    def getFitness(self):
        if self.fitness == None:
            self.evaluate()
        return self.fitness

    def getRawFitness(self):
        if self.fitness == None:
            self.evaluate()
        return self.rawFitness

    def resetFitness(self):
        "Set the fitness to None in order to force evaluation"
        self.prevFitness = self.fitness
        self.fitness = None

    def genomeUnchanged(self):
        """
        Undo a call to resetFitness.  This is used by GeneticOperators
        when they do not change the genome.  I devised this somewhat unusual
        approach in order to make things safer for people developing their
        own operators.  This way, the default behavior of the operators is
        to reset the fitness.  People who know what they're doing can then
        undo that reset (with a call to this function) and thus optimize
        their code.
        """
        self.fitness = self.prevFitness


    def copyGenome(self, genome):
        """
        Recursive copy function for copying the genome.  I used to use
        copy.deepcopy, but that is way too expensive.  This should be a
        reasonable default for most EAs.
        """
        if isinstance(genome, list):
            return [self.copyGenome(g) for g in genome]
        else:
            return copy.copy(genome)


    def clone(self):
        """
        Create a copy of an individual.  This method is also used to clear
        any information that should not be passed from parent to child.
        """
        clone = copy.copy(self)
        clone.genome = self.copyGenome(self.genome)

        #clone.fitness = None  # Force reevaluation
        clone.parents = []
        return clone


#############################################################################
#
# class ListOfListsIndividual
#
#############################################################################
class ListOfListsIndividual(Individual):
    """
    An optimized version of Individual for an class in which the genome is
    a list of lists.
    """

    def clone(self):
        """
        Create a copy of an individual.  This method is also used to clear
        any information that should not be passed from parent to child.
        """
        clone = copy.copy(self)

        clone.genome = [i[:] for i in self.genome]

        #clone.fitness = None  # Force reevaluation
        clone.parents = []
        return clone


#############################################################################
#
# class PenaltyParsimonyIndividual
#
#############################################################################
class PenaltyParsimonyIndividual(Individual):
    """
    An individual with a variable length genome.  This class implements
    simple penalty parsimony pressure (see Smith's 1980 dissertation or
    Koza's 1992 GP 1 book).
    """
    def __init__(self, encoding, penalty = 1, minPenaltySize = 0, genome = None):
        self.penalty = penalty
        self.minPenaltySize = minPenaltySize
        Individual.__init__(self, encoding, genome)
    
    def evaluate(self):
        self.rawFitness = self.encoding.evaluate(self.genome)

        # Determine if we are minimizing or maximizing.
        # I am making the assumption that fitness is a number.  This may
        # be a bad idea.
        sign = self.encoding.cmpFitness(1,0)

        sizeDiff = size(self.genome) - self.minPenaltySize
        if sizeDiff > 0:
            self.fitness = self.rawFitness - sign * self.penalty * sizeDiff
        else:
            self.fitness = self.rawFitness

        return self.fitness


#############################################################################
#
# class LexParsimonyIndividual
#
#############################################################################
class LexParsimonyIndividual(PenaltyParsimonyIndividual):
    """
    An individual with a variable length genome.  This class implements
    simple lexicographic parsimony pressure (see Sean Luke's 2002 GECCO
    paper).
    """
    def __init__(self, encoding, penalty = 0, minPenaltySize = 0, genome = None):
        PenaltyParsimonyIndividual.__init__(self, encoding, penalty,
                                            minPenaltySize, genome)
    
    def cmp(self, other):
        """
        Returns 1 if self is better than other
                0 if self = other
               -1 if self is worse than other
        Better than means '>' if maximizing or '<' if minimizing.

        Implements lexicographic parsimony pressure.
        """
        # Why did I do this None stuff again?
        if self == None and other == None:
            return 0

        if self == None:
            return -1

        if other == None:
            return 1

        result = self.encoding.cmpFitness(self.getFitness(), other.getFitness())
        if result == 0:
            return -cmp(size(self.genome), size(other.genome))
        else:
            return result


#############################################################################
#
# unit_test
#
#############################################################################
def unit_test():
    from LEAP.problem import FunctionOptimization
    from LEAP.problem import oneMaxFunction
    from LEAP.decoder import BinaryEncoding

    passed = True

    onemaxProb = FunctionOptimization(oneMaxFunction)
    onemax = BinaryEncoding(onemaxProb, 4)

    print("Test Individual")
    ind0 = Individual(onemax, "0000")
    ind1 = Individual(onemax, "1111")
    ind2 = Individual(onemax, "1100")
    ind3 = Individual(onemax, "0011")
    ind4 = Individual(onemax, "001100")
    #ind0.evaluate()
    #ind1.evaluate()
    #ind2.evaluate()
    #ind3.evaluate()
    #ind4.evaluate()

    test = cmpInd(ind1, ind0)
    #print("ind1 =", ind1)
    #print("ind0 =", ind0)
    passed = passed and (test == 1)
    print(test, "== 1")

    test = cmpInd(ind2, ind3)
    passed = passed and (test == 0)
    print(test, "== 0")

    test = cmpInd(ind2, ind4)
    passed = passed and (test == 0)
    print(test, "== 0")

    print("Test fittest()")
    ind = fittest([ind0, ind1, ind2, ind3])
    passed = passed and (ind is ind1)
    print(ind, "==", ind1)

    print("Test PenaltyParsimonyIndividual")
    pind0 = PenaltyParsimonyIndividual(onemax, 0.1, 4, genome = "1100")
    pind1 = PenaltyParsimonyIndividual(onemax, 0.1, 4, genome = "0011")
    pind2 = PenaltyParsimonyIndividual(onemax, 0.1, 4, genome = "00110")
    #pind0.evaluate()
    #pind1.evaluate()
    #pind2.evaluate()

    test = cmpInd(pind0, pind1)
    passed = passed and (test == 0)
    print(test, "== 0")

    test = cmpInd(pind0, pind2)
    passed = passed and (test == 1)
    print(test, "== 1")

    passed = passed and (pind2.getFitness() == 1.9)
    print("fitness =", pind2.getFitness(), "== 1.9")
    print("rawFitness =", pind2.rawFitness, "== 2")

    print("Test LexParsimonyIndividual")
    lind0 = LexParsimonyIndividual(onemax, genome = "1100")
    lind1 = LexParsimonyIndividual(onemax, genome = "0011")
    lind2 = LexParsimonyIndividual(onemax, genome = "00110")
    #lind0.evaluate()
    #lind1.evaluate()
    #lind2.evaluate()

    test = cmpInd(lind0, lind1)
    passed = passed and (test == 0)
    print(test, "== 0")

    test = cmpInd(lind0, lind2)
    passed = passed and (test == 1)
    print(test, "== 1")

    passed = passed and (lind0.getFitness() == 2.0)
    print("fitness =", lind0.getFitness(), "== 2.0")

    passed = passed and (lind2.getFitness() == 2.0)
    print("fitness =", lind2.getFitness(), "== 2.0")

    print()
    if passed:
        print("Passed")
    else:
        print("FAILED")


if __name__ == '__main__':
    unit_test()

