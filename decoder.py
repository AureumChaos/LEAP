#! /usr/bin/env python

# encoding.py
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
import math
import functools

from LEAP.gene import BoundedRealGene
from LEAP.gene import AdaptiveRealGene


#############################################################################
#
# int2bin
#
# A handy integer to binary conversion function I found posted at:
#
# http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/219300
#  W.J. van der Laan, 2003/11/05
#
#############################################################################
def int2bin(i, count=8):
    """
    Integer to binary
    Count is number of bits
    """
    return "".join(map(lambda j:str((i>>j)&1), range(count-1, -1, -1)))



#############################################################################
#
# bin2int
#
#############################################################################
def bin2int(binStr):
    return int(binStr, 2)



#############################################################################
#
# class Encoding
#
#############################################################################
class Encoding:
    """
    Defines how a genome is decoded into a phonome.
    It has some additional methods to do things like generate random genomes,
    fixup a genome (guarantee that it is legal) and encode a genome from a
    phenome.
    """
    def __init__(self, problem):
        # XXX I'm not sure I like the idea of having the encoding know about
        #     the problem anymore.  I'm thinking it may make more sense to
        #     separate these.
        self.problem = problem

    def decodeGenome(self, genome):
        "Decodes the genome into a phenome and returns it."
        return genome

    def encodeGenome(self, phenome):
        """
        Encodes the phenome into a genome and returns it.
        In general this function is neither needed nor used.  There are some
        situations where it has come in handy though, so I've decided to
        include it in the base class.  Feel free to leave it undefined if you
        like.
        """
        #return genome
        raise NotImplementedError

    def randomGenome(self):
        "Generates a randomized genome for this encoding."
        raise NotImplementedError

    def fixupGenome(self, genome):
        "Fixes errors in the genome.  Used by the FixupOperator."
        return genome

    def evaluate(self, genome):
        # I really shouldn't have put this here!
        "Decodes the genome and evaluates it."
        phenome = self.decodeGenome(genome)
        return self.problem.evaluate(phenome)

    def cmpFitness(self, fitness1, fitness2):
        "Determine if fitness1 is 'better than' fitness2."
        return self.problem.cmpFitness(fitness1, fitness2)


#############################################################################
#
# class FloatEncoding
#
#############################################################################
class FloatEncoding(Encoding):
    """
    Defines the genome as a list of floats.  The initRanges parameter
    is a list of tuples containing a lower and upper bound for each gene
    in a new genome.  The bounds parameter takes the same form as initRanges,
    but defines the bounds that each gene will be "clipped" to when
    fixupGenome() is called.  The fixupGenome() function is called by the
    FixupOperator().

    I recommend using BoundedRealEncoding instead.

    NOTE: If either a lower or upper bound (or both) is set to None in the
    bounds parameter (not the initRanges parameter), that bound is essentially
    eliminated.  In other words, if bound is set to [(0, None), (0, 100)],
    then only the lower bound of 0 will be enforced for the first gene.  Both
    bounds will be enforced for the second gene.
    """
    def __init__(self, problem, initRanges, bounds = None):
        if bounds != None and len(initRanges) != len(bounds):
            raise ValueError("initRanges and bounds are different lengths.")

        # instance variables
        Encoding.__init__(self, problem)
        self.genomeLength = len(initRanges)
        self.initRanges = initRanges
        self.bounds = bounds

    def randomGenome(self):
        "Generates a randomized genome for this encoding"
        genome = [random.random() * (r[1] - r[0]) + r[0] \
                  for r in self.initRanges]
        return genome

    def fixupGenome(self, genome):
        "Fixes errors in the genome.  Used by the FixupOperator."
        # Make sure each gene falls with the appropriate bounds.
        # I use 'or gene' to deal with the case where bounds[i][1] is None
        if self.bounds:
            for i, gene in enumerate(genome):
                if self.bounds[i][0] is not None:
                    gene = max(gene, self.bounds[i][0])

                if self.bounds[i][1] is not None:
                    gene = min(gene, self.bounds[i][1])

                genome[i] = gene

        return genome


#############################################################################
#
# class BoundedRealEncoding
#
#############################################################################
class BoundedRealEncoding(FloatEncoding):
    """
    Defines the genome as a list of BoundedRealGenes.  The initRanges
    parameter is a list of tuples containing a lower and upper bound for each
    gene in a new genome.  The bounds parameter takes the same form as
    initRanges, but defines the bounds that each gene will be "clipped" to.
    Clipping is handled automatically by BoundedRealGene.

    As with FloatEncoding, None values are allowed as lower and upper bounds,
    thus eliminating that bound.
    """
    def __init__(self, problem, initRanges, bounds):
        FloatEncoding.__init__(self, problem, initRanges, bounds)

    def decodeGenome(self, genome):
        "Decodes the genome into a phenome and returns it."
        # extract data from the BoundedRealGene class
        phenome = [gene.data for gene in genome]
        return phenome

    def randomGenome(self):
        "Generates a randomized genome for this encoding"
        initVals = FloatEncoding.randomGenome(self)
        genome = [BoundedRealGene(initVals[i], self.bounds[i])
                  for i in range(self.genomeLength)]
        return genome

    def fixupGenome(self, genome):
        "Fixes errors in the genome.  Used by the FixupOperator."
        return genome  # No fixing needed.


#############################################################################
#
# class AdaptiveRealEncoding
#
#############################################################################
class AdaptiveRealEncoding(BoundedRealEncoding):
    """
    Defines the genome as a list of AdaptiveRealGenes.  Each of these genes
    have a sigma (standard deviation) associated with it that is used by the
    AdaptiveMutation operator.

    The initRanges parameter is a list of tuples containing a lower and upper
    bound for each gene in a new genome.  The bounds parameter takes the same
    form as initRanges, but defines the bounds that each gene will be
    "clipped" to.  Clipping is handled automatically by AdaptiveRealGene.
    The initSigmas parameter contains a list of initial sigma values for each
    gene.

    As with FloatEncoding and BoundedRealEncoding, None values are allowed as
    lower and upper bounds, thus eliminating that bound.
    """
    def __init__(self, problem, initRanges, bounds, initSigmas=None):
        BoundedRealEncoding.__init__(self, problem, initRanges, bounds)

        if initSigmas == None:
            # I took this calculation from Mitch Potter's code.  I don't
            # know for certain if this is how the ES crowd does it.
            initSigmas = [(ir[1] - ir[0]) / math.sqrt(len(initRanges))
                          for ir in initRanges]

        if len(initSigmas) != self.genomeLength:
            raise ValueError("initRanges and initSigmas are different lengths.")

        # instance variables
        self.initSigmas = initSigmas

    def randomGenome(self):
        "Generates a randomized genome for this encoding"
        initVals = FloatEncoding.randomGenome(self)
        genome = [AdaptiveRealGene(initVals[i], self.bounds[i],
                                   self.initSigmas[i])
                  for i in range(self.genomeLength)]
        return genome


#############################################################################
#
# class NominalEncoding
#
#############################################################################
class NominalEncoding(Encoding):
    """
    Defines the genome as a list of discrete values (alleles).
    """
    def __init__(self, problem, genomeLength, alleles):
        Encoding.__init__(self, problem)
        # instance variables
        self.genomeLength = genomeLength
        self.alleles = alleles

    def randomGenome(self):
        "Generates a randomized genome for this encoding"
        genome = [random.choice(self.alleles)
                  for i in range(self.genomeLength)]
        return genome


#############################################################################
#
# class BinaryEncoding
#
#############################################################################
class BinaryEncoding(Encoding):
    """
    Defines the genome as a string (not list) of ones and zeros.
    
    I've gone back and forth on the issue of whether the genome should
    be a list or a string.  On the one hand, if I make it a list, then
    everything is very consisitent.  On the other, if I make it a string, I
    can decode binary numbers quite quickly using the int() function.
    Ultimately I've decided to opt for speed.
    """
    alleles = ['0', '1']

    def __init__(self, problem, genomeLength):
        Encoding.__init__(self, problem)

        # instance variables
        self.genomeLength = genomeLength

    def randomGenome(self):
        "Generates a randomized genome for this encoding"
        genome = ""
        for i in range(self.genomeLength):
            genome += random.choice(self.alleles)
        return genome


#############################################################################
#
# class BinaryRealEncoding
#
#############################################################################
class BinaryRealEncoding(BinaryEncoding):
    """
    Defines the genome as a string (not list) of ones and zeros.
    Each section of the genome encodes a real valued number.
    """
    def __init__(self, problem, bitsPerReals, bounds):
        if not isinstance(bitsPerReals, list):
            raise ValueError("bitsPerReals must be a list.")
        if len(bitsPerReals) != len(bounds):
            raise ValueError("bitsPerReals and bounds are different lengths.")
        BinaryEncoding.__init__(self, problem, sum(bitsPerReals))

        # instance variables
        self.bitsPerReals = bitsPerReals   # a list, # of bits per each real
        self.bounds = bounds

        # calcluate the positions of the reals in the genome
        self.realPos = [0]
        pos = 0
        for i in range(len(bitsPerReals)):
            pos += bitsPerReals[i]
            self.realPos.append(pos)

        # calculate the max integer value of each substring
        #self.maxIntVal = [2**b - 1 for b in bitsPerReals]
        self.maxIntVal = [2**b for b in bitsPerReals]

    def decodeGenome(self, genome):
        phenome = []
        for i in range(len(self.bitsPerReals)):
            bnd = self.bounds[i]
            ival = self.decodeBinary(genome[self.realPos[i] : self.realPos[i+1]])
            rval = float(ival) / self.maxIntVal[i] * (bnd[1] - bnd[0]) + bnd[0]
            phenome.append(rval)
        return phenome

    def decodeBinary(self, binary):
        return int(binary, 2)


#############################################################################
#
# class GrayRealEncoding
#
#############################################################################
class GrayRealEncoding(BinaryRealEncoding):
    """
    A reflective Gray code is used for the encoding of binary numbers.
    """
    def decodeBinary(self, binary):
        i = 0
        lastb = 0
        for g in binary:
            b = int(g) ^ lastb  # XOR
            i = (i << 1) + b
            lastb = b

        return i


#############################################################################
#
# class IntegerEncoding
#
#############################################################################
class IntegerEncoding(FloatEncoding):
    def __init__(self, problem, initRanges, bounds = None):
        FloatEncoding.__init__(self, problem, initRanges, bounds)

    def randomGenome(self):
        "Generates a randomized genome for this encoding"
        genome = [random.choice(range(r[0],r[1]))  for r in self.initRanges]
        return genome


#############################################################################
#
# class ScramblerEncoding
#
#############################################################################
class ScramblerEncoding(Encoding):
    """
    This encoding is intended to be a wrapper around another (slave) encoding.
    It will scramble the genome in order to alter the effects of the genetic
    operators (especially NPointCrossover).

    The idea is that this encoding will scramble and unscramble the genome
    without the slave encoding ever noticing.

    NOTE: This implementation is pretty inefficient.  It should really be
          using some sort of bidirectional map, maybe using 2 dictionaries.
    """
    def __init__(self, slaveEncoding):
        Encoding.__init__(self, slaveEncoding.problem)
        self.slaveEncoding = slaveEncoding
        self.map = []  # Create this later when we know the genome length

    def createMap(self, length):
        self.map = [None] * length
        indices = list(range(length))
        for i in range(length):
            self.map[i] = indices.pop(random.randrange(len(indices)))

    def measureMap(self, map = None):
        if map == None:
            map = self.map

        measure = 0
        maxdist = 0
        for i in range(len(map)):
            dist = abs(i - map[i])
            maxdist = max(maxdist, dist)
            measure += dist
        measure = float(measure) / (len(map)**2 / 2)
        #print(maxdist)
        return measure

    def scrambleGenome(self, genome):
        if not self.map:
            self.createMap(len(genome))
        #print(self.map)
        return [genome[self.map[i]] for i in range(len(genome))]

    def unscrambleGenome(self, genome):
        if not self.map:
            self.createMap(len(genome))
        unscrambled = [None] * len(genome)
        for i in range(len(genome)):
            unscrambled[self.map[i]] = genome[i]
        return unscrambled

    def decodeGenome(self, genome):
        return self.slaveEncoding.decodeGenome(self.unscrambleGenome(genome))

    def encodeGenome(self, phenome):
        return self.scrambleGenome(self.slaveEncoding.encodeGenome(phenome))

    def randomGenome(self):
        return self.scrambleGenome(self.slaveEncoding.randomGenome())

    def fixupGenome(self, genome):
        unscrambled = self.unscrambleGenome(genome)
        unscrambled = self.slaveEncoding.fixupGenome(unscrambled)
        return self.scrambleGenome(unscrambled)


#############################################################################
#
# class PerturbingEncoding
#
#############################################################################
class PerturbingEncoding(ScramblerEncoding):
    """
    This is basically the same as the ScramblerEncoding, except that one can
    control how scrambled the genome gets.
    """
    def __init__(self, slaveEncoding, pSwap = 0.5, iterations = 1):
        ScramblerEncoding.__init__(self, slaveEncoding)
        self.pSwap = pSwap
        self.iterations = iterations

    def createMap(self, length):
        self.map = list(range(length))
        for i in range(1, self.iterations + 1):
            for j in range(length-i):
                if random.random() < self.pSwap:
                    self.map[j], self.map[j+i] = self.map[j+i], self.map[j]
        #print(self.map)


#############################################################################
#
# test
#
#############################################################################
def landscape(phenome):
    return sum(phenome)

def unit_test():
    from LEAP.problem import FunctionOptimization
    from LEAP.problem import Problem

    passed = True

    realValuedProb = FunctionOptimization(landscape)
    dummyProb = Problem()

    length = 64
    measure = 0.0
    for i in range(100):
        bin = BinaryEncoding(realValuedProb, length)
        #scram = PerturbingEncoding(bin, pSwap = 0.2, iterations = 32)
        scram = ScramblerEncoding(bin)
        genome = scram.randomGenome()
        measure += scram.measureMap()
    measure /= 100
    print(measure)

    #return

    # Test FloatEncoding
    bounds = [(0.0, 1.0)] * 3
    fe = FloatEncoding(realValuedProb, bounds, bounds)
    genome = fe.randomGenome()
    inbounds = [bounds[i][0] <= genome[i] <= bounds[i][1]
                for i in range(len(genome))]
    valid = functools.reduce(lambda x,y:x and y, inbounds)

    print("FloatEncoding.randomGenome() =", genome)
    print("valid =", valid, "== True")
    passed = passed and valid

    brokenGenome = [-10.0, 100.0, 0.5]
    fixedGenome = fe.fixupGenome(brokenGenome)
    print("brokenGenome =", brokenGenome)
    print("fixedGenome =", fixedGenome, " == [0.0, 1.0, 0.5]")
    passed = passed and fixedGenome == [0.0, 1.0, 0.5]

    # Test BoundedRealEncoding
    bre = BoundedRealEncoding(realValuedProb, bounds, bounds)
    genome = bre.randomGenome()
    for i in range(len(genome)):
        genome[i].data = brokenGenome[i]
    print("genome =", genome, " == [0.0, 1.0, 0.5]")
    passed = passed and bre.decodeGenome(genome) == [0.0, 1.0, 0.5]

    # Test NominalEncoding
    alleles = ['A', 'G', 'C', 'T']
    ne = NominalEncoding(dummyProb, 10, alleles)
    genome = ne.randomGenome()
    valid = functools.reduce(lambda x,y:x and y, [g in alleles for g in genome])
    print("genome =", genome)
    print("valid =", valid, "== True")
    passed = passed and valid

    # Test BinaryRealEncoding
    bre = BinaryRealEncoding(realValuedProb, [10]*3, bounds)
    genome = bre.randomGenome()
    print("BinaryRealEncoding.randomGenome() =", genome)
    phenome = bre.decodeGenome(genome)
    print("phenome =", phenome)

    inbounds = [bounds[i][0] <= phenome[i] <= bounds[i][1]
                for i in range(len(phenome))]
    valid = functools.reduce(lambda x,y:x and y, inbounds)
    print("valid =", valid, "== True")
    passed = passed and valid


    gray = GrayRealEncoding(None, [16], [(0,1)])
    codes = ['000',
             '001',
             '011',
             '010',
             '110',
             '111',
             '101',
             '100']
    for c in codes:
        print(c, gray.decodeBinary(c))

    if passed:
        print("Passed")
    else:
        print("FAILED")


if __name__ == '__main__':
    unit_test()

