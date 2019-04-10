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

import random
import copy

#import scipy.stats

#from ca import *  # The python version
#from CA import *  # The C version

from LEAP.problem import *
from LEAP.decoder import *
from LEAP.individual import *
from LEAP.selection import *
from LEAP.operators import *
from LEAP.survival import *

from LEAP.Domains.CA.MajorityClassification import MajorityClassification


#############################################################################
#
# printPop
#
#############################################################################
def printPop(population):
    i = 0
    for ind in population:
        print(i, ":", end='')
        print(ind.genome, " ", end='')
        print(ind.fitness)
        i += 1


#############################################################################
#
# ga
#
#############################################################################
def ga():
    """
    The main loop of the Genetic Algorithm.
    """
    popSize = 100
    radius = 3
    genomeLength = 2 ** (radius * 2 + 1)
    alleles = range(30)
    maxGeneration = 20
    population = []
    children = []
    bestSoFar = None

    # Setup the reproduction pipeline
    pipeline = select = TournamentSelection(2)
#    pipeline = select = ProportionalSelection()
#    pipeline = select = RankSelection()
    pipeline = clone = CloneOperator(pipeline)
#    pipeline = crossover = NPointCrossover(pipeline, 0.8, 2)
#    pipeline = crossover = UniformCrossover(pipeline, 0.5)
    pipeline = mutate = BitFlipMutation(pipeline, 1.0/genomeLength)
#    pipeline = mutate = UniformMutation(pipeline, 1.0/60.0, alleles)
#    pipeline = survive = ElitismSurvival(pipeline, 2)
    pipeline = survive = MuPlusLambdaSurvival(pipeline, popSize, popSize)

    # Setup the problem
    problem = MajorityClassification(radius, stateSize = 51, maxSteps=100)
    encoding = BinaryEncoding(problem, genomeLength)

    # Create initial population
    bestOfGen = None
    for i in range(popSize):
        ind = Individual(encoding)
        #print(ind.genome)
        print(".", end="")
        sys.stdout.flush()
        ind.evaluate()
        if bestOfGen is None:
            bestOfGen = ind
        else:
            if cmpInd(ind,bestOfGen) == 1:
                bestOfGen = ind
        population.append(ind)
    #print "population initialized"
    bestSoFar = bestOfGen
    print()

    genome_str = ''
    for g in bestOfGen.genome:
        genome_str += str(g)
    print(0, ":", genome_str, bestOfGen.fitness)
        
    #print "numEvaluations =", numEvaluations
    #numEvaluations = 0

    # Evolution
    for gen in range(1, maxGeneration + 1):
        # print("Generation:", gen)
        # printPop(population)

        bestOfGen = None
        children = []
        pipeline.reinitialize(population)
        for i in range(popSize):
            print(".", end="")
            sys.stdout.flush()
            child = pipeline.pull()
            if child.fitness == None:
                child.evaluate()
            if cmpInd(child,bestOfGen) == 1:
                bestOfGen = child
            children.append(child)

#        parents = []
#        for i in range(popSize):
#            parents += select.apply(population)  # append selected parents
#        children = clone.apply(parents)
#        children = crossover.apply(children)
#        children = mutate.apply(children)
#        for child in children:
#            child.evaluate()
#            if cmpInd(child,bestSoFar) == 1:
#                bestSoFar = child

        genome_str = ''
        for g in bestOfGen.genome:
            genome_str += str(g)
        print()
        print(gen, ":", genome_str, bestOfGen.fitness)

        if cmpInd(bestOfGen,bestSoFar) == 1:
            bestSoFar = bestOfGen
        
        # print("numEvaluations =", numEvaluations)
        # numEvaluations = 0

        population = children

    # print out the final population
    #printPop(population)
    genome_str = ''
    for g in bestSoFar.genome:
        genome_str += str(g)
    print("best :", genome_str, bestSoFar.fitness)


#############################################################################
#
# The main part of the script
#
#############################################################################
ga()


