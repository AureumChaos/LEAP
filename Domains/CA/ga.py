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

import random
import copy

import LEAP
import scipy.stats
#from ca import *  # The python version
#from CA import *  # The C version
from MajorityClassification import *


#############################################################################
#
# printPop
#
#############################################################################
def printPop(population):
    i = 0
    for ind in population:
        print i, ":",
        print ind.genome, " ",
        print ind.fitness
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
    maxGeneration = 200
    population = []
    children = []
    bestSoFar = None

    # Setup the reproduction pipeline
    pipeline = select = LEAP.TournamentSelection(2)
#    pipeline = select = LEAP.ProportionalSelection()
#    pipeline = select = LEAP.RankSelection()
    pipeline = clone = LEAP.Clone(pipeline)
#    pipeline = crossover = LEAP.NPointCrossover(0.8, 2, pipeline)
#    pipeline = crossover = LEAP.UniformCrossover(0.5, pipeline)
    pipeline = mutate = LEAP.BitFlipMutation(1.0/genomeLength, pipeline)
#    pipeline = mutate = LEAP.UniformMutation(1.0/60.0, alleles, pipeline)
#    pipeline = survive = LEAP.ElitismSurvival(2, pipeline)
    pipeline = survive = LEAP.MuPlusLambdaSurvival(popSize, pipeline)

    # Setup the problem
    problem = MajorityClassification(radius, stateSize = 51, maxSteps=100)

    # Create initial population
    bestOfGen = None
    for i in range(popSize):
        ind = LEAP.Individual(problem)
        ind.evaluate()
        if LEAP.cmpInd(ind,bestOfGen) == 1:
            bestOfGen = ind
        population.append(ind)
    #print "population initialized"
    bestSoFar = bestOfGen

    genome_str = ''
    for g in bestOfGen.genome:
        genome_str += str(g)
    print 0, ":", genome_str, bestOfGen.fitness
        
    #print "numEvaluations =", numEvaluations
    #numEvaluations = 0

    # Evolution
    for gen in range(1, maxGeneration + 1):
        # print "Generation:", gen
        # printPop(population)

        bestOfGen = None
        children = []
        pipeline.newGeneration(population)
        for i in range(popSize):
            child = pipeline.pull()
            if child.fitness == None:
                child.evaluate()
            if LEAP.cmpInd(child,bestOfGen) == 1:
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
#            if LEAP.cmpInd(child,bestSoFar) == 1:
#                bestSoFar = child

        genome_str = ''
        for g in bestOfGen.genome:
            genome_str += str(g)
        print gen, ":", genome_str, bestOfGen.fitness

        if LEAP.cmpInd(bestOfGen,bestSoFar) == 1:
            bestSoFar = bestOfGen
        
        # print "numEvaluations =", numEvaluations
        # numEvaluations = 0

        population = children

    # print out the final population
    #printPop(population)
    genome_str = ''
    for g in bestSoFar.genome:
        genome_str += str(g)
    print "best :", genome_str, bestSoFar.fitness


#############################################################################
#
# The main part of the script
#
#############################################################################
ga()


