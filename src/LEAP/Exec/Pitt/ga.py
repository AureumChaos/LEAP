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
#import scipy.stats
from parallelParity import *
from serialParity import *


#############################################################################
#
# printInd
#
#############################################################################
def printInd(index, individual):
    print index,
    genstr = ''
    for i in individual.genome:
        genstr += str(i)
    print genstr, " ",
    print individual.fitness, len(individual.genome)


#############################################################################
#
# printPop
#
#############################################################################
def printPop(population):
    i = 0
    for ind in population:
        printInd(i, ind)
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
    # Problem parameters
    numBits = 4
    numTests = 2 ** numBits
    #digits = [0, 1]  #[0, 1, 2, 3]
    digits = ['0', '1']

    ruleSize = numBits * 2 + 1
    minRules = 5   # These numbers are only used during initialization
    maxRules = 10

    # Setup the problem
    #problem = ParallelParity(numBits)
    problem = SerialParity(numBits)
    #encoding = PittOrdinalEncoding(problem, minRules, maxRules, digits)
    encoding = PittBinaryEncoding(problem, minRules, maxRules, digits)

    # EA parameters
    alleles = digits
    mutRate = 1.0 / ((minRules + maxRules)/2 * ruleSize)

    popSize = 50
    maxGeneration = 30

    # Setup the reproduction pipeline
    pipeline = select = LEAP.TournamentSelection(2)
#    pipeline = select = LEAP.ProportionalSelection()
#    pipeline = select = LEAP.RankSelection()
    pipeline = clone = LEAP.CloneOperator(pipeline)
#    pipeline = crossover = LEAP.NPointCrossover(pipeline, 0.7, 2)
    pipeline = crossover = LEAP.VarNPointCrossover(pipeline, 0.7, 2)
#    pipeline = crossover = LEAP.UniformCrossover(0.5, pipeline)
#    pipeline = crossover = LEAP.VarSwapGenesCrossover(0.8, 3, pipeline)
#    pipeline = mutate = LEAP.BitFlipMutation(pipeline, mutRate, alleles,
#                                             linear=False)
#    pipeline = mutate = LEAP.UniformMutation(mutRate, alleles, pipeline)
    pipeline = mutate = LEAP.VarBitFlipMutation(pipeline, 1.0, alleles, \
                                                linear=False)
    pipeline = survive = LEAP.ElitismSurvival(pipeline, 2)
#    pipeline = survive = LEAP.MuPlusLambdaSurvival(popSize, pipeline)


    ea = LEAP.GenerationalEA(encoding, pipeline, popSize, \
            indClass=LEAP.LexParsimonyIndividual)
    ea.run(maxGeneration)

    best = ea.bestOfGen
    print "test =", encoding.testGeneralization(best.genome)


#############################################################################
#
# The main part of the script
#
#############################################################################

#ga()

import profile
profile.run('ga()','ga.prof')

import pstats
p = pstats.Stats('ga.prof').strip_dirs()
p.sort_stats('time').print_stats(5)


