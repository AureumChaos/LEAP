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
from ca import *


#############################################################################
#
# MajorityClassification
#
#############################################################################
class MajorityClassification(LEAP.Problem):
    """
    Cellular automata majority classification problem.
    If there are more 1's than 0's in the initial string, the CA should
    result in a string containing all 1's.  More 0's should result in all
    0's.
    """
    radius = None
    genomeLength = None
    stateSize = None
    maxSteps = None

    def __init__(self, radius = 1, stateSize = 79, maxSteps = 100):
        self.radius = radius
        neighborhood = radius * 2 + 1
        self.genomeLength = 2 ** neighborhood
        self.stateSize = stateSize
        self.maxSteps = maxSteps
    
    def episode(self, ca, stateOneBias):
        # Generate the initial state
        initState = ''
        for i in range(self.stateSize):
            if random.random() < stateOneBias:
                initState += '1'
            else:
                initState += '0'
        # find the majority
        ones = 0
        for i in initState:
            ones += int(i)
        majority = round(float(ones) / self.stateSize)

        # run the CA
        finalState = ca.run(initState, self.maxSteps)

        # calculate fitness
        ones = 0
        for i in finalState:
            ones += int(i)
#        for i in finalState2:
#            ones += int(i)
        fitness = abs(majority - float(ones) / self.stateSize)
#        fitness = fitness / 2
        return fitness

    def evaluate(self, individual):
        # convert the genome into a string
        rules = ''
        for gene in individual.genome:
            rules += str(gene)

        # build the CA
        ca = CellularAutomata(rules)

        fitness = 0.0
        fitness += self.episode(ca, 0.2)
        fitness += self.episode(ca, 0.25)
        fitness += self.episode(ca, 0.3)
        fitness += self.episode(ca, 0.35)
        fitness += self.episode(ca, 0.4)

        fitness += self.episode(ca, 0.5)
        fitness += self.episode(ca, 0.5)
        fitness += self.episode(ca, 0.5)
        fitness += self.episode(ca, 0.5)
        fitness += self.episode(ca, 0.5)

        fitness += self.episode(ca, 0.6)
        fitness += self.episode(ca, 0.65)
        fitness += self.episode(ca, 0.7)
        fitness += self.episode(ca, 0.75)
        fitness += self.episode(ca, 0.8)
        fitness = fitness / 15
        return fitness

    def cmpFitness(self, fitness1, fitness2):
        """
        Returns 1 if fitness1 is better than fitness2
                0 if fitness1 = fitness2
               -1 if fitness1 is worse than fitness2
        Better than means '>' if maximizing or '<' if minimizing.
        """
        return -cmp(fitness1.fitness, fitness2.fitness)   # Minimize

    def randomGenome(self):
        genome = []
        for i in range(self.genomeLength):
            genome.append(random.randrange(2))
        return genome


