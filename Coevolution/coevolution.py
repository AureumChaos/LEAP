#! /usr/bin/env python

##############################################################################
#
#   LEAP - Library for Evolutionary Algorithms in Python
#   Copyright (C) 2004  Jeffrey K. Bassett & Paul Wiegand
#     Modified:  9.4.2004 (rpw)   Created
#                7.5.2004 (rpw)   Modified to reflect encorder changes
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
#    This module adds functionality for coevolution to
#  the LEAP library.  It contains one basic classe:
#
#       CoevolutionaryDecoder   Manages selection of
#                               collaborators/competitors
#                               from pops, evaluation, and
#                               fitness assignment.
#
#    The most basic internal data structure is a list of
#  interactions.  Each interaction is a list of four
#  items:
#       interactionList[0]   :::   collaborator/competitor selection method
#       interactionList[1]   :::   number of interacting indiv
#       interactionList[2]   :::   decoder for component
#       interactionList[3]   :::   population
#==========================================================

# Python 2 & 3 compatibility
from __future__ import print_function

import sys
import random
import math

#from LEAP.decoder   import *
#from LEAP.problem   import *
#from LEAP.ea        import *
#from LEAP.selection import *
import LEAP


#############################################################################
#
# CoevoltuionaryDecoder
#
##############################################################################
class CoevolutionaryDecoder(LEAP.Decoder):
    """
    CoevolutionaryDecoder class manages the selection of
    collaborators and/orcompetitors from multiple populations,
    potentially multple evaluation methods, and assessment of
    a single fitness score from many objective evaluations, etc.
    """
    
    def __init__(self):
        """
        Initialize the interaction list to nothing
        I am relying on the user to append the appropriate
        lists to the object.  Also, the user will need
        to use the "whichComponent" variable to specify
        which component/population the library is
        currently processing, etc.
        """
        self.interactionList = []
        self.whichComponent = 0
        self.bestGenome  = []
        self.bestPhenome = []
        self.bestFitness = None
        

    def resetBest(self):
        self.bestInteraction  = []
        self.bestPhenome      = []
        self.bestFitness      = None


    def storeBest(self, interaction, phenome, fitness):
        if (self.bestFitness == None):
            self.bestInteraction  = interaction
            self.bestPhenome      = phenome
            self.bestFitness      = fitness
        else:
            problem = self.getCurrentDecoder().problem
            cmpFitness = problem.cmpFitness(fitness,self.bestFitness)
            if (cmpFitness == 1):
                self.bestInteraction  = interaction
                self.bestPhenome      = phenome
                self.bestFitness      = fitness

    def preStepSetup(self):
        pass                
    
    def evaluate(self,genome):
        """
        This routine is main routine called.  It gathers the
        interactions, calls the assemble routine for each
        of them, and then passes them through the objective
        function (problem domain).  The result is a list of
        fitnesses, which are resolved to a single score by
        the assessFitness routine
        """
        interaction = self.getCurrentInteraction()        
        population = interaction[3]
      
        # collect interactions in a convenient format
        interactions = self.getAllInteractions(genome, population)

        # I will assume the user will select the component under
        # evaluation (which population/EA is being processed) using
        # the "whichComponent"  member variable
        self.problem = self.getCurrentDecoder().problem
  
        # obtain a fitness score for each interaction
        fitnesses = []
        for interaction in interactions:
            phenome = self.decodeGenome(interaction)
            currFitness = self.problem.evaluate(phenome)
            fitnesses.append(currFitness)
            self.storeBest(interaction,phenome,currFitness)

        # determine a single score from the collection of fitnesses        
        finalFitness = self.assessFitness(fitnesses)
        return(finalFitness)

      
    def getAllInteractions(self,genome,population):
        """
        This routine calls the select function, collects
        all representatives for the interactions, places
        the evaluated individual in the appropriate postion,
        and calls the above reformatting reoutine.  The
        result should be a list of lists, each of which
        is an appropriate set of representatives along with
        the individual under evaluation.
        """
        interactions = []
        for interaction in self.interactionList:
            pop = interaction[3]
            if (pop is population):
                interactions.append([genome])
            else:
                interactions.append(self.selectRepresentatives(interaction))
            
#        print("DEBUG:  [preformat]  ", interactions)
        newInteractions = self.reformatAllInteractions([],interactions)
#        print("DEBUG:  [reformated] ", newInteractions)
        
        return(newInteractions)


    def selectRepresentatives(self,interaction):
        """
        This routine uses the selection method associated
        with the appropriate interaction/population to
        choose the specified number of representatives
        """
        representatives = []
        population = interaction[3]
        pipeline = selector = interaction[0]
        n = interaction[1]
      
#        selector.newGeneration(population)
        selector.reinitialize(population)
        for i in range(0,n):
            representatives.append(selector.pull().genome)
        
        return(representatives)


    def reformatAllInteractions(self, currList,restList):
        """
        A simple recursive routine to "flatten" out the
        combinations of representative interactions.
        For instance, suppose there are three populations,
        and we are evaluating someone from the second,
        choosing two representatives from the other two
        populations.  The structure:
       
         [ [someguy1, someguy2], [myguy], [someguy3, someguy4] ]
       
        becomes:
         [ [someguy1, myguy, someguy3],
           [someguy1, myguy, someguy4],
           [someguy2, myguy, someguy3],
           [someguy2, myguy, someguy4] ]
        """
        resultList = []
        if (len(restList) <= 1):
            for i in restList[0]:
                resultList.append(currList + [i])
        else:
            for i in restList[0]:
                resultList = resultList + \
                   self.reformatAllInteractions(currList+[i], restList[1:])

        return(resultList)


    def setCurrentComponent(self, whichComponent):
        self.whichComponent = whichComponent
        self.problem = self.getCurrentDecoder().problem

    def getCurrentInteraction(self):
        return(self.interactionList[self.whichComponent])

    def getCurrentPopulation(self):
        interaction = self.interactionList[self.whichComponent]
        return(interaction[3])

    def getCurrentDecoder(self):
        interaction = self.interactionList[self.whichComponent]
        return(interaction[2])

    #------- Override the following functions to specialize -----

    def decodeGenome(self,interaction):
        """
        This routine is responsible for putting a given interaction
        in a form appropriate for the objective function(problem
        domain).
        """
        assembled = []
        componentDecoder = self.getCurrentDecoder()
        for phenome in interaction:
            assembled = assembled + componentDecoder.decodeGenome(phenome)
        return(assembled)
    

    def assessFitness(self,fitnesses):
        """
        This routine determines a single fitness score from multiple
        evaluations.  The simplest solution is generally the max
        function.  Mean is also common.
        """
        finalFitness = 0
        if (self.getCurrentDecoder().problem.maximize):
            finalFitness = max(fitnesses)
        else:
            finalFitness = min(fitnesses)

        return(finalFitness)


    def randomGenome(self):
        """
        This routine provides a random genome, but should rely
        on the "whichComponent" member variable being set to
        determine for which population it is generating
        such.
        """
        componentDecoder = self.getCurrentDecoder()
        return(componentDecoder.randomGenome())



#############################################################################
#
# ComponentEA
#
##############################################################################
class ComponentEA(LEAP.GenerationalEA):
    """
    A simple EA serving as a component to a coeovlutationary algorithm    
    """
    def __init__(self, decoder, pipeline, popSize,
                 interactSelector, numInteractors, problem):
        """
        The built-in creator method sets up internal member variables.
        The ComponentEA no longer automatically intializes (or even
        creates) the population.  These are not separate methods.
        """
        self.componentDecoder = decoder
        self.decoder = None
        self.pipeline = pipeline
        self.popSize = popSize
        self.population = []

        self.whichComponent = 0
        self.interactSelector = interactSelector
        self.numInteractors = numInteractors
        self.problem = problem

        self.bestOfGen = None
        self.bestSoFar = None
        self.generation = 0


    def createPopulation(self, coevolutionarydecoder, whichComponent):
        """
        Creates a population, but does not intialize it.  Here the
        decoder is set, individuals are created, and the population
        is populated.  No evalution for intialization purposes can
        happen until after ALL populations are created.  The
        whichComponent field is provided by the calling CEA.
        """
        self.decoder = coevolutionarydecoder
        self.whichComponent = whichComponent
        self.decoder.setCurrentComponent(whichComponent)
        # Create initial population
        for i in range(self.popSize):
            ind = LEAP.Individual(self.decoder)
            self.population.append(ind)


    def initialize(self):
        """
        This routine evaluates all individuals in a newly created population
        for the first time.  It must be separated from createPopulation,
        because it relies on the CoevolutionaryDecoder to connect it with all
        other populations for evaluation.
        """
        self.decoder.setCurrentComponent(self.whichComponent)
        
        # Create initial population
        for ind in self.population:
            ind.evaluate()

        self.bestOfGen = LEAP.fittest(self.population).clone()
        self.bestSoFar = LEAP.fittest(self.bestSoFar, self.bestOfGen).clone()
        self.printStats()

    
    def step(self):
        """
        This routine steps the current EA by first setting which
        component the decoder deals with, then calling the
        super class (GenerationaleA) step() function.
        """
        self.decoder.setCurrentComponent(self.whichComponent)
        self.decoder.preStepSetup()        
        LEAP.GenerationalEA.step(self)
        

    def run(self, maxGeneration):
        """
        The ComponentEA never runs on its own, but is only stepped by a CEA.
        """
        raise(NotImplementedError)


    def printStats(self):
        """
        Print the basic statistics of the population.  The routine prints best
        of generation and best so far results.
        """
        #self.printPopulation(self.population, self.generation)
        print("Gen:", self.generation, " Pop:", self.whichComponent,\
               " Ind: BOG ", self.bestOfGen)
        print("Gen:", self.generation, " Pop:", self.whichComponent,\
               " Ind: BSF ", self.bestSoFar)


    def printPopulation(self, population, generation = None):
        """
        Print all individuals in the population.
        """
        for i in range(len(population)):
            if generation != None:
                print("Gen:", generation, "Pop:",self.whichComponent, end='')
            print("Ind:", i, "", population[i])




#############################################################################
#
# SequentialCEA
#
##############################################################################
class SequentialCEA:
    """
    A Sequential Coevolutionary Algorithm.
    """
    def __init__(self, eaList):
        """
        The built-in creator method sets up the CoevolutionaryDecoder.
        It relies on a list of ComponentEAs to population the decoder's
        interactionList.  Each EA is assigned an index (whichComponent),
        so that the decoder can be properly informed.
        """        
        self.decoder = self.createCoevolutionaryDecoder()
        self.eaList = eaList
        
        idx = 0
        for ea in eaList:
            self.decoder.interactionList.append([ea.interactSelector,
                                                 ea.numInteractors,
                                                 ea.componentDecoder,
                                                 ea.population])
            ea.createPopulation(self.decoder,idx)
            idx = idx + 1
        self.currentEA = None

        for self.currentEA in self.eaList:
            self.currentEA.initialize()


    def createCoevolutionaryDecoder(self):
        """
        Allow me to override what kind of decoder is created...
        """
        return(CoevolutionaryDecoder())

    
    def run(self, maxGeneration):
        """
        This routine simply steps until the maximum number of generations is
        reached.
        """
        self.decoder.resetBest()
        for gen in range(1, maxGeneration + 1):
            self.step()

        print("Best interaction results from run:")
        print("  ", self.decoder.bestInteraction)
        print("  ", self.decoder.bestPhenome)
        print("  ", self.decoder.bestFitness)

        
    def step(self):
        """
        The step() method of the sequential coevoltuionary algorithm calls
        the individual EA step functions one at a time, for every EA.
        """
        for self.currentEA in self.eaList:
            self.currentEA.step()
            self.printStats()            

    def printStats(self):
        """
        Passes through to current component EA printStats
        """
        self.currentEA.printStats()

    def printPopulation(self, population, generation = None):
        """
        Passes through to current component EA printPopulation()
        """
        self.currentEA.printPopulation(self.currentEA.population,generation)


class BestAndRandomSelection(LEAP.SelectionOperator):
    """
    The selects 1 best individual and n-1 random individuals.
    It is a useful selection operator for collaboration selection,
    (e.g., Wiegand et al., 2000, Potter & De Jong, 1994).
    """

    def __init__(self, parentPopulation = []):
        self.drawIndex = 0
        LEAP.SelectionOperator.__init__(self,parentPopulation)

    def reinitialize(self, parentPopulation):
        self.drawIndex = 0        
        LEAP.SelectionOperator.reinitialize(self, parentPopulation)
        
    def apply(self, parents):
        if (self.drawIndex == 0):
            selected = LEAP.fittest(parents)
        else:
            selected = random.choice(parents)

        self.drawIndex = self.drawIndex + 1
        
        return [selected]



def unit_test():
    import LEAP
    
    passed = True

    # Test recursive routine
    testList  = [ ['a1', 'a2'], ['b1'], ['c1', 'c2'] ]
    solnList  = [ ['a1', 'b1', 'c1'],
                  ['a1', 'b1', 'c2'],
                  ['a2', 'b1', 'c1'],
                  ['a2', 'b1', 'c2'] ]
    print("Testing recursive member routine reformatAllInteractions():")
    print("  Input:  ", testList)
    testDecoder = CoevolutionaryDecoder()
    resultList = testDecoder.reformatAllInteractions([],testList)
    print("  Output: ", resultList)
    if (resultList != solnList):
        passed = False
        print("  Routine Failed!")
    else:
        print("  Routine Passed!")

    print("")
    print("Testing the CCEA on a three variable Schwefel function:")
    
    # Setup the problem
    function = LEAP.schwefelFunction
    bounds   = [(-500.00, 500.00)] * 3
    maximize = LEAP.schwefelMaximize
#    function = LEAP.sphereFunction
#    bounds   = LEAP.sphereBounds
#    maximize = LEAP.sphereMaximize
    numVars  = len(bounds)

    problem = LEAP.FunctionOptimization(function, maximize = maximize)

    eaList = []

    for idx in range(numVars):
        bitsPerReal = 16
        genomeSize = bitsPerReal

        # Setup the reproduction pipeline
        pipeline = LEAP.RankSelection()
        pipeline = LEAP.CloneOperator(pipeline)
        pipeline = LEAP.NPointCrossover(pipeline, 0.8, 2)
        pipeline = mutate = LEAP.BitFlipMutation(pipeline, 1.0/genomeSize)
        decoder = LEAP.BinaryRealDecoder(problem, [bitsPerReal], [bounds[idx]])
      
        # Other parameters
        popSize       = 20
        numCollab     = 2
#        collabSelector = UniformSelection()
        collabSelector = BestAndRandomSelection()        

        ea = ComponentEA(decoder, pipeline, popSize,
                         collabSelector, numCollab, problem)
        eaList.append(ea)

    ccea = SequentialCEA(eaList)
    maxGeneration = 20    
    ccea.run(maxGeneration)
    solutionFound = ccea.decoder.bestPhenome
    trueLocation = [421]*numVars
    distance = 0.0
    for idx in range(numVars):
        distance = distance + (
                   (trueLocation[idx] - abs(solutionFound[idx])) * 
                   (trueLocation[idx] - abs(solutionFound[idx])) )
    distance = math.sqrt(distance)
    
    # This is not technically true...there must be an odd number
    # of 
    print("We expect the solution to be somewhere near x_i=+-421.00 for", \
          " all i, with an odd number of '-' values.")
    print("The fitness of this is ", 3*-421*math.sin(math.sqrt(421)), ".")
          
    print("The CCEA found a solution roughly ", distance, " from", \
          " one of these corners.")
    if ( (distance > 150) or (ccea.decoder.bestFitness > -900) ):
        print("The final result was poorer than expected.")
        passed = False
    else:
        print("The final result seems in the right ball-park.")

    print("")
    
    if passed:
        print("Passed")
    else:
        print("FAILED")


if __name__ == '__main__':
    unit_test()
