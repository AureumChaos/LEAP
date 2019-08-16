#! /usr/bin/env python

# Python 2 & 3 compatibility
from __future__ import print_function

import sys
import random
import math

import LEAP
import LEAP.Coevolution


class SANECoevolutionaryDecoder(LEAP.Coevolution.CoevolutionaryDecoder):
    
    def evaluate(self,genome):
        """
        This method handles both EA evaluations, assembling blueprints
        to apply to the problem domain for the one, and assessing
        fitness based on prior participation for the other.
        """        
        blueprintInfo = self.interactionList[0]
        componentInfo = self.interactionList[1]
        blueprintPop = blueprintInfo[3]
        componentPop = componentInfo[3]

        # If we are processing the first/blueprint EA:
        if (self.whichComponent == 0):
#            componentPop.sort(LEAP.cmpInd)
            # Use the blueprint to assemble a complete solution,
            # translate it into a phenotype, then assess its fitness
            # against the specified problem
            phenome = self.blueprintAssemble(genome,componentPop,
                                             blueprintInfo[2],componentInfo[2])
            currFitness = self.getCurrentDecoder().problem.evaluate(phenome)
            self.storeBest(phenome,phenome,currFitness)

        # If we are processing the second/component EA:
        else:
            currFitness = None
            
        return(currFitness)


    def preStepSetup(self):
        blueprintInfo = self.interactionList[0]
        componentInfo = self.interactionList[1]
        blueprintPop = blueprintInfo[3]
        componentPop = componentInfo[3]
        
        if (self.whichComponent == 1):        
            for ind in componentPop:
                genome = ind.genome
                # Collect all the fitness values from the blueprints
                # in which this component participated, then assess
                # fitness as the sum of the best five of these
                # values.
                fitList = self.getAllFitnesses(blueprintPop,componentPop,
                                               genome,blueprintInfo[2])
                fitList.sort()
                idx = 0
                currFitness = 0.0
                for fit in fitList:
                    if (idx < 5):
                        currFitness = currFitness + fit
#                print("DEBUG: ", currFitness, fitList[0:5])
                ind.fitness = currFitness
#                self.storeBest(componentInfo,fitList,currFitness)


    def blueprintAssemble(self,blueprint,componentPop,
                          blueprintDecoder,componentDecoder):
        """
        Assemble a complete solution using the blueprint as an
        index into the component population.
        """
        completeSoln = []
        for comp in blueprint:
            idx = blueprintDecoder.decodeGenome(comp)
            phenome = componentDecoder.decodeGenome(componentPop[idx].genome)
            completeSoln = completeSoln + phenome
            
        return(completeSoln)


    def getAllFitnesses(self, blueprintPop, componentPop, genome,
                        blueprintDecoder):
        """
        Search through all blueprints for references to this
        genome, and append the fitness value of container blueprint
        to the fitness list.
        """
        compIdx = -1
        idx = 0
        for ind in componentPop:
            if (genome == ind.genome):
                compIdx = idx
            idx = idx + 1
#        if (compIdx < 0):
#            print("DEBUG:   --genome=",genome)
#            for ind in componentPop:
#                print("DEBUG:     ", ind)
            
        fitList = []
        for blueprint in blueprintPop:
            bNotFound = True            
            for comp in blueprint.genome:
                idx = blueprintDecoder.decodeGenome(comp)
                if (componentPop[idx].genome == genome):
                    bNotFound = False
                    fit = blueprint.fitness
                    if (fit == None):
                        fit = 0.0
                    fitList.append(fit)
#            if (bNotFound):    
#                print("DEBUG:   could not find", compIdx, " in blueprint",blueprint, "fitList=", fitList)
#            else:
#                print("DEBUG:   found", compIdx, " in blueprint",blueprint, "fitList=", fitList)
                    
        return(fitList)


class SANE(SequentialCEA):
    def createCoevolutionaryDecoder(self):
        return(SANECoevolutionaryDecoder())


class BinomialMutation(MutationOperator):
    def __init__(self, provider, pMutate, bounds, linear = True):
        MutationOperator.__init__(self, provider, pMutate, linear)
        self.bounds = bounds


    def drawBinomial(self,probability):
        num = 1
        draw = random.uniform(0.0,1.0)
        while (draw < probability):
            num = num + 1
            draw = random.uniform(0.0,1.0)
            
        return(num)
    

    def mutateGene(self, gene):
        dir = random.choice([-1,1])
        offset = self.drawBinomial(0.5)
        g = dir*offset + gene
        #Fix this!  RPW 25.5.2004
        g = min(99,max(0,g))

        return(g)


def unit_test():
    passed = True
    
    print("")
    print("Testing the SANE CCEA on a three variable Schwefel function:")
    
    # Setup the problem
#    function = LEAP.schwefelFunction
#    fBounds   = [(-500.00, 500.00)] * 3
#    maximize = LEAP.schwefelMaximize
    function = LEAP.boxFunction
    fBounds  = LEAP.boxBounds
    maximize = LEAP.boxMaximize
    numVars  = len(fBounds)
    problem = LEAP.FunctionOptimization(function, maximize = maximize)
    bitsPerReal = 16
    compPopSize       = 40
    blueprintPopSize  = 100    
    eaList = []

    genomeSize = numVars
    nBounds = [(0,compPopSize-1)] * numVars
    pipeline1 = LEAP.RankSelection()
    pipeline1 = LEAP.CloneOperator(pipeline1)
#    pipeline1 = BinomialMutation(pipeline1, 1.0/genomeSize, nBounds[0])
    pipeline1 = LEAP.UniformMutation(pipeline1,0.5,range(0,compPopSize))
    pipeline = fixup = LEAP.FixupOperator(pipeline1)    
    decoder1 = IntegerDecoder(problem, nBounds, nBounds)
    eaList.append(ComponentEA(decoder1, pipeline1, blueprintPopSize,
                              None, 0, problem))

    genomeSize = bitsPerReal
    pipeline2 = LEAP.RankSelection()
    pipeline2 = LEAP.CloneOperator(pipeline2)
    pipeline2 = LEAP.NPointCrossover(pipeline2, 0.8, 2)
    pipeline2 = mutate = LEAP.BitFlipMutation(pipeline2, 1.0/genomeSize)
    decoder2 = LEAP.BinaryRealDecoder(problem, [bitsPerReal], [fBounds[0]])
    eaList.append(ComponentEA(decoder2, pipeline2, compPopSize,
                              None, 0, problem))

    ccea = SANE(eaList)
    maxGeneration = 30    
    ccea.run(maxGeneration)
    solutionFound = ccea.decoder.bestPhenome
    trueLocation = (1,10,1)
    distance = 0.0
    for idx in range(numVars):
        distance = distance + (
                   (trueLocation[idx] - abs(solutionFound[idx])) * 
                   (trueLocation[idx] - abs(solutionFound[idx])) )
    distance = math.sqrt(distance)
    trueLocation = (10,1,-1)
    distance2 = 0.0
    for idx in range(numVars):
        distance2 = distance2 + (
                   (trueLocation[idx] - abs(solutionFound[idx])) * 
                   (trueLocation[idx] - abs(solutionFound[idx])) )
    distance = min(distance,math.sqrt(distance2))
    
    # This is not technically true...there must be an odd number
    # of 
    print("We expect the solution to be somewhere near (1,10,1) or (10,1,-1).")
    print("The fitness of this is %f." % boxFunction(trueLocation))
          
    print("The CCEA found a solution roughly ", distance, " from", \
          " one of these minima.")
    if ( (distance > 3.4) or (ccea.decoder.bestFitness > 1.0) ):
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
