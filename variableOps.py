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

"""
variableOps.py

Operators for variable length representations.  For now this mainly means
linear representations, such as messy GAs and Pitt approach classifiers.
If I decide to add operators for GP trees, I may put them in a separate
file.
"""

# Python 2 & 3 compatibility
from __future__ import print_function

import random

from LEAP.exceptions import OperatorError
from LEAP.operators import *


#############################################################################
#
# class VarUniformMutation
#
#############################################################################
class VarUniformMutation(UniformMutation):
    """
    Mutate an integer gene by uniformly selecting a new allele from a range
    of possible values.  The mutation rate is adjusted based on the genome
    size.
    """
    def __init__(self, provider, eMutate, alleles, linear = False):
        UniformMutation.__init__(self, provider, 0.0, alleles, linear)
        self.eMutate = eMutate   # expected number of mutation per genome

    def mutateIndividual(self, individual):
        self.pMutate = self.eMutate / size(individual.genome)
        UniformMutation.mutateIndividual(self, individual)



#############################################################################
#
# class VarBitFlipMutation
#
#############################################################################
class VarBitFlipMutation(VarUniformMutation):
    """
    Mutate a binary gene by flipping 0's to 1's and vice versa.
    The mutation rate is adjusted based on the genome size.
    """
    #eMutate = 1.0  # Expected number of mutations per individual
    parentsNeeded = 1

    def __init__(self, provider, eMutate = 1.0, alleles = ['0', '1'], 
                 linear = True):
        VarUniformMutation.__init__(self, provider, eMutate, alleles, linear)



#############################################################################
#
# class VarGaussianMutation
#
#############################################################################
class VarGaussianMutation(GaussianMutation):
    """
    Mutate an float gene by adding a delta drawn from a gaussian distribution.
    The mutation rate is adjusted based on the genome size.
    """
    def __init__(self, provider, sigma, eMutate, linear = False):
        GaussianMutation.__init__(self, provider, sigma, 0.0, linear)
        self.eMutate = eMutate   # expected number of mutation per genome

    def mutateIndividual(self, individual):
        self.pMutate = self.eMutate / size(individual.genome)
        GaussianMutation.mutateIndividual(self, individual)



#############################################################################
#
# class VarComponentGaussianMutation
#
#############################################################################
class VarComponentGaussianMutation(GaussianMutation):
    """
    This is for hierarchical type genomes, specifically rules system type
    genomes.  Each top level "component" (e.g. rule) is considered for
    mutation, and when a component is chosen for mutation, all floats within
    it will be mutated according to a fixed gaussian distribution determined
    by sigma.

    There are two mechanisms for setting the mutation rate at the component
    level.  One can set pMutateComponent which indicates the probability of
    mutation.  Alternatively, the eMutateComponent indicates how many
    components one expects to be mutated on average in a given call.
    """
    def __init__(self, provider, sigma, pMutateComponent = None,
                 eMutateComponent = None):
        GaussianMutation.__init__(self, provider, sigma, 1.0, linear=False)
        if pMutateComponent == None and eMutateComponent == None:
            raise ValueError(
                  "Neither pMutateComponent nor eMutateComponent is set")
        self.pMutateComponent = pMutateComponent  # probability of mutation
        self.eMutateComponent = eMutateComponent  # expected number of mutations


    def mutateIndividual(self, individual):
        if self.pMutateComponent:
            pMut = self.pMutateComponent
        else:
            pMut = self.eMutateComponent / size(individual.genome)
        self.numMut = 0

        newgenome = []
        for component in individual.genome:
            if random.random() < pMut:
                newcomponent, dummy = self.hierarchicalMutate(component,
                                                            self.nextMutation())
                newgenome.append(newcomponent)
            else:
                newgenome.append(component)

        individual.genome = newgenome
        individual.previous.append([individual])
        individual.numMut = self.numMut
        if self.numMut == 0:
            individual.genomeUnchanged()




#############################################################################
#
# Note to self:
#
# Currently adaptive mutation assumes a mutation rate of 1.0, so no point in
# writing a variable length version.
#
# The ProxyMutationOperator has a comment in it saying that things are not
# currently set up to handle hierarchical mutations.  Since Pitt genomes are
# heirarchical, I should check into what I meant by that before writing a
# variable length version.
#
#############################################################################



#############################################################################
#
# class VarAddGeneMutation
#
#############################################################################
class VarAddGeneMutation(MutationOperator):
    """
    Adds random genes to a genome.
    """
    def __init__(self, provider, eMutate, linear = False):
        MutationOperator.__init__(self, provider, 0.0, linear = False)
        self.eMutate = eMutate   # expected number of mutation per genome


    def mutateIndividual(self, individual):
        if len(individual.genome) == 0:
            self.numToAdd = 1   # Make sure there's at least 1 gene
        else:
            self.numToAdd = 0
            self.pMutate = self.eMutate / len(individual.genome)

            # Intercept encoding
            self.encoding = individual.encoding

            # Call superclass
            MutationOperator.mutateIndividual(self, individual)

        # Add the genes now.  Use the encoding to make this more general.
        while self.numToAdd > 0:
            tempGenome = self.encoding.randomGenome()
            individual.genome += tempGenome[:self.numToAdd]
            self.numToAdd -= len(tempGenome)


    def mutateGene(self, gene):
        self.numToAdd += 1  # Keep track of how many genes to add
        return gene



#############################################################################
#
# class VarDelGeneMutation
#
#############################################################################
class VarDelGeneMutation(MutationOperator):
    """
    A gene deletion operator.
    """
    def __init__(self, provider, eMutate, linear = False):
        MutationOperator.__init__(self, provider, 0.0, linear = False)
        self.eMutate = eMutate   # expected number of mutation per genome


    def mutateIndividual(self, individual):
        self.numToDelete = 0
        if len(individual.genome) > 0:
            # Adjust mutation rate based on genome length
            self.pMutate = self.eMutate / len(individual.genome)

            # Call superclass
            MutationOperator.mutateIndividual(self, individual)

        # Delete the genes now.  Never delete the last one.
        while self.numToDelete > 0 and len(individual.genome) > 1:
            individual.genome.pop(random.randrange(len(individual.genome)))
            self.numToDelete -= 1


    def mutateGene(self, gene):
        self.numToDelete += 1  # Keep track of how many genes to delete
        return gene



#############################################################################
#
# class VarReplaceMutation
#
#############################################################################
class VarReplaceMutation(MutationOperator):
    """
    A mutation operator which replaces each affected gene with a randomly
    generated new gene.

    Note: This operator was specifically designed to be used with Pitt
          approach representations, and it uses the encoding to generator the
          new genes.
    """
    parentsNeeded = 1

    def __init__(self, provider, pMutate, linear = True):
        """
        @param provider: The preceding operator in the pipeline.
        @param pMutate: Probability of mutating a gene.
        """
        MutationOperator.__init__(self, provider, pMutate, linear=True)


    def mutateIndividual(self, individual):
        "Intercept the encoding"
        self.encoding = individual.encoding
        MutationOperator.mutateIndividual(self, individual)


    def mutateGene(self, gene):
        "Generate a completely new gene"
        # Just generate a random genome and take one of its genes.
        tempGenome = self.encoding.randomGenome()
        return tempGenome[0]



#############################################################################
#
# class VarFuchsMutation
#
#############################################################################
class VarFuchsMutation(GeneticOperator):
    """
    The mutation operator used by Matthias Fuchs and Andreas Abecker in their
    paper "Optimized Nearest-Neighbor Classifiers Using Generated Instances",
    LNCS Vol 1137.
    """
    parentsNeeded = 1

    def __init__(self, provider, Pmut, Pdel, Prnd, Pcomp, Padd):
        """
        Initialize Fuchs mutation operator.

        If an individual is selected for mutation (i.e. random < Pmut), then
        all of the following occur in order.
        1) Delete: With probability Pdel, delete ONE random rule from the
                   genome.  If there is only one rule, never delete it.
        2) Mutate: Each and every rule in the genome is subject to mutation
                   with probability Prnd.  If a rule is chosen, then with
                   probability Pcomp each component (field in the condition or
                   action/class) is changed to a randomly generated value.
        3) Add:    With probability Padd, add a single randomly generated rule
                   to the genome.  Note that Fuchs would add multiple rules.

        @param provider  The preceding operator in the pipeline.
        @param Pmut      Probability that an individual is mutated.
        @param Pdel      Probability that a rule is deleted
        @param Prnd      Probability that a rule is modified
        @param Pcomp     Probability that a rule component is modified
        @param Padd      Probability that some rule are added
        """
        GeneticOperator.__init__(self, provider)
        # Check for errors
        if not 0.0 <= Pmut <= 1.0:
            raise ValueError("Pmut not in the range [0,1]")
        if not 0.0 <= Pdel <= 1.0:
            raise ValueError("Pdel not in the range [0,1]")
        if not 0.0 <= Prnd <= 1.0:
            raise ValueError("Prnd not in the range [0,1]")
        if not 0.0 <= Pcomp <= 1.0:
            raise ValueError("Pcomp not in the range [0,1]")
        if not 0.0 <= Padd <= 1.0:
            raise ValueError("Padd not in the range [0,1]")

        self.Pmut = Pmut
        self.Pdel = Pdel
        self.Prnd = Prnd
        self.Pcomp = Pcomp
        self.Padd = Padd


    def apply(self, individuals):
        for individual in individuals:
            if random.random() < self.Pmut:
                genome = individual.genome
                # Delete a rule?
                if random.random() < self.Pdel and len(genome) > 1:
                    del(genome[random.randrange(len(genome))])

                # Modify/mutate rules
                tempGenome = individual.encoding.randomGenome()
                for rule in genome:
                    if random.random() < self.Prnd:
                        tempRule = tempGenome.pop()
                        if tempGenome == []:
                            tempGenome = individual.encoding.randomGenome()
                        for i in range(len(rule)):
                            if random.random() < self.Pcomp:
                                rule[i] = tempRule[i]

                # Add some rules?
                if random.random() < self.Padd:
                    # Fuchs added a random number of rules.  Unlike him, I
                    # tend not to use an upper bound on the number of rules in
                    # an individual.  Therefore, I'll only add 1 rule.
                    genome.append(tempGenome[0])

        return individuals



#############################################################################
#
# class VarNPointCrossover
#
#############################################################################
class VarNPointCrossover(NPointCrossover):
    """
    Sections of both parents are recombined, but those sections may not
    necessarily be equal in size.  Perhaps the best example of this type of
    crossover is in Goldberg's Messy GA (I believe this is the same, but I'm
    not actually certain).
    """

    def recombine(self, child1, child2):
        # Check for errors.
        assert(len(child1.genome) >= self.numPoints + 1-int(self.allowXoverAt0))
        assert(len(child2.genome) >= self.numPoints + 1-int(self.allowXoverAt0))

        children = [child1, child2]
        genome1 = child1.genome[0:0]  # Create empty sequence - maintain type
        genome2 = child2.genome[0:0]
        src1, src2 = 0, 1

        # Pick crossover points
        xpts = [self.pickCrossoverPoints(self.numPoints, len(child.genome)) \
                for child in children]

        # Perform the crossover
        for i in range(len(xpts[0])-1):
            genome1 += children[src1].genome[xpts[src1][i]:xpts[src1][i+1]]
            genome2 += children[src2].genome[xpts[src2][i]:xpts[src2][i+1]]
            src1, src2 = src2, src1
            
        child1.genome = genome1
        child2.genome = genome2

        # Gather some statistics
        #child1.numSwaps = ...
        #child2.numSwaps = ...

        return (child1, child2)



#############################################################################
#
# class VarNPointGeneOffsetCrossover
#
#############################################################################
class VarNPointGeneOffsetCrossover(NPointCrossover):
    """
    This crossover operator is exactly like the ones used in traditional
    Pitt approach systems (e.g. Smith's LS1 and Spears' GABIL).

    It is assumed that the genomes take the following form:
       [ [rule] [rule] [rule] ... [rule] ]

    In other words, each rule is a list of items, and the genome is a list
    of rules.  It is also assumed that all the rules are the same size.
    """

    def pickCrossoverPoints(self, numPoints, genomeSize):
        """
        Randomly choose (without replacement) crossover points.

        In this version of the function, the point to the left of the first
        gene can also be a crossover point.
        """
        pp = list(range(genomeSize))  # Possible points
        xpts = [pp.pop(random.randrange(len(pp))) for i in range(numPoints)]
        xpts.sort()
        xpts = [0] + xpts + [genomeSize]  # Add start and end
        return xpts


    def recombine(self, child1, child2):
        # Check for errors.
        if len(child1.genome) < self.numPoints + 1 or \
           len(child2.genome) < self.numPoints + 1: 
            raise OperatorError("Not enough available crossover locations.")
            # This isn't quite true anymore.  This situation is a problem
            # though, since it means that the genome can't grow anymore.

        children = [child1, child2]

        # Pick crossover points
        xpts = [self.pickCrossoverPoints(self.numPoints, len(child.genome)) \
                for child in children]
        ruleLen = len(child1.genome[1])
        offsets = [random.randrange(ruleLen) for i in range(self.numPoints)]
        offsets = [0] + offsets + [0]

        # Perform the crossover
        src1, src2 = 0, 1
        genome1 = children[src1].genome[:xpts[src1][1]]
        genome2 = children[src2].genome[:xpts[src2][1]]
        for i in range(1,len(xpts[0])-1):
            genome1.append(children[src1].genome[xpts[src1][i]][:offsets[i]] +\
                           children[src2].genome[xpts[src2][i]][offsets[i]:])
            genome2.append(children[src2].genome[xpts[src2][i]][:offsets[i]] +\
                           children[src1].genome[xpts[src1][i]][offsets[i]:])
            src1, src2 = src2, src1

            genome1 += children[src1].genome[xpts[src1][i]+1:xpts[src1][i+1]]
            genome2 += children[src2].genome[xpts[src2][i]+1:xpts[src2][i+1]]
            
        child1.genome = genome1
        child2.genome = genome2

        # Gather some statistics
        #child1.numSwaps = ...
        #child2.numSwaps = ...

        return (child1, child2)



#############################################################################
#
# class VarUniformCrossover
#
#############################################################################
class VarUniformCrossover(CrossoverOperator):
    """
    Genes are exchanged between individuals depending on a coin flip.
    """
    parentsNeeded = 2

    def __init__(self, provider, pCross, pSwap, numChildren=2):
        CrossoverOperator.__init__(self, provider, pCross, numChildren)
        self.pSwap = pSwap


    def recombine(self, child1, child2):
        g1 = child1.genome[:]
        g2 = child2.genome[:]
        genome1 = child1.genome[0:0]  # empty sequence - maintain type
        genome2 = child2.genome[0:0]

        for i in range(len(g1)):
            g = g1[i:i+1]
            if random.random() < self.pSwap:
                genome2 += g
            else:
                genome1 += g

        for i in range(len(g2)):
            g = g2[i:i+1]
            if random.random() < self.pSwap:
                genome1 += g
            else:
                genome2 += g

        child1.genome = genome1;
        child2.genome = genome2;

        return(child1, child2)


#############################################################################
#
# class VarTransGeneCrossover
#
#############################################################################
class VarTransGeneCrossover(GeneticOperator):
    """
    A single gene is transfered from one parent to another.
    """
    parentsNeeded = 2

    def __init__(self, provider, pCross = 0.5):
        self.provider = provider
        self.pCross = pCross


    def apply(self, parents):
        children = []
        # Loop through all the parents, even if there are more than 2
        while len(parents) >= 2:
            parent1 = parents.pop(0)
            parent2 = parents.pop(0)
            if random.random() < self.pCross:   # Perform the crossover?
                i = random.randrange(len(parent1.genome))
                gene = parent1.genome.pop(i)
                parent2.genome.append(gene)
            children += [parent1, parent2]  # whether modified or not

        children += parents   # add remaining odd parent (if exists)
        return children



#############################################################################
#
# class VarSwapGenesCrossover
#
#############################################################################
class VarSwapGenesCrossover(GeneticOperator):
    """
    A small number of genes (number chosen randomly) are transferred from 
    parent1 to parent2.  Another set of genes (not necessarily the same
    number) are also transferred from the parent2 to the parent1.
    """
    parentsNeeded = 2

    def __init__(self, pCross = 0.5, maxSwap = 2, provider = None):
        self.pCross = pCross
        self.maxSwap = maxSwap
        self.provider = provider


    def apply(self, parents):
        children = []
        # Loop through all the parents, even if there are more than 2
        while len(parents) >= 2:
            parent1 = parents.pop(0)
            parent2 = parents.pop(0)
            g1 = parent1.genome[:]  # copy list, but not contents
            g2 = parent2.genome[:]
            genome1 = []
            genome2 = []
            numFrom1to2 = random.randrange(0,min(self.maxSwap+1,len(g1)))
            numFrom2to1 = random.randrange(0,min(self.maxSwap+1,len(g2)))
            for i in range(numFrom1to2):
                position = random.randrange(len(g1))
                genome2.append(g1.pop(position)) # without replacement

            for i in range(numFrom2to1):
                position = random.randrange(len(g2))
                genome1.append(g2.pop(position)) # without replacement

            genome1 = genome1 + g1
            genome2 = genome2 + g2

            parent1.genome = genome1;
            parent2.genome = genome2;
            children += [parent1, parent2]

        children += parents   # add remaining odd parent (if exists)
        return children



#############################################################################
#
# unit_test
#
#############################################################################
def unit_test():

    class MyIndividual:
        def __init__(self, genome):
            self.genome = genome

    # VarNPointCrossover
    ind1 = MyIndividual('abcdefgh')
    ind2 = MyIndividual('ABCDEFGH')

    xover = VarNPointCrossover(None, 1.0, numPoints=2, numChildren=2)
    child1, child2 = xover.recombine(ind1, ind2)

    print(child1.genome)
    print(child2.genome)

    # VarNPointGeneOffsetCrossover
    ind1 = MyIndividual(['abcd','abcd','abcd'])
    ind2 = MyIndividual(['ABCD','ABCD','ABCD'])

    xover = VarNPointGeneOffsetCrossover(None, 1.0, numPoints=2, numChildren=2)
    child1, child2 = xover.recombine(ind1, ind2)

    print(child1.genome)
    print(child2.genome)

    print("Passed?")


if __name__ == '__main__':
    unit_test()


