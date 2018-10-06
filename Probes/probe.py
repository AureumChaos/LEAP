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

#import sys
#import random
#import string
#import copy

import LEAP

from measures import *

from string import *
from math import *
from numpy.linalg import *
from numpy import *

from random import *

#import rpy2.robjects as robjects


#############################################################################
#
# ProbeOperator
#
#############################################################################
class ProbeOperator(LEAP.PipelineOperator):
    """
    This is the base class for all probes.  It essentially does 3 things:
    store the measureFunction, keep track of the generation and define the
    examineTraits() interface.  
    """
    # As it turns out, some of the sub-classes will redefine the parameters to
    # examineTraits(), so maybe it doesn't really belong here.

    parentsNeeded = 1

    def __init__(self, provider, measureFunction, frequency=1):
        LEAP.PipelineOperator.__init__(self, provider)
        self.measureFunction = measureFunction
        self.frequency = frequency
        self.generation = 0

    def reinitialize(self, population):
        LEAP.PipelineOperator.reinitialize(self, population)
        self.generation += 1

    def examineTraits(self, traits):
        """
        Aggregate the traits and record them somehow (e.g. print them,
        write them to file, etc).
        """
        raise(NotImplementedError)   # Subclasses should redifine this



#############################################################################
#
# PopulationProbe
#
#############################################################################
class PopulationProbe(ProbeOperator):
    """
    Collect traits of all individuals as they pass down the pipeline.
    Subclasses of this operator can be placed at a specific point in the
    pipeline in order to probe the state of the population at that point.

    Subclasses should override the examine() method in order to aggregate
    the traits and then display or record them.
    """
    parentsNeeded = 1

    def __init__(self, provider, measureFunction, frequency=1):
        ProbeOperator.__init__(self, provider, measureFunction, frequency)
        self.traits = []

    def reinitialize(self, population):
        if self.traits != []:   # No individuals pulled the first time
            self.examineTraits(self.traits)
        ProbeOperator.reinitialize(self, population)
        self.traits = []

    def examineTraits(self, traits):
        """
        Aggregate the traits and record them somehow (e.g. print them,
        write them to file, etc).
        """
        raise(NotImplementedError)   # Subclasses should redifine this

    def apply(self, children):
        if self.generation % self.frequency == 0 or self.generation == 1:
            self.traits += [self.measureFunction(child) for child in children]
        return children



#############################################################################
#
# AveragePopulationProbe
#
#############################################################################
class AveragePopulationProbe(PopulationProbe):
    """
    Calculate the population average for a given trait.
    """
    def examineTraits(self, traits):
        avg = average(traits, 0)
        print("Gen:", self.generation, "PopAvg:", LEAP.mystr(avg))



#############################################################################
#
# OutputPopulationProbe
#
#############################################################################
class OutputPopulationProbe(PopulationProbe):
    """
    Print pairings of parents and offspring.
    """
    def __init__(self, provider, measureFunction, baseFilename, \
                 tag=None, fileSuffix = None, frequency=1):
        PopulationProbe.__init__(self, provider, measureFunction, frequency)
        suffix = fileSuffix
        if not fileSuffix:
            suffix = ".out"
            if tag:
                suffix += tag

        self.outFile = open(baseFilename + suffix, "w")
        self.first = True
        self.row = 1
        self.tag = tag


    def examineTraits(self, allTraits):
        assert(len(allTraits) > 0)
        firstTrait = allTraits[0]

        #sep = ",\t"
        sep = "\t"
        if self.first:
            self.first = False
            indepVars = ["Generation", "Index"]
            if self.tag:
                indepVars.append("Tag")

            if (getattr(firstTrait, '__iter__', False)):
                traitNames = ["Trait"+str(p+1) for p in range(len(firstTrait))]
            else:
                traitNames = ["Trait1"]

            header = sep.join(indepVars + traitNames)
            self.outFile.write(header + "\n")

        for i, traits in enumerate(allTraits):
            fields = [str(self.row), str(self.generation), str(i+1)]
            self.row += 1
            if self.tag:
                fields.append(str(self.tag))

            if (getattr(traits, '__iter__', False)):
                # Parents and offspring should have the same type of traits
                fields += [str(t) for t in traits]
            else:
                fields += [str(traits)]

            self.outFile.write(sep.join(fields) + "\n")



#############################################################################
#
# PopulationDiversityProbe
#
#############################################################################
class PopulationDiversityProbe(PopulationProbe):
    """
    Calculate the population diversity for a given trait or set of traits.
    This is the method that Ron Morrison proposed.
    """
    def __init__(self, provider, measureFunction, tag=None, frequency=1):
        PopulationProbe.__init__(self, provider, measureFunction, frequency)
        self.tag = tag

    def examineTraits(self, traits):
        if getattr(traits[0], '__iter__', False):   # is it a sequence?
            myTraits = array(traits)
        else:
            myTraits = array([[t] for t in traits])

        myTraits = array(myTraits)
        centroid = mean(myTraits,0)
        moi = sum((myTraits - centroid)**2)

        print("Gen:", self.generation, end='')
        if self.tag:
            print(" Tag:", self.tag, end='')
        print(" Diversity:", moi)



#############################################################################
#
# ParentPopulationProbe
#
#############################################################################
class ParentPopulationProbe(PopulationProbe):
    """
    A ParentPopulationProbe and OffspringPopulationProbe (or subclass) are
    used to stradle a set of operators in the pileline.  Populations are
    recorded in both places so that they can then be compared.  This allows
    one to observe effects of the stradled operators.  Note that no
    relationship between individual parents and offspring can be assumed.
    """
    def getParentTraits(self):
        return self.traits

    def examineTraits(self, Traits):
        pass



#############################################################################
#
# OffspringPopulationProbe
#
#############################################################################
class OffspringPopulationProbe(PopulationProbe):
    """
    A ParentPopulationProbe and OffspringPopulationProbe (or subclass) are
    used to stradle a set of operators in the pileline.  Populations are
    recorded in both places so that they can then be compared.  This allows
    one to observe effects of the stradled operators.  Note that no
    relationship between individual parents and offspring can be assumed.
    """
    def __init__(self, provider, measureFunction, parentProbe, tag=None, \
                 frequency=1):
        PopulationProbe.__init__(self, provider, measureFunction, frequency)
        self.parentProbe = parentProbe
        self.tag = tag

    def examineTraits(self, traits):
        self.compareTraits(self.parentProbe.getParentTraits(), self.traits)

    def compareTraits(self, parentTraits, offspringTraits):
        raise(NotImplementedError)    # Subclasses should override this


#############################################################################
#
# ParentFamilyProbe
#
#############################################################################
class ParentFamilyProbe(ProbeOperator):
    """
    This probe is designed to measure the effects of operators at the family
    level rather than the population level.  A family can be thought of as
    all the individuals involved in a single application of the operator(s)
    being measured.  For example, a mutation operator takes in one parent and
    produces one offspring, but crossover may take in two parents and produce
    two offspring.

    A ParentFamilyProbe and OffspringFamilyProbe (or subclass) are
    used to stradle a set of operators in the pipeline.  Individuals are
    recorded in both places so that they can then be compared.  This allows
    one to observe effects of the stradled operators.

    Remember that most operators (except the CloneOperator) do not create new
    individuals, but instead pass along modified versions of what they
    received.  Keep this in mind when deciding when and where to make
    measurements.

    The ultimate goal here is to make these probes work with little to no
    knowledge about the operators that they are wrapping.  Probably the most
    difficult aspect of this is figuring out how to match up individuals that
    come into this probe with those that come into the OffspringFamilyProbe.
    If everything is working properly, this should be automatic.
    """
    parentsNeeded = 1

    def __init__(self, provider, measureFunction, frequency=1):
        ProbeOperator.__init__(self, provider, measureFunction, frequency)
        self.offspringProbe = None

    def setOffspringFamilyProbe(self, offspringProbe):
        self.offspringProbe = offspringProbe

    def apply(self, parents):
        if self.generation % self.frequency == 0 or self.generation == 1:
            for parent in parents:
                parentTrait = self.measureFunction(parent)
                self.offspringProbe.submitParentTrait(parentTrait)
        return parents



#############################################################################
#
# Builtin associate functions for family probes
#
#############################################################################
def midparentAssociate(parentTraits, offspringTraits):
    return [mean(parentTraits)] * len(offspringTraits), offspringTraits

def midchildAssociate(parentTraits, offspringTraits):
    return parentTraits, [mean(offspringTraits)] * len(parentTraits)

def midparentMidchildAssociate(parentTraits, offspringTraits):
    return [mean(parentTraits)], [mean(offspringTraits)]

def combinatorialAssociate(parentTraits, offspringTraits):
    return [p for p in parentTraits for o in offspringTraits], \
           [o for p in parentTraits for o in offspringTraits]



#############################################################################
#
# OffspringFamilyProbe
#
#############################################################################
class OffspringFamilyProbe(ProbeOperator):
    """
    This probe is designed to measure the effects of operators on individuals,
    and the population as a whole.  Probes subclassed from this should be used
    in conjunction with a corresponding subclass of ParentFamilyProbe.  This
    probe will be placed in the pipeline after an operator (or set of
    operators), and the ParentFamilyProbe will be placed before.

    Remember that most operators (except the CloneOperator) do not create new
    individuals, but instead pass along modified versions of what they
    receive.  Keep this in mind when deciding when and where to make
    measurements.

    NOTE: If this class is given multiple providers, it will ignore all but
          the first.  Also, there should be no operator between the parent and
          offspring probes that uses multiple providers.

          A look-ahead approach is used to aid matching parents and children.
          As a result, one extra individual will be pulled down the pipeline
          to this point.  It will not be passed on any further though, nor
          will it be passed to the apply method.  This could cause some side
          effects if any probes are placed before these in the pipeline.
          Those earlier probes might record measurements from this extra
          individual.  A simple fix might be to tell this probe what the
          population size is so that it doesn't need to make that last pull
          (Now implemented, see expectedPulls).

          Since the relationship between parents and offspring is inferred by
          the timing by which individuals come down the pipeline, any operator
          that alters the default timing could cause problems.  For example,
          the standard mutation and crossover operators will release children
          as soon as enough parents have been pulled to produce them.  On the
          other hand, SurvivalSelection will not release any children until
          all the necessary parents have been pulled, thus making it
          impossible to determine which parents produced which offspring.
          Putting a survival selection operator between parent and offspring
          probes could also cause problems.

          This is turning into a lot of caveats :(
    """
    parentsNeeded = 1

    def __init__(self, provider, measureFunction, parentProbe, \
                 associateFunc = combinatorialAssociate, \
                 expectedPulls=None, tag=None, frequency=1):
        assert(isinstance(provider, LEAP.Operator))
        ProbeOperator.__init__(self, provider, measureFunction, frequency)
        self.parentProbe = parentProbe
        parentProbe.setOffspringFamilyProbe(self)  # for comm

        # submittedTraits contains Latest batch of traits submitted by the
        # ParentFamilyProbe
        self.submittedTraits = []  

        # parentTraits match the individuals sent to apply()
        self.parentTraits = []

        # lookahead contains the most recent individual pulled.
        self.lookahead = []

        # All parent traits collected this generation
        self.allParentTraits = []
        # All child traits collected this generation.  Parent and Offspring
        # traits will match up.  When using crossover, duplicates will exist
        # so that all parents and children are matched.
        self.allOffspringTraits = []

        self.currentPulls = 0      # Number of pulls since last reset

        # expectedPulls should either be the total number of pulls expected in
        # a given generation (often popSize), or the number of pulls per
        # family (e.g.  1 for mutation or 2 for crossover).  In some cases it
        # may not be possible to know how many pulls will be performed.
        #
        # This is parameter is optional, but it is a good idea to use it when
        # combining multiple probes in a single pipeline.
        self.expectedPulls = expectedPulls

        self.associateFunc = associateFunc
        self.tag = tag   # Used for disambiguating output


    def reinitialize(self, population):
        if self.allParentTraits != []:  # No individuals pulled on first call
            self.compareTraits(self.allParentTraits, self.allOffspringTraits)
        ProbeOperator.reinitialize(self, population)
        self.submittedTraits = []
        self.parentTraits = []   # Contains the parent traits associated
                                 # with the children sent to apply()
        if self.lookahead != []:
            print("Warning: Probe pulled too many children")
            self.lookahead = []  

        self.allParentTraits = []
        self.allOffspringTraits = []
        self.currentPulls = 0


    def submitParentTrait(self, parentTrait):
        """
        Should be called in apply() of a corresponding ParentFamilyProbe.  A
        "parentTrait" is a measurement made on an individual at an earlier
        point in the pipeline.
        """
        self.submittedTraits.append(parentTrait)


    def applyAndCache(self):
        individuals = [self.provider[0].pull()]

        while self.provider[0].isAnythingCached(after = self.parentProbe):
            individuals.append(self.provider[0].pull())

        self.parentTraits = self.submittedTraits
        self.submittedTraits = []

        # self.resetFitnesses(individuals)  # Subclasses may want to do this
        children = self.apply(individuals)
        self.addToCache(children)


#    def applyAndCache(self):
#        # This gets a little complicated.  My hope is that subclasses will
#        # be fairly simple as a result.
#        individuals = []
#
##        print("self.expectedPulls =", self.expectedPulls)
#        if self.lookahead == []:  # First pull this gen, or since reset?
##            print("get lookahead")
#            self.lookahead = [self.provider[0].pull()]
#            self.currentPulls += 1
#
#        # All parent traits for a family should already have been submitted
#        # once the first child of that family is pulled.  We may not know how
#        # many children to expect though.  Keep pulling children until a new
#        # batch of parent traits get submitted, then we know we've gone too
#        # far.  If we know how many pulls to expect, we can stop in time and
#        # avoid pulling extra individuals down the pipeline.
#        self.parentTraits = self.submittedTraits
#        self.submittedTraits = []
##        print("self.currentPulls =", self.currentPulls)
#        while self.submittedTraits == []:
#            # Prevent lookahead if possible
#            individuals += self.lookahead
#            if self.expectedPulls and self.expectedPulls == self.currentPulls:
#                self.currentPulls = 0
#                self.lookahead = []
##                print("    currentPulls == expectedPulls.  Reset")
#                break
#            self.lookahead = [self.provider[0].pull()]
#            self.currentPulls += 1
##            print("    len(individuals) =", len(individuals))
##            print("    len(self.lookahead) =", len(self.lookahead))
#
##        print("len(self.submittedTraits) =", len(self.submittedTraits) )
##        print("self.currentPulls =", self.currentPulls)
#
#        # self.resetFitnesses(individuals)  # Subclasses may want to do this
#        children = self.apply(individuals)
#        self.addToCache(children)


    def apply(self, individuals):
        if self.generation % self.frequency == 0 or self.generation == 1:
            offspringTraits = [self.measureFunction(i) for i in individuals]
            assocParents, assocOffspring = self.associateTraits( \
                                           self.parentTraits, offspringTraits)

        self.allParentTraits += assocParents
        self.allOffspringTraits += assocOffspring
        return individuals


    def associateTraits(self, parentTraits, offspringTraits):
        """
        Create two equal sized sets of traits that have a one to one
        relationship with each other.  The lists returned here will be
        gathered together and passed to compareTraits() at the end of a
        generation.

        This function offers the opportunity to perform intermediate
        calculations like midparent (average parent traits) or midchild
        (average child traits).  Of course you could just forget about the
        compareTraits() method altogether and perform all your calculations
        here.
        """
        return self.associateFunc(parentTraits, offspringTraits)


    def compareTraits(self, allParentTraits, allOffspringTraits):
        """
        This method is called once at the end of each generation.
        The traits of all parents and offspring can be compared at once.
        """
        raise(NotImplementedError)    # Subclasses should override this



#############################################################################
#
# TestOffspringFamilyProbe
#
#############################################################################
class TestOffspringFamilyProbe(OffspringFamilyProbe):
    """
    Print pairings of parents and offspring.
    """
    def associateTraits(self, parentTraits, offspringTraits):
        numRows = max(len(parentTraits), len(offspringTraits))
        for i in range(numRows):
            if i < len(self.parentTraits):
                print(self.parentTraits[i], end='')
            print("\t", end='')
            if i < len(offspringTraits):
                print(offspringTraits[i], end='')
            print()

        print("currentPulls = ", self.currntPulls)
        print()

        return OffspringFamilyProbe.associateTraits(self, parentTraits,
                                                          offspringTraits)

    def compareTraits(self, allParentTraits, allOffspringTraits):
        for parentT, offspringT in zip(allParentTraits, allOffspringTraits):
            print(parentT, offspringT)
        print()



#############################################################################
#
# OutputFamilyProbe
#
#############################################################################
class OutputFamilyProbe(OffspringFamilyProbe):
    """
    Print pairings of parents and offspring.
    """
    def __init__(self, provider, measureFunction, parentProbe, baseFilename, \
                 associateFunc = combinatorialAssociate, \
                 expectedPulls=None, tag=None, \
                 parentFileSuffix = None, offspringFileSuffix = None, \
                 frequency = 1):
        OffspringFamilyProbe.__init__(self, provider, measureFunction,
                         parentProbe, associateFunc = combinatorialAssociate, \
                         expectedPulls=expectedPulls, tag=tag,\
                         frequency=frequency)
        Psuffix = parentFileSuffix
        if not parentFileSuffix:
            Psuffix = ".P"
            if tag:
                Psuffix += tag

        Osuffix = offspringFileSuffix
        if not offspringFileSuffix:
            Osuffix = ".O"
            if tag:
                Osuffix += tag

        self.parentsFile = open(baseFilename + Psuffix, "w")
        self.offspringFile = open(baseFilename + Osuffix, "w")
        self.first = True
        self.row = 1


    def compareTraits(self, allParentTraits, allOffspringTraits):
        """
        Output all trait information recorded this generation.
        This function gets called once per generation, and all the traits
        that have been recorded this generation are passed in.  There
        should be a direct correspondence between allParentTraits[i] and 
        allOffspringTraits[i].
        """
        if len(allParentTraits) == 0 or len(allOffspringTraits) == 0:
            return
        firstTrait = allParentTraits[0]

        #sep = ",\t"
        sep = "\t"
        if self.first:
            self.first = False
            indepVars = ["Generation", "Index", "FamilyRole"]
            if self.tag:
                indepVars.append("Tag")

            if (getattr(firstTrait, '__iter__', False)):
                traitNames = ["Trait"+str(p+1) for p in range(len(firstTrait))]
            else:
                traitNames = ["Trait"]

            header = sep.join(indepVars + traitNames)
            self.parentsFile.write(header + "\n")
            self.offspringFile.write(header + "\n")

        for i, (parentT, offspringT) in \
                     enumerate(zip(allParentTraits, allOffspringTraits)):
            #print(i, parentT)
            #print(i, offspringT)
            #print()
            pfields = [str(self.row), str(self.generation), str(i+1), "parent"]
            ofields = [str(self.row), str(self.generation), str(i+1), "offspring"]
            self.row += 1
            if self.tag:
                pfields.append(str(self.tag))
                ofields.append(str(self.tag))

            if (getattr(parentT, '__iter__', False)):
                # Parents and offspring should have the same type of traits
                pfields += [str(t) for t in parentT]
                ofields += [str(t) for t in offspringT]
            else:
                pfields += [str(parentT)]
                ofields += [str(offspringT)]

            self.parentsFile.write(sep.join(pfields) + "\n")
            self.offspringFile.write(sep.join(ofields) + "\n")



#############################################################################
#
# OutputAssocFamilyProbe
#
#############################################################################
class OutputAssocFamilyProbe(OffspringFamilyProbe):
    """
    Print pairings of parents and offspring.
    """
    def __init__(self, provider, measureFunction, parentProbe, baseFilename, \
                 associateFunc = combinatorialAssociate, \
                 expectedPulls=None, tag=None, fileSuffix = None, frequency=1):
        OffspringFamilyProbe.__init__(self, provider, measureFunction,
                         parentProbe, associateFunc = combinatorialAssociate, \
                         expectedPulls=expectedPulls, tag=tag,\
                         frequency=frequency)
        Dsuffix = fileSuffix
        if not fileSuffix:
            Dsuffix = ".A"
            if tag:
                Dsuffix += tag

        self.deltaFile = open(baseFilename + Dsuffix, "w")
        self.first = True
        self.row = 1
        self.numParentsInFamily = []


    def reinitialize(self, population):
        OffspringFamilyProbe.reinitialize(self, population)
        self.numParentsInFamily = []


    def associateTraits(self, parentTraits, offspringTraits):
        """
        Create two equal sized sets of traits that have a one to one
        relationship with each other.  This modified version also tracks the
        number of parents in each family.
        """
        # Calculate associated parent Traits (apT) and associated offspring
        # Traits (aoT)
        apT, aoT = OffspringFamilyProbe.associateTraits(self, parentTraits,
                                                              offspringTraits)
        # Append the appropriate number of 
        numParents = len(parentTraits)
        for t in apT:
            self.numParentsInFamily.append(numParents)
        return (apT, aoT)


    def compareTraits(self, allParentTraits, allOffspringTraits):
        """
        Calculate and output all trait deltas between parents and offspring
        recorded this generation.  This function gets called once per
        generation, and all the traits that have been recorded this generation
        are passed in.  There should be a direct correspondence between
        allParentTraits[i] and allOffspringTraits[i].
        """
        if len(allParentTraits) == 0 or len(allOffspringTraits) == 0:
            print("Error: Incompatable number of traits")
            return
        firstTrait = allParentTraits[0]

        #sep = ",\t"
        sep = "\t"
        if self.first:
            self.first = False
            indepVars = ["Generation", "Index"]
            if self.tag:
                indepVars.append("Tag")

            traitNames = ["NumParents"]
            if (getattr(firstTrait, '__iter__', False)):
                traitNames +=["PTrait"+str(p+1) for p in range(len(firstTrait))]
                traitNames +=["OTrait"+str(p+1) for p in range(len(firstTrait))]
            else:
                traitNames +=["PTrait1 OTrait1"]

            header = sep.join(indepVars + traitNames)
            self.deltaFile.write(header + "\n")

        for i, (parentT, offspringT) in \
                     enumerate(zip(allParentTraits, allOffspringTraits)):
            #print(i, parentT)
            #print(i, offspringT)
            #print()
            dfields = [str(self.row), str(self.generation), str(i+1)]
            self.row += 1
            if self.tag:
                dfields.append(str(self.tag))

            dfields.append(str(self.numParentsInFamily[i]))
            if (getattr(parentT, '__iter__', False)):
                # Parents and offspring should have the same type of traits
                dfields += [str(pt) for pt in parentT]
                dfields += [str(ot) for ot in offspringT]
            else:
                dfields += [str(parentT), str(offspringT)]

            self.deltaFile.write(sep.join(dfields) + "\n")



#############################################################################
#
# UnivariateHeritabilityProbe
#
#############################################################################
class UnivariateHeritabilityProbe(OffspringFamilyProbe):
    """
    Calculates parent-offspring correlation.
    """
    def compareTraits(self, parentTraits, offspringTraits):
        m = cov(parentTraits, offspringTraits)
        h2 = m[0][1] / m[0][0]
        print("Gen:", self.generation,  end='')
        if self.tag:
            print(" Tag:", self.tag, end='')
        print(" h^2:", h2)



#############################################################################
#
# AverageDistanceProbe
#
#############################################################################
class AverageDistanceProbe(OffspringFamilyProbe):
    """
    Calculates the average distance between parent and offspring traits
    """
    def compareTraits(self, parentTraits, offspringTraits):
        distBar = mean([sqrt(sum((o-p)*(o-p))) for p,o in \
                                       zip(parentTraits, offspringTraits)])
        #distBar = mean([sqrt(sum(o-p * o-p)) for p,o in
        #                               zip(parentTraits, offspringTraits)])
        #delta = offspringTraits[0] - parentTraits[0]
        #distBar = sqrt(sum(delta * delta))
        print("Gen:", self.generation,  end='')
        if self.tag:
            print(" Tag:", self.tag, end='')
        print(" AvgDist:", distBar)



#############################################################################
#
# CorrelationProbe
#
#############################################################################
class CorrelationProbe(OffspringFamilyProbe):
    """
    Calculates parent-offspring correlation.
    """
    def compareTraits(self, parentTraits, offspringTraits):
        corr = corrcoef(parentTraits, offspringTraits)
        #print(zip(parentTraits, offspringTraits)
        if isnan(corr[0,1]):
            print(len(parentTraits), len(offspringTraits))
        print("Gen:", self.generation,  end='')
        if self.tag:
            print(" Tag:", self.tag, end='')
        print(" Correlation:", corr[0,1])



#############################################################################
#
# KullbackLeiblerProbe
#
#############################################################################
class KullbackLeiblerProbe(OffspringPopulationProbe):
    """
    This probe attempts to quantify to notion of heritability using the
    Kullback-Leiber diververgence.  This assumes that the parent and offspring
    traits are normally distributed.  A very big assumption, and often
    unlikely to be true.
    """
    def KLdivergence(self, mu0, Sigma0, mu1, Sigma1):
        """
        This is a normalized version of Kullback-Leibler divergence.  A
        similarity metric (in the range [0,1]) is calculated between two
        multivariate normal distributions.
        """
        N = len(mu0)
        assert(len(mu1) == N)
        assert(shape(Sigma0) == shape(Sigma1) == (N,N))

        u0 = transpose(mat(mu0))
        u1 = transpose(mat(mu1))
        S0 = mat(Sigma0)
        S1 = mat(Sigma1)

        t1 = log(det(S1) / det(S0))
        t2 = trace(inv(S1)* S0)
        d = (u1 - u0)
        t3 = float(d.T * inv(Sigma1) * d)

        kl = (t1 + t2 + t3 - N) / 2
        return kl


    def compareTraits(self, parentTraits, offspringTraits):
        # Should probably put some error checking here.
        mu0 = mean(parentTraits, axis=0)
        Sigma0 = cov(array(parentTraits).T)
        mu1 = mean(offspringTraits, axis=0)
        Sigma1 = cov(array(offspringTraits).T)

        kl_01 = self.KLdivergence(mu0, Sigma0, mu1, Sigma1)
        kl_10 = self.KLdivergence(mu1, Sigma1, mu0, Sigma0)
        sim = 1/(1 + kl_01 + kl_10)
        print("Gen:", self.generation, end='')
        if self.tag:
            print(" Tag:", self.tag, end='')
        print(" KL_similarity:", sim)

        #print("Sigma0:")
        #print(Sigma0)
        #print("Sigma1:")
        #print(Sigma1)



#############################################################################
#
# NormalityPopulationProbe
#
#############################################################################
class NormalityPopulationProbe(PopulationProbe):
    """
    Checks the population at the current point in the population to see if it
    is normally distributed.
    """
    def __init__(self, provider, measureFunction, tag=None, frequency=1):
        PopulationProbe.__init__(self, provider, measureFunction, frequency)
        self.tag = tag

        import rpy2.robjects as robjects
        robjects.r("library(dprep)")
        robjects.r("source('~/Rutils/myMardia.r')")
        self.robj = robjects


    def makeDataFrame(self, dfName, traits):
        """
        Make an R data.frame specifically formatted for used with the
        mardia() test for multivariate normality.
        """
        traitArray = array(traits)

        # Create a data.frame with the # right row num.
        self.robj.r("T1 <- 1:%d" % len(traitArray))
        self.robj.r(dfName + " <- data.frame(T1)")

        for col in range(len(traitArray[0,:])):
            traitName = "T" + str(col+1)
            traitVec = self.robj.FloatVector(traitArray[:,col])
            self.robj.r.assign('tempTraits', traitVec)
            self.robj.r(dfName + "$" + traitName + " <- tempTraits")

        self.robj.r("%s$Class = rep(1,%d)" % (dfName, len(traitArray)))


    def examineTraits(self, traits):
        self.makeDataFrame("df", traits)
        result = self.robj.r("myMardia(df)")
        print("Gen:", self.generation, end='')
        if self.tag:
            print(" Tag:", self.tag, end='')
        print(" Norm:", result[0] == 1.0, end='')
        print(" P1:", result[1], end='')
        print(" P2:", result[2])


#############################################################################
#
# MultivariateOffspringFamilyProbe
#
#############################################################################
class MultivariateOffspringFamilyProbe(OffspringFamilyProbe):
    """
    Calculates parent-offspring correlation for multiple traits.

    Oversampling can also be done.  In other words, instead of just sampling
    the population, extra children are pulled down the pipeline in order to
    get more accurate statistics.  The children never make it any further down
    the pipeline though.  Be careful when using this feature though since it
    could have some strange side effects, particularly on other probes that
    are upstream in the pipeline.

    Note: Oversampling is a little broken, but it should work OK if you set
          expectedPulls.  Using number of offspring per family is safest.
    """
    def __init__(self, provider, measureFunction, parentProbe, \
                 expectedPulls = None, filenamebase = "results", suffix = "", \
                 oversampleInterval = 0, oversampleSize = 0, \
                 oversampleSuffix = "o", frequency=1):
        OffspringFamilyProbe.__init__(self, provider, measureFunction, \
                                      parentProbe, expectedPulls, \
                                      frequency=frequency)
        self.oversampleInterval = oversampleInterval
        self.oversampleSize = oversampleSize

        s = suffix
        self.Pfile = open(filenamebase + ".P" + s, "w")
        self.Ofile = open(filenamebase + ".O" + s, "w")
        self.Gfile = open(filenamebase + ".G" + s, "w")
        self.h2file = open(filenamebase + ".h2" + s, "w")
        self.H2file = open(filenamebase + ".H2" + s, "w")
        self.dQfile = open(filenamebase + ".dQ" + s, "w")

        numVars = len(measureFunction())
        header = "Gen NumSamples " + \
                 join("C"+str(x)+"."+str(y) for x in range(numVars) \
                                            for y in range(numVars)) + "\n"
        self.Pfile.write(header)
        self.Ofile.write(header)
        self.Gfile.write(header)
        self.h2file.write(header)
        self.H2file.write(header)
        self.dQfile.write(header)

        self.gen = 0
        self.gensSinceOversample = 0

        if oversampleInterval > 0:
            os = oversampleSuffix
            self.overPfile = open(filenamebase + ".P" + os, "w")
            self.overOfile = open(filenamebase + ".O" + os, "w")
            self.overGfile = open(filenamebase + ".G" + os, "w")
            self.overh2file = open(filenamebase + ".h2" + os, "w")
            self.overH2file = open(filenamebase + ".H2" + os, "w")
            self.overdQfile = open(filenamebase + ".dQ" + os, "w")

            self.overPfile.write(header)
            self.overOfile.write(header)
            self.overGfile.write(header)
            self.overh2file.write(header)
            self.overH2file.write(header)
            self.overdQfile.write(header)


    def calcMatrices(self, parentTraits, offspringTraits):
        if parentTraits == []:
            return None, None, None, None, None  # P,O,G,h2,H2

        parentArray = array(parentTraits)    # i.e. parents
        offspringArray = array(offspringTraits)  # i.e. offspring
        deltas = offspringArray - parentArray

        X = cov(parentArray, offspringArray, rowvar=0, bias=1)
        half = X.shape[0]/2

        P = X[:half, :half]  # upper-left quarter
        O = X[half:, half:]  # lower-right quarter
        G = X[:half, half:]  # upper-right quarter
        try:
            Pinv = linalg.inv(P)
        except linalg.LinAlgError:
            Pinv = array([[-0.0] * half] * half)  # -0.0 is a warning
                
        h2 = dot(G, Pinv)
        H2 = dot(O, Pinv)

        dQ = cov(deltas, rowvar=0, bias=1)
        #print("dQ =", dQ)

        return P, O, G, h2, H2, dQ


    def reinitialize(self, population):
        if self.allPreTraits == []:   # initial call is a special case
            OffspringFamilyProbe.reinitialize(self, population)
            return

        P, O, G, h2, H2, dQ = self.calcMatrices(self.allPreTraits,
                                                self.allPostTraits)

        # Write flattened matrices to files
        numElems = P.shape[0] * P.shape[1]
        g = '"' + str(self.gen) + '" ' + str(self.gen) + " " \
            + str(len(self.allPreTraits)) + " "
        self.Pfile.write(g+ join(str(i) for i in P.reshape(numElems)) +"\n")
        self.Ofile.write(g+ join(str(i) for i in O.reshape(numElems)) +"\n")
        self.Gfile.write(g+ join(str(i) for i in G.reshape(numElems)) +"\n")
        self.h2file.write(g+ join(str(i) for i in h2.reshape(numElems)) +"\n")
        self.H2file.write(g+ join(str(i) for i in H2.reshape(numElems)) +"\n")
        self.dQfile.write(g+ join(str(i) for i in dQ.reshape(numElems)) +"\n")

        # Oversampling
        self.gensSinceOversample += 1
        if self.oversampleInterval > 0 and \
           self.gensSinceOversample == self.oversampleInterval:

            # Reinitialize the all...Traits variables in the base class.
            self.submittedTraits = []
            self.parentTraits = []   # Matches children sent to apply()
            self.offspringIndividuals = []
            self.allPreTraits = []
            self.allPostTraits = []
            self.gensSinceOversample = 0

            #while len(self.allPreTraits) < self.oversampleSize:
            for i in xrange(self.oversampleSize):
                self.pull()
                
            P, O, G, h2, H2, dQ = self.calcMatrices(self.allPreTraits,
                                                    self.allPostTraits)

            # Write flattened matrices to files
            g = '"' + str(self.gen) + '" ' + str(self.gen) + " " \
                + str(len(self.allPreTraits)) + " "
            numElems = P.shape[0] * P.shape[1]
            self.overPfile.write(g+join(str(i) for i in P.reshape(numElems))+"\n")
            self.overOfile.write(g+join(str(i) for i in O.reshape(numElems))+"\n")
            self.overGfile.write(g+join(str(i) for i in G.reshape(numElems))+"\n")
            self.overh2file.write(g+join(str(i) for i in h2.reshape(numElems)) \
                                  +"\n")
            self.overH2file.write(g+join(str(i) for i in H2.reshape(numElems)) \
                                  +"\n")
            self.overdQfile.write(g+join(str(i) for i in dQ.reshape(numElems)) \
                                  +"\n")

        self.gen += 1
        OffspringFamilyProbe.reinitialize(self, population)


# The files will close automatically
#    def __del__(self):
#        self.Pfile.close()
#        self.Ofile.close()
#        self.Qfile.close()




#############################################################################
#
# MidparentHeritabilityPostopProbe
#
#############################################################################
#class MidparentHeritabilityPostopProbe(PostOperatorProbe):
#    """
#    Calculates parent-offspring correlation for multiple traits using
#    midparent values.  In other words, the traits of all parents
#    in a family are averaged before the covariance is performed.
#
#    I'm not sure if this works or not.
#    """
#    def __init__(self, provider, parentOperatorProbe, measureFunction, \
#                 filenamebase = "results", suffix = ""):
#        PostOperatorProbe.__init__(self, provider, parentOperatorProbe, \
#                                   measureFunction)
#        s = suffix
#        self.Pfile = open(filenamebase + ".P" + s, "w")
#        self.Ofile = open(filenamebase + ".O" + s, "w")
#        self.Gfile = open(filenamebase + ".G" + s, "w")
#
#        numVars = len(measureFunction())
#        header = "Gen NumSamples " + \
#                 join("C"+str(x)+"."+str(y) for x in range(numVars) \
#                                            for y in range(numVars)) + "\n"
#        self.Pfile.write(header)
#        self.Ofile.write(header)
#        self.Gfile.write(header)
#
#        self.gen = 0
#
#
#    def calcMatrices(self, parentTraits, offspringTraits):
#        if parentTraits == []:
#            return None, None, None # P,O,G
#
#        parentArray = array(parentTraits)    # i.e. parents
#        offspringArray = array(offspringTraits)  # i.e. offspring
#
#        X = cov(parentArray, offspringArray, rowvar=0, bias=1)
#        half = X.shape[0]/2
#
#        P = X[:half, :half]  # upper-left quarter
#        O = X[half:, half:]  # lower-right quarter
#        G = X[:half, half:]  # upper-right quarter
#        try:
#            Pinv = linalg.inv(P)
#        except linalg.LinAlgError:
#            Pinv = array([[-0.0] * half] * half)  # -0.0 is a warning
#                
#        return P, O, G
#
#
#    def reinitialize(self, population):
#        if self.allPreTraits == []:   # initial call is a special case
#            PostOperatorProbe.reinitialize(self, population)
#            return
#
#        P, O, G = self.calcMatrices(self.allPreTraits, self.allPostTraits)
#
#        # Write flattened matrices to files
#        numElems = P.shape[0] * P.shape[1]
#        g = '"' + str(self.gen) + '" ' + str(self.gen) + " " \
#            + str(len(self.allPreTraits)) + " "
#        self.Pfile.write(g+ join(str(i)for i in P.reshape(numElems)) +"\n")
#        self.Ofile.write(g+ join(str(i)for i in O.reshape(numElems)) +"\n")
#        self.Gfile.write(g+ join(str(i)for i in G.reshape(numElems)) +"\n")
#
#        self.gen += 1
#        PostOperatorProbe.reinitialize(self, population)
#
#
#    def apply(self, individuals):
#        """
#        Redefine PostOperatorProbe's apply function to calculate midparent
#        values instead of defining all combinations of parents and offspring.
#        """
#        offspringTraits = [self.measureFunction(i) for i in individuals]
#
#        # Record parent/offspring traits
#        midparent = reduce(add, self.parentTraits) / len(self.parentTraits)
#        for offspring in offspringTraits:
#            self.allPreTraits.append(midparent)
#            self.allPostTraits.append(offspring)
#
#        #if len(self.parentTraits) > 0 and len(offspringTraits) > 0:
#        #    self.allPreTraits.append(average(self.parentTraits))
#        #    self.allPostTraits.append(average(offspringTraits))
#
#        return individuals
#
## The files will close automatically
##    def __del__(self):
##        self.Pfile.close()
##        self.Ofile.close()
##        self.Qfile.close()



#############################################################################
#
# OperatorProbe
#
# I don't recommend this.
# Use ParentFamilyProbe() and PostOperatorProbe() instead.
#
#############################################################################
#class OperatorProbe(LEAP.WrapperOperator):
#    """
#    A single operator (or a set of mutually exclusive operators, such as
#    crossover and no-op) is wrapped so that measurements can be made on its
#    performance both before and after it is called.
#
#    NOTE: I was planning on generalizing this and making it a base class for
#          a variety of probes.  Right now though, it measures correlation
#          between parent and offspring traits.
#
#    I like this approach less and less, but I'll keep this here just in case
#    it becomes useful.  I prefer the ParentFamilyProbe/PostOperatorProbe
#    approach above.
#    """
#    def __init__(self, provider, wrappedOps, opProbs, measureFunc,
#                 tag="opcor", measureFile = None):
#        LEAP.WrapperOperator.__init__(self, provider, wrappedOps, opProbs)
#
#        self.measureFile = measureFile
#        self.firstCall = True
#
#        self.measureFunc = measureFunc
#        self.tag = tag
#        self.zero = measureFunc()
#        self.setGeneration(0)
#
#
#    def setGeneration(self, newGen):
#        self.generation = newGen
#
#        # Create a list of empty lists.  This approach may look like overkill,
#        # but I have to make sure each element is unique.  
#        self.preMeasures = [[] for i in range(len(self.wrappedOps))]
#        self.postMeasures = [[] for i in range(len(self.wrappedOps))]
#
#        # Doing things this way is a bit of a hack
#        self.preLengths = [[] for i in range(len(self.wrappedOps))]
#        self.postLengths = [[] for i in range(len(self.wrappedOps))]
#
#
#    def reinitialize(self, population):
#        LEAP.WrapperOperator.reinitialize(self, population)
#
#        # On the first call to reinitilize, not data is available
#        if self.preMeasures != [[]] * len(self.wrappedOps):
#            ratios = [len(opPre) for opPre in self.preMeasures]
#            total = sum(ratios)
#            ratios = [float(i) / total for i in ratios]
#
#            # We will get errors if pre and post contain any empty lists
#            # For now I will just place a singe zero in any empty list
#            for i in range(len(self.preMeasures)):
#                if self.preMeasures[i] == []:
#                    self.preMeasures[i] = self.postMeasures[i] = [0.0]
#
#            deltas = [[postval-preval for preval,postval in zip(opPre,opPost)]\
#                 for opPre,opPost in zip(self.preMeasures, self.postMeasures)]
#
#            # population means
#            prebar = [E(opPre) for opPre in self.preMeasures]
#            postbar = [E(opPost) for opPost in self.postMeasures]
#            deltabar = [E(opDelta) for opDelta in deltas]
#
#            # population variances
#            varpre = [Var(opPre) for opPre in self.preMeasures]
#            varpost = [Var(opPost) for opPost in self.postMeasures]
#            vardelta = [Var(opDelta) for opDelta in deltas]
#
#            # heretability and correlation
#            covs = [cov(opPre, opPost) for opPre,opPost in
#                    zip(self.preMeasures, self.postMeasures)]
#
#            # Since these calculations can potentially produce non-results
#            # because of divide-by-zero errors, I'll comment them out and
#            # leave them to be performed later, using R for example.
#
#            #heretabilities = covs[:]
#            #for i in range(len(covs)):  # avoid divide by zeros
#            #    if varpost[i] == 0:
#            #        heretabilities[i] = None
#            #    else:
#            #        heretabilities[i] = covs[i] / varpre[i]
#
#            #correlations = covs[:]
#            #for i in range(len(covs)):  # avoid divide by zeros
#            #    if varpost[i] == 0 or varpre[i] == 0:
#            #        correlations[i] = None
#            #    else:
#            #        correlations[i] = covs[i] / math.sqrt(varpre[i]*varpost[i])
#
#            # print the results
#            print("Gen:", self.generation, end='')
#            print("Tag:", self.tag, end='')
#            for i in range(len(self.wrappedOps)):
#                print("r"+str(i+1)+":", ratios[i], end='')
#
#            for i in range(len(self.wrappedOps)):
#                print("DeltaBar"+str(i+1)+":", deltabar[i], end='')
#
#            for i in range(len(self.wrappedOps)):
#                print("VarPre"+str(i+1)+":", varpre[i], end='')
#
#            for i in range(len(self.wrappedOps)):
#                print("VarPost"+str(i+1)+":", varpost[i], end='')
#
#            for i in range(len(self.wrappedOps)):
#                print("VarDelta"+str(i+1)+":", vardelta[i], end='')
#
#            for i in range(len(self.wrappedOps)):
#                print("Cov"+str(i+1)+":", covs[i], end='')
#            print()
#
#            # print the results
#            if self.measureFile:
#                if self.firstCall:
#                    s = '"Gen", "OpNum", "Tag", "PreMeasure", ' + \
#                        '"PostMeasure", "PreLen", "PostLen"\n'
#                    self.measureFile.write(s)
#                    self.firstCall = False
#                for opInd in range(len(self.wrappedOps)):
#                    for pre,post,prelen,postlen in zip(self.preMeasures[opInd],\
#                                                     self.postMeasures[opInd],
#                                                     self.preLengths[opInd],
#                                                     self.postLengths[opInd]):
#                        s = "%d, %d, %s, %g, %g, %d, %d\n" % (self.generation,\
#                                   opInd, self.tag, pre, post, prelen, postlen)
#                        self.measureFile.write(s)
#                        
#            self.generation += 1
#
#        # Empty out the measurements to make room for next generation.
#        self.preMeasures = [[] for i in range(len(self.wrappedOps))]
#        self.postMeasures = [[] for i in range(len(self.wrappedOps))]
#
#        self.preLengths = [[] for i in range(len(self.wrappedOps))]
#        self.postLengths = [[] for i in range(len(self.wrappedOps))]
#
#
#    def apply(self, individuals):
#        # I've written other operators so that they can deal with getting more
#        # individuals than expected.  I cannot allow that here because I need
#        # to know the relationships between parents and offspring.
#        assert(len(individuals) == self.parentsNeeded)
#
#        # Measure before op
#        preMeasures = [self.measureFunc(i) for i in individuals]
#        preLengths = [len(i.genome) for i in individuals]
#
#        # Perform op
#        individuals = LEAP.WrapperOperator.apply(self, individuals)
#
#        # Measure after op
#        postMeasures = [self.measureFunc(i) for i in individuals]
#        postLengths = [len(i.genome) for i in individuals]
#
#        # Make sure all parents and offspring are associated with each other.
#        # This may mean putting duplicates measurements in the list.
#        for pre,prelen in zip(preMeasures, preLengths):
#            for post,postlen in zip(postMeasures, postLengths):
#                self.preMeasures[self.opInd].append(pre)
#                self.preLengths[self.opInd].append(prelen)
#                self.postMeasures[self.opInd].append(post)
#                self.postLengths[self.opInd].append(postlen)
#
#        return individuals
        

def indexMeasure(ind = None):
    if ind == None:
        return -1   # define a zero measurement
    return ind.popIndex


class GaussInit(LEAP.PipelineOperator):
    """
    Randomize all the genes in a genome so that each conforms to a gaussian
    distribution.
    """
    parentsNeeded = 1

    def __init__(self, provider, bounds):
        LEAP.PipelineOperator.__init__(self, provider)
        self.bounds = bounds
        self.means = [mean(bound) for bound in bounds]
        self.stdevs = [std(bound) for bound in bounds]

    def apply(self, individuals):
        for i in individuals:
            for g in range(len(i.genome)):
                i.genome[g] = gauss(self.means[g], self.stdevs[g])
        return individuals


#############################################################################
#
# unit test
#
#############################################################################
class randomizeFunctor:
    def __init__(self, function, bounds):
        self.function = function
        self.bounds = bounds

    def __call__(self, phenome):
        state = getstate();
        randPhenome = []
        total = sum(phenome)
        for p,b in zip(phenome,self.bounds):
            seed(total - p)
            if p < b[0] or p > b[1]:
                randPhenome = phenome
                break
            newp = uniform(b[0],b[1])
            randPhenome.append(newp)
        fitness = self.function(randPhenome)
        state = setstate(state);
        return fitness


if __name__ == '__main__':
    # Some parameters
    popSize = 1000
    maxGeneration = 50

    # Setup the problem
    numVars = 3
    bounds = LEAP.sphereBounds[:1] * numVars
    maximize = LEAP.sphereMaximize
    function = LEAP.sphereFunction
    #function = randomizeFunctor(LEAP.sphereFunction, bounds)

    problem = LEAP.FunctionOptimization(function, maximize = maximize)

    # ...for binary genes
    #bitsPerReal = 16
    #genomeSize = bitsPerReal * numVars
    #decoder = LEAP.BinaryRealDecoder(problem, [bitsPerReal] * numVars, bounds)

    # ...for float genes
    decoder = LEAP.FloatDecoder(problem, bounds, bounds)

    # ...for adaptive real genes
    #sigmaBounds = (0.0, bounds[0][1] - bounds[0][0])
    #initSigmas = [(bounds[0][1] - bounds[0][0]) / sqrt(numVars)] * numVars
    #decoder = LEAP.AdaptiveRealDecoder(problem, bounds, bounds, initSigmas)

    phenMeasure = ParameterMeasure(decoder)
    fitMeasure = FitnessMeasure()

    #measureFile = open("unit_test.measure", "w")
    #measure = rankMeasure

    # Setup the reproduction pipeline
    #pipeline = LEAP.TruncationSelection(popSize/2)
    pipeline = LEAP.TournamentSelection(2)
    #pipeline = LEAP.ProportionalSelection()
    #pipeline = LEAP.RankSelection()
    #pipeline = LEAP.DeterministicSelection()
#    pipeline = PriceCalcOperator(pipeline, zero=measure(), tag="SurvivalSel")
    pipeline = LEAP.CloneOperator(pipeline)
#    pipeline = NormalityPopulationProbe(pipeline, measure, tag="sel  ")
#    pipeline = klParentProbe = ParentPopulationProbe(pipeline, measure)
#    pipeline = corrParentProbe = ParentFamilyProbe(pipeline, fitMeasure)
#    pipeline = parentProbe2 = ParentFamilyProbe(pipeline, measure)
    #pipeline = LEAP.Shuffle2PointCrossover(pipeline, 0.8, 2)
    #pipeline = LEAP.NPointCrossover(pipeline, 0.8, 2)

    #numChildren = 1
    #pipeline = LEAP.NPointCrossover(pipeline, 1.0, 2, numChildren=numChildren)
    #pipeline = LEAP.UniformCrossover(pipeline, 1.0, pSwap=0.5)

#    pipeline = TestOffspringFamilyProbe(pipeline, measure, parentProbe2, \
#                                        expectedPulls = numChildren)
#    pipeline = TestOffspringFamilyProbe(pipeline, measure, parentProbe, \
#                                        expectedPulls = numChildren)
#    pipeline = CorrelationProbe(pipeline, fitMeasure, corrParentProbe, \
#                                expectedPulls = numChildren, tag="xover")
#    pipeline = UnivariateHeritabilityProbe(pipeline, fitMeasure, \
#                                corrParentProbe, expectedPulls = numChildren,\
#                                associateFunc = midparentMidchildAssociate,
#                                tag="xover")
#    pipeline = KullbackLeiblerProbe(pipeline, measure, klParentProbe,
#                                  tag="xover")
#    pipeline = NormalityPopulationProbe(pipeline, measure, tag="xover")

#    op1 = LEAP.NPointCrossover(None, 1.0, 2)
#    op2 = LEAP.DummyOperator(None, 2)
#    pipeline = OpCorCalcOperator(pipeline, [op1, op2], [0.75, 0.25], measure,
#                                 tag="crossover")

#    op1 = LEAP.NPointCrossover(None, 1.0, 2)

#    pipeline = TestPostOperatorProbe(pipeline, pre, measure)
#    pipeline = parentProbe = ParentFamilyProbe(pipeline, fitMeasure)
    pipeline = klParentProbeX = ParentPopulationProbe(pipeline, phenMeasure,\
                                                      frequency=2)
    pipeline = corrParentProbeX = ParentFamilyProbe(pipeline, fitMeasure)
    pipeline = h2ParentProbeX = ParentFamilyProbe(pipeline, fitMeasure)

    #pipeline = LEAP.UniformCrossover(pipeline, 0.8, 0.5)
    #pipeline = price1 = PriceMeasureOperator(pipeline, measure)
    #pipeline = LEAP.ProxyMutation(pipeline)
    #pipeline = LEAP.BitFlipMutation(pipeline, 1.0/genomeSize)
    #pipeline = LEAP.UniformMutation(pipeline, 1.0/genomeSize, alleles)
    #pipeline = LEAP.AdaptiveMutation(pipeline, sigmaBounds)
#    op1 = LEAP.GaussianMutation(pipeline, sigma = 1.0, pMutate = 1.0)
#    op1 = LEAP.AdaptiveMutation(pipeline, sigmaBounds)
#    pipeline = OpCorCalcOperator(pipeline, [op1], [1.0], measure,
#                                 tag="mutation", measureFile=measureFile)
#    pipeline = LEAP.GaussianMutation(pipeline, sigma = 0.01, pMutate = 1.0)
    pipeline = LEAP.RandomSearchOperator(pipeline)
    #pipeline = LEAP.FixupOperator(pipeline)

#    pipeline = TestOffspringFamilyProbe(pipeline, fitMeasure, parentProbe, \
#                                        expectedPulls = 1) # 2)
#    pipeline = CorrelationProbe(pipeline, measure, parentProbe, \
#                                        expectedPulls = popSize) # 2)
#    pipeline = MultivariatePostOperatorProbe(pipeline, measure, parentProbe,  
#                              expectedPulls = 2, \
#                              oversampleInterval = 5, oversampleSize = 10000)
    pipeline = UnivariateHeritabilityProbe(pipeline, fitMeasure, \
                              h2ParentProbeX, expectedPulls = 1, tag="mut  ",\
                              associateFunc = midparentMidchildAssociate)
    pipeline = CorrelationProbe(pipeline, fitMeasure, corrParentProbeX, \
                                expectedPulls = 1, tag="mut  ")
    pipeline = KullbackLeiblerProbe(pipeline, phenMeasure, klParentProbeX,
                                    tag="mut  ", frequency=2)
#    pipeline = NormalityPopulationProbe(pipeline, measure, tag="mut  ")

#    pipeline = price2 = PriceMeasureOperator(pipeline, measure)
    #pipeline = LEAP.ElitismSurvival(pipeline, 2)
    #pipeline = PriceRankOperator(pipeline, popSize)
    #pipeline = PriceCalcOperator(pipeline, zero=measure(),
    #                             tag="ParentSel")
#    pipeline = PriceCalcOperator(pipeline, zero=measure(), tag="ParentSel")
    #pipeline = VarianceCalcOperator(pipeline, zero=measure()) 
    #pipeline = LEAP.MuCommaLambdaSurvival(pipeline, popSize, popSize*10)
    

    initPipe = LEAP.DeterministicSelection()
#    initPipe = PriceInitOperator(initPipe)
#    initPipe = PriceMeasureOperator(initPipe, measure)
#    initPipe = GaussInit(initPipe, bounds)

    ea = LEAP.GenerationalEA(decoder, pipeline, popSize, initPipeline=initPipe)
    ea.run(maxGeneration)

#    import profile
#    profile.run('ea(params)', 'eaprof')
#
#    import pstats
#    p = pstats.Stats('eaprof')
#    p.sort_stats('time').print_stats(20)


