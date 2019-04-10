#! /usr/bin/env python

# concept.py
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

from LEAP.decoder import int2bin
from LEAP.problem import Problem
from LEAP.Domains.Concept.readUCI import readUCI


def calcDecisionThresholds(values):
    """
    This function is used to optimize the match functions.  It calculates
    threshold values which are used to determine which class the output
    from the executableObject represents.
    """
    thresholds = []
    prev = values[0]
    for i in values[1:]:
        thresholds.append(float(i)+prev/2)
        prev = i

    return thresholds


#############################################################################
#
# Match functions
#
#############################################################################
def exactMatch(output, answer, classes):
    """
    The output must match the answer exactly.

    @param output   Output previously received from an executableObject for
                    a given example
    @param answer   The correct classification associated with the same example
    @param classes  A list of all the classes possible, assumed to be sorted
    @return         True if output matches answer
    """
    return output == answer


def nearestClassMatch(output, answer, classes):
    """
    The class whos value is closest to output must match answer.

    @param output   Output previously received from an executableObject for
                    a given example
    @param answer   The correct classification associated with the same example
    @param classes  A list of all the classes possible, assumed to be sorted
    @return         True if output matches answer

    NOTE:  This function keeps persistent state information related to the
           classes for optimization purposes.
    """
    if nearestClassMatch.classes != classes:
        nearestClassMatch.classes = classes
        nearestClassMatch.thresholds = calcDecisionThresholds(classes)

    thresholds = nearestClassMatch.thresholds
    ind = 0
    for t in thresholds:
        if output[0] >= t:
            ind += 1

    return [classes[ind]] == answer


# init persistent local variable
nearestClassMatch.classes = None


def nearestIndexMatch(output, answer, classes):
    """
    The legalVal whos index is closest to output must match answer.
    This is somewhat indirect, but is less likely to be biased than
    nearestVal.  It also can be used with values other than numbers.

    @param output   Output previously received from an executableObject for
                    a given example
    @param answer   The correct classification associated with the same example
    @param classes  A list of all the classes possible, assumed to be sorted
    @return         True if output matches answer
    """
    if len(classes) == 2:
        if output[0] >= 0.5:
            return answer[0] == classes[1]
        else:
            return answer[0] == classes[0]
    else:    
        # The above optimization can probably be generalized for more than 2
        # classes.
        ind = int(round(output[0]))
        ind = max(ind, 0)
        ind = min(ind, len(classes)-1)
        return classes[ind] == answer[0]



#############################################################################
#
# ConceptLearning
#
#############################################################################
class ConceptLearning(Problem):
    """
    Concept Learning domain.
    """
    def __init__(self, matchFunc = exactMatch, classes = []):
        self.groups = []
        self.matchFunc = matchFunc
        self.classes = classes


    def cmpFitness(self, fitness1, fitness2):
        """
        Determine which fitness is better depending on whether we are
        maximizing or minimizing.

        @param fitness1: The first fitness value to be compared.
        @param fitness2: The second fitness value to be compared.

        @return:
            1 if fitness1 is "better than" fitness2
            0 if fitness1 = fitness2
            -1 if fitness1 is worse than fitness2
        """
        # Fitness is %correct, so always maximize
        return cmp(fitness1, fitness2)


    def matches(self, output, correctAnswer):
        #return output == correctAnswer
        return self.matchFunc(output, correctAnswer, self.classes)


    def evaluate(self, phenome):
        """
        Evaluate the fitness of an individual by classifying all the training
        examples.

        @param phenome: An executableObject.
        @return: The ratio of correct classifications to the total number of
                 examples.
        """
        numCorrect = 0
        for example in self.trainingSet:
            if self.matches(phenome.execute(example[0]), example[1]):
                numCorrect += 1

        fitness = float(numCorrect) / float(len(self.trainingSet))
        return fitness


    def classifyTests(self, phenome):
        """
        Perform an independent evaluation of an individual by having it
        classify a set of examples that were not used during training (the
        test set).

        @param phenome: An executableObject.
        @return: The ratio of correct classifications to the total number of
                 examples.
        """
        numCorrect = 0
        for example in self.testSet:
            if self.matches(phenome.execute(example[0]), example[1]):
                numCorrect += 1

        accuracy = float(numCorrect) / float(len(self.testSet))
        return accuracy


    def generateExampleGroups(self, numExamples, numGroups):
        """
        Generates a set of examples and stores them internally in groups.
        The number of groups is specified by the parameter numGroups.
        """
        examples = self.generateExamples(numExamples)
        random.shuffle(examples)
        self.groups = [[examples[i] for i in range(len(examples)) \
                                    if i % numGroups == j]
                       for j in range(numGroups)]


    def selectTestSetGroup(self, groupNum):
        """
        Returns the maximum number of examples available or possible.
        If there is no upper limit, None is returned.
        """
        self.testSet = self.groups[groupNum]

        self.trainingSet = []
        trainGroups = [self.groups[i] for i in range(len(self.groups)) \
                                          if i != groupNum]
        for group in trainGroups:
            self.trainingSet = self.trainingSet + group


    def getMaxExamples(self):
        """
        Returns the maximum number of examples available or possible.
        If there is no upper limit, None is returned.
        """
        raise(NotImplementedError)


    def generateExamples(self, numExamples):
        """
        Returns a list of examples for use in either training or testing.
        """
        raise(NotImplementedError)



#############################################################################
#
# BinaryConceptLearning
#
#############################################################################
class BinaryConceptLearning(ConceptLearning):
#    def __init__(self, matchFunc = nearestClassMatch, classes = [0, 1]):
#        ConceptLearning.__init__(self, matchFunc, classes)

    def __init__(self):
        ConceptLearning.__init__(self)

    def matches(self, output, correctAnswer):
        return correctAnswer[0] == (output[0] >= 0.5)



#############################################################################
#
# UCIConceptLearning
#
#############################################################################
class UCIConceptLearning(ConceptLearning):
    """
    Concept Learning domain which reads data from a UCI ML database file.

    @param UCIfilename String containing file name
    @param classIndex Index of the field that identifies the class, defaults
                      to the last field
    @param delimiter Field delimiter, defaults to comma
    @param matchFunc A function which determines if the output of an
                     executableObject matches the class of an example
    """
    def __init__(self, UCIfilename, classIndex = -1, delimiter = ',',
                 matchFunc = exactMatch, normalizeFeatures = False):
        ConceptLearning.__init__(self, matchFunc)
        self.UCIexamples, legalVals = readUCI(UCIfilename, classIndex, \
                                              delimiter)
        self.features = legalVals[0]
        self.classes = legalVals[1]

        if normalizeFeatures:
            ranges = [(f[0], f[-1]) for f in self.features]
            scaledExamples = [[[(1.0*f-r[0])/(r[1]-r[0]) for f,r in \
                             zip(e[0],ranges)], e[1]] for e in self.UCIexamples]
            self.UCIexamples = scaledExamples
            self.features = [[0.0, 1.0] for f in self.features]


    def getMaxExamples(self):
        """
        Returns the maximum number of examples available or possible.
        If there is no upper limit, None is returned.
        """
        return len(self.UCIexamples)


    def generateExamples(self, numExamples):
        """
        Returns a list of examples for use in either training or testing.
        """
        return self.UCIexamples[:numExamples]



#############################################################################
#
# OddParityConceptLearning
#
#############################################################################
class OddParityConceptLearning(ConceptLearning):
    """
    In the domain, the features are the digits in a binary number, and
    the class is a binary digit indicating the parity of the binary number.
    A one indicates odd parity, and a zero indicates an even parity.
    """
    def __init__(self, numDigits):
        ConceptLearning.__init__(self)
        self.numDigits = numDigits


    def getMaxExamples(self):
        """
        Returns the maximum number of examples available or possible.
        If there is no upper limit, None is returned.
        """
        return 2**self.numDigits


    def generateExamples(self, numExamples):
        """
        Returns a list of examples for use in either training or testing.
        """
        examples = []
        for i in range(numExamples):
            bin = int2bin(i, self.numDigits)
            features = [int(c) for c in bin]
            examples.append([features, [features.count(1)%2]])
        return examples



#############################################################################
#
# unit_test
#
#############################################################################
def unit_test():
    from LEAP.Exec.Pitt.ruleInterp import pyRuleInterp

    numDigits = 4
    parity = OddParityConceptLearning(numDigits)
    max = parity.getMaxExamples()
    parity.generateExampleGroups(max, 4)

    assert(max == 16)

    parity.selectTestSetGroup(1)
    assert(len(parity.testSet) == 4)
    assert(len(parity.trainingSet) == 12)
    print(parity.testSet[0])

    parity.selectTestSetGroup(2)
    assert(len(parity.testSet) == 4)
    assert(len(parity.trainingSet) == 12)
    print(parity.testSet[0])

    answerRuleset = [[0,1, 0,1, 0,1, 0,1,  0],
                     [0,0, 0,0, 0,0, 1,1,  1],
                     [0,0, 0,0, 1,1, 0,0,  1],
                     [0,0, 1,1, 0,0, 0,0,  1],
                     [0,0, 1,1, 1,1, 1,1,  1],
                     [1,1, 0,0, 0,0, 0,0,  1],
                     [1,1, 0,0, 1,1, 1,1,  1],
                     [1,1, 1,1, 0,0, 1,1,  1],
                     [1,1, 1,1, 1,1, 0,0,  1]]
    answer = pyRuleInterp(answerRuleset, numDigits, 1)

    train = parity.evaluate(answer)
    print("training set: ", train)
    test = parity.classifyTests(answer)
    print("testing set: ", test)

    assert(train == 1.0)
    assert(test == 1.0)

    uci = UCIConceptLearning("./wdbc.data",
                             classIndex=1, matchFunc=nearestIndexMatch,
                             normalizeFeatures=True)
    print("classes =", uci.classes)
    print("num_features =", len(uci.features))
    feature_lens = [(i[0], i[-1]) for i in uci.features]
    bounds = [(math.floor(i[0]), math.ceil(i[-1])) for i in uci.features]

    for a,b in zip(feature_lens, bounds):
        print(b, a)

    print("Passed")


if __name__ == '__main__':
    unit_test()

