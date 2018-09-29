#! /usr/bin/env python

# funcApprox.py
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
import math

import LEAP
import readUCI


#############################################################################
#
# BaseFunctionApproximation
#
#############################################################################
class BaseFunctionApproximation(LEAP.Problem):
    """
    For this problem, the EA must learn an approximation of a function.
    Fitness is measured as least mean squared error.
    """
    def __init__(self):
        self.groups = []


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
        # Fitness is sum squared error, so always minimize
        return -cmp(fitness1, fitness2)


    def compareFit(self, phenome, examples):
        """
        Checks to see how well the ruleset (phenome) matches the target
        function (as described by the examples).
        Returns the sum squared error.
        """
        fit = 0
        for example in examples:
            # example[0] is the list of inputs
            result = phenome.execute(example[0])
            # example[1] is the list of outputs.
            # For now assume only 1 output.
            err = result[0] - example[1][0]
            fit += err * err
        #return fit / (len(examples) - 1)
        return fit / len(examples)
        

    def evaluate(self, phenome):
        """
        Evaluate the fitness of an individual by classifying all the training
        examples.

        @param phenome: An executableObject.
        @return: The sum squared error.
        """
        #print "training set:"
        #for example in self.trainingSet:
        #    print example
        return self.compareFit(phenome, self.trainingSet)


    def classifyTests(self, phenome):
        """
        Perform an independent evaluation of an individual by having it
        classify a set of examples that were not used during training (the
        test set).

        @param phenome: An executableObject.
        @return: The sum squared error.
        """
        return self.compareFit(phenome, self.testSet)


    def generateExampleGroups(self, numExamples, numGroups):
        """
        Generates a set of examples and stores them internally in groups.
        The number of groups is specified by the parameter numGroups.
        """
        examples = self.generateExamples(numExamples)
        random.shuffle(examples)
        self.groups = [[examples[i] for i in range(len(examples)) \
                                    if i % numGroups == j] \
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
        raise NotImplementedError


    def generateExamples(self, numExamples):
        """
        Returns a list of examples for use in either training or testing.
        """
        raise NotImplementedError



#############################################################################
#
# UCIFunctionApproximation
#
#############################################################################
#class UCIFunctionApproximation(BaseFunctionApproximation):
#    """
#    Concept Learning domain which reads data from a UCI ML database file.
#
#    @param UCIfilename String containing file name
#    @param classIndex Index of the field that identifies the class, defaults
#                      to the last field
#    @param delimiter Field delimiter, defaults to comma
#    @param matchFunc A function which determines if the output of an
#                     executableObject matches the class of an example
#    """
#    def __init__(self, UCIfilename, classIndex = -1, delimiter = ',',
#                 matchFunc = exactMatch, normalizeFeatures = False):
#        ConceptLearning.__init__(self, matchFunc)
#        self.UCIexamples, legalVals = readUCI.readUCI(UCIfilename,
#                                                      classIndex, delimiter)
#        self.features = legalVals[0]
#        self.classes = legalVals[1]
#
#        if normalizeFeatures:
#            ranges = [(f[0], f[-1]) for f in self.features]
#            scaledExamples = [[[(1.0*f-r[0])/(r[1]-r[0]) for f,r in \
#                             zip(e[0],ranges)], e[1]] for e in self.UCIexamples]
#            self.UCIexamples = scaledExamples
#            self.features = [[0.0, 1.0] for f in self.features]
#
#
#    def getMaxExamples(self):
#        """
#        Returns the maximum number of examples available or possible.
#        If there is no upper limit, None is returned.
#        """
#        return len(self.UCIexamples)
#
#
#    def generateExamples(self, numExamples):
#        """
#        Returns a list of examples for use in either training or testing.
#        """
#        return self.UCIexamples[:numExamples]



#############################################################################
#
# FunctionApproximation
#
#############################################################################
class FunctionApproximation(BaseFunctionApproximation):
    """
    In the domain, the features are the digits in a binary number, and
    the class is a binary digit indicating the parity of the binary number.
    A one indicates odd parity, and a zero indicates an even parity.
    """
    def __init__(self, targetFunc, bounds):
        """
        @param targetFunc: The function that will be approximated.  The
                           parameters to the function are passed in as a list.
        @param bounds: A list of tuples.  Each tuple defines the lower and
                       upper bounds of one parameter.
        """
        BaseFunctionApproximation.__init__(self)
        self.targetFunc = targetFunc

        # Right now this only works with 1 parameter
        assert(len(bounds) == 1)
        self.bounds = bounds


    def getMaxExamples(self):
        """
        Returns the maximum number of examples available or possible.
        If there is no upper limit, None is returned.
        """
        return None


    def generateExamples(self, numExamples):
        """
        Returns a list of examples for use in either training or testing.
        """
        examples = []
        for i in range(numExamples):
            point = [random.uniform(b[0], b[1]) for b in self.bounds]
            examples.append([point, [self.targetFunc(point)]])
        return examples



#############################################################################
#
# unit_test
#
#############################################################################
import LEAP.Exec.Pitt
def unit_test():

    targetFunc = lambda x: math.sin(x[0])
    #targetFunc = lambda x: x[0]**2
    #targetFunc = lambda x: 0.0
    condbounds = [ (-1.0, 1.0) ]
    actionbounds = [(-10.0, 10.0)]
    allbounds = condbounds + actionbounds
    fa = FunctionApproximation(targetFunc, condbounds)
    numExamples = fa.getMaxExamples()
    if numExamples == None:
       numExamples = 12
    numGroups = 4
    fa.generateExampleGroups(numExamples, numGroups)

    fa.selectTestSetGroup(0)
    assert(len(fa.testSet) == numExamples/numGroups)
    assert(len(fa.trainingSet) == numExamples - numExamples/numGroups)
    print "Test set :"
    for ex in fa.testSet:
        print ex
    print

    print "Training set :"
    for ex in fa.trainingSet:
        print ex
    print

    print "Regenerate examples"
    oldTestSet = fa.testSet
    fa.generateExampleGroups(numExamples, numGroups)
    fa.selectTestSetGroup(0)
    fa.testSet = oldTestSet  # This is a hack

    print "Test set :"
    for ex in fa.testSet:
        print ex
    print

    print "Training set :"
    for ex in fa.trainingSet:
        print ex
    print

    fa.selectTestSetGroup(1)
    assert(len(fa.testSet) == numExamples/numGroups)
    assert(len(fa.trainingSet) == numExamples - numExamples/numGroups)
    #print "Test set =", fa.testSet[0]

    numRules = 100
    epsilon = 0.01

    print "badRuleset"
    badRuleset = []
    for ruleIndex in range(numRules):
        condition = []
        for cb in condbounds:
            condition += [random.uniform(cb[0], cb[1])] * 2
        action = []
        for ab in actionbounds:
            action = [random.uniform(ab[0], ab[1])]
        badRuleset.append(condition + action)

    #print "badRuleset =", badRuleset
    interp = LEAP.Exec.Pitt.pyInterpolatingRuleInterp(badRuleset, 1, 1)
    train = fa.evaluate(interp)
    print "training eval: ", train
    test = fa.classifyTests(interp)
    print "testing eval: ", test
    assert(train > epsilon)
    assert(test > epsilon)


    print
    print "goodRuleset"
    goodRuleset = []
    for ruleIndex in range(numRules):
        condition = []
        for b in condbounds:
            condition += [random.uniform(b[0], b[1])] * 2
        action = [targetFunc(condition)]
        goodRuleset.append(condition + action)

    #print "goodRuleset =", badRuleset
    interp = LEAP.Exec.Pitt.pyInterpolatingRuleInterp(goodRuleset, 1, 1)
    train = fa.evaluate(interp)
    print "training eval: ", train
    test = fa.classifyTests(interp)
    print "testing eval: ", test
    assert(train < epsilon)
    assert(test < epsilon)

    print
    print "shuffled goodRuleset"
    #print "goodRuleset =", goodRuleset
    random.shuffle(goodRuleset)
    interp = LEAP.Exec.Pitt.pyInterpolatingRuleInterp(goodRuleset, 1, 1)
    train2 = fa.evaluate(interp)
    print "training eval: ", train2
    test2 = fa.classifyTests(interp)
    print "testing eval: ", test2
    assert(train == train2)
    assert(test == test2)

    print
    print "Define a function using a rule interpreter"
    condbounds = [ (-10.0, 10.0) ]
    actionbounds = [(-10.0, 10.0)]
    cb = condbounds[0]
    ab = actionbounds[0]
    numRules = 10
    points = [[round(random.uniform(cb[0], cb[1]),2),
               round(random.uniform(ab[0], ab[1]))]
               for i in range(numRules)]
    points.sort()
    ruleset = [[p[0], p[0], p[1]] for p in points]
    print "ruleset =", ruleset
    targetInterp = LEAP.Exec.Pitt.pyInterpolatingRuleInterp(ruleset, 1, 1)
    targetFunc = lambda x: targetInterp.execute(x)[0]

    allbounds = condbounds + actionbounds
    fa = FunctionApproximation(targetFunc, condbounds)
    numExamples = fa.getMaxExamples()
    numExamples = 12
    numGroups = 2
    fa.generateExampleGroups(numExamples, numGroups)
    fa.selectTestSetGroup(0)
    
    interp = LEAP.Exec.Pitt.pyInterpolatingRuleInterp(ruleset, 1, 1)
    selfEval = fa.evaluate(interp)
    print "selfEval =", selfEval


    print "Passed"


if __name__ == '__main__':
    unit_test()

