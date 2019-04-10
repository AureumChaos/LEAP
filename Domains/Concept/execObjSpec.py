#! /usr/bin/env python

# execObjSpec.py
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

import math
import random

from LEAP.Domains.Concept.concept import ConceptLearning


#############################################################################
#
# ExecutableObjectSpecifiedConcept
#
#############################################################################
class ExecutableObjectSpecifiedConcept(ConceptLearning):
    """
    A Concept learning problem which uses an executable object to define the
    concept space.  This class generates examples by passing feature inputs to
    the executable object and recording the corresponding class outputs.

    Right now I assume that all the feature values are continuous (real
    valued) numbers.  A more general version would (should?) allow one to
    specify nominal and integer (ordinal?) values also.
    """
    def __init__(self, execObj, bounds):
        ConceptLearning.__init__(self)
        self.execObj = execObj
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
            features = [random.random() * (b[1] - b[0]) + b[0]
                        for b in self.bounds]
            classVal = self.execObj.execute(features)[0]

            examples.append([features, classVal])

        return examples



#############################################################################
#
# unit_test
#
#############################################################################
def unit_test():
    from LEAP.Exec.Pitt.ruleInterp import pyRuleInterp
    rules = [[0.4,1.0, 0.6,1.0, 0],
             [0.0,0.6, 0.0,0.4, 1]]
    interp = pyRuleInterp(rules, 2, 1)

    execCon = ExecutableObjectSpecifiedConcept(interp, [(0.0, 1.0)] * 2)
    execCon.generateExampleGroups(200, 3)
    execCon.selectTestSetGroup(0)

    f = open("exec.dat", mode="w")
    examples = execCon.generateExamples(100000)

    print(examples[0])
    lines = [str(e[0][0]) + " " + str(e[0][1]) + "\n" for e in examples
             if e[1] == 1]
    f.writelines(lines)

    print("Passed")


if __name__ == '__main__':
    unit_test()

