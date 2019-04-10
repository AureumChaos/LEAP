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

import math
import random

from LEAP.Domains.Concept.concept import BinaryConceptLearning


#############################################################################
#
# TwoSpiralProblem
#
#############################################################################
class TwoSpiralProblem(BinaryConceptLearning):
    """
    Classic concept learning.  Originally this was designed to test learning
    times in neural networks.  More information can be found at the CMU neural
    network learning repository.
    """
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
        for i in range( int((numExamples+1) / 2) ):
            angle = -i * math.pi / 16.0
            radius = i/16.0 + 0.5
            x = radius * math.sin(angle);
            y = radius * math.cos(angle);
            examples.append([[x,y],[1]])
            examples.append([[-x,-y],[0]])

        if len(examples) > numExamples:
            del examples[len(examples)-1]

        return examples



#############################################################################
#
# TwoSpiralRegions
#
#############################################################################
class TwoSpiralRegions(BinaryConceptLearning):
    """
    My own personal variation on the two spiral problem.  Instead of two
    spiralling lines, I define spiralling regions.  Any point in the 2D space
    can be an example.  The regions are centered on the spirals from the
    original problem.
    """
    def __init__(self, upperright = (6.5, 6.5), lowerleft = (-6.5, -6.5)):
        BinaryConceptLearning.__init__(self)
        self.top, self.right = upperright
        self.bottom, self.left = lowerleft


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
            x = random.random() * (self.right - self.left) + self.left
            y = random.random() * (self.top - self.bottom) + self.bottom
            pt_angle = math.atan2(x, y)
            pt_radius = math.sqrt(x*x + y*y)
            # By normalized I mean radians/pi, [-1, 1]
            normalized_region_angle = (0.5 - pt_radius)

            # Calculate the difference "angle", again normalized.
            # Make sure it falls in the range [-1, 1]
            diff = pt_angle / math.pi - normalized_region_angle
            diff = (diff + 1.0) % 2.0 - 1.0

            if -0.5 < diff < 0.5:  # i.e. between pi/2 and -pi/2
                classVal = True
            else:
                classVal = False

            examples.append([[x,y], [classVal]])

        return examples



#############################################################################
#
# unit_test
#
#############################################################################
def unit_test():
    twoSpiral = TwoSpiralProblem()
    twoSpiral.generateExampleGroups(194, 3)
    twoSpiral.selectTestSetGroup(0)

    f = open("spiral.dat", mode="w")
    examples = twoSpiral.generateExamples(194)
    lines = [str(e[0][0]) + " " + str(e[0][1]) + "\n" for e in examples]
    f.writelines(lines)
        

    twoSpiralReg = TwoSpiralRegions((6.5, 6.5), (-6.5, -6.5))
    #twoSpiralReg = TwoSpiralRegions((3.0, 3.0), (-3.0, -3.0))
    twoSpiralReg.generateExampleGroups(200, 3)
    twoSpiralReg.selectTestSetGroup(0)

    f = open("region.dat", mode="w")
    examples = twoSpiralReg.generateExamples(1000000)
    lines = [str(e[0][0]) + " " + str(e[0][1]) + "\n" for e in examples
             if e[1] == [1]]
    f.writelines(lines)
        
    print("Passed")


if __name__ == '__main__':
    unit_test()

