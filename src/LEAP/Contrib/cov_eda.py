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
cov_eda.py

A simple implementation of a real-valued estimation of distribution algorithm
(EDA).
"""

# Python 2 & 3 compatibility
from __future__ import print_function

import random
import numpy

from LEAP.operators import GeneticOperator
from LEAP.individual import Individual


#############################################################################
#
# class CovarianceMatrixEDA
#
#############################################################################
class CovarianceMatrixEDA(GeneticOperator):
    """
    This most closely resembles work by Yuan.
    Individuals are assumed to be a fixed linear genome of real values.
    A population of size numParents is drawn and used to calculate a
    population distribution (mean and covariance matrix).  Every time pull is
    called, a new individual is created that matches the distribution of
    selected parents.  If exploreFactor is greater than 1, the distribution of
    the offspring population will extend beyond the parent distribution by
    that amount.
    """

    def __init__(self, provider, numParents, exploreFactor = 1.0,
                 indClass = Individual, extraPullInterval = None,
                 perturb = 0.0):
        GeneticOperator.__init__(self, provider)
        self.parentsNeeded = numParents
        self.exploreFactor = exploreFactor
        self.indClass = indClass
        self.extraPullInterval = extraPullInterval
        self.covMat = None
        self.centroid = None
        self.encoding = None
        self.numPulls = 0
        self.perturb = perturb  # The calculated covariance matrix can
                                # sometimes be illegal (non-positive
                                # definite?).  Adding small perturbations
                                # (essentially mutations) helps.


    def reinitialize(self, population):
        GeneticOperator.reinitialize(self, population)
        self.covMat = None
        self.centroid = None
        self.numPulls = 0


    def pull(self):
        """
        This operator bypasses the apply() method and the cache, and just
        redefines the pull() method.
        """
        if self.covMat == None:
            #print("Calc matrix")
            parents = [self.provider[i % len(self.provider)].pull()
                       for i in range(self.parentsNeeded)]
            pGenomes = [p.genome for p in parents]
            pMatrix = numpy.transpose(numpy.array(pGenomes))
            self.centroid = pMatrix.mean(1)
            self.covMat = numpy.cov(pMatrix) * self.exploreFactor
            self.encoding = parents[0].encoding

        # Occasionally pull another parent to fool the pre/post probes
        if self.numPulls > self.extraPullInterval and \
           self.extraPullInterval != None:
            self.numPulls = 0
            self.provider[0].pull()

        # Create a new individual based on parent population distribution
        g = numpy.random.multivariate_normal(self.centroid, self.covMat) + \
            numpy.array([random.gauss(0,self.perturb) for i in self.centroid])

        self.numPulls +=1
        #print(self.numPulls,end="")

        ind = self.indClass(self.encoding, genome=list(g))
        return(ind)




#############################################################################
#
# unit_test
#
#############################################################################
def unit_test():

    print("No unit test")


if __name__ == '__main__':
    unit_test()


