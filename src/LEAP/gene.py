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

import sys
import string
import math


#############################################################################
#
# class Gene
#
# This class is not necessary.  Users can just make their genomes contain
# numbers or bits or whatever they want, and completely ignore this class.
# Of course, they may need to create genetic operators which can deal with
# the types they have chosen.
#
# The idea behind writing this class is to make it easier to mix gene
# types in an individual, or to attach additional information to a gene
# such as an adaptive sigma for gaussian mutation.
#
#############################################################################
class Gene:
    """
    Optional class used for defining genetic material for individuals.
    Can also be used to associate extra information with genes, such as
    a specific mutation operator.  This class can be useful if one is
    mixing data types in the genome.
    """
    data = None
    mutationOperator = None   # A mutation operator with a mutateGene method

    def __init__(self, data, mutationOperator = None):
        self.data = data
        self.mutationOperator = mutationOperator

    def __repr__(self):
        return str(self.data)

    # I haven't completely thought this through.  It still needs some work.
    def mutate(self):
        if self.mutationOperator != None:
            self.data = self.mutationOperator.mutateGene(self.data)
        return self
    

#############################################################################
#
# class BoundedRealGene
#
#############################################################################
class BoundedRealGene(Gene):
    """
    Stores real values, along with a tuple which contains the upper and
    lower bound.  NOTE: One (or even both) of the bounds can be set to
    None, thereby eliminating that bound.
    """
    bounds = None

    def __init__(self, data, bounds, mutationOperator = None):
        self.bounds = bounds
        Gene.__init__(self, data, mutationOperator)

    def __setattr__(self, name, value):
        "Enforce the bounds"
        if name == "data":
            # I use 'or value' to deal with the case where bounds[1] is None
            v = value
            if self.bounds[0] is not None:
                v = max(v, self.bounds[0])
            if self.bounds[1] is not None:
                v = min(v, self.bounds[1])
            self.__dict__[name] = v
        else:
            self.__dict__[name] = value


#############################################################################
#
# class AdaptiveRealGene
#
#############################################################################
class AdaptiveRealGene(BoundedRealGene):
    """
    Stores the sigma used for the adaptive mutation.
    Don't use the ProxyMutation operator with this gene.
    """
    sigma = None  # gaussian mutation stdev

    def __init__(self, data, bounds, sigma, mutationOperator = None):
        self.sigma = sigma
        BoundedRealGene.__init__(self, data, bounds, mutationOperator)

    def __str__(self):
        return "(" + str(self.data) + "," + str(self.sigma) + ")"


#############################################################################
#
# unit_test
#
#############################################################################
def unit_test():
    passed = True

    print("Test BoundedRealGene")
    gene0 = BoundedRealGene(10.0, (0.0, 100.0))

    gene0.data = 20.0
    passed = passed and (gene0.data == 20.0)
    print(gene0, "== 20.0")

    gene0.data = -5.0
    passed = passed and (gene0.data == 0.0)
    print(gene0.data, "== 0.0")

    gene0.data = 250.0
    passed = passed and (gene0.data == 100.0)
    print(gene0.data, "== 100.0")

    gene1 = BoundedRealGene(10.0, (None, None))

    gene1.data = 100000.0
    passed = passed and (gene1.data == 100000.0)
    print(gene1, "== 100000.0")

    gene1.data = -100000.0
    passed = passed and (gene1.data == -100000.0)
    print(gene1, "== -100000.0")

    print()
    if passed:
        print("Passed")
    else:
        print("FAILED")


if __name__ == '__main__':
    unit_test()

