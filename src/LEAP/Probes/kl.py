#! /usr/bin/env python

##############################################################################
#
# Kullback-Leibler implementation for 2 multivariate normal distributions.
#
##############################################################################

# Python 2 & 3 compatibility
from __future__ import print_function

#import sys
#import random
#import string
#import copy

from random import *
from math import *
from numpy import *
from numpy.linalg import *

def KLdivergence(mu0, Sigma0, mu1, Sigma1):
    """
    This is a normalized version of K-L divergence.  A similarity metric (in
    the range [0,1]) is calculated between two multivariate normal
    distributions.
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


# Gaussian mutation
def mutate(parent, mut_sigma):
    child = [i + gauss(0,mut_sigma) for i in parent]
    return child

# Uniform recombination
def recombine(parent1, parent2, pSwap = 0.5):
    c1 = []
    c2 = []
    for g1,g2 in zip(parent1, parent2):
        if random.random() < pSwap:
            c1.append(g1)
            c2.append(g2)
        else:
            c1.append(g2)
            c2.append(g1)
    return c1,c2


if __name__ == '__main__':
    
    nt = 5    # number of traits
    ni = 100  # number of individuals
    mut_sigma = .000001
    
    # ----- Generate parent population -----
    #mu0 = array([0.0] * nt)
    #Sigma0 = identity(nt)
    pop0 = array([[gauss(0,mut_sigma*100) for trait in range(nt)] for ind in range(ni)])
    #pop0 = array([[gauss(0,1)] * nt for ind in range(ni)])
    #pop0 = array([mutate(p, 0.8) for p in pop0])
    mu0 = mean(pop0, axis=0)
    Sigma0 = cov(pop0.T)
    
    print("pop0:")
    print(pop0)
    print("mu0 =", mu0)
    print("Sigma0:")
    print(Sigma0)
    
    # ----- Generate offspring population -----
    #mu1 = array([0.0] * nt)
    #Sigma1 = identity(nt)
    #pop1 = array([[gauss(0,1) for i in range(nt)] for j in range(ni)])
    pop1 = array([mutate(p, mut_sigma) for p in pop0])

    # Recombination.  Attempt at a cool list comprehension
    #pop1 = [list(recombine(p1,p2)) for (p1,p2) in zip(pop0[::2], pop0[1::2])]
    #pop1 = array(reduce(lambda x,y:x+y, pop1))  # Flatten

    mu1 = mean(pop1, axis=0)
    Sigma1 = cov(pop1.T)

    print()
    print("pop1:")
    print(pop1)
    print("mu1 =", mu1)
    print("Sigma1:")
    print(Sigma1)
    
    # ----- Compare distributions -----
    kl0 = KLdivergence(mu0, Sigma0, mu1, Sigma1)
    kl1 = KLdivergence(mu1, Sigma1, mu0, Sigma0)
    
    kldistance = kl0+kl1
    
    #print("kl0 =", kl0)
    #print("kl1 =", kl1)
    #print("kldistance =", kldistance)
    
    print()
    print("1/(1 + kldistance) =", 1.0/(1.0 + kldistance))
