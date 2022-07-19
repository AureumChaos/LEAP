#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  7 09:57:02 2022

@author: robert
"""

from scipy.optimize import minimize
import numpy as np
from numpy.linalg import eig
from math import exp


def booth(x):
    x1 = x[0]
    x2 = x[1]
    
    return (x1 + 2*x2 -7)**2 + (2*x1 + x2 -5)**2

bnds = ((-10,10),(-10,10))
res = minimize(booth, [5,-5], method='trust-constr',bounds=bnds)

def FMinterpolation(x,genomes,FMs):
    
    I = len(genomes[0])
    vx = np.linspace(-1, 1, I+1)
    vx = vx[:-1] + np.diff(x)/2
    
    rhos = np.zeros((I,I,3))
    
    rhos[:,:,0] = np.matlib.repmat(genomes[0],len(genomes[0]),1).transpose()
    rhos[:,:,1] = (x[0] - x[1])*np.square(vx) + x[1]
    rhos[:,:,2] = np.matlib.repmat(genomes[1],len(genomes[0]),1).transpose()
    
    
    # calculate alphas
    alphas = (rhos[:,:,1] - rhos[:,:,0])/(rhos[:,:,2] - rhos[:,:,0])
    
    # calculate FM
    FM = (1-alphas)*FMs[0] + alphas*FMs[1]
    # eigen
    ks,dists = eig(FM)
    
    k = ks[0]
    fiss_dist = dists[:,0]
    
    fiss_dist /= sum(fiss_dist)
    
    fitness = 0
    
    for fiss in fiss_dist:
        fitness += sum((fiss_dist/fiss -1)**2)/1000
        
    if k < 1:
        fitness *= exp(5*(1-k))
    
    return fitness
    
    
    