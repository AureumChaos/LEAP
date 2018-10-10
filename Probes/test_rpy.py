#! /usr/bin/env python

# Python 2 & 3 compatibility
from __future__ import print_function

from random import *
from rpy2 import robjects


class Individual:
    def __init__(self, phenomeLength):
        self.phenome = [gauss(i*10,1) for i in range(phenomeLength)]


if __name__ == "__main__":
    r = robjects.r

    phenomeLength = 3
    popSize = 10
    population = [Individual(phenomeLength) for i in range(popSize)]

    # Convert population phenotypes to an R data.frame
    r("T1 <- 1:%d" % popSize)
    r("df <- data.frame(T1)")  # Create a data.frame with the right row num.
    for t in range(phenomeLength):
        traitName = "T" + str(t+1)
        traits = robjects.FloatVector([ind.phenome[t] for ind in population])
        r.assign('rtraits', traits)
        r("df$" + traitName + " <- rtraits")
    r("df$" + traitName + " <- rtraits")
    r("df$Class = rep(1,%d)" % popSize)

    r("print(df)")
    r("library(dprep)")
    r("mardia(df)")

