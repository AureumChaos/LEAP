# LEAP
A general purpose Library for Evolutionary Algorithms in Python.
Written by Dr. Jeffrey K. Bassett
Contributors: Dr. R. Paul Weigand

Much of the library design was inspired by the general description of
Evolutionary Algorithms from Dr. Kenneth A. De Jong's book "Evolutionary
Computation: A Unified Approach".  He was my Ph.D. advisor, so I am quite
familiar with his approach.

As of 4/11/2018:

It's much more stable than it was, but not quite up to the level that I would
look for in a software package.  I'm working with some collaborators on
improving this, and hopefuly we can make it into something much more real soon.
Anyone is free to use it if they wish.  The current license is GPL v2, but I'm
likely to change that to something that is more permissive and more academic
(the MIT license?) at some point in the future.

I wrote this library while I was working on my dissertation some years back.
I had always thought to release it, but never actually did.  As I look over
the code now, I see that there are some pieces that I'm not completely happy
with.  It's clear that I was learning Python at the same time that I wrote
much of this.

I've now upgraded the entire library to Python 3, and almost all of it works.
It still works in Python 2 as well.  I haven't touched Paul Wiegand's code in
the Coevolution directory though, so I doubt that it even works at all.

Just to get a flavor for how the library works, I offer this simple GA:

```
from LEAP.problem import FunctionOptimization
from LEAP.encoding import BinaryRealEncoding
from LEAP.selection import ProportionalSelection
from LEAP.operators import CloneOperator, NPointCrossover, BitFlipMutation
from LEAP.survival import ElitismSurvival
from LEAP.ea import GenerationalEA

# The function we want to optimimize
def sphereFunction(phenome):
    return sum([x**2 for x in phenome])

problem = FunctionOptimization(sphereFunction, maximize=False)

numVars = 10
bounds = [(-5.12, 5.12)] * numVars

bitsPerReal = 16
genomeSize = numVars * bitsPerReal
encoding = BinaryRealEncoding(problem, [bitsPerReal] * numVars, bounds)

# Setup the reproduction pipeline
pipeline = ProportionalSelection()
pipeline = CloneOperator(pipeline)
pipeline = NPointCrossover(pipeline, 0.8, 1)  # pCross=0.8, 1 crossover point
pipeline = BitFlipMutation(pipeline, 2.0/genomeSize)  # pMut=2.0/genomeSize
pipeline = ElitismSurvival(pipeline, 2)

# More EA Parameters
popSize = 100
maxGeneration = 200

ea = GenerationalEA(encoding, pipeline, popSize)
ea.run(maxGeneration)
```
