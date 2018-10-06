#! /usr/bin/env python

# rotatedDecoder.py
#
# When I started, this seemed really clever.  Now it's become really
# complicated.  It may be worthwhile to reconsider the design.


# Python 2 & 3 compatibility
from __future__ import print_function

import random
import math
import functools

import LEAP

# Once upon a time I used to use the Numeric library for matrix operations,
# but I had a lot of trouble getting it to work for some reason, so I hand
# coded a lot of the matrix routines instead.  Things seem to work better now
# with numpy.  This is a little kludgy, but setting this to true will use the
# numpy routines, which should be faster.
USING_NUMPY = True

if USING_NUMPY:
    #import numarray
    #import Numeric
    import numpy



#############################################################################
#
# identityMatrix
#
#############################################################################
def identityMatrix(n):
    """
    """
    if USING_NUMPY:
        #return Numeric.identity(n, Numeric.Float)
        return numpy.identity(n)
    else:
        matrix = [[0.0] * n for i in range(n)]
        for i in range(n):
            matrix[i][i] = 1.0
        return matrix


#############################################################################
#
# rotationMatrix
#
#############################################################################
def rotationMatrix(i, j, n, angle = None):
    """
    Return transformation matrix of dimension n implementing a single rotation.
    If the angle is not provided, a random angle is chosen.
    """
    matrix = identityMatrix(n)
    if angle == None:
        angle = (random.random() - 0.5) * math.pi * 2
    matrix[i][i] = math.cos(angle)
    matrix[j][j] = math.cos(angle)
    matrix[i][j] = math.sin(angle)
    matrix[j][i] = -math.sin(angle)
    return matrix


#############################################################################
#
# scaleMatrix
#
#############################################################################
def scaleMatrix(n, scale):
    """
    Return transformation matrix of dimension n implementing a scale
    operation.
    """
    matrix = identityMatrix(n)
    for i in range(n):
        matrix[i][i] = scale;
    return matrix


#############################################################################
#
# matrixMultiply
#
#############################################################################
def matrixMultiply(a, b):
    """
    Return transformation matrix of dimension n implementing single rotation.
    If the angle is not provided, a random angle is chosen.
    """
    if USING_NUMPY:
        #return Numeric.matrixmultiply(a,b)
        return numpy.dot(a,b)
    else:
        # massage vectors into matrices
        if isinstance(a[0], list):
            A = a
        else:
            A = [a]
        heightA = len(A)
        widthA = len(A[0])

        # massage vectors into matrices
        if isinstance(b[0], list):
            B = b
        else:
            B = [[i] for i in b]
        heightB = len(B)
        widthB = len(B[0])

        # Check matrix shapes
        if widthA != heightB:
            raise("Error in matrixMultiply: matrix sizes do not match")

        # Build empty matrix
        C = [[0.0] * widthB for i in range(heightA)]

        # Perform the multiply
        for row in range(heightA):
            for col in range(widthB):
                sum = 0.0
                for i in range(heightB):
                    sum += A[row][i] * B[i][col]
                C[row][col] = sum

        # massage 1d matrices back into vectors
        if len(C[0]) == 1:
            C = [i[0] for i in C]
        if len(C) == 1:
            C = C[0]

        return C


#############################################################################
#
# multipleRotationsMatrix
#
#############################################################################
def multipleRotationsMatrix(n, angle = None):
    """
    Return transformation matrix of dimension n for rotation about
    multiple axes
    """
    matrix = identityMatrix(n)
    
    for i in range(n-1):
        matrix = matrixMultiply(matrix, rotationMatrix(i, i+1, n, angle))

#    for i in range(n-1):
#        matrix = matrixMultiply(matrix, rotationMatrix(0, i+1, n))
#    for i in range(n-2):
#        matrix = matrixMultiply(matrix, rotationMatrix(i+1, i-1, n))

    return matrix



##############################################################################
#
# class RotatedFloatDecoder
#
#############################################################################
class RotatedFloatDecoder(LEAP.FloatDecoder):
    """
    The genome (a list of floats) is stored in a different rotational basis.
    """
    def __init__(self, problem, initRanges, bounds = None,
                 rotationMatrix = None, angle = None):
        LEAP.FloatDecoder.__init__(self, problem, initRanges, bounds)
        if rotationMatrix is None:
            self.rotationMatrix = multipleRotationsMatrix(self.genomeLength, \
                                                          angle)
        else:
            self.rotationMatrix = rotationMatrix

    def decodeGenome(self, genome):
        if USING_NUMPY:
            genome = numpy.array(genome)
        phenome = matrixMultiply(genome, self.rotationMatrix)
        if USING_NUMPY:
            phenome = phenome.tolist()
        return phenome

    def encodeGenome(self, phenome):
        """
        What on earth is this function for?  The idea, if I recall was to
        create a way so that a randomly defined population would occupy the
        same relative space on a rotated function as it would on a non-rotated
        function.
        """
        if USING_NUMPY:
            phenome = numpy.array(phenome)
        genome = matrixMultiply(self.rotationMatrix, phenome)
        if USING_NUMPY:
            genome = genome.tolist()
        return genome

    def randomGenome(self):
        "Generates a randomized genome for this encoding"
        phenome = [random.random() * (r[1] - r[0]) + r[0] \
                   for r in self.initRanges]
        return self.encodeGenome(phenome)

    def fixupGenome(self, genome):
        "Fixes errors in the genome.  Used by the FixupOperator."
        phenome = self.decodeGenome(genome)
        phenome = LEAP.FloatDecoder.fixupGenome(self, phenome)  # clip
        genome = self.encodeGenome(phenome)
        return genome


#############################################################################
#
# class RotatedAdaptiveRealDecoder
#
#############################################################################
class RotatedAdaptiveRealDecoder(RotatedFloatDecoder):
    """
    The genome (a list of AdaptiveRealGenes) is stored in a different
    rotational basis.
    """
    def __init__(self, problem, initRanges, bounds, initSigmas,
                 rotationMatrix = None, angle = None):
        RotatedFloatDecoder.__init__(self, problem, initRanges, bounds,\
                                    rotationMatrix = rotationMatrix, \
                                    angle = angle)
        if len(initSigmas) != self.genomeLength:
            raise(ValueError,
                  "initRanges and initSigmas are different lengths.")

        # instance variables
        self.bounds = bounds   # We don't really need this here.
        self.initSigmas = initSigmas

    def decodeGenome(self, genome):
        "Decodes the genome into a phenome and returns it."
        # extract data from the BoundedRealGene class
        genome = [gene.data for gene in genome]
        phenome = RotatedFloatDecoder.decodeGenome(self, genome)
        return phenome

    def randomGenome(self):
        "Generates a randomized genome for this encoding"
        initVals = RotatedFloatDecoder.randomGenome(self)
#        genome = [LEAP.AdaptiveRealGene(initVals[i], (None, None),
#                                        self.initSigmas[i])
#                  for i in range(self.genomeLength)]
        genome = [LEAP.AdaptiveRealGene(initVals[i], self.bounds[i],
                                        self.initSigmas[i])
                  for i in range(self.genomeLength)]
        return genome

    def fixupGenome(self, genome):
        "Fixes errors in the genome.  Used by the FixupOperator."
        floatGenome = RotatedFloatDecoder.fixupGenome(self, genome) # clip
        for i in range(len(genome)):
            genome[i].data = floatGenome[i]
        return genome


#############################################################################
#
# class RotatedBinaryFloatDecoder
#
#############################################################################
class RotatedBinaryFloatDecoder(RotatedFloatDecoder):
    """
    Defines the genome as a string (not list) of ones and zeros.
    Each section of the genome encodes a real valued number.

    I probably should have named this class RotatedBinaryRealDecoder to be
    consistent with the other decoders defined in LEAP.
    """
    def __init__(self, problem, bitsPerReals, bounds, rotationMatrix = None,
                 angle = None):
        if not isinstance(bitsPerReals, list):
            raise(ValueError, "bitsPerReals must be a list.")
        if len(bitsPerReals) != len(bounds):
            raise(ValueError, "bitsPerReals and bounds are different lengths.")
        RotatedFloatDecoder.__init__(self, problem, bounds, bounds, \
                                     rotationMatrix = rotationMatrix, \
                                     angle = angle)

        # instance variables
        self.bounds = bounds
        self.bitsPerReals = bitsPerReals   # a list, # of bits per each real
        self.genomeLength = sum(bitsPerReals)

        # calculated the bounds after rotation
        lowerBounds = [b[0] for b in self.bounds]
        lowerBounds = RotatedFloatDecoder.encodeGenome(self, lowerBounds)

        upperBounds = [b[1] for b in self.bounds]
        upperBounds = RotatedFloatDecoder.encodeGenome(self, upperBounds)

        self.rotatedBounds = [(min(lowerBounds[i], upperBounds[i]), \
                               max(lowerBounds[i], upperBounds[i])) \
                              for i in range(len(lowerBounds))]

        # calcluate the positions of the reals in the genome
        self.realPos = [0]
        pos = 0
        for i in range(len(bitsPerReals)):
            pos += bitsPerReals[i]
            self.realPos.append(pos)

        # calculate the max integer value of each substring
        self.maxIntVal = [2**b - 1 for b in bitsPerReals]

    def decodeGenome(self, genome):
        phenome = []
        for i in range(len(self.bitsPerReals)):
            bnd = self.rotatedBounds[i]
            ival = int(genome[self.realPos[i] : self.realPos[i+1]], 2)
            rval = float(ival) / self.maxIntVal[i] * (bnd[1] - bnd[0]) + bnd[0]
            phenome.append(rval)
        return RotatedFloatDecoder.decodeGenome(self, phenome) # unrotate


    def encodeGenome(self, phenome):
        floatPhenome = RotatedFloatDecoder.encodeGenome(self, phenome) # rotate
        genome = ""
        for i in range(len(self.bitsPerReals)):
            bnd = self.rotatedBounds[i]
            normval = (floatPhenome[i] - bnd[0]) / (bnd[1] - bnd[0])
            ival = int(normval * self.maxIntVal[i])  # May want to round here
            genome += LEAP.int2bin(ival, self.bitsPerReals[i])
        return genome


############################################################################
#
# test
#
#############################################################################
def landscape(phenome):
    return sum(phenome)

def unit_test():
    passed = True
    epsilon = 0.001

    realValuedProb = LEAP.FunctionOptimization(landscape)
    dummyProb = LEAP.Problem()
    bounds = [(0.0, 1.0)] * 5
    #rotMatrix = rotationMatrix(0, 1, 3, math.pi)
    rotMatrix = multipleRotationsMatrix(len(bounds))
    #rotMatrix = None  # Pick one randomly
    print("rotMatrix =", rotMatrix)

    # Test RotatedFloatDecoder
    print("Test RotatedFloatDecoder")
    rfe = RotatedFloatDecoder(realValuedProb, bounds, bounds, rotMatrix)
    phenome = [0.5] * len(bounds)
    #print("phenome =", phenome)
    genome = rfe.encodeGenome(phenome)
    #print("genome =", genome)
    phenome2 = rfe.decodeGenome(genome)
    #print("phenome2 =", phenome2)
    genome = rfe.fixupGenome(genome)
    phenome3 = rfe.decodeGenome(genome)
    #print("phenome3 =", phenome3)

    errors = [(phenome[i] - phenome2[i])**2 for i in range(len(bounds))]
    error = math.sqrt(sum(errors))
    print("encode - decode:", end='')
    print("error =", error, "<", epsilon, "(" + str(error < epsilon) + ")")
    passed = passed and error < epsilon

    errors = [(phenome[i] - phenome3[i])**2 for i in range(len(bounds))]
    error = math.sqrt(sum(errors))
    print("encode - fixup - decode:", end='')
    print("error =", error, "<", epsilon, "(" + str(error < epsilon) + ")")
    passed = passed and error < epsilon

    genome = rfe.randomGenome()
    phenome = rfe.decodeGenome(genome)
    isInBounds = [bounds[i][0] <= phenome[i] <= bounds[i][1]
                  for i in range(len(phenome))]
    valid = functools.reduce(lambda x,y:x and y, isInBounds)

    #print("RotatedFloatDecoder.randomGenome() =", genome)
    #print("RotatedFloatDecoder.decodeGenome() =", phenome)
    print("valid =", valid, "== True")
    passed = passed and valid

    #brokenGenome = [-10.0, 100.0, 0.5]
    #fixedGenome = rfe.fixupGenome(brokenGenome)
    #print("brokenGenome =", brokenGenome)
    #print("fixedGenome =", fixedGenome, " == [0.0, 1.0, 0.5]")
    #passed = passed and fixedGenome == [0.0, 1.0, 0.5]

    # Test RotatedAdaptiveRealDecoder
    print()
    print("Test RotatedAdaptiveRealDecoder")
    initSigmas = [bounds[i][1] - bounds[i][0] for i in range(len(bounds))]
    nobounds = [(None, None)] * len(bounds)
    rare = RotatedAdaptiveRealDecoder(realValuedProb, bounds, nobounds, 
                                      initSigmas, rotationMatrix = rotMatrix)
    phenome = [0.5] * len(bounds)
    #print("phenome =", phenome)
    floatGenome = rare.encodeGenome(phenome)  # encodeGenome only does part
    #print("floatGenome =", floatGenome)
    realGenome = rare.randomGenome()          #  of the job.
    #print("realGenome =", realGenome)

    # The error is here.  The bounds on the genes do not allow the proper
    # values to be assigned.

    for i in range(len(realGenome)):
        realGenome[i].data = floatGenome[i]
    #print("realGenome =", realGenome)
    phenome2 = rare.decodeGenome(realGenome)
    #print("phenome2 =", phenome2)
    print("Calling fixupGenome")
    realGenome = rare.fixupGenome(realGenome)
    phenome3 = rare.decodeGenome(realGenome)
    print("phenome3 =", phenome3)

    errors = [(phenome[i] - phenome2[i])**2 for i in range(len(bounds))]
    error = math.sqrt(sum(errors))
    print("encode - decode:", end='')
    print("error =", error, "<", epsilon, "(" + str(error < epsilon) + ")")
    passed = passed and error < epsilon

    errors = [(phenome[i] - phenome3[i])**2 for i in range(len(bounds))]
    error = math.sqrt(sum(errors))
    print("encode - fixup - decode:", end='')
    print("error =", error, "<", epsilon, "(" + str(error < epsilon) + ")")
    passed = passed and error < epsilon

    # Test RotatedBinaryFloatDecoder
    print()
    print("Test RotatedBinaryFloatDecoder")
    bitsPerReals = [16] * len(bounds)
    initSigmas = [bounds[i][1] - bounds[i][0] for i in range(len(bounds))]
    rbfe = RotatedBinaryFloatDecoder(realValuedProb, bitsPerReals, bounds, 
                                     rotationMatrix = rotMatrix)
    phenome = [0.5] * len(bounds)
    genome = rbfe.encodeGenome(phenome)
    phenome2 = rbfe.decodeGenome(genome)
    genome = rbfe.fixupGenome(genome)
    phenome3 = rbfe.decodeGenome(genome)

    errors = [(phenome[i] - phenome2[i])**2 for i in range(len(bounds))]
    error = math.sqrt(sum(errors))
    print("encode - decode:", end='')
    print("error =", error, "<", epsilon, "(" + str(error < epsilon) + ")")
    passed = passed and error < epsilon

    errors = [(phenome[i] - phenome3[i])**2 for i in range(len(bounds))]
    error = math.sqrt(sum(errors))
    print("encode - fixup - decode:", end='')
    print("error =", error, "<", epsilon, "(" + str(error < epsilon) + ")")
    passed = passed and error < epsilon

    if passed:
        print("Passed")
    else:
        print("FAILED")


#############################################################################
#
# rotation_test
#
#############################################################################
def vector_test():
    v = [float(i) for i in range(3)]
    v1 = [v]
    v2 = [[i] for i in v]
    v3 = matrixMultiply(v1, v)
    print("v3 =")
    print(v3)

    m = identityMatrix(3)
    v4 = matrixMultiply(m, v)
    print("v4 =")
    print(v4)


def rotation_test():
    "Test rotation code"
    #v = Numeric.array([1.0] * 10)
    #v = Numeric.array([float(i) for i in range(10)])
    v = [float(i) for i in range(10)]
    n = len(v)
    M = identityMatrix(n)
    #M = rotationMatrix(0, 1, n, math.pi/4.0)
    #M = rotationMatrix(0, 1, n)
    #M = multipleRotationsMatrix(n)
    v1 = matrixMultiply(M, v)
    v2 = matrixMultiply(v1, M)

    print("v = ")
    print(v)
    print("v1 = ")
    print(v1)
    print("v2 = ")
    print(v2)
    print()
    #print(M)


#############################################################################
#
# main
#
#############################################################################
if __name__ == '__main__':
    print("Vector test")
    vector_test()
    print("Rotation test")
    rotation_test()
    print("Unit test")
    unit_test()
    print("More tests?")

    n = 4
    I = numpy.identity(n)
    M1 = rotationMatrix(0, 1, n, math.pi/4.0)
    M2 = rotationMatrix(0, 2, n, math.pi/4.0)
    M3 = rotationMatrix(0, 3, n, math.pi/4.0)
    #print(M1)
    #print(M2)
    #M = numpy.dot(M1, M2)
    M = numpy.dot(M2, M1)
    M = numpy.dot(M3, M)

    v = numpy.array([1,0,0,0])
    v2 = numpy.dot(v, M)
    #v2 = numpy.dot(M, v)

    print(v2)




