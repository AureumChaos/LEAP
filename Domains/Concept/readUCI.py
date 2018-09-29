#! /usr/bin/env python

import string
import sys
import re


int_re = re.compile("^\s*(\+|-)?[0-9]+\s*$");
def isInt(str):
    """
    Returns True if the string can be converted to an integer using the
    int() function.
    """
    if str == None:
        return False

    return bool(int_re.match(str))


float_re = re.compile("^\s*(\+|-)?([0-9]+.?[0-9]*|.[0-9]+)" + \
                      "((e|E)(\+|-)?[0-9]+)?\s*$")
def isFloat(str):
    """
    Returns True if the string can be converted to a float using the
    float() function.  Note that if isInt(str) is True, then so is
    isFloat(str).
    """
    if str == None:
        return False

    return bool(float_re.match(str))


def missingToNone(str):
    """
    If the string represents a missing value (i.e. '?') then None is returned.
    Otherwise the original str is returned.
    """
    if str == '?':
        return None
    return str


def transpose(matrix):
    """
    Transposes the row and column dimensions of a 2d matrix (list of lists).
    """
    return [[i[j] for i in matrix] for j in range(len(matrix[0]))]


def Int(str):
    """
    A version of the int() function that ignores None values instead of
    throwing an exception.
    """
    if str == None:
        return None
    else:
        return int(str)


def Float(str):
    """
    A version of the float() function that ignores None values instead of
    throwing an exception.
    """
    if str == None:
        return None
    else:
        return float(str)


def unique(l):
    """
    Creates a new list based on the input (l), but with duplicate values
    removed.  The values are also sorted.
    """
    u = []
    for i in l:
        if i not in u:
            u.append(i)
    u.sort()
    return u


#############################################################################
#
# readUCI
#
#############################################################################
def readUCI(dataFilename, classIndex = -1, delimiter = ','):
    """
    Reads a dataset of examples in the format used by the UCI machine learning
    repository's datafiles.
    """
    dataFile = open(dataFilename, 'r')
    all_f = string.strip(dataFile.read())
    rows = string.split(all_f, '\n')

    # Massage the row data:
    #   - split features
    #   - strip whitespace
    #   - change missing values ('?') into None
    rows = [string.split(row, delimiter) for row in rows]
    rows = [[missingToNone(string.strip(feature)) for feature in row]
            for row in rows]

    # Translate columns of strings into ints or floats if appropriate
    columns = transpose(rows)
    for c in range(len(columns)):
        ints = [isInt(i) or i == None for i in columns[c]]
        all_ints = reduce(lambda x,y:x and y, ints)

        if all_ints:
            columns[c] = [Int(x) for x in columns[c]]
        else:
            floats = [isFloat(i) or i == None for i in columns[c]]
            all_floats = reduce(lambda x,y:x and y, floats)
            if all_floats:
                columns[c] = [Float(x) for x in columns[c]]

    # Remove the column that defines the class
    classColumn = columns[classIndex]
    del columns[classIndex]

    # Build the list of legal values
    featureVals = [unique(col) for col in columns]
    classVals = unique(classColumn)
    legalVals = [featureVals, classVals]

    # Build the list of examples
    rows = transpose(columns)
    examples =  [[rows[i], [classColumn[i]]] for i in range(len(rows))]

    return examples, legalVals



#############################################################################
#
# unit_test
#
#############################################################################
def unit_test():
    passed = True

    examples, legalVals = readUCI("readUCI-test.data")

    print
    print "Examples:"
    for example in examples:
        print example

    print
    print "Legal values:"
    for featureVals in legalVals[0]:
        print featureVals
    print legalVals[1]
    print

    print "Running tests:"
    t = len(examples) == 4
    print "len(examples) == 4           (" + str(t) + ")"
    passed = passed and t

    t = [type(feature) for feature in examples[0][0]] == [str, int, float]
    print "types == [str, int, float]   (" + str(t) + ")"
    passed = passed and t

    t = reduce(lambda x,y:x and y, [e[0][1] == e[0][2] for e in examples])
    print "ints == floats               (" + str(t) + ")"
    passed = passed and t

    if passed:
        print "Passed"
    else:
        print "FAILED"


#############################################################################
#
# main
#
#############################################################################
if __name__ == '__main__':
    classIndex = -1
    delimiter = ","

    argc = len(sys.argv)
    if argc == 1:
        unit_test()
    elif argc > 4:
        print "Usage: readUCI.py [UCI datafile [classIndex ['delimiter']]]"
    else:
        filename = sys.argv[1]
        if argc > 2:
            classIndex = int(sys.argv[2])
        if argc > 3:
            delimiter = sys.argv[3]  # needs error checking

        examples, legalVals = readUCI(sys.argv[1], classIndex, delimiter)
        for example in examples:
            print example
        print
        print "Legal values:"
        for featureVals in legalVals[0]:
            print featureVals
        print legalVals[1]


