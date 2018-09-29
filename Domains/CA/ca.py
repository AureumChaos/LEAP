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

import sys
import string
import copy
import random
import math
import weave

# If Python already has a function for this, I can't find it.
def bin2int(bstr):
    "Convert a string containing a binary number to an integer."
    value = 0
    pos = len(bstr)
    for i in bstr:
        pos -= 1
        value += int(i) * (2 ** pos)
    return value


#############################################################################
#
# CellularAutomata
#
#############################################################################
class CellularAutomata:
    """
    A 1D Cellular automata class.
    """
    radius = None
    rules = None

    def __init__(self, rules):
        self.rules = rules
        self.radius = int( math.log(len(rules)) / math.log(2) ) / 2
        # print self.radius

    def getNeighborhood(self, state, loc):
        "Return the bits in 'state' which are in the neighboorhood of 'loc'"
        left = (loc - self.radius) % len(state)
        right = (loc + self.radius + 1) % len(state)
        if right > left:
            return state[left:right]
        else:
            return state[left:] + state[:right]

    def pyRun(self, initState, numSteps):
        "Python version of the run function"
        state = initState
        for step in range(numSteps):
            newState = ''
            for i in range(len(state)):
                neighbors = self.getNeighborhood(state, i)
                #print neighbors, bin2int(neighbors)
                newState += str(self.rules[bin2int(neighbors)])
            state = newState
        return newState

    def cRun(self, initState, numSteps):
        "C version of the run function"
        support_code = """
            #line 20000 "ca.py"
            #define MAX_STATE_SIZE 1000

            int bin2int(char* bstr, int bstr_len)
            {
                int i, pos, len, value = 0;

                pos = bstr_len;
                value = 0;
                for (i = 0; i < bstr_len; i++)
                {
                    pos--;
                    value += (bstr[i] - '0') * (int)pow(2.0, pos);
                }

                return value;
            }
            """

        # Define some variables
        radius = self.radius
        rules = self.rules

        params = ['radius', 'numSteps', 'rules', 'initState']
        code = \
        """
            #line 10000 "ca.py"
            // std::cout << "radius = " << radius << std::endl;
            const char* iState = std::string(initState).c_str();
            const char* sRules = std::string(rules).c_str();

            int stateLen;
            static char state1[MAX_STATE_SIZE], state2[MAX_STATE_SIZE];
            char *state, *newState, *swap_tmp, *neighbor_str;
            int neighborhood, step, bitPos, offset;

            neighborhood = radius * 2 + 1;
            stateLen = strlen(iState);
            strcpy(state1, iState);
            state = state1;
            newState = state2;

            for (step = 0; step < numSteps; step++)
            {
                strncat(state, state, neighborhood);  // Allow for wrap-around
                for (bitPos = 0; bitPos < stateLen; bitPos++)
                {
                    offset = (stateLen + bitPos - radius) % stateLen;
                    neighbor_str = state + offset;
                    newState[bitPos] = sRules[bin2int(neighbor_str,
                                                      neighborhood)];
                }
                newState[stateLen] = '\0';
                swap_tmp = state;
                state = newState;
                newState = swap_tmp;
            }

            return_val = state;
        """

        state = weave.inline(code, params, support_code = support_code);
        return state

    def run(self, state, numSteps):
        "Run the CA for 'numSteps' steps given 'state' as the initial state"
        return self.cRun(state, numSteps)


#############################################################################
#
# test
#
#############################################################################
def test():
    """
    Test the ca.
    """
    alphabet = [0, 1]
    #rules = [0, 1, 0, 1, 0, 1, 1, 1]
    #rules = [1,1,1,0,1,0,0,0,1,1,1,1,1,0,1,1,1,1,0,0,1,0,0,0,0,1,0,0,0,0,1,0]
    #rules = [1,0,1,1,1,1,0,1,1,1,0,0,1,1,1,0,1,0,0,0,1,1,0,0,1,0,0,0,0,0,0,0]
    #rules = [1,1,1,0,1,0,1,1,0,0,1,1,1,0,1,0,1,0,1,0,1,0,1,0,0,0,0,0,1,0,0,0]
    #rules = [1,1,1,1,1,0,1,1,1,0,0,0,0,0,1,0,0,0,1,1,1,1,1,0,1,0,0,0,0,0,0,0]
    #rules = [1,1,1,0,1,0,1,1,1,1,1,0,0,1,1,0,1,0,0,0,1,0,1,0,1,1,0,0,0,0,1,0]
    #rules = [1,1,1,0,1,0,1,1,1,1,1,1,1,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0]
    #rules = [0,0,0,1,0,1,0,1,0,0,0,0,0,0,0,1,0,1,1,1,0,1,1,1,0,0,1,1,1,1,1,1]
    rules = "00010101000000010111011100111111"
    myca = CellularAutomata(rules)
    stateSize = 51
    maxSteps = 100

    # initState = '001011011010011110100010110101001010101001110101011'
    initState = ''
    for i in range(stateSize):
	if random.random() < 0.7:
            initState += '1'
	else:
	    initState += '0'

    state = initState
    state2 = initState
    print state
    for i in range(maxSteps):
        state = myca.pyRun(state, 1)
        state2 = myca.cRun(state2, 1)
        if state <> state2:
            print Error
        print state

#    newState = myca.cRun(state, maxSteps)
#    newState = myca.pyRun(state, maxSteps)
#    print newState

    if state == '111111111111111111111111111111111111111111111111111':
        print "Passed"
    else:
        print "FAILED"


if __name__ == '__main__':
    test()




