#!/usr/bin/env python3
"""
   Unit tests for the class Individual
"""
import sys, os

# Apparently stupid relative imports don't work, so have this hack
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from unittest import TestCase, main

import individual
import problem
import reproduction
import encoding

class IndividualTestCases(TestCase):

    NUM_GENES = 4

    def setUp(self):
        self.problem = problem.MaxOnes()
        self.encoding = encoding.SimpleBinaryEncoding(IndividualTestCases.NUM_GENES)

        self.individual = individual.Individual(problem=self.problem,
                                                encoding=self.encoding)


    def test_create_individual(self):
        self.assertEqual(self.individual.fitness, None) # because it hasn't been evaluated yet

        self.individual.evaluate()

        self.assertNotEqual(self.individual.fitness, None)  # because now it has a value

        self.assertEqual(self.individual.encoding.total_bits, IndividualTestCases.NUM_GENES)



    def test_mutation(self):
        # intentionally set the genome to *all* ones
        self.individual.encoding.sequence = [1,1,1,1]

        # If we flip bits with 100% certainty, then we should get a new
        # individual of all zeros.
        clone = self.individual.clone() # protect parent by cloning it

        new_individual = reproduction.binary_flip_mutation(clone, 1.0)
        self.assertEqual(new_individual.encoding.sequence,[0,0,0,0] )

        # Now the mutation better have left the original untouched.
        self.assertEqual(self.individual.encoding.sequence,[1,1,1,1])




if __name__ == '__main__':
    main()
