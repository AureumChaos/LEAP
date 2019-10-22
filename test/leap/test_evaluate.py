"""
    Unit tests for cloning
"""
import unittest

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from leap import core
from leap import binary
from leap import ops


class TestEvaluate(unittest.TestCase):

    def test_evaluate(self):
        # We need an Individual with a simple encoding and a corresponding
        # problem so that we have something with which to evaluate.

        pop = [core.Individual([1, 1], decoder=core.IdentityDecoder(),
                               problem=binary.MaxOnes())]

        # The one individual hasn't been evaluated yet, so its fitness
        # should be None

        self.assertIsNone(pop[0].fitness)

        # Since we 're using generators, let's create a new sequence with
        # the hopefully now evaluated individual.

        new_pop = [i for i in ops.evaluate(iter(pop))]

        # Which should now have a fitness.

        self.assertIsNotNone (new_pop[0].fitness)

        # And so to show that there 's no copying, we can similarly refer to
        # the same individual in the original sequence to show that, yes,
        # it really did get evaluated.

        self.assertIsNotNone (pop[0].fitness)


if __name__ == '__main__':
    unittest.main()
