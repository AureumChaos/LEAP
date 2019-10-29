"""
    Unit tests for cloning
"""
import unittest

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from leap import core
from leap import binary
from leap import ops


class TestClone(unittest.TestCase):

    def test_clone(self):
        # We need an encoder and problem to ensure those float across during
        # clones.
        decoder = core.IdentityDecoder()
        problem = binary.MaxOnes()

        original = core.Individual([1, 1], decoder=decoder, problem=problem)

        cloned, args, kwargs = next(ops.clone(iter([original])))

        self.assertEqual(original, cloned)

        # Yes, but did the other state make it across OK?

        self.assertEqual(original.fitness, cloned.fitness)
        self.assertEqual(original.decoder, cloned.decoder)
        self.assertEqual(original.problem, cloned.problem)
        self.assertEqual(original.attributes, cloned.attributes)



if __name__ == '__main__':
    unittest.main()

