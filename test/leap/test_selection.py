"""
    Unit test for selection operators.
"""
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from leap import core
from leap import ops



def test_naive_cyclic_selection():
    pop = []

    pop.append(core.Individual([0, 0]))
    pop.append(core.Individual([0, 1]))

    # This selection operator will deterministically cycle through the
    # given population
    selector = ops.naive_cyclic_selection_generator(pop)

    selected, args, kwargs = next(selector)
    assert selected.genome == [0,0]

    selected, args, kwargs = next(selector)
    assert selected.genome == [0,1]

    # And now we cycle back to the first individual
    selected, args, kwargs = next(selector)
    assert selected.genome == [0,0]

