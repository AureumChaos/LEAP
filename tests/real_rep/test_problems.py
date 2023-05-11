"""Unit tests for LEAP's suite of real-valued fitness functions."""
import numpy as np
from pytest import approx

from leap_ec import Individual
from leap_ec.real_rep import problems


########################
# Tests for GriewankProblem
########################
def test_GriewankProblem_eval():
    """The value of a test point should be what we expected."""
    t = np.array((0.5, 0.5))

    # In two dimensions, the Griewank function expands like so
    expected = t[0]**2/4000 + t[1]**2/4000 - np.cos(t[0]/np.sqrt(1))*np.cos(t[1]/np.sqrt(2)) + 1

    p = problems.GriewankProblem()
    assert(approx(expected) == p.evaluate(t))



########################
# Tests for WeierstrassProblem
########################
def test_WeierstrassProblem_eval():
    """The Weierstrass function has a (0, ... ,0) in all dimensions
    and have a fitness of zero.
    """
    p = problems.WeierstrassProblem()

    assert(approx(0) == p.evaluate(np.array([0, 0])))
    assert(approx(0) == p.evaluate(np.array([0]*25)))
