"""Unit tests for the leap_ec.problem module."""
import numpy as np
import pytest
from scipy.stats import norm

from leap_ec import Individual, problem
from leap_ec.real_rep import problems as real_prob


##############################
# Tests for AverageFitnessProblem
##############################
@pytest.mark.stochastic
def test_averagefitnessproblem():
    """If we take the average fitness of a landscape with additive
    Gaussian noise, then the distance between our estimate and
    the true mean fitness should be within the 99.9% confidence
    intervals for a Gaussian distribution.
    """
    # The number of fitness samples to average over
    n = 200

    p = problem.AverageFitnessProblem(
                    wrapped_problem = real_prob.NoisyQuarticProblem(),
                    n = n)
    x = [ 1, 1, 1, 1 ]
    y = p.evaluate(x)

    # The value of the noisy-quartic is sum(i*x**4) plus additive Gaussian noise
    expected_mean = 10.0  # = 1 + 2 + 3 + 4 

    difference = y - expected_mean

    alpha = 0.999
    low, high = norm.interval(alpha, scale=1.0/np.sqrt(n))
    assert((difference >= low) and (difference <= high)), f"Expected difference from the true and estimated mean fitness to be within the confidence interval ({low}, {high}) (computed from alpha={alpha}), but observed an estimate of {difference}."
