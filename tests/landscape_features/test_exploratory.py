"""Unit tests for the exploratory package, i.e. our exploratory landscape analysis features."""
import pytest

from leap_ec.decoder import IdentityDecoder
from leap_ec.individual import Individual
from leap_ec.landscape_features import exploratory
from leap_ec.representation import Representation
from leap_ec.real_rep import initializers, problems


##############################
# Test fixtures
##############################
@pytest.fixture
def spheroid_sample():
    """A uniform sample of individuals on the spheroid function."""
    DIMENSIONS = 10
    N_SAMPLES = 50*DIMENSIONS
    problem = problems.SpheroidProblem()

    representation = Representation(
        initialize=initializers.create_real_vector(bounds=[(-5.12, 5.12)]*DIMENSIONS)
    )

    initial_sample = representation.create_population(N_SAMPLES, problem)
    Individual.evaluate_population(initial_sample)

    return problem, representation, initial_sample

@pytest.fixture
def spheroid_convex(spheroid_sample):
    """Constructs an ELAConvexity instance from samples on the spheroid.

    We made this a fixture because building ELAConexity requires some computation,
    so we want to reuse a single instance in different tests.
    """
    problem, representation, initial_sample = spheroid_sample 

    return exploratory.ELAConvexity(problem, representation, design_individuals=initial_sample)


##############################
# Tests for ELAConvexity
##############################
def test_results_length(spheroid_convex):
    """If we run ELAConvexity with the default settings, it should generate data for
    1,000 pairs of points."""
    assert(len(spheroid_convex.pairs) == 1000)
    assert(len(spheroid_convex.combinations) == 1000)
    assert(len(spheroid_convex.deltas) == 1000)

def test_spheroid_convexity(spheroid_convex):
    """If we run ELAConvexity on the spheroid function, it should come back as almost 100%
    convex."""
    # We don't check that convex_p is exactly 1.0, because occasionally the same individual
    # is chosen twice during test pairing, causing less than 100% convexity to be estimated.
    assert(1.0 - spheroid_convex.convex_p() < 0.1)
