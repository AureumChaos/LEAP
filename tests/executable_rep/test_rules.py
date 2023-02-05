"""Unit tests for the rules package."""
from gymnasium import spaces
import numpy as np
import pytest

from leap_ec.individual import Individual
from leap_ec.executable_rep.rules import PittRulesDecoder, PittRulesExecutable


##############################
# Test fixtures
##############################
@pytest.fixture
def simple_pitt_decoder():
    """A fixture that returns a PittRulesDecoder defined over a continuous sensor
    space and a discrete action space."""
    in_ = spaces.Box(low=0, high=1.0, shape=(1, 3), dtype=np.float32)
    out_ = spaces.Discrete(4)
    decoder = PittRulesDecoder(input_space=in_, output_space=out_)
    return decoder


##############################
# Tests for PittRulestDecoder.initializer()
##############################
def test_pittrulesdecoder_initialize(simple_pitt_decoder):
    """When we use a Pitt representation's default initializer to
    sample rules, all our conditions should lie within the bounds
    of the space."""
    initialize = simple_pitt_decoder.initializer(num_rules=4)

    def pairs(l):
        """A helper to iterate through a list two-by-two."""
        it = iter(l)
        return zip(l, l)

    N = 100
    for i in range(N):
        genome = initialize()
        assert(len(genome) == 4)  # 4 rules implies 4 segments
        for seg in genome:
            assert(len(seg) == 7)  # 6 condition genes + 1 action
            conditions, actions = seg[:6], seg[-1:]
            for low, high in pairs(conditions):
                assert(low >= 0.0)
                assert(low <= 1.0)
                assert(high >= 0.0)
                assert(high <= 1.0)
            for a in actions:
                assert(a >= 0)
                assert(a < 4)


##############################
# Tests for PittRulesDecoder.mutator()
##############################
def test_pittrulesdecoder_mutator(simple_pitt_decoder):
    """When we use a Pitt representation to create a mutation
    operator that applies different operators to conditions and actions, 
    respectively, then the correct operator should be applied to each."""

    mutator = simple_pitt_decoder.mutator(
        # Passing in some dummy deterministic mutators as lambdas, for a simple test
        condition_mutator=lambda segment: [ s + 0.1 for s in segment ],
        action_mutator=lambda segment: [ 0 for _ in segment ],
    )

    # An individual with two rules
    ind = Individual([
        [ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 3 ],
        [ 0, 0, 0, 0, 0, 0, 2]
    ])

    expected_genome = [
        [ 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0 ],
        [ 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0 ]
    ]

    result = next(mutator(iter([ind])))

    for expected_seg, observed_seg in zip(expected_genome, result.genome):
        assert(pytest.approx(expected_seg) == observed_seg)
