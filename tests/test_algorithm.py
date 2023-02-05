"""Unit tests for the algorithm package."""
import os

import pytest

from leap_ec import Representation, ops, context
from leap_ec.algorithm import multi_population_ea
from leap_ec.binary_rep.problems import MaxOnes
from leap_ec.binary_rep.initializers import create_binary_sequence
from leap_ec.binary_rep.ops import mutate_bitflip


##############################
# Function multi_pop_fixture()
##############################
@pytest.fixture
def multi_pop_fixture():
    """Set up a multi_population_ea() algorithm for testing."""
    pop_size = 10
    generations = 5
    num_populations = 4
    genes_per_subpopulation = [ 3, 4, 5, 8]  # The number of genes in each subpopulation's individuals

    def get_representation(length: int):
        """Return a representation that creates
        binary sequences of the given length."""
        assert(length > 0)
        return Representation(
            initialize=create_binary_sequence(length)
        )

    # Initialize a representation for each subpopulation
    representations = [ get_representation(l) for l in genes_per_subpopulation]

    ea = multi_population_ea(max_generations=generations, pop_size=pop_size,
                                num_populations=num_populations,

                                # Fitness function
                                problem=MaxOnes(), 

                                # Assign a poor initial fitness to individuals
                                init_evaluate=ops.const_evaluate(value=-100),

                                # Passing a list of representations causes
                                # different ones to be used for different subpops
                                representation=representations,

                                # Operator pipeline
                                shared_pipeline=[
                                    ops.tournament_selection,
                                    ops.clone,
                                    mutate_bitflip(expected_num_mutations=1),
                                    ops.CooperativeEvaluate(
                                        num_trials=3,
                                        collaborator_selector=ops.random_selection),
                                    ops.pool(size=pop_size)
                                ])

    # Return the iterator without running the algorithm yet
    return ea

@pytest.mark.xfail
def test_multi_population_ea(multi_pop_fixture):
    """
    Ensure that at the end of a run, the final best-so-far individual that
    is returned for each subpop is the same as the best individual in context['leap']['subpopulations'].

    This is a basic check that the reference to the population in context is
    updated correctly.
    """

    # FIXME This is foolish.  I'm comparing best-so-far to best-of-gen, so the test fails.

    # Run the algorithm and retrieve BSF records
    results = list(multi_pop_fixture)

    # Get the BSF info for the final generation
    generation, bsf_tuple = results[len(results) - 1]

    context_pops = context['leap']['subpopulations']
    context_bsf = [ max(pop).fitness for pop in context_pops ]
    
    assert(generation == 5)
    assert(len(bsf_tuple) == len(context_bsf))
    for i in range(len(bsf_tuple)):
        assert(context_bsf[i] == pytest.approx(bsf_tuple[i].fitness))
