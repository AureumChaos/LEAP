from leap_ec.multiobjective.asynchronous import enlu_inds_rank
from leap_ec.multiobjective.ops import fast_nondominated_sort, \
    crowding_distance_calc, rank_ordinal_sort
from leap_ec.multiobjective.problems import SCHProblem
from .test_ops import generate_test_pop
import numpy as np

from leap_ec.individual import Individual
from leap_ec.representation import Representation
from leap_ec.real_rep.initializers import create_real_vector
import time
import sys


def test_inds_rank1():
    """
    Tests to see if inds_rank can properly rank the test pop
    """
    pop, ranks, _ = generate_test_pop()
    
    layer_pops = []
    for ind in pop:
        enlu_inds_rank(ind, layer_pops)
    
    np.testing.assert_array_equal(
        [ind.rank for ind in pop],
        ranks
    )


def test_inds_rank2():
    """
    Tests to see if inds_rank can properly rank larger populations with many ranks
    """
    try:
        state = np.random.get_state()
        np.random.seed(111)
        
        prob = SCHProblem()
        rep = Representation(initialize=create_real_vector(bounds=[(-10, 10)]))
        
        pop = rep.create_population(1000, prob)
        Individual.evaluate_population(pop)
        
        layer_pops = []
        for ind in pop:
            enlu_inds_rank(ind, layer_pops)
        
        ranks_1 = np.array([ind.rank for ind in pop])
        
        fast_nondominated_sort(pop)
        ranks_2 = np.array([ind.rank for ind in pop])
        
        np.testing.assert_array_equal(ranks_1, ranks_2)
        
    finally:
        np.random.set_state(state)