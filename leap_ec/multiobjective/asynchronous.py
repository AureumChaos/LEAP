import numpy as np
from leap_ec import ops, util
from leap_ec.global_vars import context
from leap_ec.individual import Individual
from leap_ec.multiobjective.problems import MultiObjectiveProblem
from leap_ec.multiobjective.ops import fast_nondominated_sort, per_rank_crowding_calc
from leap_ec.global_vars import context
from leap_ec.distrib.asynchronous import steady_state


def _find_start_layer(ind, layer_pops):
    """
    Finds the highest layer in which ind is not dominated.
    """
    lo = 0
    hi = len(layer_pops) - 1
    
    while lo <= hi:
        mid = (lo + hi) // 2
        if not any(i > ind for i in layer_pops[mid]):
            hi = mid - 1
        else:
            lo = mid + 1
    
    return lo

def _split_dominated(moving_points, layer):
    """
    Splits layer into populations of points that aren't / are dominated by the
    zenith of moving_points. We use a maximization rather than minimization
    objective versus ENLU, so this is zenith / max vs nadir / min
    """
    zenith = np.max([
        ind.fitness * ind.problem.maximize
        for ind in moving_points
    ], axis=0)
    
    dominated = []
    nondominated = []
    for ind in layer:
        ord_fitness = ind.fitness * ind.problem.maximize
        if all(zenith >= ord_fitness) and any(zenith > ord_fitness):
            dominated.append(ind)
        else:
            nondominated.append(ind)
    
    return nondominated, dominated

def _set_domination(pop_a, pop_b):
    """
    Checks for each individual in pop_b whether it was dominated by an individual in pop_a.
    """
    return [
        any(li > ri for li in pop_a)
        for ri in pop_b
    ]
    
def enlu_inds_rank(start_point, layer_pops):
    """ Performs the incremental non-dominated sorting ranking process.
    
    Based on the ENLU insertion algorithm with the modification of a binary search for the start point.
    Locates the highest layer where the individual is nondominated and inserts it, propagating
    layer composition changes down the rankings.
    
        - K. Li, K. Deb, Q. Zhang and Q. Zhang, "Efficient Nondomination Level Update Method for
          Steady-State Evolutionary Multiobjective Optimization," in IEEE Transactions on
          Cybernetics, vol. 47, no. 9, pp. 2838-2849, Sept. 2017, doi: 10.1109/TCYB.2016.2621008.
    
    
    :param moving points: the set of points descending in rank from the previous layer.
        In the first recursion this is the inserted individual.
    :param layer_pops: the population separated into non-dominating layers.
    :param rank_func: the ranking function used to separate out the dominated group
        at each recursion.
    :param depth: the current layer depth the moving points set is dominating.
    """
    
    # CASE I: Find the first layer where start_point is not dominated
    start_depth = _find_start_layer(start_point, layer_pops)
    moving_points = [start_point]
    
    depth = start_depth
    while depth < len(layer_pops):
        # CASE II: If the last layer merged perfectly, done
        if not moving_points:
            return start_depth, depth
        
        # Find the individuals who are dominated by the zenith
        nondominated, dominated = _split_dominated(moving_points, layer_pops[depth])
        # Further check if those individuals are properly dominated
        true_dominated = _set_domination(moving_points, dominated)

        # CASE III: If nondominated is empty, insert moving points as a layer and re-update all ranks
        if not nondominated:
            layer_pops.insert(depth, moving_points)
            for i, lp in enumerate(layer_pops[depth:]):
                for ind in lp:
                    ind.rank = depth + i + 1
            return start_depth, depth + 1
        
        # CASE IV: Some points are dominated, propagate those onwards
        # The moving points stay in this layer, while those not dominated by the zenith
        # remain unchanged
        layer_pops[depth] = nondominated + moving_points
        moving_points = []
        
        # The truly dominated go to the next layer down, while the rest are added to
        # the current layer
        for ind, dom in zip(dominated, true_dominated):
            if dom:
                moving_points.append(ind)
            else:
                layer_pops[depth].append(ind)
                
        for ind in layer_pops[depth]:
            ind.rank = depth + 1
        depth += 1
    
    # If any points make it all the way through, they form a new layer
    if moving_points:
        for ind in moving_points:
            ind.rank = len(layer_pops) + 1
        layer_pops.append(moving_points)
    return start_depth, len(layer_pops)


class ENLUInserter:

    def __init__(self):
        # This is a 2d work list for ordering layers. Functionally it is the
        # real population, with the one the algorithm sees being overwritten
        # by this population's flattened contents
        self._layer_pops = []

    def __call__(self, ind, flat_pop, pop_size):
        start_depth, end_depth = enlu_inds_rank(ind, self._layer_pops)
        
        # Calculate crowding distance for updated layers
        for lp in self._layer_pops[start_depth:end_depth]:
            per_rank_crowding_calc(lp, lp[0].problem.maximize)
        
        # If the population is too big, drop the most crowded
        if sum(len(lp) for lp in self._layer_pops) > pop_size:
            rem_idx = min(range(len(self._layer_pops[-1])), key=lambda i: self._layer_pops[-1][i].distance)
            self._layer_pops[-1].pop(rem_idx)
            
            if self._layer_pops[-1]:
                # Since this layer is losing a member, needs recalculation of crowding
                per_rank_crowding_calc(self._layer_pops[-1], self._layer_pops[-1][0].problem.maximize)
            else:
                del self._layer_pops[-1]
        
        # Reconstruct flat_pop
        flat_pop.clear()
        for lp in self._layer_pops:
            flat_pop.extend(lp)

            
def steady_state_nsga_2(
            client, max_births: int, init_pop_size: int, pop_size: int,
            problem: MultiObjectiveProblem,
            representation,
            offspring_pipeline,
            count_nonviable=False,
            evaluated_probe=None,
            pop_probe=None,
            context=context
        ):
    """ A steady state version of the NSGA-II multi-objective evolutionary algorithm.
        Functionally, a wrapper around steady_state that chooses the inserter for you.
    
        - K. Li, K. Deb, Q. Zhang and Q. Zhang, "Efficient Nondomination Level Update Method for
          Steady-State Evolutionary Multiobjective Optimization," in IEEE Transactions on
          Cybernetics, vol. 47, no. 9, pp. 2838-2849, Sept. 2017, doi: 10.1109/TCYB.2016.2621008.

    :param client: Dask client that should already be set-up
    :param max_births: how many births are we allowing?
    :param init_pop_size: size of initial population sent directly to workers
           at start
    :param pop_size: how large should the population be?
    :param representation: of the individuals
    :param problem: to be solved
    :param offspring_pipeline: for creating new offspring from the pop
    :param count_nonviable: True if we want to count non-viable individuals
           towards the birth budget
    :param evaluated_probe: is a function taking an individual that is given
           the next evaluated individual; can be used to print newly evaluated
           individuals
    :param pop_probe: is an optional function that writes a snapshot of the
           population to a CSV formatted stream ever N births
    :return: the population containing the final individuals
    """

    # Construct the ENLU inserter for the wrapper
    inserter = ENLUInserter()
    
    # This is just a wrapper around steady state, all of the logic is the same
    # with the exception of a special inserter
    return steady_state(
            client, max_births, init_pop_size, pop_size,
            representation, problem, offspring_pipeline,
            inserter, count_nonviable, evaluated_probe,
            pop_probe, context
        )
