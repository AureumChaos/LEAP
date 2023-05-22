import toolz
import numpy as np
import bisect
import logging

from leap_ec import ops, util
from leap_ec.global_vars import context
from leap_ec.individual import Individual
from leap_ec.multiobjective.problems import MultiObjectiveProblem
from leap_ec.multiobjective.ops import fast_nondominated_sort, rank_ordinal_sort
from leap_ec.global_vars import context
from leap_ec.distrib.evaluate import evaluate, is_viable
from leap_ec.distrib.asynchronous import eval_population


# Create unique logger for this namespace
logger = logging.getLogger(__name__)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

console_handler.setFormatter(formatter)

logger.addHandler(console_handler)


def _split_dominated(moving_points, layer):
    """
    Splits layer into populations of points that are / aren't
    dominated by the nadir of moving_points.
    """
    nadir = np.min([
        ind.fitness * ind.problem.maximize
        for ind in moving_points
    ], axis=0)
    
    arr_layer = np.array(layer)
    ord_fitness = np.array([ind.fitness * ind.problem.maximize for ind in layer])
    dominated = np.all(nadir >= ord_fitness, 1) & np.any(nadir > ord_fitness, 1)
    
    return arr_layer[dominated].tolist(), arr_layer[~dominated].tolist()

def inds_rank(moving_points, layer_pops, rank_func, depth=None):
    """ Performs the incremental non-dominated sorting ranking process.
    :param moving points: the set of points descending in rank from the previous layer.
        In the first recursion this is the inserted individual.
    :param layer_pops: the population separated into non-dominating layers.
    :param rank_func: the ranking function used to separate out the dominated group
        at each recursion.
    :param depth: the current layer depth the moving points set is dominating.
    """
    
    if depth is None:
        for i, pop in enumerate(layer_pops):
            if any(ind > moving_points for ind in pop):
                inds_rank([moving_points], layer_pops, rank_func, i+1)
                return
    
    if depth >= len(layer_pops):
        layer_pops.append(moving_points)
        return
    
    if not moving_points:
        return
    
    dominated_l, nondominated_l = _split_dominated(moving_points, layer_pops[depth])
    rank_func(dominated_l + moving_points)
    
    rank_split = ([], [])
    for ind in dominated_l:
        rank_split[ind.rank-1].append(ind)
    
    layer_pops[depth] = nondominated_l + moving_points + rank_split[0]
    for ind in layer_pops[depth]:
        ind.rank = depth + 1
    
    inds_rank(rank_split[1], layer_pops, rank_func, depth + 1)


@toolz.curry
def _get_fitness(ind, obj):
    return ind.fitness[obj] * ind.problem.maximize[obj]

def _recalculate_distance(ind, objective_pops):
    ind.distance = 0
    
    for i, op in enumerate(objective_pops):
        gf = _get_fitness(obj=i)
        idx = op.index(ind)
        
        left_diff = abs(gf(ind) - gf(op[idx-1]))
        right_diff = abs(gf(op[idx+1] - gf(ind)))
        ind.distance += left_diff + right_diff
    
def insert_crowding(ind, objective_pops):
    ind.distance = 0
    
    for i, op in enumerate(objective_pops):
        # TODO: This can be expensive. Other data structures may help
        # The real time cost may be inconsequential though
        gf = _get_fitness(obj=i)
        idx = bisect.bisect_right(op, ind, key=gf)
        op.insert(idx, ind)
        
        if idx == 0:
            ind.distance = np.inf
            if len(op) > 1: _recalculate_distance(op[1], objective_pops)
        elif idx == len(op)-1:
            ind.distance = np.inf
            if len(op) > 1: _recalculate_distance(op[-2], objective_pops)
        else:
            left_diff = abs(gf(ind) - gf(op[idx-1]))
            right_diff = abs(gf(op[idx+1] - gf(ind)))
            old_diff = abs(gf(op[idx+1]) - gf(op[idx-1]))
            
            op[idx-1].distance += left_diff - old_diff
            ind.distance += left_diff + right_diff
            op[idx+1].distance += right_diff - old_diff
        
def remove_crowding(ind, objective_pops):
    for i, op in enumerate(objective_pops):
        # Same as in insert, this can be expensive.
        gf = _get_fitness(obj=i)
        idx = op.index(ind)
        
        if idx == 0:
            op[1].distance = np.inf
        elif idx == len(op)-1:
            op[-2].distance = np.inf
        else:
            left_diff = abs(gf(ind) - gf(op[idx-1]))
            right_diff = abs(gf(ind) - gf(op[idx+1]))
            old_diff = abs(gf(op[idx+1]) - gf(op[idx-1]))
            
            op[idx-1].distance += old_diff - left_diff
            op[idx+1].distance += old_diff - right_diff
        
        op.pop(idx)
            
def async_nsga_2(
            client, max_births: int, init_pop_size: int, pop_size: int,
            problem: MultiObjectiveProblem,
            representation,
            offspring_pipeline,
            rank_func,
            count_nonviable=False,
            evaluated_probe=None,
            pop_probe=None,
            context=context
        ):
    """ A steady state version of the NSGA-II multi-objective evolutionary algorithm.

    :param client: Dask client that should already be set-up
    :param max_births: how many births are we allowing?
    :param init_pop_size: size of initial population sent directly to workers
           at start
    :param pop_size: how large should the population be?
    :param representation: of the individuals
    :param problem: to be solved
    :param offspring_pipeline: for creating new offspring from the pop
    :param inserter: function with signature (new_individual, pop, popsize)
           used to insert newly evaluated individuals into the population;
           defaults to greedy_insert_into_pop()
    :param count_nonviable: True if we want to count non-viable individuals
           towards the birth budget
    :param evaluated_probe: is a function taking an individual that is given
           the next evaluated individual; can be used to print newly evaluated
           individuals
    :param pop_probe: is an optional function that writes a snapshot of the
           population to a CSV formatted stream ever N births
    :return: the population containing the final individuals
    """
    
    initial_population = representation.create_population(init_pop_size,
                                                          problem=problem)

    # fan out the entire initial population to dask workers
    as_completed_iter = eval_population(initial_population, client=client,
                                        context=context)

    # This is where we'll be putting evaluated individuals
    layer_pops = []
    objective_pops = [[] for _ in range(len(problem.maximize))]
        
    def inds_inserter(ind):
        inds_rank(ind, layer_pops, rank_func)
        insert_crowding(ind, objective_pops)
        
        if len(objective_pops[-1]) > pop_size:
            rem_idx = min(range(len(layer_pops[-1])), key=lambda i: layer_pops[-1][i].distance)
            rem_ind = layer_pops[-1].pop(rem_idx)
            remove_crowding(rem_ind, objective_pops)
            
            if not layer_pops[-1]: del layer_pops[-1]

    # Bookkeeping for tracking the number of max_births towards are fixed
    # birth budget.
    birth_counter = util.inc_births(context, start=0)

    for i, evaluated_future in enumerate(as_completed_iter):

        evaluated = evaluated_future.result()

        if evaluated_probe is not None:
            # Give a chance to do something extra with the newly evaluated
            # individual, which is *usually* a call to
            # probe.log_worker_location, but can be any function that
            # accepts an individual as an argument
            evaluated_probe(evaluated)

        logger.debug('%d evaluated: %s %s', i, str(evaluated.genome),
                     str(evaluated.fitness))

        if not is_viable(evaluated):
            if not count_nonviable:
                # if we want the non-viables to not count towards the budget
                # then we need to decrement the birth counter to ensure that
                # a new individual is spawned to replace it.
                logger.debug(f'Non-viable individual, decrementing birth'
                             f'count.  Was {birth_counter.births()}')
                births = birth_counter.do_decrement()
                logger.debug(f'Birth count now {births}')
        else:
            # is viable, so bump that birth count er
            births = birth_counter.do_increment()
            logger.debug(f'Counting a birth.  '
                         f'Births at: {births}')

        inds_inserter(evaluated)

        if pop_probe is not None:
            pop_probe(objective_pops[0])

        if birth_counter.births() < max_births:
            logger.debug(f'Creating offspring because birth count is'
                         f'{birth_counter.births()}')
            # Only create offspring if we have the budget for one
            offspring = toolz.pipe(objective_pops[0], *offspring_pipeline)

            logger.debug('created offspring: ')
            [logger.debug('%s', str(o.genome)) for o in offspring]

            # Now asynchronously submit to dask
            for child in offspring:
                future = client.submit(evaluate(context=context), child,
                                       pure=False)
                as_completed_iter.add(future)

            # Be sure to count the new kids against the birth budget
            birth_counter.do_increment(len(offspring))
        else:
            logger.debug(f'Not creating offspring because birth count is'
                         f'{birth_counter.births()}')

    return objective_pops[0]
