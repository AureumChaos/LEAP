"""An example of an evolutionary algorithm that makes use of LEAP's integer
representation.

We use a generational EA with binomial mutation of integer genes to minimize an
integer version of the Langermann function.
"""
from queue import PriorityQueue
import sys

from matplotlib import pyplot as plt
import numpy as np
import toolz

from leap_ec.algorithm import generational_ea
from leap_ec.representation import Representation
from leap_ec import ops
from leap_ec.real_rep.initializers import create_real_vector
from leap_ec.real_rep.ops import mutate_gaussian
from leap_ec import probe
from leap_ec.real_rep.problems import LangermannProblem


class AsyncEvaluationSimulator():
    def __init__(self, num_processors: int, eval_time_function):
        assert(num_processors > 0)
        assert(eval_time_function is not None)
        self.num_processors = num_processors
        self.eval_time_function = eval_time_function
        self.queue = PriorityQueue(maxsize=num_processors)

    def async_init_evaluate(self, population):
        assert(self.num_processors <= len(population))

        next_individual = iter(population)

        # First fill up all but one processor
        for _ in range(self.num_processors - 1):
            ind = next(next_individual)
            self._send_individual(ind)

        # Send off the rest of the population using steady-state logic
        evaluated = self.async_evaluate(next_individual)
        new_pop = list(evaluated)
        assert(len(new_pop) == len(population) - (self.num_processors - 1)), f"After initialization, we have {self.queue.qsize()} individuals processing and {len(new_pop)} completed; but expected {len(population) - (self.num_processors - 1)} completed."

        return new_pop

    def async_evaluate(self, next_individual):
        for ind in next_individual:
            self._send_individual(ind)
            yield self._receive_individual()

    def _send_individual(self, ind):
        eval_time = self.eval_time_function(ind)
        self.queue.put( (eval_time, ind) )

    def _receive_individual(self):
        eval_time, ind = self.queue.get()
        ind.evaluate()  # Do actual fitness evaluation here
        ind.eval_time = eval_time
        return ind


@toolz.curry
@ops.listlist_op
def steady_state_step(population: list, reproduction_pipeline: list, insert, evaluation_op = ops.evaluate):
    """An operator that performs steady-state evolution when placed in an (otherwise
    generational) pipeline.

    This is a metaheuristic component that can be parameterized to define many kinds of 
    steady-state evolution.  It takes a population, uses the `reproduction_pipeline` to
    produce a single new individual, evaluates it with the provided `evaluation_op` operator,
    and then inserts the individual returned by `evaluation_op` into the population using the
    strategy defined by `insert`.
    """
    offspring = next(toolz.pipe(population, *reproduction_pipeline))
    evaluated = next(evaluation_op(iter([ offspring ])))
    new_pop = insert(population, evaluated)
    return new_pop


##############################
# Function competition_inserter
##############################
@toolz.curry
def competition_inserter(population, new_individual, p_accept: float, replacement_selector):
    assert(replacement_selector is not None)
    indices = []
    competitor = next(replacement_selector(population, indices=indices))
    assert(len(indices) == 1)
    competitor_index = indices[0]

    new_pop = population[:]
    accept = np.random.uniform(0, 1) < p_accept
    if accept or new_individual > competitor:
        new_pop[competitor_index] = new_individual
    
    return new_pop


##############################
# main
##############################
if __name__ == '__main__':
    # Our fitness function will be the Langermann
    # This is defined over a real-valued space, but
    # we can also use it to evaluate integer-valued genomes.
    problem = LangermannProblem(maximize=False)

    l = 2
    pop_size = 10

    eval_sim = AsyncEvaluationSimulator(num_processors=5, eval_time_function=lambda x: np.random.uniform())

    ea = generational_ea(generations=1000,pop_size=pop_size,
                             init_evaluate=eval_sim.async_init_evaluate,
                             problem=problem,  # Fitness function

                             # Representation
                             representation=Representation(
                                 # Initialize a population of integer-vector genomes
                                 initialize=create_real_vector(bounds=[problem.bounds] * l)
                             ),

                             # Operator pipeline
                             pipeline=[
                                 steady_state_step(
                                     reproduction_pipeline=[
                                         ops.tournament_selection,
                                         ops.clone,
                                         mutate_gaussian(std=1.5, hard_bounds=[problem.bounds]*l,
                                                expected_num_mutations=1)
                                     ],
                                     insert=competition_inserter(p_accept=0.5,
                                         replacement_selector=ops.tournament_selection(key=lambda x: -x.fitness)
                                     ),
                                     evaluation_op=eval_sim.async_evaluate  # ops.evaluate
                                 ),
                                 # Some visualization probes so we can watch what happens
                                 probe.CartesianPhenotypePlotProbe(xlim=problem.bounds, ylim=problem.bounds,
                                        contours=problem),
                                 probe.FitnessPlotProbe(),
                                 probe.FitnessStatsCSVProbe(stream=sys.stdout)
                                 # TODO Use AttributesCSVProbe to report evaluation times
                             ]
                        )

    list(ea)
