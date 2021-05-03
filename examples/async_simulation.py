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
from leap_ec import ops, probe
from leap_ec.problem import ConstantProblem, ScalarProblem, FunctionProblem
from leap_ec.real_rep import problems as real_prob
from leap_ec.real_rep.initializers import create_real_vector
from leap_ec.real_rep.ops import mutate_gaussian

class AsyncEvaluationSimulator():
    """A queue-based simulation of asynchronous fitness evaluation.

    This evaluates individuals all on one thread—but it *pretends* that
    each evaluation takes a certain amount of *time* according to some
    distribution (given by `eval_time_function`), and that it has 
    `num_processors` processors to distribute evaluation across.

    This allows us to to produce asynchronous evolutionary algorithms 
    whose search trajectory behaves as it would if we assume the
    given eval-time distribution, but which run very fast (given a
    fitness function that is, in reality, very cheap).
    """
    def __init__(self, num_processors: int, eval_time_function):
        assert(num_processors > 0)
        assert(eval_time_function is not None)
        self.num_processors = num_processors
        self.eval_time_function = eval_time_function
        self.queue = PriorityQueue(maxsize=num_processors)

    def async_init_evaluate(self, population):
        """Evaluates an initial population, by sending individuals
        out to processors for evaluation, and building a new
        population by adding individuals in the order that they
        complete evaluating.

        This operator is meant to be used at initialization time, to
        asynchronously initialize a population in preparation for
        steady-state evolution.
        
        The population returned here is not full: after this
        function completes, there will still be `num_processors - 1`
        individuals evaluating on the processors.
        
        The result is a simulation with exactly one free processor—
        ready to be filled with an offspring individual when
        steady-state evolution begins."""
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
        """Sends individuals off to be evaluated asynchronously,
        and returns the next individuals to be completed.

        This operator is meant to be used with `steady_state_step`
        to plug into the regular evolutionary portion of an EA.
        
        The output of this function depends on what is currently 
        in the processing queue: we return what *finishes* evaluating,
        but this may be a different individual than the one we just
        *sent* to be evaluated.
        """
        for ind in next_individual:
            self._send_individual(ind)
            yield self._receive_individual()

    def _send_individual(self, ind):
        """Place an individual into the priority queue."""
        eval_time = self.eval_time_function(ind.decode())
        self.queue.put( (eval_time, ind) )

    def _receive_individual(self):
        """Retrieve the next individual from the priority queue,
        assign it a fitness value, and record its evalution time."""
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
# Class TwoBasinProblem
##############################
class TwoBasinProblem(ScalarProblem):
    def __init__(self, a: float, b: float, dimensions: int, maximize=False):
        assert(dimensions > 0)
        self.a, self.b = a, b
        self.dimensions = dimensions
        self.maximize = maximize
        self.basin_1 = real_prob.TranslatedProblem(
            problem=real_prob.GaussianProblem(width=1, height=1, maximize=maximize),
            offset=[-1]*dimensions,
            maximize=False
        )
        self.basin_2 = real_prob.TranslatedProblem(
            problem=real_prob.GaussianProblem(width=1, height=1, maximize=maximize),
            offset=[1]*dimensions,
            maximize=False
        )

    @property
    def bounds(self):
        return [-2, 2]

    def evaluate(self, phenome):
        assert(len(phenome) == self.dimensions), f"Got {len(phenome)} dimensions, expected {self.dimensions}."
        y_offset = -min(self.a, self.b) if self.a < 0 or self.b < 0 else 0
        return y_offset + self.a*self.basin_1.evaluate(phenome) + self.b*self.basin_2.evaluate(phenome)


##############################
# main
##############################
if __name__ == '__main__':
    dim = 10

    #problem = TwoBasinProblem(a=-1, b=-1, dimensions=dim)
    #problem = ConstantProblem()
    problem = FunctionProblem(lambda x: np.exp(np.sum(x)), maximize=True)
    problem.bounds = [0, 3]

    eval_time_form = problem
    #eval_time_form = TwoBasinProblem(a=-1, b=1, dimensions=dim)
    eval_time_f = lambda x: eval_time_form.evaluate(x)
    #eval_time_f = lambda x: np.random.uniform()
    #eval_time_f = lambda x: 1.0
    bounds = problem.bounds

    pad_val = 0  # Value to fix higher-dimensional values at when projecting landscapes into 2-D visuals
    real_prob.plot_2d_problem(problem, xlim=problem.bounds, ylim=problem.bounds, pad=[pad_val]*(dim - 2), title="Fitness Landscape")
    real_prob.plot_2d_function(eval_time_f, xlim=bounds, ylim=bounds, pad=[pad_val]*(dim - 2), title="Eval-Time Landscape")

    pop_size = 10

    eval_sim = AsyncEvaluationSimulator(num_processors=5, eval_time_function=eval_time_f)

    ea = generational_ea(generations=1000,pop_size=pop_size,
                             init_evaluate=eval_sim.async_init_evaluate,
                             problem=problem,  # Fitness function

                             # Representation
                             representation=Representation(
                                 # Initialize a population of integer-vector genomes
                                 initialize=create_real_vector(bounds=[problem.bounds] * dim)
                             ),

                             # Operator pipeline
                             pipeline=[
                                 steady_state_step(
                                     reproduction_pipeline=[
                                         ops.tournament_selection,
                                         ops.clone,
                                         mutate_gaussian(std=1.5, hard_bounds=[problem.bounds]*dim,
                                                expected_num_mutations=1)
                                     ],
                                     insert=competition_inserter(p_accept=0.5,
                                         replacement_selector=ops.tournament_selection(key=lambda x: -x.fitness)  # FIXME This is wrong on minimization tasks
                                     ),
                                     evaluation_op=eval_sim.async_evaluate  # ops.evaluate
                                 ),
                                 # Some visualization probes so we can watch what happens
                                 probe.CartesianPhenotypePlotProbe(xlim=problem.bounds, ylim=problem.bounds,
                                        contours=problem, pad=[pad_val]*(dim - 2)),
                                 probe.FitnessPlotProbe(),
                                 probe.FitnessStatsCSVProbe(stream=sys.stdout)
                                 # TODO Use AttributesCSVProbe to report evaluation times
                             ]
                        )

    list(ea)
