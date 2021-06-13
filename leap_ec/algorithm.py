"""
    Provides convenient monolithic functions that wrap a lot of common function-
    ality.

    * generational_ea() for a typical generational model
    * multi_population_ea() for invoking an EA using sub-populations
    * random_search() for a more naive strategy
"""
from leap_ec import util
from toolz import pipe

from leap_ec.global_vars import context
from leap_ec.individual import Individual


##############################
# Function generational_ea
##############################
def generational_ea(max_generations, pop_size, problem, representation, pipeline,
                    stop=lambda x: False, init_evaluate=Individual.evaluate_population,
                    context=context):
    """
    This function provides an evolutionary algorithm with a generational
    population model.

    When called, this initializes and evaluates a population of size
    `pop_size` using the  and then pipes it through an operator `pipeline` (
    i.e. a list of operators) to obtain offspring.  Wash, rinse, repeat.

    The algorithm is provided  here at the "metaheuristic" level: in order to
    apply it to a particular problem, you must parameterize it with
    implementations of its various components: you must decide the population
    size, how individuals are represented and initialized, the pipeline of
    reproductive operators, etc. A metaheuristic template of this kind can be
    used to implement genetic algorithms, genetic programming, certain
    evolution strategies, and all manner of other (novel) algorithms by
    passing in appropriate components as parameters.

    :param int max_generations: The max number of generations to run the algorithm for.
        Can pass in float('Inf') to run forever or until the `stop` condition is reached.
    :param int pop_size: Size of the initial population
    :param int stop: A function that accepts a population and 
        returns True iff it's time to stop evolving.
    :param `Problem` problem: the Problem that should be used to evaluate
        individuals' fitness
    :param representation: How the problem is represented in individuals
    :param list pipeline: a list of operators that are applied (in order) to
        create the offspring population at each generation
    :param init_evaluate: a function used to evaluate the initial population,
        before the main pipeline is run.  The default of
        `Individual.evaluate_population` is suitable for many cases, but you
        may wish to pass a different operator in for distributed evaluation
        or other purposes.

    :return: a generator of `(int, individual_cls)` pairs representing the
        best individual at each generation.

    The intent behind this kind of EA interface is to allow the complete
    configuration of a basic evolutionary algorithm to be defined in a clean
    and readable way.  If you define most of the components in-line when
    passing them to the named arguments, then the complete configuration of
    an algorithmic experiment forms one concise code block.  Here's what a
    basic (mu, lambda)-style EA looks like (that is, an EA that throws away
    the parents at each generation in favor of their offspring):

    >>> from leap_ec import Individual, Representation
    >>> from leap_ec.algorithm import generational_ea, stop_at_generation
    >>> from leap_ec.binary_rep.problems import MaxOnes
    >>> from leap_ec.binary_rep.initializers import create_binary_sequence
    >>> from leap_ec.binary_rep.ops import mutate_bitflip
    >>> import leap_ec.ops as ops
    >>> pop_size = 5
    >>> ea = generational_ea(max_generations=100, pop_size=pop_size,
    ...
    ...                      problem=MaxOnes(),      # Solve a MaxOnes Boolean optimization problem
    ...
    ...                      representation=Representation(
    ...                          individual_cls=Individual,     # Use the standard Individual as the prototype for the population
    ...                          initialize=create_binary_sequence(length=10)  # Initial genomes are random binary sequences
    ...                      ),
    ...
    ...                      # The operator pipeline
    ...                      pipeline=[
    ...                          ops.tournament_selection,                     # Select parents via tournament selection
    ...                          ops.clone,                          # Copy them (just to be safe)
    ...                          mutate_bitflip(expected_num_mutations=1),     # Basic mutation with a 1/L mutation rate
    ...                          ops.uniform_crossover(p_swap=0.4),  # Crossover with a 40% chance of swapping each gene
    ...                          ops.evaluate,                       # Evaluate fitness
    ...                          ops.pool(size=pop_size)             # Collect offspring into a new population
    ...                      ])
    >>> ea # doctest:+ELLIPSIS
    <generator ...>

    The algorithm evaluates lazily when you query the generator:

    >>> print(*list(ea), sep='\\n') # doctest:+ELLIPSIS
    (0, Individual(...))
    (1, Individual(...))
    (2, Individual(...))
    ...
    (100, Individual(...))

    The best individual reported from the initial population  is reported at
    generation 0) followed by the best-so-far individual at each subsequent
    generation.
    """
    # Initialize a population of pop_size individuals of the same type as
    # individual_cls
    parents = representation.create_population(pop_size, problem=problem)

    # Set up a generation counter that records the current generation to
    # context
    generation_counter = util.inc_generation(context=context)

    # Evaluate initial population
    parents = init_evaluate(parents)

    # Output the best individual in the initial population
    bsf = max(parents)
    yield (0, bsf)

    while (generation_counter.generation() < max_generations) and not stop(parents):
        # Execute the operators to create a new offspring population
        offspring = pipe(parents, *pipeline)

        if max(offspring) > bsf:  # Update the best-so-far individual
            bsf = max(offspring)

        parents = offspring  # Replace parents with offspring
        generation_counter()  # Increment to the next generation

        # Output the best-so-far individual for each generation
        yield (generation_counter.generation(), bsf)


##############################
# Function multi_population_ea
##############################
def multi_population_ea(max_generations, num_populations, pop_size, problem,
                        representation, shared_pipeline,
                        subpop_pipelines=None, stop=lambda x: False,
                        init_evaluate=Individual.evaluate_population,
                        context=context):
    """
    An EA that maintains multiple (interacting) subpopulations, i.e. for
    implementing island models.

    This effectively executes several EAs concurrently that share the same
    generation counter, and which share the same representation (
    :py:class:`~leap.Individual`, :py:class:`~leap.Decoder`) and
    objective function (:py:class:`~leap.problem.Problem`), and which share
    all or part of the same operator pipeline.

    :param int max_generations: The max number of generations to run the algorithm for.
        Can pass in float('Inf') to run forever or until the `stop` condition is reached.
    :param int num_populations: The number of separate populations to maintain.
    :param int pop_size: Size of the initial population
    :param int stop: A function that accepts a list of populations and 
        returns True iff it's time to stop evolving.
    :param `Problem` problem: the Problem that should be used to evaluate
        individuals' fitness
    :param representation: the Decoder that should be used to convert
        individual genomes into phenomes
    :param list shared_pipeline: a list of operators that every population
        will uses to create the offspring population at each generation
    :param list subpop_pipelines: a list of population-specific operator
        lists, the ith of which will only be applied to the ith population (after
        the `shared_pipeline`).  Ignored if `None`.
    :param init_evaluate: a function used to evaluate the initial population,
        before the main pipeline is run.  The default of
        `Individual.evaluate_population` is suitable for many cases, but you
        may wish to pass a different operator in for distributed evaluation
        or other purposes.

    :return: a generator of `(int, [individual_cls])` pairs representing the
        best individual in each population at each generation.

    To turn a multi-population EA into an island model, use the
    :py:func:`leap_ec.ops.migrate` operator in the shared pipeline.  This
    operator takes a `NetworkX` graph describing the topology of connections
    between islands as input.

    For example, here's how we might define a fully connected 4-island model
    that solves a :py:class:`leap_ec.real_rep.problems.SchwefelProblem` using a
    real-vector representation:

    >>> import networkx as nx
    >>> from leap_ec.algorithm import multi_population_ea
    >>> from leap_ec import ops
    >>> from leap_ec.real_rep.ops import mutate_gaussian
    >>> from leap_ec.real_rep import problems
    >>> from leap_ec.decoder import IdentityDecoder
    >>> from leap_ec.representation import Representation
    >>> from leap_ec.real_rep.initializers import create_real_vector
    >>>
    >>> topology = nx.complete_graph(4)
    >>> nx.draw(topology)
    >>> problem = problems.SchwefelProblem(maximize=False)
    ...
    >>> l = 2  # Length of the genome
    >>> pop_size = 10
    >>> ea = multi_population_ea(max_generations=1000,
    ...                         num_populations=topology.number_of_nodes(),
    ...                         pop_size=pop_size,
    ... 
    ...                         problem=problem,
    ...
    ...                         representation=Representation(
    ...                             individual_cls=Individual,
    ...                             decoder=IdentityDecoder(),
    ...                             initialize=create_real_vector(bounds=[problem.bounds] * l)
    ...                             ),
    ...
    ...                         shared_pipeline=[
    ...                             ops.tournament_selection,
    ...                             ops.clone,
    ...                             mutate_gaussian(std=30,
    ...                                             expected_num_mutations='isotropic',
    ...                                             hard_bounds=problem.bounds),
    ...                             ops.evaluate,
    ...                             ops.pool(size=pop_size),
    ...                             ops.migrate(topology=topology,
    ...                                         emigrant_selector=ops.tournament_selection,
    ...                                         replacement_selector=ops.random_selection,
    ...                                         migration_gap=50)
    ...                         ])
    >>> ea # doctest:+ELLIPSIS
    <generator ...>

    We can now run the algorithm by pulling output from its generator,
    which gives us the best individual in each population at each generation:

    >>> from itertools import islice
    >>> result = list(islice(ea, 5))  # Run the first 5 generations by pulsing the generator
    >>> print(*result, sep='\\n')
    (0, [Individual(...), Individual(...), Individual(...), Individual(...)])
    (1, [Individual(...), Individual(...), Individual(...), Individual(...)])
    (2, [Individual(...), Individual(...), Individual(...), Individual(...)])
    (3, [Individual(...), Individual(...), Individual(...), Individual(...)])
    (4, [Individual(...), Individual(...), Individual(...), Individual(...)])

    While each population is executing, `multi_population_ea` writes the
    index of the current subpopulation to `context['leap'][
    'subpopulation']`.  That way shared operators (such as
    :py:func:`leap.ops.migrate`) have the option of accessing the share
    context to learn which subpopulation they are currently working with.

    """

    if not hasattr(problem, '__len__'):
        problem = [problem for _ in range(num_populations)]

    # Initialize populations of pop_size individuals of the same type as
    # individual_cls
    pops = [representation.create_population(pop_size, problem=problem[i])
            for i in range(num_populations)]
    # Include a reference to the populations in the context object.
    # This allows operators to see all of the subpopulations.
    context['leap']['subpopulations'] = pops
    # Evaluate initial population
    pops = [init_evaluate(p) for p in pops]

    # Set up a generation counter that records the current generation to the
    # context
    generation_counter = util.inc_generation(context=context)

    # Output the best individual in the initial population
    bsf = [max(p) for p in pops]
    yield (0, bsf)

    while (generation_counter.generation() < max_generations) and not stop(pops):
        # Execute each population serially
        for i, parents in enumerate(pops):
            # Indicate the subpopulation we are currently executing in the
            # context object. This allows operators to know which
            # subpopulation the are working with.
            context['leap']['current_subpopulation'] = i
            # Execute the operators to create a new offspring population
            operators = list(shared_pipeline) + \
                        (list(subpop_pipelines[i]) if subpop_pipelines else [])
            offspring = pipe(parents, *operators)

            if max(offspring) > bsf[i]:  # Update the best-so-far individual
                bsf[i] = max(offspring)

            pops[i] = offspring  # Replace parents with offspring

        generation_counter()  # Increment to the next generation

        # Output the best-of-gen individuals for each generation
        yield (generation_counter.generation(), bsf)


##############################
# Function random_search
##############################
def random_search(evaluations, problem, representation, pipeline=(),
                  context=context):
    """This function performs random search of a solution space using the
    given representation and problem.

    Random search is often used as a control in evolutionary algorithm experiments:
    if your pet algorithm can't perform better than random search, then it's a sign
    that you've barked up the wrong tree!

    This implementation also allows you to pass in an operator pipeline, which will
    be applied to each individual.  You'd usually use this to pass in probes, for
    example, to take measurements of the population.  But you could also use it to
    hybridize random search with, say, a local refinement procedure.


    >>> from leap_ec.binary_rep.problems import MaxOnes
    >>> from leap_ec.binary_rep.initializers import create_binary_sequence
    >>> from leap_ec.decoder import IdentityDecoder
    >>> from leap_ec.representation import Representation
    >>> from leap_ec.individual import Individual
    >>> ea = random_search(evaluations=100,
    ...                    problem=MaxOnes(),      # Solve a MaxOnes Boolean optimization problem
    ...
    ...                    representation=Representation(
    ...                        individual_cls=Individual,     # Use the standard Individual as the prototype for the population
    ...                        decoder=IdentityDecoder(),     # Genotype and phenotype are the same for this task
    ...                        initialize=create_binary_sequence(length=10)  # Initial genomes are random binary sequences
    ...                    ))
    >>> ea # doctest:+ELLIPSIS
    <generator ...>

    The algorithm evaluates lazily when you query the generator:

    >>> print(*list(ea), sep='\\n') # doctest:+ELLIPSIS
    (1, Individual(...))
    (2, Individual(...))
    ...
    (100, Individual(...))

    The best individual reported from the initial population  is reported at
    generation 0) followed by the best-so-far individual at each subsequent
    generation.
    """
    # Set up an evaluation counter that records the current generation to
    # context
    evaluation_counter = util.inc_generation(context=context)
    bsf = None

    while evaluation_counter.generation() < evaluations:
        # Use the representation to sample a new individual
        population = representation.create_population(1, problem=problem)

        # Fitness evaluation
        population = Individual.evaluate_population(population)

        # Apply some operators to the new individual.
        # For example, we'd put probes here.
        population = pipe(population, *pipeline)

        if max(population) > bsf:  # Update the best-so-far individual
            bsf = max(population)

        evaluation_counter()  # Increment to the next evaluation

        # Output the best-so-far individual for each generation
        yield (evaluation_counter.generation(), bsf)


##############################
# Function stop_at_generation()
##############################
def stop_at_generation(max_generation: int, context=context):
    """A stopping criterion function that checks the 'generation' count in the `context` 
    object and returns True iff it is >= `max_generation`.

    The resulting function takes a `population` argument, which is ignored.

    For example:

    >>> from leap_ec import context
    >>> stop = stop_at_generation(100)

    If we set the generation field in the context object (this value will typically be
    updated by the algorithm as it runs) like so:

    >>> context['leap']['generation'] = 15

    Then we don't stop yet:

    >>> stop(population=[])
    False

    We do stop at the 100th generation:

    >>> context['leap']['generation'] = 100
    >>> stop([])
    True
    """
    assert(max_generation >= 0)
    assert(context is not None)
    
    def stop(population):
        return not (context['leap']['generation'] < max_generation)
    
    return stop
