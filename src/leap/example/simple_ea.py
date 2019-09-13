import click

from leap import core, real, probe
from leap import operate as op


def simple_ea(evals, pop_size, individual_cls, decoder, problem, evaluate, initialize, pipeline):
    """
    An example implementation of a basic evolutionary algorithm.

    This function initializes and evaluates a population of size `pop_size`, and then pipes it through an operator
    `pipeline` (i.e. a list of operators) to obtain offspring.  Wash, rinse, repeat.

    The algorithm here is implement at the "metaheuristic" level.  In order to apply it to a particular problem, you
    must provide implementations of its various components: you must decide the population size, how individuals are
    represented and initialized, the pipeline of reproductive operators, etc.

    :param int evals: the stopping condition—stop after `evals` fitness evaluations
    :param int pop_size: Size of the initial population
    :param class individual_cls: class representing the (sub)type of `Individual` the population should be generated
        from
    :param `Decoder` decoder: the Decoder that should be used to convert individual genomes into phenomes
    :param `Problem` problem: the Problem that should be used to evaluate individuals' fitness
    :param evaluate: the evaluation operator
    :param initialize: a function that creates the genomes for the initial population (takes an integer `n` and returns
        a list of `n` genomes
    :param list pipeline: a list of operators that are applied (in order) to the population each generation
    :return: a generator of `(int, individual_cls)` pairs representing the best individual at each generation.

    The intent behind this kind of EA interface is to allow the complete configuration of a basic evolutionary
    algorithm to be defined in a clean and readable way.  If you define most of the components in-line when passing
    them to the named arguments, then the complete configuration of an algorithmic experiment forms one concise code
    block.  Here's what a basic (mu, lambda)-style EA looks like (that is, an EA that throws away the parents at each
    generation in favor of their offspring):

    >>> from leap import core, real
    >>> from leap import operate as op
    >>> pop_size = 5  # Size of the parent population
    >>> l = 10  # The length of the genome
    >>> ea = simple_ea(evals=1000, pop_size=pop_size,
    ...                individual_cls=core.Individual,  # Use the standard Individual as the prototype for the population
    ...                decoder=core.IdentityDecoder(),  # Genotype and phenotype are the same for this task
    ...                problem=real.Spheroid(maximize=False),  # Solve a Spheroid minimization problem
    ...                evaluate=op.evaluate,  # Evaluate fitness with the basic evaluation operator
    ...
    ...                # Initialized genomes are random real-valued vectors
    ...                initialize=real.initialize_vectors_uniform(
    ...                    # Initialize each element between 0 and 1
    ...                    bounds=[[-5.12, 5.12]] * l
    ...                ),
    ...
    ...                # The operator pipeline
    ...                pipeline=[
    ...                    # Select mu parents via tournament selection
    ...                    op.tournament(n=pop_size),
    ...                    # Clone them to create offspring
    ...                    op.cloning,
    ...                    # Apply Gaussian mutation to each gene with a certain probability
    ...                    op.mutate_gaussian(prob=0.1, std=0.05)
    ...                ])
    >>> ea # doctest:+ELLIPSIS
    <generator ...>

    The algorithm evaluates lazily when you query the generator:

    >>> print(*list(ea), sep='\\n') # doctest:+ELLIPSIS
    (5, Individual(...))
    (10, Individual(...))
    (15, Individual(...))
    ...
    (1000, Individual(...))

    In this case, we see that the best individual reported from the initial population (after `pop_size = 5` evals),
    followed by reports for each generation (every five generations, since our pipeline chooses and clones `pop_size`
    parents every generation.
    """

    # Initialize a population of pop_size individuals of the same type as individual_cls
    population = individual_cls.create_population(pop_size, initialize, decoder, problem)
    # Evaluate the population's fitness once before we start the main loop
    population, _ = evaluate(population)

    i = pop_size  # Eval counter
    best = lambda pop: probe.best_of_gen(pop)[0]
    yield (i, best(population))  # Yield the best individual in the initial population
    while i < evals:
        # Run the population through the operator pipeline
        # We aren't using any context data, so we pass in None
        population, _ = op.do_pipeline(population, None, *pipeline)
        population, _ = evaluate(population)  # Evaluate the fitness of the offspring population
        i += len(population)  # Increment the eval counter by the size of the population
        yield (i, best(population))  # Yield the best individual for each generation


@click.group()
def cli():
    """Entry point for the click command-line application."""
    pass


@cli.command()
@click.option('--evals', default=100, help='Fitness evaluations to run for')
@click.option('--pop_size', default=5, help='Population size')
@click.option('--l', default=10, help='Length of the genome')
@click.option('--mutate-prob', default=0.1, help='Per-gene Gaussian mutation rate')
@click.option('--mutate-std', default=0.05, help='Standard deviation of Gaussian mutation')
def mu_comma_lambda(evals, pop_size, l, mutate_prob, mutate_std):
    """Apply a (μ, λ)-style generational EA with tournament selection and Gaussian
    mutation to the `Spheroid` function."""
    ea = simple_ea(evals=evals, pop_size=pop_size,
                   individual_cls=core.Individual,  # Use the standard Individual as the prototype for the population.
                   decoder=core.IdentityDecoder(),  # Genotype and phenotype are the same for this task.
                   problem=real.Spheroid(maximize=False),  # Solve a Spheroid minimization problem.
                   evaluate=op.evaluate,  # Evaluate fitness with the basic evaluation operator.

                   # Initialized genomes are random real-valued vectors.
                   initialize=real.initialize_vectors_uniform(
                       # Initialize each element between 0 and 1.
                       bounds=[[-5.12, 5.12]] * l
                   ),

                   # The operator pipeline.
                   pipeline=[
                       # Select mu parents via tournament selection.
                       op.tournament(n=pop_size),
                       # Clone them to create offspring.
                       op.cloning,
                       # Apply Gaussian mutation to each gene with a certain probability.
                       op.mutate_gaussian(prob=mutate_prob, std=mutate_std)
                   ])

    print('generation, best_of_gen_fitness')
    for (i, ind) in ea:
        print('{0}, {1}'.format(i, ind.fitness))


@cli.command()
@click.option('--evals', default=100, help='Fitness evaluations to run for')
@click.option('--mu', default=5, help='Population size')
@click.option('--lambda', 'lambda_', default=5, help='Population size')
@click.option('--l', default=10, help='Length of the genome')
@click.option('--mutate-prob', default=0.1, help='Per-gene Gaussian mutation rate')
@click.option('--mutate-std', default=0.05, help='Standard deviation of Gaussian mutation')
def mu_plus_lambda(evals, mu, lambda_, l, mutate_prob, mutate_std):
    """Apply a (μ + λ)-style generational EA with truncation selection and Gaussian
    mutation to the `Spheroid` function."""
    # Setup a (μ + λ) concatenation operator
    mu_plus_lambda_cat = op.MuPlusLambdaConcatenation()
    # Setup a probe to collect the BSF fitness
    bsf_probe = probe.memory_probe(probe=probe.BestSoFar(just_fitness=True))
    ea = simple_ea(evals=evals, pop_size=mu + lambda_,
                   individual_cls=core.Individual,  # Use the standard Individual as the prototype for the population.
                   decoder=core.IdentityDecoder(),  # Genotype and phenotype are the same for this task.
                   problem=real.Spheroid(maximize=False),  # Solve a Spheroid minimization problem.
                   evaluate=op.evaluate,  # Evaluate fitness with the basic evaluation operator.

                   # Initialized genomes are random real-valued vectors.
                   initialize=real.initialize_vectors_uniform(
                       # Initialize each element between 0 and 1.
                       bounds=[[-5.12, 5.12]] * l
                   ),

                   # The operator pipeline
                   pipeline=[
                       # Collect the BSF fitness at the start of each generation
                       bsf_probe,
                       # Choose the best μ individuals to serve as parents
                       op.truncation(mu=mu),
                       # Save the parents for later.
                       # (This function saves state into the MuPlusLambdaConcatenation operator)
                       mu_plus_lambda_cat.capture_parents,
                       # Clone the parents to create offspring.
                       op.cloning,
                       # Apply Gaussian mutation to each gene with a certain probability.
                       op.mutate_gaussian(prob=mutate_prob, std=mutate_std),
                       # Concatenate parents and offspring.
                       mu_plus_lambda_cat
                   ])

    print('generation, best_of_gen_fitness')
    for (i, ind) in ea:
        print('{0}, {1}'.format(i, ind.fitness))


if __name__ == '__main__':
    # Just call our click CLI interface
    cli()
