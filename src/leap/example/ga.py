from leap import core
from leap import operate as op
from leap import real


def generational(evals, mu, lambda_, individual_cls, decoder, problem, evaluate, initialize, pipeline):
    """

    :param evals:
    :param initialize:
    :param mu:
    :param lambda_:
    :param pipeline:
    :return:

    >>> from leap import core, real
    >>> from leap import operate as op
    >>> mu = 5
    >>> l = 5
    >>> ea = generational(evals=1000, mu=mu, lambda_=mu,
    ...                   individual_cls=core.Individual,  # Use the standard Individual as the prototype for the population
    ...                   decoder=core.IdentityDecoder(),  # Genotype and phenotype are the same for this task
    ...                   problem=real.Spheroid(maximize=False),  # Solve a Spheroid minimization problem
    ...                   evaluate=op.evaluate,  # Evaluate fitness with the basic evaluation operator
    ...
    ...                   # Initialized genomes are random real-valued vectors
    ...                   initialize=real.initialize_vectors_uniform(
    ...                       # Initialize each element between 0 and 1
    ...                       bounds=[[0, 1.0]] * l
    ...                   ),
    ...
    ...                   # The operator pipeline
    ...                   pipeline=[
    ...                       # Select mu parents via tournament selection
    ...                       op.tournament(n=mu),
    ...                       # Clone them to create offspring
    ...                       op.cloning,
    ...                       # Apply Gaussian mutation to each gene with a certain probability
    ...                       op.mutate_gaussian(prob=0.1, std=0.05)
    ...                   ])
    >>> ea # doctest:+ELLIPSIS
    <generator ...>

    The algorithm evaluates lazily when you query the generator:

    >>> print(*list(ea), sep='\\n') # doctest:+ELLIPSIS
    (15, Individual(...))
    (20, Individual(...))
    ...
    (1000, Individual(...))
    """

    # Initialize a population of mu + lambda_ individuals of the same type as individual_cls
    population = individual_cls.create_population(mu + lambda_, initialize, decoder, problem)
    # Evaluate the population's fitness once before we start the main loop
    population = evaluate(population)

    i = mu + lambda_ # Eval counter
    while i < evals:
        # Run the population through the operator pipeline
        # We aren't using any context data, so we pass in None
        population = op.do_pipeline(population, None, *pipeline)
        population = evaluate(population)  # Evaluate the fitness of the offspring population
        i += len(population)  # Increment the eval counter by the size of the population
        yield (i, op.best(population))  # Yield the best individual for each generation


if __name__ == '__main__':
    mu = 5  # Parent population size
    l = 10  # Length of the genome
    ea = generational(evals=1000, mu=mu, lambda_=mu,
                      individual_cls=core.Individual,  # Use the standard Individual as the prototype for the population
                      decoder=core.IdentityDecoder,  # Genotype and phenotype are the same for this task
                      problem=real.Spheroid(maximize=False),  # Solve a Spheroid minimization problem
                      evaluate=op.evaluate,  # Evaluate fitness with the basic evaluation operator

                      # Initialized genomes are random real-valued vectors
                      initialize=real.initialize_vectors_uniform(
                          # Initialize each element between 0 and 1
                          bounds=[[0, 1.0]] * l
                      ),

                      # The operator pipeline
                      pipeline=[
                          # Select mu parents via tournament selection
                          op.tournament(n=mu),
                          # Clone them to create offspring
                          op.cloning,
                          # Apply Gaussian mutation to each gene with a certain probability
                          op.mutate_gaussian(prob=0.1, std=0.05)
                      ])

    print('generation, best_of_gen_fitness')
    for (i, ind) in ea:
        print('i, {0}'.format(ind.fitness))

