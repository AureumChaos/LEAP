import toolz

from leap import operate as op


def generational(evals, mu, lambda_, initialize, evaluate, pipeline):
    """

    :param evals:
    :param initialize:
    :param mu:
    :param lambda_:
    :param pipeline:
    :return:

    >>> from leap import operate as op
    >>> from leap import real, decode
    >>> mu = 5
    >>> l = 5
    >>> ea = generational(evals=1000, mu=mu, lambda_=5,
    ...                   initialize=real.initialize_vectors(
    ...                        decode.IdentityDecoder(),
    ...                        problem = real.CosineFamilyProblem(alpha=0.6,
    ...                                                           global_optima_counts=[5]*l,
    ...                                                           local_optima_counts=[5]*l),
    ...                        bounds=[[0, 1.0]]*l),
    ...                   evaluate=op.evaluate,
    ...                   pipeline=[
    ...                        op.tournament(n=mu),
    ...                        op.cloning,
    ...                        op.mutate_gaussian(prob=0.1, std=0.05)
    ...                   ])
    >>> ea # doctest:+ELLIPSIS
    <generator ...>

    The algorithm evaluates lazily when you query the generator:

    >>> print(*list(ea), sep='\\n') # doctest:+ELLIPSIS
    (15, [...])
    (20, [...])
    ...
    (1000, [...])
    """
    population = initialize(mu + lambda_)
    population = evaluate(population)

    i = mu + lambda_
    while i < evals:
        population = toolz.pipe(population, *pipeline)
        population = evaluate(population)
        i += len(population)
        yield (i, op.best(population))
