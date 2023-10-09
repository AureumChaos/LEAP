LEAP Cookbook
=============

This is a collection of "recipes" in the spirit of the O'Reilly "Cookbook" series. That
is, it's a collection of common howtos and examples for using LEAP.

Enforcing problem bounds constraints
------------------------------------
There are two overall types of bounds enforcement within EAs, soft bounds and
hard bounds:

:soft bounds: where the boundaries are enforced only at initialization, but
    mutation allows for exploring beyond those initial boundaries

:hard bounds: boundaries are strictly enforced at initialization as well as
    during mutation and crossover.  In the latter case this can be done by
    clamping new values to a given range, or flagging an individual that violates
    such constraints as non-viable by throwing an exception during fitness
    evaluation.  (That is, during evaluation, exceptions are caught, which causes
    the individual's fitness to be set to NaN and its `is_viable` internal flag
    set to false; then selection should hopefully weed out this individual
    from the population.)

Bounds for initialization
^^^^^^^^^^^^^^^^^^^^^^^^^

When initializing a population with genomes of numeric values, such as
integers or real-valued numbers, the bounds for each gene needs to be specified so
that we know in what range to initialize the genes.

For real-valued genomes, :func:`leap_ec.real_rep.create_real_vector` takes a
list of tuples where each tuple is a pair of lower and upper bounds for each gene.
For example, the following initializes a genome with three genes, where each gene
is in the range [0, 1], [0, 1], and [-1, 100], respectively:

.. code-block:: python

    from leap_ec.real_rep import create_real_vector
    bounds = [(0, 1), (0, 1), (-1, 100)]
    genome = create_real_vector(bounds)


For integer-valued genomes, :func:`leap_ec.int_rep.create_int_vector` works identically.
That is, `create_int_vector` accepts a collection of tuples of pairs of lower and
upper bounds for each gene.

Enforcing bounds during mutation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

That's great for _initializing_ a population, but what about when we mutate?  If no
precautions are taken, then mutation is free to create genes that are well outside
the initialization bounds.  Fortunately for any numeric mutation operator, we can specify
a bounds constraint that will be enforced during mutation.  The functions
:func:`leap_ec.int_rep.ops.mutate_randint`, :func:`leap_ec.int_rep.ops.binomial`,
:func:`leap_ec.real_rep.ops.mutate_gaussian` accept a `bounds` parameter that, as with
the initializers, is a list of tuples of lower and upper bounds for each gene. Mutations
that stray outside of these bounds are clamped to the nearest boundary. `numpy.clip` is
used to efficiently clip the values to the bounds.