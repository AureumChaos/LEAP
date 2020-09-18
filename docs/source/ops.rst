Pipeline Operators
==================

.. figure:: _static/Pipeline.png

    **Figure 2: LEAP operator pipeline.** This figure depicts a typical
    LEAP operator pipeline.  First is a parent population from which the
    next operator selects individuals, which are then cloned by the next
    operator to be followed by operators for mutating and evaluating the
    individual.  (For brevity, a crossover operator was not included, but
    could also have been freely inserted into this pipeline.)  The pool
    operator is a sink for offspring, and drives the demand for the upstream
    operators to repeatedly select, clone, mutate, and evaluate individuals
    repeatedly until the pool has the desired number of offspring.  Lastly,
    another selection operator returns the final set of individuals based
    on the offspring pool and optionally the parents.

Overiew
-------

:py:class:`leap_ec.individual.Individual`, :py:class:`leap_ec.problem.Problem`,
and :py:class:`leap_ec.decoder.Decoder` are passive classes that need an
external framework to make them function.  In :doc:`concepts` the notion of a
pipeline of evolutionary algorithm (EA) operators that use these classes was
introduced.  That is, `Individual`, `Decoder`, and `Problem`are the "nouns"
and the pipeline operators a the verbs that operate on those nouns.  The
operator pipeline objective is to create a new set of evaluated individuals
from an existing set of prospective parents that can be in a new set of
prospective parents.

Fig.2 is shown again here to depict a typical set of LEAP pipeline
operators.  The pipeline generally starts with a "sink", or a parent population
from which the next operator typically selects for creating offspring. This is
followed by a clone operator that ensure the subsequent pertubation operators
do not modify the selected parents.  (And so it is critically important that
users *always* have a clone operator as a part of the offspring creation
pipeline before any mutation, crossover, or other genome altering operators.)
The pertubation operators can be mutation or also include a crossover
operator. At this point in the pipeline we have a completed offspring with no
fitness, so the next operator evaluates the offspring to assign that fitness.
Then the evaluated offspring is collected into a pool of offspring.  Once the
offspring pool reaches a desired size it returns all the offspring to another
selection operator to cull the offspring, and optionally the parents, to
return the next set of prospective parents.

Or, more explicitly:

#. Start with a collection of `Individuals` that are prospective parents
#. A selection operator for selecting one or more parents to begin the creation of a new offspring
#. A clone operator that makes a copy of the selected parents to ensure the following operators don't overwrite those parents
#. A set of mutation, crossover, or other operators that perturb the cloned individual's genome, thus (hopefully) giving the new offspring unique values
#. An operator to evaluate the new offspring
#. A pool that serves as a "sink" for evaluated offspring; this pool is sent to the next operator, or is returned from the function, once the pool reaches a specified size
#. Another selection operator to cull the offspring (and optionally parents) to return a population of new prospective parents

This is, the general sequence for most LEAP pipelines, but there will be the occasional variation
on this theme.  For example, many of the provided "canned" algorithms take just
*snippets* of an offspring creation pipeline.  E.g., :py:func:`leap_ec.distributed.asynchronous.steady_state`
has an `offspring_pipeline` parameter that doesn't have parents explicitly
as part of the pipeline; instead, for `steady_state()` it's *implied* that the parents will be provided during the
run internally.


Implementation Details
----------------------

The LEAP pipeline is implemented using the ``toolz.functoolz.pipe()`` function,
which has arguments comprised of a collection of data followed by an arbitrary
number of functions.  When invoked the data is passed as an argument to the first
function, and the output of that function is fed as an argument to the next
function --- this repeats for the rest of the functions.  The output of the last
function is returned as the overall pipeline output.
(See: https://toolz.readthedocs.io/en/latest/api.html#toolz.functoolz.pipe )

Loose-coupling via generator functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The first "data" argument is a collection of `Individuals`representing
prospective parents, which can be a sequence, such as a list or tuple.  The
design philosophy for the operator functions that follow was to ensure they
were as loosely coupled as possible. This was achieved by implementing some
operators as generator functions that accept iterators as arguments.  That way,
new operators can be spliced into the pipeline and they'd automatically "hook
up" to their neighbors.

For example, consider the following snippet:

.. code-block:: python

    gen = 0
    while gen < max_generation:
        offspring = toolz.pipe(parents,
                               ops.tournament,
                               ops.clone,
                                   mutate_bitflip,
                               ops.evaluate,
                               ops.pool(size=len(parents)))

        parents = offspring
        gen += 1

The above code snippet is an example of a very basic genetic algorithm
implementation that uses a `toolz.pipe()` function to link
together a series of operators to do the following:

#. binary tournament selection on a set of parents
#. clone those that were selected
#. perform mutation bit-flip on the clones
#. evaluate the offspring
#. accumulate as many offspring as there are parents

Since we only have mutation in the pipeline, only one parent at a time is selected
to be cloned to create an offspring.  However, let's make one change to that pipeline
by adding crossover:

.. code-block:: python

    gen = 0
    while gen < max_generation:
        offspring = toolz.pipe(parents,
                               ops.tournament,
                               ops.clone,
                                   mutate_bitflip,
                               ops.uniform_crossover, # NEW OPERATOR
                               ops.evaluate,
                               ops.pool(size=len(parents)))

        parents = offspring
        gen += 1

This does the following:

#. binary tournament selection on a set of parents
#. clone those that were selected
#. perform mutation bitflip on the clones
#. perform uniform crossover between the two offspring
#. evaluate the offspring
#. accumulate as many offspring as there are parents

Adding crossover means that now **two** parents are selected instead of one. However,
note that the tournament selection operator wasn't changed.  It automatically
selects two parents instead of one, as necessary.

Let's take a closer look at `uniform_crossover()` (this is a simplified version;
the actual code has more type checking and docstrings).

.. code-block:: python

    def uniform_crossover(next_individual: Iterator,
                          p_swap: float = 0.5) -> Iterator:
        def _uniform_crossover(ind1, ind2, p_swap):
            for i in range(len(ind1.genome)):
                if random.random() < p_swap:
                    ind1.genome[i], ind2.genome[i] = ind2.genome[i], ind1.genome[i]

            return ind1, ind2

        while True:
            parent1 = next(next_individual)
            parent2 = next(next_individual)

            child1, child2 = _uniform_crossover(parent1, parent2, p_swap)

            yield child1
            yield child2

Note that the argument `next_individual` is an `Iterator` that "hooks up" to a
previously `yielded` `Individual` from the previous pipeline operator.  The
`uniform_crossover` operator doesn't care how the previous `Individual` is made,
it just has a contract that when `next()` is invoked that it will get a new
`Individual`.  And, since this is a generator function, it `yields` the
crossed-over `Individuals`.  It also has *two* `yield` statements that
ensures both crossed-over `Individuals`are returned, thus eliminating a potential
source of genetic drift by only yielding one.

Currying Function Decorators
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Safety Decorator Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^


Examples
^^^^^^^^


API Documentation
-----------------

Base operator classes and representation agnostic functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: leap_ec.ops
    :members:
    :undoc-members:
    :show-inheritance:

Pipeline operators for binary representations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: leap_ec.binary_rep.ops
    :members:
    :undoc-members:
    :show-inheritance:

Pipeline operators for real-valued representations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: leap_ec.real_rep.ops
    :members:
    :undoc-members:
    :show-inheritance: