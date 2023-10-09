Pipeline Operators
==================

.. _pipeline:
.. figure:: _static/Pipeline.png

    **LEAP operator pipeline.** This figure depicts a typical
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


Overview
--------

:py:class:`leap_ec.individual.Individual`, :py:class:`leap_ec.problem.Problem`,
and :py:class:`leap_ec.decoder.Decoder` are passive classes that need an
external framework to make them function.  In :doc:`concepts` the notion of a
pipeline of evolutionary algorithm (EA) operators that use these classes was
introduced.  That is, `Individual`, `Decoder`, and `Problem` are the "nouns"
and the pipeline operators a the verbs that operate on those nouns.  The
operator pipeline objective is to create a new set of evaluated individuals
from an existing set of prospective parents that can be in a new set of
prospective parents.

:numref:`pipeline` is shown again here to depict a typical set of LEAP pipeline
operators.  The pipeline generally starts with a "source", or a parent population,
from which the next operator typically selects for creating offspring. This is
followed by a clone operator that ensure the subsequent pertubation operators
do not modify the selected parents.  (And so it is critically important that
users *always* have a clone operator as a part of the offspring creation
pipeline before any mutation, crossover, or other genome altering operators.)
The pertubation operators can be mutation or also include a crossover
operator. At this point in the pipeline we have a completed offspring with no
fitness, so the next operator evaluates the offspring to assign that fitness.
Then the evaluated offspring is collected into a pool of offspring that acts as a
"sink" for new individuals, and is the principal driving for the pipeline; i.e.,
it is the need to fill the sink that "pulls" individuals down the pipeline.  Once the
offspring pool reaches a desired size it returns all the offspring to another
selection operator to cull the offspring, and optionally the parents, to
return the next set of prospective parents.

Or, more explicitly:

#. Start with a collection of `Individuals` that are prospective parents as the pipeline "source"
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

The first "data" argument is a collection of `Individuals` representing
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
                               ops.tournament_selection,
                               ops.clone,
                                   mutate_bitflip,
                               ops.evaluate,
                               ops.pool(size=len(parents)))

        parents = offspring
        gen += 1

The above code snippet is an example of a very basic genetic algorithm
implementation that uses a `toolz.pipe()` function to link
together a series of operators to do the following:

#. binary tournament_selection selection on a set of parents
#. clone those that were selected
#. perform mutation bit-bitflip on the clones
#. evaluate the offspring
#. accumulate as many offspring as there are parents

Since we only have mutation in the pipeline, only one parent at a time is selected
to be cloned to create an offspring.  However, let's make one change to that pipeline
by adding crossover:

.. code-block:: python

    gen = 0
    while gen < max_generation:
        offspring = toolz.pipe(parents,
                               ops.tournament_selection,
                               ops.clone,
                                   mutate_bitflip,
                               ops.uniform_crossover, # NEW OPERATOR
                               ops.evaluate,
                               ops.pool(size=len(parents)))

        parents = offspring
        gen += 1

This does the following:

#. binary tournament_selection selection on a set of parents
#. clone those that were selected
#. perform mutation bitflip on the clones
#. perform uniform crossover between the two offspring
#. evaluate the offspring
#. accumulate as many offspring as there are parents

Adding crossover means that now **two** parents are selected instead of one. However,
note that the tournament_selection selection operator wasn't changed.  It automatically
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
it just has a contract that when `next()` is invoked that it will get another
`Individual`.  And, since this is a generator function, it `yields` the
crossed-over `Individuals`.  It also has *two* `yield` statements that
ensures both crossed-over `Individuals` are returned, thus eliminating a potential
source of genetic drift by arbitrarily only yielding one and discarding the other.

Operators for collections of `Individuals`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There is another class of operators that work on collections of `Individuals`
such as selection and pooling operators.  Generally:

*selection* pipeline operators
    accept a collection of `Individuals` and yield a selected `Individual` (and thus are generator functions)

*pooling* operators
    accept an `Iterator` from which to get the `next()` `Individual`, and returns a collection of `Individuals`

Below shows an example of a selection operator, which is a simplified version of
the `tournament_selection()` operator:

.. code-block:: python

    def tournament_selection(population: List, k: int = 2) -> Iterator:
        while True:
            choices = random.choices(population, k=k)
            best = max(choices)

            yield best

(Again, the actual :py:func:`leap_ec.ops.tournament_selection` has checks and docstrings.)

This depicts how a typical selection pipeline operator works.  It accepts a
population parameter (plus some optional parameters), and yields the selected
individual.

Below is example of a pooling operator:

.. code-block:: python

    def pool(next_individual: Iterator, size: int) -> List:
        return [next(next_individual) for _ in range(size)]

This accepts an `Iterator` from which it gets the next individual, and it
uses that iterator to accumulate a specified number of `Individuals` via a
list comprehension.  Once the desired number of `Individuals` is accumulated,
the list of those `Individuals` is returned.

Currying Function Decorators
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Some pipeline operators have user-specified parameters.  E.g., :py:func:`leap_ec.ops.pool` has
the mandatory `size` parameter.  However, given that `toolz.pipe()` takes
functions as parameters, how do we ensure that we pass in functions that have
set parameters?

Normally we would use the Standard Python Library's `functools.partial` to
set the function parameters and then pass in the function returned from that call.  However, `toolz`
has a convenient function wrapper that does the same thing, `toolz.functools.curry`.
(See: https://toolz.readthedocs.io/en/latest/api.html#toolz.functoolz.curry )
Pipeline operators that take on user-settable parameters are all wrapped with
`curry` to allow functions with parameters set to be passed into `toolz.pipe()`.


Operator Class
^^^^^^^^^^^^^^

Most of the pipeline operators are implemented as functions.  However, from
time to time an operator will need to persist state between invocations.  For
generator functions, that comes with using `yield` in that the next time that
function is invoked the next individual is returned.  However, there are some
operators that use closures, such as `:py:func:leap_ec.ops.migrate`.

In any case, sometimes if one wants persistent state in a pipeline operator a
closure or using ``yield`` isn't enough.  In which case, having a *class* that
can have objects that persist state might be useful.

To that end, :py:class:`leap_ec.ops.Operator` is an abstract base-class (ABC)
that provides a template of sorts for those kinds of classes.  That is, you would
write an `Operator` sub-class that provides a `__call__()` member function
that would allow objects of that class to be inserted into a LEAP pipeline
just like any other operator.  Presumably during execution the internal object
state would be continually be updated with book-keeping information as
`Individuals` flow through it in the pipeline.

:py:class:`leap_ec.ops.CooperativeEvaluate` is an example of using this class.

Table of Pipeline Operators
^^^^^^^^^^^^^^^^^^^^^^^^^^^
+------------------------------------+--------------------------+---------------------------+
| Representation Specificity         | Input -> Output          | Operator                  |
+====================================+==========================+===========================+
| Representation                     |  Iterator → Iterator     | clone()                   |
|                                    |                          +---------------------------+
| Agnostic                           |                          | evaluate()                |
|                                    |                          +---------------------------+
|                                    |                          | uniform_crossover()       |
|                                    |                          +---------------------------+
|                                    |                          | n_ary_crossover()         |
|                                    |                          +---------------------------+
|                                    |                          | CooperativeEvaluate       |
|                                    +--------------------------+---------------------------+
|                                    |  Iterator → population   | pool()                    |
|                                    +--------------------------+---------------------------+
|                                    |  population → population | truncation_selection()    |
|                                    |                          +---------------------------+
|                                    |                          | const_evaluate()          |
|                                    |                          +---------------------------+
|                                    |                          | insertion_selection()     |
|                                    |                          +---------------------------+
|                                    |                          | migrate()                 |
|                                    +--------------------------+---------------------------+
|                                    |  population → Iterator   | tournament_selection()    |
|                                    |                          +---------------------------+
|                                    |                          | naive_cyclic_selection()  |
|                                    |                          +---------------------------+
|                                    |                          | cyclic_selection()        |
|                                    |                          +---------------------------+
|                                    |                          | random_selection()        |
+-------------------+----------------+--------------------------+---------------------------+
| Representation    | binary_rep     |  Iterator → Iterator     | mutate_bitflip()          |
| Dependent         +----------------+--------------------------+---------------------------+
|                   | real_rep       |  Iterator → Iterator     | mutate_gaussian()         |
| Dependent         +----------------+--------------------------+---------------------------+
|                   | int_rep        |  Iterator → Iterator     | mutate_randint()          |
|                   +----------------+--------------------------+---------------------------+
|                   | segmented_rep  |  Iterator → Iterator     | apply_mutation()          |
|                   +                +                          +---------------------------+
|                   |                |                          | add_segment()             |
|                   +                +                          +---------------------------+
|                   |                |                          | remove_segment()          |
|                   +                +                          +---------------------------+
|                   |                |                          | copy_segment()            |
+-------------------+----------------+--------------------------+---------------------------+

Admittedly it can be confusing when considering the full suite of LEAP pipeline operators,
especially in remembering what kind of operators "connect" to what.  With that in mind,
the above table breaks down pipeline operators into different categories.  First,
there are two broad categories of pipeline operators --- operators that don't care
about the internal representation of `Individuals`, or "Representation Agnostic" operators;
and those operators that do depend on the internal representation, or "Representation
Dependent" operators.  Most of the operators are "Representation Agnostic" in that it doesn't matter
if a given `Individual` has a genome of bits, real-values, or some other
representation.  Only two operators are dependent on representation, and those
will be discussed later.

The next category is broken down by what kind of input and output a given
operator takes.  That is, generally, an operator takes a population (collection
of `Individuals`) or an `Iterator` from which a next `Individual` can be found.
Likewise, a given operator can return a population or yield an `Iterator` to
a next `Individual`.  So, operators that return an `Iterator` can be connected
to operators that expect an `Iterator` for input.  Similarly, an operator that
expects a population can be connected directly to a collection of `Individuals`
(e.g., be the second argument to ``toolz.pipe()``) or to an operator that
returns a collection of `Individuals`.

If you are familiar with evolutionary algorithms, most of these connections are
just common sense.  For example, selection operators would select from a
population.

With regards to "Representation Dependent" operators there currently are only
two: :py:func:`leap_ec.binary_rep.mutate_bitflip()` and
:py:func:`leap_ec.real_rep.mutate_gaussian`.  The former relies on a genome of
all bits, and the latter of real-values.  In the future, LEAP will support other
representations that will similarly have their own operators.

.. warning:: **Are all operators really representation agnostic?**
    In reality, most of the operators assume that `Individual.genome` is a
    `numpy` array, which may not always be the case.  For example, the user
    may come up with a representation that employs, say, a sparse matrix.  In
    that case, the crossover operators will fail.

    In the future we intend on adding support for other popular representations
    that will show up as LEAP sub-packages. (I.e., just as `binary_rep` and
    `real_rep` provide support for binary and real-value representations.)

    So, in a sense, for where it matters, LEAP currently assumes some sort of
    sequence for genomes though, again, plans are afoot to add more representation
    types.  In the interim, you will have to add your own operators to support
    new non-sequence genomic representations.

Type-checking Decorator Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

However, to help minimize the chances that pipeline operators would be mis-used
the operators have function decorates that due parameter type-checking to
ensure the correct parameters are being passed in.  These are:

`iteriter_op`
    This checks for signatures of type `Iterator` -> `Iterator`

`listlist_op`
    Checks for population -> population type operators

`listiter_op`
    Checks for population -> population type operators

`iterlist_op`
    Checks for population -> `Iterator` type operators

These can be found in `leap_ec.ops`.

API Documentation
-----------------

Base operator classes and representation agnostic functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: leap_ec.ops
    :members:
    :undoc-members:
    :show-inheritance:
    :noindex:

Pipeline operators for binary representations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: leap_ec.binary_rep.ops
    :members:
    :undoc-members:
    :show-inheritance:
    :noindex:

Pipeline operators for real-valued representations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: leap_ec.real_rep.ops
    :members:
    :undoc-members:
    :show-inheritance:
    :noindex:

Pipeline operators for segmented representations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: leap_ec.segmented_rep.ops
    :members:
    :undoc-members:
    :show-inheritance:
    :noindex:
