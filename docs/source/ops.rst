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
snippets of an offspring creation pipeline.  E.g., :py:func:`leap_ec.distributed.asynchronous.steady_state`
has an `offspring_pipeline` parameter that doesn't have parents explicitly
as part of the pipeline; instead, it's *implied* that the parents will be provided during the
run.

Implementation Details
----------------------

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