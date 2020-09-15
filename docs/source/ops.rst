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
introduced.  Fig.2 is shown again here to depict a typical set of LEAP pipeline
operators.

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