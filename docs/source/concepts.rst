LEAP Concepts
===========================

.. inheritance-diagram:: leap_ec.core.Individual leap_ec.problem.Problem leap_ec.core.Representation
    :parts: 2
    :caption: **Figure 1: The core classes**  `Individual`, `Problem`, and
        `Representation` are the three core classes upon which the rest of the
        toolkit rests.

.. figure:: _static/Pipeline.png

        **Figure 1: LEAP operator pipeline.** This figure depicts a typical
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


.. toctree::

    Individuals <individuals>
    Representations <representation>
    Problems <problem>
    Operators <ops>
    Context <context>
    Probes <probes>
    Visualization <visualization>
