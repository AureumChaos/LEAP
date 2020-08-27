LEAP Concepts
===========================

.. figure:: _static/top-level-class-diagram.svg

        **Figure 1: The core classes**  `Individual`, `Problem`, and
        `Representation` are the three classes upon which the rest of the
        toolkit rests.

Three classes work in tandem to represent and evaluate solutions: `Individual`,
`Problem`, and `Decoder`.  The relationship between these classes is depicted
in Figure 1, and shows that the `Individual` is the design's keystone, and encapsulates
posed solution to a `Problem`.  `Problem` implements the semantics for a given
problem to be solved, and which `Individual` uses to compute its fitness.
`Problem` also implements how any two given `Individuals` are "better than" or
"equivalent" to one another.

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
