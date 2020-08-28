LEAP Concepts
===========================

This section summarizes the main classes and the operator pipeline that use
them.

Core Classes
------------

.. figure:: _static/top-level-class-diagram.svg
    :align: center

    **Figure 1: The core classes**  `Individual`, `Problem`, and
    `Representation` are the three classes upon which the rest of the
    toolkit rests.

Three classes work in tandem to represent and evaluate solutions: `Individual`,
`Problem`, and `Decoder`.  The relationship between these classes is depicted
in Figure 1, and shows that the `Individual` is the design's keystone, and encapsulates
posed solutions to a `Problem`.  `Problem` implements the semantics for a given
problem to be solved, and which `Individual` uses to compute its fitness.
`Problem` also implements how any two given `Individuals` are "better than" or
"equivalent" to one another.  The `Decoder` translates an `Individuals` genome
into a phenome, or values meaningful to the associated `Problem` for fitness
evaluation; for example, a `Decoder` may translate a bit sequence into a vector
of real-values that are then passed to the `Problem` as parameters during
evaluation.

Operator Pipeline
-----------------

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

If the above classes are the "nouns" of LEAP, the pipeline operators are the
"verbs" that work on those "nouns."  The overarching concept of the pipeline is
similar to *nix style text processing command lines, where a sequence
of operators pipe output of one text processing utility into the next one with
the last one returning the final results.  For example::

    > cut -d, -f 4,5,8 results.csv | head -4 | column -t -s,
    birth_id  scenario  fitness
    2         2         -23.2
    1         14        6.0
    0         36        31.0

This shows the output of `cut` is passed to `head` and the out of that is
passed to the formatter `column`, which then sends its output to stdout.


Detailed Explanations
---------------------

More detailed explanations of the concepts shared here are given in the following
sections.

.. toctree::

    Individuals <individuals>
    Representations <representation>
    Problems <problem>
    Operators <ops>
    Context <context>
    Probes <probes>
    Visualization <visualization>
