LEAP Concepts
===========================

This section summarizes the main classes and the operator pipeline that use
them.

Core Classes
------------

.. _class-diagram:
.. figure:: _static/top-level-class-diagram.svg
    :align: center

    **The core classes**  `Individual`, `Problem`, and
    `Decoder` are the three classes upon which the rest of the
    toolkit rests.

Three classes work in tandem to represent and evaluate solutions: `Individual`,
`Problem`, and `Decoder`.  The relationship between these classes is depicted
in :numref:`class-diagram`, and shows that the `Individual` is the design's keystone, and encapsulates
posed solutions to a `Problem`.  `Problem` implements the semantics for a given
problem to be solved, and which `Individual` uses to compute its fitness.
`Problem` also implements how any two given `Individuals` are "better than" or
"equivalent" to one another.  The `Decoder` translates an `Individuals` genome
into a phenome, or values meaningful to the associated `Problem` for fitness
evaluation; for example, a `Decoder` may translate a bit sequence into a vector
of real-values that are then passed to the `Problem` as parameters during
evaluation.

.. _operator-pipeline:

Operator Pipeline
-----------------

If the above classes are the "nouns" of LEAP, the pipeline operators are the
"verbs" that work on those "nouns."  The overarching concept of the pipeline is
similar to \*nix style text processing command lines, where a sequence
of operators pipe output of one text processing utility into the next one with
the last one returning the final results.  For example::

    > cut -d, -f 4,5,8 results.csv | head -4 | column -t -s,
    birth_id  scenario  fitness
    2         2         -23.2
    1         14        6.0
    0         36        31.0

This shows the output of `cut` is passed to `head` and the output of that is
passed to the formatter `column`, which then sends its output to stdout.

Here is an example of a LEAP pipeline:

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
#. perform mutation bitflip on the clones
#. evaluate the offspring
#. accumulate as many offspring as there are parents

Essentially the `ops.` functions are python co-routines that are driven by the
last function, `ops.pool()` , that makes requests of the upstream operators to
fill a pool of offspring.  Once the pool is filled, it is returned as the next
set of offspring, which are then assigned to become the parents for the next
generation.  (`mutate_bitflip` is in `ops` but the one for binary
representations; i.e., `binary_rep/ops.py`.  And, since `ops` is already used,
we just directly import `mutate_bitflip`, which is why it does not have the
`ops` qualifier.)

.. _pipeline-figure:
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

:numref:`pipeline-figure` depicts a general pattern of LEAP pipeline operators. Typically, the
first pipeline element is a source for individuals followed by some form of
selection operator and then a clone operator to create an offspring that is
initially just a copy of the selected parent.  Following that there are one
or more pertubation operators, and though there is only a mutation operator
shown in the figure, there can be other configurations that also include
crossover, among other pertubation operators.  Next, there is an operator
to evaluate offspring as they come through pipeline where they are collected
by a pooling operator.  And, lastly, there can be a survival selection operator to
determine survivors for the next generation, such as truncation selection. (The
above code snippet does not have survival selection because it replaces the
parents with the offspring for every generation.)

Detailed Explanations
---------------------

More detailed explanations of the concepts shared here are given in the following
sections.

.. toctree::

    Individuals <individuals>
    Decoders <decoders>
    Representations <representation>
    Problems <problem>
    Operators <ops>
    Contexts <context>
    Probes <probes>
    Parsimony <parsimony>
    Visualizations <visualization>
