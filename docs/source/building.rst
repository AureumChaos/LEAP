.. _building:

Implementing Tailored Evolutionary Algorithms with LEAP
=======================================================

The :ref:`prebuilt` , `ea_solve()`, `generational_ea()`,
and `asynchronous.steady_state()` may not be sufficient to address your problem.
This could be that you want to access the state of objects outside the
pipeline during a run, or that you want to add complex bookkeeping
not easily supported by a prebuilt, among many other possible reasons.

This leaves assembling a bespoke evolutionary algorithm (EA) using low-level LEAP
components.  Generally, to do that you will need to do the following:

* Come up with a suitable representation for your problem

  * What is the genome going to look like?  Is it an indirect representation
    like a binary representation that must be decoded?  Or a phenotypic, or
    direct representation, such as a real-valued vector?  Or something else?
  * How are genomes going to be decoded into a phenotypic representation
    suitable for the associated Problem class?
  * Is the default `Individual` class suitable?  Or will one of its subclasses
    be more appropriate?  Will you have to write your own to keep additional
    state?

* Define a `Problem` sub-class
* Implement a loop wrapped around a pipeline of appropriate pipeline operators
* Determine what output to generate
* Optionally visualizing a run

These will be described in more detail in the following sub-sections.

Deciding on a suitable representation
-------------------------------------

The first design decision you will have to make is how to best represent your
problem.  There are two broad categories of representations, *genotypic* and
*phenotypic*.  A *genotypic* representation is a form of indirect representation
whereby problem values use data that must be decoded into values that make
sense to the associated `Problem` class you will also define.  One popular
example is using a binary encoding that must be decoded into values, usually
integers or real-value sequences, that can then be used by a `Problem`
instance to evaluate an individual.  (However, there are some problems that
directly use the binary sequences without having to interpret the values,
such as the `MaxOnes` problem.)  A *phenotypic* representation is able to
directly represent problem relevant values in some way, usually as a vector
of real-values that correspond to problem parameters.

Decoders for binary representations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you use a binary representation, then you will almost certainly need to
define an associated decoder to convert binary sequences within `Individual`
genomes into values that the associated `Problem` can use.  A variety of
premade binary decoders can be found in :py:mod:`leap_ec.binary_rep.decoders`,
and these can be used to convert binary sequences to integers or real values.
Gray code versions of binary decoders are also included.

.. note::
    **Gray encoding** Gray encoding is an alternative integer representation
    that use binary sequences. Gray encoding resolves
    the issue where bit flip mutation of higher order bits would greatly change
    the values, whereas a Grey encoded binary integer will only change the value
    a small amount regardless of which bit was flipped in the binary sequence.
    `(See also: Grey code) <https://en.wikipedia.org/wiki/Gray_code>`_

Impact on representation on choice of pipeline operators
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There will be two areas where you representation choice is going to have an
impact on the code you write.  First is in how you initialize individuals with
random genomes.  The second will be mutation and possibly crossover pipeline
operators tailored to that representation.  The mutation and crossover pipeline
operators are generally going to be specific to the underlying representation.
For example, bit flip mutation is relevant to binary representations, and a
Gaussian mutation is appropriate for real-value representations.  There are
sub-packages for integer, real, and binary representations that have an
`ops.py` that will contain pertubation (mutation) operators appropriate for
the associated representation.

LEAP supports three numeric representations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There are three numeric representations supported by LEAP, that for binary,
integer, and real values.  You can find initializers that create random values
for those value types in their respective sub-packages.  You can find them
in :py:mod:`leap_ec.binary_rep.initializers`, :py:mod:`leap_ec.int_rep.initializers`,
and :py:mod:`leap_ec.real_rep.initializers`, respectively.

Support for exotic representations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

LEAP is flexible enough to support other, more exotic representations, such as
graphs and matrices.  However, you will have to write your own initializers
and mutation (and possibly crossover) operators to support such novel
genome types.

Defining a `Problem` subclass
-----------------------------
The :ref:`problem` are where you implement how to evaluate an individual
to solve your problem.  You will need to create a `Problem` sub-class and
implement its `evaluate()` member function accordingly.

`Problem` is an abstract base class (ABC), so you *must* subclass from it.
Moreover, there are a number of `Problem` subclasses, so you will need to pick
one that is the best fit for your situation.  More than likely you will
subclass from `ScalarProblem` since it supports real-valued fitnesses and it
handles fitnesses that are NaNs, which can happen if you use `RobustIndividual`
or `DistributedIndividual` and an exception is thrown during evaluation. (I.e.,
it was impossible to assign a fitness because the evaluation failed, so we
signal that by assigning NaN as the fitness.)

If you use `ScalarProblem` or one of its subclasses, you will also have to
specify whether it is a maximization problem via the boolean parameter passed
into the class constructor.

There are a number of example `Problem` implementations that can be found in
`real_rep.problems` many of which are popular benchmarks.

Possibly defining or choosing a special `Individual` subclass
-------------------------------------------------------------

:ref:`individuals` encapsulate a posed solution to a problem and an associated
fitness after evaluation.  For most situations the default `Individual`
class should be fine.  However, you can also use :ref:`robust-individual` if
you want individuals to handle exceptions that may be thrown during evaluation.
If you are using the synchronous or asynchronous distributed evaluation
support, then you may consider using :ref:`distributed-individual`, which itself
is a subclass of `RobustIndividual`, but also assigns a UUID to each individual,
a unique birth ID integer, and start and stop evaluation times in UNIX epoch
time.

Of course, if none of those `Individual` classes meet your needs, you can freely
create your own `Individual` subclass.  For example, you may want a subclass
that performs additional bookkeeping, such as perhaps maintaining links to
its parents and any clones (offspring).


Putting all that together
-------------------------

Now that you have chosen a representation, an associated `Decoder`, a `Problem`,
and an `Individual` class, you are now ready to assemble those components into
a functional evolutionary algorithm.  Generally, your code will follow this
pattern::

    parents ← create_initial_random_population()

    While not done:

        offspring ← toolz.pipe(parents, *pipeline_ops)
        parents ← offspring

That is, first a population of parents are randomly created, and then we fall
into a loop where we create offspring from those parents by generation until we
are done with some sort of arbitrary stopping criteria.  Within the loop the old
parents are replaced with the offspring.  There is, of course, a lot more nuance to
that with actual evolutionary algorithms, but that captures the essence of
EAs.

The part where the offspring are created merits more discussion.  We rely on
`toolz.pipe()` to take a source of individuals, the current parents, from
which to generate a set of offspring.  Individuals are selected by demand
from the given sequent of pipeline operators, where each of these operators will
manipulate the individuals that pass through them in some way.  This concept is
described in more detail in :ref:`operator-pipeline`.

Evolutionary algorithm examples
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There are a number of examples to steer by found in `examples/simple`.  In
particular:

* `simple_ep.py` -- simple example of an Evolutionary Program
* `simple_es.py` -- simple example of an Evolutionary Strategy
* `simple_ga.py` -- simple example of a Genetic Algorithm
* `simple_ev.py` -- simple example of an Evolutionary Algorithm as defined in
  Ken De Jong's *Evolutionary Computation: A Unified Approach*.
