.. _building:

Implementing Tailored Evolutionary Algorithms with LEAP
=======================================================

The :ref:`prebuilt` LEAP functions, `ea_solve()`, `generational_ea()`,
and `asynchronous.steady_state()` may not be sufficient to address your problem.
This could be that you want to access the state of objects outside the
pipeline during a run, or that you want to add complex bookkeeping
not easily supported by a prebuilt, among many other possible reasons.

This leaves assembling a bespoke evolutionary algorithm using low-level LEAP
components.  Generally, you will need to do the following:

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

Coming up with a suitable representation
----------------------------------------

The first design decision you will have to make is how to best represent your
problem.  There are two broad categories of representations, genotypic and
phenotypic.  A genotypic representation is a form of indirect representation
whereby problem values use data that must be decoded into values that make
sense to the associated `Problem` class you will also define.  One popular
example is using a binary encoding that must be decoded into values, usually
integers or real-value sequences, that can then be used by a `Problem`
instance to evaluate an individual.  (However, there are some problems that
directly use the binary sequences without having to interpret the values,
such as the `MaxOnes` problem.)

If you use a binary representation, then you will almost certainly need to
define an associated decoder to convert binary sequences within `Individual`
genomes into values that the associated `Problem` can use.  A variety of
premade binary decoders can be found in :py:mod:`leap_ec.binary_rep.decoders`,
and these can be used to convert binary sequences to integers or real values.
Gray code versions of binary decoders are also included.

.. pull-quote::
    **Gray encoding** Gray encoding is an alternative integer representation
    for binary sequences that represent integer values. Gray encoding resolves
    the issue where bit flip mutation of higher order bits would greatly change
    the values, whereas a Grey encoded binary integer will only change the value
    a small amount regardless of which bit was flipped in the binary sequence.
    `(See also: Grey code) <https://en.wikipedia.org/wiki/Gray_code>`_

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
