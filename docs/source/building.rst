.. _building:

Building New Evolutionary Algorithms with LEAP
==============================================

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
