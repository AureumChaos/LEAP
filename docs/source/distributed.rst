Distributed LEAP
================
LEAP supports synchronous and asynchronous distributed concurrent fitness evaluations that
can significantly speed-up runs.  LEAP uses dask_, which
is a popular distributed processing python package, to implement
parallel fitness evaluations, and which allows easy scaling from laptops to supercomputers.

.. _dask https://dask.org/

Synchronous fitness evaluations
-------------------------------
Synchronous fitness evaluations are essentially a map/reduce approach where individuals
are fanned out to computing resources to be concurrently evaluated, and then
the calling process waits until all the evaluations are done.  This is particularly
suited for by-generation approaches where offspring are evaluated in a
batch, and progress in the EA only proceeds when all individuals have been evaluated.

Examples
^^^^^^^^

There is a jupyter notebook that walks through a synchronous implementation in
`examples/simple_sync_distributed.ipynb`.

Asynchronous fitness evaluations
--------------------------------
Asynchronous fitness evaluations are a little more involved in that the EA immediately integrates
newly evaluated individuals into the population -- it doesn't wait until all
the individuals have finished evaluating before proceeding.  More specifically,
LEAP implements an asynchronous steady-state evolutionary algorithm (ASEA).

.. figure:: _static/asea.png

    Algorithm 1: Asynchronous steady-state evolutionary algorithm concurrently
    updates a population as individuals are evaluated.

Algorithm 1 shows the details of how an ASEA works.  Newly evaluated individuals
are inserted into the population, which now leaves a computing resource free.
Offspring are created from one or more selected parents, and are then assigned
to that computing resource, thus assuring minimal idle time between evaluations.
This is particularly important within HPC contexts, as it is often the case that
such resources are costly, and therefore there is an implicit need to minimize
wasting such resources.  By contrast, a synchronous distributed approach risks
idling computing resources

Examples
^^^^^^^^
There is also a jupyter notebook walkthrough for the asynchronous implementation,
`examples/simple_async_distributed.ipynb`.  Moreover, there is standalone
code in `examples/simple_distributed.py`.