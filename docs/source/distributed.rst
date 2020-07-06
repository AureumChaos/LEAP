Distributed LEAP
================
LEAP supports synchronous and asynchronous distributed concurrent fitness evaluations that
can significantly speed-up runs.  LEAP uses dask (`https://dask.org/`), which
is a popular distributed processing python package, to implement
parallel fitness evaluations, and which allows easy scaling from laptops to supercomputers.

Synchronous fitness evaluations
-------------------------------
Synchronous fitness evaluations are essentially a map/reduce approach where individuals
are fanned out to computing resources to be concurrently evaluated, and then
the calling process waits until all the evaluations are done.  This is particularly
suited for by-generation approaches where offspring are evaluated in a
batch, and progress in the EA only proceeds when all individuals have been evaluated.

Components
^^^^^^^^^^
`leap_ec.distributed.synchronous` provides two components to implement synchronous
individual parallel evaluations.

* `leap_ec.distributed.synchronous.eval_population`
    which evaluates an entire population in parallel, and returns the evaluated population
* `leap_ec.distributed.synchronous.eval_pool`
    is a pipeline operator that
    will collect offspring and then evaluate them all at once in parallel; the
    evaluated offspring are returned

Example
^^^^^^^
The following shows a simple example of how to use the synchronous parallel
fitness evaluation in LEAP.

.. literalinclude:: ../../examples/simple_sync_distributed.py
    :linenos:
    :language: python
    :lines: 5-42

This example of a basic genetic algorithm that solves the MAX ONES problem
does not use a provided monolithic entry point, such as found with
`ea_solve()` or `generational_ea()` but, instead, directly uses LEAP's pipeline
architecture.  Here, we create a simple `dask` `Client` that uses the default
local cores to do the parallel evaluations.  The first step is to create the
initial random population, and then distribute those to dask workers for evaluation
via `synchronous.eval_population()`, and which returns a set of fully evaluated
parents.  The `for` loop supports the number of generations we want, and provides
a sequence of pipeline operators to create offspring from selected parents.  For
concurrently evaluating newly created offspring, we use `synchronous.eval_pool`,
which is just a variant of the `leap_ec.ops.pool` operator that relies on `dask`
to evaluate individuals in parallel.

.. note:: If you wanted to use resources on a cluster or supercomputer, you would
    start up `dask-scheduler` and `dask-worker`s first, and then point the `Client`
    at the scheduler file used by the scheduler and workers.  Distributed LEAP is agnostic on what kind of dask
    client is passed as a `client` parameter -- it will generically perform the same
    whether running on local cores or on a supercomputer.

Separate Examples
^^^^^^^^^^^^^^^^^

There is a jupyter notebook that walks through a synchronous implementation in
`examples/simple_sync_distributed.ipynb`.  The above example can also be found
at `examples/simple_sync_distributed.py`.

Asynchronous fitness evaluations
--------------------------------
Asynchronous fitness evaluations are a little more involved in that the EA immediately integrates
newly evaluated individuals into the population -- it doesn't wait until all
the individuals have finished evaluating before proceeding.  More specifically,
LEAP implements an asynchronous steady-state evolutionary algorithm (ASEA).

.. figure:: _static/asea.png

    Algorithm 1: Asynchronous steady-state evolutionary algorithm concurrently
    updates a population as individuals are evaluated. (Mark Coletti, Eric Scott,
    Jeffrey K. Bassett. **Library for Evolutionary Algorithms in Python (LEAP)**.
    Genetic and Evolutionary Computation Conference, 2020. Cancun, MX. To be
    printed.)

Algorithm 1 shows the details of how an ASEA works.  Newly evaluated individuals
are inserted into the population, which then leaves a computing resource available.
Offspring are created from one or more selected parents, and are then assigned
to that computing resource, thus assuring minimal idle time between evaluations.
This is particularly important within HPC contexts as it is often the case that
such resources are costly, and therefore there is an implicit need to minimize
wasting such resources.  By contrast, a synchronous distributed approach risks
wasting computing resources because computing resources that finish evaluating
individuals before the last individual is evaluated will idle until the next
generation.

Separate Examples
^^^^^^^^^^^^^^^^^
There is also a jupyter notebook walkthrough for the asynchronous implementation,
`examples/simple_async_distributed.ipynb`.  Moreover, there is standalone
code in `examples/simple_async_distributed.py`.