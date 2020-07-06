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

Example
^^^^^^^

.. code-block:: Python
    :linenos:

    from pprint import pformat

    from dask.distributed import Client, LocalCluster

    from leap_ec import core
    from leap_ec import ops
    from leap_ec import binary_problems
    from leap_ec.distributed import asynchronous
    from leap_ec.distributed.probe import log_worker_location, log_pop
    from leap_ec.distributed.individual import DistributedIndividual

    MAX_BIRTHS = 500
    INIT_POP_SIZE = 20
    POP_SIZE = 20
    GENOME_LENGTH = 5

    with Client(scheduler_file='scheduler.json') as client:
        final_pop = asynchronous.steady_state(client, # dask client
                                      births=MAX_BIRTHS,
                                      init_pop_size=INIT_POP_SIZE,
                                      pop_size=POP_SIZE,

                                      representation=core.Representation(
                                          decoder=core.IdentityDecoder(),
                                          initialize=core.create_binary_sequence(
                                              GENOME_LENGTH),
                                          individual_cls=DistributedIndividual),

                                      problem=binary_problems.MaxOnes(),

                                      offspring_pipeline=[
                                          ops.random_selection,
                                          ops.clone,
                                          ops.mutate_bitflip,
                                          ops.pool(size=1)],

                                      evaluated_probe=track_workers_func,
                                      pop_probe=track_pop_func)

    print(f'Final pop: \n{pformat(final_pop)}')

The above example is quite different from the synchronous code given earlier.  Unlinke,
with the synchronous code, the asynchronous code does provide a monolithic function
entry point, `asynchronous.steady_state()`.  The first thing to note is that
by nature this EA has a birth budget, not a generation budget, and which is set
to 500 in `MAX_BIRTHS`, and passed in via the `births` parameter.  We also need
to know the size of the initial population, which is given in `init_pop_size`.
And, of course, we need the size of the population that is perpetually updated
during the lifetime of the run, and which is passed in via the `pop_size`
parameter.

The `representation` parameter we have seen before in the other monolithic
functions, such as `generational_ea`, which encapsulates the mechanisms for
making an individual and how the individual's state is stored.  In this case,
because it's the MAX ONES problem, we use the `IdentityDecoder` because we want
to use the raw bits as is, and we specify a factory function for creating
binary sequences GENOME_LENGTH in size; and, lastly, we override the default
class with a new class, `DistributedIndividual`, that contains some additional
bookkeeping useful for an ASEA, and is described later.

Also noteworthy is that the `Client` has a `scheduler_file` specified, which
indicates that a dask scheduler and one or more dask workers have already been
started and are awaiting tasking to evaluate individuals.


DistributedIndividual
^^^^^^^^^^^^^^^^^^^^^
`DistributedIndividual` is a subclass of `Individual` that contains some additional
state that may be useful for distributed fitness evaluations.

:uuid: is UUID assigned to that individual upon creation
:birth_id: is a unique, monotonically increasing integer assigned to each
    indidividual on creation, and denotes its birth order
:start_eval_time: is when evaluation began for this individul; it's set in
    `distributed.evaluate.evaluate()` and is in `time_t` format.
:stop_eval_time: when evaluation completed in `time_t` format.

`is_viable` and `exception` are set as with the base class, `core.Individual`.

.. note:: The `uuid` is useful if one wanted to save, say, a model or some other
    state in a file; using the `uuid` in the file name will make it easier to associate
    the file with a given individual later during a run's post mortem analysis.

.. note:: The `start_eval_time` and `end_eval_time` can be useful for checking
    whether individuals that take less time to evaluate come to dominate the
    population, which can be important in ASEA parameter tuning.  E.g., initially
    the population will come to be dominated by individuals that evaluated quickly
    even if they represent inferior solutions; however, eventually, better solutions
    that take longer to evaluate will come to dominate the population; so, if
    one observes that shorter solutions still dominate the population, then
    increasing the `max_births` should be considered, if feasible, to allow time
    for solutions that need longer to evaluate time to make a representative
    presence in the population.


Separate Examples
^^^^^^^^^^^^^^^^^
There is also a jupyter notebook walkthrough for the asynchronous implementation,
`examples/simple_async_distributed.ipynb`.  Moreover, there is standalone
code in `examples/simple_async_distributed.py`.