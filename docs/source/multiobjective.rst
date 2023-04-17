Multiobjective Optimization
===========================

LEAP supports multi-objective optimization via an implementation of [NSGA-II]_.
There are two ways of using this functionality -- using a single function,
`leap_ec.mulitobjective.nsga2.generalized_nsga_2`, or by assembling a bespoke NSGA-II using pipeline
operators.  We will cover both approaches here.

Using `generalized_nsga_2`
--------------------------

`leap_ec.mulitobjective.nsga2.generalized_nsga_2` is similar to other LEAP
metaheuristic functions, such as `generational_ea`.  It has arguments for specifying
the maximum number of generations, population size, stopping criteria, problem
representation, and others.

Note that by default a faster rank sorting algorithm is used [Burlacu]_, but if it is
important to use the original NSGA-II rank sorting algorithm, then that can be provided
by specifying `leap_ec.mulitobjective.ops.fast_nondominated_sort` for the `rank_func`
argument.

Example
^^^^^^^

.. code-block:: Python
    :linenos:

    from leap_ec.representation import Representation
    from leap_ec.ops import random_selection, clone, evaluate, pool
    from leap_ec.real_rep.initializers import create_real_vector
    from leap_ec.real_rep.ops import mutate_gaussian
    from leap_ec.multiobjective.nsga2 import generalized_nsga_2
    from leap_ec.multiobjective.problems import SCHProblem
    pop_size = 10
    max_generations = 5
    final_pop = generalized_nsga_2(
        max_generations=max_generations, pop_size=pop_size,

        problem=SCHProblem(),

        representation=Representation(
            initialize=create_real_vector(bounds=[(-10, 10)])
        ),

        pipeline=[
            random_selection,
            clone,
            mutate_gaussian(std=0.5, expected_num_mutations=1),
            evaluate,
            pool(size=pop_size),
        ]
    )

The above code snippet shows how to set up NSGA-II for one of the benchmark
multiobjective problems, `SCHProblem`.  We specify the maximum number of generations,
the population size, representation, and give a reproduction pipeline.  The
representation is a simple single valued gene, that we see on line 15 is initialized
in the range of [-10,10].

The reproduction pipeline given on lines 18-24 is used to create the offspring for each generation. It is spliced
into another pipeline so that the offspring created via this pipeline are then passed to the rank sorting and
crowding distance functions.  Then truncation selection based on rank and crowding distance is used to return the
final set of offspring that then become the parents for the next generation.

Creating a tailored NSGA-II
---------------------------

However, it may be desirable to have fine-grained control over the NSGA-II implementation, maybe to
more conveniently perform some necessary ancillary calculations during a run.  In that case, the
lower-level NSGA-II operators can be directly used in a full LEAP pipeline, as shown below.

Example
^^^^^^^

.. code-block:: Python
    :linenos:



.. [NSGA-II] Deb, Kalyanmoy, Amrit Pratap, Sameer Agarwal, and T. A. M. T. Meyarivan.
            "A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II." IEEE
            transactions on evolutionary computation 6, no. 2 (2002): 182-197.

.. [Burlacu] Bogdan Burlacu. 2022. Rank-based Non-dominated Sorting. arXiv.
      DOI:https://doi.org/10.48550/ARXIV.2203.13654