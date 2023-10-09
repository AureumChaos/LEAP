Common Problems
===============

Here we address common problems that may arise when using LEAP.

Missing pipeline operator arguments
-----------------------------------

If you see an error like this:

```
TypeError: mutate_binomial() missing 1 required positional argument: 'next_individual'
```

The corresponding code may look like this:

.. code-block:: python

    int_ops.mutate_binomial(std=[context['leap']['std0'],
                                 context['leap']['std1']],
                            hard_bounds=[(1, 127), (0, 255)],
                            probability=context['leap']['mutation']),

In this case, the API for :func:`leap_ec.int_rep.ops.mutate_binomial` had changed
such that the argument `hard_bounds` had been shortened to `bounds`.  Renaming that
argument to `bounds` fixed this instance of the problem.

In general, if you see an error like this, you should check the API documentation and
ensure that all mandatory function arguments are getting passed into the pipeline
operator.
