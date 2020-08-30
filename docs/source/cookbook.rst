LEAP Cookbook
=============


Enforcing problem bounds constraints
------------------------------------
There are two overall types of bounds enforcement within EAs, soft bounds and
hard bounds:

:soft bounds: where the boundaries are enforced only at initialization, but
    mutation allows for exploring beyond those initial boundaries

:hard bounds: boundaries are strictly enforced at initialization as well as during
    mutation and crossover.  In the latter this can be done by clamping new values
    to a given range, or flagging an individual that violations such constraints
    as non-viable by throwing an exception during fitness evaluation.  (That is,
    during evaluations, exceptions are caught, the individual's fitness is set to
    NaN and its `is_viable` internal flag set to false; then selection should
    hopefully weed out this individual from the population.)

.. TODO Add details for relevant functions and operators as well as provide examples

