.. _individuals:

Individuals
===========
This section covers the class `Individual` in more detail.

Class Summary
-------------

.. _individual-class:
.. figure:: _static/individual-class-diagram.svg
    :align: center

    **The `Individual` class**  This class diagram shows the detail
    for `Individual`.  In additional to the association with `Decoder` and
    `ProbLem`, each `Individual` has a `genome` and `fitness`.  There are also
    several member functions for cloning, decoding, and evaluating individuals.
    Not shown are such member functions as `__repr__()` and `__str__()`.

An `Individual` poses a unique instance of a solution to the associated `Problem`.
Each `Individual` has a `genome`, which contains state representing that
posed solution.  The `genome` can be a sequence or a matrix or a tree or some
other data structure, but in practice a `genome` is usually a binary or a real-value
sequence represented as a numpy array.  Every `Individual` is connected to an
associated `Problem` and relies on the `Problem` to evaluate its fitness and to
compare itself with another `Individual` to determine the better of the two.

The `clone()` method will create a duplicate of a given `Individual`; the new
`Individual` gets a deep copy of the `genome` and refers to the same `Problem` and
`Decoder`; also, the clone gets its own UUID and has its `self.parents` set updated
to include the individual from which it was cloned (i.e., its parent).
`evaluate()` calls `evaluate_imp()` that, in turn, calls `decode()` to
translate the `genome` into phenomes, or values meaningful to the `Problem`, and
then passes those values to the `Problem` where it returns a fitness. This
fitness is then assigned to the `Individual`.

The reason for the indirection using `evaluate_imp()` is that `evaluate_imp()`
allows sub-classes to pass ancillary information to `Problem` during evaluation.
For example, an `Individual` may have a UUID that the `Problem` needs in order
to create a file or sub-directory using that UUID.  `evaluate_imp()` can be
over-ridden in a sub-class to pass along the UUID in addition to the decoded
`genome`.

The `@total_ordering` class wrapper is used to expand the member functions
`__lt__()` and `__eq__()` that are, in turn, heavily used in sorting, selection,
and comparison operators.

.. _robust-individual:

`RobustIndividual`
^^^^^^^^^^^^^^^^^^^

`RobustIndividual` is a sub-class of `Individual` that over-rides `evaluate()`
to handle exceptions thrown during evaluation.  If no exceptions are thrown,
then `self.is_viable` is set to `True`.  If an exception happens, then the
following occurs:

* `self.is_viable` is set to `False`
* `self.fitness` is set to `math.nan`
* `self.exception` is assigned the `Exception` object

In turn, this class has another sub-class `leap_ec.distributed.individual.DistributedIndividual`.

Class API
-------------
.. inheritance-diagram:: leap_ec.individual.RobustIndividual

.. autoclass:: leap_ec.individual.Individual
    :members:
    :undoc-members:
    :noindex:

    .. automethod:: __init__

.. autoclass:: leap_ec.individual.RobustIndividual
    :members:
    :undoc-members:
    :noindex:

    .. automethod:: __init__

.. autoclass:: leap_ec.individual.WholeEvaluatedIndividual
    :members:
    :undoc-members:
    :noindex:

    .. automethod:: __init__