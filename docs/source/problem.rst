.. _problem:

Problems
========
This section covers `Problem` classes in more detail.

Class Summary
-------------

.. _problem-class:
.. figure:: _static/problem_class_diagram.svg

    **The `Problem` abstract-base class**  This class diagram shows the
    detail for `Problem`, which is an abstract base class (ABC).  It has three
    abstract methods that must be over-ridden by subclasses. `evaluate()`
    takes a phenome from an individual and compute a fitness from that.
    `worse_than()` and `equivalent()` compare fitnesses from two different
    individuals and, as the name suggests, respectively returns the worst
    of the two or the equivalent within the `Problem` context.

As shown in :numref:`problem-class`, the `Problem` abstract-base class has three abstract
methods. `evaluate()` takes
a phenome that was `decode()d` from an `Individual`'s genome, and returns a value
denoting the quality, or fitness, of that individual.  `Problems` are also used
to compare the fitnesses between `Individuals`.  `worse_than()` returns true if
the first individual is less fit than the second.  Similarly, `equivalent()` is
used to determine if two given fitnesses are effectively the same.


Class API
---------
.. inheritance-diagram:: leap_ec.problem

.. automodule:: leap_ec.problem
    :members:
    :undoc-members:
    :noindex:

Binary Problems API
-------------------
.. inheritance-diagram:: leap_ec.binary_rep.problems

.. automodule:: leap_ec.binary_rep.problems
    :members:
    :undoc-members:
    :noindex:


Real-value Problems API
-----------------------
.. inheritance-diagram:: leap_ec.real_rep.problems

.. automodule:: leap_ec.real_rep.problems
    :members:
    :undoc-members:
    :noindex:
