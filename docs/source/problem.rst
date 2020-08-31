Problems
========
This section covers `Problems`s in more detail.

Class Summary
-------------

.. figure:: _static/problem_class_diagram.svg
    :align: center

    **Figure 1: The `Problem` abstract-base class**  This class diagram shows the
    detail for `Problem`, which is an abstract base class (ABC).  It has three
    abstract methods that must be over-ridden by subclasses. `evaluate()`
    takes a phenome from an individual and compute a fitness from that.
    `worse_than()` and `equivalent()` compare fitnesses from two different
    individuals and, as the name suggests, respectively returns the worst
    of the two or the equivalent within the `Problem` context.


