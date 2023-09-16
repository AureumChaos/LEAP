Parsimony Pressure
==================

One common problem with variable length representations is "bloat" whereby
genome lengths will gradually increase over time.  This may be due to over-fitting
or the accumulation of "junk DNA" over time.

LEAP currently provides two approaches to mitigating bloat.  First is a very
simple "genome tax," or penalty by genome length, popularized by
:cite:t:`Koza1992`.  The second is lexicographical
parsimony, or "tie breaking parsimony," where the individual with the shortest
genome is returned if their respective fitnesses happen to be
equivalent :cite:t:`Luke2002`.

.. footbibliography::

API
---

.. automodule:: leap_ec.parsimony
    :members:
    :undoc-members:
    :noindex:
