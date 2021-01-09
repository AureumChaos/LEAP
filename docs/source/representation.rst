Representations
===============
When implementing an EA, one of the first design decisions that a practitioner
must make is how to represent their problem in an individual.  In this section
we share how to structure individuals to represent a posed solution instance
for a given problem.

Generally, each representation has a specific function, or set of functions, to
create genomes for values of that representation type.  There is sometimes also
a decoders tailored to translate genomes into desired values.  And, lastly, there
will be a set of pipeline operators specific to that representation.

In summary, representations can have the following:

:initializers:
    These are for creating random genomes of that representation.
:decoders:
    These are for translating genomes to usable values; note not all representations need
    decoders in that you can directly use the genome values, and which is typical
    for real-valued and integer representations. (Which are a type of _phenotypic_
    representation.)
:pipeline operators:
    Most all representations will have pipeline operators that are specific to
    that type

Binary representations
----------------------
A common representation for individuals is as a string of binary digits.

.. automodule:: leap_ec.binary_rep.decoders
    :members:
    :undoc-members:
    :show-inheritance:
    :noindex:

.. automodule:: leap_ec.binary_rep.initializers
    :members:
    :undoc-members:
    :show-inheritance:
    :noindex:

.. automodule:: leap_ec.binary_rep.ops
    :members:
    :undoc-members:
    :show-inheritance:
    :noindex:

Real-valued representations
---------------------------
Another common representation is a vector of real-values.

.. automodule:: leap_ec.real_rep.initializers
    :members:
    :undoc-members:
    :show-inheritance:
    :noindex:

.. automodule:: leap_ec.real_rep.ops
    :members:
    :undoc-members:
    :show-inheritance:
    :noindex:


Integer representations
-----------------------
A vector of all integer values is also a common representation.

.. automodule:: leap_ec.int_rep.initializers
    :members:
    :undoc-members:
    :show-inheritance:
    :noindex:

.. automodule:: leap_ec.int_rep.ops
    :members:
    :undoc-members:
    :show-inheritance:
    :noindex:

Segmented representations
-------------------------
Segmented representations are a wrapper around another, arbitrary
representation, such as a binary, real-valued, or integer representation.
Segmented representations allow for sequences of value "chunks". For example,
a Pitt Approach could be implemented using this representation where each
segment represents a single rule.  Another example would be each segment
represents associated hyper-parameters for a convolutional neural network
layer.

.. automodule:: leap_ec.segmented_rep.decoders
    :members:
    :undoc-members:
    :show-inheritance:
    :noindex:

.. automodule:: leap_ec.segmented_rep.initializers
    :members:
    :undoc-members:
    :show-inheritance:
    :noindex:

.. automodule:: leap_ec.segmented_rep.ops
    :members:
    :undoc-members:
    :show-inheritance:
    :noindex:


Mixed representations
---------------------
There is currently no explicit support for mixed representations, but there are
plans to implement such at some point.  There are a few strategies for implementing
mixed values:

* use a binary representation with an associated `Decoder` that decodes values
  into desired target value formats, such as sequences that are a blend of
  integers, floating point, and categorical variables
* use a floating point representation that has an associated decoder for mapping
  certain floating point values to integer or categorical values; an associated
  mutation function may be necessary to implement pertubations that make sense
  for individual genes
* likewise use an integer representation with tailored associated decoders and
  mutators to decode and change values in a bespoke way

`Representation` convenience class
----------------------------------
Since the notion of a representation includes how individuals are created,
how they're decoded, and are bound to a particular class for
an individual, the class :py:class:`leap_ec.representation.Representation`
was created to bundle those together.  By default, the class for individual is
:py:class:`leap_ec.individual.Individual`, but can of course be any of its
subclasses.

The :py:class:`leap_ec.representation.Representation` is used in
:py:class:`leap_ec.algorithm.generational_ea`.  In the future this may become
a formal python dataclass and be more integrated into LEAP in other functions.

.. automodule:: leap_ec.representation
    :members:
    :undoc-members:
    :show-inheritance:
    :noindex:

