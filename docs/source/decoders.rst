Decoders
========
This section covers the `Decoder` class in more detail.

Class Summary
-------------

.. _decoder-class:
.. figure:: _static/decoder_class_diagram.svg

    **The Decoder abstract-base class**  This class diagram shows the
    detail for `Decoder`, which is an abstract base class (ABC).  It has just
    a single abstract function, `decode()`, that is intended to be defined by
    subclasses.

The abstract-base class, `Decoder` , has one function intended to be overridden
by sub-classes, `decode()`, that returns a phenome meaningful to a given
`Problem`, which is usually a sequence of values.  There are a number of
supplied `Decoder` classes mostly for converting binary strings
into integers or real values.

Note that there is also support for Gray encoding.  See `BinarytoIntGrayDecoder`
and `BinaryToRealGreyDecoder`.


Class API
---------

Decoder
^^^^^^^

.. inheritance-diagram:: leap_ec.decoder.Decoder

.. autoclass:: leap_ec.decoder.Decoder
    :members:
    :undoc-members:
    :noindex:

    .. automethod:: __init__


IdentityDecoder
^^^^^^^^^^^^^^^

.. inheritance-diagram:: leap_ec.decoder.IdentityDecoder

.. autoclass:: leap_ec.decoder.IdentityDecoder
    :members:
    :undoc-members:
    :noindex:

    .. automethod:: __init__

BinaryToIntDecoder
^^^^^^^^^^^^^^^^^^

.. inheritance-diagram:: leap_ec.binary_rep.decoders.BinaryToIntDecoder

.. autoclass:: leap_ec.binary_rep.decoders.BinaryToIntDecoder
    :members:
    :undoc-members:
    :noindex:

    .. automethod:: __init__

BinaryToRealDecoderCommon
^^^^^^^^^^^^^^^^^^^^^^^^^

.. inheritance-diagram:: leap_ec.binary_rep.decoders.BinaryToRealDecoderCommon

.. autoclass:: leap_ec.binary_rep.decoders.BinaryToRealDecoderCommon
    :members:
    :undoc-members:
    :noindex:

    .. automethod:: __init__

BinaryToRealDecoder
^^^^^^^^^^^^^^^^^^^

.. inheritance-diagram:: leap_ec.binary_rep.decoders.BinaryToRealDecoder

.. autoclass:: leap_ec.binary_rep.decoders.BinaryToRealDecoder
    :members:
    :undoc-members:
    :noindex:

    .. automethod:: __init__

BinaryToIntGreyDecoder
^^^^^^^^^^^^^^^^^^^^^^

.. inheritance-diagram:: leap_ec.binary_rep.decoders.BinaryToIntGreyDecoder

.. autoclass:: leap_ec.binary_rep.decoders.BinaryToIntGreyDecoder
    :members:
    :undoc-members:
    :noindex:

    .. automethod:: __init__

BinaryToRealGreyDecoder
^^^^^^^^^^^^^^^^^^^^^^^

.. inheritance-diagram:: leap_ec.binary_rep.decoders.BinaryToRealGreyDecoder

.. autoclass:: leap_ec.binary_rep.decoders.BinaryToRealGreyDecoder
    :members:
    :undoc-members:
    :noindex:

    .. automethod:: __init__
