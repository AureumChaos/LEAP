Decoders
========
This section covers `Decoder`s in more detail.

Class Summary
-------------

.. figure:: _static/decoder_class_diagram.svg
    :align: center

    **Figure 1: The `Decoder` abstract-base class**  This class diagram shows the
    detail for `Decoder`, which is an abstract base class (ABC).  It has just
    a single abstract function, `decode()`, that is intended to be defined by
    subclasses.

The abstract-base class, `Decoder` has one function intended to be overridden
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

.. inheritance-diagram:: leap_ec.core.Decoder

.. autoclass:: leap_ec.core.Decoder
    :members:
    :undoc-members:

    .. automethod:: __init__


IdentityDecoder
^^^^^^^^^^^^^^^

.. inheritance-diagram:: leap_ec.core.IdentityDecoder

.. autoclass:: leap_ec.core.IdentityDecoder
    :members:
    :undoc-members:

    .. automethod:: __init__

BinaryToIntDecoder
^^^^^^^^^^^^^^^^^^

.. inheritance-diagram:: leap_ec.core.BinaryToIntDecoder

.. autoclass:: leap_ec.core.BinaryToIntDecoder
    :members:
    :undoc-members:

    .. automethod:: __init__

BinaryToRealDecoderCommon
^^^^^^^^^^^^^^^^^^^^^^^^^

.. inheritance-diagram:: leap_ec.core.BinaryToRealDecoderCommon

.. autoclass:: leap_ec.core.BinaryToRealDecoderCommon
    :members:
    :undoc-members:

    .. automethod:: __init__

BinaryToRealDecoder
^^^^^^^^^^^^^^^^^^^

.. inheritance-diagram:: leap_ec.core.BinaryToRealDecoder

.. autoclass:: leap_ec.core.BinaryToRealDecoder
    :members:
    :undoc-members:

    .. automethod:: __init__

BinaryToIntGreyDecoder
^^^^^^^^^^^^^^^^^^^^^^

.. inheritance-diagram:: leap_ec.core.BinaryToIntGreyDecoder

.. autoclass:: leap_ec.core.BinaryToIntGreyDecoder
    :members:
    :undoc-members:

    .. automethod:: __init__

BinaryToRealGreyDecoder
^^^^^^^^^^^^^^^^^^^^^^^

.. inheritance-diagram:: leap_ec.core.BinaryToRealGreyDecoder

.. autoclass:: leap_ec.core.BinaryToRealGreyDecoder
    :members:
    :undoc-members:

    .. automethod:: __init__

