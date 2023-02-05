Probes
======

Probes are special pipeline operators that can be used to echo state of
passing individuals or other data.  For example, you might want to print
the state of an individual with two probes, one before a mutation operator
is applied, and another afterwards to observe the effects of mutation.

Simple printing probes
----------------------

`print_probe()` and `print_individual` will write out the entire population
or a single individual to a given stream.  The default stream is `sys.stdout`.

.. autofunction:: leap_ec.probe.print_probe
    :noindex:

.. autofunction:: leap_ec.probe.print_individual
    :noindex:


Information probes
------------------

These are probes do more than passive reporting of data that passes through
the pipeline -- they actually do some data processing and report that.

.. autoclass:: leap_ec.probe.BestSoFarProbe
    :members:
    :undoc-members:
    :noindex:
