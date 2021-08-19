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

.. autofunction:: leap_ec.probe.print_individual
