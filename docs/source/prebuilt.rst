Prebuilt Algorithms
===================
.. _figtiers:
.. figure:: _static/tiers.png
    :align: left

    **The three tiers of tailorability for using LEAP**
    LEAP has three levels of abstraction with gradually increasing order of
    customization.  The top-level has ea_solve() that is ideal for real-valued
    optimization problems.  The mid-level has two functions that allows for some
    tailoring of the pipeline and representation, generational_ea() and
    steady_state().  The bottom-most tier provides maximum flexibility and
    control over an EA's implementation, and involves the practitioner assembling
    bespoke EAs using LEAP low-level components, as shown by the code snippet.


:numref:`figtiers` depicts he top-level entry-point, `ea_solve()`, and has the
least customization, but is ideal for real-valued optimization problems. The
mid-level allows for more user customization. `generational_ea()` allows for
implementing most traditional evolutionary algorithms, such as genetic
algorithms and evolutionary programs. `asynchronous.steady_state()` is used to
implement an asynchronous steady-state EA suitable for HPC platforms as they
make the best use of HPC resources. The bottom-most level provides the
greatest amount of flexability, and is where users implement their
evolutionary algorithms using low-level LEAP components.

.. autoclass:: leap_ec.simple.ea_solve
    :members:
    :undoc-members:
    :noindex:
