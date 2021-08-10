Prebuilt Algorithms
===================
.. _fig1:
.. figure:: _static/tiers.png
    :align: center

    **Figure :numref:`fig1` The three tiers of tailorability for using LEAP**
    LEAP has three levels of abstraction with gradually increasing order of
    customization.  The top- level entry-point, `ea_solve`, has the least
    customization and is ideal for real-valued optimization problems. The
    mid-level allows for more user customization.  `generational_ea` allows for
    implementing most traditional evolutionary algorithms, such as genetic
    algorithms and evolutionary programs.  The bottom-most level provides the
    greatest amount of flexability where users specify their evolutionary
    algorithm implemention using low-level LEAP components.


foo
