Visualization
=============

Being able to visualize a running evolutionary algorithm is important.  Here we describe
special pipeline operators that use matplotlib to visualize the state of the population.

Prebuilt Algorithms
-------------------

LEAP's "prebuilt" algorithms (also sometimes referred to as "monolithic functions")
have optional support for visualizations.

:class:`leap_ec.simple.ea_solve` has two optional arguments that control visualization: `viz` and `viz_ylim`.
`viz` is a boolean that controls whether or not to display the visualization; if `True`
a matplot lib window will appear and update during a run with the . `viz_ylim`
is used to supply the initial bounds for the y-axis of the visualization.  The plotting
is carried out via an instance of :class:`leap_ec.probe.FitnessPlotProbe`, and which is
added as a last pipeline operator; this means that it will be plotting the
created offspring with each iteration.

The other monolithic function, :class:`leap_ec.algorithm.generational_ea`, offers
more fine-tuned control over visualization.  Since the user specifies the pipeline,
a visualization pipeline operator can be added anywhere in the pipeline.  Moreover,
since the user is specifying the visualization operator, they're free to tailor
how the visualization is done.  For example, the user can specify a custom title and
the update frequency.

Tailored evolutionary algorithms
--------------------------------

Of course many practitioners will want to build their own evolutionary algorithms and
forgo the use of the aforementioned monolithic functions.  For these users, LEAP offers
a number of pipeline operators that can be used to visualize the state of the population
merely by inserting an instance of one of these into the pipeline.  A full list of
such operators is in the next section.

Visualization Pipeline Operators
--------------------------------

* :class:`leap_ec.probe.FitnessPlotProbe`
    A pipeline operator that plots the fitness of the population with each iteration.
* :class:`leap_ec.probe.PopulationMetricsPlotProbe`
    A pipeline operator that plots user-specified metrics of the population with each invocation.  The user is free to specify which metrics to plot. Please refer to `examples/simple/onemax_style_problems.py` for an example of how to use this operator.
* :class:`leap_ec.probe.CartesianPhenotypePlotProbe`
    A pipeline operator that plots the phenotypes of the population with each iteration.  This operator is only useful for problems where the phenotype is a 2D point.
* :class:`leap_ec.probe.HistPhenotypePlotProbe`
    A pipeline operator that shows a dynamic histogram of phenotypes.
* :class:`leap_ec.probe.HeatMapPhenotypeProbe`
    A pipeline operator that shows a heatmap of phenotypes.
* :class:`leap_ec.probe.SumPhenotypePlotProbe`
    This operator plots the sum of the phenotype vector with each iteration. For example, this is good for the MAXONES problem that is literally the sum of all the ones in a binary vector.

Examples
--------

You can find examples on how to use these probes in the following:

* `examples/advanced/custom_stopping_condition.py`
* `examples/advanced/neural_network_cartpole.py`
* `examples/advanced/island_model.py`
* `examples/advanced/cgp_images.py`
* `examples/advanced/cgp.py`
* `examples/advanced/real_rep_with_diversity_metrics.py`
* `examples/advanced/multitask_island_model.py`
* `examples/advanced/external_simulation.py`
* `examples/advanced/pitt_rules_cartpole.py`
* `examples/distributed/simple_sync_distributed.py`
* `examples/simple/int_rep.py`
* `examples/simple/one+one_es.py`
* `examples/simple/onemax_style_problems.py`
* `examples/simple/real_rep_genewise_mutation.py`
