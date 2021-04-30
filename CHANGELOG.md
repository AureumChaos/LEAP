# LEAP CHANGES BY VERSION

Being a terse compilation by version of changes.

## 0.6.0

* New features
  * Added `landscape_features` package with some initial exploratory landscape analysis tools
  * Added elitism
  * Added a new example demonstrating integer representations
  * Added a `mutate_binomial()` operator for integer representations
  * Added visualization of ANN weights for `SimpleNeuralNetworkExecutable` phenotypes
  * Added metrics for logging population diversity

* Documentation
  * Added documentation on `leap_ec.context` and updated software development
  guidelines to encourage its use if tracking persistent state outside of 
  function calls was necessary.

* CI/CD
  * Added a `make test-slow` harness that runs the `examples/` scripts
  * Improved test coverage

* Bugfixes
  * Fix `viz` parameter when calling `simple.ea_solve()`
  * Tell `dask` that functions are impure by default, to make sure it doesn't cache results

* API changes
  * Renamed `leap.distributed` package to `leap.distrib` (to avoid conflicts with ex. `dask.distributed`)
  * Significantly refactored the `executable_rep.rules` package to simplify learning classifier systems
  * Added probability parameter for the `uniform_crossover` operator
  * `mutate_gaussian` now accepts a list of gene-wise hard bounds
  * Added `notes` columns parameter to `FitnessStatsCSVProbe`
  * Added a `pad_inputs` parameter to `TruthTableProblem` to handle varying-dimension inputs
  * Renamed `PlotTrajectoryProbe` to the more descriptive `CartesianPhenotypePlotProbe`
  * Renamed `PopulationPlotProbe` to the more descriptive `PopulationMetricsPlotProbe`
  * Added `FitnessPlotProbe` as a convenience wrapper for `PoulationMetricsPlotProbe`
  * Default behavior changes
    * `Representation` now uses a phenotypic representation (`IdentityDecoder`) by default
    * Mutation operators no longer have default mutation rates (they must be explicitly set by the user).
    * Set default `p_swap = 0.2` for `uniform_crossover`, instead of 0.5
    * Set default `num_points = 2` for `n_ary_crossover`, instead of 1
    * Set default value for `context` parameter on probes, so users needn't set it
    * standardized on making `context` last function argument that defaults to
    `leap_ec.context.context`


## 0.5.0, 1/9/2021

* Added probability parameter for the `n_ary_crossover` operator
* Greatly improved test coverage
* Added support for static- and variable-length segments, which are fixed-length "chunks" of values
* Added a simple neural network representation, `executable_rep.neural_network`, and made it the default for `examples/openai_gym.py`
* Changed the `Executable` interface to act as a `Callable` object (rather than using a custom `output()` method)
* Added `statistical_helpers` to assist with writing unit tests for stochastic algorithms
* Added support for integer representations, via the `int_rep` package
* Added a Cartesian genetic programming (CGP) representation, `executable_rep.cgp`, with example in `examples/cgp.py`
* Added support for heterogeneous island models, demoed in `examples/multitask_island_model.py`


## 0.4.0, 9/19/2020

* Significantly added to online [documentation](https://leap-gmu.readthedocs.io/en/latest/index.html)
* Major code reorganization
    * exception management for `Individual` has been moved to `RobustIndividual`
    * `DistributedIndividual` now inherits from `RobustIndividual`
    * `core.py` has been broken out to separate modules
        * `Individual` and `RobustIndividual` now in `individual.py`
        * representation specific entities moved to new sub-packages, `binary_rep`
          and `real_rep`
        * `Representation` now in `representation.py`
        * `Decoder` now in `decoder.py`
    * documentation, doctests, examples, Jupyter notebooks, and unit tests updated accordingly 
* added ability to pass ancillary information during evaluation, such as UUIDs
  that could be used to name output files and directories, yet do not have a 
  direct impact on fitness


## 0.3.1

* Apply `Representation` consistently throughout LEAP, particularly the top-level monolithic functions
* Added probe to `leap_ec.distributed.asynchronous.steady_state()` to take regular snapshots of the population


## 0.3, 6/14/2020

* fix how non-viable individuals sort themselves when compared since the prior method of comparing `math.nan` to `math.nan` yielded non-ideal behavior 
* minor maintenance tweaks


## 0.2, 6/14/2020

* changed package name to `leap_ec` from `leap` to mitigate pypi namespace collisions
* minor maintenance tweaks


## 0.1

* first major "mature" release of LEAP
