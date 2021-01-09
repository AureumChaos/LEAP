# LEAP CHANGES BY VERSION

Being a terse compilation by version of changes.

## 0.6.0

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
