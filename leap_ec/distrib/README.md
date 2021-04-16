* `asynchronous.py` -- support for asynchronous steady-state evolutionary algorithms
* `synchronous.py` -- support for synchronous EAs; i.e., essentially a map/reduce approach suitable for by-generation oriented EAs
* `evaluate.py` -- provides `evaluate()` that is common to both asynchronous and synchronous implementations
* `logging.py` -- optional `dask` worker plugin that adds a logger for each worker 
