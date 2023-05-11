Context
=======

From time to time pipeline operators need to consult some sort of state such as the
current generation.  E.g., `ops.migrate` uses the `context` to track subpopulations.

`context` is found in `leap_ec.context` and is just a dictionary.  The default
element, `leap`, is reserved for LEAP data.

Summary of current `leap_ec.context` reserved state:

* `context['leap']` is for storing general LEAP running state, such as current
   generation.
* `context['leap']['distributed']` is for storing `leap.distributed` running state
* `context['leap']['distributed']['non_viable']` accumulates counts of non-viable
   individuals during `distributed.eval_pool()` and
   `distributed.async_eval_pool()` runs.
