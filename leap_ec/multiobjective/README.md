This sub-package is for LEAP support for multiobjective optimization.

As a side-effect some operators will add state to individuals as they pass 
through a LEAP pipeline.

`fast_nondominated_sort()` adds these attributes:

* `dominates`, a list
* `dominated_by`, integer, count of others that dominate current individual
* `rank`, integer, valid values in regular expression `[1-9]+`

Alternatively, we could have used a sub-class to manage this additional 
state, but we felt that could unnecessarily complicate implementations, 
particularly in situations where a user had already defined their own 
`Individual` subclass for their own specialized needs.  This approach 
provides more flexability in thtat 
