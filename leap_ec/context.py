#!/usr/bin/env python3
"""
This defines a global context that is a dictionary of dictionaries.  The
intent is for certain operators and functions to add to and modify this
context.  Third party operators and functions will just add a new top-level
dedicated key.
context['leap'] is for storing general LEAP running state, such as current
   generation.
context['leap']['distributed'] is for storing leap.distributed running state
context['leap']['distributed']['non_viable'] accumulates counts of non-viable
   individuals during distributed.eval_pool() and
   distributed.async_eval_pool() runs.
"""
context = {'leap': {'distributed': {'non_viable': 0}}}
