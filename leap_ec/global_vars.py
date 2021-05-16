#!/usr/bin/env python3
"""
This defines a global context that is a dictionary of dictionaries.  The
intent is for certain operators and functions to add to and modify this
context.  Third party operators and functions will just add a new top-level
dedicated key.
context['leap'] is for storing general LEAP running state, such as current
   generation.
context['leap']['distrib'] is for storing leap.distrib running state
context['leap']['distrib']['non_viable'] accumulates counts of non-viable
   individuals during distrib.eval_pool() and
   distrib.async_eval_pool() runs.
"""
context = {'leap': {'distrib': {'non_viable': 0}}}
