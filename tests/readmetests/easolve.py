#!/usr/bin/env python3
"""
    Ensures that the first code snippet in top-level README.md works.
"""

from leap_ec.simple import ea_solve

def f(x):
    """A real-valued function to optimized."""
    return sum(x)**2

ea_solve(f, bounds=[(-5.12, 5.12) for _ in range(5)], maximize=True)
