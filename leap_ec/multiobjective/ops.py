#!/usr/bin/env python3
"""
    LEAP pipeline operators for multiobjective optimization.
"""
from typing import Iterator
import random
from toolz import curry

import numpy as np

from leap_ec.ops import compute_expected_probability, listlist_op, iteriter_op

@curry
@listlist_op
def fast_nondominated_sort(population: list) -> list:
    raise NotImplementedError

@curry
@listlist_op
def crowding_distance_calc(population: list) -> list:
    raise NotImplementedError

@curry
@listlist_op
def sort_by_dominance(population: list) -> list:
    raise NotImplementedError
