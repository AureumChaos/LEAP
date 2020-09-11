#!/usr/bin/env python3
"""
    Classes related to individuals that represent posed solutions.


    TODO Need to decide if the logic in __eq__ and __lt__ is overly complex.
    I like that this reduces the dependency on Individuals on a Problem
    (because sometimes you have a super simple situation that doesn't require
    explicitly couching your problem in a Problem subclass.)
"""
from math import nan
import abc
from copy import deepcopy
from functools import total_ordering
import random

from toolz import curry
from toolz.itertoolz import pluck

from leap_ec import util




