#! /usr/bin/env python

import numpy

x = [1.0]
xp = [0.0, 2.0]
fp = [10.0, 6.0]
y = numpy.interp(x, xp, fp)

