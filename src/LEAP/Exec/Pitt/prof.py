#! /usr/bin/env python

import pstats
p = pstats.Stats('ga.prof').strip_dirs()

p.sort_stats('time').print_stats()
p.sort_stats('cumulative').print_stats()



