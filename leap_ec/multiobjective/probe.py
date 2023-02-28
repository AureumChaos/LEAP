#!/usr/bin/env python3
""" Visualization pipeline operators tailored for multiple objectives
"""
from leap_ec.probe import PopulationMetricsPlotProbe
from leap_ec.global_vars import context


class MOPopulationMetricsPlotProbe2D(PopulationMetricsPlotProbe):
    """
        For plotting two fitnesses against one another with optional Pareto
        frontier calculation.
    """

    def __init__(self, ax=None, xlim=(0, 100), ylim=(0, 1), modulo=1,
                 title='2D fitnesses', x_axis_value=None, context=context):
        super().__init__(ax=ax,
                         xlim=xlim, ylim=ylim, modulo=modulo, title=title,
                         x_axis_value=x_axis_value, context=context)



if __name__ == '__main__':
    probe = MOPopulationMetricsPlotProbe2D()

    pass
