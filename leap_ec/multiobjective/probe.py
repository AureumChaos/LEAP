#!/usr/bin/env python3
""" Visualization pipeline operators tailored for multiple objectives
"""
from leap_ec.probe import PopulationMetricsPlotProbe
from leap_ec.global_vars import context



##############################
# Class ParetoPlotProbe
##############################
class ParetoPlotProbe2D:
    """
    Plot a 2D Pareto front of a population that has been assigned
    multi-objective fitness values.

    If the fitness space has more than two dimensions, only the first two are
    plotted.
    """
    def __init__(self, ax=None, xlim=(0, 1), ylim=(0, 1),
                 title='Pareto Front',
                 step=1, context=context):
        """
        :param Axes ax: Matplotlib axes to plot to (if `None`, a new figure will
            be created).
        :param xlim: Bounds of the horizontal axis.
        :type xlim: (float, float)
        :param ylim: Bounds of the vertical axis.
        :type ylim: (float, float)
        :param int step: take and plot a measurement every `step` steps (
            default 1).
        :param title: title to print on the plot
        :param context: set a context object to query for the current generation.
            Defaults to the standard `leap_ec.context` object.
        """
        if ax is None:
            _, ax = plt.subplots()

        self.sc = ax.scatter([], [])

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        self.ax = ax
        self.left, self.right = xlim
        self.bottom, self.top = ylim
        self.x = np.array([])
        self.y = np.array([])
        plt.title(title)
        self.step = step
        self.context = context

    def __call__(self, population):
        assert (population is not None)
        assert ('leap' in self.context)
        assert ('generation' in self.context['leap'])
        step = self.context['leap']['generation']

        if step % self.step == 0:
            self.x = np.array([ind.fitness[0] for ind in population])
            self.y = np.array([ind.fitness[1] for ind in population])
            self.sc.set_offsets(np.c_[self.x, self.y])
            self.__rescale_ax()
            self.ax.figure.canvas.draw()
            plt.pause(0.000001)
        return population

    def __rescale_ax(self):
        if np.min(self.x) < self.left:
            self.ax.set_xlim(left=np.min(self.x))
        if np.max(self.x) > self.right:
            self.ax.set_xlim(right=np.max(self.x))
        if np.min(self.y) < self.bottom:
            self.ax.set_ylim(bottom=np.min(self.y))
        if np.max(self.y) > self.top:
            self.ax.set_ylim(top=np.max(self.y))



if __name__ == '__main__':
    probe = ParetoPlotProbe2D()

    pass
