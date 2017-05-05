# Utilities for manually plotting a semilogx plot.
#
# Call plot_semilogx() first on all the data to be plotted, then call
# set_ticks_and_labels().

import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker

class FakeLogLocator(mpl.ticker.Locator):
    """ Helper to produce log-spaced ticks on a non-log axis."""
    def __call__(self):
        vmin, vmax = 10**(self.axis.get_view_interval())
        return self.tick_values(vmin, vmax)

    def tick_values(self, vmin, vmax):
        min_decade = int(math.floor(np.log10(vmin)))
        max_decade = int(math.floor(np.log10(vmax)))
        res = []
        for dec in range(min_decade, max_decade):
            delta = 10^dec
            i = np.array(range(10**dec, 10**(dec+1), 10**dec))
            res.extend(np.log10(i))
        return res

def set_axis_limits(ax, max_logx, min_logx):
    nticks = max_logx - min_logx + 1
    ax.set_xlim(left=min_logx, right=max_logx)
    ax.xaxis.set_major_locator(mpl.ticker.LinearLocator(numticks=nticks))
    ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter(r"$10^{%d}$"))
    ax.xaxis.set_minor_locator(FakeLogLocator())
    ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())

def plot_semilogx(ax, xarray, yarray, **kwargs):
    logx = np.log10(xarray)
    ax.plot(logx, yarray, **kwargs)
