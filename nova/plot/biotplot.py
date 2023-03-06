"""Manage biot plot utility functions."""
from dataclasses import dataclass, field

import numpy as np
from nova.frame.baseplot import Plot


@dataclass
class LinePlot(Plot):
    """Single contour plot base class."""

    color: str = field(init=False, default='lightgray', repr=False)
    linewidth: float = field(init=False, default=1.5, repr=False)
    linestyle: str = field(init=False, default='solid', repr=False)
    alpha: float = field(init=False, default=0.9, repr=False)

    def plot_kwargs(self, **kwargs):
        """Return line plot kwargs."""
        return dict(color=self.color, linewidth=self.linewidth,
                    alpha=self.alpha, linestyle=self.linestyle) | kwargs


@dataclass
class BiotPlot(LinePlot):
    """Biot plot base class."""

    levels: int | list[float] | np.ndarray = 31

    def contour_kwargs(self, **kwargs):
        """Return contour plot kwargs."""
        return dict(colors=self.color, linewidths=self.linewidth,
                    alpha=self.alpha, linestyles=self.linestyle,
                    levels=self.levels) | kwargs
