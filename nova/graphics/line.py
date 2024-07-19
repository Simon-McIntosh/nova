"""Manage biot plot utility functions."""

from dataclasses import dataclass, field

import numpy as np
from nova.graphics.plot import Plot


@dataclass
class Line(Plot):
    """Single contour plot base class."""

    color: str = field(init=False, default="lightgray", repr=False)
    linewidth: float = field(init=False, default=1.5, repr=False)
    linestyle: str = field(init=False, default="solid", repr=False)
    alpha: float = field(init=False, default=0.9, repr=False)

    def plot_kwargs(self, **kwargs):
        """Return line plot kwargs."""
        return (
            dict(
                color=self.color,
                linewidth=self.linewidth,
                alpha=self.alpha,
                linestyle=self.linestyle,
            )
            | kwargs
        )


@dataclass
class Chart(Line):
    """Multi-contour base class."""

    levels: int | list[float] | np.ndarray = 31

    def contour_kwargs(self, **kwargs):
        """Return contour plot kwargs."""
        return (
            dict(
                colors=self.color,
                linewidths=self.linewidth,
                alpha=self.alpha,
                linestyles=self.linestyle,
                levels=self.levels,
            )
            | kwargs
        )

    def label_contour(self, label="", **kwargs):
        """Add label to contour plot."""
        if label:
            label_kwargs = {
                attr[:-1]: kwargs[attr]
                for attr in ["colors", "linewidths", "linestyles"]
            }
            self.axes.add_line(
                self.mpl["lines"].Line2D([], [], label=label, **label_kwargs)
            )
            loc = kwargs.get("loc", "upper right")
            self.legend(loc=loc)
