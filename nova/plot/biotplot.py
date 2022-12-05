"""Manage biot plot utility functions."""
from dataclasses import dataclass

from nova.frame.baseplot import Plot


@dataclass
class BiotPlot(Plot):
    """Biot plot base class."""

    levels: int | list[float] = 31

    def contour_kwargs(self, **kwargs):
        """Return contour plot kwargs."""
        return dict(colors='lightgray', linewidths=1.5, alpha=0.9,
                    linestyles='solid', levels=self.levels) | kwargs
