"""Manage biot calculation for wall contours."""
from dataclasses import dataclass

import numpy as np

from nova.biot.biotpoint import BiotPoint


@dataclass
class BiotFirstWall(BiotPoint):
    """Compute interaction for a series of discrete points."""

    def solve(self):
        """Solve Biot wall-pannel mid-points."""
        points = self.Loc['plasma', 'poly'][0].boundary
        firstwall = np.empty((2*len(points)-1, 2))
        firstwall[::2] = points
        firstwall[1::2] = (points[:-1, :] + points[1:, :]) / 2
        super().solve(firstwall)

    def plot(self, axes=None, **kwargs):
        """Plot wall pannels."""
        if len(self.data) == 0:
            return
        self.axes = axes
        kwargs = dict(marker=None, linestyle='-', ms=4, color='gray',
                      linewidth=1.5) | kwargs
        self.axes.plot(self.data.x, self.data.z, **kwargs)
