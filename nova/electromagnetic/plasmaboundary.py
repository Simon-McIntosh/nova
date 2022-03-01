"""Manage biot calculation for wall contours."""
from dataclasses import dataclass

import numpy as np

from nova.electromagnetic.biotpoint import BiotPoint


@dataclass
class PlasmaBoundary(BiotPoint):
    """Compute interaction for a series of discrete points."""

    def solve(self, points):
        """Solve Biot wall-pannel mid-points."""
        firstwall = np.empty((2*len(points)-1, 2))
        firstwall[::2] = points
        firstwall[1::2] = (points[:-1, :] + points[1:, :]) / 2
        super().solve(firstwall)

    def plot(self, axes=None, **kwargs):
        """Plot wall pannels."""
        self.axes = axes
        kwargs = dict(marker='o', linestyle='', ms=4, color='C0') | kwargs
        self.axes.plot(self.data.x, self.data.z, **kwargs)
