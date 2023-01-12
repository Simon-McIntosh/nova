"""Manage biot calculation for wall contours."""
from dataclasses import dataclass

import numpy as np

from nova.biot.biotpoint import BiotPoint
from nova.geometry.polyframe import PolyFrame


@dataclass
class BiotFirstWall(BiotPoint):
    """Compute interaction for a series of discrete points."""

    @property
    def loop(self):
        """Return first wall loop."""
        try:
            return self.Loc['plasma', 'poly'][0].boundary
        except AttributeError:
            return PolyFrame.loads(self.Loc['plasma', 'poly'][0]).boundary

    def solve(self):
        """Solve Biot wall-pannel mid-points."""
        loop = self.loop
        self.data.coords['x'] = loop[:, 0]
        self.data.coords['z'] = loop[:, 1]
        firstwall = np.empty((2*len(loop)-1, 2))
        firstwall[::2] = loop
        firstwall[1::2] = (loop[:-1, :] + loop[1:, :]) / 2
        super().solve(firstwall)

    def plot(self, axes=None, **kwargs):
        """Plot wall pannels."""
        if len(self.data) == 0:
            return
        self.axes = axes
        kwargs = dict(marker=None, linestyle='-', ms=4, color='gray',
                      linewidth=1.5) | kwargs
        self.axes.plot(self.data.x, self.data.z, **kwargs)
