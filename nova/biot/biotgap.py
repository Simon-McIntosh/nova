"""Manage biot calculation plasma gap flux probes."""
from dataclasses import dataclass, field

import numpy as np

from nova.biot.biotpoint import BiotPoint


@dataclass
class BiotGap(BiotPoint):
    """Compute flux interaction for a series of discrete gap probes."""

    attrs: list[str] = field(default_factory=lambda: ['Psi'])

    @property
    def boundary(self):
        """Return first wall boundary."""
        return self.Loc['plasma', 'poly'][0].boundary

    def solve(self, points, angle, length=1, resolution=):
        """Solve linear gap flux probes."""

        self.data.coords['xo'] = points[:, 0]
        self.data.coords['zo'] = points[:, 1]
        self.data.coords['angle'] = angle



        super().solve(np.c_[sample['radius'], sample['height']])

    def plot(self, axes=None, **kwargs):
        """Plot wall-gap probes."""
        if len(self.data) == 0:
            return
        self.axes = axes
        kwargs = dict(marker=None, linestyle='-', ms=4, color='gray',
                      linewidth=1.5) | kwargs
        self.axes.plot(self.data.x, self.data.z, **kwargs)
        if wallflux:
            super().plot(axes=axes)
