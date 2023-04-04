"""Manage biot calculation for wall contours."""
from dataclasses import dataclass, field

import numpy as np

from nova.biot.point import Point
from nova.biot.field import Sample
from nova.biot.limiter import Limiter

# Limiter,


@dataclass
class PlasmaWall(Limiter, Point):
    """Compute interaction for a series of discrete points."""

    attrs: list[str] = field(default_factory=lambda: ['Psi'])

    def __post_init__(self):
        """Initialize wallflux version."""
        super().__post_init__()
        self.version['limitflux'] = None

    @property
    def plasma_polarity(self):
        """Return plasma polarity."""
        return np.sign(self.saloc['Ic'][self.plasma_index])

    def check_limiter(self):
        """Check validity of upstream data -> update limiter flux."""
        if (version := self.aloc_hash['Ic']) != self.version['limitflux'] or \
                self.version['Psi'] != self.subframe.version['nturn']:
            self.update_wall(self.psi, self.plasma_polarity)
            self.version['limitflux'] = version

    def __getattribute__(self, attr):
        """Extend getattribute to intercept field null data access."""
        if attr == 'data_w':
            self.check_limiter()
        return super().__getattribute__(attr)

    @property
    def boundary(self):
        """Return first wall boundary."""
        return self.Loc['plasma', 'poly'][0].boundary

    def solve(self):
        """Solve Biot wall-pannel nodes with a delta subpannel spacing."""
        sample = Sample(self.boundary, delta=0.1)
        super().solve(np.c_[sample['radius'], sample['height']])

    def plot(self, axes=None, limitflux=False, **kwargs):
        """Plot wall pannels."""
        if len(self.data) == 0:
            return
        self.axes = axes
        kwargs = dict(marker=None, linestyle='-', ms=4, color='gray',
                      linewidth=1.5) | kwargs
        self.axes.plot(self.data.x, self.data.z, **kwargs)
        if limitflux:
            super().plot(axes=axes)
