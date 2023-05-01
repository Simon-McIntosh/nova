"""Manage biot calculation for wall contours."""
from dataclasses import dataclass, field

import numpy as np

from nova.biot.point import Point
from nova.biot.field import Sample
from nova.biot.limiter import Limiter


@dataclass
class PlasmaWall(Limiter, Point):
    """Compute interaction for a series of discrete points."""

    attrs: list[str] = field(default_factory=lambda: ['Psi'])

    def __post_init__(self):
        """Initialize limiter flux version."""
        super().__post_init__()
        self.version['limitflux'] = None

    def __getattribute__(self, attr):
        """Extend getattribute to intercept field null data access."""
        match attr:
            case 'data_w':
                self.check_limiter()
        return super().__getattribute__(attr)

    def check_limiter(self):
        """Check validity of upstream data -> update limiter flux."""
        self.check('psi')
        if self.version['limitflux'] != self.version['psi']:
            self.update_wall(self.psi, self.polarity)
            self.version['limitflux'] = self.version['psi']

    def __getitem__(self, attr):
        """Implement dict-like access to wall limiter flux attributes."""
        match attr:
            case 'w_point':
                return self.w_point
            case 'w_psi':
                return self.w_psi
        if hasattr(self, '__getitem__'):
            return super().__getitem__(attr)

    @property
    def boundary(self):
        """Return first wall boundary."""
        return self.Loc['plasma', 'poly'][0].boundary

    def solve(self, boundary=None):
        """Solve Biot wall-pannel nodes with a delta subpannel spacing."""
        if self.number is None:
            return
        if boundary is None:
            boundary = self.boundary
        sample = Sample(boundary, delta=-self.number)
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
