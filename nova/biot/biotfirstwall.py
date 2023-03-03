"""Manage biot calculation for wall contours."""
from dataclasses import dataclass, field

import numpy as np

from nova.biot.biotpoint import BiotPoint
from nova.biot.wallflux import WallFlux
from nova.geometry.polyframe import PolyFrame


@dataclass
class BiotFirstWall(WallFlux, BiotPoint):
    """Compute interaction for a series of discrete points."""

    attrs: list[str] = field(default_factory=lambda: ['Psi'])

    def __post_init__(self):
        """Initialize fieldnull version."""
        super().__post_init__()
        self.version['wallflux'] = None

    @property
    def plasma_polarity(self):
        """Return plasma polarity."""
        return np.sign(self.saloc['Ic'][self.plasma_index])

    def check_wall(self):
        """Check validity of upstream data, update wall flux if nessisary."""
        if (version := self.aloc_hash['Ic']) != self.version['wallflux'] or \
                self.version['Psi'] != self.subframe.version['nturn']:
            self.update_wall(self.psi, self.plasma_polarity)
            self.version['wallflux'] = version

    def __getattribute__(self, attr):
        """Extend getattribute to intercept field null data access."""
        if attr == 'data_w':
            self.check_wall()
        return super().__getattribute__(attr)

    @property
    def loop(self):
        """Return first wall loop."""
        try:
            return self.Loc['plasma', 'poly'][0].boundary
        except AttributeError:
            return PolyFrame.loads(self.Loc['plasma', 'poly'][0]).boundary

    def solve(self):
        """Solve Biot wall-pannel nodes and mid-points."""
        loop = self.loop
        self.data.coords['x'] = loop[:, 0]
        self.data.coords['z'] = loop[:, 1]
        firstwall = np.empty((2*len(loop)-1, 2))
        firstwall[::2] = loop
        firstwall[1::2] = (loop[:-1, :] + loop[1:, :]) / 2
        super().solve(firstwall)

    def plot(self, axes=None, wallflux=True, **kwargs):
        """Plot wall pannels."""
        if len(self.data) == 0:
            return
        self.axes = axes
        if wallflux:
            super().plot(axes=axes)
        kwargs = dict(marker=None, linestyle='-', ms=4, color='gray',
                      linewidth=1.5) | kwargs
        self.axes.plot(self.data.x, self.data.z, **kwargs)
