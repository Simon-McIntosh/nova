"""Manage plasma attributes."""
from dataclasses import dataclass, field

import numpy as np

from nova.electromagnetic.framesetloc import FrameSetLoc
from nova.electromagnetic.poloidalgrid import PoloidalGrid
from nova.electromagnetic.polyplot import Axes
from nova.geometry.polygon import Polygon
from nova.utilities.pyplot import plt


@dataclass
class PlasmaGrid(PoloidalGrid):
    """Mesh rejoin interior to firstwall."""

    turn: str = 'hexagon'
    tile: bool = field(init=False, default=True)
    required: list[str] = field(default_factory=lambda: ['x', 'z', 'dl', 'dt'])
    default: dict = field(init=False, default_factory=lambda: {
        'nturn': 1, 'part': 'plasma', 'name': 'Plasma', 'plasma': True,
        'active': True})

    def set_conditional_attributes(self):
        """Set conditional attrs for plasma grid."""
        self.ifthen('delta', -1, 'turn', 'rectangle')
        self.ifthen('turn', 'rectangle', 'tile', False)

    def insert(self, *required, iloc=None, **additional):
        """
        Extend PoloidalGrid.insert.

        Add plasma to coilset and generate bounding plasma grid.

        Plasma inserted into frame with subframe meshed accoriding
        to delta and trimmed to the plasma's boundary curve.

        """
        return super().insert(*required, iloc=iloc, **additional)


@dataclass
class FirstWall(Axes, PlasmaGrid, FrameSetLoc):
    """Set plasma separatix, ionize plasma filaments."""

    name: str = 'firstwall'

    def __post_init__(self):
        """Update subframe metadata."""
        self.subframe.metaframe.metadata = \
            {'additional': ['plasma', 'area', 'nturn'],
             'array': ['nturn']}
        self.subframe.update_columns()
        super().__post_init__()

    def insert(self, *args, required=None, iloc=None, **additional):
        """Insert plasma and update plasma nturn version (xxhash)."""
        super().insert(*args, required=None, iloc=None, **additional)
        if self.sloc['plasma'].sum() > 1:
            self.normalize_multiframe()
        self.update_aloc_hash('nturn')

    def normalize_multiframe(self):
        """Nnormalize turn number for multiframe plasmas."""
        self.linkframe(self.Loc['plasma', :].index.tolist())
        self.Loc['plasma', 'nturn'] = \
            self.Loc['plasma', 'area'] / np.sum(self.Loc['plasma', 'area'])
        self.loc['plasma', 'nturn'] = \
            self.loc['plasma', 'area'] / np.sum(self.loc['plasma', 'area'])

    @property
    def poly(self) -> Polygon:
        """Return firstwall polygon."""
        return self.Loc['plasma', 'poly'][0]

    def plot(self, axes=None, boundary=True):
        """Plot plasma boundary and separatrix."""
        self.axes = axes
        if boundary:
            self.poly.plot_boundary(self.axes, color='gray', lw=1.5)
        self.subframe.polyplot('plasma')
        plt.axis('equal')
        plt.axis('off')
