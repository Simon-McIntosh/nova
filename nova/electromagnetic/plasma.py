"""Manage plasma attributes."""
from dataclasses import dataclass, field

import descartes
import numba
import numpy as np
import numpy.typing as npt

from nova.electromagnetic.framesetloc import FrameSetLoc, ArrayLocIndexer
from nova.electromagnetic.poloidalgrid import PoloidalGrid
from nova.electromagnetic.polyplot import Axes
from nova.geometry import inpoly
from nova.geometry.polygon import Polygon
from nova.utilities.pyplot import plt


# self.biot_instances = ['plasmafilament', 'plasmagrid']

'''
self.plasmagrid.generate_grid(**kwargs)
grid_factor = self.dPlasma/self.plasmagrid.dx
# self._add_vertical_stabilization_coils()
self.plasmagrid.cluster_factor = 1.5*grid_factor
self.plasmafilament.add_plasma()
'''


@dataclass
class PlasmaGrid(PoloidalGrid):
    """Grid plasma region."""

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
class Plasma(PlasmaGrid, FrameSetLoc, Axes):
    """Set plasma separatix, ionize plasma filaments."""

    loop: npt.ArrayLike = field(init=False, default=None, repr=False)
    _aloc: ArrayLocIndexer = field(init=False)

    def __post_init__(self):
        """Update subframe metadata."""
        self.subframe.metaframe.metadata = \
            {'additional': ['plasma', 'ionize', 'area', 'nturn'],
             'array': ['plasma', 'ionize', 'area', 'nturn', 'x', 'z']}
        self.subframe.update_columns()
        print(self.subframe)
        self._update_aloc()

    def _update_aloc(self):
        """Update array loc indexer."""
        self._aloc = self.aloc  # initialize ArrayLocIndexer

    def __getattr__(self, attr: str) -> npt.ArrayLike:
        """Access data array attributes."""
        try:
            return self._aloc[attr]
        except KeyError as err:
            raise KeyError(f'attr <{attr}> not available in {self._aloc}') \
                from err

    def __len__(self):
        """Return number of plasma filaments."""
        return self.index.sum()

    def __str__(self):
        """Return string representation of plasma subframe."""
        return self.loc['ionize', ['x', 'z', 'section', 'area',
                                   'Ic', 'It', 'nturn']].__str__()

    def insert(self, *args, required=None, iloc=None, **additional):
        """Insert plasma, normalize turn number for multiframe plasmas."""
        super().insert(*args, required=None, iloc=None, **additional)
        if self.sloc['plasma'].sum() == 1:
            return
        self.linkframe(self.Loc['plasma', :].index.tolist())
        self.Loc['plasma', 'nturn'] = \
            self.Loc['plasma', 'area'] / np.sum(self.Loc['plasma', 'area'])
        self.loc['plasma', 'nturn'] = \
            self.loc['plasma', 'area'] / np.sum(self.loc['plasma', 'area'])

    @property
    def plasma_version(self):
        """Manage unique separtrix identifier - store id in metaframe data."""
        return self.subframe.version['plasma']

    @property
    def boundary(self) -> Polygon:
        """Return vessel boundary."""
        return Polygon(self.Loc['plasma', 'poly'][0])

    @property
    def separatrix(self):
        """Return input plasma separatrix trimmed to first wall."""
        return Polygon(self.loop).poly.intersection(self.boundary.poly)

    def update_separatrix(self, loop):
        """
        Update plasma separatrix.

        Updates coil and subcoil geometries. Ionizes subcoil plasma filaments.
        Sets plasma update turn and current flags to True.

        Parameters
        ----------
        loop : array-like (n, 2), Polygon, dict[str, list[float]], list[float]
            Bounding loop.

        """
        if self._aloc.version != self.subframe.version['index']:
            self._update_aloc()
        points = np.c_[self.x, self.z][self.plasma]
        try:
            ionize = inpoly.polymultipoint(points, loop)
        except numba.TypingError:
            loop = Polygon(loop).boundary
            ionize = inpoly.polymultipoint(points, loop)
        self.loop = loop
        self.subframe.version['plasma'] = id(loop)
        self.ionize[self.plasma] = ionize
        self.nturn[self.plasma] = 0
        area = self.area[self.ionize]
        self.nturn[self.ionize] = area / np.sum(area)

    def plot(self, axes=None, boundary=True):
        """Plot plasma boundary and separatrix."""
        self.axes = axes
        if (poly := self.separatrix) is not None:
            self.axes.add_patch(descartes.PolygonPatch(
                poly.__geo_interface__,
                facecolor='C4', alpha=0.75, linewidth=0.5, zorder=-10))
        if boundary:
            self.boundary.plot_boundary(self.axes, color='gray')
        plt.axis('equal')
        plt.axis('off')

    @property
    def polarity(self):
        """Return plasma polarity."""
        return np.sign(self.sloc['Plasma', 'Ic'])
