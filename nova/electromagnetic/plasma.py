"""Manage plasma attributes."""
from dataclasses import dataclass, field

import descartes
import numba
import numpy as np
import numpy.typing as npt

from nova.electromagnetic.framesetloc import FrameSetLoc
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

    number: int = field(init=False, default=0)
    loop: npt.ArrayLike = field(init=False, default=None, repr=False)
    #boundary: Polygon = field(init=False, repr=False, default=None)
    #separatrix: Polygon = field(init=False, repr=False, default=None)
    index: npt.ArrayLike = field(init=False, repr=False, default=None)
    points: npt.ArrayLike = field(init=False, repr=False, default=None)
    ionize: npt.ArrayLike = field(init=False, repr=False, default=None)
    nturn: npt.ArrayLike = field(init=False, repr=False, default=None)
    area: npt.ArrayLike = field(init=False, repr=False, default=None)

    def __post_init__(self):
        """Update subframe metadata."""
        self.subframe.metaframe.metadata = \
            {'additional': ['ionize'],
             'array': ['plasma', 'ionize', 'area', 'nturn']}
        self.subframe.update_columns()

    def __len__(self):
        """Return number of plasma filaments."""
        return self.index.sum()

    def __str__(self):
        """Return string representation of plasma subframe."""
        return self.loc['ionize', ['x', 'z', 'section', 'area',
                                   'Ic', 'It', 'nturn']].__str__()

    def insert(self, *args, required=None, iloc=None, **additional):
        """Store plasma index and plasma boundary and generate STR tree."""
        super().insert(*args, required=None, iloc=None, **additional)
        self.normalize()
        self.generate()

    def normalize(self):
        """Normalize plasma turn number for multiframe plasmas."""
        if self.sloc['plasma'].sum() == 1:
            return
        self.linkframe(self.Loc['plasma', :].index.tolist())
        self.Loc['plasma', 'nturn'] = \
            self.Loc['plasma', 'area'] / np.sum(self.Loc['plasma', 'area'])
        self.loc['plasma', 'nturn'] = \
            self.loc['plasma', 'area'] / np.sum(self.loc['plasma', 'area'])

    def generate(self):
        """Generate plasma attributes, build STR tree."""
        if self.loc['plasma'].sum() > 0:
            self.index = self.loc['plasma']
            self.points = self.loc['plasma', ['x', 'z']].to_numpy(copy=True)
            self.ionize = self.loc['ionize']
            self.nturn = self.loc['nturn']
            self.area = self.loc['area']

    @property
    def boundary(self) -> Polygon:
        """Return vessel boundary."""
        return Polygon(self.Loc['plasma', 'poly'][0])

    @property
    def separatrix(self):
        """Manage plasma separatrix."""
        return Polygon(self.loop).poly.intersection(self.boundary.poly)

    @separatrix.setter
    def separatrix(self, loop):
        self.loop = loop
        self.version = loop

    @property
    def version(self):
        """Manage unique separtrix identifier - store id in metaframe data."""
        return self.subframe.version['plasma']

    @version.setter
    def version(self, loop: npt.ArrayLike):
        self.subframe.version['plasma'] = id(loop)

    def update(self, loop):
        """
        Update plasma separatrix.

        Updates coil and subcoil geometries. Ionizes subcoil plasma filaments.
        Sets plasma update turn and current flags to True.

        Parameters
        ----------
        loop : array-like (n, 2), Polygon, dict[str, list[float]], list[float]
            Bounding loop.

        """
        try:
            ionize = inpoly.polymultipoint(self.points, loop)
        except numba.TypingError:
            loop = Polygon(loop).boundary
            ionize = inpoly.polymultipoint(self.points, loop)
        self.separatrix = loop
        self.ionize[self.index] = ionize
        self.nturn[self.index] = 0
        area = self.area[self.ionize]
        self.nturn[self.ionize] = area / np.sum(area)

    def plot(self, axes=None, boundary=True):
        """Plot plasma boundary and separatrix."""
        self.axes = axes
        if self.separatrix is not None:
            self.axes.add_patch(descartes.PolygonPatch(
                self.separatrix.__geo_interface__,
                facecolor='C4', alpha=0.75, linewidth=0.5, zorder=-10))
        if boundary:
            self.boundary.plot_boundary(self.axes, color='gray')
        plt.axis('equal')
        plt.axis('off')

    @property
    def polarity(self):
        """Return plasma polarity."""
        return np.sign(self.sloc['Plasma', 'Ic'])
