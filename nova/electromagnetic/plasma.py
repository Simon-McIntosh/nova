"""Manage plasma attributes."""
from dataclasses import dataclass, field
from functools import cached_property
from typing import ClassVar

import numba
import numpy as np
import pyvista
import xarray

from nova.electromagnetic.framesetloc import FrameSetLoc
from nova.electromagnetic.plasmaboundary import PlasmaBoundary
from nova.electromagnetic.plasmagrid import PlasmaGrid
from nova.electromagnetic.poloidalgrid import PoloidalGrid
from nova.electromagnetic.polyplot import Axes
from nova.geometry.polygon import Polygon
from nova.geometry.pointloop import PointLoop
from nova.utilities.pyplot import plt


@dataclass
class PlasmaVTK(PlasmaGrid):
    """Extend PlasmaGrid dataset with VTK methods."""

    data: xarray.Dataset = field(repr=False, default_factory=xarray.Dataset)
    mesh: pyvista.PolyData = field(init=False, repr=False)
    classnames: ClassVar[list[str]] = ['PlasmaGrid', 'PlasmaVTK']

    def __post_init__(self):
        """Load biot dataset."""
        super().__post_init__()
        self.load_data()
        assert self.data.attrs['classname'] in self.classnames
        self.build_mesh()

    def build_mesh(self):
        """Build vtk mesh."""
        points = np.c_[self.data.x, np.zeros(self.data.dims['x']), self.data.z]
        faces = np.c_[np.full(self.data.dims['tri_index'], 3),
                      self.data.triangles]
        self.mesh = pyvista.PolyData(points, faces=faces)

    def plot(self, **kwargs):
        """Plot vtk mesh."""
        self.mesh['psi'] = self.psi
        kwargs = dict(color='purple', line_width=2,
                      render_lines_as_tubes=True) | kwargs
        plotter = pyvista.Plotter()
        plotter.add_mesh(self.mesh)
        plotter.add_mesh(self.mesh.contour(), **kwargs)
        plotter.show()


@dataclass
class MeshPlasma(PoloidalGrid):
    """Mesh plasma region."""

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
class Plasma(Axes, MeshPlasma, FrameSetLoc):
    """Set plasma separatix, ionize plasma filaments."""

    name: str = 'plasma'
    grid: PlasmaGrid = field(repr=False, default=None)
    boundary: PlasmaBoundary = field(repr=False, default=None)

    def __post_init__(self):
        """Update subframe metadata."""
        self.subframe.metaframe.metadata = \
            {'additional': ['plasma', 'ionize', 'area', 'nturn'],
             'array': ['plasma', 'ionize', 'area', 'nturn', 'x', 'z']}
        self.subframe.update_columns()
        super().__post_init__()

    def __len__(self):
        """Return number of plasma filaments."""
        return self.aloc.plasma.sum()

    def __str__(self):
        """Return string representation of plasma subframe."""
        return self.loc['ionize', ['x', 'z', 'section', 'area',
                                   'Ic', 'It', 'nturn']].__str__()

    @cached_property
    def pointloop(self):
        """Return pointloop instance, used to check loop membership."""
        if self.sloc['plasma'].sum() == 0:
            raise AttributeError('No plasma filaments found.')
        return PointLoop(self.loc['plasma', ['x', 'z']].to_numpy())

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
    def firstwall(self) -> Polygon:
        """Return vessel boundary."""
        return self.Loc['plasma', 'poly'][0]

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
        if self.sloc['plasma'].sum() == 0:
            return
        try:
            inloop = self.pointloop.update(loop)
        except numba.TypingError:
            loop = Polygon(loop).boundary
            inloop = self.pointloop.update(loop)
        self.loop = loop

        plasma = self.aloc['plasma']
        ionize = self.aloc['ionize']
        nturn = self.aloc['nturn']
        area = self.aloc['area']
        ionize[plasma] = inloop
        self._update_nturn(plasma, ionize, nturn, area)
        self.update_aloc_hash('nturn')

    @staticmethod
    @numba.njit
    def _update_nturn(plasma, ionize, nturn, area):
        nturn[plasma] = 0
        ionize_area = area[ionize]
        nturn[ionize] = ionize_area / np.sum(ionize_area)

    @property
    def polarity(self):
        """Return plasma polarity."""
        return np.sign(self.sloc['Plasma', 'Ic'])

    def plot(self, axes=None, boundary=True):
        """Plot plasma boundary and separatrix."""
        self.axes = axes
        if boundary:
            self.firstwall.plot_boundary(self.axes, color='gray', lw=1.5)
        self.subframe.polyplot('plasma')
        plt.axis('equal')
        plt.axis('off')
