"""Manage plasma attributes."""
from dataclasses import dataclass, field
from functools import cached_property
from typing import ClassVar

import descartes
import numba
import numpy as np
import numpy.typing as npt
import pyvista

import xarray

from nova.database.netcdf import netCDF
from nova.electromagnetic.framesetloc import FrameSetLoc
from nova.electromagnetic.plasmaboundary import PlasmaBoundary
from nova.electromagnetic.plasmagrid import PlasmaGrid
from nova.electromagnetic.poloidalgrid import PoloidalGrid
from nova.electromagnetic.polyplot import Axes
from nova.geometry.polygon import Polygon
from nova.geometry.pointloop import PointLoop
from nova.utilities.pyplot import plt

from numba import njit


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
class Plasma(Axes, netCDF, MeshPlasma, FrameSetLoc):
    """Set plasma separatix, ionize plasma filaments."""

    name: str = 'plasma'
    grid: PlasmaGrid = field(repr=False, default=None)
    boundary: PlasmaBoundary = field(repr=False, default=None)
    loop: npt.ArrayLike = field(init=False, default=None, repr=False)

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

    def store(self, filename: str, path=None):
        """Extend netCDF.store, store data as netCDF in hdf5 file."""
        self.data = xarray.Dataset()
        self.data['loop_coorinates'] = ['x', 'z']
        self.data['loop'] = ('loop_index', 'loop_coorinates'), self.loop
        super().store(filename, path)

    def load(self, filename: str, path=None):
        """Extend netCDF.load, load data from hdf5."""
        super().load(filename, path)
        self.loop = self.data['loop'].data
        return self

    @cached_property
    def pointloop(self):
        """Return pointloop instance, used to check loop membership."""
        if self.sloc['plasma'].sum() == 0:
            raise AttributeError('No plasma filaments found.')
        return PointLoop(self.loc['plasma', ['x', 'z']].to_numpy())

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
    def firstwall(self) -> Polygon:
        """Return vessel boundary."""
        return self.Loc['plasma', 'poly'][0]

    @property
    def separatrix(self):
        """Return input plasma separatrix trimmed to first wall."""
        if self.loop is None:
            self.update_separatrix(self.firstwall)
        return Polygon(self.loop).poly.intersection(self.firstwall.poly)

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
        self.update_loc_indexer()
        if self.sloc['plasma'].sum() == 0:
            return
        try:
            inloop = self.pointloop.update(loop)
        except numba.TypingError:
            loop = Polygon(loop).boundary
            inloop = self.pointloop.update(loop)
        self.loop = loop
        self.subframe.version['plasma'] = id(loop)
        plasma = self.aloc['plasma']
        ionize = self.aloc['ionize']
        nturn = self.aloc['nturn']
        area = self.aloc['area']
        ionize[plasma] = inloop
        self._update_nturn(plasma, ionize, nturn, area)

    @staticmethod
    @njit
    def _update_nturn(plasma, ionize, nturn, area):
        nturn[plasma] = 0
        ionize_area = area[ionize]
        nturn[ionize] = ionize_area / np.sum(ionize_area)

    def plot(self, axes=None, boundary=True):
        """Plot plasma boundary and separatrix."""
        self.axes = axes
        if (poly := self.separatrix) is not None and not poly.is_empty:
            self.axes.add_patch(descartes.PolygonPatch(
                poly.__geo_interface__,
                facecolor='C4', alpha=0.75, linewidth=0.5, zorder=-10))
        if boundary:
            self.firstwall.plot_boundary(self.axes, color='gray', lw=1.5)
        plt.axis('equal')
        plt.axis('off')

    @property
    def polarity(self):
        """Return plasma polarity."""
        return np.sign(self.sloc['Plasma', 'Ic'])
