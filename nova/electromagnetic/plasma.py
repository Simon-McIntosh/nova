"""Forward free-boundary equilibrium solver."""
from dataclasses import dataclass, field
from functools import cached_property

import descartes
import numba
import numpy as np
import scipy.spatial

from nova.database.netcdf import netCDF
from nova.electromagnetic.biotplasmagrid import BiotPlasmaGrid
from nova.electromagnetic.biotplasmaboundary import BiotPlasmaBoundary
from nova.electromagnetic.framesetloc import FrameSetLoc
from nova.electromagnetic.polyplot import Axes
from nova.geometry.polygon import Polygon
from nova.geometry.pointloop import PointLoop


@dataclass
class Plasma(Axes, netCDF, FrameSetLoc):
    """Set plasma separatix, ionize plasma filaments."""

    name: str = 'plasma'
    grid: BiotPlasmaGrid = field(repr=False, default=None)
    boundary: BiotPlasmaBoundary = field(repr=False, default=None)

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

    @property
    def psi(self):
        """Return grid an boundary psi."""
        return np.append(self.grid.psi, self.boundary.psi)

    @property
    def polarity(self):
        """Return plasma polarity."""
        return np.sign(self.sloc['Plasma', 'Ic'])

    @cached_property
    def pointloop(self):
        """Return pointloop instance, used to check loop membership."""
        if self.sloc['plasma'].sum() == 0:
            raise AttributeError('No plasma filaments found.')
        return PointLoop(self.loc['plasma', ['x', 'z']].to_numpy())

    @property
    def separatrix(self):
        """Return plasma separatrix, the convex hull of active filaments."""
        index = self.loc['plasma', 'nturn'] > 0
        points = self.loc['plasma', ['x', 'z']][index].values
        hull = scipy.spatial.ConvexHull(points)
        vertices = np.append(hull.vertices, hull.vertices[0])
        return points[vertices]

    @separatrix.setter
    def separatrix(self, loop):
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

    def update(self, psi):
        """Update plasma seperatrix."""
        psi_grid = psi[:self.grid.target_number]
        self.grid.update_null(psi_grid)

        #print(self.grid.x_psi.min())
        psi_boundary = psi[-self.boundary.target_number:]
        #s_psi = np.min([psi_boundary.min(), self.grid.x_psi[0]])
        try:
            s_psi = self.grid.x_psi[0]
            print(s_psi)
        except IndexError:
            s_psi = psi_boundary.min()


        plasma = self.aloc['plasma']
        ionize = self.aloc['ionize']
        nturn = self.aloc['nturn']
        area = self.aloc['area']

        ionize[plasma] = (psi_grid < s_psi) & \
            (self.aloc['z'][plasma] > -2.5)
        self._update_nturn(plasma, ionize, nturn, area)
        self.update_aloc_hash('nturn')

    def plot(self, turns=False):
        """Plot separatirx as polygon patch."""
        if turns:
            self.subframe.polyplot('plasma')
        poly = Polygon(self.separatrix).poly
        if not poly.is_empty:
            self.axes.add_patch(descartes.PolygonPatch(
                poly.__geo_interface__,
                facecolor='C4', alpha=0.75, linewidth=0, zorder=-10))
        self.boundary.plot()
        self.grid.plot()


'''

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
'''
