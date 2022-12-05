"""Forward free-boundary equilibrium solver."""
from dataclasses import dataclass, field
from functools import cached_property
from importlib import import_module

import descartes
import numba
import numpy as np

from nova.database.netcdf import netCDF
from nova.biot.biotplasmagrid import BiotPlasmaGrid
from nova.biot.biotplasmaboundary import BiotPlasmaBoundary
from nova.frame.baseplot import Plot
from nova.frame.framesetloc import FrameSetLoc
from nova.geometry.polygon import Polygon
from nova.geometry.pointloop import PointLoop


@dataclass
class Plasma(Plot, netCDF, FrameSetLoc):
    """Set plasma separatix, ionize plasma filaments."""

    name: str = 'plasma'
    grid: BiotPlasmaGrid = field(repr=False, default_factory=BiotPlasmaGrid)
    boundary: BiotPlasmaBoundary = field(repr=False,
                                         default_factory=BiotPlasmaBoundary)

    def __post_init__(self):
        """Update subframe metadata."""
        self.subframe.metaframe.metadata = \
            {'additional': ['plasma', 'ionize', 'area', 'nturn'],
             'array': ['plasma', 'ionize', 'area', 'nturn', 'x', 'z']}
        self.subframe.update_columns()
        super().__post_init__()

    def __len__(self):
        """Return number of plasma filaments."""
        return self.aloc['plasma'].sum()

    def __str__(self):
        """Return string representation of plasma subframe."""
        return self.loc['ionize', ['x', 'z', 'section', 'area',
                                   'Ic', 'It', 'nturn']].__str__()

    @property
    def psi(self):
        """Return concatenated array of grid and boundary psi values."""
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
        hull = import_module('scipy.spatial').ConvexHull(points)
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

    def __XXX_residual(self, Psi):

        self.grid.operator['Psi'].matrix[:, 115] = Psi
        self.grid.version['psi'] = None

        plasma = self.aloc['plasma']
        ionize = self.aloc['ionize']
        nturn = self.aloc['nturn']
        area = self.aloc['area']

        ionize[:] = 0
        ionize[plasma] = self.grid.psi < -50

        self._update_nturn(plasma, ionize, nturn, area)
        self.grid.operator['Psi'].update_turns(True)

        return Psi - self.grid.operator['Psi'].matrix[:, 115]

    def _XXX_residual(self, nturn):
        """Update plasma seperatrix."""

        '''
        psi_grid = psi[:self.grid.target_number]
        self.grid.update_null(psi_grid)

        #print(self.grid.x_psi.min())
        psi_boundary = psi[-self.boundary.target_number:]
        #s_psi = np.min([psi_boundary.min(), self.grid.x_psi[0]])
        '''

        '''
        try:
            s_psi = self.grid.x_psi[0]
        except IndexError:
            s_psi = self.boundary.psi.min()
        '''
        nturn /= sum(nturn)

        # update nturn
        plasma = self.aloc['plasma']
        self.aloc['nturn'][plasma] = nturn
        self.update_aloc_hash('nturn')


        # solve rhs

        psi = self.boundary.psi.min()
        '''
        if len(self.grid.x_psi) > 0:
            psi = self.grid.x_psi.min()
        else:
            psi = psi_boundary
        print(psi, [psi_boundary, self.grid.x_psi])
        '''

        self.aloc['ionize'] = 0
        self.aloc['ionize'][plasma] = self.grid.psi < psi
        self.aloc['nturn'][plasma] = 0
        ionize_area = self.aloc['area'][self.aloc['ionize']]
        self.aloc['nturn'][self.aloc['ionize']] = \
            ionize_area / np.sum(ionize_area)

        print(sum(nturn), sum(self.aloc['nturn'][self.aloc['ionize']]))

        #self._update_nturn(plasma, ionize,
        #                   self.aloc['nturn'], self.aloc['area'])
        self.update_aloc_hash('nturn')
        return nturn - self.aloc['nturn'][plasma]

    def plot(self, turns=True, **kwargs):
        """Plot separatirx as polygon patch."""
        if turns:
            self.subframe.polyplot('plasma')
        '''
        poly = Polygon(self.separatrix).poly
        if not poly.is_empty:
            self.axes.add_patch(descartes.PolygonPatch(
                poly.__geo_interface__,
                facecolor='C4', alpha=0.75, linewidth=0, zorder=-10))
        '''
        self.boundary.plot()
        self.grid.plot(**kwargs)


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
