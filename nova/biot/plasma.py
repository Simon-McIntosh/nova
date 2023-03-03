"""Forward free-boundary equilibrium solver."""
from dataclasses import dataclass, field
from functools import cached_property

import descartes
import numba
import numpy as np
import scipy.spatial

from nova.database.netcdf import netCDF
from nova.biot.biotfirstwall import BiotFirstWall
from nova.biot.biotplasmagrid import BiotPlasmaGrid
from nova.frame.baseplot import Plot
from nova.frame.framesetloc import FrameSetLoc
from nova.geometry.polygon import Polygon
from nova.geometry.pointloop import PointLoop


@numba.njit
def update_nturn(inloop, plasma, ionize, nturn, area):
    """Update plasma turns."""
    ionize[plasma] = inloop
    nturn[plasma] = 0
    ionize_area = area[ionize]
    nturn[ionize] = ionize_area / np.sum(ionize_area)


@dataclass
class Plasma(Plot, netCDF, FrameSetLoc):
    """Set plasma separatix, ionize plasma filaments."""

    name: str = 'plasma'
    grid: BiotPlasmaGrid = field(repr=False, default_factory=BiotPlasmaGrid)
    wall: BiotFirstWall = field(repr=False, default_factory=BiotFirstWall)

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

    def solve(self):
        """Solve interaction matricies across plasma grid."""
        self.wall.solve()
        self.grid.solve()

    @property
    def psi_axis(self):
        """Return on-axis poloidal flux."""
        if self.grid.o_point_number > 1:
            raise IndexError('multiple field nulls found within firstwall\n'
                             f'{self.grid.data_o}')
        return self.grid.o_psi[0]

    @property
    def boundary_index(self):
        """Return x-point index for plasma boundary."""
        return np.argmin(abs(self.grid.x_psi - self.psi_axis))

    @property
    def x_point(self):
        """Return coordinates of primary x-point."""
        return self.grid.x_points[self.boundary_index]

    @property
    def psi_boundary(self):
        """Return boundary poloidal flux."""
        if self.grid.x_point_number == 1:
            return self.grid.x_psi[0]
        return self.grid.x_psi[self.boundary_index]

    @property
    def psi(self):
        """Return concatenated array of grid and boundary psi values."""
        return np.append(self.grid.psi, self.boundary.psi)

    @cached_property
    def index(self):
        """Return plasma index."""
        return self.plasma_index

    @property
    def polarity(self):
        """Return plasma polarity."""
        return np.sign(self.saloc['Ic'][self.index])

    @cached_property
    def pointloop(self):
        """Return pointloop instance, used to check loop membership."""
        if self.saloc['plasma'].sum() == 0:
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
        if self.saloc['plasma'].sum() == 0:
            return
        try:
            inloop = self.pointloop.update(loop)
        except numba.TypingError:
            loop = Polygon(loop).boundary
            inloop = self.pointloop.update(loop)
        update_nturn(inloop, self.aloc['plasma'], self.aloc['ionize'],
                     self.aloc['nturn'], self.aloc['area'])
        self.update_aloc_hash('nturn')

    @property
    def nturn(self):
        """Manage plasma turns."""
        plasma = self.aloc['plasma']
        return self.aloc['nturn'][plasma]

    @nturn.setter
    def nturn(self, nturn):
        plasma = self.aloc['plasma']
        self.aloc['nturn'][plasma] = nturn
        self.update_aloc_hash('nturn')

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
        limit = np.argmin(self.wall.psi)

        self.grid.plot(**kwargs)
        self.wall.plot()


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
