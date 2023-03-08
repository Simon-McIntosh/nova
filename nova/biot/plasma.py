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
from nova.biot.error import PlasmaTopologyError
from nova.biot.flux import Flux, LCFS
from nova.frame.baseplot import Plot
from nova.frame.framesetloc import FrameSetLoc
from nova.geometry.polygon import Polygon
from nova.geometry.pointloop import PointLoop


@numba.njit(cache=True)
def update_nturn(select, plasma, ionize, nturn, area):
    """Update plasma turns."""
    ionize[plasma] = select
    nturn[plasma] = 0
    ionize_area = area[ionize]
    nturn[ionize] = ionize_area / np.sum(ionize_area)


@dataclass
class Plasma(Plot, netCDF, FrameSetLoc):
    """Set plasma separatix, ionize plasma filaments."""

    name: str = 'plasma'
    grid: BiotPlasmaGrid = field(repr=False, default_factory=BiotPlasmaGrid)
    wall: BiotFirstWall = field(repr=False, default_factory=BiotFirstWall)
    flux: Flux = field(repr=False, default_factory=Flux)
    lcfs: LCFS | None = field(init=False, default=None)

    def __post_init__(self):
        """Update subframe metadata."""
        self.subframe.metaframe.metadata = \
            {'additional': ['plasma', 'ionize', 'area', 'nturn'],
             'array': ['plasma', 'ionize', 'area', 'nturn', 'x', 'z']}
        self.subframe.update_columns()
        super().__post_init__()
        self.version['lcfs'] = None

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
        self.flux.solve()

    def update_lcfs(self):
        """Update last closed flux surface."""
        self.lcfs = self.flux.lcfs(self.psi_boundary)

    def check_lcfs(self):
        """Check validity of upstream data, update wall flux if nessisary."""
        if (version := self.grid.version['fieldnull']) is None or \
                version != self.version['lcfs']:
            self.update_lcfs()
            self.version['lcfs'] = version

    def __getattribute__(self, attr):
        """Extend getattribute to intercept field null data access."""
        if attr == 'lcfs':
            self.check_lcfs()
        return super().__getattribute__(attr)

    @property
    def psi_axis(self):
        """Return on-axis poloidal flux."""
        if self.grid.o_point_number > 1:
            raise IndexError('multiple field nulls found within firstwall\n'
                             f'{self.grid.data_o}')
        return self.grid.o_psi[0]

    @property
    def x_point_index(self):
        """Return x-point index for plasma boundary."""
        if self.grid.x_point_number == 0:
            raise PlasmaTopologyError('no x-points within first wall')
        return np.argmin(abs(self.grid.x_psi - self.psi_axis))

    @property
    def x_point(self):
        """Return coordinates of primary x-point."""
        return self.grid.x_points[self.x_point_index]

    @property
    def o_point(self):
        """Return o-point coordinates."""
        return self.grid.o_points[0]

    @property
    def psi_boundary(self):
        """Return boundary poloidal flux."""
        if self.grid.x_point_number == 0:
            return self.wall.w_psi
        if self.grid.x_point_number == 1:
            return self.grid.x_psi[0]
        return self.grid.x_psi[self.x_point_index]

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

    def psi_mask(self, psi):
        """Return plasma filament psi-mask."""
        return self.polarity*self.grid.psi > self.polarity*psi

    def x_mask(self):
        """Return plasma filament x-mask."""
        mask = np.ones(self.aloc['plasma'].sum(), dtype=bool)
        if self.grid.x_point_number == 0:
            return mask
        z_plasma = self.aloc['plasma', 'z']
        for x_point in self.grid.x_points:
            if x_point[1] < self.o_point[1]:
                mask &= z_plasma > x_point[1]
            else:
                mask &= z_plasma < x_point[1]
        return mask

    def ionize(self, index):
        """Return plasma filament selection mask."""
        match index:
            case int(psi) | float(psi):
                return self.psi_mask(psi) & self.x_mask()
            case [int(psi) | float(psi), float(z_min)]:
                return self.psi_mask(psi) & self.aloc['plasma', 'z'] > z_min
            case [int(psi) | float(psi), float(z_min), float(z_max)]:
                z_plasma = self.aloc['plasma', 'z']
                return self.psi_mask(psi) & z_plasma > z_min & z_plasma < z_max
            case _:
                try:
                    return self.pointloop.update(index)
                except numba.TypingError:
                    index = Polygon(index).boundary
                    return self.pointloop.update(index)

    @property
    def separatrix(self):
        """Return plasma separatrix, the convex hull of active filaments."""
        index = self.loc['plasma', 'nturn'] > 0
        points = self.loc['plasma', ['x', 'z']][index].values
        hull = scipy.spatial.ConvexHull(points)
        vertices = np.append(hull.vertices, hull.vertices[0])
        return points[vertices]

    @separatrix.setter
    def separatrix(self, index):
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
        select = self.ionize(index)
        update_nturn(select, self.aloc['plasma'], self.aloc['ionize'],
                     self.aloc['nturn'], self.aloc['area'])
        self.update_aloc_hash('nturn')

    @property
    def nturn(self):
        """Manage plasma turns."""
        return self.aloc['plasma', 'nturn']

    @nturn.setter
    def nturn(self, nturn):
        self.aloc['plasma', 'nturn'] = nturn
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
        self.grid.plot(**kwargs)
        self.wall.plot()
