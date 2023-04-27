"""Forward free-boundary equilibrium solver."""
from dataclasses import dataclass, field
from functools import cached_property

from descartes import PolygonPatch
import numba
import numpy as np
from scipy.constants import mu_0
from scipy.interpolate import interp1d
import scipy.spatial

from nova.database.netcdf import netCDF
from nova.biot.error import PlasmaTopologyError
from nova.biot.levelset import LevelSet
from nova.biot.plasmagrid import PlasmaGrid
from nova.biot.plasmawall import PlasmaWall
from nova.frame.baseplot import Plot
from nova.frame.framesetloc import FrameSetLoc
from nova.geometry.polygon import Polygon
from nova.geometry.pointloop import PointLoop
from nova.geometry.select import bisect, bisect_right
from nova.geometry.separatrix import LCFS


@dataclass
class Plasma(Plot, netCDF, FrameSetLoc):
    """Set plasma separatix, ionize plasma filaments."""

    name: str = 'plasma'
    grid: PlasmaGrid = field(repr=False, default_factory=PlasmaGrid)
    wall: PlasmaWall = field(repr=False, default_factory=PlasmaWall)
    levelset: LevelSet = field(repr=False, default_factory=LevelSet)
    lcfs: LCFS | None = field(init=False, default=None)

    def __post_init__(self):
        """Update subframe metadata."""
        self.subframe.metaframe.metadata = \
            {'additional': ['plasma', 'ionize', 'area', 'nturn'],
             'array': ['plasma', 'ionize', 'area', 'nturn', 'x', 'z']}
        self.subframe.update_columns()
        super().__post_init__()
        self.version['lcfs'] = None
        self.version['psi_o'] = None
        self.version['psi_x'] = None
        self.version['psi_w'] = None

    def __len__(self):
        """Return number of plasma filaments."""
        return self.aloc['plasma'].sum()

    def __str__(self):
        """Return string representation of plasma subframe."""
        return self.loc['ionize', ['x', 'z', 'section', 'area',
                                   'Ic', 'It', 'nturn']].__str__()

    def solve(self, boundary=None):
        """Solve interaction matricies across plasma grid."""
        self.wall.solve(boundary)
        self.grid.solve()
        self.levelset.solve()

    def update_lcfs(self):
        """Update last closed flux surface."""
        if len(self.levelset) == 0:
            raise RuntimeError('solve levelset - nlevelset is None')
        points = self.levelset(self.psi_boundary)
        mask = self.x_mask(points[:, 1])
        self.lcfs = LCFS(points[mask])

    def _check_lcfs(self):
        """Check validity of upstream data, update wall flux if nessisary."""
        if (version := self.grid.version['fieldnull']) is None or \
                version != self.version['lcfs']:
            self.update_lcfs()
            self.version['lcfs'] = version

    def _check_psi_axis(self):
        """Check validity of psi_axis attribute."""
        if (version := self.grid.version['fieldnull']) is None or \
                version != self.version['psi_o']:
            self._clear_cache(['psi_axis'])
            self.version['psi_o'] = version

    def _check_psi_x(self):
        """Check validity of psi_x attribute."""
        if (version := self.grid.version['fieldnull']) is None or \
                version != self.version['psi_x']:
            self._clear_cache(['psi_x', 'psi_boundary'])
            self.version['psi_x'] = version

    def _check_psi_w(self):
        """Check validity of psi_w attribute."""
        if (version := self.wall.version['limitflux']) is None or \
                version != self.version['psi_w']:
            self._clear_cache(['psi_w', 'psi_boundary'])
            self.version['psi_w'] = version

    def __getattribute__(self, attr):
        """Extend getattribute to intercept grid and wall data access."""
        match attr:
            case 'lcfs':
                self._check_lcfs()
            case 'psi_axis':
                self._check_psi_axis()
            case 'psi_x' | 'psi_boundary':
                self._check_psi_x()
            case 'psi_w' | 'psi_boundary':
                self._check_psi_w()
        return super().__getattribute__(attr)

    @property
    def i_plasma(self):
        """Return total plasma current."""
        return self.saloc['plasma', 'Ic'][0]

    @property
    def li_3(self):
        """Return normalized plasma inductance."""
        filament_volume = self.aloc['ionize', 'volume']
        volume = np.sum(filament_volume)
        poloidal_field = self.grid.bp[self.aloc['plasma', 'ionize']]
        surface = np.sum(poloidal_field**2 * filament_volume) / volume
        # boundary = (mu_0 * self.i_plasma / self.lcfs.length)**2
        radius = 6.2 # self.lcfs.geometric_radius
        boundary = (mu_0 * self.i_plasma)**2 * radius / (2*volume)
        return surface / boundary

    @property
    def x_point_index(self):
        """Return x-point index for plasma separatrix."""
        if self.grid.x_point_number == 0:
            raise PlasmaTopologyError('no x-points within first wall')
        return np.argmax(self.polarity*(self.grid.x_psi - self.psi_axis))

    @property
    def x_point(self):
        """Return coordinates of primary x-point."""
        return self.grid.x_points[self.x_point_index]

    @property
    def o_point(self):
        """Return o-point coordinates."""
        return self.grid.o_points[0]

    @cached_property
    def psi_axis(self):
        """Return on-axis poloidal flux."""
        if self.grid.o_point_number > 1:
            raise IndexError('multiple field nulls found within firstwall\n'
                             f'{self.grid.data_o}')
        return self.grid.o_psi[0]

    @cached_property
    def psi_x(self):
        """Return poloidal flux at primary x-point."""
        if self.grid.x_point_number == 1:
            return self.grid.x_psi[0]
        return self.grid.x_psi[self.x_point_index]

    @cached_property
    def psi_w(self):
        """Return poloidal flux at limiter point."""
        return self.wall.w_psi

    @cached_property
    def psi_boundary(self):
        """Return boundary poloidal flux."""
        if self.grid.x_point_number == 0:
            return self.psi_w
        if self.polarity < 0:
            return np.min([self.psi_x, self.psi_w])
        return np.max([self.psi_x, self.psi_x])

    @cached_property
    def psi_index(self):
        """Return plasma wall number."""
        return np.cumsum([self.grid.number, self.wall.number])

    @property
    def psi(self):
        """Manage concatenated array of grid and wall flux values."""
        return np.r_[self.grid.psi, self.wall.psi]

    @psi.setter
    def psi(self, psi):
        self.grid['psi'] = psi[:self.psi_index[0]]
        self.wall['psi'] = psi[slice(*self.psi_index[0:2])]

    @property
    def psi_norm(self):
        """Return normalized grid flux."""
        psi_axis = self.psi_axis
        psi_boundary = self.psi_boundary
        return (self.grid.psi - self.psi_axis) / (psi_boundary - psi_axis)

    @cached_property
    def index(self):
        """Return plasma frame index."""
        return self.plasma_index

    @cached_property
    def _slice(self):
        """Return plasma filament slice."""
        start = bisect(self.aloc['plasma'], True)
        number = bisect_right(~self.aloc['plasma'][start:], False)
        if number != np.sum(self.aloc['plasma']):
            raise IndexError('plasma filaments are non-contiguous.')
        return slice(start, start+number)

    @cached_property
    def _nturn(self):
        """Return a view of the plasma's nturn array."""
        return self.aloc['nturn'][self._slice]

    @cached_property
    def _ionize(self):
        """Return a view of the plasma's ionize array."""
        return self.aloc['ionize'][self._slice]

    @cached_property
    def _area(self):
        """Return a view of the plasma's area array."""
        return self.aloc['area'][self._slice]

    @property
    def nturn(self):
        """Manage ionized plasma turn attribute."""
        return self._nturn[self.ionize]

    @nturn.setter
    def nturn(self, nturn):
        self._nturn[self.ionize] = nturn
        self.update_aloc_hash('nturn')

    @property
    def ionize(self):
        """Manage plasma ionization property."""
        return self._ionize

    @ionize.setter
    def ionize(self, mask):
        self._nturn[:] = 0
        self._ionize[:] = mask
        self._clear_cache(['area'])

    @cached_property
    def area(self):
        """Return ionized indexed area."""
        return self._area[self._ionize]

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

    def x_mask(self, z_plasma: np.ndarray):
        """Return plasma filament x-mask."""
        mask = np.ones(len(z_plasma), dtype=bool)
        if self.grid.x_point_number == 0:
            return mask
        for x_point in self.grid.x_points:
            if x_point[1] < self.o_point[1]:
                mask &= z_plasma > x_point[1]
            else:
                mask &= z_plasma < x_point[1]
        return mask

    def ionize_mask(self, index):
        """Return plasma filament selection mask."""
        match index:
            case int(psi) | float(psi):
                z_plasma = self.aloc['plasma', 'z']
                mask = self.psi_mask(psi)
                try:
                    return mask & self.x_mask(z_plasma)
                except IndexError:
                    return mask
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
        convexhull = points[vertices]
        tangent = convexhull[1:] - convexhull[:-1]
        length = np.append(0, np.cumsum(np.linalg.norm(tangent, axis=1)))
        _length = np.linspace(0, length[-1], 250)
        return np.c_[interp1d(length, convexhull[:, 0], 'quadratic')(_length),
                     interp1d(length, convexhull[:, 1], 'quadratic')(_length)]

    @separatrix.setter
    def separatrix(self, index):
        """
        Update plasma separatrix.

        Updates coil and subcoil geometries. Ionizes subcoil plasma filaments.
        Sets plasma update turn and current flags to True.

        Parameters
        ----------
        index : array-like (n, 2), Polygon, dict[str, list[float]], list[float]
            Bounding loop.

        """
        try:
            mask = self.ionize_mask(index)
        except (AttributeError, StopIteration) as error:
            raise AttributeError('use coilset.firstwall.insert '
                                 'to define plasma rejoin') from error
        self.ionize = mask
        self.nturn = self.area / np.sum(self.area)

    def plot(self, turns=True, axes=None, **kwargs):
        """Plot separatirx as polygon patch."""
        self.axes = axes
        if turns:
            self.subframe.polyplot('plasma')
        else:
            poly = Polygon(self.separatrix).poly
            if not poly.is_empty:
                self.axes.add_patch(PolygonPatch(
                    poly.__geo_interface__,
                    facecolor='C4', alpha=0.75, linewidth=0, zorder=-10))
        levels = self.levelset.plot(**kwargs)
        if levels is None:
            self.grid.plot(**kwargs)
        self.wall.plot(limitflux=False)
