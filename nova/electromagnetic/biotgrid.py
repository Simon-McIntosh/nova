"""Generate grids for BiotGrid methods."""
from dataclasses import dataclass, field, InitVar
from typing import Union

from matplotlib.collections import LineCollection
import numpy as np
import shapely.geometry
import xarray

from nova.electromagnetic.biotdata import BiotMatrix
from nova.electromagnetic.biotfilament import Biot
from nova.electromagnetic.coilset import CoilSet
from nova.electromagnetic.framelink import FrameLink
from nova.electromagnetic.framespace import FrameSpace
from nova.electromagnetic.polyplot import Axes


@dataclass
class GridCoord:
    """Manage grid coordinates."""

    start: float
    stop: float
    _num: int = field(init=False, repr=False, default=0)

    def __post_init__(self):
        """Calculate coordinate spacing."""
        self.delta = self.stop - self.start

    def __len__(self):
        """Return coordinate dimension."""
        return self.num

    def __call__(self):
        """Return coordinate point vector."""
        return np.linspace(self.start, self.stop, self.num)

    @property
    def limit(self):
        """Return coordinate limits."""
        return self.start, self.stop

    @property
    def num(self):
        """Manage coordinate number."""
        return self._num

    @num.setter
    def num(self, num):
        self._num = np.max([int(np.ceil(num)), 1])


@dataclass
class Grid(Axes):
    """Generate grid."""

    number: InitVar[int]
    limit: list[float]
    data: xarray.Dataset = field(init=False, repr=False)

    def __post_init__(self, number):
        """Build grid coordinates."""
        self.xcoord = GridCoord(*self.limit[:2])
        self.zcoord = GridCoord(*self.limit[2:])
        self.xcoord.num = self.xcoord.delta / np.sqrt(
            self.xcoord.delta*self.zcoord.delta / number)
        self.zcoord.num = number / self.xcoord.num
        self.data = xarray.Dataset(
            coords=dict(x=self.xcoord(), z=self.zcoord()))
        x2d, z2d = np.meshgrid(self.data.x, self.data.z, indexing='ij')
        self.data['x2d'] = (['x', 'z'], x2d)
        self.data['z2d'] = (['x', 'z'], z2d)

    def __len__(self):
        """Return grid number."""
        return self.xcoord.num * self.zcoord.num

    @property
    def shape(self):
        """Return grid shape."""
        return self.xcoord.num, self.zcoord.num

    def plot(self, axes=None, **kwargs):
        """Plot grid."""
        self.axes = axes  # set plot axes
        kwargs = {'linewidth': 0.4, 'color': 'gray',
                  'alpha': 0.5, 'zorder': -100} | kwargs
        for num, step in zip(self.shape, [1, -1]):
            lines = np.zeros((num, 2, 2))
            for i in range(2):
                index = tuple([slice(None), -i][::step])
                lines[:, i, 0] = self.data.x2d[index]
                lines[:, i, 1] = self.data.z2d[index]
            segments = LineCollection(lines, **kwargs)
            self.axes.add_collection(segments, autolim=True)
        self.axes.autoscale_view()


@dataclass
class Expand:
    """Calculate grid limit as a factor expansion about multipoly bounds."""

    frame: FrameLink
    xmin: float = 1e-12,
    fix_aspect: bool = False

    def __post_init__(self):
        """Extract multipolygon bounding box."""
        poly = shapely.geometry.MultiPolygon(self.frame.poly.to_list())
        self.limit = [*poly.bounds[::2], *poly.bounds[1::2]]
        self.xcoord = GridCoord(*self.limit[:2])
        self.zcoord = GridCoord(*self.limit[2:])

    def __call__(self, factor):
        """Return expanded limit."""
        delta_x, delta_z = self.xcoord.delta, self.zcoord.delta
        if not self.fix_aspect:
            delta_x = delta_z = np.mean([delta_x, delta_z])
        limit = self.limit + factor/2 * np.array([-delta_x, delta_x,
                                                  -delta_z, delta_z])
        if limit[0] < self.xmin:
            limit[0] = self.xmin
        return limit


@dataclass
class BiotGrid(Axes):
    """Compute interaction across grid."""

    frame: FrameSpace
    data: BiotMatrix = field(init=False, repr=False)

    def solve(self, number: int, limit: Union[float, list[float]]):
        """Solve Biot interaction across grid."""
        if isinstance(limit, (int, float)):
            limit = Expand(self.frame)(limit)
        grid = Grid(number, limit)
        target = dict(x=grid.data.x2d.values.flatten(),
                      z=grid.data.z2d.values.flatten())
        self.data = Biot(self.frame, target, reduce=[True, False],
                         columns=['Psi', 'Br', 'Bz']).data
        # insert grid data
        self.data.coords['x'] = grid.data.x
        self.data.coords['z'] = grid.data.z
        self.data.coords['x2d'] = (['x', 'z'], grid.data.x2d)
        self.data.coords['z2d'] = (['x', 'z'], grid.data.z2d)

    @property
    def shape(self):
        """Return grid shape."""
        return self.data.dims['x'], self.data.dims['z']

    def plot(self, axes=None):
        self.axes = axes

        Psi = np.dot(self.data.Psi, self.frame.subspace['Ic'])
        self.axes.contour(self.data.x, self.data.z,
                          Psi.reshape(*self.shape).T, 21)


if __name__ == '__main__':

    coilset = CoilSet(dcoil=-35, dplasma=-40)
    coilset.coil.insert(10, 0.5, 0.95, 0.95, section='hex', turn='r',
                        nturn=-0.8)
    coilset.coil.insert(10, -0.5, 0.95, 0.95, section='hex')
    coilset.coil.insert(11, 0, 0.95, 0.1, section='sk', nturn=-1.8)
    coilset.coil.insert(12, 0, 0.6, 0.9, section='r', turn='sk')
    coilset.plasma.insert({'ellip': [11.5, 0.8, 1.7, 0.4]})
    coilset.link(['Coil0', 'Plasma'], 2)

    biotgrid = BiotGrid(coilset.subframe)
    biotgrid.solve(1e3, 0.1)

    coilset.sloc['Ic'] = 1

    coilset.plot()
    biotgrid.plot()

    '''
    def generate(self, regen=False, **kwargs):
        """
        Generate grid for use as targets in Biot Savart calculations.

        Compare kwargs to current grid settings, update grid on-demand and
        return update flag

        Parameters
        ----------
        regen : bool
            Force grid regeneration.
        boundary : str
            Grid boundary flag ['expand_limit' or 'limit'].

        Returns
        -------
        regenerate_grid : bool
            Generation flag.

        """
        self.regen = regen
        self._set_boundary(**kwargs)
        self.assemble_biotset()
        grid_attributes = {}  # grid attributes
        for key in ['n', 'limit', 'expand', 'levels', 'nlevels']:
            attribute = kwargs.get(key, getattr(self, key))
            if key in ['n', 'nlevels']:
                attribute = int(attribute)
            grid_attributes[key] = attribute
        # calculate coilset limit
        grid_attributes['expand_limit'] = \
            self._get_expand_limit(grid_attributes['expand'])
        # switch to coilset limit if limit is None
        if grid_attributes['limit'] is None:
            self.grid_boundary = 'expand_limit'
        regenerate_grid = \
            not np.array_equal(grid_attributes[self.boundary],
                               self.grid_boundary) or \
            grid_attributes['expand'] != self.expand or \
            grid_attributes['n'] != self.n or self.regen
        if regenerate_grid:
            self._generate_grid(**grid_attributes)
            self.grid_polygon = shapely.geometry.box(
                *self.grid_boundary[::2], *self.grid_boundary[1::2])
        return regenerate_grid

    def _generate_grid(self):
        if self.n > 0:
            mg = MeshGrid(self.n, self.grid_boundary)  # set mesh
            self.n2d = [mg.nx, mg.nz]  # shape
            self.x, self.z = mg.x, mg.z  # axes
            # trace index interpolators
            self.x_index = interp1d(range(self.n2d[0]), self.x)
            self.z_index = interp1d(range(self.n2d[1]), self.z)
            # grid deltas
            self.dx = np.diff(self.grid_boundary[:2])[0] / (mg.nx - 1)
            self.dz = np.diff(self.grid_boundary[2:])[0] / (mg.nz - 1)
            # 2d coordinates
            self.x2d = mg.x2d
            self.z2d = mg.z2d
            self.target.drop_coil()  # clear target
            self.target.add_coil(self.x2d.flatten(), self.z2d.flatten(),
                                 name='Grid', delim='')
            self.assemble_biotset()
            self.update_biot = True

    @property
    def grid_boundary(self):
        """
        Manage grid bounding box based on self.boundary.

        Parameters
        ----------
        boundary : str
            Boundary flag ether 'limit' or 'expand_limit'.
            Set self.regen=True if boundary flag is changed.

        Raises
        ------
        IndexError
            Flag not in [limit, expand_limit].

        Returns
        -------
        limit : array_like, shape(4,)
            Grid bounding box [xmin, xmax, zmin, zmax]

        """
        if self.boundary == 'limit':
            return self.limit
        elif self.boundary == 'expand_limit':
            return self.expand_limit
        else:
            raise IndexError(f'boundary {self.boundary} not in '
                             '[limit, expand_limit]')

    @grid_boundary.setter
    def grid_boundary(self, boundary):
        if boundary != self.boundary:
            self.regen = True  # force update after boundary switch
        if boundary in ['limit', 'expand_limit']:
            self.boundary = boundary
        else:
            raise IndexError(f'boundary label {boundary} not in '
                             '[limit, expand_limit]')

    def _set_boundary(self, **kwargs):
        if 'boundary' in kwargs:  # set flag directly [limit, expand_limit]
            boundary = kwargs['boundary']
        elif 'expand' in kwargs:  # then expand (coilset limit)
            boundary = 'expand_limit'
        elif 'limit' in kwargs:  # then limit
            boundary = 'limit'
        else:  # defaults to coilset limit
            boundary = 'expand_limit'
        self.grid_boundary = boundary


    def _get_expand_limit(self, expand=None, xmin=1e-3,
                          exclude=['Zfb0', 'Zfb1']):
        if expand is None:
            expand = self.expand  # use default
        if self.source.empty:
            raise IndexError('source coilframe empty')
        index = np.full(self.source.nC, True)
        for coil in exclude:  # exclude coils (feedback pair)
            index[self.source.coil == coil] = False
        x, z, = self.source.x[index], self.source.z[index]
        dx, dz = self.source.dx[index], self.source.dz[index]
        limit = np.array([(x - dx/2).min(), (x + dx/2).max(),
                          (z - dz/2).min(), (z + dz/2).max()])
        return self._expand_limit(limit, expand, xmin)

    @staticmethod
    def _expand_limit(limit, expand, xmin, fix_aspect=False):
        dx, dz = np.diff(limit[:2])[0], np.diff(limit[2:])[0]
        if not fix_aspect:
            dx = dz = np.mean([dx, dz])
        limit += expand/2 * np.array([-dx, dx, -dz, dz])
        if limit[0] < xmin:
            limit[0] = xmin
        return limit



@dataclass
class BiotGrid:



    @property
    def grid_polygon(self):
        """
        Manage boundary polygon.

        Returns
        -------
        grid_polygon : shapely.polygon or None
            Grid bounding box.

        """
        return self._grid_polygon

    @grid_polygon.setter
    def grid_polygon(self, polygon):
        self._grid_polygon = polygon
        self._update_coil_center = True



    def plot_flux(self, ax=None, lw=1, color='lightgrey', **kwargs):
        """
        Plot constant flux contours.

        Parameters
        ----------
        ax : Axes, optional
            Plot axes.
        lw : float, optional
            line weight.
        color : str, optional
            line color. The default is 'lightgrey'.
        **kwargs : dict, optional
            Keyword arguments passed to ax.contour.

        Returns
        -------
        levels : QuadContourSet.levels
            Contour levels.

        """
        if self.n > 0:
            if ax is None:
                ax = plt.gca()
            levels = kwargs.get('levels', self.levels)
            if levels is None:
                levels = self.nlevels
            QuadContourSet = ax.contour(
                    self.x2d, self.z2d, self.Psi,
                    levels, colors=color, linestyles='-',
                    linewidths=lw,
                    alpha=0.9, zorder=4)
            if self.levels is None:
                self.levels = QuadContourSet.levels
            plt.axis('equal')
            return QuadContourSet.levels

    def plot_field(self, ax=None, **kwargs):
        """
        Generate field quiver plot.

        Returns
        -------
        None.

        """
        if self.n > 0:
            if ax is None:
                ax = plt.gca()
            levels = kwargs.get('levels', self.levels)
            if levels is None:
                levels = self.nlevels
            ax.contour(
                    self.x2d, self.z2d, self.B,
                    31, linestyles='-', alpha=0.9, zorder=4)


    '''

    '''
    """
    Grid interaction methods and data.

    Key Attributes
    --------------
        n : int
            Grid dimension.
        limit : array-like, shape(4,)
            Grid limits.
        expand : float
            Grid expansion beyond coils.
        nlevels : int
            Number of contour levels.
        boundary : str
            Limit boundary ['limit', 'expand_limit'].

    Derived Attributes
    ------------------
        n2d : array-like, shape(2,)
            As meshed dimension.
        x2d : array-like shape(*n2D)
            2D x-coordinates (radial).
        z2d : array-like shape(*n2D)
            2D z-coordinates.
        levels : array-like
            Contour levels.
        Psi : array-like
            Poloidal flux.
        Bx : array-like
            Radial field.
        Bz : array-like
            Vertical field.

    """

    _biot_attributes = ['n', 'n2d', 'limit', 'expand_limit', 'boundary',
                        'grid_polygon', 'expand', 'nlevels', 'levels',
                        'x', 'z', 'x_index', 'z_index', 'x2d', 'z2d',
                        'dx', 'dz', 'target']

    _default_biot_attributes = {'n': 1e4, 'expand': 0.05, 'nlevels': 51,
                                'boundary': 'coilset'}
    '''
