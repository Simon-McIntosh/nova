"""Generate grids for BiotGrid methods."""
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import shapely.geometry
import xarray

from nova.biot.biotframe import Target
from nova.biot.operate import Operate
from nova.biot.solve import Solve
from nova.biot.fieldnull import FieldNull
from nova.graphics.plot import Plot
from nova.frame.error import GridError
from nova.frame.framelink import FrameLink
from nova.geometry.pointloop import PointLoop
from nova.graphics.line import Chart


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
class Gridgen(Plot):
    """Generate rectangular 2d grid."""

    ngrid: int | None = field(default=None)
    limit: np.ndarray | None = field(default=None)
    xcoord: list[float] = field(init=False, repr=False)
    zcoord: list[float] = field(init=False, repr=False)
    data: xarray.Dataset = field(init=False, repr=False)

    def __post_init__(self):
        """Build grid coordinates."""
        self.xcoord, self.zcoord = self.generate()
        self.data = xarray.Dataset(
            coords=dict(x=self.xcoord, z=self.zcoord))
        x2d, z2d = np.meshgrid(self.data.x, self.data.z, indexing='ij')
        self.data['x2d'] = (['x', 'z'], x2d)
        self.data['z2d'] = (['x', 'z'], z2d)

    def generate(self):
        """Return grid coordinates."""
        if len(self.limit) == 2:  # grid coordinates
            xcoord, zcoord = self.limit
            self.ngrid = len(xcoord) * len(zcoord)
            self.limit = [xcoord[0], xcoord[-1], zcoord[0], zcoord[-1]]
            return xcoord, zcoord
        if len(self.limit) == 4:  # grid limits
            xgrid = GridCoord(*self.limit[:2])
            zgrid = GridCoord(*self.limit[2:])
            xgrid.num = xgrid.delta / np.sqrt(
                xgrid.delta*zgrid.delta / self.ngrid)
            zgrid.num = self.ngrid / xgrid.num
            self.ngrid = xgrid.num * zgrid.num
            return xgrid(), zgrid()
        raise IndexError(f'len(limit) {len(self.limit)} not in [2, 4]')

    def __len__(self):
        """Return grid resolution."""
        return len(self.xcoord) * len(self.zcoord)

    @property
    def shape(self):
        """Return grid shape."""
        return len(self.xcoord), len(self.zcoord)

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
            segments = self.mpl['LineCollection'](lines, **kwargs)
            self.axes.add_collection(segments, autolim=True)
        self.axes.autoscale_view()


@dataclass
class Expand:
    """Calculate grid limit as a factor expansion about multipoly bounds."""

    frame: FrameLink
    index: str | slice | np.ndarray = field(
        default_factory=lambda: slice(None))
    xmin: float = 1e-12
    fix_aspect: bool = False

    def __post_init__(self):
        """Extract multipolygon bounding box."""
        if isinstance(self.index, str):
            index = self.index
            self.index = getattr(self.frame, self.index)
            if sum(self.index) == 0:
                raise GridError(index)
        poly = shapely.geometry.MultiPolygon(
            [polygon.poly for polygon in self.frame.poly[self.index]])
        self.limit = np.array([*poly.bounds[::2], *poly.bounds[1::2]])
        self.xcoord = GridCoord(*self.limit[:2])
        self.zcoord = GridCoord(*self.limit[2:])

    def __call__(self, factor) -> np.ndarray:
        """Return expanded limit."""
        delta_x, delta_z = self.xcoord.delta, self.zcoord.delta
        if not self.fix_aspect:
            delta_x = delta_z = np.mean([delta_x, delta_z])
        limit = self.limit + factor/2 * np.array([-delta_x, delta_x,
                                                  -delta_z, delta_z])
        if limit[0] < self.xmin:
            limit[0] = self.xmin
        return limit


class BaseGrid(Chart, FieldNull, Operate):
    """Flux grid baseclass."""

    attrs: list[str] = field(default_factory=lambda: ['Br', 'Bz', 'Psi'])

    def __post_init__(self):
        """Initialize fieldnull version."""
        super().__post_init__()
        self.version['fieldnull'] = None

    def check_null(self):
        """Check validity of upstream data, update field null if nessisary."""
        self.check('psi')
        if self.version['fieldnull'] != self.version['psi']:
            self.update_null(self.psi_)
            self.version['fieldnull'] = self.version['psi']

    def __getattribute__(self, attr):
        """Extend getattribute to intercept field null data access."""
        if attr == 'data_x' or attr == 'data_o':
            self.check_null()
        return super().__getattribute__(attr)

    def plot_svd(self, **kwargs):
        """Plot influence of SVD reduction."""
        for svd, color, linestyle in zip([False, True], ['C7', 'C3'],
                                         ['solid', 'dashed']):
            self.update_turns('Psi', svd)
            kwargs |= dict(colors=color, linestyles=linestyle)
            self.plot(**kwargs)


@dataclass
class Grid(BaseGrid):
    """Compute interaction across grid."""

    def solve(self, number: int, limit: float | np.ndarray = 0,
              index: str | slice | np.ndarray = slice(None)):
        """Solve Biot interaction across grid."""
        with self.solve_biot(number) as number:
            if number is not None:
                if isinstance(limit, (int, float)):
                    limit = Expand(self.subframe, index)(limit)
                grid = Gridgen(number, limit)
                self.solve2d(grid.data.x2d.values, grid.data.z2d.values)

    def solve2d(self, x2d, z2d):
        """Solve interaction across rectangular grid."""
        target = Target(dict(x=x2d.flatten(), z=z2d.flatten()), label='Grid')
        self.data = Solve(self.subframe, target, reduce=[True, False],
                          name=self.name, attrs=self.attrs).data
        self.data.coords['x'] = x2d[:, 0]
        self.data.coords['z'] = z2d[0]
        self.data.coords['x2d'] = (['x', 'z'], x2d)
        self.data.coords['z2d'] = (['x', 'z'], z2d)

    @cached_property
    def pointloop(self):
        """Return pointloop instance, used to check loop membership."""
        points = np.array([self.data.x2d.data.flatten(),
                           self.data.z2d.data.flatten()]).T
        return PointLoop(points)

    def mask(self, boundary: np.ndarray):
        """Return boundary mask."""
        return self.pointloop.update(boundary).reshape(self.shape)

    @property
    def shape(self):
        """Return grid shape."""
        return self.data.dims['x'], self.data.dims['z']

    def plot(self, attr='psi', axes=None, nulls=True, clabel=None, **kwargs):
        """Plot contours."""
        if len(self.data) == 0:
            return
        self.axes = axes
        if nulls:
            super().plot(axes=axes)
        if isinstance(attr, str):
            attr = getattr(self, f'{attr}_')
        QuadContourSet = self.axes.contour(self.data.x, self.data.z, attr.T,
                                           **self.contour_kwargs(**kwargs))
        if isinstance(kwargs.get('levels', None), int):
            self.levels = QuadContourSet.levels
        if clabel is not None:
            self.axes.clabel(QuadContourSet, **clabel)
        return np.array(QuadContourSet.levels)

    def plot_grid(self):
        """Plot grid."""
        Grid.plot(self)
