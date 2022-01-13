"""Generate grids for BiotGrid methods."""
from dataclasses import dataclass, field, InitVar
from typing import Union

from matplotlib.collections import LineCollection
import numpy as np
from numpy import typing as npt
import shapely.geometry
import xarray

from nova.electromagnetic.biotframe import BiotFrame
from nova.electromagnetic.biotsolve import BiotSolve
from nova.electromagnetic.biotdata import BiotData
from nova.electromagnetic.fieldnull import FieldNull
from nova.electromagnetic.framelink import FrameLink
from nova.electromagnetic.polyplot import Axes
from nova.utilities.xpu import asnumpy


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
    index: Union[slice, npt.ArrayLike] = slice(None)
    xmin: float = 1e-12
    fix_aspect: bool = False

    def __post_init__(self):
        """Extract multipolygon bounding box."""
        if isinstance(self.index, str):
            self.index = getattr(self.frame, self.index)
        poly = shapely.geometry.MultiPolygon(
            [polygon.poly for polygon in self.frame.poly[self.index]])
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
class BiotGrid(Axes, BiotData):
    """Compute interaction across grid."""

    attrs: list[str] = field(default_factory=lambda: ['Br', 'Bz', 'Psi'])
    levels: Union[int, list[float]] = 31
    _null: FieldNull = field(init=False, default=None)

    def solve_biot(self, number: int, limit: Union[float, list[float]],
                   index: Union[str, slice, npt.ArrayLike] = slice(None)):
        """Solve Biot interaction across grid."""
        if isinstance(limit, (int, float)):
            limit = Expand(self.subframe, index)(limit)
        grid = Grid(number, limit)
        target = BiotFrame(dict(x=grid.data.x2d.values.flatten(),
                                z=grid.data.z2d.values.flatten()),
                           label='Grid')
        self.data = BiotSolve(self.subframe, target, reduce=[True, False],
                              attrs=self.attrs).data
        # insert grid data
        self.data.coords['x'] = grid.data.x
        self.data.coords['z'] = grid.data.z
        self.data.coords['x2d'] = (['x', 'z'], grid.data.x2d.data)
        self.data.coords['z2d'] = (['x', 'z'], grid.data.z2d.data)
        # initialize field null instance
        self._null = FieldNull(self.coords)
        self.version['current'] = id(None)

    @property
    def null(self):
        if (version := id(self.current)) != self.version['current']:
            print('update')
            #  TODO add plasma change check version['plasma']
            #  TODO add version check to topology unittest
            #  TODO subclass null from biotgrid of full intergration !!! :)
            shape = self.data.dims['x'], self.data.dims['z']
            psi, bn = self.psi.reshape(shape), self.bn.reshape(shape)
            self._null.update(psi, bn)
            self.version['current'] = version
        return self._null

    @property
    def coords(self):
        """Return 1D grid coordinates."""
        return self.data.x.data, self.data.z.data

    @property
    def shape(self):
        """Return grid shape."""
        return self.data.dims['x'], self.data.dims['z']

    def plot(self, axes=None, **kwargs):
        """Plot poloidal flux contours."""
        self.axes = axes
        kwargs = dict(colors='lightgray', linewidths=1.5, alpha=0.9,
                      linestyles='solid', levels=self.levels) | kwargs
        psi = asnumpy(self.psi.reshape(*self.shape).T)
        QuadContourSet = self.axes.contour(self.data.x, self.data.z, psi,
                                           **kwargs)
        if isinstance(kwargs['levels'], int):
            self.levels = QuadContourSet.levels
