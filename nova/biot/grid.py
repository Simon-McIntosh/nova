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
    """Generate rectangular 2d or 3d grid."""

    number: int | None = field(default=None)
    limit: np.ndarray | None = field(default=None)
    data: xarray.Dataset = field(init=False, repr=False, default_factory=xarray.Dataset)

    def __post_init__(self):
        """Build grid coordinates."""
        if len(self.data) == 0:
            self.generate()

    def generate(self):
        """Return grid coordinates."""
        match len(self.limit):
            case 2:  # 2d grid coordinates
                xcoord, zcoord = self.limit
                self.number = len(xcoord) * len(zcoord)
                self.limit = [xcoord[0], xcoord[-1], zcoord[0], zcoord[-1]]
                coords = dict(zip("xyz", [xcoord, [0], zcoord]))
            case 3:  # 3d grid coordinates
                xcoord, ycoord, zcoord = self.limit
                self.number = len(xcoord) * len(ycoord) * len(zcoord)
                self.limit = [
                    xcoord[0],
                    xcoord[-1],
                    ycoord[0],
                    ycoord[-1],
                    zcoord[0],
                    zcoord[-1],
                ]
                coords = dict(zip("xyz", [xcoord, ycoord, zcoord]))
            case 4:  # 2d grid limits
                xgrid = GridCoord(*self.limit[:2])
                zgrid = GridCoord(*self.limit[2:])
                xgrid.num = xgrid.delta / np.sqrt(
                    xgrid.delta * zgrid.delta / self.number
                )
                zgrid.num = self.number / xgrid.num
                self.number = xgrid.num * zgrid.num
                coords = dict(zip("xyz", [xgrid(), [0], zgrid()]))
            case 6:  # 3d grid limits
                xgrid = GridCoord(*self.limit[:2])
                ygrid = GridCoord(*self.limit[2:4])
                zgrid = GridCoord(*self.limit[-2:])
                grid_volume = xgrid.delta * ygrid.delta * zgrid.delta
                grid_delta = (grid_volume / self.number) ** (1 / 3)
                xgrid.num = xgrid.delta / grid_delta
                ygrid.num = ygrid.delta / grid_delta
                zgrid.num = self.number / (xgrid.num * ygrid.num)
                self.number = xgrid.num * ygrid.num * zgrid.num
                coords = dict(zip("xyz", [xgrid(), ygrid(), zgrid()]))
            case _:
                raise IndexError(f"len(limit) {len(self.limit)} not in [2, 3, 4, 6]")

        self.data = xarray.Dataset(coords=coords)
        X, Y, Z = np.meshgrid(self.data.x, self.data.y, self.data.z, indexing="ij")
        self.data["X"] = (list("xyz"), X)
        self.data["Y"] = (list("xyz"), Y)
        self.data["Z"] = (list("xyz"), Z)
        # if len(coords["y"]) == 1:  # 2d grid
        self.data["x2d"] = (["x", "z"], X[:, 0])
        self.data["z2d"] = (["x", "z"], Z[:, 0])

    def __len__(self):
        """Return grid resolution."""
        return np.prod(self.shape)

    @property
    def shape(self):
        """Return grid shape."""
        if len(self.ycoord) == 1:
            return len(self.xcoord), len(self.zcoord)
        return len(self.xcoord), len(self.ycoord), len(self.zcoord)

    def plot(self, axes=None, **kwargs):
        """Plot 2d grid."""
        self.axes = axes  # set plot axes
        kwargs = {
            "linewidth": 0.4,
            "color": "gray",
            "alpha": 0.5,
            "zorder": -100,
        } | kwargs
        for num, step in zip(self.shape, [1, -1]):
            lines = np.zeros((num, 2, 2))
            for i in range(2):
                index = tuple([slice(None), -i][::step])
                lines[:, i, 0] = self.data.x2d[index]
                lines[:, i, 1] = self.data.z2d[index]
            segments = self.mpl["LineCollection"](lines, **kwargs)
            self.axes.add_collection(segments, autolim=True)
        self.axes.autoscale_view()


@dataclass
class Expand:
    """Calculate grid limit as a factor expansion about multipoly bounds."""

    frame: FrameLink
    index: str | slice | np.ndarray = field(default_factory=lambda: slice(None))
    xmin: float = 1e-12
    fix_aspect: bool = False

    def __post_init__(self):
        """Extract multipolygon bounding box."""
        if isinstance(self.index, str):
            index = self.index
            self.index = getattr(self.frame, self.index)
            if sum(self.index) == 0:
                raise GridError(index)
        """
        poly = shapely.geometry.MultiPolygon(
            [
                polygon.poly
                for polygon in self.frame.poly[self.index]
                if isinstance(polygon.poly, shapely.geometry.Polygon)
            ]
        )
        """
        poly = shapely.ops.unary_union(
            [polygon.poly for polygon in self.frame.poly[self.index]]
        )
        self.limit = np.array([*poly.bounds[::2], *poly.bounds[1::2]])
        self.xcoord = GridCoord(*self.limit[:2])
        self.zcoord = GridCoord(*self.limit[2:])

    def __call__(self, factor) -> np.ndarray:
        """Return expanded limit."""
        delta_x, delta_z = self.xcoord.delta, self.zcoord.delta
        if not self.fix_aspect:
            delta_x = delta_z = np.mean([delta_x, delta_z])
        limit = self.limit + factor / 2 * np.array(
            [-delta_x, delta_x, -delta_z, delta_z]
        )
        if limit[0] < self.xmin:
            limit[0] = self.xmin
        return limit


class BaseGrid(Chart, FieldNull, Operate):
    """Flux grid baseclass."""

    attrs: list[str] = field(default_factory=lambda: ["Br", "Bz", "Psi"])

    def __post_init__(self):
        """Initialize fieldnull version."""
        super().__post_init__()
        self.version["fieldnull"] = None

    def check_null(self):
        """Check validity of upstream data, update field null if nessisary."""
        self.check("psi")
        if self.version["fieldnull"] != self.version["psi"]:
            self.update_null(self.psi_)
            self.version["fieldnull"] = self.version["psi"]

    def __getattribute__(self, attr):
        """Extend getattribute to intercept field null data access."""
        if attr == "data_x" or attr == "data_o":
            self.check_null()
        return super().__getattribute__(attr)

    def plot_svd(self, **kwargs):
        """Plot influence of SVD reduction."""
        for svd, color, linestyle in zip(
            [False, True], ["C7", "C3"], ["solid", "dashed"]
        ):
            self.update_turns("Psi", svd)
            kwargs |= dict(colors=color, linestyles=linestyle)
            self.plot(**kwargs)


@dataclass
class Grid(BaseGrid):
    """Compute interaction across regular grid."""

    def solve(
        self,
        number: int | None = None,
        limit: float | np.ndarray | None = 0,
        index: str | slice | np.ndarray = slice(None),
        grid: xarray.Dataset = xarray.Dataset(),
    ):
        """Solve Biot interaction across grid."""
        with self.solve_biot(number) as number:
            if len(grid) > 0:
                assert all([attr in grid for attr in "XYZ"])
                self.number = np.prod(grid.X.shape)
                limit = None
            else:
                if number is None:
                    return
                if isinstance(limit, (int, float)):
                    limit = Expand(self.subframe, index)(limit)
                grid = Gridgen(number, limit).data
            target = Target(
                {attr.lower(): grid[attr].data.flatten() for attr in "XYZ"},
                label="Grid",
            )
            self.data = Solve(
                self.subframe,
                target,
                reduce=[True, False],
                name=self.name,
                attrs=self.attrs,
            ).data
            self.data = self.data.merge(
                grid, compat="override", combine_attrs="drop_conflicts"
            )

    def _delta(self, coordinate):
        """Return grid spacing along coordinate."""
        if coordinate not in self.data:
            return None
        points = self.data[coordinate].to_numpy()
        if len(points) == 1:
            return None
        delta = points[1] - points[0]
        assert np.allclose(np.diff(points), delta)
        return delta

    @property
    def delta(self):
        """Return grid spacing along each dimension."""
        return tuple(self._delta(coordinate) for coordinate in "xyz")

    @cached_property
    def pointloop(self):
        """Return pointloop instance, used to check loop membership."""
        points = np.array(
            [self.data.x2d.data.flatten(), self.data.z2d.data.flatten()]
        ).T
        return PointLoop(points)

    def mask(self, boundary: np.ndarray):
        """Return boundary mask."""
        return self.pointloop.update(boundary).reshape(self.shape)

    @property
    def shape(self):
        """Return grid shape."""
        if self.data.sizes["y"] == 1:
            return self.data.sizes["x"], self.data.sizes["z"]
        return tuple(self.data.sizes[attr] for attr in "xyz")

    def plot(
        self,
        attr="psi",
        coords="xz",
        index=slice(None),
        axes=None,
        nulls=True,
        clabel=None,
        **kwargs,
    ):
        """Plot contours."""
        if len(self.data) == 0:
            return
        self.axes = axes
        if nulls and hasattr(self, "psi"):
            super().plot(axes=axes)
        if isinstance(attr, str):
            attr = getattr(self, f"{attr}_")[index]
        QuadContourSet = self.axes.contour(
            self.data[coords[0]],
            self.data[coords[1]],
            attr.T,
            **self.contour_kwargs(**kwargs),
        )
        if isinstance(kwargs.get("levels", None), int):
            self.levels = QuadContourSet.levels
        if clabel is not None:
            self.axes.clabel(QuadContourSet, **clabel)
        return np.array(QuadContourSet.levels)

    def plot_grid(self):
        """Plot grid."""
        Gridgen.plot(self)
