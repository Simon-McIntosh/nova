"""Manage methods related to plasma profile control points."""
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
from scipy.optimize import minimize
import xarray

from nova.geometry.curve import Curve
from nova.geometry.quadrant import Quadrant
from nova.geometry.kdtree import KDTree
from nova.graphics.plot import Plot
from nova.imas.machine import Wall


@dataclass
class Points:
    """Manage access to plasma shape point groups."""

    name: str = ""
    data: xarray.Dataset = field(default_factory=xarray.Dataset, repr=False)
    axis: str = ""
    group: str = field(init=False, default="")
    attrs: list[str] = field(init=False, default_factory=list)
    labels: list[str] = field(init=False, default_factory=list)

    def __post_init__(self):
        """Check dataset for self-consistency."""
        self._select_axis()
        self.group = "_".join([name for name in [self.name, self.axis] if name != ""])

    def _select_axis(self):
        """Select point attribures attrs from data."""
        match self.axis:
            case "":
                self._set_attrs()
            case "major":
                self._set_attrs("upper", "lower")
            case "minor":
                self._set_attrs("inner", "outer")
            case "ids_fix":
                self._set_attrs("upper", "lower")
            case "dual":
                self._set_attrs("upper", "lower", "inner", "outer")
            case "square":
                self._set_attrs(
                    "upper_outer", "upper_inner", "lower_inner", "lower_outer"
                )
            case _:
                raise ValueError(f"axis {self.axis} not in [major, minor, dual].")

    def _set_attrs(self, *labels):
        """Set attribute list."""
        if len(labels) == 0:
            attrs = [self.name]
        else:
            attrs = ["_".join([self.name, label]) for label in labels]
        self.attrs = [attr for attr in attrs if attr in self.data]
        self.labels = list(labels)

    def _fixattr(self, attr):
        """Implement missing minor triangularity ids workarround."""
        if self.axis == "ids_fix":
            attr = {"outer": "upper", "inner": "lower"}[attr]  # TODO fix ids
        return attr

    def _getattr(self, attr):
        """Return resolved attribute name."""
        if attr in [self.group, self.name, "mean"]:
            return attr
        attr = self._fixattr(attr)  # TODO fix ids
        return f"{self.name}_{attr}"

    def __getitem__(self, attr: str):
        """Return item from data."""
        label = attr
        attr = self._getattr(attr)  # TODO fix ids
        try:
            if isinstance(self.data, xarray.Dataset):
                data = self.data[attr].data
                try:
                    return data.item()
                except ValueError:
                    return data
                raise TypeError(f"xarray {attr}: {self.data}")
            else:
                return self.data[attr]
        except KeyError as error:
            if label in self.labels and attr not in [self.group, self.name]:
                return self.mean
            raise KeyError(f"invalid attr {attr}") from error
        raise KeyError(f"Mapping attr {attr} not found.")

    def __setitem__(self, attr: str, value):
        """Update data attribute."""
        attr = self._getattr(attr)  # TODO fix ids
        if attr == self.group:  # and attr in self.data:
            factor = value / self.mean  # self.data[attr]
            for subattr in self.attrs:
                self.data[subattr] *= factor
        self.data[attr] = value

    def __getattribute__(self, attr):
        """Extend getattribute to provide access to self.data."""
        try:
            return super().__getattribute__(attr)
        except AttributeError as attribute_error:
            _attr = self._fixattr(attr)  # TODO fix ids
            if _attr not in self.labels:
                raise attribute_error
        return self[attr]

    @cached_property
    def complete(self):
        """Return True if all attributes in labels are defined."""
        return np.all(
            ["_".join([self.name, label]) in self.attrs for label in self.labels]
        )

    def _attr_mean(self):
        """Return mean of attribute set."""
        if len(self.attrs) == 0:
            return self[self.name]
        return np.mean([self.data[attr] for attr in self.attrs])

    @property
    def mean(self):
        """Manage mean attribute."""
        if self.complete:
            return self._attr_mean()
        try:
            return self[self.group]
        except KeyError:
            return self._attr_mean()

    @mean.setter
    def mean(self, value):
        self[self.group] = value


@dataclass
class ControlPoints(Plot):
    """Defined plasma separatrix control points from plasma parameters."""

    data: xarray.Dataset = field(default_factory=xarray.Dataset, repr=False)
    square: bool = False
    strike: bool = False

    @property
    def point_attrs(self):
        """Return point attributes."""
        attrs = ["outer", "upper", "inner", "lower"]
        if self.square:
            attrs.extend(["upper_outer", "upper_inner", "lower_inner", "lower_outer"])
        if self.strike:
            attrs.extend(["inner_strike", "outer_strike"])
        return np.array(attrs)

    @property
    def point_index(self) -> np.ndarray:
        """Return constraint point index excluding strike points."""
        return np.array(
            [
                i
                for i, attr in enumerate(self.point_attrs)
                if attr.split("_")[-1] != "strike"
            ],
            dtype=int,
        )

    @property
    def control_points(self):
        """Return control points."""
        return np.c_[
            [
                point
                for attr in self.point_attrs
                if not np.allclose((point := getattr(self, attr)), (0, 0))
            ]
        ]

    def __getitem__(self, attr):
        """Return attribute from data if present else from super."""
        if hasattr(super(), "__getitem__"):
            return super().__getitem__(attr)
        if attr in self.data:
            if isinstance(self.data, xarray.Dataset):
                data = self.data[attr].data
                try:
                    return data.item()
                except ValueError:
                    return data
                raise TypeError(f"xarray {attr}: {self.data[attr]}")
            return self.data[attr]
        raise KeyError(f"mapping attr {attr} not found")

    def __setitem__(self, attr, value):
        """Update coef attribute."""
        if hasattr(super(), "__getitem__"):
            super().__setitem__(attr, value)
        else:
            try:
                self.data[attr].data = value
            except KeyError:
                dimension = len(np.shape(value))
                coord = tuple(
                    coord for coord, _ in zip(["point", "index"], range(dimension))
                )
                self.data[attr] = coord, np.array(value)

    @property
    def _pointdata(self):
        """Return point data."""
        if hasattr(self, "itime"):
            return self
        return self.data

    @property
    def axis(self):
        """Return location of geometric axis."""
        return self["geometric_axis"]

    @property
    def elongation(self):
        """Return plasma elongation."""
        return self["elongation"]

    @property
    def minor_radius(self):
        """Return minor radius."""
        return self["minor_radius"]

    @cached_property
    def triangularity(self):
        """Return triangularity points instance."""
        return Points("triangularity", self._pointdata, "dual")

    @cached_property
    def triangularity_major(self):
        """Return major triangularity points instance."""
        return Points("triangularity", self._pointdata, "major")

    @cached_property
    def triangularity_minor(self):
        """Return minor triangularity points instance."""
        return Points("elongation", self._pointdata, "ids_fix")  # TODO fix IDS

    @cached_property
    def squareness(self):
        """Return squareness points instance."""
        return Points("squareness", self._pointdata, "square")

    @property
    def upper(self):
        """Return upper control point."""
        return self.axis + self.minor_radius * np.array(
            [-self.triangularity.upper, self.elongation]
        )

    @property
    def lower(self):
        """Return lower control point."""
        return self.axis - self.minor_radius * np.array(
            [self.triangularity.lower, self.elongation]
        )

    @property
    def inner(self):
        """Return inner control point."""
        return self.axis + self.minor_radius * np.array(
            [-1, self.triangularity_minor.inner]
        )

    @property
    def outer(self):
        """Return outer control point."""
        return self.axis + self.minor_radius * np.array(
            [1, self.triangularity_minor.outer]
        )

    @property
    def upper_outer(self):
        """Return upper outer control point."""
        return Quadrant(self.outer, self.upper).separatrix_point(
            self["squareness_upper_outer"]
        )

    @property
    def upper_inner(self):
        """Return upper inner control point."""
        return Quadrant(self.inner, self.upper).separatrix_point(
            self["squareness_upper_inner"]
        )

    @property
    def lower_inner(self):
        """Return lower inner control point."""
        return Quadrant(self.inner, self.lower).separatrix_point(
            self["squareness_lower_inner"]
        )

    @property
    def lower_outer(self):
        """Return lower outer control point."""
        return Quadrant(self.outer, self.lower).separatrix_point(
            self["squareness_lower_outer"]
        )

    @property
    def inner_strike(self):
        """Return inner strike point."""
        return self["strike_point"][0]

    @property
    def outer_strike(self):
        """Return outer strike point."""
        return self["strike_point"][1]

    def plot(self, index=None, axes=None, **kwargs):
        """Plot control points."""
        self.get_axes("2d", axes)
        points = np.array(
            [point for point in self.control_points if not np.allclose(point, (0, 0))]
        )
        self.axes.plot(*points.T, "o", color="C2", ms=10 / 4)
        if hasattr(super(), "plot"):
            super().plot(index, axes, **kwargs)


@dataclass
class PlasmaPoints(ControlPoints):
    """Fit plasma separatrix control points to first wall."""

    def __post_init__(self):
        """Generate minimum gap and create copy of instance data."""
        if hasattr(super(), "__post_init__"):
            super().__post_init__()
        self.update_minimum_gap()

    def update_minimum_gap(self):
        """Update minimum gap."""
        try:
            if "time" in self.data:
                self.data["minimum_gap"] = "time", self.minimum_gap()
            else:
                self.data["minimum_gap"] = np.min(self.point_gap(self.point_index))
        except KeyError:
            self.data["minimum_gap"] = 0

    def minimum_gap(self):
        """Return minimum wall normal point gap vector."""
        gap = np.zeros_like(self.data["time"])
        time_index = self.time_index
        for i in range(self.data.dims["time"]):
            self.time_index = i
            self.clear_cache()
            gap[i] = np.min(self.point_gap(self.point_index))
        self.time_index = time_index
        return gap

    @property
    def limiter(self) -> bool:
        """Return limiter flag."""
        return self["boundary_type"] == 0

    @cached_property
    def firstwall(self):
        """Return firstwall contour (main chamber)."""
        wall = Curve(Wall().boundary, pad_width=0, kind="linear")
        return wall.boundary(np.linspace(0, 1, 1000))

    @cached_property
    def midpoint(self):
        """Return pannel midpoints."""
        return self.firstwall[:-1] + np.diff(self.firstwall, axis=0) / 2

    @cached_property
    def normal(self):
        """Return wall pannel normal vectors."""
        tangent = np.c_[
            -np.diff(self.firstwall, axis=0), np.zeros(len(self.firstwall) - 1)
        ]
        tangent /= np.linalg.norm(tangent, axis=1)[:, np.newaxis]
        return np.cross([0, 0, 1], tangent)[:, :2]

    @cached_property
    def kd_tree(self):
        """Return kd_tree instance for fast nearest neighbour lookup."""
        return KDTree(self.midpoint, np.inf)

    def _control_points(self, point_index: slice | int = slice(None)):
        """Return indexed control points."""
        attrs = self.point_attrs[point_index]
        if isinstance(attrs, str):
            attrs = [attrs]
        return np.c_[
            [
                point
                for attr in attrs
                if not np.allclose(point := getattr(self, attr), (0, 0))
            ]
        ]

    def _midpoint_index(self, points):
        """Return pannel midpoint index closest to requested control points."""
        return np.array([self.kd_tree.query(point)[1] for point in points])

    def control_midpoints(self, point_index: slice | int = slice(None)):
        """Return wall pannel midpoints closest to requested control points."""
        points = self._control_points(point_index)
        return self.midpoint[self._midpoint_index(points)]

    def point_gap(self, point_index: slice | int = slice(None)):
        """Return vector of pannel normal wall gaps."""
        points = self._control_points(point_index)
        index = self._midpoint_index(points)
        vector = points - self.midpoint[index]
        return np.einsum("ij,ij->i", self.normal[index], vector)

    def _update_point_gap(self, radii, minor_radius, point_index):
        """Return vector of pannel normal wall gaps."""
        self._update_radii(radii, minor_radius)
        gap = self.point_gap(point_index)
        if self["boundary_type"] == 1:
            return gap - self["minimum_gap"]
        return gap

    def _update_radii(self, radii, minor_radius):
        """Update major and minor radii."""
        self["minor_radius"] = radii[0]
        self["geometric_axis"][0] = radii[1]
        match self["boundary_type"]:
            case 0:  # limiter
                return self.inner[0] + abs(radii[0] - minor_radius)
            case 1:  # single null
                if not np.allclose(self["x_point"], (0, 0)):
                    x_point_delta = self["x_point"] - self.lower
                    self["geometric_axis"] += x_point_delta
                return -radii[0]
            case _:
                raise NotImplementedError(
                    f'boundary type {self["boundary_type"]} not implemented'
                )

    def fit(self):
        """First wall contour with gap constrainted contorl points."""
        minor_radius = self["minor_radius"]
        firstwall_limit = (np.min(self.firstwall[:, 0]), np.max(self.firstwall[:, 0]))

        bounds = [
            (0, 0.5 * (firstwall_limit[1] - firstwall_limit[0])),
            (firstwall_limit[0], firstwall_limit[1]),
        ]
        constraints = [
            {
                "type": "ineq",
                "fun": self._update_point_gap,
                "args": (
                    minor_radius,
                    index,
                ),
            }
            for index in self.point_index
        ]
        sol = minimize(
            self._update_radii,
            [self["minor_radius"], self["geometric_axis"][0]],
            args=(minor_radius,),
            bounds=bounds,
            constraints=constraints,
            method="SLSQP",
        )
        if not sol.success:
            raise ValueError(f"minimizer failed to converge {sol}")
        self._update_radii(sol.x, minor_radius)
        return sol

    def plot(self, index=None, axes=None, **kwargs):
        """Plot control points and first wall."""
        super().plot()
        wall = Wall()
        for segment in wall.segments:
            self.axes.plot(
                segment[:, 0], segment[:, 1], "-", ms=4, color="gray", linewidth=1.5
            )


if __name__ == "__main__":
    data = xarray.Dataset(
        {
            "geometric_axis": ("point", [6.0, 0.0]),
            "minor_radius": 1.75,
            "elongation": 2.0,
            "triangularity": 0.2,
            "triangularity_upper": 0.2,
            "triangularity_lower": 0.5,
            "elongation_upper": 0.1,  # triangularity outer
            "elongation_lower": 0.5,  # triangularity_inner
            "squareness_upper_outer": 0.1,
            "squareness_upper_inner": 0.1,
            "squareness_lower_inner": 0.1,
            "squareness_lower_outer": 0.1,
            "squareness": 1,
            "minimum_gap": 0.2,
            "boundary_type": 1,
            "x_point": ("point", [5.1, -3.4]),
            "strike_point": (("index", "point"), [[4.2, -3.8], [5.6, -4.4]]),
        }
    )

    plasmapoints = PlasmaPoints(data, square=True, strike=True)
    plasmapoints.elongation

    sol = plasmapoints.fit()
    print(plasmapoints.control_points)
    print(np.array([plasmapoints.point_gap(i) for i in plasmapoints.point_index]))
    plasmapoints.plot()
