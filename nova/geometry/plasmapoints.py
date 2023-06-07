"""Manage methods related to plasma profile control points."""
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
from scipy.optimize import minimize
import xarray

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
    attrs: list[str] = field(init=False, default_factory=list)
    labels: list[str] = field(init=False, default_factory=list)

    def __post_init__(self):
        """Check dataset for self-consistency."""
        self._select_axis()

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

    def __getitem__(self, attr: str):
        """Return item from data."""
        try:
            return self.data[f"{self.name}_{attr}"]
        except KeyError as error:
            if attr in self.labels:
                return self.mean
            raise KeyError(f"invalid attr {attr}") from error

    def __setitem__(self, key: str, value):
        """Update data attribute."""
        if key == "mean":
            factor = value / self.mean
            for attr in self.attrs:
                self.data[attr] *= factor
            return
        self.data[f"{self.name}_{key}"] = value

    def __getattr__(self, attr):
        """Provide attribute get."""
        if self.axis == "ids_fix":
            attr = {"outer": "upper", "inner": "lower"}[attr]
        if attr in self.labels:
            return self[attr]
        raise AttributeError(f"attribute {attr} not defined")

    @property
    def mean(self):
        """Manage mean attribute."""
        if len(self.attrs) == 0:
            return self.data[self.name]
        return np.mean([self.data[attr] for attr in self.attrs])

    @mean.setter
    def mean(self, value):
        self["mean"] = value


@dataclass
class PlasmaPoints(Plot):
    """Calculate plasma profile control points from plasma parameters."""

    data: xarray.Dataset = field(default_factory=xarray.Dataset)
    square: bool = False

    @property
    def point_attrs(self):
        """Return point attributes."""
        attrs = ["outer", "upper", "inner", "lower"]
        if self.square:
            attrs.extend(["upper_outer", "upper_inner", "lower_inner", "lower_outer"])
        return attrs

    @property
    def control_points(self):
        """Return control points."""
        return np.c_[[getattr(self, attr) for attr in self.point_attrs]]

    def __getitem__(self, attr):
        """Return attribute from data if present else from super."""
        if hasattr(super(), "__getitem__"):
            return super().__getitem__(attr)
        if attr in self.data:
            return self.data[attr]
        raise KeyError(f"mapping attr {attr} not found")

    def __setitem__(self, attr, value):
        """Update coef attribute."""
        if hasattr(super(), "__getitem__"):
            super().__setitem__(attr, value)
        else:
            self.data[attr] = value

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
        # TODO update once IDS is fixed
        return Points("elongation", self._pointdata, "ids_fix")

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
        """Return upper control point."""
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

    @cached_property
    def firstwall(self):
        """Return firstwall contour (main chamber)."""
        return Wall().segment(index=0)

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
        return np.cross([0, 0, 1], tangent)[:, :2]

    def plot(self, index=None, axes=None, **kwargs):
        """Plot control points and first wall."""
        self.get_axes("2d", axes)
        wall = Wall()
        for segment in wall.segments:
            self.axes.plot(
                segment[:, 0], segment[:, 1], "-", ms=4, color="gray", linewidth=1.5
            )
        self.axes.plot(*self.control_points.T, "o", color="C2", ms=10 / 4)

    @cached_property
    def kd_tree(self):
        """Return kd_tree instance for fast nearest neighbour lookup."""
        return KDTree(self.midpoint, np.inf)

    def point_gap(self, point_index=slice(None)):
        """Return vector of pannel normal wall gaps."""
        attrs = self.point_attrs[point_index]
        if isinstance(attrs, str):
            attrs = [attrs]
        points = np.c_[[getattr(self, attr) for attr in attrs]]
        index = np.array([self.kd_tree.query(point)[1] for point in points])
        vector = points - self.midpoint[index]
        gap = np.einsum("ij,ij->i", self.normal[index], vector)
        if self["boundary_type"] == 1:
            gap -= self["minimum_gap"]
        return gap

    def _update_point_gap(self, x, minor_radius, point_index):
        """Return vector of pannel normal wall gaps."""
        self._update(x, minor_radius)
        return self.point_gap(point_index)

    def _update(self, x, minor_radius):
        """Update major and minor radii."""
        self["minor_radius"] = x[0]
        self["geometric_axis"][0] = x[1]
        match self["boundary_type"]:
            case 0:  # limiter
                return self.inner[0] + abs(x[0] - minor_radius)
            case 1:  # single null
                x_point_delta = self["x_point"] - self.lower
                self["geometric_axis"][1] += x_point_delta[1]
                return -x[0] + abs(x_point_delta[0])
            case _:
                raise NotImplementedError(
                    f'boundary type {self["boundary_type"]} not implemented'
                )

    def fit(self):
        """First wall contour with gap constrainted contorl points."""
        minor_radius = self["minor_radius"]
        constraints = [
            {
                "type": "ineq",
                "fun": self._update_point_gap,
                "args": (
                    minor_radius,
                    index,
                ),
            }
            for index in range(len(self.control_points))
        ]

        sol = minimize(
            self._update,
            [self["minor_radius"], self["geometric_axis"][0]],
            args=(minor_radius,),
            constraints=constraints,
            method="SLSQP",
        )
        self._update(sol.x, minor_radius)


if __name__ == "__main__":
    plasmapoints = PlasmaPoints(
        {
            "geometric_axis": [6, 0],
            "minor_radius": 3.75,
            "elongation": 2.0,
            "triangularity": 0.2,
            "triangularity_upper": 0.2,
            "triangularity_lower": 0.5,
            "elongation_upper": 0.1,  # triangularity outer
            "elongation_lower": 0.5,  # triangularity_inner
            "squareness": 1,
            "minimum_gap": 0.2,
            "boundary_type": 0,
            "x_point": [5.1, -3.4],
        }
    )

    plasmapoints.fit()
    plasmapoints.plot()
