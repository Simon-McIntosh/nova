"""Manage methods related to plasma profile control points."""
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import xarray

from nova.geometry.quadrant import Quadrant


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
class PlasmaPoints:
    """Calculate plasma profile control points from plasma parameters."""

    data: xarray.Dataset = field(default_factory=xarray.Dataset)

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
    def geometric_radius(self):
        """Return geometric raidus."""
        return self["geometric_radius"]

    @property
    def geometric_height(self):
        """Return geometric height."""
        return self["geometric_height"]

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
