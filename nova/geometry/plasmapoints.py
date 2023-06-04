"""Manage methods related to plasma profile control points."""
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np


@dataclass
class FourPoint:
    """Manage access to (upper, lower, inner, outer) type plasma shape parameters."""

    segment: str
    coef: dict[str, float]

    def __post_init__(self):
        """Check dataset for self-consistency."""
        self.check_consistency()

    def check_consistency(self):
        """Check data consistency."""
        if all(
            attr in self.coef
            for attr in [self.segment, self.upper_attr, self.lower_attr]
        ):
            assert np.isclose(self.mean, (self.upper + self.lower) / 2)
        if all(
            attr in self.coef
            for attr in [self.segment, self.inner_attr, self.outer_attr]
        ):
            assert np.isclose(self.minor, (self.inner + self.outer) / 2)

    @property
    def upper_attr(self) -> str:
        """Return upper attribute name."""
        return f"{self.segment}_upper"

    @property
    def lower_attr(self) -> str:
        """Return lower attribute name."""
        return f"{self.segment}_lower"

    @property
    def inner_attr(self) -> str:
        """Return inner attribute name."""
        return f"{self.segment}_inner"

    @property
    def outer_attr(self) -> str:
        """Return outer attribute name."""
        return f"{self.segment}_outer"

    @property
    def mean(self):
        """Return mean attribute."""
        match self.coef:
            case {self.segment: mean}:
                return mean
            case {self.upper_attr: upper, self.lower_attr: lower}:
                return (upper + lower) / 2
            case {self.upper_attr: upper}:
                return upper
            case {self.lower_attr: lower}:
                return lower
            case _:
                raise KeyError(
                    "attributes required to reconstruct "
                    f"mean {self.segment} not found in {self.coef}"
                )

    @property
    def upper(self):
        """Return upper attribute."""
        match self.coef:
            case {self.upper_attr: upper}:
                return upper
            case {self.lower_attr: lower}:
                return 2 * self.mean - lower
            case _:
                return self.mean

    @property
    def lower(self):
        """Return lower attribute."""
        match self.coef:
            case {self.lower_attr: lower}:
                return lower
            case {self.upper_attr: upper}:
                return 2 * self.mean - upper
            case _:
                return self.mean


@dataclass
class PlasmaPoints:
    """Calculate plasma profile control points from plasma parameters."""

    coef: dict[str, float] = field(default_factory=dict)
    plasma_shape: dict[str, FourPoint] = field(init=False, default_factory=dict)

    profile_attrs: ClassVar[list[str]] = [
        "geometric_radius",
        "geometric_height",
        "minor_radius",
        "elongation",
        "triangularity",
    ]

    def __post_init__(self):
        """Initialise updown."""
        if hasattr(super(), "__post_init__"):
            super().__post_init__()
        self.plasma_shape["kappa"] = FourPoint("elongation", self.coef)
        self.plasma_shape["delta"] = FourPoint("triangularity", self.coef)

    def update_coefficents(self, *args, **kwargs):
        """Update plasma profile coefficients."""
        self.coef = kwargs
        self.coef |= {attr: arg for attr, arg in zip(self.profile_attrs, args)}
        self.plasma_shape["kappa"] = FourPoint("elongation", self.coef)
        self.plasma_shape["delta"] = FourPoint("triangularity", self.coef)
        return self

    @property
    def minor_radius(self):
        """Return minor radius."""
        return self["minor_radius"]

    @property
    def geometric_radius(self):
        """Return geometric raidus."""
        return self["geometric_radius"]

    @property
    def geometric_height(self):
        """Return geometric height."""
        return self["geometric_height"]

    @property
    def x_point(self):
        """Return x-point."""
        return self["x_point"]

    def __getitem__(self, attr):
        """Return attribute from coef if present else from super."""
        if attr in self.coef:
            return self.coef[attr]
        if hasattr(super(), "__getitem__"):
            return super().__getitem__(attr)

    def __setitem__(self, attr, value):
        """Update coef attribute."""
        self.coef[attr] = value

    def check_consistency(self):
        """Check data consistency."""
        for attr in ["kappa", "delta"]:
            self.plasma_shape[attr].check_consistency()

    def set_x_point(self, x_point):
        """Adjust lower elongation and triangularity to match x_point."""
        if x_point is None:
            return
        self["triangularity_lower"] = (
            self.geometric_radius - self.x_point[0]
        ) / self.minor_radius
        if "triangularity" in self.coef:
            self["triangularity_upper"] = self["triangularity"]
            del self.coef["triangularity"]
        self["elongation_lower"] = (
            self.geometric_height - self.x_point[1]
        ) / self.minor_radius
        assert abs(self["triangularity_lower"]) < 1
        self.check_consistency()

    @property
    def elongation(self):
        """Return plasma elongation."""
        return self.plasma_shape["kappa"].mean

    @property
    def elongation_upper(self):
        """Return upper plasma elongation."""
        return self.plasma_shape["kappa"].upper

    @property
    def elongation_lower(self):
        """Return lower plasma elongation."""
        return self.plasma_shape["kappa"].lower

    @property
    def triangularity(self):
        """Return plasma triangularity."""
        return self.plasma_shape["delta"].mean

    @property
    def triangularity_upper(self):
        """Return plasma triangularity."""
        return self.plasma_shape["delta"].upper

    @property
    def triangularity_lower(self):
        """Return plasma triangularity."""
        return self.plasma_shape["delta"].lower

    def adjust_elongation_lower(self):
        """Adjust lower elongation for single-null compliance."""
        if self.elongation_lower < (
            min_kappa := 2 * (1 - self.triangularity_lower**2) ** 0.5
        ):
            delta_kappa = 1e-3 + min_kappa - self.elongation_lower
            self["elongation_lower"] = self.elongation_lower + delta_kappa
            self["geometric_height"] += self.minor_radius * delta_kappa
            self.plasma_shape["kappa"].check_consistency()
