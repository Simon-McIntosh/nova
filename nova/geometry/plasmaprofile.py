"""Manage methods related to plasma profiles."""
import bisect
from dataclasses import dataclass, field
from functools import cached_property
from typing import ClassVar

import numpy as np
from scipy.interpolate import interp1d

from nova.geometry.plasmapoints import PlasmaPoints


@dataclass
class PlasmaProfile(PlasmaPoints):
    """
    Generate Separatrix profiles from plasma shape parameters.

    Simple, General, Realistic, Robust, Analytic, Tokamak Equilibria I.
    Limiter and Divertor Tokamaks

    L. Guazzotto1 and J. P. Freidberg2
    """

    point_number: int = 1000
    profile: np.ndarray | None = field(init=False, repr=False, default=None)
    theta_o: float = field(init=False, default=0.0)

    profile_attrs: ClassVar[list[str]] = [
        "geometric_radius",
        "geometric_height",
        "minor_radius",
        "elongation",
        "triangularity",
    ]

    def __post_init__(self):
        """Initialize point array."""
        self.profile = np.zeros((self.point_number, 2))
        if hasattr(super(), "__post_init__"):
            super().__post_init__()

    def update_coefficents(self, *args, **kwargs):
        """Update plasma profile coefficients."""
        coef = kwargs | {attr: arg for attr, arg in zip(self.profile_attrs, args)}
        for attr, value in coef.items():
            self[attr] = value
        return self

    @property
    def geometric_axis(self):
        """Manage geometric axis attribute."""
        return np.array([self["geometric_radius"], self["geometric_height"]], float)

    @property
    def points(self) -> np.ndarray:
        """Return profile r,z points."""
        return self.profile + self.geometric_axis[np.newaxis, :]

    @points.setter
    def points(self, points):
        """Update zero centered point array."""
        delta = np.linalg.norm(points[1:] - points[:-1], axis=1)
        length = np.append(0, np.cumsum(delta))
        linspace = np.linspace(0, length[-1], self.point_number)
        try:
            self.profile[:] = interp1d(length, points, "quadratic", 0)(linspace)
        except ValueError:
            self.profile[:] = np.zeros((self.point_number, 2))

    @cached_property
    def theta(self):
        """Return full theta array."""
        return np.linspace(0, 2 * np.pi, self.point_number)

    @cached_property
    def upper_point_number(self):
        """Return number of upper profile points."""
        return bisect.bisect_right(self.theta, np.pi)

    @property
    def x_point_number(self):
        """Return number of profile points to the lower x-point."""
        return bisect.bisect_right(self.theta, self.theta_o + np.pi)

    @cached_property
    def theta_upper(self):
        """Return upper theta array 0<=theta<np.pi."""
        return self.theta[: self.upper_point_number]

    @property
    def theta_lower_hfs(self):
        """Return lower theta array on the high field side."""
        return self.theta[self.upper_point_number : self.x_point_number]

    @property
    def theta_lower_lfs(self):
        """Return lower theta array on the low field side."""
        return np.linspace(-self.theta_o, 0, self.point_number - self.x_point_number)

    @staticmethod
    def miller_profile(theta, minor_radius, elongation, triangularity):
        """Return Miller profile."""
        return (
            minor_radius
            * np.c_[
                np.cos(theta + np.arcsin(triangularity) * np.sin(theta)),
                elongation * np.sin(theta),
            ]
        )

    def limiter(self, *args, **kwargs):
        """Update points - symetric limiter."""
        self.update_coefficents(*args, **kwargs)
        self.points = self.miller_profile(
            self.theta, self.minor_radius, self.elongation, self.triangularity.mean
        )
        return self

    @property
    def x_point(self):
        """Return x-point."""
        return self["x_point"]

    def check_consistency(self):
        """Check data consistency."""
        for attr in ["kappa", "delta"]:
            self.plasma_shape[attr].check_consistency()

    def set_x_point(self, x_point, elongation):
        """Adjust lower elongation and triangularity to match x_point."""
        if x_point is None:
            return
        self.triangularity.lower = (
            self.geometric_axis[0] - self.x_point[0]
        ) / self.minor_radius
        elongation["lower"] = (
            self.geometric_axis[1] - self.x_point[1]
        ) / self.minor_radius
        assert abs(self.triangularity.lower) < 1

    def adjust_elongation_lower(self, elongation):
        """Adjust lower elongation for single-null compliance."""
        if elongation["lower"] < (
            min_kappa := 2 * (1 - self.triangularity.lower**2) ** 0.5
        ):
            delta_kappa = 1e-3 + min_kappa - elongation["lower"]
            elongation["lower"] += delta_kappa
            elongation["upper"] -= delta_kappa
            # self["geometric_height"] += self.minor_radius * delta_kappa

    def single_null(self, *args, **kwargs):
        """Update points - lower single null."""
        self.update_coefficents(*args, **kwargs)
        elongation = {"upper": self.elongation, "lower": self.elongation}
        self.set_x_point(kwargs.get("x_point", None), elongation)
        self.adjust_elongation_lower(elongation)

        print(
            self.theta_upper,
            self.minor_radius,
            elongation["upper"],
            self.triangularity.upper,
        )
        upper = self.miller_profile(
            self.theta_upper,
            self.minor_radius,
            elongation["upper"],
            self.triangularity.upper,
        )
        x_i = (1 - self.triangularity.lower**2) ** 0.5 / (
            elongation["lower"] - (1 - self.triangularity.lower**2) ** 0.5
        )
        k_o = elongation["lower"] / (1 - x_i**2) ** 0.5
        x_1 = (x_i - self.triangularity.lower) / (1 - x_i)
        x_2 = (x_i + self.triangularity.lower) / (1 - x_i)
        self.theta_o = np.arctan((1 - x_i**2) ** 0.5 / x_i)
        lower_hfs = (
            self.minor_radius
            * np.c_[
                x_1 + (1 + x_1) * np.cos(self.theta_lower_hfs),
                k_o * np.sin(self.theta_lower_hfs),
            ]
        )
        lower_lfs = (
            self.minor_radius
            * np.c_[
                -x_2 + (1 + x_2) * np.cos(self.theta_lower_lfs),
                k_o * np.sin(self.theta_lower_lfs),
            ]
        )
        self.points = np.vstack((lower_lfs[:-1], upper, lower_hfs, lower_lfs[0]))
        return self

    def plot(self, axes=None, **kwargs):
        """Plot last closed flux surface."""
        self.get_axes("2d", axes)
        self.axes.plot(*self.points.T, "-", lw=1.5, color="C6")


if __name__ == "__main__":
    geometric_axis = (5.2, 0)
    minor_radius, elongation, triangularity, triangularity_minor = 0.5, 1.5, 0.3, 0.1
    profile = PlasmaProfile(point_number=201).single_null(
        *geometric_axis,
        minor_radius,
        elongation,
        triangularity,
        triangularity_minor,
    )

    profile.plot()
