"""Manage methods related to plasma profiles."""
import bisect
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
from scipy.interpolate import interp1d

from nova.geometry.plasmapoints import PlasmaPoints
from nova.graphics.plot import Plot


@dataclass
class PlasmaProfile(Plot, PlasmaPoints):
    """
    Generate Separatrix profiles from plasma shape parameters.

    Simple, General, Realistic, Robust, Analytic, Tokamak Equilibria I.
    Limiter and Divertor Tokamaks

    L. Guazzotto1 and J. P. Freidberg2
    """

    point_number: int = 1000
    profile: np.ndarray | None = field(init=False, repr=False, default=None)

    def __post_init__(self):
        """Initialize point array."""
        self.profile = np.zeros((self.point_number, 2))
        super().__post_init__()

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
        return bisect.bisect_right(self.theta, self["theta_o"] + np.pi)

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
        return np.linspace(-self["theta_o"], 0, self.point_number - self.x_point_number)

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
            self.theta, self.minor_radius, self.elongation, self.triangularity
        )
        return self

    def single_null(self, *args, **kwargs):
        """Update points - lower single null."""
        self.update_coefficents(*args, **kwargs)
        self.set_x_point(kwargs.get("x_point", None))
        self.adjust_elongation_lower()
        upper = self.miller_profile(
            self.theta_upper,
            self.minor_radius,
            self.elongation_upper,
            self.triangularity_upper,
        )
        x_i = (1 - self.triangularity_lower**2) ** 0.5 / (
            self.elongation_lower - (1 - self.triangularity_lower**2) ** 0.5
        )
        k_o = self.elongation_lower / (1 - x_i**2) ** 0.5
        x_1 = (x_i - self.triangularity_lower) / (1 - x_i)
        x_2 = (x_i + self.triangularity_lower) / (1 - x_i)
        self.coef["theta_o"] = np.arctan((1 - x_i**2) ** 0.5 / x_i)
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
    minor_radius, elongation, triangularity = 0.5, 1.5, 0.3
    profile = PlasmaProfile(point_number=201).limiter(
        *geometric_axis, minor_radius, elongation, triangularity
    )

    profile.plot()
