"""Generate of an artifical an separatrix from shape parameters."""
import bisect
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

from nova.frame.baseplot import Plot


@dataclass
class LCFS(Plot):
    """Calculate plasma shape parameters from the last closed flux surface."""

    points: np.ndarray

    def __call__(self, attrs: list[str]):
        """Return attribute shape vector."""
        return np.array([getattr(self, attr) for attr in attrs])

    @cached_property
    def radius(self):
        """Return surface radius."""
        return self.points[:, 0]

    @cached_property
    def height(self):
        """Return surface height."""
        return self.points[:, 1]

    @cached_property
    def r_max(self):
        """Return maximum radius, Rmax."""
        return self.radius.max()

    @cached_property
    def r_min(self):
        """Return minimum radius, Rmin."""
        return self.radius.min()

    @cached_property
    def geometric_radius(self):
        """Return geometric radius, Rgeo."""
        return (self.r_max + self.r_min) / 2

    @cached_property
    def minor_radius(self):
        """Return minor radius, a."""
        return (self.r_max - self.r_min) / 2

    @cached_property
    def z_max(self):
        """Return maximum height, Zmax."""
        return self.height.max()

    @cached_property
    def z_min(self):
        """Return minimum height, Zmin."""
        return self.height.min()

    @cached_property
    def inverse_aspect_ratio(self):
        """Return inverse aspect ratio, epsilon."""
        return self.minor_radius / self.geometric_radius

    @cached_property
    def elongation(self):
        """Return elongation, kappa."""
        return (self.z_max - self.z_min) / (2*self.minor_radius)

    @cached_property
    def r_zmax(self):
        """Return radius at maximum height, Rzmax."""
        return self.radius[np.argmax(self.height)]

    @cached_property
    def r_zmin(self):
        """Return radius at minimum height, Rzmin."""
        return self.radius[np.argmin(self.height)]

    @cached_property
    def triangularity(self):
        """Return triangularity, del."""
        r_zmean = (self.r_zmax + self.r_zmin) / 2
        return (self.geometric_radius - r_zmean) / self.minor_radius

    @cached_property
    def upper_triangularity(self):
        """Return upper triangularity, del_u."""
        return (self.geometric_radius - self.r_zmax) / self.minor_radius

    @cached_property
    def lower_triangularity(self):
        """Return lower triangularity, del_l."""
        return (self.geometric_radius - self.r_zmin) / self.minor_radius

    def plot(self, label=False):
        """Plot last closed flux surface and key geometrical points."""
        self.get_axes('2d')
        self.axes.plot(self.radius, self.height, color='k', alpha=0.25)
        if label:
            self.axes.plot(self.r_max, self.height[np.argmax(self.radius)],
                           'o', label='Rmax')
            self.axes.plot(self.r_min, self.height[np.argmin(self.radius)],
                           'o', label='Rmin')
            self.axes.plot(self.geometric_radius,
                           (self.z_max + self.z_min) / 2, 'o', label='Rgeo')
            self.axes.plot(self.r_zmax, self.z_max, 'o', label='Zmax')
            self.axes.plot(self.r_zmin, self.z_min, 'o', label='Zmin')
            self.axes.legend(loc='center')


@dataclass
class Miller(Plot):
    """
    Generate Miller profiles from plasma shape parameters.

    Simple, General, Realistic, Robust, Analytic, Tokamak Equilibria I.
    Limiter and Divertor Tokamaks

    L. Guazzotto1 and J. P. Freidberg2
    """

    radius: float
    height: float
    point_number: int = 250
    theta_x: float | None = None
    _points: np.ndarray = field(init=False, repr=False)
    attrs: dict[str, float] = field(init=False, default_factory=dict)

    def __post_init__(self):
        """Initialize point array."""
        self._points = np.zeros((self.point_number, 2))
        super().__post_init__()

    @property
    def axis(self):
        """Return geometric axis."""
        return np.array([self.radius, self.height])

    @property
    def points(self) -> np.ndarray:
        """Return profile r,z points."""
        return self._points + self.axis[np.newaxis, :]

    @points.setter
    def points(self, points):
        """Update zero centered point array."""
        self._points[:] = points

    @cached_property
    def theta(self):
        """Return full theta array."""
        return np.linspace(0, 2*np.pi, self.point_number)

    @cached_property
    def upper_point_number(self):
        """Return number of upper profile points."""
        return bisect.bisect_right(self.theta, np.pi)

    @cached_property
    def x_point_number(self):
        """Return number of profile points to the lower x-point."""
        return bisect.bisect_right(self.theta, self.theta_x)

    @cached_property
    def theta_upper(self):
        """Return upper theta array 0<=theta<np.pi."""
        return self.theta[:self.upper_point_number]

    @cached_property
    def theta_lower_hfs(self):
        """Return lower theta array on the high field side."""
        return self.theta[self.upper_point_number:self.x_point_number]

    @cached_property
    def theta_lower_lfs(self):
        """Return lower theta array on the low field side."""
        return np.linspace(-(self.theta_x - np.pi), 0,
                           self.point_number-self.x_point_number)

    @staticmethod
    def miller(theta, minor_radius, elongation, triangularity):
        """Return Miller profile."""
        del_hat = np.arcsin(triangularity)
        return minor_radius * np.c_[np.cos(theta + del_hat * np.sin(theta)),
                                    elongation * np.sin(theta)]

    def limiter(self, minor_radius, elongation, triangularity):
        """Update points - symetric limiter."""
        self.points = self.miller(self.theta, minor_radius,
                                  elongation, triangularity)
        self.attrs = dict(minor_radius=minor_radius, elongation=elongation,
                          triangularity=triangularity)
        return self

    def single_null(self, minor_radius, elongation, triangularity,
                    elongation_x=None, triangularity_x=None):
        """Update points - lower single null."""
        if elongation_x is None:
            elongation_x = elongation
        if triangularity_x is None:
            triangularity_x = triangularity
        upper = self.miller(self.theta_upper, minor_radius,
                            elongation, triangularity)
        self.theta_x = np.arctan2(elongation_x, triangularity_x) + np.pi
        k_o = -elongation_x / np.sin(self.theta_x)

        x_1 = (-triangularity_x - np.cos(self.theta_x)) / \
            (np.cos(self.theta_x) + 1)
        x_2 = (-triangularity_x - np.cos(-(self.theta_x - np.pi))) / \
            (np.cos(-(self.theta_x - np.pi)) - 1)
        lower_hfs = minor_radius * np.c_[
            x_1 + (1 + x_1) * np.cos(self.theta_lower_hfs),
            k_o * np.sin(self.theta_lower_hfs)]
        lower_lfs = minor_radius * np.c_[
            -x_2 + (1 + x_2) * np.cos(self.theta_lower_lfs),
            k_o * np.sin(self.theta_lower_lfs)]
        self.points = np.vstack((upper, lower_hfs, lower_lfs))
        return self

    def plot(self):
        """Plot last closed flux surface."""
        self.get_axes('2d')
        self.axes.plot(*self.points.T, 'gray')


if __name__ == '__main__':

    separatrix = Miller(5, 0, 2000).single_null(2, 1.75, 0.3)
    separatrix.plot()
