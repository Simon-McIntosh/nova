"""Calculate shape interpolators and parameters from a separatrix point string."""
from dataclasses import dataclass, field
from functools import cached_property, wraps

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar

from nova.geometry.quadrant import Quadrant
from nova.graphics.plot import Plot


def negate(func):
    """Return negated interpolator output."""

    @wraps(func)
    def wrapper(*args):
        return -func(*args)

    return wrapper


@dataclass
class Peak:
    """Wrapped (loop) interpolation and minimization."""

    length: np.ndarray
    value: int
    pad_width: int = 6
    kind: str = "quadratic"

    def __post_init__(self):
        """Wrap inputs."""
        if self.pad_width > 0:
            segments = self.length[1:] - self.length[:-1]
            segments = np.pad(segments, self.pad_width, "wrap")
            self.length = np.append(0, np.cumsum(segments))
            self.length /= self.length[-1]
            self.value = np.pad(self.value, self.pad_width, "wrap")

    def __call__(self, length):
        """Return call to interpolator."""
        return self.interpolator(length)

    @cached_property
    def interpolator(self):
        """Return 1d interpolator."""
        return interp1d(
            self.length,
            self.value,
            self.kind,
            assume_sorted=True,
            bounds_error=False,
            fill_value="extrapolate",
        )

    @staticmethod
    def _minimize(function):
        """Wrap scipy minimize_scalar."""
        return minimize_scalar(function, bounds=(0, 1), method="bounded")

    @cached_property
    def minimum(self):
        """Return location and value of minimium."""
        res = self._minimize(self.interpolator)
        return res.x, res.fun

    @cached_property
    def maximum(self):
        """Return location and value of minimium."""
        res = self._minimize(negate(self.interpolator))
        return res.x, -res.fun

    @property
    def min_length(self):
        """Return normalized minimum length."""
        return self.minimum[0]

    @property
    def min_value(self):
        """Return normalized minimum length."""
        return self.minimum[1]

    @property
    def max_length(self):
        """Return normalized maximum length."""
        return self.maximum[0]

    @property
    def max_value(self):
        """Return normalized minimum length."""
        return self.maximum[1]


@dataclass
class Curve:
    """Calculate geometric properties for a curve traced by an input point array."""

    points: np.ndarray
    pad_width: int = 0
    kind: str = "quadratic"

    @cached_property
    def segment_length(self):
        """Return separatrix segment lengths."""
        return np.linalg.norm(self.points[1:] - self.points[:-1], axis=1)

    @cached_property
    def cumulative_length(self):
        """Return normalized cumulative separatrix length."""
        return np.append(0, np.cumsum(self.segment_length)) / self.length

    @cached_property
    def radius(self):
        """Return surface radius interpolator."""
        return Peak(
            self.cumulative_length, self.points[:, 0], self.pad_width, self.kind
        )

    @cached_property
    def height(self):
        """Return surface height interpolator."""
        return Peak(
            self.cumulative_length, self.points[:, 1], self.pad_width, self.kind
        )

    def boundary(self, length):
        """Return lcfs interpolated to zero-pad normalized length."""
        radius = Peak(self.cumulative_length, self.points[:, 0], 0, self.kind)
        height = Peak(self.cumulative_length, self.points[:, 1], 0, self.kind)
        return np.c_[radius(length), height(length)]

    @cached_property
    def length(self):
        """Return length of last closed flux surface."""
        return self.segment_length.sum()

    @property
    def r_max(self):
        """Return minimum radius, Rmin."""
        return self.radius.max_value

    @property
    def r_min(self):
        """Return minimum radius, Rmin."""
        return self.radius.min_value

    @cached_property
    def z_max(self):
        """Return maximum height, Zmax."""
        return self.height.max_value

    @cached_property
    def z_min(self):
        """Return minimum height, Zmin."""
        return self.height.min_value

    @cached_property
    def r_zmax(self):
        """Return radius at maximum height, Rzmax."""
        return self.radius(self.height.max_length)

    @cached_property
    def r_zmin(self):
        """Return radius at minimum height, Rzmin."""
        return self.radius(self.height.min_length)

    @cached_property
    def z_rmax(self):
        """Return height at maximum radius, Zrmax."""
        return self.height(self.radius.max_length)

    @cached_property
    def z_rmin(self):
        """Return height at minimum radius, Zrmin."""
        return self.height(self.radius.min_length)


@dataclass
class PointGeometry(Curve):
    """Calculate geometric axis and derived parameters from bounding box."""

    @cached_property
    def geometric_radius(self):
        """Return geometric radius, Rgeo."""
        return (self.r_max + self.r_min) / 2

    @cached_property
    def geometric_height(self):
        """Return geometric height, Zgeo."""
        return (self.z_max + self.z_min) / 2

    @cached_property
    def geometric_axis(self):
        """Return geometric axis."""
        return np.r_[self.geometric_radius, self.geometric_height]

    @cached_property
    def minor_radius(self):
        """Return minor radius, a."""
        return (self.r_max - self.r_min) / 2

    @cached_property
    def inverse_aspect_ratio(self):
        """Return inverse aspect ratio, epsilon."""
        return self.minor_radius / self.geometric_radius


@dataclass
class Elongation(PointGeometry):
    """Extend Point Geometry to include plasma elongation."""

    @cached_property
    def elongation(self):
        """Return elongation, kappa."""
        return (self.z_max - self.z_min) / (2 * self.minor_radius)


@dataclass
class Triangularity(PointGeometry):
    """Extend Point Geometry to include plasma triangularity."""

    @cached_property
    def triangularity(self):
        """Return triangularity, del."""
        r_zmean = (self.r_zmax + self.r_zmin) / 2
        return (self.geometric_radius - r_zmean) / self.minor_radius

    @cached_property
    def triangularity_upper(self):
        """Return upper triangularity, del_u."""
        return (self.geometric_radius - self.r_zmax) / self.minor_radius

    @cached_property
    def triangularity_lower(self):
        """Return lower triangularity, del_l."""
        return (self.geometric_radius - self.r_zmin) / self.minor_radius

    @cached_property
    def triangularity_inner(self):
        """Return inner triangularity, del_i."""
        return (self.z_rmin - self.geometric_height) / self.minor_radius

    @cached_property
    def triangularity_outer(self):
        """Return outer triangularity, del_o."""
        return (self.z_rmax - self.geometric_height) / self.minor_radius


@dataclass
class Squareness(Plot, PointGeometry):
    """Extend point geometry to inculde squarness calculation."""

    quadrant_points: np.ndarray = field(init=False, repr=False)
    quadrant_theta: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        """Initialize quadrant points."""
        self.update_quadrants()

    def update_quadrants(self):
        """Update separatrix quadrant-local poloidal angle."""
        points = np.copy(self.points)
        for i in range(4):
            points[self.quadrant_mask(i)] -= self.quadrant_axis(i)
        theta = np.arctan2(points[:, 1], points[:, 0])
        theta[theta < 0] += 2 * np.pi
        index = np.argsort(theta)
        theta = theta[index]
        points = self.points[index]
        self.quadrant_theta, unique_index = np.unique(theta, True)
        self.quadrant_points = points[unique_index]

    @cached_property
    def quadrant_radius(self):
        """Return quadrant radius peak interpolator."""
        return Peak(self.quadrant_theta, self.quadrant_points[:, 0], pad_width=0)

    @cached_property
    def quadrant_height(self):
        """Return quadrant height peak interpolator."""
        return Peak(self.quadrant_theta, self.quadrant_points[:, 1], pad_width=0)

    def quadrant_mask(self, index: int):
        """Return quadrant mask."""
        match index:
            case 0:  # upper_outer
                return (self.points[:, 0] >= self.r_zmax) & (
                    self.points[:, 1] >= self.z_rmax
                )
            case 1:  # upper_inner
                return (self.points[:, 0] < self.r_zmax) & (
                    self.points[:, 1] > self.z_rmin
                )
            case 2:  # lower_inner
                return (self.points[:, 0] <= self.r_zmin) & (
                    self.points[:, 1] <= self.z_rmin
                )
            case 3:  # lower_outer
                return (self.points[:, 0] > self.r_zmin) & (
                    self.points[:, 1] < self.z_rmax
                )
            case _:
                raise IndexError(f"quadrant index {index} not 0-3")

    def quadrant_axis(self, index: int):
        """Return local quadrant axis."""
        match index:
            case 0:  # upper_outer
                return self.r_zmax, self.z_rmax
            case 1:  # upper_inner
                return self.r_zmax, self.z_rmin
            case 2:  # lower_inner
                return self.r_zmin, self.z_rmin
            case 3:  # lower_outer
                return self.r_zmin, self.z_rmax
            case _:
                raise IndexError(f"quadrant index {index} not 0-3")

    def quadrant_point(self, index: int):
        """Return quadrant point."""
        theta = index * np.pi / 2 + self.quadrant(index).theta
        return np.array([self.quadrant_radius(theta), self.quadrant_height(theta)])

    def quadrant(self, index):
        """Return quadrant index."""
        match index:
            case 0:  # upper_outer
                minor_point = (self.r_max, self.z_rmax)
                major_point = (self.r_zmax, self.z_max)
            case 1:  # upper_inner
                minor_point = (self.r_min, self.z_rmin)
                major_point = (self.r_zmax, self.z_max)
            case 2:  # lower_inner
                minor_point = (self.r_min, self.z_rmin)
                major_point = (self.r_zmin, self.z_min)
            case 3:  # lower_outer
                minor_point = (self.r_max, self.z_rmax)
                major_point = (self.r_zmin, self.z_min)
            case _:
                raise IndexError(f"quadrant index {index} not 0-3")
        return Quadrant(minor_point, major_point)

    def squareness(self, index):
        """Return quadrant squareness."""
        return self.quadrant(index).squareness(self.quadrant_point(index))

    @cached_property
    def squareness_upper_outer(self):
        """Return upper outer squareness."""
        return self.squareness(0)

    @cached_property
    def squareness_upper_inner(self):
        """Return upper inner squareness."""
        return self.squareness(1)

    @cached_property
    def squareness_lower_inner(self):
        """Return lower inner squareness."""
        return self.squareness(2)

    @cached_property
    def squareness_lower_outer(self):
        """Return lower outer squareness."""
        return self.squareness(3)

    def plot_quadrants(self, axes=None):
        """Plot parametric curves."""
        self.set_axes("2d", axes)
        for i in range(4):
            self.quadrant(i).plot(self.axes)


@dataclass
class LCFS(Elongation, Triangularity, Squareness, Plot):
    """Calculate plasma shape parameters from the last closed flux surface."""

    points: np.ndarray

    def __call__(self, attrs: list[str]):
        """Return attribute shape vector."""
        return np.array([getattr(self, attr) for attr in attrs])

    def plot(self, label=False):
        """Plot last closed flux surface and key geometrical points."""
        self.get_axes("2d")
        self.axes.plot(*self.points.T, color="k", alpha=0.25)
        if label:
            self.axes.plot(self.r_max, self.z_rmax, "o", label="Rmax")
            self.axes.plot(self.r_min, self.z_rmin, "o", label="Rmin")
            self.axes.plot(
                self.geometric_radius, (self.z_max + self.z_min) / 2, "o", label="Rgeo"
            )
            self.axes.plot(self.r_zmax, self.z_max, "o", label="Zmax")
            self.axes.plot(self.r_zmin, self.z_min, "o", label="Zmin")
            self.axes.legend(loc="center")
        self.plot_quadrants(self.axes)
        for quadrant in range(4):
            self.axes.plot(*self.quadrant_point(quadrant), "k.", ms=8)


if __name__ == "__main__":
    from nova.geometry.plasmaprofile import PlasmaProfile

    geometric_axis = (5.2, 0)
    minor_radius, elongation, triangularity = 0.5, 1.5, 0.3

    profile = PlasmaProfile(point_number=201).limiter(
        *geometric_axis, minor_radius, elongation, triangularity
    )
    shape = LCFS(profile.points)

    profile.plot()
    shape.plot(True)

    # square = Squareness(profile.points,
    #                    shape.z_rmax, shape.r_zmax, shape.z_rmin, shape.r_zmin)
    # square.plot()
