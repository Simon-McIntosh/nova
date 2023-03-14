"""Generate of an artifical an separatrix from shape parameters."""
import bisect
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
from scipy.interpolate import interp1d

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
class UpDown:
    """Manage access to upper/lower type plasma shape parameters."""

    attr: str
    data: dict[str, float]

    def __post_init__(self):
        """Check dataset for self-consistency."""
        self.check_consistency()

    def check_consistency(self):
        """Check data consistency."""
        if all(attr in self.data for attr in
               [self.attr, self.upper_attr, self.lower_attr]):
            assert np.isclose(self.mean, (self.upper + self.lower) / 2)

    @property
    def upper_attr(self) -> str:
        """Return upper attribute name."""
        return f'upper_{self.attr}'

    @property
    def lower_attr(self) -> str:
        """Return lower attribute name."""
        return f'lower_{self.attr}'

    @property
    def mean(self):
        """Return mean attribute."""
        match self.data:
            case {self.attr: mean}:
                return mean
            case {self.upper_attr: upper, self.lower_attr: lower}:
                return (upper + lower) / 2
            case {self.upper_attr: upper}:
                return upper
            case {self.lower_attr: lower}:
                return lower
            case _:
                raise KeyError('attributes required to reconstruct '
                               f'mean {self.attr} not found in {self.data}')

    @property
    def upper(self):
        """Return upper attribute."""
        match self.data:
            case {self.upper_attr: upper}:
                return upper
            case {self.lower_attr: lower}:
                return 2*self.mean - lower
            case _:
                return self.mean

    @property
    def lower(self):
        """Return lower attribute."""
        match self.data:
            case {self.lower_attr: lower}:
                return lower
            case {self.upper_attr: upper}:
                return 2*self.mean - upper
            case _:
                return self.mean


@dataclass
class PlasmaShape:
    """Manage plasma shape parameters."""

    radius: float = 0
    height: float = 0
    data: dict[str, float] = field(default_factory=dict)
    kappa: UpDown = field(init=False, repr=False)
    delta: UpDown = field(init=False, repr=False)

    def __post_init__(self):
        """Initialise updown."""
        self.kappa = UpDown('elongation', self.data)
        self.delta = UpDown('triangularity', self.data)

    def __call__(self, *args, **kwargs):
        """Update plasma shape."""
        self.data['geometric_axis'] = \
            np.array([self.radius, self.height], float)
        attrs = ['minor_radius', 'elongation', 'triangularity']
        if isinstance(args[0], tuple):
            attrs = ['geometric_axis'] + attrs
        self.data |= kwargs | {attr: arg for arg, attr in zip(args, attrs)}
        self['geometric_axis'] = np.array(self.geometric_axis, float)
        self.kappa = UpDown('elongation', self.data)
        self.delta = UpDown('triangularity', self.data)
        self.radius, self.height = self.geometric_axis
        return self

    def __getattr__(self, attr):
        """Return unspecified attributes directly from data dict."""
        return self.data[attr]

    def __getitem__(self, attr):
        """Return data attribute."""
        return self.data[attr]

    def __setitem__(self, attr, value):
        """Update data attribute."""
        self.data[attr] = value

    def check_consistency(self):
        """Check data consistency."""
        for attr in ['kappa', 'delta']:
            getattr(self, attr).check_consistency()

    @property
    def x_point(self):
        """Manage x-point."""
        return self['x_point']

    @x_point.setter
    def x_point(self, x_point):
        """Adjust lower elongation and triangularity to match x_point."""
        if x_point is None:
            return
        self['lower_triangularity'] = \
            (self.geometric_axis[0] - self.x_point[0]) / self.minor_radius
        if 'triangularity' in self.data:
            self['upper_triangularity'] = self['triangularity']
            del self.data['triangularity']
        self['lower_elongation'] = \
            (self.geometric_axis[1] - self.x_point[1]) / self.minor_radius
        assert abs(self['lower_triangularity']) < 1
        self.check_consistency()

    @property
    def elongation(self):
        """Return plasma elongation."""
        return self.kappa.mean

    @property
    def upper_elongation(self):
        """Return upper plasma elongation."""
        return self.kappa.upper

    @property
    def lower_elongation(self):
        """Return lower plasma elongation."""
        return self.kappa.lower

    @property
    def triangularity(self):
        """Return plasma triangularity."""
        return self.delta.mean

    @property
    def upper_triangularity(self):
        """Return plasma triangularity."""
        return self.delta.upper

    @property
    def lower_triangularity(self):
        """Return plasma triangularity."""
        return self.delta.lower

    def adjust_lower_elongation(self):
        """Adjust lower elongation for single-null compliance."""
        if self.lower_elongation < (min_kappa :=
                                    2*(1 - self.lower_triangularity**2)**0.5):
            delta_kappa = 1e-3 + min_kappa - self.lower_elongation
            self['lower_elongation'] = self.lower_elongation + delta_kappa
            self.geometric_axis[1] += self.minor_radius * delta_kappa
            self.height = self.geometric_axis[1]
            self.kappa.check_consistency()


@dataclass
class Separatrix(Plot, PlasmaShape):
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
    def points(self) -> np.ndarray:
        """Return profile r,z points."""
        assert isinstance(self.geometric_axis, np.ndarray)
        return self.profile + self.geometric_axis[np.newaxis, :]

    @points.setter
    def points(self, points):
        """Update zero centered point array."""
        delta = np.linalg.norm(points[1:] - points[:-1], axis=1)
        length = np.append(0, np.cumsum(delta))
        linspace = np.linspace(0, length[-1], self.point_number)
        self.theta = interp1d(length, self.theta, 'linear', 0)(linspace)
        self.profile[:] = interp1d(length, points, 'quadratic', 0)(linspace)

    @cached_property
    def theta(self):
        """Return full theta array."""
        return np.linspace(0, 2*np.pi, self.point_number)

    @cached_property
    def upper_point_number(self):
        """Return number of upper profile points."""
        return bisect.bisect_right(self.theta, np.pi)

    @property
    def x_point_number(self):
        """Return number of profile points to the lower x-point."""
        return bisect.bisect_right(self.theta, self['theta_o'] + np.pi)

    @cached_property
    def theta_upper(self):
        """Return upper theta array 0<=theta<np.pi."""
        return self.theta[:self.upper_point_number]

    @property
    def theta_lower_hfs(self):
        """Return lower theta array on the high field side."""
        return self.theta[self.upper_point_number:self.x_point_number]

    @property
    def theta_lower_lfs(self):
        """Return lower theta array on the low field side."""
        return np.linspace(-self['theta_o'], 0,
                           self.point_number - self.x_point_number)

    @staticmethod
    def miller_profile(theta, minor_radius, elongation, triangularity):
        """Return Miller profile."""
        return minor_radius * np.c_[
            np.cos(theta + np.arcsin(triangularity) * np.sin(theta)),
            elongation * np.sin(theta)]

    def limiter(self, *args, **kwargs):
        """Update points - symetric limiter."""
        self(*args, **kwargs)
        self.points = self.miller_profile(self.theta, self.minor_radius,
                                          self.elongation, self.triangularity)
        return self

    def single_null(self, *args, **kwargs):
        """Update points - lower single null."""
        self(*args, **kwargs)
        self.x_point = kwargs.get('x_point', None)
        self.adjust_lower_elongation()
        upper = self.miller_profile(
            self.theta_upper, self.minor_radius,
            self.upper_elongation, self.upper_triangularity)
        x_i = (1 - self.lower_triangularity**2)**0.5 / \
            (self.lower_elongation - (1 - self.lower_triangularity**2)**0.5)
        k_o = self.lower_elongation / (1 - x_i**2)**0.5
        x_1 = (x_i - self.lower_triangularity) / (1 - x_i)
        x_2 = (x_i + self.lower_triangularity) / (1 - x_i)
        self.data['theta_o'] = np.arctan((1 - x_i**2)**0.5 / x_i)
        lower_hfs = self.minor_radius * np.c_[
            x_1 + (1 + x_1) * np.cos(self.theta_lower_hfs),
            k_o * np.sin(self.theta_lower_hfs)]
        lower_lfs = self.minor_radius * np.c_[
            -x_2 + (1 + x_2) * np.cos(self.theta_lower_lfs),
            k_o * np.sin(self.theta_lower_lfs)]
        self.points = np.vstack((lower_lfs[:-1], upper,
                                 lower_hfs, lower_lfs[0]))
        return self

    def plot(self):
        """Plot last closed flux surface."""
        self.get_axes('2d')
        self.axes.plot(*self.points.T, '-', lw=1.5, color='C6')
        if 'x_point' in self.data:
            self.axes.plot(*self.x_point, 'x',
                           ms=6, mec='C3', mew=1, mfc="none")


if __name__ == '__main__':

    separatrix = Separatrix().single_null(
        (5, 0), 2, 1.8, 0.3, x_point=(4, -4.5))
    separatrix.plot()
