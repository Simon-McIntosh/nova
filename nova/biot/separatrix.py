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
class UpperLower:
    """Manage access to upper/lower type plasma shape parameters."""

    attr: str
    data: dict[str, float]

    def __post_init__(self):
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

    data: dict[str, float]
    kappa: UpperLower = field(init=False)
    delta: UpperLower = field(init=False)

    def __post_init__(self):
        """Initialize upper / lower attribute selection logic."""
        self.kappa = UpperLower('elongation', self.data)
        self.delta = UpperLower('triangularity', self.data)

    def __getattr__(self, attr):
        """Return unspecified attributes directly from data."""
        return self.data[attr]

    def __getitem__(self, attr):
        """Return data attribute."""
        return self.data[attr]

    def __setitem__(self, attr, value):
        """Update data attribute."""
        self.data[attr] = value

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

    def set_single_null(self):
        """Adjust lower elongation compliant with single null topologies."""
        if self.lower_elongation < 2*(1 - self.lower_triangularity**2)**0.5:
            self['lower_elongation'] = \
                1e-3 + 2*(1 - self.lower_triangularity**2)**0.5
            self['upper_elongation'] = \
                2*self.elongation - self.lower_elongation

    def get_upper(self):
        """Return upper shape parameters."""
        return self.upper_elongation, self.upper_triangularity

    def get_lower(self):
        """Return lower shape parameters."""
        return self.lower_elongation, self.lower_triangularity


@dataclass
class Separatrix(Plot):
    """
    Generate Separatrix profiles from plasma shape parameters.

    Simple, General, Realistic, Robust, Analytic, Tokamak Equilibria I.
    Limiter and Divertor Tokamaks

    L. Guazzotto1 and J. P. Freidberg2
    """

    radius: float = 0
    height: float = 0
    point_number: int = 1000
    _points: np.ndarray = field(init=False, repr=False)
    data: dict[str, float] = field(init=False, default_factory=dict)
    attrs: list[str] = field(init=False, default_factory=list)

    def __post_init__(self):
        """Initialize point array."""
        self._points = np.zeros((self.point_number, 2))
        super().__post_init__()

    def __getitem__(self, attr):
        """Return data item."""
        return self.data[attr]

    @property
    def axis(self):
        """Return geometric axis."""
        return np.array([self.radius, self.height])

    @axis.setter
    def axis(self, axis):
        self.radius, self.height = axis

    @property
    def x_point(self):
        """Return location of lower x-point."""
        return self.data['_x-point'] + self.axis

    @x_point.setter
    def x_point(self, x_point):
        """Position plasma relitive to x-point."""
        if x_point is None:
            return
        self.radius -= (self.x_point[0] - x_point[0])
        self.height -= (self.x_point[1] - x_point[1])

    @property
    def points(self) -> np.ndarray:
        """Return profile r,z points."""
        return self._points + self.axis[np.newaxis, :]

    @points.setter
    def points(self, points):
        """Update zero centered point array."""
        delta = np.linalg.norm(points[1:] - points[:-1], axis=1)
        length = np.append(0, np.cumsum(delta))
        linspace = np.linspace(0, length[-1], self.point_number)
        self._points[:] = interp1d(length, points, 'quadratic', 0)(linspace)

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
        return bisect.bisect_right(self.theta, self['theta_o'] + np.pi)

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
        return np.linspace(-self['theta_o'], 0,
                           self.point_number - self.x_point_number)

    @staticmethod
    def miller(theta, minor_radius, elongation, triangularity):
        """Return Miller profile."""
        del_hat = np.arcsin(triangularity)
        return minor_radius * np.c_[np.cos(theta + del_hat * np.sin(theta)),
                                    elongation * np.sin(theta)]

    def plasma_shape(self, *args, **kwargs):
        """Return plasma shape instance."""
        kwargs |= {attr: arg for arg, attr in
                   zip(args, ['minor_radius', 'elongation', 'triangularity'])}
        self.attrs = list(kwargs.keys())
        return PlasmaShape(kwargs)

    def limiter(self, *args, **kwargs):
        """Update points - symetric limiter."""
        plasma = self.plasma_shape(*args, **kwargs)
        self.points = self.miller(self.theta, plasma.minor_radius,
                                  plasma.elongation, plasma.triangularity)
        return self

    def single_null(self, *args, **kwargs):
        """Update points - lower single null."""
        plasma = self.plasma_shape(*args, **kwargs)
        plasma.set_single_null()
        minor_radius = plasma.minor_radius
        kappa_u, delta_u = plasma.get_upper()
        kappa_x, delta_x = plasma.get_lower()
        upper = self.miller(self.theta_upper, minor_radius, kappa_u, delta_u)
        x_i = (1 - delta_x**2)**0.5 / (kappa_x - (1 - delta_x**2)**0.5)
        k_o = kappa_x / (1 - x_i**2)**0.5
        x_1 = (x_i - delta_x) / (1 - x_i)
        x_2 = (x_i + delta_x) / (1 - x_i)
        self.data['theta_o'] = np.arctan((1 - x_i**2)**0.5 / x_i)
        lower_hfs = minor_radius * np.c_[
            x_1 + (1 + x_1) * np.cos(self.theta_lower_hfs),
            k_o * np.sin(self.theta_lower_hfs)]
        lower_lfs = minor_radius * np.c_[
            -x_2 + (1 + x_2) * np.cos(self.theta_lower_lfs),
            k_o * np.sin(self.theta_lower_lfs)]
        self.data['_x-point'] = lower_lfs[0]
        self.points = np.vstack((lower_lfs[:-1], upper,
                                 lower_hfs, lower_lfs[0]))
        self.x_point = kwargs.get('x_point', None)
        return self

    def plot(self):
        """Plot last closed flux surface."""
        self.get_axes('2d')
        self.axes.plot(*self.points.T, 'C3')


if __name__ == '__main__':

    separatrix = Separatrix().single_null(2, 2, 0.3, lower_triangularity=0.5,
                                      x_point=(0, -0.2))
    separatrix.plot()
