"""Generate of an artifical an separatrix from shape parameters."""
import bisect
from dataclasses import dataclass, field
from functools import cached_property, wraps
from typing import ClassVar

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar

from nova.frame.baseplot import Plot


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
    kind: str = 'quadratic'

    def __post_init__(self):
        """Wrap inputs."""
        if self.pad_width > 0:
            segments = self.length[1:] - self.length[:-1]
            segments = np.pad(segments, self.pad_width, 'wrap')
            self.length = np.append(0, np.cumsum(segments))
            self.length /= self.length[-1]
            self.value = np.pad(self.value, self.pad_width, 'wrap')

    def __call__(self, length):
        """Return call to interpolator."""
        return self.interpolator(length)

    @cached_property
    def interpolator(self):
        """Return 1d interpolator."""
        return interp1d(self.length, self.value, self.kind)

    @staticmethod
    def _minimize(fun):
        """Wrap scipy minimize_scalar."""
        return minimize_scalar(fun, bounds=(0, 1), method='bounded')

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
class LCFS(Plot):
    """Calculate plasma shape parameters from the last closed flux surface."""

    points: np.ndarray

    def __call__(self, attrs: list[str]):
        """Return attribute shape vector."""
        return np.array([getattr(self, attr) for attr in attrs])

    @cached_property
    def segment_length(self):
        """Return separatrix segment lengths."""
        return np.linalg.norm(self.points[1:] - self.points[:-1], axis=1)

    @cached_property
    def cumaulative_length(self):
        """Return normalized cumaulative separatrix length."""
        return np.append(0, np.cumsum(self.segment_length)) / self.length

    @cached_property
    def radius(self):
        """Return surface radius interpolator."""
        return Peak(self.cumaulative_length, self.points[:, 0])

    @cached_property
    def height(self):
        """Return surface height interpolator."""
        return Peak(self.cumaulative_length, self.points[:, 1])

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
    def geometric_radius(self):
        """Return geometric radius, Rgeo."""
        return (self.r_max + self.r_min) / 2

    @cached_property
    def geometric_height(self):
        """Return geometric height, Zgeo."""
        return (self.z_max + self.z_min) / 2

    @cached_property
    def minor_radius(self):
        """Return minor radius, a."""
        return (self.r_max - self.r_min) / 2

    @cached_property
    def z_max(self):
        """Return maximum height, Zmax."""
        return self.height.max_value

    @cached_property
    def z_min(self):
        """Return minimum height, Zmin."""
        return self.height.min_value

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

    def plot(self, label=False):
        """Plot last closed flux surface and key geometrical points."""
        self.get_axes('2d')
        self.axes.plot(*self.points.T, color='k', alpha=0.25)
        if label:
            self.axes.plot(self.r_max, self.z_rmax, 'o', label='Rmax')
            self.axes.plot(self.r_min, self.z_rmin, 'o', label='Rmin')
            self.axes.plot(self.geometric_radius,
                           (self.z_max + self.z_min) / 2, 'o', label='Rgeo')
            self.axes.plot(self.r_zmax, self.z_max, 'o', label='Zmax')
            self.axes.plot(self.r_zmin, self.z_min, 'o', label='Zmin')
            self.axes.legend(loc='center')


@dataclass
class UpDown:
    """Manage access to upper/lower type plasma shape parameters."""

    segment: str
    coef: dict[str, float]

    def __post_init__(self):
        """Check dataset for self-consistency."""
        self.check_consistency()

    def check_consistency(self):
        """Check data consistency."""
        if all(attr in self.coef for attr in
               [self.segment, self.upper_attr, self.lower_attr]):
            assert np.isclose(self.mean, (self.upper + self.lower) / 2)

    @property
    def upper_attr(self) -> str:
        """Return upper attribute name."""
        return f'{self.segment}_upper'

    @property
    def lower_attr(self) -> str:
        """Return lower attribute name."""
        return f'{self.segment}_lower'

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
                raise KeyError('attributes required to reconstruct '
                               f'mean {self.segment} not found in {self.coef}')

    @property
    def upper(self):
        """Return upper attribute."""
        match self.coef:
            case {self.upper_attr: upper}:
                return upper
            case {self.lower_attr: lower}:
                return 2*self.mean - lower
            case _:
                return self.mean

    @property
    def lower(self):
        """Return lower attribute."""
        match self.coef:
            case {self.lower_attr: lower}:
                return lower
            case {self.upper_attr: upper}:
                return 2*self.mean - upper
            case _:
                return self.mean


@dataclass
class PlasmaProfile:
    """Generate plasma profile from plasma parameters."""

    coef: dict[str, float] = field(default_factory=dict)
    plasma_shape: dict[str, UpDown] = field(init=False, default_factory=dict)

    profile_attrs: ClassVar[list[str]] = \
        ['geometric_radius', 'geometric_height',
         'minor_radius', 'elongation', 'triangularity']

    def __post_init__(self):
        """Initialise updown."""
        if hasattr(super(), '__post_init__'):
            super().__post_init__()
        self.plasma_shape['kappa'] = UpDown('elongation', self.coef)
        self.plasma_shape['delta'] = UpDown('triangularity', self.coef)

    def update_coefficents(self, *args, **kwargs):
        """Update plasma profile coefficients."""
        self.coef = kwargs
        self.coef |= {attr: arg for attr, arg in zip(self.profile_attrs, args)}
        self.plasma_shape['kappa'] = UpDown('elongation', self.coef)
        self.plasma_shape['delta'] = UpDown('triangularity', self.coef)
        return self

    @property
    def minor_radius(self):
        """Return minor radius."""
        return self['minor_radius']

    @property
    def geometric_radius(self):
        """Return geometric raidus."""
        return self['geometric_radius']

    @property
    def geometric_height(self):
        """Return geometric height."""
        return self['geometric_height']

    @property
    def x_point(self):
        """Return x-point."""
        return self['x_point']

    def __getitem__(self, attr):
        """Return attribute from coef if present else from pulse data."""
        if attr in self.coef:
            return self.coef[attr]
        if hasattr(super(), '__getitem__'):
            return super().__getitem__(attr)

    def __setitem__(self, attr, value):
        """Update coef attribute."""
        self.coef[attr] = value

    def check_consistency(self):
        """Check data consistency."""
        for attr in ['kappa', 'delta']:
            self.plasma_shape[attr].check_consistency()

    def set_x_point(self, x_point):
        """Adjust lower elongation and triangularity to match x_point."""
        if x_point is None:
            return
        self['triangularity_lower'] = \
            (self.geometric_radius - self.x_point[0]) / self.minor_radius
        if 'triangularity' in self.coef:
            self['triangularity_upper'] = self['triangularity']
            del self.coef['triangularity']
        self['elongation_lower'] = \
            (self.geometric_height - self.x_point[1]) / self.minor_radius
        assert abs(self['triangularity_lower']) < 1
        self.check_consistency()

    @property
    def elongation(self):
        """Return plasma elongation."""
        return self.plasma_shape['kappa'].mean

    @property
    def elongation_upper(self):
        """Return upper plasma elongation."""
        return self.plasma_shape['kappa'].upper

    @property
    def elongation_lower(self):
        """Return lower plasma elongation."""
        return self.plasma_shape['kappa'].lower

    @property
    def triangularity(self):
        """Return plasma triangularity."""
        return self.plasma_shape['delta'].mean

    @property
    def triangularity_upper(self):
        """Return plasma triangularity."""
        return self.plasma_shape['delta'].upper

    @property
    def triangularity_lower(self):
        """Return plasma triangularity."""
        return self.plasma_shape['delta'].lower

    def adjust_elongation_lower(self):
        """Adjust lower elongation for single-null compliance."""
        if self.elongation_lower < (min_kappa :=
                                    2*(1 - self.triangularity_lower**2)**0.5):
            delta_kappa = 1e-3 + min_kappa - self.elongation_lower
            self['elongation_lower'] = self.elongation_lower + delta_kappa
            self['geometric_height'] += self.minor_radius * delta_kappa
            self.plasma_shape['kappa'].check_consistency()


@dataclass
class Separatrix(Plot, PlasmaProfile):
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
        return np.array([self['geometric_radius'],
                         self['geometric_height']], float)

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
            self.profile[:] = interp1d(length, points,
                                       'quadratic', 0)(linspace)
        except ValueError:
            self.profile[:] = np.zeros((self.point_number, 2))

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
        self.update_coefficents(*args, **kwargs)
        self.points = self.miller_profile(self.theta, self.minor_radius,
                                          self.elongation, self.triangularity)
        return self

    def single_null(self, *args, **kwargs):
        """Update points - lower single null."""
        self.update_coefficents(*args, **kwargs)
        self.set_x_point(kwargs.get('x_point', None))
        self.adjust_elongation_lower()
        upper = self.miller_profile(
            self.theta_upper, self.minor_radius,
            self.elongation_upper, self.triangularity_upper)
        x_i = (1 - self.triangularity_lower**2)**0.5 / \
            (self.elongation_lower - (1 - self.triangularity_lower**2)**0.5)
        k_o = self.elongation_lower / (1 - x_i**2)**0.5
        x_1 = (x_i - self.triangularity_lower) / (1 - x_i)
        x_2 = (x_i + self.triangularity_lower) / (1 - x_i)
        self.coef['theta_o'] = np.arctan((1 - x_i**2)**0.5 / x_i)
        lower_hfs = self.minor_radius * np.c_[
            x_1 + (1 + x_1) * np.cos(self.theta_lower_hfs),
            k_o * np.sin(self.theta_lower_hfs)]
        lower_lfs = self.minor_radius * np.c_[
            -x_2 + (1 + x_2) * np.cos(self.theta_lower_lfs),
            k_o * np.sin(self.theta_lower_lfs)]
        self.points = np.vstack((lower_lfs[:-1], upper,
                                 lower_hfs, lower_lfs[0]))
        return self

    def plot(self, axes=None, **kwargs):
        """Plot last closed flux surface."""
        self.get_axes('2d', axes)
        self.axes.plot(*self.points.T, '-', lw=1.5, color='C6')


if __name__ == '__main__':

    geometric_axis = (5.2, 0)
    minor_radius, elongation, triangularity = 0.5, 1.4, 0.3
    profile = Separatrix(point_number=21).limiter(
        *geometric_axis, minor_radius, elongation, triangularity)
    shape = LCFS(profile.points)

    profile.plot()
    shape.plot(True)
