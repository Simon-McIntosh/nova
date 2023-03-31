"""Generate of an artifical an separatrix from shape parameters."""
import bisect
from dataclasses import dataclass, field
from functools import cached_property
from typing import ClassVar

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.spatial import KDTree

from nova.frame.baseplot import Plot
from nova.imas.pulse_schedule import PulseSchedule


@dataclass
class PlasmaShape(Plot):
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
        return f'upper_{self.segment}'

    @property
    def lower_attr(self) -> str:
        """Return lower attribute name."""
        return f'lower_{self.segment}'

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
        self['lower_triangularity'] = \
            (self.geometric_radius - self.x_point[0]) / self.minor_radius
        if 'triangularity' in self.coef:
            self['upper_triangularity'] = self['triangularity']
            del self.coef['triangularity']
        self['lower_elongation'] = \
            (self.geometric_height - self.x_point[1]) / self.minor_radius
        assert abs(self['lower_triangularity']) < 1
        self.check_consistency()

    @property
    def elongation(self):
        """Return plasma elongation."""
        return self.plasma_shape['kappa'].mean

    @property
    def upper_elongation(self):
        """Return upper plasma elongation."""
        return self.plasma_shape['kappa'].upper

    @property
    def lower_elongation(self):
        """Return lower plasma elongation."""
        return self.plasma_shape['kappa'].lower

    @property
    def triangularity(self):
        """Return plasma triangularity."""
        return self.plasma_shape['delta'].mean

    @property
    def upper_triangularity(self):
        """Return plasma triangularity."""
        return self.plasma_shape['delta'].upper

    @property
    def lower_triangularity(self):
        """Return plasma triangularity."""
        return self.plasma_shape['delta'].lower

    def adjust_lower_elongation(self):
        """Adjust lower elongation for single-null compliance."""
        if self.lower_elongation < (min_kappa :=
                                    2*(1 - self.lower_triangularity**2)**0.5):
            delta_kappa = 1e-3 + min_kappa - self.lower_elongation
            self['lower_elongation'] = self.lower_elongation + delta_kappa
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
        self.adjust_lower_elongation()
        upper = self.miller_profile(
            self.theta_upper, self.minor_radius,
            self.upper_elongation, self.upper_triangularity)
        x_i = (1 - self.lower_triangularity**2)**0.5 / \
            (self.lower_elongation - (1 - self.lower_triangularity**2)**0.5)
        k_o = self.lower_elongation / (1 - x_i**2)**0.5
        x_1 = (x_i - self.lower_triangularity) / (1 - x_i)
        x_2 = (x_i + self.lower_triangularity) / (1 - x_i)
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


@dataclass
class LCFS(Separatrix, PulseSchedule):
    """Fit Last Closed Flux Surface to Pulse Schedule parameters."""

    gap_head: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        """Extract geometric axis from pulse schedule dataset."""
        super().__post_init__()
        self.extract_geometric_axis()

    def extract_geometric_axis(self):
        """Extract geometric radius and height from geometric axis."""
        self.data['geometric_radius'] = self.data.geometric_axis[:, 0]
        self.data['geometric_height'] = self.data.geometric_axis[:, 1]

    @cached_property
    def gap_tail(self) -> np.ndarray:
        """Return gap tail from pulse schedule dataset."""
        return self.data.gap_tail.data

    @cached_property
    def gap_vector(self):
        """Return gap vactor from pulse schedule dataset."""
        return self.data.gap_vector.data

    def update(self):
        """Extend GetSlice.update to include gap_head calculation."""
        super().update()
        try:  # clear cached property
            delattr(self, 'topology')
        except AttributeError:
            pass
        self.gap_head = self.get('gap')[:, np.newaxis] * self.gap_vector
        self.gap_head += self.gap_tail

    @cached_property
    def hfs_radius(self):
        """Return first wall radius on the high field side."""
        return self.wall_segment[:, 0].min()

    @cached_property
    def lfs_gap_index(self):
        """Return low field side gap index."""
        return self.gap_tail[:, 0] > self.hfs_radius + 1e-3

    @cached_property
    def topology(self):
        """Return plasma topology discriptor."""
        if self.get('x_point')[0, 0] < self.hfs_radius + 1e-3:
            return 'limiter'
        return 'single_null'

    def update_separatrix(self, coef: np.ndarray):
        """Update plasma boundary points."""
        if self.topology == 'limiter':
            return self.limiter(*coef)
        return self.single_null(*coef, x_point=self.get('x_point')[0])

    def kd_tree(self, points: np.ndarray) -> np.ndarray:
        """Return boundary point selection index using a 2d partition tree."""
        return KDTree(self.points).query(self.gap_head)[1]

    def kd_index(self, coef: np.ndarray) -> np.ndarray:
        """Update separatrix with coef and return gap-point selection index."""
        self.update_separatrix(coef)
        return self.kd_tree(self.points)

    def objective(self, coef: np.ndarray) -> float:
        """Return lcfs fitting objective."""
        index = self.kd_index(coef)
        error = np.linalg.norm(self.points[index, :] - self.gap_head, axis=1)
        return np.mean(error**2)

    def gap_constraint(self, coef: np.ndarray) -> np.ndarray:
        """Return lcfs fillting constraints."""
        index = self.kd_index(coef)
        gap_delta = np.einsum('ij,ij->i', self.gap_vector,
                              self.points[index, :] - self.gap_head)
        if self.topology == 'limiter':
            return gap_delta[self.lfs_gap_index]
        return gap_delta

    def limiter_constraint(self, coef: np.ndarray) -> np.ndarray:
        """Return lcfs radial hfs limiter constraint."""
        self.update_separatrix(coef)
        return np.array([self.points[:, 0].min() - self.hfs_radius])

    @property
    def constraints(self):
        """Return gap and limiter constraints."""
        gap_constraint = dict(type='ineq', fun=self.gap_constraint)
        if self.topology == 'limiter':
            limiter_constraint = dict(type='eq', fun=self.limiter_constraint)
            return [gap_constraint, limiter_constraint]
        return gap_constraint

    @property
    def bounds(self):
        """Return parameter bounds."""
        return [(-1, 1) if attr == 'triangularity' else (None, None)
                for attr in self.profile_attrs]

    @property
    def coef_o(self):
        """Return IDS profile coeffients."""
        return [self.get(attr) for attr in self.profile_attrs]

    def initialize(self):
        """Initialize analytic separatrix with pulse schedule data."""
        self.objective(self.coef_o)

    def fit(self):
        """Fit analytic separatrix to pulse schedule gaps."""
        self.initialize()
        if np.allclose(self.get('gap'), 0):
            return
        sol = minimize(self.objective, self.coef_o, method='SLSQP',
                       bounds=self.bounds, constraints=self.constraints)
        self.objective(sol.x)

    def plot(self, axes=None, **kwargs):
        """Plot first wall, gaps, and plasma profile."""
        if not np.isclose(self.geometric_radius, 0):
            super().plot(axes=axes, **kwargs)
        if self.topology == 'single_null':
            self.axes.plot(*self.get('x_point')[0], 'x',
                           ms=6, mec='C3', mew=1, mfc='none')
        self.plot_gaps()

    def _make_frame(self, time):
        """Make frame for annimation."""
        self.axes.clear()
        max_time = np.min([self.data.time[-1], self.max_time])
        try:
            self.itime = bisect.bisect_left(
                self.data.time, max_time * time / self.duration)
        except ValueError:
            pass
        self.initialize()
        self.fit()
        self.plot()
        return self.mpy.mplfig_to_npimage(self.fig)

    def annimate(self, duration: float, filename='gaps'):
        """Generate annimiation."""
        self.duration = duration
        self.max_time = 15
        self.set_axes('2d')
        animation = self.mpy.editor.VideoClip(
            self._make_frame, duration=duration)
        animation.write_gif(f'{filename}.gif', fps=10)


if __name__ == '__main__':

    pulse, run = 135003, 5
    lcfs = LCFS(pulse, run)

    lcfs.time = 11.656 - 0.5

    lcfs.fit()
    lcfs.plot()

    # lcfs.annimate(10, 'gaps_fit_limiter')

    #lcfs.plot_gaps()
    #separatrix = Separatrix().single_null(
    #    (5, 0), 2, 1.8, 0.3, x_point=(4, -4.5))
    #separatrix.plot()
