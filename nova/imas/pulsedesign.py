"""Generate feed-forward coil current waveforms from pulse schedule IDS."""
from dataclasses import dataclass, field
from functools import cached_property
from typing import ClassVar

import numpy as np
from scipy.optimize import minimize

from nova.frame.baseplot import Plot
from nova.geometry.separatrix import Separatrix
from nova.imas.pulseschedule import PulseSchedule


@dataclass
class ControlPoint(PulseSchedule):
    """Build control points from pulse schedule data."""

    point_attrs: ClassVar[list[str]] = ['outer', 'upper', 'inner', 'lower',
                                        'inner_strike', 'outer_strike']

    @property
    def control_points(self):
        """Return point array."""
        return np.c_[[getattr(self, attr) for attr in self.point_attrs
                      if not np.allclose(getattr(self, attr), (0, 0))]]

    @property
    def axis(self):
        """Return location of geometric axis."""
        return self['geometric_axis']

    @property
    def minor_radius(self):
        """Return minor radius."""
        return self['minor_radius']

    @property
    def elongation(self):
        """Return elongation."""
        return self['elongation']

    @property
    def triangularity_upper(self):
        """Return upper triangularity."""
        return self['triangularity_upper']

    @property
    def triangularity_lower(self):
        """Return lower triangularity."""
        return self['triangularity_lower']

    @property
    def triangularity_outer(self):
        """Return outer triangularity."""
        return 0

    @property
    def triangularity_inner(self):
        """Return inner triangularity."""
        return 0

    @property
    def outer(self):
        """Return outer control point."""
        return self.axis + self.minor_radius*np.array(
            [1, self.triangularity_outer])

    @property
    def upper(self):
        """Return upper control point."""
        return self.axis + self.minor_radius*np.array(
            [-self.triangularity_upper, self.elongation])

    @property
    def inner(self):
        """Return inner control point."""
        return self.axis - self.minor_radius*np.array(
            [1, -self.triangularity_inner])

    @property
    def lower(self):
        """Return upper control point."""
        return self.axis - self.minor_radius*np.array(
            [self.triangularity_lower, self.elongation])

    @property
    def inner_strike(self):
        """Return inner strike point."""
        return self['strike_point'][0]

    @property
    def outer_strike(self):
        """Return outer strike point."""
        return self['strike_point'][1]

    @property
    def coef(self) -> dict:
        """Return plasma profile coefficents."""
        return {'geometric_radius': self.axis[0],
                'geometric_height': self.axis[1]} | {
                    attr: getattr(self, attr) for attr in
                    ['minor_radius', 'elongation',
                     'triangularity_upper', 'triangularity_lower']}

    def plot_profile(self):
        """Plot analytic profile."""
        profile = Separatrix(point_number=121)
        profile.limiter(**self.coef).plot()

    def plot(self):
        """Plot control points and first wall."""
        self.get_axes('2d')
        for segment in ['wall', 'divertor']:
            self.axes.plot(self.data[segment][:, 0], self.data[segment][:, 1],
                           '-', ms=4, color='gray', linewidth=1.5)

        constraint = PointConstraint(self.control_points)
        constraint.poloidal_flux = 0
        constraint.radial_field = 0, [0, 2, 3]
        constraint.vertical_field = 0, [1, 3]
        constraint.plot()


@dataclass
class ConstraintData:
    """Manage masked constraint data."""

    point_number: int
    array: np.ndarray = field(init=False)
    mask: np.ndarray = field(init=False)

    def __post_init__(self):
        """Initialize data and mask arrays."""
        self.array = np.zeros(self.point_number, float)
        self.mask = np.ones(self.point_number, bool)

    def __len__(self):
        """Return constraint number."""
        return np.sum(~self.mask)

    def update(self, data, index=None):
        """Update constraint."""
        if index is None:
            index = self.point_index
        self.array[index] = data
        self.mask[index] = False

    @cached_property
    def point_index(self):
        """Return full point index."""
        return np.arange(self.point_number)

    @property
    def index(self):
        """Return select point index."""
        return self.point_index[~self.mask]

    @property
    def data(self):
        """Return select data."""
        return self.array[~self.mask]


@dataclass
class PointConstraint(Plot):
    """Manage flux and field constraints."""

    points: np.ndarray
    constraint: dict[str, ConstraintData] = \
        field(init=False, default_factory=dict)

    attrs: ClassVar[list[str]] = ['psi', 'br', 'bz']

    def __post_init__(self):
        """Initialize constraint data."""
        super().__post_init__()
        for attr in self.attrs:
            self.constraint[attr] = ConstraintData(self.point_number)

    def __len__(self):
        """Return contstraint number."""
        return np.sum([len(self[attr]) for attr in self.attrs])

    @cached_property
    def point_number(self):
        """Return point number."""
        return len(self.points)

    @cached_property
    def point_index(self):
        """Return full point index."""
        return np.arange(self.point_number)

    def __getitem__(self, attr: str):
        """Return constraint data."""
        return self.constraint[attr]

    def index(self, attr: str):
        """Return constraint point index."""
        if attr == 'null':
            return np.intersect1d(self['br'].index, self['bz'].index,
                                  assume_unique=True)
        if attr == 'radial':
            return self.point_index[~self['br'].mask & self['bz'].mask]
        if attr == 'vertical':
            return self.point_index[self['br'].mask & ~self['bz'].mask]
        return self[attr].index

    def _points(self, attr: str):
        """Return constraint points."""
        return self.points[self.index(attr)]

    def update(self, attr: str, constraint):
        """Update constraint."""
        match constraint:
            case (value, index):
                self[attr].update(value, index)
            case value:
                self[attr].update(value)

    @property
    def poloidal_flux(self):
        """Return poloidal flux constraints."""
        return self['psi'].data

    @poloidal_flux.setter
    def poloidal_flux(self, constraint):
        """Set poloidal flux constraint."""
        self.update('psi', constraint)

    @property
    def radial_field(self):
        """Return radial_field constraints."""
        return self['br'].data

    @radial_field.setter
    def radial_field(self, constraint):
        """Set radial field constraint."""
        self.update('br', constraint)

    @property
    def vertical_field(self):
        """Return vertical_field constraints."""
        return self['bz'].data

    @vertical_field.setter
    def vertical_field(self, constraint):
        """Set vertical field constraint."""
        self.update('bz', constraint)

    def plot(self, axes=None, ms=10, color='C2'):
        """Plot constraint."""
        self.axes = axes
        self.axes.plot(*self.points.T, 'o', color=color, ms=ms/4)
        self.axes.plot(*self._points('psi').T, 's', ms=ms, mec=color,
                       mew=1, mfc='none')
        self.axes.plot(*self._points('radial').T, '|', ms=ms, mec=color)
        self.axes.plot(*self._points('vertical').T, '_', ms=ms, mec=color)
        self.axes.plot(*self._points('null').T, 'x', ms=ms, mec=color)


if __name__ == '__main__':

    point = ControlPoint(135013, 2, 'iter', 1)

    point.itime = 13
    point.plot()
