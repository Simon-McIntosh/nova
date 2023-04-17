"""Generate feed-forward coil current waveforms from pulse schedule IDS."""
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from nova.frame.baseplot import Plot
from nova.geometry.separatrix import Separatrix
from nova.imas.pulseschedule import PulseSchedule


@dataclass
class ControlPoint(PulseSchedule):
    """Build control points from pulse schedule data."""

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
    def point_upper(self):
        """Return upper control point."""
        return self.axis + self.minor_radius*np.array(
            [-self.triangularity_upper, self.elongation])

    @property
    def point_lower(self):
        """Return upper control point."""
        return self.axis - self.minor_radius*np.array(
            [self.triangularity_lower, self.elongation])

    @property
    def point_outer(self):
        """Return outer control point."""
        return self.axis + self.minor_radius*np.array(
            [1, self.triangularity_outer])

    @property
    def point_inner(self):
        """Return inner control point."""
        return self.axis - self.minor_radius*np.array(
            [1, -self.triangularity_inner])

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
        self.axes.plot(*self.axis, '+')
        self.axes.plot(*self.point_upper, 'o')
        self.axes.plot(*self.point_lower, 'o')
        self.axes.plot(*self.point_outer, 'o')
        self.axes.plot(*self.point_inner, 'o')


@dataclass
class PointConstraint(Plot):
    """Manage point-wise flux and field constraints."""

    radius: float
    height: float
    psi: bool = True
    radial_field: bool | int | float = False
    vertical_field: bool | int | float = False

    def _field_marker(self, ):
        """Return field marker."""
        match self.radial_field, self.vertical_field:
            case (False, False):
                return None
            case (True, True):
                return 'x'
            case (True, False) | (int() | float(), 0 | False):
                return '|'
            case (False, True) | (0 | False, int() | float()):
                return '_'
            case (float() | int(), float() | int()):
                return '1'
            case _:
                raise ValueError('malformed field components')

    def plot(self, axes=None):
        """Plot constraint."""
        self.axes = axes
        self.axes.plot(self.radius, self.height, 'k.', ms=6)
        if self.psi:
            self.axes.plot(self.radius, self.height, 's',
                           ms=12, mec='C2', mew=1, mfc='none')
        if (marker := self._field_marker()) is not None:
            self.axes.plot(self.radius, self.height, marker,
                           ms=12, color='C2')


@dataclass
class Constraint(Plot):
    """Manage multiple field and flux constraints."""

    points: np.ndarray = field(init=False)

    def append(self, radius, height, point_type)


if __name__ == '__main__':

    point = ControlPoint(135013, 2, 'iter', 1)

    point.itime = 20
    #point.plot()

    point_constraint = PointConstraint(point.point_upper[0],
                                       point.point_upper[1], False, 0, 0).plot()
