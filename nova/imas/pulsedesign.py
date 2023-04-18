"""Generate feed-forward coil current waveforms from pulse schedule IDS."""
from dataclasses import dataclass, field
from functools import cached_property
from typing import ClassVar

import numpy as np
from scipy import optimize

from nova.biot.biot import Nbiot
from nova.frame.baseplot import Plot
from nova.geometry.separatrix import Separatrix
from nova.imas.database import Ids
from nova.imas.equilibrium import Equilibrium
from nova.imas.machine import Machine
from nova.imas.pf_active import PF_Active
from nova.imas.pulseschedule import PulseSchedule
from nova.linalg.regression import MoorePenrose


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
        index = [i for i in index if i < self.point_number]
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
class Constraint(Plot):
    """Manage flux and field constraints."""

    points: np.ndarray = field(default_factory=lambda: np.array([]))
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
        if self.point_number == 0:
            return
        self.axes = axes
        self.axes.plot(*self.points.T, 'o', color=color, ms=ms/4)
        self.axes.plot(*self._points('psi').T, 's', ms=ms, mec=color,
                       mew=1, mfc='none')
        self.axes.plot(*self._points('radial').T, '|', ms=2*ms, mec=color)
        self.axes.plot(*self._points('vertical').T, '_', ms=2*ms, mec=color)
        self.axes.plot(*self._points('null').T, 'x', ms=2*ms, mec=color)


@dataclass
class ControlPoint(PulseSchedule):
    """Build control points from pulse schedule data."""

    control: Constraint = field(init=False, default_factory=Constraint)
    strike: Constraint = field(init=False, default_factory=Constraint)

    point_attrs: ClassVar[dict[str, list[str]]] = {
        'boundary': ['outer', 'upper', 'inner', 'lower'],
        'strike': ['inner_strike', 'outer_strike']}

    def update_control_point(self, psi=0):
        """Update control point constraints."""
        self.control = Constraint(self.control_points)
        self.control.poloidal_flux = psi
        self.control.radial_field = 0, [0, 2]
        self.control.vertical_field = 0, [1, 3]

    def update_strike_point(self, psi=0):
        """Update strike point constraints."""
        self.strike = Constraint(self.strike_points)
        self.strike.poloidal_flux = psi

    def update_constraints(self, psi=0):
        """Update flux and field constraints."""
        self.update_control_point(psi)
        self.update_strike_point(psi)

    def update(self):
        """Update source equilibrium."""
        super().update()
        self.update_constraints()

    @property
    def limiter(self) -> bool:
        """Return limiter flag."""
        return np.allclose(self['x_point'], (0, 0))

    @property
    def control_points(self):
        """Return control points."""
        points = np.c_[[getattr(self, attr)
                        for attr in self.point_attrs['boundary']]]
        if self.limiter:
            return points
        return points[:3]

    @property
    def strike_points(self):
        """Return strike points."""
        if self.limiter:
            return np.array([])
        return np.c_[[getattr(self, attr)
                      for attr in self.point_attrs['strike']]]

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

    def plot(self, index=None, axes=None, **kwargs):
        """Plot control points and first wall."""
        self.get_axes('2d', axes)
        for segment in ['wall', 'divertor']:
            self.axes.plot(self.data[segment][:, 0], self.data[segment][:, 1],
                           '-', ms=4, color='gray', linewidth=1.5)
        self.control.plot()
        self.strike.plot()


@dataclass
class PulseDesign(Machine, ControlPoint):
    """Generate coilset voltage and current waveforms."""

    pf_active: Ids | bool | str = 'iter_md'
    pf_passive: Ids | bool | str = 'iter_md'
    wall: Ids | bool | str = 'iter_md'
    tplasma: str = 'hex'
    dplasma: int | float = -2500
    ninductance: Nbiot = 0
    nlevelset: Nbiot = 2500

    def update_constraints(self):
        """Extend ControlPoint.update_constraints to include psi proxy."""
        super().update_constraints(self['loop_voltage'])  # for DINA benchmark

    def update(self):
        """Extend itime update."""
        super().update()
        self.sloc['plasma', 'Ic'] = self['i_plasma']

    def _control_point_constraint(self):
        """Return control psi matrix and data."""
        index = self.levelset.kd_query(self.points)
        Psi = self.levelset.Psi[index]
        psi = float(self['loop_psi'])*np.ones(len(Psi))
        plasma = Psi[:, self.plasma_index] * self.saloc['plasma', 'Ic']
        return Psi[:, self.saloc['coil']], psi-plasma

    def residual(self, nturn):
        """Return psi grid residual."""
        nturn /= np.sum(nturn)
        self.plasma.nturn = nturn
        self.update_gap()
        #self.update_lcfs()
        #sol = optimize.root(plasma_shape, self.saloc['coil', 'Ic'])
        #self.saloc['coil', 'Ic'] = sol.x
        #self.plasma.separatrix = self.plasma.psi_boundary
        residual = self.aloc['plasma', 'nturn'] - nturn
        return residual

    def solve(self):
        """Solve waveform."""
        nturn = optimize.newton_krylov(
            self.residual, self.aloc['plasma', 'nturn'],
            x_tol=1e-3, f_tol=1e-3, maxiter=10)
        self.residual(nturn)
        #self.aloc['plasma', 'nturn'] = nturn
        #self.update_gap()


@dataclass
class Benchmark(PulseDesign):
    """Benchmark pulse design with source IDSs."""

    equilibrium: Equilibrium = field(init=False, repr=False)
    pf_active: PF_Active = field(init=False, repr=False)

    def __post_init__(self):
        """Load source equilibrium instance."""
        self.equilibrium = Equilibrium(self.pulse, self.run, occurrence=0)
        self.pf_active = PF_Active(self.pulse, self.run, occurrence=0)
        super().__post_init__()

    def update(self):
        """Extend update to include source IDSs."""
        self.equilibrium.time = self.time
        self.pf_active.time = self.time

    def plot(self, index=None, axes=None, **kwargs):
        """Extend plot to include source flux map and separatrix."""
        super().plot(index, axes, **kwargs)
        self.equilibrium.plot_2d(axes=self.axes)
        self.equilibrium.plot_boundary(self.axes)


if __name__ == '__main__':

    #point = ControlPoint(135013, 2, 'iter', 1)
    #point.itime = 0
    #point.plot()

    design = PulseDesign(135013, 2, 'iter', 1)

    design.itime = 0
    design.plot('plasma')
