"""Generate feed-forward coil current waveforms from pulse schedule IDS."""
from dataclasses import dataclass, field
from functools import cached_property
from typing import ClassVar

import numpy as np
from scipy.optimize import minimize, newton_krylov, LinearConstraint
from scipy.sparse.linalg import LinearOperator
from scipy.spatial import distance_matrix

from nova.biot.biot import Nbiot
from nova.frame.baseplot import Plot
from nova.geometry.separatrix import Quadrant, Separatrix
from nova.imas.database import Ids
from nova.imas.equilibrium import Equilibrium
from nova.imas.machine import Machine
from nova.imas.pf_active import PF_Active
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
            return np.intersect1d(self['br'].index[self['br'].data == 0],
                                  self['bz'].index[self['bz'].data == 0],
                                  assume_unique=True)
        if attr == 'radial':
            return np.intersect1d(
                self['br'].index[self['br'].data == 0],
                self.point_index[self['bz'].mask], assume_unique=True)
        if attr == 'vertical':
            return np.intersect1d(
                self['bz'].index[self['bz'].data == 0],
                self.point_index[self['br'].mask], assume_unique=True)
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
                       mew=2, mfc='none')
        self.axes.plot(*self._points('radial').T, '|', ms=2*ms, mec=color)
        self.axes.plot(*self._points('vertical').T, '_', ms=2*ms, mec=color)
        self.axes.plot(*self._points('null').T, 'x', ms=2*ms, mec=color)


@dataclass
class ControlPoint(Equilibrium):
    """Build control points from pulse schedule data."""

    control: Constraint = field(init=False, default_factory=Constraint)
    strike: Constraint = field(init=False, default_factory=Constraint)

    point_attrs: ClassVar[dict[str, list[str]]] = {
        'boundary': ['outer', 'upper', 'inner', 'lower',
                     'upper_outer', 'upper_inner',
                     'lower_inner', 'lower_outer'],
        'strike': ['inner_strike', 'outer_strike']}

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
        return points#[:3]

    @property
    def strike_points(self):
        """Return strike points."""
        if self.limiter:
            return np.array([])
        return np.c_[[getattr(self, attr)
                      for attr in self.point_attrs['strike']]]

    def update_control_point(self, psi=0):
        """Update control point constraints."""
        self.control = Constraint(self.control_points)
        self.control.poloidal_flux = psi, [0, 1, 2, 3, 4, 5]
        self.control.radial_field = 0, [0, 2]
        self.control.vertical_field = 0, [1, 3]
        if not self.limiter:
            self.control.radial_field = 0, [3]

    def update_field_constraint(self):
        """Update boundary field inequality constraints."""
        self.field = Constraint(self.control_points[4:])
        #self.field.radial_field = 0, [1]
        self.field.radial_field = 0, [0]

    def update_strike_point(self, psi=0):
        """Update strike point constraints."""
        self.strike = Constraint()#self.strike_points)
        self.strike.poloidal_flux = psi

    def update_constraints(self, psi=0):
        """Update flux and field constraints."""
        self.update_control_point(psi)
        self.update_strike_point(psi)
        #self.update_field_constraint()

    def update(self):
        """Update source equilibrium."""
        super().update()
        self.update_constraints()

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
        return self['elongation_upper']  # TODO update once IDS is fixed

    @property
    def triangularity_inner(self):
        """Return inner triangularity."""
        return self['elongation_lower']  # TODO update once IDS is fixed

    @property
    def upper(self):
        """Return upper control point."""
        return self.axis + self.minor_radius*np.array(
            [-self.triangularity_upper, self.elongation])

    @property
    def lower(self):
        """Return upper control point."""
        return self.axis - self.minor_radius*np.array(
            [self.triangularity_lower, self.elongation])

    @property
    def inner(self):
        """Return inner control point."""
        return self.axis + self.minor_radius*np.array(
            [-1, self.triangularity_inner])

    @property
    def outer(self):
        """Return outer control point."""
        return self.axis + self.minor_radius*np.array(
            [1, self.triangularity_outer])

    @property
    def upper_outer(self):
        """Return upper outer control point."""
        return Quadrant(self.outer, self.upper).separatrix_point(
            self['squareness_upper_outer'])

    @property
    def upper_inner(self):
        """Return upper inner control point."""
        return Quadrant(self.inner, self.upper).separatrix_point(
            self['squareness_upper_inner'])

    @property
    def lower_inner(self):
        """Return lower inner control point."""
        return Quadrant(self.inner, self.lower).separatrix_point(
            self['squareness_lower_inner'])

    @property
    def lower_outer(self):
        """Return lower outer control point."""
        return Quadrant(self.outer, self.lower).separatrix_point(
            self['squareness_lower_outer'])

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
class ITER(Machine):
    """ITER machine description."""

    pf_active: Ids | bool | str = 'iter_md'
    pf_passive: Ids | bool | str = False
    wall: Ids | bool | str = 'iter_md'
    tplasma: str = 'hex'
    dplasma: int | float = -3000

    def __post_init__(self):
        """Disable vs3 current updates."""
        super().__post_init__()
        self.saloc['free'][-2] = False

from nova.imas.profile import Profile

@dataclass
class PulseDesign(ITER, ControlPoint, Profile):
    """Generate coilset voltage and current waveforms."""

    name: str = 'equilibrium'
    nwall: Nbiot = 10
    nlevelset: Nbiot = 3000
    ninductance: Nbiot = 0
    nforce: Nbiot = 0
    nfield: Nbiot = 0

    def update_constraints(self):
        """Extend ControlPoint.update_constraints to include psi proxy."""
        # for DINA benchmark
        super().update_constraints(-self['psi_boundary'])  # COCOS11

    def update(self):
        """Extend itime update."""
        super().update()
        self.sloc['plasma', 'Ic'] = self['ip']

    def _constrain(self, constraint, field_weight=100):
        """Return coupling matrix and vectors."""
        if len(constraint) == 0:
            return
        point_index = np.array([self.levelset.kd_query(point) for point in
                                constraint.points])
        _matrix, _vector = [], []
        for attr in constraint.attrs:
            if len(constraint[attr]) == 0:
                continue
            index = point_index[constraint[attr].index]
            matrix = getattr(self.levelset, attr.capitalize())[index]
            vector = constraint[attr].data - \
                matrix[:, self.plasma_index] * self.saloc['plasma', 'Ic']
            if attr != 'psi':
                matrix *= np.sqrt(field_weight)
                vector *= np.sqrt(field_weight)
            _matrix.append(matrix)
            _vector.append(vector)
        matrix = np.vstack(_matrix)
        vector = np.hstack(_vector)
        return matrix[:, self.saloc['free']], vector

    def _stack(self, *args):
        """Stack coupling matrix and vectors."""
        matrix = np.vstack([arg[0] for arg in args if arg is not None])
        data = np.hstack([arg[1] for arg in args if arg is not None])
        return matrix, data

    def solve_current(self):
        """Solve coil currents given flux and field targets."""
        coupling = [self._constrain(self.control),
                    self._constrain(self.strike)]
        matrix, vector = self._stack(*coupling)
        self.saloc['free', 'Ic'] = MoorePenrose(matrix, gamma=1e-5) / vector


    def fun(self, x):
        """Return optimization goal."""
        #return np.sum(self.superframe.Ic.values * self.inductance.psi)
        return np.sum(x**2)

    def hess(self, x):
        return np.zeros((len(x), len(x)))

    def optimize_current(self):
        """Optimize external coil currents."""
        coupling = [self._constrain(self.control)]
                    #self._constrain(self.strike)]
        matrix, vector = self._stack(*coupling)

        fmatrix, fvector = self._constrain(self.field)

        print(fmatrix)

        self.solve_current()

        constraints = [LinearConstraint(matrix, vector, vector),
                       LinearConstraint(fmatrix, fvector, fvector)]

        sol = minimize(self.fun, self.saloc['free', 'Ic'],
                       hess=self.hess, method='trust-constr',
                       constraints=constraints)
        self.saloc['free', 'Ic'] = sol.x

        print(self._constrain(self.field)[1])
        print(np.dot(self._constrain(self.field)[0],
                     self.saloc['free', 'Ic']))
        #print(sol)

    @property
    def psi_boundary(self):
        """Return boundary psi."""
        if self.limiter:
            return self.plasma.psi_w
        return self.plasma.psi_x

    def residual(self, xin):
        """Return psi grid residual."""
        self.plasma.nturn = xin[:-1]
        self.solve_current()
        self.plasma.separatrix = xin[-1]

        #nturn = self.plasma.nturn * self.aloc['plasma', 'x']**-1
        #self.plasma.nturn = nturn / np.sum(nturn)
        #print('a')

        xout = np.r_[self.plasma.nturn, np.sum(self.plasma.nturn)]
        residual = xout - np.r_[xin[:-1], 1]
        residual[-1] /= self.plasmagrid.number
        return residual

    def psi_residual(self, psi):  # psi on grid and wall
        """Return psi residual."""
        self.plasma.psi = psi
        self.plasma.separatrix = self.plasma.psi_boundary
        self.solve_current()
        return np.r_[self.plasmagrid.psi, self.plasmawall.psi] - psi

    def solve(self):
        """Solve waveform."""
        self.solve_current()
        psi = np.r_[self.plasmagrid.psi, self.plasmawall.psi]
        psi = newton_krylov(self.psi_residual, self.plasma.psi, verbose=True)
        self.psi_residual(psi)

    def plot(self, index=None, axes=None, **kwargs):
        """Extend plot to include plasma contours."""
        super().plot(index, axes, **kwargs)
        self.plasma.plot()


@dataclass
class Benchmark(PulseDesign):
    """Benchmark pulse design with source IDSs."""

    source_data: dict[str, Ids] = field(init=False, repr=False,
                                        default_factory=dict)

    def __post_init__(self):
        """Load source equilibrium instance."""
        self.source_data['equilibrium'] = Equilibrium(self.pulse, self.run)
        self.source_data['pf_active'] = PF_Active(self.pulse, self.run)
        super().__post_init__()

    def __getitem__(self, attr):
        """Extend getitem to include source data lookup."""
        if attr in self.source_data:
            return self.source_data[attr]
        return super().__getitem__(attr)

    def update(self):
        """Extend update to include source IDSs."""
        super().update()
        for attr in self.source_data:
            self[attr].time = self.time

    def plot(self, index=None, axes=None, **kwargs):
        """Extend plot to include source flux map and separatrix."""
        super().plot(index, axes, **kwargs)
        self['equilibrium'].plot_boundary(self.axes, 'C2')


if __name__ == '__main__':


    design = PulseDesign(135013, 2, 'iter', 1)
    #design = Benchmark(135013, 2, 'iter', 1)
    # design.strike = Constraint()
    # design.control.points[3, 1] += 0.5

    design.itime = 30
    #design.control.points[3, 0] += 0.2
    #design.control.points[3, 1] += 0.6
    #design.strike = Constraint()

    #design.saloc['free', 'Ic'] = np.array([-25165.58379627,  39487.41871743,  14124.85852006,  71118.5512022 ,
    #       -48672.04102026, -22220.38133958,  41509.18374367,  22574.05540302,
    #         7335.78074361,  81871.25182582, -70840.0819651 ])


    design.solve()

    design.plot('plasma')
    design.levelset.plot_levelset(-design['psi_boundary'], False, color='k')  # Cocos
    design.levelset.plot_levelset(design.psi_boundary, False, color='C3')
