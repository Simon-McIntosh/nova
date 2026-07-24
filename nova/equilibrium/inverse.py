"""Inverse equilibrium design: boundary-shape targets to conductor currents.

The forward problem takes conductor currents to a flux map; the inverse problem
here takes a prescribed plasma boundary — a set of control points on the target
separatrix, with the flux and field conditions that hold there — and solves for
the external currents that realise it. The plasma current distribution is
carried on the same :math:`j_\\phi(\\psi_N)` profile ladder as the forward
solve, so shape targets and profile response are resolved together by Picard
iteration: solve currents against the present current distribution, re-solve
the separatrix, repeat.

All quantities are raw SI; poloidal flux is the total flux
:math:`\\Phi = 2 \\pi R A_\\phi` in Wb.
"""

from dataclasses import dataclass, field
from functools import cached_property
from typing import ClassVar

import numpy as np
from scipy.optimize import minimize, newton_krylov, LinearConstraint

from nova.geometry.plasmapoints import PlasmaPoints
from nova.graphics.plot import Plot
from nova.imas.dataset import Ids
from nova.imas.profiles import Profile
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
    constraint: dict[str, ConstraintData] = field(init=False, default_factory=dict)

    attrs: ClassVar[list[str]] = ["psi", "br", "bz"]

    def __post_init__(self):
        """Initialize constraint data."""
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
        if attr == "null":
            return np.intersect1d(
                self["br"].index[self["br"].data == 0],
                self["bz"].index[self["bz"].data == 0],
                assume_unique=True,
            )
        if attr == "radial":
            return np.intersect1d(
                self["br"].index[self["br"].data == 0],
                self.point_index[self["bz"].mask],
                assume_unique=True,
            )
        if attr == "vertical":
            return np.intersect1d(
                self["bz"].index[self["bz"].data == 0],
                self.point_index[self["br"].mask],
                assume_unique=True,
            )
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
        return self["psi"].data

    @poloidal_flux.setter
    def poloidal_flux(self, constraint):
        """Set poloidal flux constraint."""
        self.update("psi", constraint)

    @property
    def radial_field(self):
        """Return radial_field constraints."""
        return self["br"].data

    @radial_field.setter
    def radial_field(self, constraint):
        """Set radial field constraint."""
        self.update("br", constraint)

    @property
    def vertical_field(self):
        """Return vertical_field constraints."""
        return self["bz"].data

    @vertical_field.setter
    def vertical_field(self, constraint):
        """Set vertical field constraint."""
        self.update("bz", constraint)

    def plot(self, axes=None, ms=10, color="C2"):
        """Plot constraint."""
        if self.point_number == 0:
            return
        self.axes = axes
        self.axes.plot(*self._points("psi").T, "s", ms=ms, mec=color, mew=2, mfc="none")
        self.axes.plot(*self._points("radial").T, "|", ms=2 * ms, mec=color)
        self.axes.plot(*self._points("vertical").T, "_", ms=2 * ms, mec=color)
        self.axes.plot(*self._points("null").T, "x", ms=2 * ms, mec=color)


@dataclass
class Control(PlasmaPoints, Profile):
    """Extract control points and flux profiles from equilibrium data."""

    equilibrium: Ids | bool | str = True
    pf_active: Ids | bool | str = "iter_md"
    pf_passive: Ids | bool | str = False
    constraint: Constraint = field(init=False, default_factory=Constraint, repr=False)

    def update_constraints(self, psi=0):
        """Update flux and field constraints."""
        self.constraint = Constraint(self.control_points)
        self.constraint.poloidal_flux = psi, range(4)
        if self.square:
            self.constraint.poloidal_flux = psi, range(4, 8)
        if self.strike and not self.limiter:
            self.constraint.poloidal_flux = (
                psi,
                self.constraint.point_number + np.array([-2, -1]),
            )
        self.constraint.radial_field = 0, [0, 2]
        self.constraint.vertical_field = 0, [1, 3]
        if not self.limiter:
            self.constraint.radial_field = 0, [3]

    def update(self):
        """Update source equilibrium."""
        super().update()
        self.update_constraints()

    def plot(self, index=None, axes=None, **kwargs):
        """Extend PlasmaPoints.plot to include constraints."""
        super().plot(index, axes, **kwargs)
        self.constraint.plot()


@dataclass
class InverseDesign(Control):
    """Solve external conductor currents from plasma boundary-shape targets.

    The control points are the four (optionally eight) points where the target
    separatrix touches its bounding box, plus the strike points of a diverted
    boundary. Each point carries a subset of three conditions — boundary flux,
    zero radial field, zero vertical field — assembled into one overdetermined
    linear system in the free conductor currents, with the plasma column's own
    contribution moved to the right-hand side. Field rows are scaled by
    ``sqrt(field_weight)`` so field and flux residuals trade off on comparable
    terms, and the system is inverted by a Tikhonov-regularized pseudo-inverse
    whose factor scales with :math:`|I_p|`, keeping the regularization
    dimensionally consistent as the current waveform ramps.

    Because the plasma current distribution depends on the flux map that the
    solved currents produce, :meth:`solve` alternates the current solve with a
    separatrix update (Picard iteration) from an initial elliptical guess.
    :meth:`_solve` is the Newton-Krylov alternative, which drives the same
    coupled flux residual as the forward solve.

    The class is a solver mixin over a coilset: it reads the conductor coupling
    matrices (``levelset``), the free-current selection (``saloc``) and the
    plasma component from the host, and the boundary targets and profile
    coefficients from the host's time-indexed source dataset.
    """

    gamma: float = 1e-12
    field_weight: float | int = 50

    def update_constraints(self):
        """Extend Control.update_constraints to include boundary psi."""
        super().update_constraints(-self["psi_boundary"])  # COCOS

    def _constrain(self, constraint):
        """Return coupling matrix and vectors."""
        if len(constraint) == 0:
            return
        point_index = np.array(
            [self.levelset.kd_query(point) for point in constraint.points]
        )
        _matrix, _vector = [], []
        for attr in constraint.attrs:
            if len(constraint[attr]) == 0:
                continue
            index = point_index[constraint[attr].index]
            matrix = getattr(self.levelset, attr.capitalize())[index]
            vector = (
                constraint[attr].data
                - matrix[:, self.plasma_index] * self.saloc["plasma", "Ic"]
            )
            if attr != "psi":
                matrix *= np.sqrt(self.field_weight)
                vector *= np.sqrt(self.field_weight)
            _matrix.append(matrix)
            _vector.append(vector)
        matrix = np.vstack(_matrix)
        vector = np.hstack(_vector)
        return matrix[:, self.saloc["free"]], vector

    def _stack(self, *args):
        """Stack coupling matrix and vectors."""
        matrix = np.vstack([arg[0] for arg in args if arg is not None])
        data = np.hstack([arg[1] for arg in args if arg is not None])
        return matrix, data

    def solve_current(self):
        """Solve coil currents given flux and field targets."""
        coupling = [self._constrain(self.constraint)]
        matrix, vector = self._stack(*coupling)
        gamma = self.gamma * abs(self["ip"])
        self.saloc["free", "Ic"] = MoorePenrose(matrix, gamma=gamma) / vector

    def fun(self, xin, matrix, vector):
        """Return optimization goal."""
        return np.linalg.norm(matrix @ xin - vector)

    def hess(self, x):
        """Return Hessian for a linear operator."""
        return np.zeros((len(x), len(x)))

    def optimize_current(self):
        """Optimize external coil currents."""
        coupling = [self._constrain(self.constraint)]
        matrix, vector = self._stack(*coupling)
        fmatrix, fvector = self._constrain(self.field)
        self.solve_current()
        constraints = [
            LinearConstraint(matrix, vector, vector),
            LinearConstraint(fmatrix, fvector, fvector),
        ]
        sol = minimize(
            self.fun,
            self.saloc["free", "Ic"],
            hess=self.hess,
            method="trust-constr",
            constraints=constraints,
        )
        self.saloc["free", "Ic"] = sol.x

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
        xout = np.r_[self.plasma.nturn, np.sum(self.plasma.nturn)]
        residual = xout - np.r_[xin[:-1], 1]
        residual[-1] /= self.plasmagrid.number
        return residual

    def psi_residual(self, psi):
        """Return psi residual."""
        self.plasma.psi = psi
        with self.plasma.profile(self.p_prime, self.ff_prime):
            self.plasma.separatrix = self.plasma.psi_boundary
        self.solve_current()
        return np.r_[self.plasmagrid.psi, self.plasmawall.psi] - psi

    def _solve(self, verbose=True):
        """Solve waveform with Newton Krylov scheame."""
        self.solve_current()
        psi = np.r_[self.plasmagrid.psi, self.plasmawall.psi]
        psi = newton_krylov(self.psi_residual, self.plasma.psi, verbose=verbose, iter=5)
        self.psi_residual(psi)

    def solve(self, verbose=False):
        """Solve waveform using basic Picard itteration."""
        self.plasma.separatrix = {
            "ellipse": np.r_[
                self["geometric_axis"],
                1 * self["minor_radius"] * np.array([1, self["elongation"]]),
            ]
        }
        for _ in range(3):
            self.solve_current()
            with self.plasma.profile():
                self.plasma.separatrix = self.plasma.psi_boundary
        self.solve_current()
