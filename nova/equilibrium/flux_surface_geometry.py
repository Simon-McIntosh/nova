r"""Flux-surface-averaged geometry read from one converged flux map.

The solve publishes a field; a one-dimensional transport balance cannot
consume a field. What it consumes is the metric of the nested surface family
that field defines: how much volume each surface encloses, how fast that
volume grows with the label, and the surface averages that turn a
three-dimensional divergence into a one-dimensional flux. This module
assembles exactly that record, once per converged state, with the same
convention discipline the field itself carries.

Everything here is a host-side read of a solved map. It carries no traced
state and enters no fixed point, so a record may be assembled from any
uniform structured lattice — a solve result, a stored map, an analytic
family — without the accelerated solve's dependencies.

Surface label
-------------
Surfaces are cut on the normalised poloidal flux the topology read already
publishes,

.. math::
    \psi_N = \frac{\Phi - \Phi_a}{\Phi_b - \Phi_a},

with :math:`\Phi` the TOTAL poloidal flux in Wb pinned by
:mod:`nova.equilibrium.convention`, :math:`\Phi_a` its value on the magnetic
axis and :math:`\Phi_b` its value on the boundary. Because the label is a
ratio it is blind to the sign of the flux span, so a map that falls outward
and a map that rises outward produce the same labelling and the same record.

The published radial coordinate is the toroidal-flux radius

.. math::
    \Phi_{\mathrm{tor}}(\psi_N) = \int \!\! \int B_\phi \, \mathrm{d}A
    = \int_0^{\psi_N} q \, |\mathrm{d}\Phi|, \qquad
    \rho = \sqrt{\frac{\Phi_{\mathrm{tor}}}{\pi B_0}}, \qquad
    B_0 = \frac{F_b}{R_0},

built on a caller-supplied reference radius :math:`R_0` and the boundary
value of the toroidal-field function. Both :math:`\Phi_{\mathrm{tor}}` and
:math:`B_0` carry the sign of :math:`F`, so their ratio is positive and
:math:`\rho` is a real, increasing radial label whichever direction the
toroidal field points.

The surface average and every metric on it
------------------------------------------
The average of a quantity over one surface is the volume average of the thin
shell the surface bounds,

.. math::
    \langle A \rangle
    = \frac{\oint A \, R \, \mathrm{d}l / |\nabla \Phi|}
           {\oint R \, \mathrm{d}l / |\nabla \Phi|},

the loop running once round the closed surface in the poloidal plane. Five
line integrals of that family carry the whole record. Writing them as

.. math::
    L_1 = \oint \frac{R \, \mathrm{d}l}{|\nabla \Phi|}, \quad
    L_2 = \oint \frac{\mathrm{d}l}{R \, |\nabla \Phi|}, \quad
    G_1 = \oint |\nabla \Phi| R \, \mathrm{d}l, \quad
    G_2 = \oint \frac{|\nabla \Phi| \, \mathrm{d}l}{R}, \quad
    P = \oint R \, \mathrm{d}l,

the published quantities are

.. math::
    \frac{\mathrm{d}V}{\mathrm{d}\Phi}
        = 2 \pi L_1 \, \mathrm{sgn}(\Phi_b - \Phi_a), \qquad
    q = F L_2, \qquad
    \left\langle \frac{1}{R^2} \right\rangle = \frac{L_2}{L_1},

.. math::
    \left\langle |\nabla \rho| \right\rangle
        = \left| \frac{\mathrm{d}\rho}{\mathrm{d}\Phi} \right| \frac{P}{L_1},
    \qquad
    \left\langle |\nabla \rho|^2 \right\rangle
        = \left( \frac{\mathrm{d}\rho}{\mathrm{d}\Phi} \right)^2
          \frac{G_1}{L_1}, \qquad
    \left\langle \frac{|\nabla \rho|^2}{R^2} \right\rangle
        = \left( \frac{\mathrm{d}\rho}{\mathrm{d}\Phi} \right)^2
          \frac{G_2}{L_1},

with :math:`\mathrm{d}\rho/\mathrm{d}\Phi = q / (2 \pi B_0 \rho)` following
from :math:`\Phi_{\mathrm{tor}} = \pi B_0 \rho^2`. The volume derivative on
the published coordinate follows without ever dividing by a vanishing radius,

.. math::
    \frac{\mathrm{d}V}{\mathrm{d}\rho}
    = \frac{4 \pi^2 B_0 \rho}{F \langle R^{-2} \rangle},

which is the form used here, and the three published averages satisfy
:math:`q = F \langle R^{-2}\rangle (\mathrm{d}V/\mathrm{d}\Phi) / (2\pi)`
identically. The enclosed volume and poloidal cross-section come from the
same traced contour by the plane divergence theorem,
:math:`V = \pi \oint R^2 \mathrm{d}Z` and :math:`A = \oint R \mathrm{d}Z`,
which is an independent route to the volume the derivative above reports.

Where the factors come from
---------------------------
Only two constants appear and they are different constants. The
:math:`2\pi` in :math:`\mathrm{d}V/\mathrm{d}\Phi` and in
:math:`\mathrm{d}V/\mathrm{d}\rho` is the toroidal revolution in
:math:`\mathrm{d}V = 2 \pi R \, \mathrm{d}A`. The :math:`2\pi` that the
usual textbook safety factor carries is already gone: the field-line pitch is

.. math::
    q = \frac{1}{2\pi} \oint \frac{F \, \mathrm{d}l}{R^2 B_{\mathrm{pol}}},
    \qquad
    B_{\mathrm{pol}} = \frac{|\nabla \Phi|}{2 \pi R},

and the convention's own :math:`2\pi` in the poloidal field cancels it
exactly, leaving :math:`q = F \oint \mathrm{d}l / (R |\nabla \Phi|)`. That
cancellation is a property of carrying the TOTAL flux and is the one place
the flux convention touches this module. :math:`\mu_0` appears nowhere: the
record reads a flux map and a toroidal-field function and integrates
geometry, so no current enters it.

The toroidal-field function is the same one the source declares. It is
recovered from the diamagnetic gradient through the boundary-inward
integration pinned in :mod:`nova.equilibrium.convention`, and its sign branch
is taken from the declared boundary value, which is the direction the vacuum
toroidal field points.

Axis and edge
-------------
Both ends of the family are singular in the same removable way. At the axis
the surface shrinks to a point: the contour integrals stop being resolvable
on a lattice long before the label reaches zero, while every average they
form has a finite limit. Each average is an even function of the surface
minor radius at a smooth O-point and :math:`\psi_N` is quadratic in that
radius, so each is analytic in :math:`\psi_N`; the record therefore traces no
surface inside a declared floor and carries the averages to the axis with a
low-order fit in :math:`\psi_N` over the innermost traced surfaces. The
extensive quantities need no fit — volume, area, toroidal flux, :math:`\rho`
and :math:`\mathrm{d}V/\mathrm{d}\rho` all vanish at the axis by
construction, while :math:`\mathrm{d}V/\mathrm{d}\Phi`, :math:`q`, :math:`F`
and the four averages reach finite limits there.

At the edge the record stops one declared step short of the separatrix. A
diverted separatrix passes through an X-point where :math:`|\nabla \Phi|`
vanishes and :math:`q` diverges logarithmically, so the outermost resolvable
surface, not the separatrix, is what :math:`\rho_{\mathrm{norm}} = 1` labels.

Two records make a moving grid
------------------------------
The record is per converged state, and the label grid is an input rather than
an output. Two records built on the same requested
:math:`\rho_{\mathrm{norm}}` and the same reference radius therefore align
element for element even though the surfaces they describe have moved, which
is what a balance written at fixed normalised label needs: the boundary
radius :math:`\rho_b` and the volume derivative are then plain first
differences between the two, and :meth:`FluxSurfaceGeometry.motion` forms
them after checking the alignment. The vacuum reference is a machine
constant, so the label grid is the only thing held fixed and all of the grid
motion lives in :math:`\rho_b` and in the metric.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.constants import mu_0
from scipy.interpolate import CubicSpline, RectBivariateSpline

from nova.equilibrium.convention import flux_function_toroidal_field

if TYPE_CHECKING:
    from nova.equilibrium.conservation import FluxLattice

__all__ = [
    "FluxSurfaceGeometry",
    "GridMotion",
    "PlasmaInternalGeometry",
    "SurfaceGeometryError",
    "source_field_function",
]

#: Default number of nested surfaces in the decoder loop block, laid out on
#: ``psi_norm = k / n_surface`` for ``k = 0 .. n_surface`` with the axis row
#: (``psi_norm = 0``) carried as the repeated magnetic-axis point and the
#: outermost row the LCFS (``psi_norm = 1``).
DEFAULT_SURFACES = 11

#: Default number of stored angles on each traced loop surface.
DEFAULT_ANGLES = 64

#: Default TORAX radial face count; the profile block publishes the
#: ``n_rho + 1`` face values on ``rho_norm = k / n_rho``.
DEFAULT_N_RHO = 25

#: Innermost traced surfaces the axis fit is taken over, and its degree in
#: normalised flux. Two coefficients beyond the limit absorb the leading
#: curvature of an average without chasing the tracing noise of the smallest
#: resolvable contours.
AXIS_FIT_SURFACES = 8
AXIS_FIT_DEGREE = 2

#: Newton steps taken on the interpolant's gradient to place the axis. The
#: supplied seed comes from a different fit of the same map, so the record
#: re-solves the stationary point on its own interpolant; the flux label is
#: otherwise cut about an axis the surfaces are not nested on.
AXIS_NEWTON_STEPS = 6

#: Bisection steps per contour point. The bracket starts one ladder interval
#: wide, so this drives the surface radius to the interpolant's own floor.
BISECTION_STEPS = 50

#: Ladder samples per lattice cell along each ray. The bracket search takes
#: the FIRST level crossing outward from the axis, so the ladder only has to
#: be fine enough not to step over a surface.
LADDER_PER_CELL = 4


class SurfaceGeometryError(ValueError):
    """Raised when a flux map does not carry a resolvable surface family."""


class GridMotion(NamedTuple):
    """Rates a pair of records imply for a balance at fixed label.

    ``boundary_rate`` moves the whole coordinate — every surface at fixed
    normalised label sits at a different physical radius once the boundary
    radius changes — and ``volume_derivative_rate`` moves the metric those
    surfaces carry. A conservative balance written on the normalised label
    needs both.
    """

    interval: float
    boundary_rate: float
    volume_derivative_rate: NDArray[np.float64]


@dataclass(frozen=True)
class FluxSurfaceGeometry:
    """Flux-surface-averaged geometry of one converged equilibrium.

    Arrays are ordered from the magnetic axis outwards on the requested
    normalised toroidal-flux label and share its length. Every definition,
    every factor of :math:`2\\pi` and the sign of every flux is pinned in the
    module docstring.
    """

    rho_tor_norm: NDArray[np.float64]
    rho_tor: NDArray[np.float64]
    psi_norm: NDArray[np.float64]
    poloidal_flux: NDArray[np.float64]
    toroidal_flux: NDArray[np.float64]
    volume: NDArray[np.float64]
    area: NDArray[np.float64]
    volume_derivative: NDArray[np.float64]
    volume_flux_derivative: NDArray[np.float64]
    field_function: NDArray[np.float64]
    safety_factor: NDArray[np.float64]
    inverse_square_radius: NDArray[np.float64]
    gradient_rho: NDArray[np.float64]
    gradient_rho_squared: NDArray[np.float64]
    gradient_rho_squared_over_radius_squared: NDArray[np.float64]

    # -- the same surface family read one level down: the averages TORAX
    # consumes as its standard geometry (grad psi per radian of toroidal
    # angle, magnetic-field averages) and the fixed-shape extrema -- they are
    # all line integrals or extremum reads of the loops already traced --
    inverse_radius: NDArray[np.float64]
    gradient_psi: NDArray[np.float64]
    gradient_psi_squared: NDArray[np.float64]
    gradient_psi_squared_over_radius_squared: NDArray[np.float64]
    field_squared: NDArray[np.float64]
    inverse_field_squared: NDArray[np.float64]
    int_dl_over_bp: NDArray[np.float64]
    area_derivative: NDArray[np.float64]
    r_in: NDArray[np.float64]
    r_out: NDArray[np.float64]
    elongation: NDArray[np.float64]
    triangularity_upper: NDArray[np.float64]
    triangularity_lower: NDArray[np.float64]
    enclosed_toroidal_current: NDArray[np.float64]

    boundary_rho_tor: float
    vacuum_field: float
    reference_radius: float
    axis_flux: float
    boundary_flux: float
    magnetic_axis: tuple[float, float]

    def __post_init__(self):
        """Validate the label grid and freeze every published array."""
        size = np.size(self.rho_tor_norm)
        for name in self.profile_names():
            values = np.ascontiguousarray(getattr(self, name), dtype=float)
            if values.shape != (size,):
                raise SurfaceGeometryError(
                    f"{name} carries {values.shape} on a label grid of {size}"
                )
            if not np.all(np.isfinite(values)):
                raise SurfaceGeometryError(f"{name} is not finite")
            values.setflags(write=False)
            object.__setattr__(self, name, values)
        if np.any(np.diff(self.rho_tor_norm) <= 0.0):
            raise SurfaceGeometryError(
                "the normalised label must increase from axis to edge"
            )
        if np.any(np.diff(self.rho_tor) <= 0.0):
            raise SurfaceGeometryError(
                "the toroidal-flux radius must increase from axis to edge"
            )

    @staticmethod
    def profile_names() -> tuple[str, ...]:
        """Return the names of the arrays carried on the label grid."""
        return (
            "rho_tor_norm",
            "rho_tor",
            "psi_norm",
            "poloidal_flux",
            "toroidal_flux",
            "volume",
            "area",
            "volume_derivative",
            "volume_flux_derivative",
            "field_function",
            "safety_factor",
            "inverse_square_radius",
            "gradient_rho",
            "gradient_rho_squared",
            "gradient_rho_squared_over_radius_squared",
            "inverse_radius",
            "gradient_psi",
            "gradient_psi_squared",
            "gradient_psi_squared_over_radius_squared",
            "field_squared",
            "inverse_field_squared",
            "int_dl_over_bp",
            "area_derivative",
            "r_in",
            "r_out",
            "elongation",
            "triangularity_upper",
            "triangularity_lower",
            "enclosed_toroidal_current",
        )

    @property
    def size(self) -> int:
        """Return the number of label nodes the record carries."""
        return int(np.size(self.rho_tor_norm))

    def aligned_with(self, other: FluxSurfaceGeometry) -> bool:
        """Return whether two records difference element for element."""
        return (
            self.size == other.size
            and np.array_equal(self.rho_tor_norm, other.rho_tor_norm)
            and self.reference_radius == other.reference_radius
        )

    def motion(self, later: FluxSurfaceGeometry, interval: float) -> GridMotion:
        """Return the grid-motion rates two interval endpoints imply.

        The two records must share the label grid and the reference radius,
        because a difference taken at fixed normalised label is only a rate
        when the label means the same surface fraction at both ends.
        """
        if not self.aligned_with(later):
            raise SurfaceGeometryError(
                "records differenced at fixed label must share the label grid "
                "and the reference radius"
            )
        if not interval > 0.0:
            raise SurfaceGeometryError("the interval between records must be positive")
        return GridMotion(
            interval=float(interval),
            boundary_rate=(later.boundary_rho_tor - self.boundary_rho_tor) / interval,
            volume_derivative_rate=(later.volume_derivative - self.volume_derivative)
            / interval,
        )

    @classmethod
    def from_flux_map(
        cls,
        lattice: FluxLattice,
        flux: ArrayLike,
        field_function: Callable[[NDArray[np.float64]], ArrayLike],
        *,
        axis: tuple[float, float],
        boundary_flux: float,
        reference_radius: float | None = None,
        rho_tor_norm: ArrayLike | None = None,
        surfaces: int = 65,
        angles: int = 128,
        edge_psi_norm: float = 0.995,
        axis_psi_norm: float = 0.02,
        degree: int = 3,
    ) -> FluxSurfaceGeometry:
        """Return the geometry record of one flux map.

        ``lattice`` is read for its uniform ``radius`` and ``height`` axes
        alone. ``flux`` is the total poloidal flux [Wb] flattened in C order
        over those axes; a longer vector is truncated to them, so the solve's
        concatenated grid-and-wall state may be passed whole.
        ``field_function`` maps normalised flux to :math:`F = R B_\\phi`
        [T m].

        ``axis`` seeds the magnetic-axis search and ``boundary_flux`` is the
        flux of the bounding surface, both as the topology read publishes
        them. ``reference_radius`` defaults to the lattice mid-radius, which
        is a machine constant rather than a solved quantity and so keeps the
        toroidal-flux radius comparable between two states.

        ``surfaces`` sets the internal family, laid out uniformly in the
        square root of normalised flux so its nodes are near-uniform in
        radius, and ``angles`` sets the rays each surface is traced on.
        ``edge_psi_norm`` is the outermost surface resolved and
        ``axis_psi_norm`` the innermost; inside it the averages come from the
        axis fit.
        """
        node_radius = np.ascontiguousarray(lattice.radius, dtype=float)
        node_height = np.ascontiguousarray(lattice.height, dtype=float)
        count = node_radius.size * node_height.size
        state = np.asarray(flux, dtype=float).ravel()
        if state.size < count:
            raise SurfaceGeometryError(
                f"the flux map carries {state.size} nodes; the lattice indexes {count}"
            )
        interpolant = RectBivariateSpline(
            node_radius,
            node_height,
            state[:count].reshape(node_radius.size, node_height.size),
            kx=degree,
            ky=degree,
            s=0,
        )
        if reference_radius is None:
            reference_radius = 0.5 * (node_radius[0] + node_radius[-1])

        centre = _refine_axis(interpolant, axis, node_radius, node_height)
        axis_flux = float(interpolant.ev(*centre))
        span = float(boundary_flux) - axis_flux
        if span == 0.0:
            raise SurfaceGeometryError("the axis and the boundary carry the same flux")

        psi_norm = float(edge_psi_norm) * np.linspace(0.0, 1.0, int(surfaces)) ** 2
        traced = psi_norm >= float(axis_psi_norm)
        if int(traced.sum()) < AXIS_FIT_SURFACES + 2:
            raise SurfaceGeometryError(
                "too few surfaces lie outside the axis floor to fit its limit"
            )
        trace = _trace_surfaces(
            interpolant,
            centre,
            axis_flux,
            span,
            psi_norm[traced],
            node_radius,
            node_height,
            int(angles),
        )
        return cls._assemble(
            psi_norm,
            traced,
            trace,
            field_function,
            axis_flux=axis_flux,
            span=span,
            centre=centre,
            reference_radius=float(reference_radius),
            rho_tor_norm=rho_tor_norm,
        )

    @classmethod
    def from_equilibrium(
        cls, lattice: FluxLattice, source, equilibrium, **options
    ) -> FluxSurfaceGeometry:
        """Return the geometry record of one converged forward solve.

        The three arguments are what a solve already holds: the lattice it
        was carried on, the immutable source state it consumed and the typed
        result it returned. The toroidal-field function is rebuilt from the
        source's own diamagnetic gradient, so the record inherits the flux
        functions the solve was given rather than re-reading the field.
        """
        topology = equilibrium.topology
        axis = np.asarray(topology.axis, dtype=float)
        return cls.from_flux_map(
            lattice,
            np.asarray(equilibrium.flux, dtype=float),
            source_field_function(source, float(topology.flux_span)),
            axis=(float(axis[0]), float(axis[1])),
            boundary_flux=float(topology.boundary_flux),
            **options,
        )

    @classmethod
    def _assemble(
        cls,
        psi_norm: NDArray[np.float64],
        traced: NDArray[np.bool_],
        trace: _TracedSurfaces,
        field_function: Callable[[NDArray[np.float64]], ArrayLike],
        *,
        axis_flux: float,
        span: float,
        centre: tuple[float, float],
        reference_radius: float,
        rho_tor_norm: ArrayLike | None,
    ) -> FluxSurfaceGeometry:
        """Return the record the traced surfaces and the source imply."""
        contour = trace.integrals
        inner = psi_norm[traced]
        traced_field = np.asarray(field_function(inner), dtype=float)
        field = np.asarray(field_function(psi_norm), dtype=float)
        # the reference field is the vacuum one at the plasma boundary rather
        # than the outermost resolved surface, so it stays a machine constant
        # while the surfaces move
        boundary_field = float(np.asarray(field_function(np.ones(1)))[0])
        if not np.all(field * boundary_field > 0.0):
            raise SurfaceGeometryError(
                "the toroidal-field function changes sign across the plasma, "
                "so no real toroidal-flux radius exists"
            )
        vacuum_field = boundary_field / reference_radius

        # the two inverse-gradient loops carry every intensive quantity that
        # survives the axis, so they are the pair the fit is taken over
        inverse_gradient = _to_axis(psi_norm, inner, contour.inverse_gradient, traced)
        pitch_loop = _to_axis(psi_norm, inner, contour.pitch_loop, traced)
        safety_factor = field * pitch_loop

        # the label runs uniformly in the square root of normalised flux, so
        # the toroidal-flux integrand carries a factor that vanishes at the
        # axis and the quadrature needs no special first interval; it is
        # taken as the antiderivative of the interpolant the record resamples
        # through, so the enclosed flux is never the coarser of the two
        label = np.sqrt(psi_norm / psi_norm[-1])
        drive = 2.0 * psi_norm[-1] * abs(span) * safety_factor * label
        toroidal_flux = CubicSpline(label, drive).antiderivative()(label)
        rho_tor = np.sqrt(np.maximum(toroidal_flux / (np.pi * vacuum_field), 0.0))
        if np.any(np.diff(rho_tor) <= 0.0):
            raise SurfaceGeometryError("the enclosed toroidal flux is not monotone")

        # the enclosed integrals vanish linearly in normalised flux, so the
        # axis fit is taken on the ratio and the vanishing factor restored
        volume = _to_axis_linear(psi_norm, inner, contour.volume, traced)
        area = _to_axis_linear(psi_norm, inner, contour.area, traced)

        # the gradient family is finite at the axis only as a product of a
        # vanishing loop and a diverging label derivative, so it is formed on
        # the traced surfaces and carried in by the same fit
        rho_flux_derivative = safety_factor[traced] / (
            2.0 * np.pi * vacuum_field * rho_tor[traced]
        )
        loop = contour.inverse_gradient
        gradient_rho = _to_axis(
            psi_norm,
            inner,
            abs(rho_flux_derivative) * contour.radius_perimeter / loop,
            traced,
        )
        gradient_rho_squared = _to_axis(
            psi_norm,
            inner,
            rho_flux_derivative**2 * contour.gradient_radius / loop,
            traced,
        )
        gradient_rho_squared_over_radius_squared = _to_axis(
            psi_norm,
            inner,
            rho_flux_derivative**2 * contour.gradient_inverse_radius / loop,
            traced,
        )

        # the TORAX-read averages live one level down from the gradient_rho
        # family: per-radian poloidal-flux gradients and the magnetic-field
        # averages, all formed as ratios of the same traced loops, so each is
        # analytic in normalised flux at the axis like every other average
        inverse_radius = _to_axis(
            psi_norm,
            inner,
            contour.perimeter_inverse_gradient / contour.inverse_gradient,
            traced,
        )
        gradient_psi = _to_axis(
            psi_norm,
            inner,
            contour.radius_perimeter / (2.0 * np.pi * contour.inverse_gradient),
            traced,
        )
        gradient_psi_squared = _to_axis(
            psi_norm,
            inner,
            contour.gradient_radius / ((2.0 * np.pi) ** 2 * contour.inverse_gradient),
            traced,
        )
        gradient_psi_squared_over_radius_squared = _to_axis(
            psi_norm,
            inner,
            contour.gradient_inverse_radius
            / ((2.0 * np.pi) ** 2 * contour.inverse_gradient),
            traced,
        )
        field_squared = _to_axis(
            psi_norm,
            inner,
            (
                contour.gradient_inverse_radius / (2.0 * np.pi) ** 2
                + traced_field**2 * contour.pitch_loop
            )
            / contour.inverse_gradient,
            traced,
        )
        # 1/B^2 carries the field function inside an inverse-gradient loop, so
        # its loop is formed on the raw traced geometry rather than on a
        # precomputed one
        angle_weight = 2.0 * np.pi / trace.radius.shape[0]
        inverse_field_numerator = (
            np.sum(
                trace.radius**3
                * trace.arc
                / (
                    trace.gradient
                    * (
                        trace.gradient**2 / (2.0 * np.pi) ** 2
                        + traced_field[None, :] ** 2
                    )
                ),
                axis=0,
            )
            * angle_weight
        )
        inverse_field_squared = _to_axis(
            psi_norm,
            inner,
            inverse_field_numerator / contour.inverse_gradient,
            traced,
        )
        enclosed_toroidal_current = _to_axis_linear(
            psi_norm,
            inner,
            contour.gradient_inverse_radius / (2.0 * np.pi * mu_0),
            traced,
        )

        # the fixed-shape extrema collapse onto the axis point, so the rows
        # inside the axis floor are pinned directly instead of fitted; the
        # geometry itself (not the flux label) is the smooth coordinate
        axis_radius_value = float(centre[0])
        axis_height_value = float(centre[1])
        r_in = np.empty_like(psi_norm)
        r_out = np.empty_like(psi_norm)
        z_upper = np.empty_like(psi_norm)
        z_lower = np.empty_like(psi_norm)
        r_in[traced], r_in[~traced] = trace.r_in, axis_radius_value
        r_out[traced], r_out[~traced] = trace.r_out, axis_radius_value
        z_upper[traced], z_upper[~traced] = trace.z_upper, axis_height_value
        z_lower[traced], z_lower[~traced] = trace.z_lower, axis_height_value
        minor_radius = np.empty_like(r_in)
        major_radius_local = np.empty_like(r_in)
        minor_radius[traced] = 0.5 * (trace.r_out - trace.r_in)
        major_radius_local[traced] = 0.5 * (trace.r_out + trace.r_in)
        elongation = np.ones_like(psi_norm)
        elongation[traced] = (trace.z_upper - trace.z_lower) / (
            trace.r_out - trace.r_in
        )
        triangularity_upper = np.zeros_like(psi_norm)
        triangularity_lower = np.zeros_like(psi_norm)
        triangularity_upper[traced] = (
            major_radius_local[traced] - trace.r_upper
        ) / minor_radius[traced]
        triangularity_lower[traced] = (
            major_radius_local[traced] - trace.r_lower
        ) / minor_radius[traced]

        boundary_rho_tor = float(rho_tor[-1])
        internal = rho_tor / boundary_rho_tor
        internal[0], internal[-1] = 0.0, 1.0
        requested = (
            np.linspace(0.0, 1.0, psi_norm.size)
            if rho_tor_norm is None
            else np.ascontiguousarray(rho_tor_norm, dtype=float)
        )
        if requested.min() < 0.0 or requested.max() > 1.0:
            raise SurfaceGeometryError(
                "the requested normalised label must lie between the axis and "
                "the outermost resolved surface"
            )

        def resample(values):
            """Return one internal profile on the requested label grid.

            The internal family is dense and every profile on it is smooth,
            so the resampling is taken at an order high enough that it does
            not floor the lattice error the record is otherwise limited by.
            """
            return CubicSpline(internal, values, extrapolate=False)(requested)

        # the two loops are resampled and the metrics built from them AFTER,
        # so the published columns satisfy the pitch identity exactly rather
        # than to the accuracy of three independent resamplings; the same
        # reason publishes the enclosed toroidal flux from the radius that
        # defines it instead of resampling it separately, and writes
        # dV/drho as 4 pi^2 B_0 rho / (F <R^-2>) so the vanishing surface at
        # the axis appears as a factor rather than as a quotient
        label_flux = resample(psi_norm)
        loop_inverse = resample(inverse_gradient)
        loop_pitch = resample(pitch_loop)
        radius = requested * boundary_rho_tor
        published_field = resample(field)
        published_pitch = published_field * loop_pitch
        published_inverse_square = loop_pitch / loop_inverse
        published_volume_derivative = (
            4.0
            * np.pi**2
            * vacuum_field
            * radius
            / (published_field * published_inverse_square)
        )
        published_inverse_radius = resample(inverse_radius)
        published_gradient_psi = resample(gradient_psi)
        published_gradient_psi_squared = resample(gradient_psi_squared)
        published_gradient_psi_squared_over_radius_squared = resample(
            gradient_psi_squared_over_radius_squared
        )
        published_field_squared = resample(field_squared)
        published_inverse_field_squared = resample(inverse_field_squared)
        published_int_dl_over_bp = resample(np.abs(2.0 * np.pi * inverse_gradient))
        published_r_in = resample(r_in)
        published_r_out = resample(r_out)
        published_elongation = resample(elongation)
        published_triangularity_upper = resample(triangularity_upper)
        published_triangularity_lower = resample(triangularity_lower)
        published_enclosed_current = resample(enclosed_toroidal_current)
        published_enclosed_current[0] = 0.0
        return cls(
            rho_tor_norm=requested,
            rho_tor=radius,
            psi_norm=label_flux,
            poloidal_flux=axis_flux + label_flux * span,
            toroidal_flux=np.pi * vacuum_field * radius**2,
            volume=resample(volume),
            area=resample(area),
            volume_derivative=published_volume_derivative,
            volume_flux_derivative=2.0 * np.pi * loop_inverse * np.sign(span),
            field_function=published_field,
            safety_factor=published_pitch,
            inverse_square_radius=published_inverse_square,
            gradient_rho=resample(gradient_rho),
            gradient_rho_squared=resample(gradient_rho_squared),
            gradient_rho_squared_over_radius_squared=resample(
                gradient_rho_squared_over_radius_squared
            ),
            inverse_radius=published_inverse_radius,
            gradient_psi=published_gradient_psi,
            gradient_psi_squared=published_gradient_psi_squared,
            gradient_psi_squared_over_radius_squared=(
                published_gradient_psi_squared_over_radius_squared
            ),
            field_squared=published_field_squared,
            inverse_field_squared=published_inverse_field_squared,
            int_dl_over_bp=published_int_dl_over_bp,
            area_derivative=published_inverse_radius
            * published_volume_derivative
            / (2.0 * np.pi),
            r_in=published_r_in,
            r_out=published_r_out,
            elongation=published_elongation,
            triangularity_upper=published_triangularity_upper,
            triangularity_lower=published_triangularity_lower,
            enclosed_toroidal_current=published_enclosed_current,
            boundary_rho_tor=boundary_rho_tor,
            vacuum_field=float(vacuum_field),
            reference_radius=float(reference_radius),
            axis_flux=float(axis_flux),
            boundary_flux=float(axis_flux + span),
            magnetic_axis=(float(centre[0]), float(centre[1])),
        )

    @classmethod
    def internal_geometry(
        cls,
        lattice: FluxLattice,
        flux: ArrayLike,
        field_function: Callable[[NDArray[np.float64]], ArrayLike],
        *,
        axis: tuple[float, float],
        boundary_flux: float,
        reference_radius: float | None = None,
        n_surface: int = DEFAULT_SURFACES,
        n_theta: int = DEFAULT_ANGLES,
        n_rho: int = DEFAULT_N_RHO,
        edge_psi_norm: float = 0.995,
        axis_psi_norm: float = 0.02,
        diverted: bool = False,
    ) -> PlasmaInternalGeometry:
        """Return the fixed-shape loop block and the TORAX face profiles.

        Two grids, deliberately: the decoder loop block carries the nested
        surface *positions* at ``n_surface`` levels ``psi_norm = k / n_surface``
        (the axis row is the repeated magnetic axis, the last row the LCFS at
        ``psi_norm = 1``, traced for coordinates only because at an X-point the
        gradient is not defined there), and the profile block carries every
        surface average on the ``(n_rho + 1)`` TORAX faces
        ``rho_norm = k / n_rho`` so a consumer that interpolates its own
        profiles onto those faces reads the record without resampling error.
        """
        if int(n_surface) < 2:
            raise SurfaceGeometryError(
                "the loop block needs an axis row and at least one traced row"
            )
        if int(n_theta) < 4:
            raise SurfaceGeometryError("the loop block needs at least four angles")
        if int(n_rho) < 1:
            raise SurfaceGeometryError("the profile block needs at least one face")

        node_radius = np.ascontiguousarray(lattice.radius, dtype=float)
        node_height = np.ascontiguousarray(lattice.height, dtype=float)
        count = node_radius.size * node_height.size
        state = np.asarray(flux, dtype=float).ravel()
        if state.size < count:
            raise SurfaceGeometryError(
                f"the flux map carries {state.size} nodes; the lattice indexes {count}"
            )
        interpolant = RectBivariateSpline(
            node_radius,
            node_height,
            state[:count].reshape(node_radius.size, node_height.size),
            kx=3,
            ky=3,
            s=0,
        )
        centre = _refine_axis(interpolant, axis, node_radius, node_height)
        axis_flux = float(interpolant.ev(*centre))
        span = float(boundary_flux) - axis_flux
        if span == 0.0:
            raise SurfaceGeometryError("the axis and the boundary carry the same flux")

        # the loop block reuses the reader's own rays; the outermost row is the
        # LCFS at psi_norm 1, traced for coordinates only because the poloidal
        # gradient vanishes at an X-point there
        surface_psi_norm = np.linspace(0.0, 1.0, int(n_surface))
        trace_n_theta = 2 * int(n_theta)
        trace = _trace_surfaces(
            interpolant,
            centre,
            axis_flux,
            span,
            surface_psi_norm[1:],
            node_radius,
            node_height,
            trace_n_theta,
        )
        surface_r = np.empty((int(n_surface), int(n_theta)))
        surface_z = np.empty((int(n_surface), int(n_theta)))
        surface_r[0] = float(centre[0])
        surface_z[0] = float(centre[1])
        surface_r[1:] = trace.radius[::2].T
        surface_z[1:] = trace.height[::2].T

        # the profile block is the reader record placed directly on the TORAX
        # face grid, so a consumer interpolating its own profiles onto those
        # faces reads the record without resampling error
        record = cls.from_flux_map(
            lattice,
            flux,
            field_function,
            axis=axis,
            boundary_flux=boundary_flux,
            reference_radius=reference_radius,
            rho_tor_norm=np.linspace(0.0, 1.0, int(n_rho) + 1),
            edge_psi_norm=edge_psi_norm,
            axis_psi_norm=axis_psi_norm,
        )
        r_major = 0.5 * (float(record.r_in[-1]) + float(record.r_out[-1]))
        a_minor = 0.5 * (float(record.r_out[-1]) - float(record.r_in[-1]))
        return PlasmaInternalGeometry(
            record=record,
            surface_psi_norm=surface_psi_norm,
            surface_psi=axis_flux + surface_psi_norm * span,
            surface_r=surface_r,
            surface_z=surface_z,
            surface_angle=trace.angle[::2],
            r_major=r_major,
            a_minor=a_minor,
            b0=float(record.field_function[-1]) / r_major,
            boundary_toroidal_flux=float(record.toroidal_flux[-1]),
            n_rho=int(n_rho),
            n_theta=int(n_theta),
            n_surface=int(n_surface),
            diverted=bool(diverted),
        )


class PlasmaInternalGeometry(NamedTuple):
    """Fixed-shape loop block and TORAX face profiles of one converged map.

    ``record`` is the reader's own geometry record placed on the TORAX face
    grid and carries every profile column the frame publishes; the loop block
    is the fixed ``(n_surface, n_theta)`` surface coordinates a decoder draws
    without re-deriving them, with row 0 the repeated magnetic axis and row
    ``n_surface - 1`` the LCFS.
    """

    record: FluxSurfaceGeometry
    surface_psi_norm: NDArray[np.float64]
    surface_psi: NDArray[np.float64]
    surface_r: NDArray[np.float64]
    surface_z: NDArray[np.float64]
    surface_angle: NDArray[np.float64]
    r_major: float
    a_minor: float
    b0: float
    boundary_toroidal_flux: float
    n_rho: int
    n_theta: int
    n_surface: int
    diverted: bool


class _SurfaceIntegrals(NamedTuple):
    """Contour loops of one traced surface family, in the docstring's names."""

    inverse_gradient: NDArray[np.float64]
    pitch_loop: NDArray[np.float64]
    gradient_radius: NDArray[np.float64]
    gradient_inverse_radius: NDArray[np.float64]
    radius_perimeter: NDArray[np.float64]
    volume: NDArray[np.float64]
    area: NDArray[np.float64]

    #: ``\oint dl / |\nabla\Phi|``, the ``<1/R>`` numerator: the same
    #: inverse-gradient loop as the pitch loop without the extra ``1/R``.
    perimeter_inverse_gradient: NDArray[np.float64]


class _TracedSurfaces(NamedTuple):
    """One traced surface family: the loops plus the raw traced geometry.

    ``radius``/``height``/``arc``/``gradient`` are packed
    ``(angles, n_surface)`` in the same angle order the loops are summed over,
    and the six extrema arrays are per surface.  The raw geometry is what
    lets a later assembly read off the fixed-shape columns (R_in, R_out,
    elongation, triangularity) and the two field averages whose integrand
    carries the toroidal-field function, which the geometry-only loop can not
    hold.
    """

    integrals: _SurfaceIntegrals
    angle: NDArray[np.float64]
    radius: NDArray[np.float64]
    height: NDArray[np.float64]
    arc: NDArray[np.float64]
    gradient: NDArray[np.float64]
    r_in: NDArray[np.float64]
    r_out: NDArray[np.float64]
    z_lower: NDArray[np.float64]
    z_upper: NDArray[np.float64]
    r_lower: NDArray[np.float64]
    r_upper: NDArray[np.float64]


def source_field_function(source, flux_span: float) -> Callable:
    """Return ``F(psi_N)`` [T m] the declared source implies.

    The diamagnetic gradient is integrated inward from the boundary through
    the primitive pinned in :mod:`nova.equilibrium.convention`, on the same
    fixed node set the integral observations use, so the record and the
    receipts read one toroidal-field function. The branch of the square root
    is the declared boundary value's, which is the direction the vacuum
    toroidal field points.
    """
    # the source's gradients are typed for the traced solve; the pinned
    # integrator is imported where it is used so the geometry read itself
    # stays a numpy capability
    from nova.equilibrium.observation import gradient_tail

    boundary = float(source.boundary_field_function)

    def field(psi_norm):
        """Return the toroidal-field function at one normalised flux."""
        tail = np.asarray(gradient_tail(source.core.ff_prime, np.asarray(psi_norm)))
        squared = flux_function_toroidal_field(boundary, flux_span, tail)
        return np.sign(boundary) * np.sqrt(np.maximum(squared, 0.0))

    return field


def _axis_fit(psi_norm: NDArray[np.float64], values: NDArray[np.float64]):
    """Return the innermost-surface fit of one quantity in normalised flux.

    A flux-surface average is an even function of the surface minor radius at
    a smooth O-point and the normalised flux is quadratic in that radius, so
    the average is analytic in normalised flux and a low-order fit over the
    innermost traced surfaces carries it to the axis.
    """
    window = slice(0, min(AXIS_FIT_SURFACES, psi_norm.size))
    return np.polynomial.Polynomial.fit(
        psi_norm[window], values[window], AXIS_FIT_DEGREE
    )


def _to_axis(
    psi_norm: NDArray[np.float64],
    inner: NDArray[np.float64],
    values: NDArray[np.float64],
    traced: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Return one traced quantity extended over the untraced axis region."""
    extended = np.empty_like(psi_norm)
    extended[traced] = values
    extended[~traced] = _axis_fit(inner, values)(psi_norm[~traced])
    return extended


def _to_axis_linear(
    psi_norm: NDArray[np.float64],
    inner: NDArray[np.float64],
    values: NDArray[np.float64],
    traced: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Return one enclosed integral extended over the untraced axis region."""
    extended = np.empty_like(psi_norm)
    extended[traced] = values
    extended[~traced] = psi_norm[~traced] * _axis_fit(inner, values / inner)(
        psi_norm[~traced]
    )
    return extended


def _refine_axis(
    interpolant: RectBivariateSpline,
    seed: tuple[float, float],
    node_radius: NDArray[np.float64],
    node_height: NDArray[np.float64],
) -> tuple[float, float]:
    """Return the stationary point of the interpolant nearest the seed.

    The step is capped at one cell because a Newton step on a spline Hessian
    can leave the lattice while the gradient is still large.
    """
    step = np.array(
        [node_radius[1] - node_radius[0], node_height[1] - node_height[0]],
        dtype=float,
    )
    point = np.array(seed, dtype=float)
    for _ in range(AXIS_NEWTON_STEPS):
        gradient = np.array(
            [
                float(interpolant.ev(*point, dx=1)),
                float(interpolant.ev(*point, dy=1)),
            ]
        )
        mixed = float(interpolant.ev(*point, dx=1, dy=1))
        curvature = np.array(
            [
                [float(interpolant.ev(*point, dx=2)), mixed],
                [mixed, float(interpolant.ev(*point, dy=2))],
            ]
        )
        if float(np.linalg.det(curvature)) <= 0.0:
            raise SurfaceGeometryError(
                "the seeded axis is not an elliptic stationary point of the map"
            )
        point = point - np.clip(np.linalg.solve(curvature, gradient), -step, step)
        if not (
            node_radius[0] < point[0] < node_radius[-1]
            and node_height[0] < point[1] < node_height[-1]
        ):
            raise SurfaceGeometryError("the axis search left the lattice")
    return float(point[0]), float(point[1])


def _ray_reach(
    centre: tuple[float, float],
    cosine: NDArray[np.float64],
    sine: NDArray[np.float64],
    node_radius: NDArray[np.float64],
    node_height: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return how far each ray runs before it leaves the lattice.

    One cell is held back from every side, because the interpolant is only an
    interpolant strictly inside the data it was built on.
    """
    margin = np.array(
        [node_radius[1] - node_radius[0], node_height[1] - node_height[0]],
        dtype=float,
    )
    lower = np.array([node_radius[0], node_height[0]]) + margin
    upper = np.array([node_radius[-1], node_height[-1]]) - margin
    reach = np.full(cosine.size, np.inf)
    for index, direction in enumerate((cosine, sine)):
        safe = np.where(direction == 0.0, 1.0, direction)
        near = (lower[index] - centre[index]) / safe
        far = (upper[index] - centre[index]) / safe
        reach = np.minimum(
            reach, np.where(direction == 0.0, np.inf, np.maximum(near, far))
        )
    return reach


def _trace_surfaces(
    interpolant: RectBivariateSpline,
    centre: tuple[float, float],
    axis_flux: float,
    span: float,
    psi_norm: NDArray[np.float64],
    node_radius: NDArray[np.float64],
    node_height: NDArray[np.float64],
    angles: int,
) -> _SurfaceIntegrals:
    """Return the contour loops of one nested surface family.

    Each surface is found by casting rays from the axis and bisecting the
    first outward crossing of the level, which keeps a surface inside the
    separatrix even where the map turns over beyond it. The rays make the
    surface a smooth periodic function of the polar angle, so a uniform
    trapezoidal rule on that angle converges spectrally and the only
    discretisation left in the record is the interpolant's own.
    """
    angle = 2.0 * np.pi * np.arange(angles) / angles
    cosine, sine = np.cos(angle), np.sin(angle)
    reach = _ray_reach(centre, cosine, sine, node_radius, node_height)
    cosine, sine = cosine[:, None], sine[:, None]

    def normalised(distance):
        """Return the normalised flux at a ray distance from the axis."""
        radius = centre[0] + distance * cosine
        height = centre[1] + distance * sine
        return (interpolant.ev(radius, height) - axis_flux) / span

    samples = LADDER_PER_CELL * max(node_radius.size, node_height.size)
    ladder = reach[:, None] * np.linspace(0.0, 1.0, samples + 1)[None, :]
    climb = normalised(ladder)

    crossed = climb[:, None, :] >= psi_norm[None, :, None]
    if not bool(np.all(crossed.any(axis=-1))):
        raise SurfaceGeometryError(
            "a requested surface does not close inside the lattice; the "
            "boundary flux does not bound this map"
        )
    index = np.argmax(crossed, axis=-1)
    lower = np.take_along_axis(ladder, np.maximum(index - 1, 0), axis=1)
    upper = np.take_along_axis(ladder, index, axis=1)
    for _ in range(BISECTION_STEPS):
        middle = 0.5 * (lower + upper)
        below = normalised(middle) < psi_norm[None, :]
        lower = np.where(below, middle, lower)
        upper = np.where(below, upper, middle)
    distance = 0.5 * (lower + upper)

    radius = centre[0] + distance * cosine
    height = centre[1] + distance * sine
    radial = interpolant.ev(radius, height, dx=1)
    vertical = interpolant.ev(radius, height, dy=1)
    gradient = np.hypot(radial, vertical)

    # the surface radius turns with the angle at the rate that holds the flux
    # constant, so the tangent needs no differencing of the traced points
    turn = (
        distance
        * (radial * sine - vertical * cosine)
        / (radial * cosine + vertical * sine)
    )
    arc = np.hypot(turn, distance)
    vertical_step = turn * sine + distance * cosine
    weight = 2.0 * np.pi / angles

    def loop(integrand):
        """Return one closed line integral over the traced angles."""
        return weight * np.sum(integrand, axis=0)

    # the fixed-shape columns come straight off the traced points, so the
    # family also carries the per-surface extrema and the raw geometry the
    # field averages need
    lower = np.argmin(height, axis=0)
    upper = np.argmax(height, axis=0)
    r_in = np.min(radius, axis=0)
    r_out = np.max(radius, axis=0)
    return _TracedSurfaces(
        integrals=_SurfaceIntegrals(
            inverse_gradient=loop(radius * arc / gradient),
            pitch_loop=loop(arc / (radius * gradient)),
            gradient_radius=loop(gradient * radius * arc),
            gradient_inverse_radius=loop(gradient * arc / radius),
            radius_perimeter=loop(radius * arc),
            volume=np.pi * loop(radius**2 * vertical_step),
            area=loop(radius * vertical_step),
            perimeter_inverse_gradient=loop(arc / gradient),
        ),
        angle=angle,
        radius=radius,
        height=height,
        arc=arc,
        gradient=gradient,
        r_in=r_in,
        r_out=r_out,
        z_lower=height[lower, np.arange(psi_norm.size)],
        z_upper=height[upper, np.arange(psi_norm.size)],
        r_lower=radius[lower, np.arange(psi_norm.size)],
        r_upper=radius[upper, np.arange(psi_norm.size)],
    )
