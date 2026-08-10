r"""Flux-surface-averaged geometry against an exact family and a solved map.

The record is validated twice over, on two problems that share no code.

The first is exact. The static limit of the analytic rotating family has
closed-form flux surfaces: on ``psi = psi_a - k_p u^2 / 2 - k_z Z^2`` with
``u = R^2 - R_0^2`` every surface is the smooth loop ``u = u_m cos t``,
``Z = Z_m sin t``, so this module carries its own machine-precision contour
quadrature of the integrals the record forms, and its own closed forms for the
three that reduce,

    V(psi_N) = pi^2 psi_a psi_N sqrt(2 / (k_p k_z)),
    <1/R^2>  = 1 / sqrt(R_0^4 - u_m^2) = 1 / (R_in R_out),
    q(psi_N) = F / (2 sqrt(2 k_p k_z) sqrt(R_0^4 - u_m^2)),

with ``u_m = sqrt(2 psi_a psi_N / k_p)`` the boundary label of the surface.
Those closed forms are checked against the reference module's own independent
quadratures first, so the yardstick is itself verified before anything from
the package is read. The flux is then sampled on a lattice, the record is
assembled from the lattice alone, and every published column is compared with
the exact value at the label the exact toroidal-flux map assigns — so the
comparison tests the surface integrals, the toroidal-flux quadrature and the
coordinate inversion together rather than one of them at a time.

The second is a free-boundary solve. A ring of conductors is fitted to hold an
analytic seed, the shipped absolute source is driven above the marginal point,
and the record is taken from the converged result through the public entry
point alone. Nothing about that map is known in closed form, so what is pinned
there is the contract: finiteness, monotonicity, the axis the topology
published, and the two identities that hold whatever produced the map.

Pinned here: the exact surface geometry and its convergence order under
lattice refinement, the safety factor as the pitch the published averages
imply, the enclosed volume against its own flux derivative, the surface-area
identity the gradient family carries, the axis and edge limits, the alignment
two interval endpoints need, and the refusals a map with no resolvable family
earns.

Running this file as ``python -m tests.test_equilibrium_flux_surface_geometry
figures`` from the repository root regenerates the evidence figures instead of
running the suite.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.constants import mu_0
from scipy.interpolate import CubicSpline

from nova.utilities.importmanager import skip_import
from tests.rotating_equilibrium_references import reference_cases

with skip_import("jax"):
    import jax.numpy as jnp

    from nova.biot.greens import hybrid_greens
    from nova.equilibrium.conservation import FluxLattice
    from nova.equilibrium.domain import PlasmaDomain
    from nova.equilibrium.flux_surface_geometry import (
        FluxSurfaceGeometry,
        SurfaceGeometryError,
    )
    from nova.equilibrium.forward import ForwardProfile
    from nova.equilibrium.source import DomainProfile, ForwardSource
    from nova.jax.config import configure_dtypes

#: Wb of total poloidal flux per Wb/rad. The analytic family is written in
#: flux per radian; the record consumes the total flux.
TOTAL_FLUX_FACTOR = 2.0 * np.pi

#: Lattice the exact comparison is read on, and the refinement ladder the
#: convergence order is measured over.
REFERENCE_NODES = 65
REFINEMENT_NODES = (33, 49, 65, 97)

#: Angular nodes of this module's own exact contour quadrature. The surfaces
#: are analytic and periodic in the parametrising angle, so the midpoint rule
#: is spectral and this is far past what double precision resolves.
EXACT_ANGLES = 2048

#: Label window the exact comparison is read over. The innermost surfaces are
#: a fraction of a cell across on the coarsest lattice, which measures the
#: lattice rather than the record.
COMPARISON_WINDOW = (0.35, 0.96)

#: Pre-registered agreement with the exact family at ``REFERENCE_NODES``, as a
#: relative sup-norm over the comparison window. The metrics built from the
#: inverse-gradient loops sit an order below the two built from the enclosed
#: area, which is the cancellation in ``oint R^2 dZ`` rather than a different
#: discretisation.
EXACT_TOLERANCE = {
    "safety_factor": 2.0e-7,
    "field_function": 1.0e-8,
    "volume": 5.0e-6,
    "area": 5.0e-6,
    "volume_flux_derivative": 2.0e-7,
    "volume_derivative": 2.0e-7,
    "inverse_square_radius": 1.0e-7,
    "gradient_rho": 5.0e-7,
    "gradient_rho_squared": 1.0e-6,
    "gradient_rho_squared_over_radius_squared": 1.0e-6,
    "rho_tor": 1.0e-7,
    "psi_norm": 1.0e-6,
    "toroidal_flux": 1.0e-7,
}

#: Columns whose lattice convergence is measured. They are the ones built from
#: the traced gradient alone, so their order is the interpolant's and nothing
#: else's.
CONVERGENT_COLUMNS = (
    "safety_factor",
    "volume_flux_derivative",
    "inverse_square_radius",
    "volume_derivative",
)
CONVERGENCE_ORDER = 2.0

#: The axis limits are reached by a fit over the innermost traced surfaces
#: rather than by a surface of their own, so they carry their own tolerance
#: against the closed forms.
AXIS_TOLERANCE = 1.0e-5
AXIS_POSITION_TOLERANCE = 1.0e-5

#: The cross-route identities. The volume derivative is differenced off the
#: published label grid, so it is read an order looser than the surface area,
#: which is a direct comparison of two loops. The pitch identity holds to
#: round-off on any map, because the record builds those three columns from
#: one resampled pair of loops rather than resampling each of them.
VOLUME_IDENTITY_TOLERANCE = 1.0e-4
SURFACE_AREA_TOLERANCE = 1.0e-6
PITCH_IDENTITY_TOLERANCE = 1.0e-12

#: Free-boundary solve. The lattice is the smallest that still resolves a
#: core, a scrape-off band and a material exclusion at once.
SOLVE_NODES = 25
CONDUCTORS = 12
EVALUATIONS = 200
P_PRIME = -3.0e5
FF_PRIME = -0.25
AXIS_RADIUS = 1.0
SEED_SPAN = 0.35
DRIVE = 1.4
BOUNDARY_FIELD_FUNCTION = 5.0
SOLVE_RESIDUAL_TOLERANCE = 1.0e-6

#: Agreement between the record's outermost enclosed volume and the volume the
#: integral observation sums over the labelled core. They are different
#: objects — a contour integral of the last resolved surface against a sum of
#: whole cells over a boolean mask — so they agree only to the cell size.
CELL_VOLUME_TOLERANCE = 0.1

FIGURE_DIRECTORY = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "figures"
    / "flux-function-forward-equilibrium"
)


# --------------------------------------------------------------------------
# the analytic family: its own exact quadrature and closed forms
# --------------------------------------------------------------------------
def _field_function(case):
    """Return ``F(psi_N)`` [T m] of one member of the analytic family."""

    def field(psi_norm):
        """Return the toroidal-field function at one normalised flux."""
        return case.f_function(case.axis_flux * (1.0 - np.asarray(psi_norm)))

    return field


def _exact_loops(case, psi_norm: float) -> dict[str, float]:
    """Return the exact contour loops of one static-limit surface.

    The surface is parametrised by ``u = u_m cos t`` and ``Z = Z_m sin t``,
    which satisfies ``k_p u^2 / 2 + k_z Z^2 = psi_a psi_N`` identically for
    every ``t``. Both the surface and every integrand on it are analytic and
    periodic, so the midpoint rule converges spectrally.
    """
    drop = case.axis_flux * psi_norm
    label_max = np.sqrt(2.0 * drop / case.pressure_coefficient)
    height_max = np.sqrt(drop / case.field_coefficient)
    angle = 2.0 * np.pi * np.arange(EXACT_ANGLES) / EXACT_ANGLES
    label = label_max * np.cos(angle)
    radius = np.sqrt(case.major_radius**2 + label)
    height = height_max * np.sin(angle)
    radial = -TOTAL_FLUX_FACTOR * 2.0 * radius * case.pressure_coefficient * label
    vertical = -TOTAL_FLUX_FACTOR * 2.0 * case.field_coefficient * height
    gradient = np.hypot(radial, vertical)
    radius_step = -label_max * np.sin(angle) / (2.0 * radius)
    height_step = height_max * np.cos(angle)
    arc = np.hypot(radius_step, height_step)
    weight = 2.0 * np.pi / EXACT_ANGLES

    def loop(integrand):
        """Return one closed line integral of the exact surface."""
        return float(weight * np.sum(integrand))

    return {
        "inverse_gradient": loop(radius * arc / gradient),
        "pitch": loop(arc / (radius * gradient)),
        "gradient_radius": loop(gradient * radius * arc),
        "gradient_inverse_radius": loop(gradient * arc / radius),
        "radius_perimeter": loop(radius * arc),
        "volume": np.pi * loop(radius**2 * height_step),
        "area": loop(radius * height_step),
    }


def _closed_form_root(case, psi_norm):
    """Return ``sqrt(R_0^4 - u_m^2)``, the root both closed forms share."""
    label = 2.0 * case.axis_flux * np.asarray(psi_norm) / case.pressure_coefficient
    return np.sqrt(case.major_radius**4 - label)


def _closed_form_safety(case, psi_norm):
    """Return the closed-form safety factor of the static limit."""
    drive = 2.0 * np.sqrt(2.0 * case.pressure_coefficient * case.field_coefficient)
    return _field_function(case)(psi_norm) / (drive * _closed_form_root(case, psi_norm))


def _closed_form_volume(case, psi_norm):
    """Return the closed-form enclosed volume of the static limit."""
    return (
        np.pi**2
        * case.axis_flux
        * np.asarray(psi_norm)
        * np.sqrt(2.0 / (case.pressure_coefficient * case.field_coefficient))
    )


def _exact_reference(case, rho_tor_norm, nodes: int = 4001) -> dict[str, np.ndarray]:
    """Return every exact column at the label the requested grid names.

    The toroidal-flux map is built once from the closed-form safety factor on a
    dense grid and inverted, so each requested normalised radius is turned into
    the normalised flux it really labels before any surface integral is taken.
    Comparing there rather than at the record's own normalised flux is what
    puts the coordinate inversion inside the test.
    """
    boundary_field = float(_field_function(case)(1.0))
    vacuum_field = boundary_field / case.major_radius
    span = -TOTAL_FLUX_FACTOR * case.axis_flux
    label = np.linspace(0.0, 1.0, nodes)
    drive = 2.0 * abs(span) * _closed_form_safety(case, label**2) * label
    toroidal = CubicSpline(label, drive).antiderivative()(label)
    radius = np.sqrt(toroidal / (np.pi * vacuum_field))
    normalised = radius / radius[-1]
    normalised[0], normalised[-1] = 0.0, 1.0

    requested = np.asarray(rho_tor_norm, dtype=float)
    psi_norm = np.clip(CubicSpline(normalised, label**2)(requested), 0.0, 1.0)
    rho_tor = requested * radius[-1]
    exact = {
        "psi_norm": psi_norm,
        "rho_tor": rho_tor,
        "boundary_rho_tor": float(radius[-1]),
        "toroidal_flux": CubicSpline(normalised, toroidal)(requested),
        "field_function": _field_function(case)(psi_norm),
    }
    for column in (
        "safety_factor",
        "volume",
        "area",
        "volume_flux_derivative",
        "volume_derivative",
        "inverse_square_radius",
        "gradient_rho",
        "gradient_rho_squared",
        "gradient_rho_squared_over_radius_squared",
        "surface_area",
    ):
        exact[column] = np.zeros_like(requested)
    for index, level in enumerate(psi_norm):
        if level <= 0.0:
            continue
        loops = _exact_loops(case, float(level))
        safety = float(exact["field_function"][index]) * loops["pitch"]
        # d rho / d Phi from Phi_tor = pi B_0 rho^2, the relation the record
        # uses to carry the gradient family onto the published label
        derivative = safety / (2.0 * np.pi * vacuum_field * rho_tor[index])
        exact["safety_factor"][index] = safety
        exact["volume"][index] = loops["volume"]
        exact["area"][index] = loops["area"]
        exact["volume_flux_derivative"][index] = (
            2.0 * np.pi * loops["inverse_gradient"] * np.sign(span)
        )
        exact["volume_derivative"][index] = (
            2.0 * np.pi * loops["inverse_gradient"] / abs(derivative)
        )
        exact["inverse_square_radius"][index] = (
            loops["pitch"] / loops["inverse_gradient"]
        )
        exact["gradient_rho"][index] = (
            abs(derivative) * loops["radius_perimeter"] / loops["inverse_gradient"]
        )
        exact["gradient_rho_squared"][index] = (
            derivative**2 * loops["gradient_radius"] / loops["inverse_gradient"]
        )
        exact["gradient_rho_squared_over_radius_squared"][index] = (
            derivative**2 * loops["gradient_inverse_radius"] / loops["inverse_gradient"]
        )
        exact["surface_area"][index] = 2.0 * np.pi * loops["radius_perimeter"]
    return exact


def _analytic_lattice(case, nodes: int, pad: float = 0.18):
    """Return a lattice enclosing one member and the flux sampled on it."""
    inboard, outboard = case.boundary_midplane_radii()
    half_height = np.sqrt(case.axis_flux / case.field_coefficient)
    width = outboard - inboard
    lattice = FluxLattice(
        np.linspace(inboard - pad * width, outboard + pad * width, nodes),
        np.linspace(-(1.0 + pad) * half_height, (1.0 + pad) * half_height, nodes),
    )
    coordinate = lattice.coordinate
    flux = TOTAL_FLUX_FACTOR * case.flux(coordinate[:, 0], coordinate[:, 1])
    return lattice, flux


def _analytic_record(case, nodes: int, rho_tor_norm, **options):
    """Return the record of one analytic member sampled on a lattice.

    The axis seed is deliberately offset from the exact axis, so the search
    the record runs on its own interpolant is exercised rather than handed the
    answer.
    """
    lattice, flux = _analytic_lattice(case, nodes)
    half_height = np.sqrt(case.axis_flux / case.field_coefficient)
    return FluxSurfaceGeometry.from_flux_map(
        lattice,
        flux,
        _field_function(case),
        axis=(1.03 * case.major_radius, 0.02 * half_height),
        boundary_flux=0.0,
        reference_radius=case.major_radius,
        rho_tor_norm=rho_tor_norm,
        edge_psi_norm=1.0,
        **options,
    )


@pytest.fixture(scope="module")
def label_grid():
    """Return the normalised label grid every analytic record is read on."""
    return np.linspace(0.0, 1.0, 41)


@pytest.fixture(scope="module")
def members():
    """Return the static limit of every member of the analytic family."""
    return {name: case.static_limit() for name, case in reference_cases().items()}


@pytest.fixture(scope="module")
def records(members, label_grid):
    """Return one record per member at the reference lattice."""
    return {
        name: _analytic_record(case, REFERENCE_NODES, label_grid)
        for name, case in members.items()
    }


@pytest.fixture(scope="module")
def exact(members, label_grid):
    """Return the exact columns of every member on the label grid."""
    return {name: _exact_reference(case, label_grid) for name, case in members.items()}


def _window(label_grid):
    """Return the label mask the exact comparison is read over."""
    lower, upper = COMPARISON_WINDOW
    return (label_grid > lower) & (label_grid < upper)


def _relative_error(actual, expected, mask):
    """Return the relative sup-norm of one column over the read window."""
    return float(np.max(np.abs(actual[mask] - expected[mask]) / np.abs(expected[mask])))


# --------------------------------------------------------------------------
# the yardstick, verified without the package under test
# --------------------------------------------------------------------------
def test_the_closed_forms_agree_with_the_reference_quadratures(members):
    """This module's own exact machinery reproduces the reference module.

    The closed forms and the contour quadrature written here are checked
    against the reference module's independent routes to the same numbers
    before a single record is read, so a later failure cannot be blamed on the
    yardstick.
    """
    for name, case in members.items():
        loops = _exact_loops(case, 1.0)
        assert loops["volume"] == pytest.approx(case.plasma_volume(), rel=1.0e-12), name
        assert float(_closed_form_volume(case, 1.0)) == pytest.approx(
            case.plasma_volume(), rel=1.0e-12
        ), name
        for level in (0.25, 0.5, 0.9):
            quadrature = (
                float(_field_function(case)(level)) * _exact_loops(case, level)["pitch"]
            )
            reference = case.safety_factor(case.axis_flux * (1.0 - level))
            assert quadrature == pytest.approx(reference, rel=1.0e-10), (name, level)
            assert float(_closed_form_safety(case, level)) == pytest.approx(
                reference, rel=1.0e-10
            ), (name, level)
        assert float(_closed_form_safety(case, 0.0)) == pytest.approx(
            case.axis_safety_factor, rel=1.0e-12
        ), name
        # <1/R^2> reduces to the inverse product of the midplane radii, the
        # same root both closed forms are built on
        inboard, outboard = case.surface_midplane_radii(0.5 * case.axis_flux)
        assert 1.0 / float(_closed_form_root(case, 0.5)) == pytest.approx(
            1.0 / (inboard * outboard), rel=1.0e-10
        ), name


# --------------------------------------------------------------------------
# exact validation of the record
# --------------------------------------------------------------------------
def test_the_record_reproduces_the_exact_surface_geometry(records, exact, label_grid):
    """Every published column matches the exact family to its tolerance."""
    mask = _window(label_grid)
    for name, record in records.items():
        for column, tolerance in EXACT_TOLERANCE.items():
            error = _relative_error(getattr(record, column), exact[name][column], mask)
            assert error < tolerance, (name, column, error)
        assert record.boundary_rho_tor == pytest.approx(
            exact[name]["boundary_rho_tor"], rel=EXACT_TOLERANCE["rho_tor"]
        ), name


def test_the_record_converges_on_the_lattice(members, exact, label_grid):
    """Refining the lattice drives the surface metrics down at second order.

    Only the columns built from the traced gradient are read here. The
    interpolant carries the flux one order above its own first derivative, so
    the record's floor is set by the gradient the surface averages divide by,
    and a measured order below two would mean a discretisation the record
    should not have.
    """
    name = "moderate-rotation-conventional"
    case = members[name]
    mask = _window(label_grid)
    errors = {column: [] for column in CONVERGENT_COLUMNS}
    for nodes in REFINEMENT_NODES:
        record = _analytic_record(case, nodes, label_grid)
        for column in CONVERGENT_COLUMNS:
            errors[column].append(
                _relative_error(getattr(record, column), exact[name][column], mask)
            )
    spacing = np.log([1.0 / nodes for nodes in REFINEMENT_NODES])
    for column, measured in errors.items():
        assert measured[-1] < measured[0], (column, measured)
        order = float(np.polyfit(spacing, np.log(measured), 1)[0])
        assert order > CONVERGENCE_ORDER, (column, order, measured)


def test_the_axis_limits_reach_their_closed_forms(members, records):
    """The extrapolated axis values are the ones the family fixes there.

    Every intensive average has a finite axis limit that the record reaches by
    a fit in normalised flux, and two of them are known exactly: the pitch
    collapses onto the quadratic form at the O-point and the inverse square
    radius onto the axis radius. The extensive columns vanish there instead,
    and the volume derivative on the poloidal flux stays finite because it is
    the one metric the shrinking surface does not scale.
    """
    for name, case in members.items():
        record = records[name]
        assert record.safety_factor[0] == pytest.approx(
            case.axis_safety_factor, rel=AXIS_TOLERANCE
        ), name
        assert record.inverse_square_radius[0] == pytest.approx(
            1.0 / case.major_radius**2, rel=AXIS_TOLERANCE
        ), name
        assert record.magnetic_axis[0] == pytest.approx(
            case.major_radius, rel=AXIS_POSITION_TOLERANCE
        ), name
        assert record.magnetic_axis[1] == pytest.approx(
            0.0, abs=AXIS_POSITION_TOLERANCE * case.major_radius
        ), name
        for column in ("volume", "area", "toroidal_flux", "rho_tor", "psi_norm"):
            assert getattr(record, column)[0] == 0.0, (name, column)
        assert record.volume_derivative[0] == 0.0, name
        assert abs(record.volume_flux_derivative[0]) > 0.0, name
        assert record.field_function[0] == pytest.approx(
            float(case.f_function(case.axis_flux)), rel=1.0e-12
        ), name


def test_the_edge_surface_is_the_declared_cut(members, records):
    """The outermost node is the resolved boundary the record declares.

    The analytic family has no X-point, so the record is taken all the way to
    the plasma boundary and the enclosed volume there is the family's own.
    """
    for name, case in members.items():
        record = records[name]
        assert record.rho_tor_norm[-1] == 1.0, name
        assert record.rho_tor[-1] == pytest.approx(record.boundary_rho_tor, rel=1e-15)
        assert record.psi_norm[-1] == pytest.approx(1.0, abs=1.0e-9), name
        assert record.volume[-1] == pytest.approx(case.plasma_volume(), rel=1.0e-6), (
            name
        )
        assert record.field_function[-1] == pytest.approx(case.boundary_f, rel=1e-12)
        assert record.vacuum_field == pytest.approx(case.vacuum_field, rel=1e-12), name
        assert record.reference_radius == pytest.approx(case.major_radius, rel=1e-15)


# --------------------------------------------------------------------------
# identities the record must satisfy whatever produced the map
# --------------------------------------------------------------------------
def test_the_safety_factor_is_the_pitch_the_averages_imply(records):
    """``q = F <R^-2> |dV/dPhi| / (2 pi)`` on every published node.

    This is the convention pin. The identity holds with a single factor of
    ``2 pi`` only because the flux is the TOTAL poloidal flux; written in flux
    per radian it would carry ``4 pi^2``, so a record that satisfied the
    textbook form here would be reporting the wrong flux.
    """
    for name, record in records.items():
        implied = (
            record.field_function
            * record.inverse_square_radius
            * np.abs(record.volume_flux_derivative)
            / (2.0 * np.pi)
        )
        error = float(
            np.max(
                np.abs(implied - record.safety_factor) / np.abs(record.safety_factor)
            )
        )
        assert error < PITCH_IDENTITY_TOLERANCE, (name, error)


def test_the_volume_derivative_matches_the_enclosed_volume(records):
    """The enclosed volume differentiates into the published derivative.

    The two reach the same number by different routes: the volume is the plane
    divergence theorem applied to the traced contour, the derivative is a line
    integral of the traced gradient over the same contour. Agreement is a
    check on the tracing, not an algebraic identity.
    """
    for name, record in records.items():
        label = record.rho_tor_norm
        volume_step = CubicSpline(label, record.volume).derivative()(label)
        flux_step = CubicSpline(label, record.poloidal_flux).derivative()(label)
        interior = (label > 0.05) & (label < 0.98)
        differenced = volume_step[interior] / flux_step[interior]
        published = record.volume_flux_derivative[interior]
        error = float(np.max(np.abs(differenced - published) / np.abs(published)))
        assert error < VOLUME_IDENTITY_TOLERANCE, (name, error)


def test_the_gradient_family_carries_the_surface_area(records, exact, label_grid):
    """``(dV/drho) <|grad rho|>`` is the surface area of the same contour.

    The metric that turns a transport flux density into a flux through a
    surface is exactly this product, so it is read against the independent
    perimeter loop of the exact surface rather than against itself.
    """
    mask = label_grid > 0.2
    for name, record in records.items():
        published = record.volume_derivative * record.gradient_rho
        error = _relative_error(published, exact[name]["surface_area"], mask)
        assert error < SURFACE_AREA_TOLERANCE, (name, error)
        # a mean square never falls below a squared mean
        assert np.all(
            record.gradient_rho**2 <= record.gradient_rho_squared * (1.0 + 1.0e-12)
        ), name


def test_the_toroidal_flux_radius_is_monotone(records):
    """The published coordinate increases and its metrics stay positive."""
    for name, record in records.items():
        assert np.all(np.diff(record.rho_tor) > 0.0), name
        assert np.all(np.diff(record.toroidal_flux) > 0.0), name
        assert np.all(np.diff(record.volume) > 0.0), name
        assert np.all(np.diff(record.area) > 0.0), name
        assert np.all(record.safety_factor > 0.0), name
        assert np.all(record.inverse_square_radius > 0.0), name
        assert np.all(record.volume_derivative[1:] > 0.0), name
        assert np.all(record.gradient_rho[1:] > 0.0), name
        # these members carry a flux that falls outward, so the signed volume
        # derivative on the poloidal flux is negative throughout
        assert np.all(record.volume_flux_derivative < 0.0), name


# --------------------------------------------------------------------------
# two records make a moving grid
# --------------------------------------------------------------------------
def test_two_records_difference_at_fixed_label(members, records, label_grid):
    """A pair on one label grid yields the rates a moving grid needs.

    The second state is the same family driven harder, which holds a smaller
    plasma: the boundary radius falls, so the whole coordinate contracts while
    the normalised label is unmoved, and the metric on that label moves with
    it. Those two rates are what a balance written at fixed normalised label
    picks up beyond the transport fluxes themselves.
    """
    name = "moderate-rotation-conventional"
    case = members[name]
    driven = replace(case, pressure_coefficient=1.15 * case.pressure_coefficient)
    later = _analytic_record(driven, REFERENCE_NODES, label_grid)
    early = records[name]

    assert early.aligned_with(later)
    interval = 0.05
    motion = early.motion(later, interval)
    assert motion.interval == interval
    assert later.boundary_rho_tor < early.boundary_rho_tor
    assert motion.boundary_rate == pytest.approx(
        (later.boundary_rho_tor - early.boundary_rho_tor) / interval, rel=1e-12
    )
    assert motion.boundary_rate < 0.0
    assert motion.volume_derivative_rate.shape == early.volume_derivative.shape
    assert np.all(np.isfinite(motion.volume_derivative_rate))
    assert np.max(np.abs(motion.volume_derivative_rate)) > 0.0

    coarse = _analytic_record(case, REFERENCE_NODES, np.linspace(0.0, 1.0, 21))
    assert not early.aligned_with(coarse)
    with pytest.raises(SurfaceGeometryError, match="share the label grid"):
        early.motion(coarse, interval)
    with pytest.raises(SurfaceGeometryError, match="interval"):
        early.motion(later, 0.0)


def test_a_map_without_a_resolvable_family_is_refused(members, label_grid):
    """The record fails loudly rather than publishing an unbounded read."""
    name = "moderate-rotation-conventional"
    case = members[name]
    lattice, flux = _analytic_lattice(case, REFERENCE_NODES)
    field = _field_function(case)
    with pytest.raises(SurfaceGeometryError, match="does not bound this map"):
        FluxSurfaceGeometry.from_flux_map(
            lattice,
            flux,
            field,
            axis=(case.major_radius, 0.0),
            boundary_flux=-4.0 * TOTAL_FLUX_FACTOR * case.axis_flux,
        )
    with pytest.raises(SurfaceGeometryError, match="the lattice indexes"):
        FluxSurfaceGeometry.from_flux_map(
            lattice, flux[:10], field, axis=(case.major_radius, 0.0), boundary_flux=0.0
        )
    with pytest.raises(SurfaceGeometryError, match="between the axis"):
        _analytic_record(case, REFERENCE_NODES, np.linspace(0.0, 1.4, 21))

    # a hyperbolic point carries no nested family at all, and the search that
    # places the axis is what has to say so
    coordinate = lattice.coordinate
    centre = 0.5 * (lattice.radius[0] + lattice.radius[-1])
    saddle = (coordinate[:, 0] - centre) ** 2 - coordinate[:, 1] ** 2
    with pytest.raises(SurfaceGeometryError, match="elliptic stationary point"):
        FluxSurfaceGeometry.from_flux_map(
            lattice, saddle, field, axis=(centre, 0.0), boundary_flux=1.0
        )


# --------------------------------------------------------------------------
# the same record taken from a free-boundary solve
# --------------------------------------------------------------------------
def _solovev_terms():
    """Return the seed's quartic, offset and vertical coefficients."""
    alpha = np.pi**2 * mu_0 * P_PRIME / 2.0
    return alpha, -2.0 * alpha * AXIS_RADIUS**2, 2.0 * np.pi**2 * FF_PRIME


def _solovev(radius, height):
    """Return the analytic seed flux [Wb] the conductors are fitted to."""
    alpha, offset, beta = _solovev_terms()
    return alpha * radius**4 + offset * radius**2 + beta * height**2


def _wall_loop(points=41):
    """Return a material boundary lying on one seed flux surface."""
    alpha, offset, beta = _solovev_terms()
    wall_flux = _solovev(AXIS_RADIUS, 0.0) - SEED_SPAN
    inner, outer = np.sqrt(np.sort(np.roots([alpha, offset, -wall_flux])))
    centre, half = 0.5 * (inner + outer), 0.5 * (outer - inner)
    angle = 2 * np.pi * np.arange(points) / points
    radius = centre + half * np.cos(angle)
    argument = np.clip((wall_flux - _solovev(radius, 0.0)) / beta, 0.0, None)
    return np.c_[radius, np.sign(np.sin(angle)) * np.sqrt(argument)], wall_flux


def _green_block(target, source, section=0.05):
    """Return the total-flux coupling [Wb/A] of one source set on one target."""
    return np.stack(
        [
            hybrid_greens(target[:, 0], target[:, 1], a, z, section, section)[0]
            for a, z in source
        ],
        axis=1,
    )


def _edge_vanishing_profile(amplitude):
    """Return an absolute gradient that falls linearly to zero at the edge."""

    def gradient(psi_norm):
        """Return the tapered value at one normalised flux."""
        return amplitude * (1.0 - jnp.clip(jnp.asarray(psi_norm), 0.0, 1.0))

    return gradient


def _flat_profile(amplitude):
    """Return a constant absolute gradient."""

    def gradient(psi_norm):
        """Return the constant value at every normalised flux."""
        return jnp.full_like(jnp.asarray(psi_norm, dtype=jnp.float64), amplitude)

    return gradient


def _bootstrap_solve():
    """Return the free-boundary solve, its analytic seed and its wall."""
    configure_dtypes()
    lattice = FluxLattice(
        np.linspace(0.6, 1.42, SOLVE_NODES), np.linspace(-0.42, 0.42, SOLVE_NODES)
    )
    coordinate = lattice.coordinate
    wall, wall_flux = _wall_loop()
    seed_flux = _solovev(coordinate[:, 0], coordinate[:, 1])
    wall_seed = _solovev(wall[:, 0], wall[:, 1])
    inside = seed_flux >= wall_flux

    angle = 2 * np.pi * np.arange(CONDUCTORS) / CONDUCTORS
    conductor = np.c_[1.0 + 0.62 * np.cos(angle), 0.62 * np.sin(angle)]
    coupling = {
        "plasma_to_grid": _green_block(coordinate, coordinate),
        "plasma_to_wall": _green_block(wall, coordinate),
        "source_to_grid": _green_block(coordinate, conductor),
        "source_to_wall": _green_block(wall, conductor),
    }

    def build(core, current):
        """Return the solve for one declared source and conductor state."""
        return ForwardProfile.from_lattice(
            lattice,
            ForwardSource(core=core, boundary_field_function=BOUNDARY_FIELD_FUNCTION),
            external_current=current,
            wall_coordinate=wall,
            polarity=1,
            inside_material=inside,
            **coupling,
        )

    seed = jnp.asarray(np.r_[seed_flux, wall_seed])
    flat = build(
        DomainProfile(p_prime=_flat_profile(P_PRIME), ff_prime=_flat_profile(FF_PRIME)),
        np.zeros(CONDUCTORS),
    )
    cell_current = np.asarray(flat.operator.cell_current(seed))
    target = np.r_[
        seed_flux - coupling["plasma_to_grid"] @ cell_current,
        wall_seed - coupling["plasma_to_wall"] @ cell_current,
    ]
    weight = np.r_[inside.astype(float), np.ones(len(wall))]
    matrix = np.r_[coupling["source_to_grid"], coupling["source_to_wall"]]
    current = np.linalg.lstsq(matrix * weight[:, None], target * weight, rcond=None)[0]
    profile = build(
        DomainProfile(
            p_prime=_edge_vanishing_profile(2.0 * DRIVE * P_PRIME),
            ff_prime=_edge_vanishing_profile(2.0 * DRIVE * FF_PRIME),
        ),
        current,
    )
    return profile, seed, wall


@pytest.fixture(scope="module")
def solved():
    """Return the converged free-boundary solve, its wall and its record."""
    profile, seed, wall = _bootstrap_solve()
    result = profile.solve(seed, route="anderson", evaluations=EVALUATIONS)
    record = FluxSurfaceGeometry.from_equilibrium(
        profile.lattice,
        profile.source,
        result,
        rho_tor_norm=np.linspace(0.0, 1.0, 33),
    )
    return profile, result, wall, record


def test_the_solved_map_carries_a_self_consistent_geometry_record(solved):
    """The public entry point turns a converged solve into a valid record.

    Nothing about this map is known in closed form, so what is pinned is the
    contract: the record reads the same axis the topology published, its
    coordinate is monotone, its metrics are positive and finite, the enclosed
    volume lands within a cell of the volume the integral observation sums,
    and both identities still hold.
    """
    _profile, result, _wall, record = solved
    assert float(result.fixed_point.residual) < SOLVE_RESIDUAL_TOLERANCE
    counts = np.asarray(result.domains.cell_count())
    assert counts[PlasmaDomain.CORE] > 50
    assert counts[PlasmaDomain.COMMON_SOL] > 0
    assert counts[PlasmaDomain.EXCLUDED_MATERIAL] > 0

    axis = np.asarray(result.topology.axis)
    assert record.magnetic_axis[0] == pytest.approx(float(axis[0]), abs=0.02)
    assert record.magnetic_axis[1] == pytest.approx(float(axis[1]), abs=0.02)

    for column in FluxSurfaceGeometry.profile_names():
        assert np.all(np.isfinite(getattr(record, column))), column
    assert np.all(np.diff(record.rho_tor) > 0.0)
    assert np.all(np.diff(record.volume) > 0.0)
    assert np.all(np.diff(record.toroidal_flux) > 0.0)
    assert np.all(record.safety_factor > 0.0)
    assert record.volume_derivative[0] == 0.0
    assert np.all(record.volume_derivative[1:] > 0.0)

    implied = (
        record.field_function
        * record.inverse_square_radius
        * np.abs(record.volume_flux_derivative)
        / (2.0 * np.pi)
    )
    assert np.allclose(implied, record.safety_factor, rtol=PITCH_IDENTITY_TOLERANCE)

    label = record.rho_tor_norm
    volume_step = CubicSpline(label, record.volume).derivative()(label)
    flux_step = CubicSpline(label, record.poloidal_flux).derivative()(label)
    interior = (label > 0.1) & (label < 0.95)
    differenced = volume_step[interior] / flux_step[interior]
    published = record.volume_flux_derivative[interior]
    assert np.allclose(differenced, published, rtol=1.0e-2)

    # the source declares its boundary toroidal-field function, and the last
    # resolved surface has to land back on it
    assert record.field_function[-1] == pytest.approx(
        BOUNDARY_FIELD_FUNCTION, rel=1.0e-3
    )
    assert record.vacuum_field == pytest.approx(
        BOUNDARY_FIELD_FUNCTION / record.reference_radius, rel=1e-12
    )
    assert record.volume[-1] == pytest.approx(
        float(result.moments.volume), rel=CELL_VOLUME_TOLERANCE
    )


# --------------------------------------------------------------------------
# evidence figures
# --------------------------------------------------------------------------
def _domain_figure(axes, profile, result, wall):
    """Draw the solved flux map beside the domain partition it labels."""
    lattice = profile.lattice
    shape = lattice.shape
    extent = (
        lattice.radius[0],
        lattice.radius[-1],
        lattice.height[0],
        lattice.height[-1],
    )
    grid_flux = np.asarray(result.flux)[: lattice.node_count].reshape(shape)
    psi_norm = np.asarray(result.domains.psi_norm).reshape(shape)
    label = np.asarray(result.domains.label).reshape(shape)
    radius, height = np.meshgrid(lattice.radius, lattice.height, indexing="ij")
    axis = np.asarray(result.topology.axis)
    closed = np.r_[wall, wall[:1]]

    left, right = axes
    left.contour(radius, height, grid_flux, levels=24, colors="0.75", linewidths=0.5)
    left.contour(radius, height, psi_norm, levels=[1.0], colors="C3", linewidths=1.6)
    left.plot(closed[:, 0], closed[:, 1], "k-", lw=1.0)
    left.plot(axis[0], axis[1], "kx", ms=7)
    left.annotate(
        "magnetic axis",
        (axis[0], axis[1]),
        textcoords="offset points",
        xytext=(9, -11),
        fontsize=8,
    )
    left.annotate(
        "last closed surface",
        (axis[0], 0.232),
        textcoords="offset points",
        xytext=(0, -14),
        color="C3",
        fontsize=8,
        ha="center",
    )
    upper = int(np.argmax(wall[:, 1]))
    left.annotate(
        "material boundary",
        (wall[upper, 0], wall[upper, 1]),
        textcoords="offset points",
        xytext=(-26, 26),
        fontsize=8,
        ha="center",
        arrowprops={"arrowstyle": "-", "lw": 0.6, "color": "0.35"},
    )
    left.set_title("solved flux map", fontsize=9)

    palette = np.array(
        [
            [0.93, 0.93, 0.93],
            [0.84, 0.32, 0.22],
            [0.31, 0.50, 0.76],
            [0.95, 0.76, 0.24],
        ]
    )
    right.imshow(
        np.transpose(palette[label], (1, 0, 2)),
        origin="lower",
        extent=extent,
        interpolation="nearest",
    )
    right.plot(closed[:, 0], closed[:, 1], "k-", lw=0.8)
    empty = []
    for domain, text, offset in (
        (PlasmaDomain.CORE, "core", (0.0, 0.0)),
        (PlasmaDomain.COMMON_SOL, "common SOL", (0.0, 0.21)),
        (PlasmaDomain.PRIVATE_FLUX, "private flux", (0.0, 0.0)),
        (PlasmaDomain.EXCLUDED_MATERIAL, "excluded material", (0.0, 0.34)),
    ):
        selected = label == domain
        if not selected.any():
            empty.append(text)
            continue
        right.annotate(
            text,
            (
                float(radius[selected].mean()) + offset[0],
                float(height[selected].mean()) + offset[1],
            ),
            fontsize=8,
            ha="center",
            color="0.1",
        )
    if empty:
        # an empty branch is a physical statement about the configuration, not
        # a gap in the partition: a wall-limited plasma has no X-point and so
        # no closed surface disconnected from the axis
        right.annotate(
            f"{', '.join(empty)}: empty — no X-point on this branch",
            (0.5, 0.02),
            xycoords="axes fraction",
            fontsize=8,
            ha="center",
            color="0.35",
        )
    right.set_title("topology-qualified domains", fontsize=9)

    for panel in axes:
        panel.set_xlabel("$R$ [m]", fontsize=8)
        panel.set_aspect("equal")
        panel.tick_params(labelsize=7)
    left.set_ylabel("$Z$ [m]", fontsize=8)


def _profile_figure(figure, case, label_grid):
    """Draw the analytic profiles and the order the lattice drives them at."""
    exact = _exact_reference(case, label_grid)
    ladder = {
        nodes: _analytic_record(case, nodes, label_grid) for nodes in REFINEMENT_NODES
    }
    reference = ladder[REFERENCE_NODES]
    shown = (
        ("safety_factor", "$q$", float(_closed_form_safety(case, 0.0)), 0.72),
        (
            "volume_derivative",
            r"$V' = \mathrm{d}V/\mathrm{d}\rho$  [m$^2$]",
            0.0,
            0.56,
        ),
        (
            "inverse_square_radius",
            r"$\langle 1/R^2 \rangle$  [m$^{-2}$]",
            1.0 / case.major_radius**2,
            0.40,
        ),
        (
            "field_function",
            r"$F = R B_\phi$  [T m]",
            float(case.f_function(case.axis_flux)),
            0.12,
        ),
    )
    mask = _window(label_grid)
    grid = figure.add_gridspec(2, 4, height_ratios=(1.45, 1.0))
    for index, (column, title, axis_limit, _offset) in enumerate(shown):
        panel = figure.add_subplot(grid[0, index])
        # the exact reference has no surface at the axis, so its line starts
        # one node out and the closed-form limit is marked there instead
        panel.plot(exact["rho_tor"][1:], exact[column][1:], "-", color="0.2", lw=1.2)
        panel.plot(0.0, axis_limit, "+", ms=8, mew=1.2, color="0.2")
        panel.plot(
            reference.rho_tor[::3],
            getattr(reference, column)[::3],
            "o",
            ms=3.8,
            mfc="none",
            color="C3",
        )
        panel.set_title(title, fontsize=9)
        panel.set_xlabel(r"$\rho_{\mathrm{tor}}$  [m]", fontsize=8)
        panel.tick_params(labelsize=7)
        if index == 0:
            panel.annotate(
                "exact family",
                (exact["rho_tor"][24], exact[column][24]),
                textcoords="offset points",
                xytext=(-16, 22),
                fontsize=8,
                color="0.2",
            )
            panel.annotate(
                "record from the lattice",
                (reference.rho_tor[33], getattr(reference, column)[33]),
                textcoords="offset points",
                xytext=(-118, -4),
                fontsize=8,
                color="C3",
            )
            panel.annotate(
                "fitted axis limit against\nits closed form",
                (0.0, axis_limit),
                textcoords="offset points",
                xytext=(22, 26),
                fontsize=8,
                color="0.2",
                arrowprops={"arrowstyle": "-", "lw": 0.6, "color": "0.5"},
            )

    order_panel = figure.add_subplot(grid[1, :])
    spacing = np.array(REFINEMENT_NODES, dtype=float)
    for column, title, _limit, height in shown:
        deviation = np.array(
            [
                _relative_error(getattr(record, column), exact[column], mask)
                for record in ladder.values()
            ]
        )
        order = -float(np.polyfit(np.log(spacing), np.log(deviation), 1)[0])
        line = order_panel.loglog(spacing, deviation, "o-", ms=4, lw=1.0)[0]
        # a column already at round-off has no order to report; saying so is
        # the honest label, since a slope fitted through noise is not one
        reading = (
            f"order {order:.1f}" if deviation.max() > 1.0e-9 else "already at round-off"
        )
        # the four series end within a factor of two of one another, so the
        # labels are placed on the empty right of the panel and led back to
        # their curve rather than stacked on top of each other
        order_panel.annotate(
            f"{title.split('  ')[0]}   {reading}",
            (spacing[-1], deviation[-1]),
            textcoords="axes fraction",
            xytext=(0.68, height),
            fontsize=8,
            color=line.get_color(),
            va="center",
            arrowprops={"arrowstyle": "-", "lw": 0.6, "color": line.get_color()},
        )
    guide = 4.0e-7 * (spacing / spacing[0]) ** -3.0
    order_panel.loglog(spacing, guide, "--", color="0.55", lw=1.0)
    order_panel.annotate(
        "third order",
        (spacing[1], guide[1]),
        textcoords="offset points",
        xytext=(-58, -4),
        fontsize=8,
        color="0.55",
    )
    order_panel.set_xlabel("lattice nodes per axis", fontsize=8)
    order_panel.set_ylabel(
        f"relative deviation, sup over $\\rho/\\rho_b \\in$ {COMPARISON_WINDOW}",
        fontsize=8,
    )
    order_panel.set_xlim(spacing[0] * 0.92, spacing[-1] * 2.0)
    order_panel.set_xticks(spacing)
    order_panel.set_xticks([], minor=True)
    order_panel.set_xticklabels([f"{int(nodes)}" for nodes in spacing])
    order_panel.tick_params(labelsize=7)
    figure.suptitle(
        f"flux-surface geometry of the static member '{case.name}': the record "
        "against the exact family, and the order the lattice drives it at",
        fontsize=10,
    )


def render_figures(directory: Path = FIGURE_DIRECTORY) -> list[Path]:
    """Write the evidence figures and return the paths written."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    written = []

    profile, seed, wall = _bootstrap_solve()
    result = profile.solve(seed, route="anderson", evaluations=EVALUATIONS)
    figure, axes = plt.subplots(1, 2, figsize=(8.8, 4.6), constrained_layout=True)
    _domain_figure(axes, profile, result, wall)
    path = directory / "static-solve-domains.png"
    figure.savefig(path, dpi=200)
    plt.close(figure)
    written.append(path)

    case = reference_cases()["moderate-rotation-conventional"].static_limit()
    figure = plt.figure(figsize=(11.2, 6.0), constrained_layout=True)
    _profile_figure(figure, case, np.linspace(0.0, 1.0, 41))
    path = directory / "geometry-profiles.png"
    figure.savefig(path, dpi=200)
    plt.close(figure)
    written.append(path)
    return written


if __name__ == "__main__":
    if "figures" in sys.argv[1:]:
        for written in render_figures():
            print(written)
    else:
        pytest.main([__file__])
