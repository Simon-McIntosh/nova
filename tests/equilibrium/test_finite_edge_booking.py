"""The booked current stays continuous where the finite edge crosses a sample.

An EFIT profile keeps a finite toroidal current density on the last closed flux
surface and zero beyond it, so the integrated current density moves smoothly
with the flux level that carries that surface. The booked route integrates the
density over the clipped polygon, so the integral follows the level.

The value-only route applies the five-node cubic cell-average rule to densities
evaluated at cell centres. A cell whose centre crosses the level switches a
weighted contribution from the finite edge value to zero at once, so the
integrated current takes a step at the crossing.
"""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

from dataclasses import dataclass
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks.diiid_forward_gs_match import _booked_profile, _moment_couplings
from nova.equilibrium import topology
from nova.equilibrium.conservation import FluxLattice
from nova.equilibrium.current import Quadrature, book, book_tangent
from nova.equilibrium.forward import ForwardProfile, _cubic_cell_average_stencil
from nova.equilibrium.forward_operator import point_route_current
from nova.equilibrium.solve_request import SampledFluxFunction
from nova.equilibrium.source import DomainProfile, ForwardSource

assert jax.config.jax_enable_x64 is True

#: Six sample offsets of the per-cell quadratic fit.
SAMPLES = np.array(
    [[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0], [1.0, 1.0]]
)

#: Cells per axis of the patch the value-only route reads. The route averages
#: each cell against its four axial neighbours, so the patch has to carry them;
#: the cell the sweep crosses is the middle one.
POINT_PATCH = 3
POINT_PATCH_CENTRE = (POINT_PATCH * POINT_PATCH) // 2

#: Samples of the swept level, either side of the crossing. The cut cell sweeps
#: two percent of the cell's flux half-range. The sector sweeps a tenth of that,
#: because the normal form represents the level set only in the neighbourhood of
#: the saddle: wider sweeps reach a level at which a straight boundary interval
#: enters the sector fan, an entry event of the representation rather than of the
#: integrated density.
CUT_CELL_HALF_WIDTH = 0.02
SECTOR_HALF_WIDTH = 2.0e-3
LEVEL_COUNT = 21

#: Relative agreement required between the analytic tangent and a central
#: difference of the booked current.
TANGENT_RTOL = 1e-6

#: Offsets from the crossing level at which the tangent is checked, bracketing
#: the crossing on both sides. The exact crossing level is deliberately excluded
#: for the sector: there the level set crosses a saddle and the fan gains a new
#: straight boundary interval, so the booked current is continuous across that
#: level but its derivative steps, and a central difference taken across it
#: measures the event rather than the derivative. The sector offsets are wider
#: for the same reason the sector sweep is narrower: the normal form owns the
#: level set only near the saddle.
TANGENT_OFFSETS = {
    "cut-cell": (-0.004, 0.004),
    "x-point-sector": (-0.01, 0.01),
}

#: The value-only route's step must exceed the booked route's largest step by
#: this factor for the comparator to separate the two routes.
STEP_CONTRAST = 100.0


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class FiniteEdgeDensity:
    """A density with a finite edge value and nothing beyond the level.

    The density is linear in normalized flux on the confined side, so it carries
    a finite edge value and a defined slope everywhere the booked polygon
    reaches. A density clipped at the axis would introduce an interior kink the
    booked interior does not carry, and the tangent would then disagree with a
    central difference for a reason unrelated to the moving boundary.
    """

    edge: object = 1.5
    slope: object = 1.0

    def current_density(self, radius, normalized):
        del radius
        confined = self.edge + self.slope * normalized
        return jnp.where(normalized > 1.0, 0.0, confined)

    def density_gradient(self, radius, normalized):
        del radius
        return jnp.where(normalized <= 1.0, self.slope, 0.0)


def _design(points):
    x, y = np.asarray(points, dtype=np.float64).T
    return np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], axis=1)


def _field(coefficients, points):
    return _design(points) @ np.asarray(coefficients, dtype=np.float64)


def _case(kind):
    """Return the cell, its cell-local quadratic field and its crossing level.

    ``cut-cell`` carries a field linear in the vertical coordinate, so the level
    set crosses the cell as one straight cut. ``x-point-sector`` carries a
    saddle at the cell centre, so the level set crosses as two conical sectors
    the normal form owns. Both sweeps pass through the level at which the
    cell-centre sample would cross.
    """

    vertices = np.array([[1.0, -1.0], [3.0, -1.0], [3.0, 1.0], [1.0, 1.0]])
    centre = np.array([2.0, 0.0])
    if kind == "cut-cell":
        coefficients = np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
        sector = False
        half_width = CUT_CELL_HALF_WIDTH
    elif kind == "x-point-sector":
        coefficients = np.array([0.0, 0.0, 0.0, 1.0, 0.0, -1.0])
        sector = True
        half_width = SECTOR_HALF_WIDTH
    else:
        raise ValueError(f"unknown cell case {kind!r}")
    return SimpleNamespace(
        vertices=vertices,
        centre=centre,
        coefficients=np.asarray(coefficients, dtype=np.float64),
        axis=-1.0,
        crossing=0.0,
        sector=sector,
        half_width=half_width,
    )


def _geometry(case):
    return SimpleNamespace(
        vertices=jnp.asarray(case.vertices[None]),
        vertex_count=jnp.asarray([4]),
        centre=jnp.asarray(case.centre[None]),
        pitch=jnp.ones(1),
        sample_points=jnp.asarray((SAMPLES + case.centre)[None]),
        fit_inverse=jnp.asarray(np.linalg.inv(_design(SAMPLES))[None]),
    )


def _saddle(case):
    """Return the separatrix jet of the clip region at the cell centre.

    The signed field the booking clips on is ``sigma * psi`` with ``sigma`` of
    the same sign as the density profile, so its Hessian at the saddle is the
    negated field Hessian. Its sectors are therefore the confined pair. The
    cubic term is a small nonzero value: the ray construction takes a cubic
    separator per sector, and a field with no third derivative leaves it
    degenerate.
    """
    return topology.saddle_normal_form(
        jnp.asarray(case.centre),
        jnp.asarray([[-2.0, 0.0], [0.0, 2.0]]),
        jnp.asarray([[[6.0e-3, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]),
        1e-12,
        jnp.zeros((2, 2, 2, 2)),
    )


def _read(case, level, coefficients=None):
    """Return the topology read of the cell at one flux level.

    The fragment areas and edge intervals come from the package's own
    level-set reader, so the fixture sees the same clip region the solver does.
    """
    coefficients = case.coefficients if coefficients is None else coefficients
    coefficients = jnp.asarray(coefficients, dtype=jnp.float64)
    policy = SimpleNamespace(fragment_capacity=4, edge_tolerance=1e-12)
    area, intervals, valid, _, *_ = topology._fragments_at_level(
        _geometry(case),
        coefficients[None],
        -1.0,
        level,
        _saddle(case),
        jnp.asarray([case.sector]),
        policy,
    )
    selected = area > 1e-14
    return SimpleNamespace(
        field_coefficients=coefficients[None],
        axis_flux=jnp.asarray(case.axis),
        boundary_flux=jnp.asarray(level),
        fragment_selected=selected,
        membership=jnp.sum(jnp.where(selected, area, 0.0), axis=1) / jnp.asarray(4.0),
        normal_form_cells=jnp.asarray([case.sector]),
        saddle_form=_saddle(case),
        edge_interval=intervals,
        valid=valid,
        qualified=valid,
    )


def _quadrature():
    # The production polarity: the clip keeps the confined side of the level.
    return Quadrature(segments=64, sigma=-1.0)


def _booked(case, level, coefficients=None, profile=None):
    coefficients = case.coefficients if coefficients is None else coefficients
    profile = FiniteEdgeDensity() if profile is None else profile
    psi = jnp.asarray(_field(coefficients, SAMPLES)[None])
    booked = book(
        psi,
        _read(case, level, coefficients),
        _geometry(case),
        profile,
        _quadrature(),
    )
    return np.asarray(booked.cell_current)[0]


def _area(case):
    x, y = case.vertices.T
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


class CentreSampledDensity:
    """The value-only route's source, already sampled at one cell centre."""

    def __init__(self, current):
        self._current = jnp.asarray(current)

    def cell_current(self, radius, area, masks):
        del radius, area
        return jnp.where(masks.profile_participation, self._current, 0.0)


def _pitch(case):
    """Return the cell's own radial and vertical pitch."""

    return np.array(
        [
            case.vertices[1, 0] - case.vertices[0, 0],
            case.vertices[2, 1] - case.vertices[1, 1],
        ]
    )


def _point(case, level):
    """Integrate the level through the operator's own value-only route.

    The patch carries the read cell and its four axial neighbours, because a
    cell average is not a point sample: the route gathers the neighbour cells
    with the production stencil and weights, so a change to the rule the
    solver applies moves this control with it.
    """

    offsets = np.array(
        [
            (radial, height)
            for radial in range(POINT_PATCH)
            for height in range(POINT_PATCH)
        ],
        dtype=np.float64,
    ) - 0.5 * (POINT_PATCH - 1)
    pitch = _pitch(case)
    centres = case.centre + offsets * pitch
    normalized = (_field(case.coefficients, offsets) - case.axis) / (level - case.axis)
    density = np.asarray(
        FiniteEdgeDensity().current_density(0.0, jnp.asarray(normalized))
    )
    stencil, weight = _cubic_cell_average_stencil((POINT_PATCH, POINT_PATCH))
    area = np.full(centres.shape[0], _area(case))
    masks = SimpleNamespace(
        profile_participation=jnp.ones(centres.shape[0], dtype=bool)
    )
    current = point_route_current(
        CentreSampledDensity(density * area),
        jnp.asarray(centres),
        jnp.asarray(area),
        masks,
        stencil,
        weight,
    )
    return float(np.asarray(current)[POINT_PATCH_CENTRE])


def _levels(case, width=None):
    width = case.half_width if width is None else width
    return case.crossing + np.linspace(-width, width, LEVEL_COUNT)


#: Unit flux functions of the profile the builder's operator is assembled with.
SURFACES = np.linspace(0.0, 1.0, 5)


def _booked_operator():
    """Return the operator the DIII-D builder attaches to a booked profile.

    The lattice, the wall and the moment blocks are the builder's own objects
    and its own booked-route seam, so what this reads is the route the builder
    attaches rather than a declaration about it.
    """

    radius = np.linspace(1.0, 2.0, 9)
    height = np.linspace(-1.0, 1.0, 9)
    lattice = FluxLattice(radius, height)
    wall = np.array([[1.0, -1.0], [2.0, -1.0], [2.0, 1.0], [1.0, 1.0]])
    grid_moments, wall_moments, sample_coordinate, sample_moments = _moment_couplings(
        lattice, wall
    )
    nodes = lattice.node_count
    sample_rows = int(np.shape(sample_coordinate)[0])
    conductors = 3
    source = ForwardSource(
        core=DomainProfile(
            p_prime=SampledFluxFunction(SURFACES, np.ones_like(SURFACES)),
            ff_prime=SampledFluxFunction(SURFACES, np.ones_like(SURFACES)),
        )
    )
    profile = ForwardProfile.from_lattice(
        lattice,
        source,
        external_current=np.zeros(nodes),
        source_to_grid=np.zeros((nodes, conductors)),
        plasma_to_grid=np.zeros((nodes, nodes)),
        source_to_wall=np.zeros((len(wall), conductors)),
        plasma_to_wall=np.zeros((len(wall), nodes)),
        wall_coordinate=wall,
        polarity=1,
    )
    booked = _booked_profile(
        profile,
        grid_moments=grid_moments,
        wall_moments=wall_moments,
        sample_coordinate=sample_coordinate,
        sample_moments=sample_moments,
        source_to_sample=np.zeros((sample_rows, conductors)),
        plasma_to_sample=np.zeros((sample_rows, nodes)),
    )
    return booked.operator


def test_the_diiiid_builder_books_a_finite_edge():
    operator = _booked_operator()
    sample = operator.sample
    print(
        f"BUILDER linear_moments {bool(operator.use_linear_moments)} "
        f"sample {None if sample is None else sample.node_number} "
        f"columns {None if sample is None else int(sample.source_target.shape[1])}"
    )
    assert operator.use_linear_moments, (
        "the builder left the operator on the value-only point-current route, so "
        "the source integral steps where a sample crosses the edge"
    )
    assert sample is not None, (
        "the builder attached no sampling target, so the booked route has no "
        "nodes to clip the boundary on"
    )
    assert sample.source_target.shape[1] == operator.grid.source_target.shape[1], (
        "the sampling target carries a shorter conductor response than the grid, "
        "so the solve cannot contract it against the conductor vector"
    )
    assert sample.node_number > operator.grid.node_number, (
        "the sampling nodes are the cell vertices, so they outnumber the centres"
    )


@pytest.mark.parametrize("kind", ["cut-cell", "x-point-sector"])
def test_the_finite_edge_integral_takes_no_step(kind):
    case = _case(kind)
    values = np.array([_booked(case, level) for level in _levels(case)])
    steps = np.abs(np.diff(values))
    print(
        f"ROUTE booked {kind} max_step {steps.max():.6e} "
        f"scale {np.abs(values).max():.6e}"
    )
    assert steps.max() <= 1e-2 * np.abs(values).max()


@pytest.mark.parametrize("kind", ["cut-cell", "x-point-sector"])
def test_the_value_only_route_steps_at_the_crossing(kind):
    case = _case(kind)
    booked = np.array([_booked(case, level) for level in _levels(case)])
    point = np.array([_point(case, level) for level in _levels(case)])
    booked_step = np.abs(np.diff(booked)).max()
    point_step = np.abs(np.diff(point)).max()
    print(f"CONTROL {kind} booked {booked_step:.6e} point {point_step:.6e}")
    assert point_step > STEP_CONTRAST * booked_step


@pytest.mark.parametrize("kind", ["cut-cell", "x-point-sector"])
def test_book_tangent_matches_a_central_difference(kind):
    case = _case(kind)
    coefficients = np.asarray(case.coefficients, dtype=np.float64)
    direction = np.array([0.3, -0.7, 0.2, 0.5, -0.4, 0.6])
    psi = jnp.asarray(_field(coefficients, SAMPLES)[None])
    psi_tangent = jnp.asarray(_field(direction, SAMPLES)[None])
    profile = FiniteEdgeDensity()
    quadrature = _quadrature()
    for step in TANGENT_OFFSETS[kind]:
        level = case.crossing + step
        booked = book(psi, _read(case, level), _geometry(case), profile, quadrature)
        tangent = book_tangent(
            psi,
            psi_tangent,
            _read(case, level),
            jnp.asarray(direction[None]),
            jnp.asarray(0.0),
            jnp.asarray(0.0),
            _geometry(case),
            profile,
            profile.density_gradient,
            quadrature,
            booked,
        )
        offset = 1e-5
        measured = (
            _booked(case, level, coefficients + offset * direction)
            - _booked(case, level, coefficients - offset * direction)
        ) / (2 * offset)
        analytic = float(np.asarray(tangent.cell_current)[0])
        print(
            f"TANGENT {kind} step {step:+.4f} "
            f"analytic {analytic:.12e} measured {measured:.12e}"
        )
        assert abs(measured) > 1e-3
        np.testing.assert_allclose(analytic, measured, rtol=TANGENT_RTOL)
