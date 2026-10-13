"""Analytic-separatrix authority for the closed-form fixture exterior."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from shapely.geometry import Polygon

from nova.equilibrium.forward_operator import ForwardFluxOperator
from nova.jax.config import configure_dtypes
from scripts.analytic_oracle_fixtures import measure as fixture
from scripts.prototypes.support import geometry
from scripts.prototypes.support.geometry import polygon_rings
from tests.rotating_equilibrium_references import reference_cases

# One clipped cell of the diverted exact support, as the support owner cuts it:
# a thirteen-vertex near-edge sliver whose fan from its first vertex is not a
# decomposition because the ring doubles back on itself.  Translated to its own
# first vertex so the fixture is small and independent of the machine.
NON_CONVEX_CELL_RING = np.array(
    [
        [0.0, 0.0],
        [-0.010210227050424514, 0.0],
        [-0.009620135671014607, -0.000992275430406142],
        [-0.009021939455008399, -0.0019983622547328217],
        [-0.008422888689574082, -0.0030060715065394206],
        [-0.007822980999301565, -0.004015410036983358],
        [-0.007222213996537441, -0.005026384733812339],
        [-0.006620585280497915, -0.006039002518609005],
        [-0.0060180924396249225, -0.007053270356079944],
        [-0.00474472192758113, -0.005555899559215538],
        [-0.0034764670390978747, -0.004067178773966784],
        [-0.002213276277150955, -0.0025870044128499803],
        [-0.0009550989996911508, -0.0011152747390490436],
    ]
)


class _UnitDensity:
    """A stand-in source whose density is uniformly one.

    The region a moment rule sums is separable from the density it integrates:
    under a uniform density the moments of any polygon are its shoelace area
    and first moments in closed form, so comparing against those isolates a rule
    that sums the wrong region from one that merely loses accuracy.
    """

    @staticmethod
    def toroidal_current_density(x, y):
        return np.ones_like(np.asarray(x, dtype=np.float64))


def _shoelace_moments(ring: np.ndarray) -> np.ndarray:
    x, y = ring[:, 0], ring[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    area = 0.5 * float(np.sum(cross))
    return np.array(
        [
            area,
            float(np.sum((x + np.roll(x, -1)) * cross)) / 6.0,
            float(np.sum((y + np.roll(y, -1)) * cross)) / 6.0,
        ]
    )


def test_clipped_analytic_moments_close_the_weak_fixture_without_a_topology_read(
    monkeypatch,
):
    """The explicit analytic clip closes its own map to binary64 roundoff."""
    configure_dtypes()
    assert jnp.asarray(1.0).dtype == jnp.float64
    case = reference_cases()["weak-rotation-reactor"].static_limit()
    machine = fixture.cached_machine(
        case,
        -110,
        wall_nodes=fixture.WALL_POINT_COUNT,
    )
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    exact = fixture.exact_state(case, coordinates)
    empty_operator = fixture.forward_operator(case, machine)

    def refuse_production_read(*_args, **_kwargs):
        raise AssertionError(
            "fixture construction consulted the production topology read"
        )

    with monkeypatch.context() as isolated:
        isolated.setattr(
            ForwardFluxOperator,
            "_fixed_design_read",
            refuse_production_read,
        )
        physical = fixture.exact_current_moments(
            case, empty_operator, exact, analytic=case, support_arm="legacy"
        )

    coefficients = empty_operator.coupling_current_moments(physical)
    internal = fixture._internal_flux_image(empty_operator, coefficients)
    exterior = exact - internal
    operator = fixture.forward_operator(case, machine, exterior)
    mapped = np.asarray(operator.external()) + fixture._internal_flux_image(
        operator, coefficients
    )

    assert np.max(np.abs(mapped - exact)) <= 1.0e-12
    assert np.count_nonzero(np.asarray(physical.cell_current)) > 0
    whole = fixture.whole_cell_current_moments(case, empty_operator, exact)
    assert not np.array_equal(
        np.asarray(physical.cell_current), np.asarray(whole.cell_current)
    )


def test_clipped_region_moments_follow_the_signed_fan_on_a_non_convex_cell():
    """A clipped cell's moments are the shoelace moments of the region it clips.

    The fixture's exact arm integrates the clipped cells through the shared
    moment owner, which sums each fan triangle's *signed* jacobian.  On a convex
    ring any per-triangle convention telescopes, so the two rules agree and the
    choice is invisible; on the non-convex rings the independent analytic clip
    actually produces they do not, and weighting a reversed triangle by the
    magnitude of its jacobian adds that region a second time instead of
    cancelling it.  Under a uniform density the true moments are closed form,
    so this measures the region a rule sums rather than the accuracy of either
    quadrature.
    """
    configure_dtypes()
    assert jnp.asarray(1.0).dtype == jnp.float64

    polygon = Polygon(NON_CONVEX_CELL_RING)
    assert polygon.is_valid and not polygon.is_empty
    rings = polygon_rings(polygon)
    assert len(rings) == 1
    ring = rings[0]

    edge_a, edge_b = ring[1:-1] - ring[0], ring[2:] - ring[0]
    cross = edge_a[:, 0] * edge_b[:, 1] - edge_a[:, 1] * edge_b[:, 0]
    assert np.any(cross < 0.0) and np.any(cross > 0.0), (
        "the fixture polygon must be non-convex about its first vertex, or the "
        "rule this test guards is not exercised"
    )

    centre = ring.mean(axis=0)
    reference = _shoelace_moments(ring)
    reference[1:] -= centre * reference[0]
    assert reference[0] > 0.0

    moments = np.asarray(
        geometry.analytic_moments(_UnitDensity(), [polygon], np.asarray([centre]))
    )[:, 0]

    relative = np.abs(moments - reference) / np.abs(reference)
    assert np.max(relative) <= 1.0e-12, (moments, reference)
    assert not hasattr(fixture, "_polygon_current_moments")
    assert not hasattr(fixture, "_signed_ring_area")
