"""Qualified, connectivity-vetoed wall-height residual exclusion."""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nova.equilibrium.connectivity_boundary import wall_height_shadow_mask
from nova.equilibrium.domain import DomainMasks, PlasmaDomain
from nova.equilibrium.fixed_point import picard
from nova.equilibrium.forward_operator import ForwardFluxOperator


def _points(*heights: float) -> jax.Array:
    """Return fixed-shape X-point coordinates at the requested heights."""
    rows = [[1.0, height] for height in heights]
    while len(rows) < 2:
        rows.append([jnp.nan, jnp.nan])
    return jnp.asarray(rows[:2])


def _height_mask(
    wall_height,
    primary_height,
    qualified_heights,
    *,
    private_wall=None,
    previous=None,
    band=0.02,
):
    """Evaluate the production height helper with compact test operands."""
    wall_height = jnp.asarray(wall_height, dtype=jnp.float64)
    if private_wall is None:
        private_wall = jnp.ones(wall_height.shape, dtype=bool)
    if previous is None:
        previous = jnp.zeros(wall_height.shape, dtype=bool)
    primary = jnp.asarray([1.0, primary_height], dtype=wall_height.dtype)
    return wall_height_shadow_mask(
        wall_height,
        0.0,
        primary,
        _points(*qualified_heights),
        private_wall,
        previous,
        jnp.asarray(band, dtype=wall_height.dtype),
        0.1,
    )


def test_only_qualified_primary_and_opposite_candidate_define_limits() -> None:
    """Raw notch and coil-null extrema cannot define a wall limit."""
    wall_height = jnp.linspace(-1.0, 1.0, 37)
    unqualified = _height_mask(wall_height, -0.5, ())
    qualified = _height_mask(wall_height, -0.5, (-0.5, 0.5))

    np.testing.assert_array_equal(unqualified, np.zeros(37, dtype=bool))
    assert int(jnp.sum(qualified)) > 0


@pytest.mark.parametrize(
    ("geometry", "wall_count", "review_false_exclusions"),
    [
        ("disconnected double null", 37, 19),
        ("snowflake minus", 37, 37),
        ("wall notch MAST", 37, 19),
        ("wall notch DIII-D", 84, 38),
        ("coil-null column MAST", 37, 35),
        ("coil-null column DIII-D", 84, 77),
        ("axis crossing", 37, 37),
    ],
)
def test_common_flux_wall_rows_have_zero_false_residual_exclusions(
    geometry, wall_count, review_false_exclusions
) -> None:
    """Height limits cannot suppress a wall equation without private proof."""
    del geometry
    wall_height = jnp.linspace(-1.2, 1.2, wall_count)
    private_wall = jnp.zeros(wall_count, dtype=bool)
    shadow = _height_mask(
        wall_height,
        -0.3,
        (-0.3, 0.3),
        private_wall=private_wall,
    )
    trial = jnp.arange(wall_count, dtype=jnp.float64)
    mapped = trial + 1.0
    residual = trial - jnp.where(shadow, trial, mapped)

    assert review_false_exclusions > 0
    assert int(jnp.sum(shadow)) == 0
    np.testing.assert_array_equal(residual, -jnp.ones(wall_count))


def test_missing_candidate_carries_promoted_window_eager_and_jit() -> None:
    """A transiently absent side leaves its accepted wall participation fixed."""
    wall_height = jnp.asarray([-1.0, -0.25, 0.25, 1.0])

    def evaluate(points, previous):
        return wall_height_shadow_mask(
            wall_height,
            0.0,
            jnp.asarray([1.0, -0.5]),
            points,
            jnp.ones(4, dtype=bool),
            previous,
            0.02,
            0.1,
        )

    for call in (evaluate, jax.jit(evaluate)):
        promoted = call(_points(-0.5, 0.5), jnp.zeros(4, dtype=bool))
        missing = call(_points(), promoted)
        np.testing.assert_array_equal(promoted, [True, False, False, True])
        np.testing.assert_array_equal(missing, promoted)


def test_equality_hover_has_zero_chatter_flips_eager_and_jit() -> None:
    """Sub-band height motion preserves the previously promoted equality bit."""
    heights = jnp.asarray([0.795, 0.800, 0.805, 0.799, 0.801])

    def trajectory(values):
        def promote(previous, height):
            point = jnp.asarray([1.0, height])
            current = wall_height_shadow_mask(
                jnp.asarray([0.8]),
                0.0,
                point,
                jnp.stack((point, jnp.asarray([jnp.nan, jnp.nan]))),
                jnp.asarray([True]),
                previous,
                0.02,
                0.1,
            )
            return current, current

        return jax.lax.scan(promote, jnp.asarray([False]), values)[1]

    for call in (trajectory, jax.jit(trajectory)):
        masks = call(heights)
        assert int(jnp.sum(masks[1:] != masks[:-1])) == 0


def _operator_with_components() -> tuple[ForwardFluxOperator, jax.Array]:
    """Return a minimal operator with independent flood and wall evidence."""
    operator = object.__new__(ForwardFluxOperator)
    operator.grid = SimpleNamespace(node_number=3)
    operator.wall = SimpleNamespace(
        node_number=2,
        coordinate=jnp.asarray([[1.0, -0.8], [1.0, 0.8]]),
    )
    operator.sample = SimpleNamespace(node_number=1)
    operator._wall_height_hysteresis = jnp.asarray(0.02)
    operator._x_qualification_distance = jnp.asarray(0.1)
    masks = DomainMasks(
        label=jnp.asarray(
            [PlasmaDomain.CORE, PlasmaDomain.PRIVATE_FLUX, PlasmaDomain.CORE],
            dtype=jnp.int8,
        ),
        psi_norm=jnp.zeros(3),
    )
    topology = SimpleNamespace(
        axis=jnp.asarray([1.0, 0.0]), x_point=jnp.asarray([1.0, -0.5])
    )
    operator._fixed_design_read = lambda _physical, _requested=None: (
        masks,
        topology,
        ~masks.private_flux,
        jnp.asarray(True),
    )
    operator._connectivity_read = lambda _physical, _topology, classify=False: {
        "xset": _points(-0.5, 0.5),
        "private_wall_node_mask": jnp.asarray([True, False]),
    }
    return operator, masks.private_flux


def test_composed_mask_preserves_flood_and_vetoes_common_wall() -> None:
    """The flood stays the interior authority and common wall rows participate."""
    operator, expected_flood = _operator_with_components()
    trial = jnp.arange(operator.node_number, dtype=jnp.float32)

    for evaluate in (
        operator.residual_shadow_components,
        jax.jit(operator.residual_shadow_components),
    ):
        flood_shadow, wall_shadow = evaluate(trial)
        np.testing.assert_array_equal(flood_shadow, expected_flood)
        np.testing.assert_array_equal(wall_shadow, [True, False])

    combined = operator.residual_shadow_mask(trial)
    np.testing.assert_array_equal(combined, [False, True, False, True, False, False])
    mapped = trial + 10.0
    excluded = operator._exclude_shadow_residual(trial, mapped)
    np.testing.assert_array_equal(excluded, [10.0, 1.0, 12.0, 3.0, 14.0, 15.0])


def test_picard_uses_carried_mask_for_residual_then_promotes_it() -> None:
    """The residual consumes the old mask before the accepted state advances it."""

    def mapped(state):
        return state + 1.0

    def shadowed(state, shadow):
        del state
        return jnp.where(shadow, 0.0, 1.0)

    def proposed(state, previous):
        return jnp.where(state > 0.5, jnp.ones_like(previous), previous)

    result = picard(
        mapped,
        jnp.asarray([0.0]),
        evaluations=2,
        relaxation=1.0,
        shadow_mask_fn=lambda _state: jnp.asarray([False]),
        promoted_shadow_mask_fn=proposed,
        shadowed_map_fn=shadowed,
    )

    np.testing.assert_array_equal(result.state, [0.0])
    np.testing.assert_array_equal(result.shadow_mask_changes, [1, 0])
