"""Wall-height augmentation of the residual shadow."""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium.connectivity_boundary import wall_height_shadow_mask
from nova.equilibrium.domain import DomainMasks, PlasmaDomain
from nova.equilibrium.forward_operator import ForwardFluxOperator


def _x_points(*heights: float) -> jax.Array:
    """Return fixed-shape finite X-point rows at the requested heights."""
    return jnp.asarray([[1.0, height, 0.0] for height in heights])


def test_height_shadow_changes_only_at_wall_node_crossings_eager_and_jit() -> None:
    """Moving a height limit changes exactly the wall nodes it crosses."""
    wall_height = jnp.asarray([-1.0, -0.25, 0.4, 1.0])

    for evaluate in (wall_height_shadow_mask, jax.jit(wall_height_shadow_mask)):
        before = evaluate(wall_height, 0.0, _x_points(-0.5, 0.75))
        after_upper_crossing = evaluate(wall_height, 0.0, _x_points(-0.5, 1.25))
        after_lower_crossing = evaluate(wall_height, 0.0, _x_points(-1.25, 1.25))

        np.testing.assert_array_equal(before, [True, False, False, True])
        np.testing.assert_array_equal(
            before ^ after_upper_crossing, [False, False, False, True]
        )
        np.testing.assert_array_equal(
            after_upper_crossing ^ after_lower_crossing,
            [True, False, False, False],
        )


def _operator_with_components() -> tuple[ForwardFluxOperator, jax.Array]:
    """Return a minimal operator with independent fixed flood and wall data."""
    operator = object.__new__(ForwardFluxOperator)
    operator.grid = SimpleNamespace(node_number=3)
    operator.wall = SimpleNamespace(
        node_number=2,
        coordinate=jnp.asarray([[1.0, -0.8], [1.0, 0.8]]),
    )
    operator.sample = SimpleNamespace(node_number=1)
    operator.topology = SimpleNamespace(
        split_flux_map=lambda physical: (physical[:3], physical[3:])
    )
    operator._fixed_design_topology = SimpleNamespace(
        grid=lambda _grid_flux: (jnp.zeros((1, 3)), _x_points(-0.5, 0.5))
    )
    masks = DomainMasks(
        label=jnp.asarray(
            [PlasmaDomain.CORE, PlasmaDomain.PRIVATE_FLUX, PlasmaDomain.CORE],
            dtype=jnp.int8,
        ),
        psi_norm=jnp.zeros(3),
    )
    topology = SimpleNamespace(axis=jnp.asarray([1.0, 0.0]))
    operator._fixed_design_read = lambda _physical, _requested=None: (
        masks,
        topology,
        ~masks.private_flux,
        jnp.asarray(True),
    )
    return operator, masks.private_flux


def test_wall_augmentation_preserves_flood_component_bitwise() -> None:
    """The saddle-aware flood remains the complete interior shadow authority."""
    operator, expected_flood = _operator_with_components()
    trial = jnp.arange(operator.node_number, dtype=jnp.float32)

    for evaluate in (
        operator.residual_shadow_components,
        jax.jit(operator.residual_shadow_components),
    ):
        flood_shadow, wall_shadow = evaluate(trial)
        np.testing.assert_array_equal(flood_shadow, expected_flood)
        np.testing.assert_array_equal(wall_shadow, [True, True])

    combined = operator.residual_shadow_mask(trial)
    np.testing.assert_array_equal(combined[: operator.grid.node_number], expected_flood)
    np.testing.assert_array_equal(combined, [False, True, False, True, True, False])

    mapped = trial + 10.0
    excluded = operator._exclude_shadow_residual(trial, mapped)
    np.testing.assert_array_equal(excluded, [10.0, 1.0, 12.0, 3.0, 4.0, 15.0])


def test_height_augmentation_reduces_synthetic_promotion_mask_flips() -> None:
    """Holding an excluded wall node removes its induced flood-mask excursions."""
    wall_shadow = wall_height_shadow_mask(jnp.asarray([0.8]), 0.0, _x_points(-0.5, 0.5))

    def promotion_masks(augment: bool) -> jax.Array:
        def promote(wall_value, _unused):
            flood_shadow = jnp.asarray([wall_value > 0.5])
            shadow = jnp.concatenate(
                (flood_shadow, wall_shadow if augment else jnp.zeros(1, dtype=bool))
            )
            mapped_wall = 1.0 - wall_value
            next_wall = jnp.where(shadow[1], wall_value, mapped_wall)
            return next_wall, ~shadow

        return jax.lax.scan(promote, jnp.asarray(0.0), xs=None, length=5)[1]

    def flip_count(participation: jax.Array) -> int:
        return int(jnp.sum(participation[1:] != participation[:-1]))

    baseline = promotion_masks(False)
    augmented = promotion_masks(True)
    assert flip_count(augmented) < flip_count(baseline)
    assert flip_count(baseline) == 4
    assert flip_count(augmented) == 0
