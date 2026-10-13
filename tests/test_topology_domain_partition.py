"""Production domain partition authority on committed topology operands."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nova.utilities.importmanager import skip_import

with skip_import("jax"):
    import jax
    import jax.numpy as jnp

    from nova.equilibrium.domain import (
        PlasmaDomain,
        axis_connected_component,
        classify_domains,
    )
    from nova.geometry.hexstencil import hex_stencil


OPERANDS = (
    Path(__file__).parents[1]
    / "docs/figures/topology-visual-corroboration/mast-topology-operands.npz"
)


def _cached_partition(row: int, *, compiled: bool):
    with np.load(OPERANDS, allow_pickle=False) as stored:
        coordinate = stored[f"row_{row:02d}_cell_rz"]
        cached_label = stored[f"row_{row:02d}_domain_labels"]
        axis = stored[f"row_{row:02d}_selected_o"][0]

    radius = np.unique(coordinate[:, 0])
    height = np.unique(coordinate[:, 1])
    shape = (height.size, radius.size)
    closed_flat = np.isin(
        cached_label,
        (int(PlasmaDomain.CORE), int(PlasmaDomain.PRIVATE_FLUX)),
    )
    inside_flat = cached_label != int(PlasmaDomain.EXCLUDED_MATERIAL)
    confined = jnp.asarray(closed_flat.reshape((radius.size, height.size)).T)
    rings = jnp.asarray(hex_stencil(shape))
    link_admissible = jnp.ones(rings.shape, dtype=bool)
    seed_index = int(np.argmin(np.sum((coordinate - axis) ** 2, axis=1)))
    seed_flat = np.zeros(coordinate.shape[0], dtype=bool)
    seed_flat[seed_index] = True
    seed = jnp.asarray(seed_flat.reshape((radius.size, height.size)).T)

    component_read = axis_connected_component
    if compiled:
        component_read = jax.jit(component_read)
        component = component_read(confined, rings, link_admissible, seed)
    else:
        with jax.disable_jit():
            component = component_read(confined, rings, link_admissible, seed)
    connected = component.T.reshape(-1)
    masks = classify_domains(
        jnp.zeros_like(connected, dtype=jnp.float64),
        jnp.asarray(closed_flat),
        connected,
        jnp.asarray(inside_flat),
    )
    return cached_label, np.asarray(connected), masks


@pytest.mark.parametrize("row", [0, 1])
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
def test_committed_partition_uses_the_axis_hex_component(row, compiled):
    """The published core/private split is the cached axis component exactly."""
    cached_label, connected, masks = _cached_partition(row, compiled=compiled)
    np.testing.assert_array_equal(np.asarray(masks.core), connected)
    np.testing.assert_array_equal(
        np.asarray(masks.private_flux),
        np.isin(
            cached_label,
            (int(PlasmaDomain.CORE), int(PlasmaDomain.PRIVATE_FLUX)),
        )
        & ~connected,
    )
