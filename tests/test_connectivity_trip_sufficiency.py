"""Fixed-trip sufficiency and reciprocal-edge invariants for hex labels."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from nova.utilities.importmanager import skip_import

with skip_import("jax"):
    import jax
    import jax.numpy as jnp

    from nova.equilibrium import flux_surface_connectivity
    from nova.equilibrium.connectivity_boundary import (
        _canonicalize_reciprocal_hex_edges,
        _saddle_aware_axis_component,
    )
    from nova.geometry.hexstencil import hex_stencil


def _serpentine_links(size):
    """Return a Hamiltonian path through a square half-offset hex raster."""
    rings = hex_stencil((size, size))
    row_for_centre = {int(centre): row for row, centre in enumerate(rings[:, 0])}
    path = []
    for row in range(1, size - 1):
        columns = range(1, size - 1) if row % 2 == 1 else range(size - 2, 0, -1)
        path.extend(row * size + column for column in columns)

    links = np.zeros_like(rings, dtype=bool)
    links[:, 0] = True
    for left, right in zip(path[:-1], path[1:], strict=True):
        left_row = row_for_centre[left]
        right_row = row_for_centre[right]
        left_slots = np.flatnonzero(rings[left_row] == right)
        right_slots = np.flatnonzero(rings[right_row] == left)
        assert left_slots.size == right_slots.size == 1
        links[left_row, left_slots[0]] = True
        links[right_row, right_slots[0]] = True
    return rings, links, path


@pytest.mark.parametrize("size", [33, 65])
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
def test_serpentine_component_saturates_beyond_grid_diameter(size, compiled):
    """The static cell-count cap labels a path longer than both grid axes."""
    rings, links, path = _serpentine_links(size)
    confined = jnp.zeros((size, size), dtype=bool).at[1:-1, 1:-1].set(True)
    seed = jnp.zeros((size, size), dtype=bool).at[1, 1].set(True)

    diameter_labels = (
        flux_surface_connectivity.label_saddle_aware_hex_connected_components(
            confined,
            jnp.asarray(rings),
            jnp.asarray(links),
            2 * size,
        )
    )
    diameter_component = np.asarray(diameter_labels) == int(diameter_labels[1, 1])
    assert int(np.count_nonzero(diameter_component)) == {33: 133, 65: 261}[size]

    component_read = _saddle_aware_axis_component
    if compiled:
        component_read = jax.jit(component_read)
    component = component_read(
        confined,
        jnp.asarray(rings),
        jnp.asarray(links),
        seed,
    )
    assert int(np.count_nonzero(np.asarray(component))) == len(path)


def test_reciprocal_edge_decisions_are_canonical():
    """One closed direction closes both representations of a shared edge."""
    rings = hex_stencil((5, 5))
    centre_row = len(rings) // 2
    centre = int(rings[centre_row, 0])
    interior_centres = set(rings[:, 0])
    slot = next(
        slot
        for slot in range(1, rings.shape[1])
        if rings[centre_row, slot] in interior_centres
    )
    neighbour = int(rings[centre_row, slot])
    neighbour_row = int(np.flatnonzero(rings[:, 0] == neighbour)[0])
    reverse_slot = int(np.flatnonzero(rings[neighbour_row] == centre)[0])
    links = np.ones_like(rings, dtype=bool)
    links[centre_row, slot] = False

    canonical = np.asarray(
        _canonicalize_reciprocal_hex_edges(jnp.asarray(rings), jnp.asarray(links))
    )
    assert not canonical[centre_row, slot]
    assert not canonical[neighbour_row, reverse_slot]

    label_source = inspect.getsource(
        flux_surface_connectivity._iterate_component_labels
    )
    component_source = inspect.getsource(_saddle_aware_axis_component)
    assert "lax.fori_loop" in label_source
    assert "while_loop" not in label_source
    assert "while_loop" not in component_source
