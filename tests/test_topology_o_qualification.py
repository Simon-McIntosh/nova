"""Material and connectivity qualification of magnetic-axis candidates."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nova.utilities.importmanager import skip_import

with skip_import("jax"):
    import jax
    import jax.numpy as jnp

    from nova.biot.null import Null1D, Null2D
    from nova.equilibrium.fixed_point import kink_aware_newton_krylov
    from nova.equilibrium.forward_operator import axis_cell_seed
    from nova.equilibrium.topology import (
        NoQualifiedAxisError,
        Topology,
        require_qualified_axis,
    )
    from nova.geometry.hexstencil import hex_stencil
    from nova.jax.config import configure_dtypes


OPERANDS = (
    Path(__file__).parents[1]
    / "docs/figures/topology-visual-corroboration/mast-topology-operands.npz"
)


@pytest.fixture(scope="module", autouse=True)
def _double_precision():
    """Match production topology precision."""
    configure_dtypes()


def _topology_fixture():
    radius = np.linspace(0.0, 2.0, 7)
    height = np.linspace(-2.0, 2.0, 7)
    radial, vertical = np.meshgrid(radius, height, indexing="ij")
    coordinate = np.c_[radial.ravel(), vertical.ravel()]
    grid = Null2D.from_coordinates(coordinate, hex_stencil((7, 7)), maxsize=5)
    wall = Null1D(
        jnp.asarray(
            [[0.0, -2.0], [2.0, -2.0], [2.0, 2.0], [0.0, 2.0]],
            dtype=jnp.float64,
        )
    )
    return Topology(grid, wall), coordinate


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
def test_isolated_wall_candidate_loses_to_connected_trimmed_owner(compiled):
    """Owner admission validates only a candidate joined to governed material."""
    topology, coordinate = _topology_fixture()
    top = np.array([1.0, 2.0, 10.0, 0.0])
    interior = np.array([1.0, 0.0, 5.0, 0.0])
    vmap_o = jnp.asarray(
        np.vstack((top, interior, np.full((3, 4), np.nan))), dtype=jnp.float64
    )
    vmap_x = jnp.asarray(
        np.vstack((np.array([1.0, -1.0, 0.0, 0.0]), np.full((4, 4), np.nan))),
        dtype=jnp.float64,
    )
    data_w = jnp.asarray([2.0, 0.0, -1.0, 0.0], dtype=jnp.float64)
    central = (np.abs(coordinate[:, 0] - 1.0) <= 0.7) & (
        np.abs(coordinate[:, 1]) <= 0.7
    )
    flux = np.where(central, 1.0, -1.0)
    top_owner = int(np.argmin(np.sum((coordinate - top[:2]) ** 2, axis=1)))
    interior_owner = int(np.argmin(np.sum((coordinate - interior[:2]) ** 2, axis=1)))
    flux[top_owner] = top[2]
    flux[interior_owner] = interior[2]
    material = central.copy()
    material[top_owner] = False
    material[interior_owner] = False

    qualify = topology.qualified_o_candidates
    if compiled:
        qualify = jax.jit(qualify)
        qualified = qualify(
            vmap_o, vmap_x, data_w, 1, jnp.asarray(flux), jnp.asarray(material)
        )
    else:
        with jax.disable_jit():
            qualified = qualify(
                vmap_o, vmap_x, data_w, 1, jnp.asarray(flux), jnp.asarray(material)
            )
    np.testing.assert_array_equal(
        np.asarray(qualified), [False, True, False, False, False]
    )

    selected = topology.o_point_data(vmap_o, 1, qualified)
    np.testing.assert_array_equal(np.asarray(selected), interior)
    seed, admitted = axis_cell_seed(
        jnp.asarray(coordinate), selected[:2], jnp.asarray(material)
    )
    assert int(np.sum(seed)) == 1
    assert bool(np.asarray(seed)[interior_owner])
    assert int(np.sum(np.asarray(admitted) != material)) == 1


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
def test_committed_wall_notch_cannot_be_selected_without_qualification(compiled):
    """The committed top-notch axis produces a named fail-closed result."""
    topology, _coordinate = _topology_fixture()
    with np.load(OPERANDS, allow_pickle=False) as stored:
        wall_notch = stored["row_04_selected_o"][0]
    np.testing.assert_allclose(
        wall_notch, [0.36626597682950957, -1.932135898814253], atol=5.0e-9
    )
    vmap_o = jnp.asarray(
        np.vstack((np.r_[wall_notch, 10.0, 0.0], np.full((4, 4), np.nan))),
        dtype=jnp.float64,
    )
    qualified = jnp.zeros(5, dtype=bool)

    with pytest.raises(
        NoQualifiedAxisError, match="no qualified magnetic-axis candidate"
    ) as raised:
        if not compiled:
            with jax.disable_jit():
                topology.o_point_data(vmap_o, 1, qualified)
        else:
            selection = jax.jit(topology.o_point_qualification)(vmap_o, 1, qualified)
            require_qualified_axis(selection.admitted)
    assert not isinstance(raised.value, jax.errors.JaxRuntimeError)


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
def test_disqualified_mid_iterate_is_backtracked_without_raising(compiled):
    """An empty trial component rejects that factor without aborting Newton."""
    topology, coordinate = _topology_fixture()
    candidate = np.array([1.0, 0.0, 5.0, 0.0])
    vmap_o = jnp.asarray(
        np.vstack((candidate, np.full((4, 4), np.nan))), dtype=jnp.float64
    )
    vmap_x = jnp.asarray(
        np.vstack((np.array([1.0, -1.0, 0.0, 0.0]), np.full((4, 4), np.nan))),
        dtype=jnp.float64,
    )
    data_w = jnp.asarray([2.0, 0.0, -1.0, 0.0], dtype=jnp.float64)
    central = (np.abs(coordinate[:, 0] - 1.0) <= 0.7) & (
        np.abs(coordinate[:, 1]) <= 0.7
    )
    owner = int(np.argmin(np.sum((coordinate - candidate[:2]) ** 2, axis=1)))
    material = central.copy()
    material[owner] = False
    connected_flux = np.where(central, 1.0, -1.0)
    connected_flux[owner] = candidate[2]
    empty_flux = np.full(coordinate.shape[0], -1.0)
    empty_flux[owner] = candidate[2]

    empty_qualified = topology.qualified_o_candidates(
        vmap_o,
        vmap_x,
        data_w,
        1,
        jnp.asarray(empty_flux),
        jnp.asarray(material),
    )
    assert not np.any(np.asarray(empty_qualified))
    finite_candidates = np.asarray(vmap_o)[np.isfinite(np.asarray(vmap_o)[:, 0])]
    for data_o in finite_candidates:
        boundary = topology.boundary(jnp.asarray(data_o), vmap_x, data_w, 1)
        assert np.all(np.isfinite(np.asarray(boundary)[:3]))

    def trial_qualification(state):
        flux = jnp.where(state[0] > 1.5, empty_flux, connected_flux)
        qualified = topology.qualified_o_candidates(
            vmap_o, vmap_x, data_w, 1, flux, jnp.asarray(material)
        )
        return topology.o_point_qualification(vmap_o, 1, qualified)

    def solve():
        return kink_aware_newton_krylov(
            lambda state: jnp.full_like(state, 2.0),
            jnp.zeros(1, dtype=jnp.float64),
            strategy="nonmonotone",
            newton_steps=2,
            gmres_iterations=1,
            warmup=0,
            admissibility_fn=trial_qualification,
        )

    def solve_with_terminal_qualification():
        result = solve()
        terminal = topology.o_point_qualification(vmap_o, 1, empty_qualified)
        return result, terminal

    if compiled:
        result, terminal = jax.jit(solve_with_terminal_qualification)()
    else:
        result, terminal = solve_with_terminal_qualification()
    np.testing.assert_allclose(np.asarray(result.state), [1.5])
    np.testing.assert_allclose(np.asarray(result.accepted_factors), [0.5, 0.5])
    assert np.all(np.asarray(result.candidate_admissibility[:, 1:]))
    assert not np.any(np.asarray(result.candidate_admissibility[:, 0]))

    with pytest.raises(
        NoQualifiedAxisError, match="no qualified magnetic-axis candidate"
    ) as raised:
        require_qualified_axis(terminal.admitted)
    assert not isinstance(raised.value, jax.errors.JaxRuntimeError)
