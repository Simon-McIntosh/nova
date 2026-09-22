"""Wall-carrier ownership follows polygon containment across a narrow leg."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks.topology_parity_harness import (
    _production_operator,
    exit_code,
    finalize_receipt,
)
from nova.equilibrium.domain import DomainMasks, PlasmaDomain
from nova.equilibrium.forward_operator import _wall_node_cell_owners
from nova.jax.config import configure_dtypes


def _narrow_leg(monkeypatch, wall=None, wall_flux=None):
    configure_dtypes()
    assert jax.config.jax_enable_x64
    radius = np.array([0.6, 0.8, 1.0, 1.2, 1.4])
    height = np.arange(-2.0, 3.0)
    edges = np.array([0.5, 0.7, 0.9, 1.04, 1.3, 1.5])
    coordinate = np.c_[np.repeat(radius, len(height)), np.tile(height, len(radius))]
    polygons = tuple(
        np.array(
            [
                [edges[i], z - 0.5],
                [edges[i + 1], z - 0.5],
                [edges[i + 1], z + 0.5],
                [edges[i], z + 0.5],
            ]
        )
        for i in range(len(radius))
        for z in height
    )
    if wall is None:
        wall = np.array(
            [
                [1.06, -1.4],
                [1.07, -1.0],
                [1.4, -1.0],
                [1.4, 2.0],
                [0.5, 2.0],
                [0.5, -1.4],
            ]
        )
    operator = _production_operator(coordinate, polygons, wall, np.ones(25, bool))
    operator.polarity = -1
    private = (coordinate[:, 0] == 1.2) & (coordinate[:, 1] < 0)
    masks = DomainMasks(
        jnp.where(
            private, int(PlasmaDomain.PRIVATE_FLUX), int(PlasmaDomain.COMMON_SOL)
        ),
        jnp.ones(25),
    )
    axis = jnp.array([1.2, 1.0])
    saddle = jnp.array([[1.2, 0.0, 1.0, 0.0]])

    class Census:
        direct_sample_count = 0

        def __call__(self, _flux):
            return jnp.array([[1.2, 1.0, 0.0, -1.0]]), saddle

    operator._fixed_design_topology = SimpleNamespace(
        grid=Census(), contained_x_candidates=lambda rows: jnp.isfinite(rows[:, 2])
    )
    state = SimpleNamespace(
        axis=axis,
        axis_flux=jnp.array(0.0),
        x_point=saddle[0, :2],
        x_point_flux=saddle[0, 2],
    )
    monkeypatch.setattr(
        operator, "_fixed_design_read", lambda *_args: (masks, state, None, True)
    )
    if wall_flux is None:
        wall_flux = jnp.array([0.5, 0.6, 2.0, 2.0, 2.0, 2.0])
    physical = jnp.r_[jnp.zeros(25), wall_flux]
    return operator, masks, physical, state


def test_narrow_private_leg_uses_wall_node_containing_cell(monkeypatch):
    operator, masks, physical, _state = _narrow_leg(monkeypatch)
    nearest = np.asarray(masks.private_flux[operator._wall_carrier_index])
    expected = np.array([True, True, False, False, False, False])
    assert np.count_nonzero(nearest != expected) == 2
    actual = operator._carrier_shadow_read(physical, masks)["private_wall_node_mask"]
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("polarity", [-1, 1])
def test_uncontained_nodes_use_own_flux_and_saddle_height(monkeypatch, polarity):
    wall = np.array([[1.2, -3.0], [1.21, -3.0], [1.2, 3.0], [1.6, -3.0]])
    operator, masks, physical, state = _narrow_leg(
        monkeypatch, wall, jnp.array([0.5, 2.0, 0.5, jnp.nan])
    )
    if polarity > 0:
        operator.polarity = polarity
        physical = -physical
        state.x_point_flux = -state.x_point_flux
    read = jax.jit(lambda flux: operator._carrier_shadow_read(flux, masks, state))
    result = read(physical)
    assert int(result["wall_node_fallback_count"]) == 4
    np.testing.assert_array_equal(result["wall_node_fallback_mask"], True)
    np.testing.assert_array_equal(
        result["private_wall_node_mask"], [True, False, False, False]
    )
    state.x_point = jnp.full(2, jnp.nan)
    state.x_point_flux = jnp.asarray(jnp.nan)
    absent = operator._carrier_shadow_read(physical, masks, state)
    assert not np.any(absent["private_wall_node_mask"])


def test_polygon_ownership_includes_edges_and_marks_uncovered_nodes():
    polygons = (
        np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
        np.array([[1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0]]),
    )
    points = np.array([[0.5, 0.5], [1.0, 0.5], [1.5, 0.5], [3.0, 0.5]])
    np.testing.assert_array_equal(
        _wall_node_cell_owners(points, polygons), [0, 0, 1, -1]
    )
    np.testing.assert_array_equal(_wall_node_cell_owners(points, ()), -1)


@pytest.mark.parametrize("marginal", [False, True])
def test_receipt_refuses_a_wall_node_disagreement_with_identical_cells(marginal):
    row = {
        "identity": "synthetic narrow leg",
        "replayable": True,
        "replay_completed": True,
        "marginal_solver_basin": marginal,
        "marginal_flag_source": "synthetic",
        "disposition": "measured",
        "compared_cell_count": 25,
        "differing_cell_count": 0,
        "differing_cells": [],
        "selected_primaries": {"axis": {"matches": True}},
        "classification": {"finding": False},
        "wall_node_census": {"node_count": 6, "differing_node_count": 0},
    }
    receipt = {"schema": "nova.topology-cell-parity", "rows": [row]}
    assert finalize_receipt(receipt, pending=False)["passes"]
    row["wall_node_census"]["differing_node_count"] = 2
    finalize_receipt(receipt, pending=False)
    assert receipt["validation_errors"] == [
        "synthetic narrow leg: wall-node private flags differ"
    ]
    assert not receipt["passes"]
    assert exit_code(receipt) == 1
