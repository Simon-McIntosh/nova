"""Private wall flags follow node flux and admitted saddle height limits."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks.topology_parity_harness import (
    _production_operator,
    bilinear_node_flux,
    exit_code,
    finalize_receipt,
)
from nova.equilibrium.domain import DomainMasks, PlasmaDomain
from nova.equilibrium.topology import private_wall_node_read
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


def test_narrow_private_leg_uses_wall_node_flux(monkeypatch):
    operator, masks, physical, state = _narrow_leg(monkeypatch)
    nearest = np.asarray(masks.private_flux[operator._wall_carrier_index])
    expected = (
        np.asarray(physical[operator.grid.node_number :]) <= float(state.x_point_flux)
    ) & (np.asarray(operator.wall.coordinate[:, 1]) < float(state.x_point[1]))
    assert np.count_nonzero(nearest != expected) == 2
    actual = operator._carrier_shadow_read(physical, masks)["private_wall_node_mask"]
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("label, flux", [(0, 0.5), (3, 2.0)])
def test_contained_wall_nodes_use_flux_instead_of_cell_label(monkeypatch, label, flux):
    operator, masks, physical, state = _narrow_leg(monkeypatch)
    masks = masks._replace(label=jnp.full_like(masks.label, label))
    physical = physical.at[operator.grid.node_number :].set(flux)
    wall_flux = np.asarray(physical[operator.grid.node_number :])
    height_band = np.asarray(operator.wall.coordinate[:, 1]) < state.x_point[1]
    expected = (wall_flux <= float(state.x_point_flux)) & height_band
    actual = operator._carrier_shadow_read(physical, masks, state)
    np.testing.assert_array_equal(actual["private_wall_node_mask"], expected)


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
    np.testing.assert_array_equal(
        result["wall_node_height_band"], [True, True, False, True]
    )
    np.testing.assert_array_equal(
        result["private_wall_node_mask"], [True, False, False, False]
    )
    state.x_point = jnp.full(2, jnp.nan)
    state.x_point_flux = jnp.asarray(jnp.nan)
    absent = operator._carrier_shadow_read(physical, masks, state)
    assert not np.any(absent["private_wall_node_mask"])


def test_private_height_bands_exclude_axis_interval_and_require_a_saddle():
    configure_dtypes()
    height = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    saddles = jnp.array([[1.0, -1.0], [1.0, 1.0], [jnp.nan, jnp.nan]])
    reading = private_wall_node_read(jnp.full(5, 0.5), height, 0.0, 1.0, -1.0, saddles)
    np.testing.assert_array_equal(
        reading["private_wall_node_mask"], [True, False, False, False, True]
    )
    absent = private_wall_node_read(
        jnp.full(5, 0.5), height, 0.0, 1.0, -1.0, jnp.full_like(saddles, jnp.nan)
    )
    assert not np.any(absent["private_wall_node_mask"])


def test_bilinear_raster_reads_nodes_between_nonuniform_grid_samples():
    radius = np.array([1.0, 2.0, 4.0])
    height = np.array([-2.0, 0.0, 3.0])
    points = np.array([[1.5, -0.5], [3.0, 2.0], [4.0, 3.0]])

    def field(r, z):
        return 2.0 * r - 3.0 * z + r * z

    values = field(radius[:, None], height[None, :])
    np.testing.assert_array_equal(
        bilinear_node_flux(values, radius, height, points),
        field(points[:, 0], points[:, 1]),
    )
    with pytest.raises(ValueError, match="interpolation domain"):
        bilinear_node_flux(values, radius, height, [[0.0, 0.0]])


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
