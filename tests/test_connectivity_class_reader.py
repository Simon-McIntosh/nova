"""Regression coverage for supplied topology-classification operands."""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.receipt_raster_check import _profile_and_seed
from nova.utilities.importmanager import skip_import

with skip_import("jax"):
    import jax.numpy as jnp

    from nova.equilibrium import connectivity_boundary as cb
    from nova.equilibrium.topology import TopologyClass
    from nova.jax.config import configure_dtypes


def test_shadowed_supplied_wall_vetoes_valid_exact_node():
    """The supplied MAST wall stays authoritative when rediscovery succeeds."""
    configure_dtypes()
    case, profile, _target_current, _carrier, _policy = _profile_and_seed()
    operator = profile.operator
    state = jnp.asarray(case["state"])
    physical = state[: operator.physical_node_number]
    requested = jnp.asarray(int(TopologyClass.DIVERTED), dtype=jnp.int8)
    _masks, topology, _connected, axis_admitted = operator._fixed_design_read(
        physical, requested
    )

    radial, vertical, shape = operator.connectivity_grid_axes()
    field, wall_flux = operator.topology.split_flux_map(physical)
    _vmap_o, classification_x = operator._fixed_design_topology.grid(field)
    radial_count, vertical_count = shape
    field = field.reshape((radial_count, vertical_count)).T
    _axis_seed, inside = operator.connectivity_axis_seed(topology.axis)
    inside = inside.reshape((radial_count, vertical_count)).T
    wall_r = operator.wall.coordinate[:, 0]
    wall_z = operator.wall.coordinate[:, 1]
    surface = cb.fit_tensor_spline(radial, vertical, field)
    classification_wall = jnp.concatenate(
        (topology.wall_point, topology.wall_point_flux[None])
    )
    wall_surface_flux = surface(classification_wall[0], classification_wall[1])
    flux_scale = jnp.maximum(jnp.abs(wall_surface_flux), jnp.ptp(field))
    classification_wall = classification_wall.at[2].set(
        wall_surface_flux + 0.1 * flux_scale
    )

    result = cb._read_ingredients(
        field,
        radial,
        vertical,
        inside,
        topology.axis[0],
        topology.axis[1],
        96,
        18,
        wall_r,
        wall_z,
        wall_flux,
        jnp.asarray(jnp.nan),
        True,
        classification_x,
        classification_wall,
        surface,
    )

    assert bool(axis_admitted)
    assert bool(result["class_wall"]["valid"])
    assert int(result["class_wall"]["node_index"]) == 20
    np.testing.assert_allclose(
        [result["class_wall"]["node_r"], result["class_wall"]["node_z"]],
        [0.2800000011920929, -1.683500051498413],
        rtol=0.0,
        atol=2.0e-12,
    )
    assert bool(result["class_wall_shadowed"])
    rediscovered_wall_level = (
        result["class_wall"]["psi"] - result["psi_axis"]
    ) / result["span_safe"]
    assert float(rediscovered_wall_level) == pytest.approx(0.3334, abs=5.0e-3)
    assert float(result["class_u_x"]) == pytest.approx(0.4383, abs=5.0e-4)
    assert np.isposinf(float(result["u_wall_c"]))
    assert np.isposinf(float(result["class_u_wall"]))
    assert bool(result["class_x_valid"])
    achieved = jnp.where(
        result["class_x_valid"] & (result["class_u_x"] <= result["class_u_wall"]),
        int(TopologyClass.DIVERTED),
        int(TopologyClass.LIMITED),
    )
    topology_consistent = achieved == requested
    assert int(achieved) == int(TopologyClass.DIVERTED)
    assert bool(topology_consistent)
