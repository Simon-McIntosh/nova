"""Spline-consistent separatrix thresholding for active-set masks."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks import efit_forward_parity_slice as parity
from benchmarks.receipt_raster_check import _profile_and_seed
from nova.equilibrium.flux_surface_connectivity import (
    fit_tensor_spline,
    polish_census_stationary_points,
    polish_stationary_points,
)
from nova.equilibrium.topology import TopologyClass


def _cubic_interpolation_disagreement(size: int) -> float:
    radial = jnp.linspace(-1.5, 1.5, size)
    vertical = jnp.linspace(-1.2, 1.2, size)
    radial_grid, vertical_grid = jnp.meshgrid(radial, vertical)
    values = radial_grid**3 / 3.0 - radial_grid + vertical_grid**2 / 2.0
    surface = fit_tensor_spline(radial, vertical, values)
    seeds = jnp.asarray(((0.9, 0.03), (-0.9, -0.03)), dtype=values.dtype)
    polished = polish_stationary_points(
        surface,
        seeds,
        jnp.ones(2, dtype=bool),
        stationary_steps=32,
    )
    selected = jnp.column_stack(
        (
            polished["position_rz"],
            polished["value"],
            jnp.zeros(2, dtype=values.dtype),
        )
    )
    _axis, _saddle, receipt = polish_census_stationary_points(
        values,
        radial,
        vertical,
        selected[1, 2],
        jnp.asarray(-1.0, dtype=values.dtype),
        selected[0],
        selected[1],
        surface=surface,
    )
    return float(
        jnp.abs(receipt["selected_value"][1] - receipt["local_value_evidence"][1])
    )


def test_stationary_interpolation_deadband_contracts_with_grid_refinement():
    """Spline-to-local disagreement contracts with the grid-cell scale."""
    disagreement = np.asarray(
        [_cubic_interpolation_disagreement(size) for size in (9, 17, 33)]
    )

    assert np.all(disagreement > 0.0)
    np.testing.assert_allclose(
        disagreement[1:] / disagreement[:-1],
        0.125,
        rtol=0.03,
        atol=0.0,
    )


@pytest.mark.slow
def test_spline_boundary_deadband_preserves_the_first_active_set_trip():
    """A sub-cell value correction cannot trigger a private-flood cascade."""
    case, profile, target_current, _cache_receipt, _policy = _profile_and_seed()
    seed = jnp.asarray(case["state"])
    requested = TopologyClass.DIVERTED
    physical = seed[: profile.operator.physical_node_number]
    masks, state, _connected, admitted = profile.operator._fixed_design_read(
        physical, requested
    )
    initial = profile.operator._fixed_design_topology.read_qualification(
        physical,
        profile.operator.polarity,
        profile.operator.inside_material,
        requested,
    )
    initial_shadow = profile.operator.residual_shadow_mask(seed, requested)

    np.testing.assert_allclose(
        np.asarray(state.axis),
        [0.9052605679418458, 0.0034524725279145934],
        rtol=0.0,
        atol=2.0e-15,
    )
    np.testing.assert_allclose(
        np.asarray(state.x_point),
        [0.5862131601685471, -1.1966931466743165],
        rtol=0.0,
        atol=2.0e-15,
    )
    assert float(state.axis_flux) == pytest.approx(0.3423115841995916, abs=2.0e-15)
    assert float(state.x_point_flux) == pytest.approx(-0.12933198764991063, abs=2.0e-15)
    np.testing.assert_array_equal(
        np.asarray(initial.polish_receipt["spline_authored"]), True
    )
    assert bool(initial.polish_receipt["boundary_comparison_spline_authored"])
    assert float(
        initial.polish_receipt["boundary_interpolation_uncertainty"]
    ) == pytest.approx(1.1840979492272474e-4, abs=2.0e-15)
    assert int(np.asarray(initial.boundary_uncertain).sum()) == 1
    assert bool(admitted)
    assert int(np.asarray(masks.private_flux).sum()) == 23
    # Flux-owned wall flags from 635195df8 contribute 14 shadowed wall nodes.
    # Golden source revision: c199578f903e99c28f41bd7cbcfb0b80b58b2a5e.
    # Regenerated with: JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
    # PYTHONPATH=. /home/ITER/mcintos/Code/nova/.venv/bin/python -m pytest -q
    # -s -p no:cacheprovider -m 'slow or not slow' --timeout=900
    # tests/test_mask_threshold_consistency.py
    grid_count = profile.operator.grid.node_number
    physical_count = profile.operator.physical_node_number
    assert int(np.asarray(initial_shadow).sum()) == 37
    assert int(np.asarray(initial_shadow[:grid_count]).sum()) == 23
    assert int(np.asarray(initial_shadow[grid_count:physical_count]).sum()) == 14

    prescribed_current = jnp.asarray(profile.operator.prescribed_field.current)
    solved = profile.solve_branch(
        seed,
        requested,
        target_current=target_current,
        prescribed_current=prescribed_current,
        route="newton_krylov",
        tolerance=1.0e-3,
        warmup=parity.WARMUP_SWEEPS,
        newton_steps=parity.NEWTON_STEPS,
        gmres_iterations=parity.GMRES_ITERATIONS,
        active_set_steps=1,
    )
    fixed = solved.equilibrium.fixed_point
    print(
        "active-set evidence",
        np.asarray(fixed.active_set_mask_differences),
        float(fixed.residual),
        flush=True,
    )

    assert int(fixed.active_set_iterations) == 1
    np.testing.assert_array_equal(
        np.asarray(fixed.active_set_mask_differences), np.asarray([24])
    )
    assert float(fixed.residual) == pytest.approx(
        0.003492851288148042, rel=0.0, abs=2.0e-15
    )
