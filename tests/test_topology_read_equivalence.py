"""Equivalence pins for the state-threaded connectivity binding search."""

from __future__ import annotations

import numpy as np

from nova.utilities.importmanager import skip_import

with skip_import("jax"):
    import jax.numpy as jnp

    from nova.equilibrium.connectivity_boundary import (
        traced_boundary_read,
        traced_emit_boundary_read,
        traced_iteration_boundary_read,
        traced_smooth_boundary_read,
    )
    from nova.jax.config import configure_dtypes


def _fixture_fields():
    rg = np.linspace(0.2, 1.8, 45)
    zg = np.linspace(-1.2, 1.2, 57)
    rr, zz = np.meshgrid(rg, zg)
    inside = ((rr - 1.0) / 0.72) ** 2 + (zz / 1.05) ** 2 <= 1.0
    limited = np.exp(-(((rr - 1.0) ** 2 + zz**2) / 0.3**2))
    diverted = np.exp(-(((rr - 1.0) ** 2 + (zz - 0.25) ** 2) / 0.28**2))
    diverted += 0.9 * np.exp(-(((rr - 1.0) ** 2 + (zz + 0.75) ** 2) / 0.28**2))
    return rg, zg, inside, (limited, diverted)


def _read(psi, rg, zg, inside, previous_flood_level=np.nan):
    return traced_boundary_read(
        jnp.asarray(psi),
        jnp.asarray(rg),
        jnp.asarray(zg),
        jnp.asarray(inside),
        jnp.asarray(1.0),
        jnp.asarray(0.0),
        48,
        10,
        32,
        jnp.linspace(0.0, 2.0 * jnp.pi, 8, endpoint=False),
        previous_flood_level=jnp.asarray(previous_flood_level),
    )


def test_warm_bracket_reproduces_cold_binding_levels_exactly():
    configure_dtypes()
    rg, zg, inside, fields = _fixture_fields()
    cold_levels = []
    warm_levels = []

    for field in fields:
        prior = _read(field, rg, zg, inside)
        moved = field * (1.0 + 2.0e-4 * (np.asarray(zg)[:, None] + 0.3))
        cold = _read(moved, rg, zg, inside)
        warm = _read(moved, rg, zg, inside, prior["s_flood"])
        cold_levels.append(np.asarray(cold["s_flood"]))
        warm_levels.append(np.asarray(warm["s_flood"]))
        assert bool(warm["binding_search_warm"])

    np.testing.assert_array_equal(np.asarray(warm_levels), np.asarray(cold_levels))


def test_first_read_and_bracket_miss_use_the_cold_sweep():
    configure_dtypes()
    rg, zg, inside, fields = _fixture_fields()
    field = fields[0]
    cold = _read(field, rg, zg, inside)
    missed = _read(field, rg, zg, inside, previous_flood_level=0.02)

    assert not bool(cold["binding_search_warm"])
    assert int(cold["binding_search_evaluations"]) == 48
    assert not bool(missed["binding_search_warm"])
    assert int(missed["binding_search_evaluations"]) == 48 + 6
    np.testing.assert_array_equal(missed["s_flood"], cold["s_flood"])


def test_coarse_iteration_preserves_the_full_resolution_emit():
    configure_dtypes()
    rg, zg, inside, fields = _fixture_fields()
    field = fields[1]
    args = (
        jnp.asarray(field),
        jnp.asarray(rg),
        jnp.asarray(zg),
        jnp.asarray(inside),
        jnp.asarray(1.0),
        jnp.asarray(0.0),
        48,
        10,
        32,
    )
    full_iteration = traced_smooth_boundary_read(*args, temperature=jnp.asarray(1.0e-3))
    coarse_iteration = traced_iteration_boundary_read(
        *args,
        temperature=jnp.asarray(1.0e-3),
        previous_flood_level=full_iteration["s_flood"],
        resolution_stride=2,
    )
    emitted = traced_emit_boundary_read(*args, temperature=jnp.asarray(1.0e-3))
    reference_emit = traced_smooth_boundary_read(*args, temperature=jnp.asarray(1.0e-3))

    for key in ("psi_axis", "psi_bnd", "s_soft", "radii", "core_weight"):
        np.testing.assert_array_equal(emitted[key], reference_emit[key])

    boundary_span = abs(float(reference_emit["psi_out"] - reference_emit["psi_axis"]))
    iterate_boundary_difference = (
        abs(float(coarse_iteration["psi_bnd"] - full_iteration["psi_bnd"]))
        / boundary_span
    )
    iterate_core_difference = float(
        np.max(
            np.abs(
                np.asarray(coarse_iteration["core_weight"])
                - np.asarray(full_iteration["core_weight"])
            )
        )
    )
    assert np.isfinite(iterate_boundary_difference)
    assert np.isfinite(iterate_core_difference)
    assert iterate_boundary_difference > 0.0
    assert iterate_core_difference > 0.0
    assert coarse_iteration["core_weight"].shape == field.shape
