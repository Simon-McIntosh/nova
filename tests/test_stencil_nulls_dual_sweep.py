"""Staggered critical-point orbits at rectangular partition boundaries."""

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium.stencil_nulls import (
    critical_point_candidates_batch,
    gradient_cell_degree,
)
from nova.equilibrium.flux_surface_connectivity import polish_stationary_points
from nova.linalg.tensor_spline import fit_tensor_spline


def _saddle(radius, height, centre_radius, centre_height, coefficients):
    radial_offset = radius[None, :] - centre_radius
    vertical_offset = height[:, None] - centre_height
    radial_square, vertical_square, cross = coefficients
    return (
        radial_square * radial_offset**2
        + vertical_square * vertical_offset**2
        + cross * radial_offset * vertical_offset
    )


def _census(fields, radius, height):
    return jax.device_get(
        critical_point_candidates_batch(
            fields,
            radius,
            height,
            np.ones(fields.shape[-2:], dtype=bool),
            k_slots=8,
            material_dilate=0,
            target_index=-1,
            noise_sigma=0.0,
            dual_sweep=True,
        )
    )


def test_single_orbit_family_has_zero_degree_at_partition_boundaries():
    coordinates = np.arange(-4.0, 5.0)
    edge = _saddle(coordinates, coordinates, 0.0, 0.25, (2.0, -1.0, 1.5))
    node = _saddle(coordinates, coordinates, 0.0, 0.0, (1.0, -1.3, 0.7))

    degree, _winding, *_ = gradient_cell_degree(
        jnp.asarray(np.stack((edge, node))), coordinates, coordinates
    )

    np.testing.assert_array_equal(np.asarray(degree), 0)


def test_staggered_orbits_recover_one_saddle_at_each_partition_boundary():
    coordinates = np.arange(-4.0, 5.0)
    expected = np.asarray([[0.0, 0.25], [0.0, 0.0]])
    fields = np.stack(
        [
            _saddle(coordinates, coordinates, *expected[0], (2.0, -1.0, 1.5)),
            _saddle(coordinates, coordinates, *expected[1], (1.0, -1.3, 0.7)),
        ]
    )

    result = _census(fields, coordinates, coordinates)

    np.testing.assert_array_equal(result["cluster_count"], 1)
    np.testing.assert_array_equal(result["dual_candidate_count"], 1)
    np.testing.assert_array_equal(result["resolved"][:, 0], True)
    np.testing.assert_array_equal(result["native_signed_index"][:, 0], -1)
    np.testing.assert_allclose(result["r"][:, 0], expected[:, 0], atol=2.0e-7)
    np.testing.assert_allclose(result["z"][:, 0], expected[:, 1], atol=2.0e-7)
    np.testing.assert_array_equal(np.sum(result["present"], axis=1), 1)


def test_generic_interior_saddle_keeps_the_primal_refinement_exact():
    coordinates = np.arange(-4.0, 5.0)
    field = _saddle(coordinates, coordinates, 0.25, 0.25, (1.0, -1.3, 0.7))

    single = jax.device_get(
        critical_point_candidates_batch(
            field[None],
            coordinates,
            coordinates,
            np.ones(field.shape, dtype=bool),
            k_slots=8,
            material_dilate=0,
            target_index=-1,
            noise_sigma=0.0,
            dual_sweep=False,
        )
    )
    eager = _census(field[None], coordinates, coordinates)
    compiled = jax.device_get(
        jax.jit(
            critical_point_candidates_batch,
            static_argnames=(
                "k_slots",
                "material_dilate",
                "target_index",
                "dual_sweep",
            ),
        )(
            jnp.asarray(field[None]),
            jnp.asarray(coordinates),
            jnp.asarray(coordinates),
            jnp.ones(field.shape, dtype=bool),
            k_slots=8,
            material_dilate=0,
            target_index=-1,
            noise_sigma=0.0,
            dual_sweep=True,
        )
    )

    for result in (eager, compiled):
        assert int(result["primal_candidate_count"][0]) == 1
        assert int(result["dual_candidate_count"][0]) == 1
        assert int(result["cluster_count"][0]) == 1
        assert int(np.sum(result["present"][0])) == 1
        assert int(result["native_signed_index"][0, 0]) == -1
        for key in ("r", "z", "psi"):
            np.testing.assert_array_equal(result[key][0, 0], single[key][0, 0])
        np.testing.assert_array_equal(result["orbit_family"][0, 0], 0)


def test_polishing_union_slots_never_perturbs_a_primal_candidate():
    """Every census family remains an independent stationary-polish lane."""
    coordinates = jnp.arange(-4.0, 5.0)
    field = _saddle(coordinates, coordinates, 0.25, 0.25, (1.0, -1.3, 0.7))
    census = _census(field[None], coordinates, coordinates)
    seeds = jnp.stack((census["r"][0], census["z"][0]), axis=-1)
    present = jnp.asarray(census["present"][0])
    spline = fit_tensor_spline(coordinates, coordinates, jnp.asarray(field))

    union = polish_stationary_points(spline, seeds, present)
    primal_slot = int(np.flatnonzero(np.asarray(census["orbit_family"][0]) == 0)[0])
    primal = polish_stationary_points(
        spline, seeds[primal_slot : primal_slot + 1], jnp.asarray([True])
    )

    np.testing.assert_array_equal(
        np.asarray(union["position_rz"][primal_slot]),
        np.asarray(primal["position_rz"][0]),
    )
