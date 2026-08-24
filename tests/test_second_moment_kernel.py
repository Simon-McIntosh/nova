"""Verification of fixed six-column source-cell flux coupling."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from nova.biot.greens import greens_psi, traced_filament_greens
from nova.biot.second_moment_kernel import (
    flux_density_columns,
    monopole_taylor_columns,
    taylor_displaced_flux,
)
from nova.jax.config import configure_dtypes

configure_dtypes()

RECEIPT = (
    Path(__file__).parents[1]
    / "docs/figures/coefficient-space-newton/second-order-kernel.json"
)


def _rectangle(centre, pitch):
    half = 0.5 * pitch
    return np.asarray(centre) + np.asarray(
        [[-half, -half], [half, -half], [half, half], [-half, half]]
    )


def _subdivide(triangles):
    first, second, third = np.moveaxis(triangles, 1, 0)
    first_second = 0.5 * (first + second)
    second_third = 0.5 * (second + third)
    third_first = 0.5 * (third + first)
    return np.concatenate(
        (
            np.stack((first, first_second, third_first), axis=1),
            np.stack((first_second, second, second_third), axis=1),
            np.stack((third_first, second_third, third), axis=1),
            np.stack((first_second, second_third, third_first), axis=1),
        )
    )


def _midpoint_columns(target_r, target_z, vertices, levels):
    centre = np.mean(vertices, axis=0)
    triangles = np.stack(
        (
            np.broadcast_to(vertices[0], (len(vertices) - 2, 2)),
            vertices[1:-1],
            vertices[2:],
        ),
        axis=1,
    )
    for _ in range(levels):
        triangles = _subdivide(triangles)
    point = triangles.mean(axis=1)
    first = triangles[:, 1] - triangles[:, 0]
    second = triangles[:, 2] - triangles[:, 0]
    weight = 0.5 * np.abs(first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0])
    local = point - centre
    basis = np.column_stack(
        (
            np.ones(len(point)),
            local[:, 0],
            local[:, 1],
            local[:, 0] ** 2,
            local[:, 0] * local[:, 1],
            local[:, 1] ** 2,
        )
    )
    kernel = greens_psi(
        np.asarray(target_r)[:, None],
        np.asarray(target_z)[:, None],
        point[None, :, 0],
        point[None, :, 1],
    )
    return np.einsum("tq,qc,q->tc", kernel, basis, weight) / weight.sum()


def test_quadratic_weights_match_richardson_extrapolation():
    vertices = _rectangle((3.0, 0.0), 0.16)
    target_r = np.asarray([2.72, 3.31, 4.2, 6.0])
    target_z = np.asarray([0.13, -0.21, 0.7, -1.4])
    coarse = _midpoint_columns(target_r, target_z, vertices, 6)
    fine = _midpoint_columns(target_r, target_z, vertices, 7)
    richardson = (4.0 * fine - coarse) / 3.0
    computed = flux_density_columns(np, target_r, target_z, vertices, order=10)
    np.testing.assert_allclose(
        computed[:, 3:], richardson[:, 3:], rtol=2.0e-8, atol=2.0e-20
    )


def test_every_cell_has_fixed_six_column_shape_under_jit_and_vmap():
    centres = np.asarray([[2.85, -0.12], [3.0, 0.0], [3.15, 0.12]])
    cells = np.stack([_rectangle(centre, 0.12) for centre in centres])
    target_r = np.asarray([2.65, 3.35, 4.1, 5.0])
    target_z = np.asarray([0.22, -0.28, 0.6, -0.9])
    host = np.stack(
        [flux_density_columns(np, target_r, target_z, cell) for cell in cells]
    )

    batched = jax.jit(
        jax.vmap(
            lambda cell: flux_density_columns(
                jnp, jnp.asarray(target_r), jnp.asarray(target_z), cell
            )
        )
    )(jnp.asarray(cells))
    assert batched.shape == (len(cells), len(target_r), 6)
    np.testing.assert_allclose(np.asarray(batched), host, rtol=8.0e-12, atol=2.0e-20)


def test_centroid_shift_matches_direct_displaced_monopole_through_quadratic_order():
    source = jnp.asarray([3.0, 0.0])
    target_r = jnp.asarray([2.65, 3.35, 4.2, 5.5])
    target_z = jnp.asarray([0.24, -0.31, 0.8, -1.2])
    pitch = 0.12
    direction = jnp.asarray([0.8, -0.6])
    columns = jax.jit(monopole_taylor_columns)(target_r, target_z, source)
    residual = []
    for fraction in (0.005, 0.01, 0.02, 0.04, 0.08):
        displacement = pitch * fraction * direction
        direct = traced_filament_greens(
            jnp,
            target_r,
            target_z,
            source[0] + displacement[0],
            source[1] + displacement[1],
        )[0]
        approximation = taylor_displaced_flux(columns, displacement)
        residual.append(
            float(jnp.max(jnp.abs(direct - approximation)) / jnp.max(jnp.abs(direct)))
        )
    assert residual[-1] < 2.0e-5
    fitted_order = np.polyfit(
        np.log(np.asarray([0.005, 0.01, 0.02, 0.04, 0.08])),
        np.log(np.asarray(residual)),
        1,
    )[0]
    assert 2.8 < fitted_order < 3.2


def test_receipt_carries_required_quantitative_evidence():
    receipt = json.loads(RECEIPT.read_text())
    assert set(receipt["weighted_second_order_members"]) == {"RR", "RZ", "ZZ"}
    for result in receipt["weighted_second_order_members"].values():
        assert result["evaluation"] in {"closed_form", "quadrature"}
        assert result["richardson_relative_error"] < 2.0e-8
    assert receipt["centroid_shift_identity"]["maximum_relative_residual"] < 2.0e-5
    assert receipt["centroid_shift_identity"]["fitted_residual_order"] > 2.8
    assert receipt["fixed_shape"]["columns_per_cell"] == 6
    assert receipt["fixed_shape"]["jit_vmap_agreement"]
    assert receipt["cost"]["coupling_memory_ratio_six_over_three"] == 2.0
