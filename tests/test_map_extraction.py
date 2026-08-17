"""Receipts-bearing reads of a prescribed Grad--Shafranov map."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.constants import mu_0
from scipy.sparse.linalg import spsolve

from nova.equilibrium import (
    apply_delta_star,
    extract_flux_functions,
    sample_chord_psi_norm,
    vacuum_region_receipt,
)
from nova.equilibrium.conservation import FluxLattice, delta_star
from nova.jax.config import configure_dtypes

PROFILE_RELATIVE_TOLERANCE = 2.0e-8
VACUUM_RELATIVE_TOLERANCE = 2.0e-10
CHORD_ABSOLUTE_TOLERANCE = 2.0e-14
P_PRIME = -2.2e5
FF_PRIME = -0.18


@dataclass(frozen=True)
class PrescribedEquilibrium:
    """One finite-difference equilibrium with a compact source region."""

    radius: np.ndarray
    height: np.ndarray
    flux: np.ndarray
    psi_norm: np.ndarray
    plasma: np.ndarray
    vacuum: np.ndarray
    axis_flux: float
    boundary_flux: float


def _operator_matrix(radius: np.ndarray, height: np.ndarray):
    """Return the structured Delta-star matrix with a fixed-flux boundary."""

    radial_count = radius.size
    vertical_count = height.size
    radial_step = radius[1] - radius[0]
    vertical_step = height[1] - height[0]
    matrix = sparse.lil_array(
        (radial_count * vertical_count, radial_count * vertical_count)
    )

    def flat(radial_index: int, vertical_index: int) -> int:
        return radial_index * vertical_count + vertical_index

    for radial_index, radial_value in enumerate(radius):
        for vertical_index in range(vertical_count):
            row = flat(radial_index, vertical_index)
            if radial_index in (0, radial_count - 1) or vertical_index in (
                0,
                vertical_count - 1,
            ):
                matrix[row, row] = 1.0
                continue
            matrix[row, flat(radial_index - 1, vertical_index)] = (
                1.0 / radial_step** 2 + 1.0 / (2.0 * radial_value * radial_step)
            )
            matrix[row, flat(radial_index + 1, vertical_index)] = (
                1.0 / radial_step** 2 - 1.0 / (2.0 * radial_value * radial_step)
            )
            matrix[row, flat(radial_index, vertical_index - 1)] = 1.0 / vertical_step**2
            matrix[row, flat(radial_index, vertical_index + 1)] = 1.0 / vertical_step**2
            matrix[row, row] = -2.0 / radial_step**2 - 2.0 / vertical_step**2
    return matrix.tocsr()


def _prescribed_equilibrium() -> PrescribedEquilibrium:
    """Solve a compact-source equilibrium under Nova's total-flux convention."""

    configure_dtypes()
    radius = np.linspace(0.62, 1.38, 65)
    height = np.linspace(-0.38, 0.38, 65)
    radius_map, height_map = np.meshgrid(radius, height, indexing="ij")
    elliptical_radius = np.sqrt(
        ((radius_map - 1.0) / 0.31) ** 2 + (height_map / 0.29) ** 2
    )
    plasma = elliptical_radius <= 0.62
    source = 4.0 * np.pi**2 * (mu_0 * radius_map**2 * P_PRIME + FF_PRIME)
    right_hand_side = np.where(plasma, source, 0.0)
    right_hand_side[[0, -1], :] = 0.0
    right_hand_side[:, [0, -1]] = 0.0
    flux = spsolve(
        _operator_matrix(radius, height), right_hand_side.reshape(-1)
    ).reshape(radius_map.shape)
    axis_flux = float(np.max(flux))
    boundary_flux = 0.0
    psi_norm = (flux - axis_flux) / (boundary_flux - axis_flux)
    vacuum = elliptical_radius >= 0.74
    return PrescribedEquilibrium(
        radius=radius,
        height=height,
        flux=flux,
        psi_norm=psi_norm,
        plasma=plasma,
        vacuum=vacuum,
        axis_flux=axis_flux,
        boundary_flux=boundary_flux,
    )


def test_prescribed_profiles_round_trip_and_vacuum_receipt_passes():
    equilibrium = _prescribed_equilibrium()
    surfaces = np.linspace(0.05, 0.55, 11)
    extracted = extract_flux_functions(
        equilibrium.radius,
        equilibrium.height,
        equilibrium.flux,
        equilibrium.psi_norm,
        surfaces=surfaces,
        plasma_mask=equilibrium.plasma,
    )
    resolved = extracted.reliable
    assert np.count_nonzero(resolved) >= 8
    p_error = np.max(np.abs(extracted.p_prime[resolved] / P_PRIME - 1.0))
    ff_error = np.max(np.abs(extracted.ff_prime[resolved] / FF_PRIME - 1.0))
    assert p_error < PROFILE_RELATIVE_TOLERANCE
    assert ff_error < PROFILE_RELATIVE_TOLERANCE
    assert extracted.current.finite

    vacuum = vacuum_region_receipt(
        extracted.current,
        equilibrium.vacuum,
        relative_tolerance=VACUUM_RELATIVE_TOLERANCE,
    )
    assert vacuum.passed
    assert vacuum.relative_rms < VACUUM_RELATIVE_TOLERANCE


def test_axis_and_stationary_label_regions_inflate_uncertainty():
    equilibrium = _prescribed_equilibrium()
    surfaces = np.asarray([0.0, 0.08, 0.2, 0.4, 0.6, 0.8])
    baseline = extract_flux_functions(
        equilibrium.radius,
        equilibrium.height,
        equilibrium.flux,
        equilibrium.psi_norm,
        surfaces=surfaces,
        plasma_mask=equilibrium.plasma,
        min_samples=3,
    )
    assert baseline.uncertainty_inflation[0] > baseline.uncertainty_inflation[3]

    stationary_labels = equilibrium.psi_norm.copy()
    stationary_labels[47:52, 29:36] = 0.6
    stationary = extract_flux_functions(
        equilibrium.radius,
        equilibrium.height,
        equilibrium.flux,
        stationary_labels,
        surfaces=surfaces,
        plasma_mask=equilibrium.plasma,
        min_samples=3,
    )
    assert stationary.minimum_gradient[4] == 0.0
    assert stationary.uncertainty_inflation[4] > 100.0
    assert not stationary.reliable[4]


def test_synthetic_chords_return_normalised_flux():
    equilibrium = _prescribed_equilibrium()
    radial_indices = np.arange(8, 57, 4)
    vertical_indices = np.arange(8, 57, 4)
    middle_radius = equilibrium.radius.size // 2
    middle_height = equilibrium.height.size // 2
    coordinates = np.stack(
        [
            np.c_[
                equilibrium.radius[radial_indices],
                np.full(radial_indices.size, equilibrium.height[middle_height]),
            ],
            np.c_[
                np.full(vertical_indices.size, equilibrium.radius[middle_radius]),
                equilibrium.height[vertical_indices],
            ],
        ]
    )
    expected = np.stack(
        [
            equilibrium.psi_norm[radial_indices, middle_height],
            equilibrium.psi_norm[middle_radius, vertical_indices],
        ]
    )
    sampled = sample_chord_psi_norm(
        equilibrium.radius,
        equilibrium.height,
        equilibrium.flux,
        coordinates,
        axis_flux=equilibrium.axis_flux,
        boundary_flux=equilibrium.boundary_flux,
    )
    assert sampled.finite
    assert np.all(sampled.inside_grid)
    assert np.allclose(sampled.psi_norm, expected, atol=CHORD_ABSOLUTE_TOLERANCE)


def test_delta_star_receipt_marks_the_centred_stencil_only():
    equilibrium = _prescribed_equilibrium()
    receipt = apply_delta_star(
        equilibrium.radius, equilibrium.height, equilibrium.flux.reshape(-1)
    )
    mesh = FluxLattice(equilibrium.radius, equilibrium.height)
    expected_operator = np.asarray(
        delta_star(mesh, equilibrium.flux.reshape(-1))
    ).reshape(equilibrium.flux.shape)
    assert np.array_equal(receipt.valid.reshape(-1), np.asarray(mesh.interior()))
    assert np.allclose(
        receipt.delta_star_flux[receipt.valid],
        expected_operator[receipt.valid],
        rtol=0.0,
        atol=0.0,
    )
    assert receipt.valid[2:-2, 2:-2].all()
    assert not receipt.valid[:2, :].any()
    assert not receipt.valid[-2:, :].any()
    assert not receipt.valid[:, :2].any()
    assert not receipt.valid[:, -2:].any()
    expected_current = np.broadcast_to(
        -2.0
        * np.pi
        * (
            equilibrium.radius[:, None] * P_PRIME
            + FF_PRIME / (mu_0 * equilibrium.radius[:, None])
        ),
        receipt.toroidal_current_density.shape,
    )
    core = equilibrium.plasma & receipt.valid
    relative_error = np.max(
        np.abs(receipt.toroidal_current_density[core] / expected_current[core] - 1.0)
    )
    assert relative_error < PROFILE_RELATIVE_TOLERANCE
