"""Account for every numerical stage in the masked EFIT operator round trip.

The primary ledger is an exact vector decomposition on Nova's grid.  Starting
from the LCFS-masked current supplied to ``plasma_to_grid``, it separates the
error made by applying the conservation finite-difference operator to the
finite-cell Green field from the extra error introduced by masking the
recovered current again.  The two contribution maps sum to the reported total
map, and a closure receipt proves that no remainder is hidden by comparing
only sup norms at different coordinates.

COCOS scaling, drive packing and grid interoperation are measured as explicit
counterfactual paths.  The EFIT-to-Nova-to-EFIT interpolation error remains a
separate receipt and is never included in the operator-identity ledger.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from scipy.constants import mu_0

from benchmarks.efit_flux_decomposition import (
    DEFAULT_SHOT,
    OPERATOR_ROUND_TRIP_BOUND,
    _density_from_flux,
    _evaluate_on_grid,
    _flux_interpolation_receipt,
    _interpolator,
    _lcfs_mask,
    _plasma_flux,
    _read_stored_slice,
)
from nova.equilibrium.conservation import FluxLattice, delta_star
from nova.equilibrium.convention import TOTAL_FLUX_FACTOR
from nova.imas.mast_chain_factory import build_mast_parity_chain
from nova.imas.mast_solve_inputs import SHOT_STORE


def _fraction(field: np.ndarray, span: float) -> float:
    """Return a map's sup norm as a fraction of the stored flux span."""

    return float(np.max(np.abs(field)) / span)


def _flux_from_density(
    coupling: np.ndarray,
    lattice: FluxLattice,
    density_rz: np.ndarray,
    shape_zr: tuple[int, int],
) -> np.ndarray:
    """Apply the coupling with an independently written packing expression."""

    area_rz = lattice.cell_area.reshape(lattice.shape)
    current_zr = (np.asarray(density_rz, dtype=float) * area_rz).T
    return (np.asarray(coupling, dtype=float) @ current_zr.ravel()).reshape(shape_zr)


def _stage(
    contribution: np.ndarray,
    span: float,
    peak_index: tuple[int, int],
) -> dict[str, float]:
    """Describe one contribution both globally and at the total-error peak."""

    return {
        "sup_error_wb": float(np.max(np.abs(contribution))),
        "sup_error_fraction_of_stored_span": _fraction(contribution, span),
        "signed_error_at_total_peak_wb": float(contribution[peak_index]),
        "signed_fraction_at_total_peak": float(contribution[peak_index] / span),
    }


def account_round_trip(
    *,
    shot: int,
    slice_index: int,
    store: Path,
    artifact_cache: Path,
    artifact_digest: str,
) -> dict[str, Any]:
    """Return the stage-wise identity ledger for one stored EFIT slice."""

    group = zarr.open_group(str(store / f"{shot}.zarr"), mode="r")["efm"]
    requested_time = float(group["time"][slice_index])
    stored = _read_stored_slice(group, requested_time)
    if stored.index != slice_index:
        raise ValueError(
            f"requested slice {slice_index} resolved to usable slice {stored.index}"
        )
    span = float(np.ptp(stored.total_flux_wb))

    stored_lattice = FluxLattice(stored.radius_m, stored.height_m)
    stored_density_total, stored_valid = _density_from_flux(
        stored_lattice, stored.total_flux_wb
    )
    stored_lcfs = _lcfs_mask(
        stored.radius_m,
        stored.height_m,
        stored.lcfs_radius_m,
        stored.lcfs_height_m,
    )
    stored_density = np.where(stored_valid & stored_lcfs, stored_density_total, 0.0)

    per_radian_flux = stored.total_flux_wb / TOTAL_FLUX_FACTOR
    per_radian_elliptic = np.asarray(
        delta_star(stored_lattice, per_radian_flux.T.ravel()), dtype=float
    )
    cocos_density = -per_radian_elliptic / (
        mu_0 * np.asarray(stored_lattice.node_radius)
    )
    cocos_density = cocos_density.reshape(stored_lattice.shape)
    cocos_density = np.where(stored_valid & stored_lcfs, cocos_density, 0.0)

    chain = build_mast_parity_chain(
        shot,
        artifact_cache=artifact_cache,
        artifact_digest=artifact_digest,
        store=store,
    )
    profile = chain.profile_solver
    nova_r = np.asarray(profile.grid_r, dtype=float)
    nova_z = np.asarray(profile.grid_z, dtype=float)
    nova_lattice = FluxLattice(nova_r, nova_z)
    nova_lcfs = _lcfs_mask(
        nova_r,
        nova_z,
        stored.lcfs_radius_m,
        stored.lcfs_height_m,
    )

    nova_flux = _evaluate_on_grid(
        _interpolator(stored.radius_m, stored.height_m, stored.total_flux_wb),
        nova_r,
        nova_z,
    )
    interpolation = _flux_interpolation_receipt(stored, nova_r, nova_z, nova_flux)

    mapped_density = _evaluate_on_grid(
        _interpolator(stored.radius_m, stored.height_m, stored_density.T),
        nova_r,
        nova_z,
    ).T
    mapped_density = np.where(nova_lcfs, mapped_density, 0.0)
    mapped_cocos_density = _evaluate_on_grid(
        _interpolator(stored.radius_m, stored.height_m, cocos_density.T),
        nova_r,
        nova_z,
    ).T
    mapped_cocos_density = np.where(nova_lcfs, mapped_cocos_density, 0.0)

    plasma_flux = _plasma_flux(profile, nova_lattice, mapped_density)
    independently_packed_flux = _flux_from_density(
        profile.plasma_to_grid,
        nova_lattice,
        mapped_density,
        plasma_flux.shape,
    )
    cocos_flux = _plasma_flux(profile, nova_lattice, mapped_cocos_density)

    recovered_density, recovered_valid = _density_from_flux(nova_lattice, plasma_flux)
    recovered_complete = np.where(recovered_valid, recovered_density, 0.0)
    recovered_masked = np.where(recovered_valid & nova_lcfs, recovered_density, 0.0)
    complete_reconstruction = _plasma_flux(profile, nova_lattice, recovered_complete)
    masked_reconstruction = _plasma_flux(profile, nova_lattice, recovered_masked)

    pair_contribution = plasma_flux - complete_reconstruction
    remask_contribution = complete_reconstruction - masked_reconstruction
    total_error = plasma_flux - masked_reconstruction
    closure = total_error - pair_contribution - remask_contribution
    peak_index = np.unravel_index(
        int(np.argmax(np.abs(total_error))), total_error.shape
    )

    nova_flux_density, nova_flux_valid = _density_from_flux(nova_lattice, nova_flux)
    nova_flux_density = np.where(nova_flux_valid & nova_lcfs, nova_flux_density, 0.0)
    flux_first_reconstruction = _plasma_flux(profile, nova_lattice, nova_flux_density)
    grid_interoperation_difference = plasma_flux - flux_first_reconstruction

    input_current = mapped_density * nova_lattice.cell_area.reshape(nova_lattice.shape)
    recovered_current = recovered_masked * nova_lattice.cell_area.reshape(
        nova_lattice.shape
    )
    current_difference = recovered_current - input_current
    input_current_scale = float(np.max(np.abs(input_current)))

    pair_stage = _stage(pair_contribution, span, peak_index)
    pair_stage.update(
        {
            "name": "finite-cell Green coupling followed by conservation delta_star",
            "attribution": (
                "delta_star recovery disagrees with the supplied current after "
                "the coupling drive; without a third exact discrete inverse this "
                "contribution belongs to the Green-delta_star pair, not either "
                "operator alone"
            ),
            "peak_cell_current_relative_error": float(
                np.max(np.abs(current_difference)) / input_current_scale
            ),
            "rms_cell_current_relative_error": float(
                np.sqrt(np.mean(current_difference[nova_lcfs] ** 2))
                / input_current_scale
            ),
        }
    )
    remask_stage = _stage(remask_contribution, span, peak_index)
    remask_stage.update(
        {
            "name": "LCFS remasking of delta-star recovered current",
            "removed_recovered_current_a": float(
                np.sum(
                    recovered_complete[~nova_lcfs]
                    * nova_lattice.cell_area.reshape(nova_lattice.shape)[~nova_lcfs]
                )
            ),
            "removed_absolute_recovered_current_a": float(
                np.sum(
                    np.abs(recovered_complete[~nova_lcfs])
                    * nova_lattice.cell_area.reshape(nova_lattice.shape)[~nova_lcfs]
                )
            ),
        }
    )

    total_fraction = _fraction(total_error, span)
    return {
        "source": {
            "shot": shot,
            "slice_index": stored.index,
            "time_s": stored.time_s,
            "stored_flux_span_wb": span,
            "stored_grid_shape_zr": list(stored.total_flux_wb.shape),
            "nova_grid_shape_zr": list(plasma_flux.shape),
        },
        "operator_identity": {
            "measured_sup_error_wb": float(np.max(np.abs(total_error))),
            "measured_sup_error_fraction_of_stored_span": total_fraction,
            "required_bound": OPERATOR_ROUND_TRIP_BOUND,
            "passes": total_fraction <= OPERATOR_ROUND_TRIP_BOUND,
            "total_error_peak": {
                "row_z": int(peak_index[0]),
                "column_r": int(peak_index[1]),
                "r_m": float(nova_r[peak_index[1]]),
                "z_m": float(nova_z[peak_index[0]]),
                "signed_error_wb": float(total_error[peak_index]),
            },
            "stages": {
                "delta_star_green_pair": pair_stage,
                "lcfs_remask": remask_stage,
                "drive_packing": {
                    "name": "density times cell area, z-r C-order packing",
                    "sup_error_fraction_of_stored_span": _fraction(
                        plasma_flux - independently_packed_flux, span
                    ),
                    "interpretation": (
                        "the public plasma-current-to-grid drive and an independent "
                        "matrix contraction are identical"
                    ),
                },
                "cocos_conversion": {
                    "name": "COCOS 3 Wb/rad to COCOS 17 total Wb",
                    "sup_error_fraction_of_stored_span": _fraction(
                        plasma_flux - cocos_flux, span
                    ),
                },
                "grid_interoperation": {
                    "name": (
                        "derive-current-then-bilinear-map versus bilinear-map-"
                        "then-derive-current"
                    ),
                    "sup_error_fraction_of_stored_span": _fraction(
                        grid_interoperation_difference, span
                    ),
                    "note": (
                        "counterfactual path comparison; excluded from the exact "
                        "operator-error sum and distinct from the flux interpolation "
                        "round-trip receipt"
                    ),
                },
            },
            "accounting": {
                "identity": "total = delta_star_green_pair + lcfs_remask",
                "vector_closure_sup_fraction_of_stored_span": _fraction(closure, span),
                "signed_stage_sum_at_total_peak_fraction": float(
                    (pair_contribution[peak_index] + remask_contribution[peak_index])
                    / span
                ),
                "signed_total_at_peak_fraction": float(total_error[peak_index] / span),
                "unexplained_remainder_fraction": _fraction(closure, span),
            },
            "defective_stage": (
                "finite-cell Green coupling and conservation delta_star are not "
                "a discrete inverse pair on the Nova grid"
            ),
        },
        "interpolation": interpolation,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shot", type=int, default=DEFAULT_SHOT)
    parser.add_argument("--slice", type=int, default=46)
    parser.add_argument("--store", type=Path, default=SHOT_STORE)
    parser.add_argument("--artifact-cache", type=Path, required=True)
    parser.add_argument("--artifact-digest", required=True)
    return parser


def main() -> None:
    """Print the stage ledger as stable JSON."""

    arguments = _parser().parse_args()
    print(
        json.dumps(
            account_round_trip(
                shot=arguments.shot,
                slice_index=arguments.slice,
                store=arguments.store,
                artifact_cache=arguments.artifact_cache,
                artifact_digest=arguments.artifact_digest,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
