"""Explain unresolved X-point fit states on the frozen EFIT map cohort.

The production boundary read accepts an unresolved fitted position as geometric
evidence while preserving its certification state.  This study reports the
candidate-level certification path and repeats the emitted read with only the
candidate state promoted to resolved.  Exact comparisons therefore distinguish
diagnostic state from geometry, topology, and domain-label consequences.
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import zarr

from benchmarks.efit_topology_boundary_score import (
    FROZEN_SHOTS,
    _BoundaryGrid,
    _live_flux_map,
    _slice_candidates,
    _valid_points,
)
from nova.equilibrium import connectivity_boundary as boundary_kernel
from nova.equilibrium.connectivity_boundary import LCFS_ANGLES
from nova.equilibrium.stencil_nulls import (
    CLASS_MARGIN_THRESHOLD,
    ROOT_SUPPORT_LIMIT,
    STATE_ABSENT,
    STATE_RESOLVED,
)
from nova.equilibrium.wall_mask import inside_polygon
from nova.imas.mast_chain_factory import BOUNDARY_RADIAL_SAMPLES
from nova.imas.mast_vacuum_cohort import SHOT_STORE

FLUX_LEVEL_COUNT = 48
FLUX_BISECTION_COUNT = 12
TARGET_SADDLE_INDEX = -1


def _finite(value: Any) -> float | int | bool | None:
    """Return a strict-JSON scalar."""

    item = np.asarray(value).item()
    if isinstance(item, bool | np.bool_):
        return bool(item)
    if isinstance(item, int | np.integer):
        return int(item)
    return float(item) if np.isfinite(item) else None


def _shot_input(store: Path, shot: int):
    """Read the deterministic production-map slice and its limiter grid."""

    group = zarr.open_group(str(store / f"{shot}.zarr"), mode="r")["efm"]
    candidates = _slice_candidates(group)
    current = np.asarray(group["plasma_current_c"][...], dtype=np.float64)
    if not np.any(candidates):
        raise ValueError(f"shot {shot} has no geometry-valid live flux-map row")
    slice_index = int(np.argmax(np.where(candidates, np.abs(current), -np.inf)))

    rg = np.asarray(group["gridr"][...], dtype=np.float64)
    zg = np.asarray(group["gridz"][...], dtype=np.float64)
    limiter_r = np.asarray(group["limiterr"][...], dtype=np.float64)
    limiter_z = np.asarray(group["limiterz"][...], dtype=np.float64)
    valid = _valid_points(np.column_stack([limiter_r, limiter_z]))
    limiter_r = limiter_r[valid]
    limiter_z = limiter_z[valid]
    radius, height = np.meshgrid(rg, zg)
    inside = np.asarray(
        inside_polygon(
            radius.ravel(),
            height.ravel(),
            limiter_r,
            limiter_z,
        ),
        dtype=bool,
    ).reshape(radius.shape)
    grid = _BoundaryGrid(
        rg=rg,
        zg=zg,
        inside_limiter=inside,
        limiter_r=limiter_r,
        limiter_z=limiter_z,
        wall_r=limiter_r,
        wall_z=limiter_z,
    )
    flux = _live_flux_map(group, slice_index, rg.size)
    axis_seed = (float(0.5 * (limiter_r.min() + limiter_r.max())), 0.0)
    return slice_index, float(group["time"][slice_index]), flux, grid, axis_seed


def _boundary_read(flux: np.ndarray, grid: _BoundaryGrid, axis):
    """Run the production topology-reader configuration."""

    return boundary_kernel.host_boundary_read(
        flux,
        grid,
        axis,
        n_levels=FLUX_LEVEL_COUNT,
        n_bisect=FLUX_BISECTION_COUNT,
        n_ray=BOUNDARY_RADIAL_SAMPLES,
        angles=LCFS_ANGLES,
    )


def _candidate_ingredients(flux: np.ndarray, grid: _BoundaryGrid, axis):
    """Return the kernel's binding-candidate evidence at the emitted read."""

    wall_r, wall_z = boundary_kernel._densify_wall(grid)
    return boundary_kernel._read_ingredients(
        jnp.asarray(flux),
        jnp.asarray(grid.rg),
        jnp.asarray(grid.zg),
        jnp.asarray(grid.inside_limiter),
        jnp.asarray(axis[0]),
        jnp.asarray(axis[1]),
        FLUX_LEVEL_COUNT,
        FLUX_BISECTION_COUNT,
        jnp.asarray(wall_r),
        jnp.asarray(wall_z),
        jnp.asarray([jnp.nan]),
        jnp.asarray(jnp.nan),
        True,
    )


def _candidate_record(evidence: dict[str, np.ndarray], index: int) -> dict[str, Any]:
    """Report one retained candidate's complete resolution path."""

    def get(key):
        return _finite(evidence[key][index])

    native_index = get("native_signed_index")
    fitted_index = get("fitted_signed_index")
    confidence_index = get("confidence_fitted_signed_index")
    root_support = get("root_support_cell")
    class_margin = get("class_margin")
    cluster_index = get("cluster_index_sum")
    cluster_certified = get("cluster_index_certified")
    return {
        "slot": index,
        "fit_state": get("state"),
        "position_rz_m": [get("r"), get("z")],
        "normalised_class_margin": class_margin,
        "required_class_margin": CLASS_MARGIN_THRESHOLD,
        "root_support_cell": root_support,
        "maximum_root_support_cell": ROOT_SUPPORT_LIMIT,
        "native_signed_index": native_index,
        "fitted_signed_index": fitted_index,
        "confidence_fitted_signed_index": confidence_index,
        "cluster_signed_index": cluster_index,
        "required_cluster_signed_index": TARGET_SADDLE_INDEX,
        "cluster_index_certified": cluster_certified,
        "cluster_containment_radius_cell": get("cluster_containment_radius"),
        "cluster_member_index_sum": get("member_index_sum"),
        "cluster_size": get("cluster_size"),
        "enclosing_loop_degree": get("enclosing_loop_degree"),
        "confirming_loop_degree": get("confirming_loop_degree"),
        "enclosing_loop_margin": get("enclosing_loop_margin"),
        "confirming_loop_margin": get("confirming_loop_margin"),
        "boundary_snr": get("boundary_snr"),
        "scale_support_count": get("scale_support"),
        "scale_drift_cell": get("scale_drift_cell"),
        "normalised_root_residual": get("normalized_residual"),
        "position_sigma_cell": get("position_sigma_cell"),
        "passes_fit_indices": bool(
            native_index == TARGET_SADDLE_INDEX
            and fitted_index == TARGET_SADDLE_INDEX
            and confidence_index == TARGET_SADDLE_INDEX
        ),
        "passes_class_margin": bool(class_margin >= CLASS_MARGIN_THRESHOLD),
        "passes_root_support": bool(root_support <= ROOT_SUPPORT_LIMIT),
        "passes_cluster_index": bool(
            cluster_certified and cluster_index == TARGET_SADDLE_INDEX
        ),
    }


def _emitted_fields(read, ray_centre: np.ndarray, flux, grid: _BoundaryGrid):
    """Materialise exactly the topology fields emitted by the MAST labeler."""

    ring = ray_centre + np.column_stack(
        [
            read.radii * np.cos(LCFS_ANGLES),
            read.radii * np.sin(LCFS_ANGLES),
        ]
    )
    span = read.psi_bnd - read.psi_axis
    normalised = (flux - read.psi_axis) / (
        span if abs(span) > np.finfo(float).tiny else np.nan
    )
    core = grid.inside_limiter & np.isfinite(normalised) & (normalised <= 1.0)
    common = grid.inside_limiter & ~core
    xset = np.asarray(read.xset, dtype=np.float64)
    finite_x = xset[np.all(np.isfinite(xset), axis=1)]
    return {
        "x_point_positions_m": xset,
        "primary_x_point_m": finite_x[0]
        if finite_x.size
        else np.asarray([np.nan, np.nan]),
        "lcfs_radii_m": np.asarray(read.radii, dtype=np.float64),
        "lcfs_positions_m": ring,
        "topology_diverted": np.asarray(read.is_diverted),
        "core_mask": core,
        "common_scrape_off_mask": common,
        "private_flux_mask": np.zeros_like(core),
        "excluded_material_mask": ~grid.inside_limiter,
    }


def _maximum_difference(left: np.ndarray, right: np.ndarray) -> float | None:
    """Return a finite maximum difference, treating matching NaNs as equal."""

    left = np.asarray(left)
    right = np.asarray(right)
    if left.dtype == np.bool_ and right.dtype == np.bool_:
        return float(np.any(np.logical_xor(left, right)))
    finite = np.isfinite(left) & np.isfinite(right)
    if not np.any(finite):
        return 0.0 if np.array_equal(left, right, equal_nan=True) else None
    return float(np.max(np.abs(left[finite] - right[finite])))


def _compare_emitted(standard: dict[str, Any], resolved: dict[str, Any]):
    """Compare production output against a candidate-state-only counterfactual."""

    result = {}
    for key in standard:
        left = np.asarray(standard[key])
        right = np.asarray(resolved[key])
        result[key] = {
            "exactly_equal": bool(np.array_equal(left, right, equal_nan=True)),
            "maximum_absolute_difference": _maximum_difference(left, right),
        }
    return result


@contextmanager
def _force_present_candidates_resolved():
    """Temporarily change only retained candidate states during JAX tracing."""

    original = boundary_kernel.xpoint_candidates

    def state_override(*args, **kwargs):
        result = dict(original(*args, **kwargs))
        result["state"] = jnp.where(
            result["present"],
            jnp.asarray(STATE_RESOLVED, dtype=result["state"].dtype),
            jnp.asarray(STATE_ABSENT, dtype=result["state"].dtype),
        )
        return result

    boundary_kernel.xpoint_candidates = state_override
    jax.clear_caches()
    try:
        yield
    finally:
        boundary_kernel.xpoint_candidates = original
        jax.clear_caches()


def build_report(
    store: Path = SHOT_STORE, shots: tuple[int, ...] = FROZEN_SHOTS
) -> dict[str, Any]:
    """Build candidate-state and state-only consequence evidence."""

    inputs = {}
    standard_reads = {}
    ray_centres = {}
    per_shot = []
    for shot in shots:
        slice_index, time_s, flux, grid, axis_seed = _shot_input(store, shot)
        initial = _boundary_read(flux, grid, axis_seed)
        ray_centre = np.asarray(initial.axis, dtype=np.float64)
        read = _boundary_read(flux, grid, tuple(ray_centre))
        ingredients = _candidate_ingredients(flux, grid, tuple(ray_centre))
        evidence = {key: np.asarray(value) for key, value in ingredients["xc"].items()}
        present = np.flatnonzero(evidence["present"])
        u_x = np.asarray(ingredients["u_x"], dtype=np.float64)
        valid = np.asarray(ingredients["x_valid"], dtype=bool)
        flood_level = float(ingredients["s_flood"])
        binding_slot = int(
            np.argmin(np.where(valid, np.abs(u_x - flood_level), np.inf))
        )
        candidates = [_candidate_record(evidence, int(index)) for index in present]
        binding = candidates[list(present).index(binding_slot)]
        binding["distance_from_flood_level"] = float(
            abs(u_x[binding_slot] - flood_level)
        )
        binding["normalised_flux"] = float(u_x[binding_slot])
        inputs[shot] = (flux, grid)
        standard_reads[shot] = read
        ray_centres[shot] = ray_centre
        per_shot.append(
            {
                "shot": shot,
                "slice_index": slice_index,
                "time_s": time_s,
                "boundary_resolved": bool(read.boundary_resolved),
                "binding_fit_state": int(read.x_binding_state),
                "emitted_x_point_fit_states": np.asarray(
                    read.xset_state, dtype=int
                ).tolist(),
                "retained_candidate_count": len(candidates),
                "retained_candidates": candidates,
                "binding_candidate": binding,
            }
        )

    counterfactual_reads = {}
    with _force_present_candidates_resolved():
        for shot in shots:
            flux, grid = inputs[shot]
            counterfactual_reads[shot] = _boundary_read(
                flux, grid, tuple(ray_centres[shot])
            )

    for row in per_shot:
        shot = row["shot"]
        flux, grid = inputs[shot]
        standard = standard_reads[shot]
        counterfactual = counterfactual_reads[shot]
        row["state_only_counterfactual"] = {
            "standard_boundary_resolved": bool(standard.boundary_resolved),
            "forced_boundary_resolved": bool(counterfactual.boundary_resolved),
            "emitted_fields": _compare_emitted(
                _emitted_fields(standard, ray_centres[shot], flux, grid),
                _emitted_fields(counterfactual, ray_centres[shot], flux, grid),
            ),
        }

    unresolved = [row for row in per_shot if not row["boundary_resolved"]]
    all_comparisons = [
        comparison
        for row in per_shot
        for comparison in row["state_only_counterfactual"]["emitted_fields"].values()
    ]
    devices = [
        {
            "platform": device.platform,
            "device_kind": getattr(device, "device_kind", type(device).__name__),
        }
        for device in jax.devices()
    ]
    return {
        "store": str(store),
        "shots": list(shots),
        "backend": {
            "jax_version": jax.__version__,
            "devices": devices,
            "x64_enabled": bool(jax.config.jax_enable_x64),
        },
        "configuration": {
            "flux_level_count": FLUX_LEVEL_COUNT,
            "flux_bisection_count": FLUX_BISECTION_COUNT,
            "radial_samples": BOUNDARY_RADIAL_SAMPLES,
            "output_angles": len(LCFS_ANGLES),
            "required_saddle_index": TARGET_SADDLE_INDEX,
            "required_class_margin": CLASS_MARGIN_THRESHOLD,
            "maximum_root_support_cell": ROOT_SUPPORT_LIMIT,
        },
        "per_shot": per_shot,
        "summary": {
            "shots_reported": len(per_shot),
            "unresolved_shots": [row["shot"] for row in unresolved],
            "resolved_shots": [
                row["shot"] for row in per_shot if row["boundary_resolved"]
            ],
            "unresolved_binding_cluster_indices": {
                str(row["shot"]): row["binding_candidate"]["cluster_signed_index"]
                for row in unresolved
            },
            "required_binding_cluster_index": TARGET_SADDLE_INDEX,
            "all_non_diagnostic_emitted_fields_exactly_equal": bool(
                all(item["exactly_equal"] for item in all_comparisons)
            ),
            "maximum_non_diagnostic_emitted_difference": max(
                item["maximum_absolute_difference"] or 0.0 for item in all_comparisons
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    """Write strict JSON evidence to standard output."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=SHOT_STORE)
    parser.add_argument("--shots", nargs="+", type=int, default=FROZEN_SHOTS)
    arguments = parser.parse_args(argv)
    report = build_report(arguments.store, tuple(arguments.shots))
    json.dump(report, sys.stdout, indent=2, allow_nan=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
