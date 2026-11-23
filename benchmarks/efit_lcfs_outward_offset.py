"""Diagnose signed LCFS offsets against stored EFIT geometry.

The production eight-angle ring is evaluated on the same deterministic slices
as the geometry scorecard.  Each radial offset is separated into the stored
polygon-to-map-contour discrepancy, connectivity boundary-flux selection, and
the remaining ray-cast discrepancy.  Sector proximity to stored X-points and
the limiter is retained so localized geometry effects can be distinguished
from a ring-wide displacement.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from benchmarks.efit_topology_boundary_score import (
    FROZEN_SHOTS,
    _BoundaryGrid,
    _live_flux_map,
    _point_to_polyline_distance,
    _slice_candidates,
    _stored_lcfs,
    _stored_x_points,
    _valid_points,
)
from nova.equilibrium.connectivity_boundary import host_boundary_read
from nova.equilibrium.labels import LCFS_ANGLES
from nova.equilibrium.wall_mask import inside_polygon
from nova.imas.mast_chain_factory import BOUNDARY_RADIAL_SAMPLES
from nova.imas.mast_vacuum_cohort import SHOT_STORE
from nova.imas.parity_tolerances import ScorecardField, registered_tolerances

PRODUCTION_LEVEL_COUNT = 48
PRODUCTION_BISECTION_COUNT = 12
REFINED_LEVEL_COUNT = 96
REFINED_BISECTION_COUNT = 18
LCFS_FLUX_FRACTION = 0.999


def _cross(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return the scalar cross product of two-dimensional vectors."""

    return left[..., 0] * right[..., 1] - left[..., 1] * right[..., 0]


def _ray_polygon_radius(
    centre: np.ndarray, direction: np.ndarray, polygon: np.ndarray
) -> float:
    """Return the first forward intersection of a ray and closed polygon."""

    start = np.asarray(polygon, dtype=np.float64)
    edge = np.roll(start, -1, axis=0) - start
    relative = start - centre
    denominator = _cross(np.broadcast_to(direction, edge.shape), edge)
    valid_denominator = np.abs(denominator) > np.finfo(np.float64).eps
    ray_fraction = np.divide(
        _cross(relative, edge),
        denominator,
        out=np.full(len(edge), np.nan),
        where=valid_denominator,
    )
    edge_fraction = np.divide(
        _cross(relative, np.broadcast_to(direction, edge.shape)),
        denominator,
        out=np.full(len(edge), np.nan),
        where=valid_denominator,
    )
    intersections = ray_fraction[
        valid_denominator
        & (ray_fraction >= 0.0)
        & (edge_fraction >= -1.0e-12)
        & (edge_fraction <= 1.0 + 1.0e-12)
    ]
    if not intersections.size:
        return float("nan")
    return float(np.min(intersections))


def _bilinear_flux(
    flux: np.ndarray, grid_r: np.ndarray, grid_z: np.ndarray, points: np.ndarray
) -> np.ndarray:
    """Interpolate a regular stored flux map at finite in-grid points."""

    points = np.asarray(points, dtype=np.float64)
    radius = points[:, 0]
    height = points[:, 1]
    radial_index = np.clip(np.searchsorted(grid_r, radius) - 1, 0, len(grid_r) - 2)
    vertical_index = np.clip(np.searchsorted(grid_z, height) - 1, 0, len(grid_z) - 2)
    radial_fraction = (radius - grid_r[radial_index]) / (
        grid_r[radial_index + 1] - grid_r[radial_index]
    )
    vertical_fraction = (height - grid_z[vertical_index]) / (
        grid_z[vertical_index + 1] - grid_z[vertical_index]
    )
    lower = (1.0 - radial_fraction) * flux[vertical_index, radial_index] + (
        radial_fraction * flux[vertical_index, radial_index + 1]
    )
    upper = (1.0 - radial_fraction) * flux[vertical_index + 1, radial_index] + (
        radial_fraction * flux[vertical_index + 1, radial_index + 1]
    )
    return (1.0 - vertical_fraction) * lower + vertical_fraction * upper


def _ray_flux_radius(
    flux: np.ndarray,
    grid_r: np.ndarray,
    grid_z: np.ndarray,
    centre: np.ndarray,
    direction: np.ndarray,
    maximum_radius: float,
    target_flux: float,
) -> float:
    """Locate the first target-flux crossing along an axis-centred ray."""

    radii = np.linspace(0.0, maximum_radius, 4097)
    points = centre + radii[:, None] * direction
    residual = _bilinear_flux(flux, grid_r, grid_z, points) - target_flux
    crossing = np.flatnonzero(residual[:-1] * residual[1:] <= 0.0)
    if not crossing.size:
        return float("nan")
    index = int(crossing[0])
    left = residual[index]
    right = residual[index + 1]
    fraction = 0.0 if right == left else -left / (right - left)
    return float(radii[index] + fraction * (radii[index + 1] - radii[index]))


def _correlation(left: list[float], right: list[float]) -> float | None:
    """Return a finite Pearson correlation, or ``None`` for constant inputs."""

    lhs = np.asarray(left, dtype=np.float64)
    rhs = np.asarray(right, dtype=np.float64)
    finite = np.isfinite(lhs) & np.isfinite(rhs)
    if np.count_nonzero(finite) < 3:
        return None
    lhs = lhs[finite]
    rhs = rhs[finite]
    if np.ptp(lhs) == 0.0 or np.ptp(rhs) == 0.0:
        return None
    return float(np.corrcoef(lhs, rhs)[0, 1])


def _read_boundary(
    flux: np.ndarray,
    grid: _BoundaryGrid,
    axis_seed: tuple[float, float],
    level_count: int,
    bisection_count: int,
) -> tuple[Any, np.ndarray, np.ndarray]:
    """Run the two-pass production boundary read and construct its ring."""

    rough = host_boundary_read(
        flux,
        grid,
        axis_seed,
        n_levels=level_count,
        n_bisect=bisection_count,
        n_ray=BOUNDARY_RADIAL_SAMPLES,
        angles=LCFS_ANGLES,
    )
    ray_centre = np.asarray(rough.axis, dtype=np.float64)
    read = host_boundary_read(
        flux,
        grid,
        tuple(ray_centre),
        n_levels=level_count,
        n_bisect=bisection_count,
        n_ray=BOUNDARY_RADIAL_SAMPLES,
        angles=LCFS_ANGLES,
    )
    directions = np.column_stack([np.cos(LCFS_ANGLES), np.sin(LCFS_ANGLES)])
    ring = ray_centre + np.asarray(read.radii)[:, None] * directions
    return read, ray_centre, ring


def _angle_rows(
    flux: np.ndarray,
    grid: _BoundaryGrid,
    stored_boundary: np.ndarray,
    stored_x_points: np.ndarray,
    stored_axis_flux: float,
    stored_boundary_flux: float,
    read: Any,
    ray_centre: np.ndarray,
    ring: np.ndarray,
) -> list[dict[str, Any]]:
    """Decompose every output-angle radial residual into named components."""

    directions = np.column_stack([np.cos(LCFS_ANGLES), np.sin(LCFS_ANGLES)])
    point_distance = _point_to_polyline_distance(ring, stored_boundary)
    point_inside = np.asarray(
        inside_polygon(
            ring[:, 0], ring[:, 1], stored_boundary[:, 0], stored_boundary[:, 1]
        ),
        dtype=bool,
    )
    point_signed = np.where(point_inside, -point_distance, point_distance)
    limiter = np.column_stack([grid.limiter_r, grid.limiter_z])
    flux_span = stored_boundary_flux - stored_axis_flux
    registered_target_flux = stored_axis_flux + LCFS_FLUX_FRACTION * flux_span
    rows = []
    for index, (angle, direction, extracted_point) in enumerate(
        zip(LCFS_ANGLES, directions, ring, strict=True)
    ):
        stored_radius = _ray_polygon_radius(ray_centre, direction, stored_boundary)
        limiter_radius = _ray_polygon_radius(ray_centre, direction, limiter)
        selected_contour_radius = _ray_flux_radius(
            flux,
            grid.rg,
            grid.zg,
            ray_centre,
            direction,
            limiter_radius,
            float(read.psi_lcfs),
        )
        registered_contour_radius = _ray_flux_radius(
            flux,
            grid.rg,
            grid.zg,
            ray_centre,
            direction,
            limiter_radius,
            registered_target_flux,
        )
        extracted_radius = float(read.radii[index])
        nearest_x_distance = (
            float(np.min(np.linalg.norm(stored_x_points - extracted_point, axis=1)))
            if len(stored_x_points)
            else None
        )
        rows.append(
            {
                "angle_index": index,
                "angle_degrees": float(np.rad2deg(angle)),
                "signed_point_to_stored_lcfs_m": float(point_signed[index]),
                "extracted_radius_m": extracted_radius,
                "stored_polygon_radius_m": stored_radius,
                "signed_radial_offset_m": extracted_radius - stored_radius,
                "polygon_to_registered_map_contour_m": (
                    registered_contour_radius - stored_radius
                ),
                "registered_to_selected_flux_contour_m": (
                    selected_contour_radius - registered_contour_radius
                ),
                "selected_contour_to_extracted_ray_m": (
                    extracted_radius - selected_contour_radius
                ),
                "radial_decomposition_closure_m": (
                    extracted_radius
                    - stored_radius
                    - (registered_contour_radius - stored_radius)
                    - (selected_contour_radius - registered_contour_radius)
                    - (extracted_radius - selected_contour_radius)
                ),
                "nearest_stored_x_point_distance_m": nearest_x_distance,
                "limiter_clearance_from_stored_lcfs_m": (
                    limiter_radius - stored_radius
                ),
            }
        )
    return rows


def _setting_score(
    flux: np.ndarray,
    grid: _BoundaryGrid,
    axis_seed: tuple[float, float],
    stored_boundary: np.ndarray,
    stored_x_points: np.ndarray,
    stored_axis_flux: float,
    stored_boundary_flux: float,
    level_count: int,
    bisection_count: int,
) -> dict[str, Any]:
    """Score one globally applied connectivity-resolution setting."""

    read, ray_centre, ring = _read_boundary(
        flux, grid, axis_seed, level_count, bisection_count
    )
    rows = _angle_rows(
        flux,
        grid,
        stored_boundary,
        stored_x_points,
        stored_axis_flux,
        stored_boundary_flux,
        read,
        ray_centre,
        ring,
    )
    signed_point = np.asarray([row["signed_point_to_stored_lcfs_m"] for row in rows])
    signed_radial = np.asarray([row["signed_radial_offset_m"] for row in rows])
    absolute_point = np.abs(signed_point)
    top_two = np.sort(absolute_point)[-2:]
    component_means = {
        key: float(np.nanmean([row[key] for row in rows]))
        for key in (
            "polygon_to_registered_map_contour_m",
            "registered_to_selected_flux_contour_m",
            "selected_contour_to_extracted_ray_m",
        )
    }
    positive_fraction = float(np.mean(signed_radial > 0.0))
    return {
        "flux_level_count": level_count,
        "flux_bisection_count": bisection_count,
        "ray_sample_count": BOUNDARY_RADIAL_SAMPLES,
        "ray_centre_rz_m": ray_centre.tolist(),
        "selected_boundary_flux_Wb_per_rad": float(read.psi_bnd),
        "selected_ray_flux_Wb_per_rad": float(read.psi_lcfs),
        "boundary_resolved": bool(read.boundary_resolved),
        "signed_mean_point_distance_m": float(np.mean(signed_point)),
        "mean_absolute_point_distance_m": float(np.mean(absolute_point)),
        "signed_mean_radial_offset_m": float(np.mean(signed_radial)),
        "radial_offset_standard_deviation_m": float(np.std(signed_radial)),
        "positive_angle_fraction": positive_fraction,
        "top_two_angle_absolute_offset_fraction": float(
            np.sum(top_two) / np.sum(absolute_point)
        ),
        "angular_distribution": (
            "ring-wide outward with angular modulation"
            if positive_fraction == 1.0
            else "ring-wide inward with angular modulation"
            if positive_fraction == 0.0
            else "mixed-sign angular sectors"
        ),
        "mean_radial_decomposition_m": component_means,
        "per_output_angle": rows,
    }


def _shot_score(store: Path, shot: int) -> dict[str, Any]:
    """Evaluate both numerical settings on one deterministic frozen slice."""

    group = zarr.open_group(str(store / f"{shot}.zarr"), mode="r")["efm"]
    candidates = _slice_candidates(group)
    if not np.any(candidates):
        raise ValueError(f"shot {shot} has no geometry-valid live flux-map row")
    current = np.asarray(group["plasma_current_c"][...], dtype=np.float64)
    slice_index = int(np.argmax(np.where(candidates, np.abs(current), -np.inf)))
    grid_r = np.asarray(group["gridr"][...], dtype=np.float64)
    grid_z = np.asarray(group["gridz"][...], dtype=np.float64)
    limiter_r = np.asarray(group["limiterr"][...], dtype=np.float64)
    limiter_z = np.asarray(group["limiterz"][...], dtype=np.float64)
    limiter_valid = _valid_points(np.column_stack([limiter_r, limiter_z]))
    limiter_r = limiter_r[limiter_valid]
    limiter_z = limiter_z[limiter_valid]
    radial_mesh, vertical_mesh = np.meshgrid(grid_r, grid_z)
    inside_limiter = np.asarray(
        inside_polygon(
            radial_mesh.ravel(),
            vertical_mesh.ravel(),
            limiter_r,
            limiter_z,
        )
    ).reshape(radial_mesh.shape)
    grid = _BoundaryGrid(
        rg=grid_r,
        zg=grid_z,
        inside_limiter=inside_limiter,
        limiter_r=limiter_r,
        limiter_z=limiter_z,
        wall_r=limiter_r,
        wall_z=limiter_z,
    )
    flux = _live_flux_map(group, slice_index, grid_r.size)
    axis_seed = (float(0.5 * (limiter_r.min() + limiter_r.max())), 0.0)
    stored_boundary = _stored_lcfs(group, slice_index)
    stored_x_points = _stored_x_points(group, slice_index)
    settings = {
        "production": _setting_score(
            flux,
            grid,
            axis_seed,
            stored_boundary,
            stored_x_points,
            float(group["psi_axis"][slice_index]),
            float(group["psi_boundary"][slice_index]),
            PRODUCTION_LEVEL_COUNT,
            PRODUCTION_BISECTION_COUNT,
        ),
        "refined": _setting_score(
            flux,
            grid,
            axis_seed,
            stored_boundary,
            stored_x_points,
            float(group["psi_axis"][slice_index]),
            float(group["psi_boundary"][slice_index]),
            REFINED_LEVEL_COUNT,
            REFINED_BISECTION_COUNT,
        ),
    }
    bound = float(registered_tolerances()[ScorecardField.LCFS_DISTANCE_M].bound)
    for setting in settings.values():
        setting["registered_bound_m"] = bound
        setting["passes_registered_bound"] = bool(
            setting["mean_absolute_point_distance_m"] <= bound
        )
    return {
        "shot": shot,
        "slice_index": slice_index,
        "time_s": float(group["time"][slice_index]),
        "source_grid_resolution": [int(grid_z.size), int(grid_r.size)],
        "settings": settings,
    }


def _mechanism_evidence(per_shot: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate sector and decomposition measurements across all shots."""

    production_rows = [
        angle
        for shot in per_shot
        for angle in shot["settings"]["production"]["per_output_angle"]
    ]
    signed = [row["signed_radial_offset_m"] for row in production_rows]
    x_proximity = [
        1.0 / row["nearest_stored_x_point_distance_m"]
        if row["nearest_stored_x_point_distance_m"] is not None
        else np.nan
        for row in production_rows
    ]
    limiter_proximity = [
        1.0 / row["limiter_clearance_from_stored_lcfs_m"] for row in production_rows
    ]
    polygon_map = [
        row["polygon_to_registered_map_contour_m"] for row in production_rows
    ]
    selected_flux = [
        row["registered_to_selected_flux_contour_m"] for row in production_rows
    ]
    ray_residual = [
        row["selected_contour_to_extracted_ray_m"] for row in production_rows
    ]
    component_means = {
        "polygon_to_registered_map_contour_m": float(np.nanmean(polygon_map)),
        "registered_to_selected_flux_contour_m": float(np.nanmean(selected_flux)),
        "selected_contour_to_extracted_ray_m": float(np.nanmean(ray_residual)),
    }
    dominant_component = max(component_means, key=lambda key: abs(component_means[key]))
    polygon_correlation = _correlation(signed, polygon_map)
    failing_shots = [
        shot
        for shot in per_shot
        if not shot["settings"]["production"]["passes_registered_bound"]
    ]
    return {
        "signed_radial_offset_correlation_with_x_point_proximity": _correlation(
            signed, x_proximity
        ),
        "signed_radial_offset_correlation_with_limiter_proximity": _correlation(
            signed, limiter_proximity
        ),
        "signed_radial_offset_correlation_with_polygon_map_discrepancy": (
            polygon_correlation
        ),
        "signed_radial_offset_correlation_with_selected_flux_displacement": (
            _correlation(signed, selected_flux)
        ),
        "signed_radial_offset_correlation_with_ray_cast_residual": _correlation(
            signed, ray_residual
        ),
        "all_shot_mean_decomposition_m": component_means,
        "largest_mean_component": dominant_component,
        "named_mechanism": (
            "stored LCFS polygon-to-flux-map contour mismatch"
            if polygon_correlation is not None and polygon_correlation > 0.9
            else "no single measured component dominates the signed offset"
        ),
        "failing_shot_findings": {
            str(shot["shot"]): {
                "angular_distribution": shot["settings"]["production"][
                    "angular_distribution"
                ],
                "outward_angle_count": int(
                    8 * shot["settings"]["production"]["positive_angle_fraction"]
                ),
                "output_angle_count": 8,
                "top_two_angle_absolute_offset_fraction": shot["settings"][
                    "production"
                ]["top_two_angle_absolute_offset_fraction"],
                "mean_radial_decomposition_m": shot["settings"]["production"][
                    "mean_radial_decomposition_m"
                ],
                "production_to_refined_signed_point_change_m": (
                    shot["settings"]["refined"]["signed_mean_point_distance_m"]
                    - shot["settings"]["production"]["signed_mean_point_distance_m"]
                ),
                "passes_registered_bound_after_refinement": shot["settings"]["refined"][
                    "passes_registered_bound"
                ],
            }
            for shot in failing_shots
        },
        "interpretation_rule": (
            "The largest absolute mean radial component names the ring-wide "
            "mechanism; correlations and per-shot angular concentration test "
            "whether X-point or limiter sectors instead explain the sign."
        ),
    }


def build_report(
    store: Path = SHOT_STORE, shots: tuple[int, ...] = FROZEN_SHOTS
) -> dict[str, Any]:
    """Build six-shot outward-offset attribution evidence."""

    per_shot = [_shot_score(store, shot) for shot in shots]
    bound = float(registered_tolerances()[ScorecardField.LCFS_DISTANCE_M].bound)
    return {
        "store": str(store),
        "shots": list(shots),
        "slice_selection": (
            "largest absolute finite efm/plasma_current_c among rows carrying "
            "a live flux map, magnetic axis and declared LCFS"
        ),
        "registered_lcfs_mean_absolute_distance_bound_m": bound,
        "boundary_span_ray_fraction": LCFS_FLUX_FRACTION,
        "boundary_span_directional_effect": (
            "A value below one moves the ray target inward and therefore cannot "
            "create a positive outward offset."
        ),
        "settings_applied_identically_to_every_shot": {
            "production": [PRODUCTION_LEVEL_COUNT, PRODUCTION_BISECTION_COUNT],
            "refined": [REFINED_LEVEL_COUNT, REFINED_BISECTION_COUNT],
        },
        "per_shot": per_shot,
        "mechanism_evidence": _mechanism_evidence(per_shot),
        "summary": {
            setting: {
                "pass_count": sum(
                    shot["settings"][setting]["passes_registered_bound"]
                    for shot in per_shot
                ),
                "per_shot_signed_mean_point_distance_m": {
                    str(shot["shot"]): shot["settings"][setting][
                        "signed_mean_point_distance_m"
                    ]
                    for shot in per_shot
                },
                "per_shot_mean_absolute_point_distance_m": {
                    str(shot["shot"]): shot["settings"][setting][
                        "mean_absolute_point_distance_m"
                    ]
                    for shot in per_shot
                },
            }
            for setting in ("production", "refined")
        },
    }


def main(argv: list[str] | None = None) -> int:
    """Write strict JSON diagnostic evidence to standard output."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=SHOT_STORE)
    parser.add_argument("--shots", type=int, nargs="+", default=FROZEN_SHOTS)
    arguments = parser.parse_args(argv)
    report = build_report(arguments.store, tuple(arguments.shots))
    json.dump(report, sys.stdout, indent=2, allow_nan=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
