"""Score Nova topology extraction directly against stored EFIT geometry.

Each shot contributes the geometry-valid flux-map row carrying the largest
absolute fitted plasma current.  Stored axis, LCFS and X-point values qualify
the scoring row but are never passed to the Nova connectivity read.  The read
first obtains a sub-grid magnetic axis from the machine-centre seed, then reads
the boundary again about that extracted axis; ray radii are defined relative
to the axis supplied to the kernel.

The report keeps signed coordinate residuals and a signed LCFS offset alongside
the non-negative distances used by the registered parity bounds.  LCFS sign is
negative when Nova's extracted surface lies inside the stored polygon and
positive when it lies outside.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from nova.equilibrium.connectivity_boundary import host_boundary_read
from nova.equilibrium.wall_mask import inside_polygon
from nova.imas.mast_vacuum_cohort import SHOT_STORE
from nova.imas.parity_tolerances import ScorecardField, registered_tolerances

FROZEN_SHOTS = (21978, 21983, 21985, 21986, 21989, 22086)
RAY_COUNT = 512
FLUX_LEVEL_COUNT = 96
FLUX_BISECTION_COUNT = 18


@dataclass(frozen=True)
class _BoundaryGrid:
    """Geometry fields consumed by Nova's host connectivity adapter."""

    rg: np.ndarray
    zg: np.ndarray
    inside_limiter: np.ndarray
    limiter_r: np.ndarray
    limiter_z: np.ndarray
    wall_r: np.ndarray
    wall_z: np.ndarray


def _valid_points(points: np.ndarray) -> np.ndarray:
    """Return a mask for finite positions within broad machine bounds."""

    points = np.asarray(points, dtype=np.float64)
    return (
        np.all(np.isfinite(points), axis=-1)
        & (points[..., 0] > 0.0)
        & (points[..., 0] < 10.0)
        & (np.abs(points[..., 1]) < 10.0)
    )


def _live_flux_map(
    group: zarr.Group, slice_index: int, radial_count: int
) -> np.ndarray:
    """Return the finite ``(height, radius)`` plane from a padded EFM map."""

    raw = np.asarray(group["psirz"][slice_index], dtype=np.float64)
    columns = np.flatnonzero(np.all(np.isfinite(raw), axis=0))
    if columns.size != radial_count:
        raise ValueError(
            f"slice {slice_index} carries {columns.size} live radial columns, "
            f"expected {radial_count}"
        )
    return raw[:, columns]


def _stored_lcfs(group: zarr.Group, slice_index: int) -> np.ndarray:
    """Read the declared number of finite stored LCFS vertices."""

    count_value = float(group["lcfsn_c"][slice_index])
    if not np.isfinite(count_value) or count_value != int(count_value):
        raise ValueError(f"slice {slice_index} has invalid LCFS count {count_value}")
    count = int(count_value)
    boundary = np.column_stack(
        [group["lcfs_r"][slice_index, :count], group["lcfs_z"][slice_index, :count]]
    ).astype(np.float64)
    if count < 3 or not np.all(_valid_points(boundary)):
        raise ValueError(f"slice {slice_index} has no finite stored LCFS polygon")
    return boundary


def _stored_x_points(group: zarr.Group, slice_index: int) -> np.ndarray:
    """Return the finite members of the stored unordered X-point pair."""

    points = np.asarray(
        [
            [
                float(group["xpoint1_rc"][slice_index]),
                float(group["xpoint1_zc"][slice_index]),
            ],
            [
                float(group["xpoint2_rc"][slice_index]),
                float(group["xpoint2_zc"][slice_index]),
            ],
        ],
        dtype=np.float64,
    )
    return points[_valid_points(points)]


def _point_to_polyline_distance(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Return each point's Euclidean distance to a closed piecewise-linear curve."""

    segment_start = np.asarray(polygon, dtype=np.float64)
    segment_end = np.roll(segment_start, -1, axis=0)
    segment = segment_end - segment_start
    denominator = np.sum(segment * segment, axis=1)
    offset = np.asarray(points, dtype=np.float64)[:, None, :] - segment_start[None]
    fraction = np.divide(
        np.sum(offset * segment[None], axis=2),
        denominator[None],
        out=np.zeros((len(points), len(polygon)), dtype=np.float64),
        where=denominator[None] > 0.0,
    )
    fraction = np.clip(fraction, 0.0, 1.0)
    separation = offset - fraction[:, :, None] * segment[None]
    return np.min(np.linalg.norm(separation, axis=2), axis=1)


def _boundary_scores(extracted: np.ndarray, stored: np.ndarray) -> dict[str, float]:
    """Return signed and unsigned polyline distances between two boundaries."""

    extracted_to_stored = _point_to_polyline_distance(extracted, stored)
    stored_to_extracted = _point_to_polyline_distance(stored, extracted)
    extracted_inside = np.asarray(
        inside_polygon(extracted[:, 0], extracted[:, 1], stored[:, 0], stored[:, 1]),
        dtype=bool,
    )
    signed = np.where(extracted_inside, -extracted_to_stored, extracted_to_stored)
    return {
        "signed_mean_distance_m": float(np.mean(signed)),
        "mean_absolute_distance_m": float(np.mean(extracted_to_stored)),
        "symmetric_mean_absolute_distance_m": float(
            0.5 * (np.mean(extracted_to_stored) + np.mean(stored_to_extracted))
        ),
        "maximum_absolute_distance_m": float(
            max(np.max(extracted_to_stored), np.max(stored_to_extracted))
        ),
    }


def _x_point_scores(extracted: np.ndarray, stored: np.ndarray) -> dict[str, Any]:
    """Match two unordered fixed-capacity point sets at minimum total distance."""

    extracted = np.asarray(extracted, dtype=np.float64)
    stored = np.asarray(stored, dtype=np.float64)
    match_count = min(len(extracted), len(stored))
    matches: list[dict[str, Any]] = []
    if match_count:
        cost = np.linalg.norm(extracted[:, None, :] - stored[None, :, :], axis=2)
        choices = (
            (
                float(sum(cost[left, right] for left, right in zip(lhs, rhs))),
                lhs,
                rhs,
            )
            for lhs in itertools.permutations(range(len(extracted)), match_count)
            for rhs in itertools.permutations(range(len(stored)), match_count)
        )
        _, left_indices, right_indices = min(choices, key=lambda item: item[0])
        for left, right in zip(left_indices, right_indices):
            delta = extracted[left] - stored[right]
            matches.append(
                {
                    "extracted_index": int(left),
                    "stored_index": int(right),
                    "extracted_rz_m": extracted[left].tolist(),
                    "stored_rz_m": stored[right].tolist(),
                    "signed_delta_rz_m": delta.tolist(),
                    "distance_m": float(np.linalg.norm(delta)),
                }
            )
    distances = [match["distance_m"] for match in matches]
    return {
        "extracted_count": int(len(extracted)),
        "stored_count": int(len(stored)),
        "count_agreement": bool(len(extracted) == len(stored)),
        "matches": matches,
        "maximum_matched_distance_m": max(distances) if distances else None,
    }


def _finite_or_none(value: float) -> float | None:
    """Keep strict JSON free of non-finite numeric sentinels."""

    return float(value) if np.isfinite(value) else None


def _slice_candidates(group: zarr.Group) -> np.ndarray:
    """Return rows carrying a live map, stored axis and stored LCFS."""

    time = np.asarray(group["time"][...], dtype=np.float64)
    current = np.asarray(group["plasma_current_c"][...], dtype=np.float64)
    axes = np.column_stack(
        [group["magnetic_axis_r"][...], group["magnetic_axis_z"][...]]
    ).astype(np.float64)
    lcfs = np.stack([group["lcfs_r"][...], group["lcfs_z"][...]], axis=-1).astype(
        np.float64
    )
    live_map = np.asarray(
        [
            np.count_nonzero(np.all(np.isfinite(plane), axis=0))
            == group["gridr"].shape[0]
            for plane in group["psirz"]
        ],
        dtype=bool,
    )
    lcfs_count = np.asarray(group["lcfsn_c"][...], dtype=np.float64)
    lcfs_valid = np.sum(_valid_points(lcfs), axis=1)
    return (
        np.isfinite(time)
        & np.isfinite(current)
        & _valid_points(axes)
        & np.isfinite(lcfs_count)
        & (lcfs_count >= 3)
        & (lcfs_valid >= lcfs_count)
        & live_map
    )


def _score_shot(store: Path, shot: int, angles: np.ndarray) -> dict[str, Any]:
    """Extract and score one deterministic stored-map slice."""

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
    flux_map = _live_flux_map(group, slice_index, grid_r.size)
    machine_centre_seed = (float(0.5 * (limiter_r.min() + limiter_r.max())), 0.0)
    rough = host_boundary_read(
        flux_map,
        grid,
        machine_centre_seed,
        n_levels=FLUX_LEVEL_COUNT,
        n_bisect=FLUX_BISECTION_COUNT,
        n_ray=angles.size,
        angles=angles,
        lcfs_norm=1.0,
    )
    extracted = host_boundary_read(
        flux_map,
        grid,
        rough.axis,
        n_levels=FLUX_LEVEL_COUNT,
        n_bisect=FLUX_BISECTION_COUNT,
        n_ray=angles.size,
        angles=angles,
        lcfs_norm=1.0,
    )
    if not extracted.found:
        raise ValueError(f"shot {shot} slice {slice_index} has no extracted boundary")

    extracted_axis = np.asarray(extracted.axis, dtype=np.float64)
    stored_axis = np.asarray(
        [
            group["magnetic_axis_r"][slice_index],
            group["magnetic_axis_z"][slice_index],
        ],
        dtype=np.float64,
    )
    axis_delta = extracted_axis - stored_axis
    axis_distance = float(np.linalg.norm(axis_delta))

    extracted_boundary = extracted_axis + np.column_stack(
        [extracted.radii * np.cos(angles), extracted.radii * np.sin(angles)]
    )
    extracted_boundary = extracted_boundary[_valid_points(extracted_boundary)]
    if extracted_boundary.shape[0] < 3:
        raise ValueError(
            f"shot {shot} slice {slice_index} extracted fewer than three LCFS points"
        )
    stored_boundary = _stored_lcfs(group, slice_index)
    boundary = _boundary_scores(extracted_boundary, stored_boundary)

    extracted_x_points = np.asarray(extracted.xset, dtype=np.float64)
    extracted_x_points = extracted_x_points[_valid_points(extracted_x_points)]
    stored_x_points = _stored_x_points(group, slice_index)
    x_points = _x_point_scores(extracted_x_points, stored_x_points)
    stored_diverted = bool(len(stored_x_points))
    extracted_diverted = bool(extracted.is_diverted)

    tolerances = registered_tolerances()
    axis_bound = tolerances[ScorecardField.MAGNETIC_AXIS_DISTANCE_M]
    lcfs_bound = tolerances[ScorecardField.LCFS_DISTANCE_M]
    x_point_bound = tolerances[ScorecardField.X_POINT_DISTANCE_M]
    x_point_distance = x_points["maximum_matched_distance_m"]
    x_point_pass = bool(
        x_points["count_agreement"]
        and (x_point_distance is not None or not stored_diverted)
        and (x_point_distance is None or x_point_bound.passes(x_point_distance))
    )
    topology_agreement = extracted_diverted == stored_diverted
    resolution = {
        "vertical_points": int(grid_z.size),
        "radial_points": int(grid_r.size),
        "label": f"({grid_z.size},{grid_r.size})",
    }
    return {
        "shot": int(shot),
        "slice_index": slice_index,
        "time_s": float(group["time"][slice_index]),
        "selection_value_plasma_current_c_A": float(current[slice_index]),
        "source_grid_resolution": resolution,
        "magnetic_axis": {
            "extracted_rz_m": extracted_axis.tolist(),
            "stored_rz_m": stored_axis.tolist(),
            "signed_delta_rz_m": axis_delta.tolist(),
            "distance_m": axis_distance,
            "registered_bound_m": float(axis_bound.bound),
            "passes_registered_bound": bool(axis_bound.passes(axis_distance)),
        },
        "lcfs": {
            **boundary,
            "extracted_point_count": int(extracted_boundary.shape[0]),
            "stored_point_count": int(stored_boundary.shape[0]),
            "ray_flux_fraction_of_boundary_span": 0.999,
            "registered_metric": "mean_absolute_distance_m",
            "registered_bound_m": float(lcfs_bound.bound),
            "passes_registered_bound": bool(
                lcfs_bound.passes(boundary["mean_absolute_distance_m"])
            ),
        },
        "x_points": {
            **x_points,
            "registered_bound_m": float(x_point_bound.bound),
            "passes_registered_bound_and_count": x_point_pass,
        },
        "topology": {
            "extracted_class": "diverted" if extracted_diverted else "limited",
            "stored_class": "diverted" if stored_diverted else "limited",
            "class_agreement": topology_agreement,
            "agreement_resolution": resolution if topology_agreement else None,
            "class_disagreement": None
            if topology_agreement
            else {
                "observed_at": resolution,
                "resolved_at": None,
                "reason": "no finer measured flux map is stored",
            },
            "boundary_resolved": bool(extracted.boundary_resolved),
            "class_margin": _finite_or_none(extracted.class_margin),
            "class_margin_is_infinite": bool(np.isinf(extracted.class_margin)),
            "x_point_fit_states": np.asarray(extracted.xset_state, dtype=int).tolist(),
            "x_candidate_count": int(extracted.x_candidate_count),
            "x_unresolved_count": int(extracted.x_unresolved_count),
        },
        "flux_landmarks": {
            "extracted_axis_flux_Wb_per_rad": float(extracted.axis_psi),
            "stored_axis_flux_Wb_per_rad": float(group["psi_axis"][slice_index]),
            "extracted_boundary_flux_Wb_per_rad": float(extracted.psi_bnd),
            "stored_boundary_flux_Wb_per_rad": float(
                group["psi_boundary"][slice_index]
            ),
        },
    }


def build_report(
    store: Path = SHOT_STORE, shots: tuple[int, ...] = FROZEN_SHOTS
) -> dict[str, Any]:
    """Score one deterministic stored-map slice for every requested shot."""

    angles = np.linspace(0.0, 2.0 * np.pi, RAY_COUNT, endpoint=False)
    per_shot = [_score_shot(store, shot, angles) for shot in shots]
    axis_distances = [row["magnetic_axis"]["distance_m"] for row in per_shot]
    lcfs_distances = [row["lcfs"]["mean_absolute_distance_m"] for row in per_shot]
    x_distances = [
        row["x_points"]["maximum_matched_distance_m"]
        for row in per_shot
        if row["x_points"]["maximum_matched_distance_m"] is not None
    ]
    topology_agreement_count = sum(
        row["topology"]["class_agreement"] for row in per_shot
    )
    return {
        "store": str(store),
        "shots": list(shots),
        "slice_selection": (
            "largest absolute finite efm/plasma_current_c among rows carrying "
            "a live flux map, magnetic axis and declared LCFS"
        ),
        "extraction": {
            "implementation": (
                "nova.equilibrium.connectivity_boundary.host_boundary_read"
            ),
            "initial_axis_seed": "stored-limiter radial midpoint at Z=0",
            "axis_refinement": (
                "repeat the boundary read about its extracted sub-grid axis"
            ),
            "ray_count": RAY_COUNT,
            "flux_level_count": FLUX_LEVEL_COUNT,
            "flux_bisection_count": FLUX_BISECTION_COUNT,
            "lcfs_ray_flux_fraction": 0.999,
        },
        "registered_bounds": {
            "magnetic_axis_distance_m": float(
                registered_tolerances()[ScorecardField.MAGNETIC_AXIS_DISTANCE_M].bound
            ),
            "lcfs_distance_m": float(
                registered_tolerances()[ScorecardField.LCFS_DISTANCE_M].bound
            ),
            "x_point_distance_m": float(
                registered_tolerances()[ScorecardField.X_POINT_DISTANCE_M].bound
            ),
            "topology_class_agreement_fraction": float(
                registered_tolerances()[
                    ScorecardField.TOPOLOGY_CLASS_AGREEMENT_FRACTION
                ].bound
            ),
        },
        "per_shot": per_shot,
        "summary": {
            "shots_scored": len(per_shot),
            "maximum_axis_distance_m": max(axis_distances),
            "axis_bound_pass_count": sum(
                row["magnetic_axis"]["passes_registered_bound"] for row in per_shot
            ),
            "maximum_lcfs_mean_absolute_distance_m": max(lcfs_distances),
            "lcfs_bound_pass_count": sum(
                row["lcfs"]["passes_registered_bound"] for row in per_shot
            ),
            "maximum_x_point_distance_m": max(x_distances) if x_distances else None,
            "x_point_bound_and_count_pass_count": sum(
                row["x_points"]["passes_registered_bound_and_count"] for row in per_shot
            ),
            "topology_class_agreement_count": topology_agreement_count,
            "topology_class_agreement_fraction": topology_agreement_count
            / len(per_shot),
            "class_disagreements": [
                {
                    "shot": row["shot"],
                    **row["topology"]["class_disagreement"],
                }
                for row in per_shot
                if row["topology"]["class_disagreement"] is not None
            ],
            "all_registered_geometry_bounds_pass": bool(
                all(
                    row["magnetic_axis"]["passes_registered_bound"]
                    and row["lcfs"]["passes_registered_bound"]
                    and row["x_points"]["passes_registered_bound_and_count"]
                    and row["topology"]["class_agreement"]
                    for row in per_shot
                )
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    """Write strict JSON topology and boundary evidence to standard output."""

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
