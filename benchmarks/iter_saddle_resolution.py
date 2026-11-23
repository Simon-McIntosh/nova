"""Diagnose ITER EQDSK critical-point detection and limiter containment.

The measurement runs Nova's signed-degree detector over the entire computational
grid.  The all-true geometric mask is intentional: containment is evaluated only
after the complete critical-point table has been recovered.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib.util import find_spec
import json
import math
from pathlib import Path
from typing import Any

import jax
import numpy as np

from nova.equilibrium.stencil_nulls import critical_point_candidates_batch
from nova.equilibrium.wall_mask import inside_polygon
from nova.io import geqdsk
from nova.jax.config import configure_dtypes


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT / "docs/figures/primary-xpoint-evidence/iter-saddle-resolution.json"
)
SEARCH_CAPACITY = 64
EXPECTED_X_REGION_M = {"r": [4.0, 6.0], "z": [-4.0, -3.0]}


def _eqdsk_path() -> Path:
    specification = find_spec("torax")
    if specification is None or specification.origin is None:
        raise RuntimeError("TORAX installation is required for the ITER equilibrium")
    return (
        Path(specification.origin).resolve().parent
        / "data/third_party/geo/iterhybrid_cocos17.eqdsk"
    )


def _strict(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _strict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict(item) for item in value]
    if isinstance(value, np.ndarray):
        return _strict(value.tolist())
    if isinstance(value, np.generic):
        return _strict(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _closed_contour(radius: np.ndarray, height: np.ndarray) -> np.ndarray:
    contour = np.column_stack((radius, height)).astype(np.float64)
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour


def _is_axis_aligned_rectangle(contour: np.ndarray) -> bool:
    if len(contour) != 5 or not np.array_equal(contour[0], contour[-1]):
        return False
    unique_radius = np.unique(contour[:-1, 0])
    unique_height = np.unique(contour[:-1, 1])
    corners = {
        (float(radius), float(height))
        for radius in unique_radius
        for height in unique_height
    }
    return (
        len(unique_radius) == 2
        and len(unique_height) == 2
        and corners == {tuple(point) for point in contour[:-1]}
    )


def _search_index(
    flux: np.ndarray,
    radius: np.ndarray,
    height: np.ndarray,
    target_index: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    full_grid = np.ones(flux.shape, dtype=bool)
    result = jax.device_get(
        critical_point_candidates_batch(
            flux[None],
            radius,
            height,
            full_grid,
            k_slots=SEARCH_CAPACITY,
            material_dilate=0,
            target_index=target_index,
            noise_sigma=0.0,
        )
    )
    present = np.asarray(result["present"])[0]
    points = []
    for slot in np.flatnonzero(present):
        points.append(
            {
                "coordinate_m": [
                    float(np.asarray(result["r"])[0, slot]),
                    float(np.asarray(result["z"])[0, slot]),
                ],
                "flux_wb_per_rad": float(np.asarray(result["psi"])[0, slot]),
                "signed_index": int(np.asarray(result["native_signed_index"])[0, slot]),
                "resolved": bool(np.asarray(result["resolved"])[0, slot]),
                "state": int(np.asarray(result["state"])[0, slot]),
                "source_cell": int(np.asarray(result["source_cell"])[0, slot]),
                "native_winding": float(np.asarray(result["native_winding"])[0, slot]),
                "confidence": float(np.asarray(result["confidence"])[0, slot]),
                "cluster_index_certified": bool(
                    np.asarray(result["cluster_index_certified"])[0, slot]
                ),
            }
        )
    summary = {
        "target_signed_index": target_index,
        "candidate_count_before_compaction": int(result["candidate_count"][0]),
        "cluster_count": int(result["cluster_count"][0]),
        "reported_count": len(points),
        "resolved_count": int(np.sum(np.asarray(result["resolved"])[0])),
        "overflow": bool(result["overflow"][0]),
        "work_overflow": bool(result["work_overflow"][0]),
        "candidate_index_sum": int(result["candidate_index_sum"][0]),
        "domain_signed_index": int(result["domain_signed_index"][0]),
    }
    return summary, points


def measure() -> dict[str, Any]:
    configure_dtypes()
    source = _eqdsk_path()
    data = geqdsk.read(str(source))
    radius = np.asarray(data["x"], dtype=np.float64)
    height = np.asarray(data["z"], dtype=np.float64)
    flux = np.asarray(data["psi"], dtype=np.float64).T
    limiter = _closed_contour(data["xlim"], data["zlim"])
    boundary = np.column_stack((data["xbdry"], data["zbdry"])).astype(np.float64)

    searches = []
    critical_points = []
    for target_index in (-1, 1):
        summary, points = _search_index(flux, radius, height, target_index)
        searches.append(summary)
        critical_points.extend(points)

    point_radius = np.asarray(
        [point["coordinate_m"][0] for point in critical_points], dtype=np.float64
    )
    point_height = np.asarray(
        [point["coordinate_m"][1] for point in critical_points], dtype=np.float64
    )
    contained = inside_polygon(point_radius, point_height, limiter[:, 0], limiter[:, 1])
    for point, is_contained in zip(critical_points, contained, strict=True):
        point["inside_eqdsk_limiter"] = bool(is_contained)

    resolved_saddles = [
        point
        for point in critical_points
        if point["signed_index"] == -1 and point["resolved"]
    ]
    contained_saddles = [
        point for point in resolved_saddles if point["inside_eqdsk_limiter"]
    ]
    rejected_saddles = [
        point for point in resolved_saddles if not point["inside_eqdsk_limiter"]
    ]
    if rejected_saddles:
        situation = "detected_then_rejected_by_containment"
        repair = "replace the containment geometry with the governed first wall"
    elif resolved_saddles:
        raise RuntimeError(
            "signed-degree search found a contained saddle, which is neither "
            "permitted diagnosis"
        )
    else:
        situation = "never_detected"
        repair = (
            "repair the detection or field-data seam; the complete signed-degree "
            "census contains no saddle even though the grid covers the expected "
            "lower-divertor region, so changing containment cannot recover it"
        )

    lower_boundary_index = int(np.argmin(boundary[:, 1]))
    expected_x = boundary[lower_boundary_index]
    limiter_is_rectangle = _is_axis_aligned_rectangle(limiter)
    if not limiter_is_rectangle:
        raise RuntimeError("the ITER EQDSK limiter is no longer a bounding rectangle")
    if any(search["overflow"] or search["work_overflow"] for search in searches):
        raise RuntimeError("critical-point search capacity did not report every point")

    grid_extent = {
        "r_m": [float(radius[0]), float(radius[-1])],
        "z_m": [float(height[0]), float(height[-1])],
    }
    expected_location_covered = bool(
        radius[0] <= expected_x[0] <= radius[-1]
        and height[0] <= expected_x[1] <= height[-1]
    )
    return {
        "schema": "nova-iter-saddle-resolution",
        "measurement": (
            "full-grid signed-degree critical-point census and post-search "
            "EQDSK-limiter containment classification"
        ),
        "source": {
            "path": str(source),
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "cocos": 17,
            "flux_unit": "Wb/rad",
        },
        "typing_hypothesis": {
            "status": "falsified_before_this_measurement",
            "reason": (
                "Nova types native critical-point cells by the signed winding "
                "degree of grad-psi; the counter-clockwise cell traversal gives "
                "a saddle index of -1 and needs continuity of grad-psi, which a "
                "C1 flux field supplies"
            ),
        },
        "grid": {
            "shape_nz_nr": list(flux.shape),
            "extent": grid_extent,
            "spacing_m": [
                float(np.diff(radius).mean()),
                float(np.diff(height).mean()),
            ],
            "expected_iter_x_region_m": EXPECTED_X_REGION_M,
            "expected_lower_x_from_plasma_boundary_m": expected_x.tolist(),
            "expected_location_covered": expected_location_covered,
            "interpretation": (
                "the grid includes the lower-divertor region where the supplied "
                "plasma boundary reaches its lowest point"
            ),
        },
        "eqdsk_limiter": {
            "declared_point_count": int(data["nlim"]),
            "stored_points_rz_m": limiter.tolist(),
            "unique_point_count": int(len(np.unique(limiter[:-1], axis=0))),
            "closed": bool(np.array_equal(limiter[0], limiter[-1])),
            "extent_r_m": [float(limiter[:, 0].min()), float(limiter[:, 0].max())],
            "extent_z_m": [float(limiter[:, 1].min()), float(limiter[:, 1].max())],
            "classification": "computational_grid_bounding_rectangle",
            "is_axis_aligned_rectangle": limiter_is_rectangle,
            "is_first_wall": False,
            "is_plasma_limiting_surface": False,
            "maximum_inset_from_grid_edge_m": float(
                max(
                    limiter[:, 0].min() - radius[0],
                    radius[-1] - limiter[:, 0].max(),
                    limiter[:, 1].min() - height[0],
                    height[-1] - limiter[:, 1].max(),
                )
            ),
            "interpretation": (
                "the EQDSK limiter block contains only the four computational "
                "domain corners plus closure, so it supplies neither an ITER "
                "first-wall outline nor a plasma-limiting contour"
            ),
        },
        "search": {
            "routine": "critical_point_candidates_batch",
            "typing": "signed degree of the native finite-difference gradient",
            "containment_mask": "none; all grid samples eligible",
            "material_dilate": 0,
            "noise_sigma_wb_per_rad": 0.0,
            "capacity_per_signed_index": SEARCH_CAPACITY,
            "summaries": searches,
        },
        "critical_points": critical_points,
        "diagnosis": {
            "resolved_saddle_count": len(resolved_saddles),
            "resolved_saddles_inside_eqdsk_limiter": len(contained_saddles),
            "resolved_saddles_rejected_by_eqdsk_limiter": len(rejected_saddles),
            "situation": situation,
            "repair_direction": repair,
            "physical_geometry_follow_on": (
                "a governed ITER first-wall contour is still required before "
                "wall reachability can be interpreted physically"
            ),
        },
        "verdict": (
            "NEVER DETECTED: the full-grid signed-degree census reports only the "
            "magnetic-axis extremum and no saddle, before any containment predicate; "
            "the unavailable panel is therefore a detection or field-data fault, "
            "not limiter rejection or a physical property of the diverted equilibrium"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    payload = measure()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(_strict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(_strict(payload["diagnosis"]), sort_keys=True))


if __name__ == "__main__":
    main()
