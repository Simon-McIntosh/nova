"""Discriminate physical and numerical origins of exterior label current.

The landed exact-moment tare supplies the cohort and exterior-current patches.
This study does not alter a current, flux function, label, or machine geometry.
It asks whether the detectable patch positions persist across shots, whether
the label's strict Grad--Shafranov inconsistency accounts for their current,
whether their magnitude resembles released conductor currents, and whether
the measurement survives a factor-two grid decimation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import deque
from pathlib import Path
from typing import Any

import jax
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from benchmarks import diiid_exact_clipped_tare as exact_tare
from benchmarks import diiid_unclaimed_current_patches as patches
from benchmarks.diiid_label_resolve_gate import _operator
from benchmarks.diiid_negative_tail_attribution import _current_vector
from benchmarks.diiid_root_existence import _profile_source
from benchmarks.diiid_state_of_play_figures import boundary_gradient_minimum
from nova.equilibrium.map_extraction import apply_delta_star
from nova.imas.diiid_description import DiiidDescriptionRegistry, vacuum_response
from nova.jax.config import configure_dtypes

DEFAULT_DATA = exact_tare.DEFAULT_DATA
DEFAULT_OUTPUT = Path("docs/figures/diiid-forward-onboarding/unclaimed-origin")
SOURCE_RECEIPT = patches.DEFAULT_OUTPUT / patches.RECEIPT_NAME
PREREGISTRATION_NAME = "unclaimed_current_origin_preregistration.json"
RECEIPT_NAME = "unclaimed_current_origin_receipt.json"
FIGURE_NAME = "unclaimed_current_origin.png"
CHECKPOINT_NAME = "unclaimed_current_origin_frames.jsonl"
SETTLEMENT_RECEIPT_NAME = "cluster_lead_settlement_receipt.json"

FRAME_COUNT = 60
SHOT_COUNT = 20
DETECTABLE_PATCH_COUNT = 391
NATIVE_CELL_M = 0.021
CLUSTER_RADIUS_M = 2.0 * NATIVE_CELL_M
CLUSTER_MINIMUM_PATCHES = 3
POSITION_MINIMUM_SHOTS = 15
NON_GS_DOMINANT_FRACTION = 0.80
NON_GS_LABEL_CONTENT_FRACTION = 0.9968
RESOLUTION_MAXIMUM_RELATIVE_CHANGE = 0.20
LANDED_MEDIAN_UNCLAIMED_AMPERE_TURNS = 452_070.90359150956
LANDED_MEDIAN_UNCLAIMED_FRACTION = 0.6149271950681132
LEAD_CLUSTER_CENTRES_RZ_M = {
    2: (2.3475159346026016, 1.2498395802321838),
    3: (2.357307745680109, -1.2496513053830223),
}


def preregistration() -> dict[str, Any]:
    """Return the complete discriminator and verdict policy."""

    return {
        "cohort": {
            "frames": FRAME_COUNT,
            "shots": SHOT_COUNT,
            "detectable_patches": DETECTABLE_PATCH_COUNT,
            "authority": str(SOURCE_RECEIPT),
            "screen": (
                "every shot absent from the complete 603-shot polarity population"
            ),
        },
        "position_stability": {
            "algorithm": "Euclidean DBSCAN over R-Z centroids",
            "cluster_radius_m": CLUSTER_RADIUS_M,
            "minimum_patches": CLUSTER_MINIMUM_PATCHES,
            "native_cell_m": NATIVE_CELL_M,
            "physical_position_rule": (
                "a cluster spans at least fifteen of twenty shots and both its "
                "radial and vertical peak-to-peak spreads do not exceed 0.021 m"
            ),
        },
        "non_gs_accounting": {
            "construction": (
                "extract p-prime and FF-prime from each label, solve the fixed-border "
                "profile equation, apply Delta-star to label minus solution, and "
                "integrate exterior L1 apparent toroidal current"
            ),
            "dominant_fraction": NON_GS_DOMINANT_FRACTION,
            "landed_irreducible_strict_gs_residual_fraction": (
                NON_GS_LABEL_CONTENT_FRACTION
            ),
            "coefficients_fitted": 0,
        },
        "magnitude_plausibility": {
            "comparison": (
                "per-frame unclaimed L1 ampere-turns against the absolute released "
                "ampere-turn current of every shipped poloidal conductor"
            ),
            "landed_median_unclaimed_ampere_turns": (
                LANDED_MEDIAN_UNCLAIMED_AMPERE_TURNS
            ),
            "landed_median_fraction_of_extracted_plasma_current": (
                LANDED_MEDIAN_UNCLAIMED_FRACTION
            ),
            "interpretation": "scale comparison only; it cannot establish origin alone",
        },
        "resolution_dependence": {
            "operation": (
                "take every second R and Z node, then repeat Delta-star and "
                "patch detection"
            ),
            "maximum_stable_relative_change": RESOLUTION_MAXIMUM_RELATIVE_CHANGE,
            "unchanged": [
                "label map values on retained nodes",
                "LCFS mask values on retained nodes",
                "tare-floor fraction",
                "patch connectivity and detectability rule",
            ],
        },
        "verdict": {
            "artefact": (
                "non-GS apparent current explains at least 80 percent and either no "
                "position-stable cluster exists or the decimated total or patch count "
                "changes by more than 20 percent"
            ),
            "physical": (
                "a position-stable cluster exists, non-GS apparent current explains "
                "at most 20 percent, and both resolution changes are at most 20 percent"
            ),
            "mixed": "neither the artefact nor physical conjunction is met",
        },
    }


def write_preregistration(output: Path) -> Path:
    """Persist the discriminator policy before reading scored artifacts."""

    output.mkdir(parents=True, exist_ok=True)
    path = output / PREREGISTRATION_NAME
    encoded = json.dumps(preregistration(), indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != encoded:
        raise RuntimeError("on-disk origin preregistration differs from policy")
    path.write_text(encoded)
    return path


def _distribution(values: list[float] | np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "minimum": float(np.min(array)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "q75": float(np.quantile(array, 0.75)),
        "maximum": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def cluster_centroids(
    coordinates: np.ndarray,
    radius_m: float = CLUSTER_RADIUS_M,
    minimum_patches: int = CLUSTER_MINIMUM_PATCHES,
) -> np.ndarray:
    """Return deterministic DBSCAN labels for two-dimensional coordinates."""

    coordinate = np.asarray(coordinates, dtype=float)
    if coordinate.ndim != 2 or coordinate.shape[1] != 2:
        raise ValueError("coordinates must have shape (patch, 2)")
    neighbourhood = cKDTree(coordinate).query_ball_point(coordinate, radius_m)
    labels = np.full(len(coordinate), -1, dtype=int)
    visited = np.zeros(len(coordinate), dtype=bool)
    cluster = 0
    for seed in range(len(coordinate)):
        if visited[seed]:
            continue
        visited[seed] = True
        adjacent = sorted(neighbourhood[seed])
        if len(adjacent) < minimum_patches:
            continue
        labels[seed] = cluster
        queue = deque(adjacent)
        queued = set(adjacent)
        while queue:
            point = queue.popleft()
            if not visited[point]:
                visited[point] = True
                neighbours = sorted(neighbourhood[point])
                if len(neighbours) >= minimum_patches:
                    for neighbour in neighbours:
                        if neighbour not in queued:
                            queued.add(neighbour)
                            queue.append(neighbour)
            if labels[point] < 0:
                labels[point] = cluster
        cluster += 1
    return labels


def position_summary(
    patch_records: list[dict[str, Any]], labels: np.ndarray
) -> dict[str, Any]:
    """Summarise cluster occupancy and physical-position qualification."""

    clusters = []
    for label in sorted(set(labels) - {-1}):
        indices = np.flatnonzero(labels == label)
        coordinates = np.asarray(
            [
                [
                    patch_records[index]["centroid_r_m"],
                    patch_records[index]["centroid_z_m"],
                ]
                for index in indices
            ]
        )
        shots = {patch_records[index]["shot"] for index in indices}
        frames = {
            (patch_records[index]["shot"], patch_records[index]["frame"])
            for index in indices
        }
        radial_spread = float(np.ptp(coordinates[:, 0]))
        vertical_spread = float(np.ptp(coordinates[:, 1]))
        clusters.append(
            {
                "cluster": int(label),
                "patches": int(len(indices)),
                "shots": int(len(shots)),
                "frames": int(len(frames)),
                "centroid_r_m": float(np.mean(coordinates[:, 0])),
                "centroid_z_m": float(np.mean(coordinates[:, 1])),
                "radial_standard_deviation_m": float(np.std(coordinates[:, 0])),
                "vertical_standard_deviation_m": float(np.std(coordinates[:, 1])),
                "radial_peak_to_peak_m": radial_spread,
                "vertical_peak_to_peak_m": vertical_spread,
                "radial_spread_in_native_cells": radial_spread / NATIVE_CELL_M,
                "vertical_spread_in_native_cells": vertical_spread / NATIVE_CELL_M,
                "physical_position_stable": bool(
                    len(shots) >= POSITION_MINIMUM_SHOTS
                    and radial_spread <= NATIVE_CELL_M
                    and vertical_spread <= NATIVE_CELL_M
                ),
            }
        )
    clusters.sort(key=lambda item: (-item["shots"], -item["patches"], item["cluster"]))
    return {
        "algorithm": "Euclidean DBSCAN",
        "patches": len(patch_records),
        "clustered_patches": int(np.count_nonzero(labels >= 0)),
        "noise_patches": int(np.count_nonzero(labels < 0)),
        "cluster_count": len(clusters),
        "native_cell_m": NATIVE_CELL_M,
        "position_stable_cluster_count": sum(
            item["physical_position_stable"] for item in clusters
        ),
        "largest_clusters": clusters[:12],
    }


def origin_verdict(
    non_gs_fraction: float,
    stable_cluster_count: int,
    total_relative_change: float,
    patch_count_relative_change: float,
) -> dict[str, Any]:
    """Apply the preregistered conjunctions without fitting a boundary."""

    resolution_stable = bool(
        abs(total_relative_change) <= RESOLUTION_MAXIMUM_RELATIVE_CHANGE
        and abs(patch_count_relative_change) <= RESOLUTION_MAXIMUM_RELATIVE_CHANGE
    )
    position_stable = stable_cluster_count > 0
    if non_gs_fraction >= NON_GS_DOMINANT_FRACTION and (
        not position_stable or not resolution_stable
    ):
        verdict = "artefact"
        carrier = "non-GS accounting"
    elif non_gs_fraction <= 0.20 and position_stable and resolution_stable:
        verdict = "physical"
        carrier = "position stability with resolution invariance"
    else:
        verdict = "mixed"
        carrier = "no preregistered conjunction is decisive"
    return {
        "verdict": verdict,
        "carrying_discriminator": carrier,
        "non_gs_explained_fraction": non_gs_fraction,
        "position_stable_cluster_count": stable_cluster_count,
        "resolution_total_relative_change": total_relative_change,
        "resolution_patch_count_relative_change": patch_count_relative_change,
        "resolution_stable": resolution_stable,
    }


def _source_patch_records(
    source: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[tuple[str, int], dict[str, Any]]]:
    records = source["records"]
    by_frame = {(item["shot"], int(item["frame"])): item for item in records}
    flat: list[dict[str, Any]] = []
    for item in records:
        for patch in item["patches"]:
            if patch["detectable_above_tare_floor"]:
                flat.append(
                    {
                        "shot": item["shot"],
                        "frame": int(item["frame"]),
                        **patch,
                    }
                )
    return flat, by_frame


def _sha256(path: Path) -> str:
    """Return the byte digest of one banked evidence input."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_stamp() -> dict[str, str]:
    """Return the clean committed source identity used by a measurement."""

    status = subprocess.run(
        ["git", "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise RuntimeError("cluster settlement requires a clean checkout")

    def revision(name: str) -> str:
        return subprocess.run(
            ["git", "rev-parse", name],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    return {"commit": revision("HEAD"), "tree": revision("HEAD^{tree}")}


def _lead_current_summary(
    patch_records: list[dict[str, Any]], labels: np.ndarray, cluster: int
) -> dict[str, Any]:
    """Report a lead's signed member-patch current on the banked scale."""

    indices = np.flatnonzero(labels == cluster)
    signed = np.asarray(
        [patch_records[index]["signed_current_a"] for index in indices], dtype=float
    )
    if not len(signed):
        raise RuntimeError(f"banked cluster {cluster} has no member patches")
    median = float(np.median(signed))
    return {
        "aggregation_for_fraction": "median signed current across member patches",
        "member_patch_count": int(len(signed)),
        "signed_current_a": _distribution(signed),
        "median_signed_current_a": median,
        "median_signed_current_fraction_of_unclaimed_median": (
            median / LANDED_MEDIAN_UNCLAIMED_AMPERE_TURNS
        ),
        "member_patch_signed_current_sum_a": float(np.sum(signed)),
        "unclaimed_current_median_a_turn": LANDED_MEDIAN_UNCLAIMED_AMPERE_TURNS,
    }


def _nearest_feature(point: np.ndarray, coordinates: np.ndarray) -> dict[str, Any]:
    """Return the nearest coordinate and its distance from one lead."""

    distance = np.linalg.norm(coordinates - point[None, :], axis=1)
    nearest = int(np.argmin(distance))
    return {
        "distance_m": float(distance[nearest]),
        "coordinate_rz_m": coordinates[nearest].tolist(),
        "within_one_native_cell": bool(distance[nearest] <= NATIVE_CELL_M),
    }


def _fixed_feature_summary(
    point: np.ndarray,
    radius: np.ndarray,
    height: np.ndarray,
    wall: np.ndarray,
    x_points: np.ndarray,
) -> dict[str, Any]:
    """Measure proximity to every declared fixed geometric feature."""

    radial_boundary = min(abs(point[0] - radius[0]), abs(point[0] - radius[-1]))
    vertical_boundary = min(abs(point[1] - height[0]), abs(point[1] - height[-1]))
    radial_node = float(np.min(np.abs(radius - point[0])))
    vertical_node = float(np.min(np.abs(height - point[1])))
    grid_boundary = min(radial_boundary, vertical_boundary)
    grid_node_line = min(radial_node, vertical_node)
    wall_vertex = _nearest_feature(point, wall)
    x_point_locus = _nearest_feature(point, x_points)
    grid = {
        "boundary_distance_m": float(grid_boundary),
        "radial_boundary_distance_m": float(radial_boundary),
        "vertical_boundary_distance_m": float(vertical_boundary),
        "nearest_node_line_distance_m": float(grid_node_line),
        "nearest_radial_node_line_distance_m": radial_node,
        "nearest_vertical_node_line_distance_m": vertical_node,
        "boundary_within_one_native_cell": bool(grid_boundary <= NATIVE_CELL_M),
        "node_line_within_one_native_cell": bool(grid_node_line <= NATIVE_CELL_M),
        "radial_extent_m": [float(radius[0]), float(radius[-1])],
        "vertical_extent_m": [float(height[0]), float(height[-1])],
    }
    fixed_coincidence = bool(
        wall_vertex["within_one_native_cell"]
        or grid["boundary_within_one_native_cell"]
        or grid["node_line_within_one_native_cell"]
        or x_point_locus["within_one_native_cell"]
    )
    return {
        "tolerance_m": NATIVE_CELL_M,
        "wall_vertex": wall_vertex,
        "efit_grid": grid,
        "frames_x_point_locus": {
            **x_point_locus,
            "frame_count": int(len(x_points)),
        },
        "coincides_with_any_declared_feature": fixed_coincidence,
    }


def _decimated_lead_matches(
    patch_records: list[dict[str, Any]], labels: np.ndarray
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    """Match banked leads to clusters after the registered grid decimation."""

    position = position_summary(patch_records, labels)
    clusters = position["largest_clusters"]
    matches: dict[int, dict[str, Any]] = {}
    for lead, centre_values in LEAD_CLUSTER_CENTRES_RZ_M.items():
        centre = np.asarray(centre_values, dtype=float)
        candidates = []
        for item in clusters:
            candidate = np.asarray(
                [item["centroid_r_m"], item["centroid_z_m"]], dtype=float
            )
            candidates.append((float(np.linalg.norm(candidate - centre)), item))
        if not candidates:
            matches[lead] = {
                "survives": False,
                "nearest_decimated_cluster": None,
                "distance_m": None,
                "reason": "no decimated detectable cluster exists",
            }
            continue
        distance, nearest = min(candidates, key=lambda item: item[0])
        survives = bool(
            distance <= CLUSTER_RADIUS_M and nearest["shots"] >= POSITION_MINIMUM_SHOTS
        )
        matches[lead] = {
            "survives": survives,
            "matching_rule": (
                "nearest decimated DBSCAN cluster within 0.042 m and spanning "
                "at least fifteen shots"
            ),
            "distance_m": distance,
            "distance_in_native_cells": distance / NATIVE_CELL_M,
            "nearest_decimated_cluster": nearest,
        }
    return matches, position


def settle_cluster_leads(
    data: Path, output: Path, *, workers: int = 1
) -> dict[str, Any]:
    """Settle the two banked position-stable leads without rewriting banked data."""

    stamp = _source_stamp()
    configure_dtypes()
    source_path = SOURCE_RECEIPT
    origin_path = DEFAULT_OUTPUT / RECEIPT_NAME
    input_digests = {
        str(origin_path): _sha256(origin_path),
        str(source_path): _sha256(source_path),
    }
    source = json.loads(source_path.read_text())
    origin = json.loads(origin_path.read_text())
    patch_records, source_by_frame = _source_patch_records(source)
    coordinates = np.asarray(
        [[item["centroid_r_m"], item["centroid_z_m"]] for item in patch_records]
    )
    labels = cluster_centroids(coordinates)
    for cluster, expected_values in LEAD_CLUSTER_CENTRES_RZ_M.items():
        members = coordinates[labels == cluster]
        measured = np.mean(members, axis=0)
        if not np.allclose(measured, expected_values, rtol=0.0, atol=1e-12):
            raise RuntimeError(f"banked cluster {cluster} centroid changed")
    banked_centres = {
        item["cluster"]: (item["centroid_r_m"], item["centroid_z_m"])
        for item in origin["position_stability"]["largest_clusters"]
    }
    for cluster, expected in LEAD_CLUSTER_CENTRES_RZ_M.items():
        if cluster not in banked_centres or not np.allclose(
            banked_centres[cluster], expected, rtol=0.0, atol=1e-12
        ):
            raise RuntimeError(f"origin receipt cluster {cluster} changed")

    affected = exact_tare.polarity_population()
    selected, limited_rows = exact_tare.select_frames(
        sorted(data.glob("*.parquet")),
        affected,
        SHOT_COUNT,
        FRAME_COUNT // SHOT_COUNT,
    )
    selected_keys = [(item.path.name, item.frame) for item in selected]
    if set(selected_keys) != set(source_by_frame):
        raise RuntimeError(
            "selected decimation cohort differs from banked patch cohort"
        )
    rows = {name: exact_tare._read(data / name) for name in limited_rows}
    first = rows[selected[0].path.name]
    radius, height = exact_tare.canonical_axes(first)
    mesh, geometry, width, vertical_extent = exact_tare.rectangular_geometry(
        radius, height
    )
    prepared = [
        exact_tare.prepare_frame(item, rows[item.path.name], radius, height)
        for item in selected
    ]
    source_mask = np.any(
        np.stack([item.participation_zr.reshape(-1) for item in prepared]), axis=0
    )
    source_indices = np.flatnonzero(source_mask & np.asarray(mesh.interior()))
    blocks = exact_tare.response_blocks(
        mesh, source_indices, width, vertical_extent, max(1, workers)
    )
    integrate = exact_tare.moment_integrator(mesh, geometry)
    with np.load(patches.VESSEL_ARTIFACT) as vessel:
        wall = np.asarray(vessel["limiter_contour_rz_m"], dtype=float)

    decimated_records: list[dict[str, Any]] = []
    x_points = []
    for prepared_frame in prepared:
        key = (prepared_frame.selected.path.name, prepared_frame.selected.frame)
        row = rows[key[0]]
        exact_vectors = integrate(
            prepared_frame.psi_norm_zr,
            prepared_frame.participation_zr,
            prepared_frame.profile_surface,
            prepared_frame.p_prime,
            prepared_frame.ff_prime,
        )
        exact_current, exact_radial, exact_vertical, _boundary = (
            np.asarray(value) for value in jax.block_until_ready(exact_vectors)
        )
        exact_flux_zr = (
            blocks[0] @ exact_current[source_indices]
            + blocks[1] @ exact_radial[source_indices]
            + blocks[2] @ exact_vertical[source_indices]
        ).reshape(prepared_frame.label_total_zr.shape)
        tared_total_zr = prepared_frame.label_total_zr - exact_flux_zr
        decimated_radius = radius[::2]
        decimated_height = height[::2]
        decimated_core = prepared_frame.core_rz[::2, ::2]
        decimated_delta = apply_delta_star(
            decimated_radius, decimated_height, tared_total_zr[::2, ::2].T
        )
        decimated_density = np.asarray(decimated_delta.toroidal_current_density)
        decimated_exterior = (
            ~decimated_core & decimated_delta.valid & np.isfinite(decimated_density)
        )
        detected, _metrics, _masks = patches.locate_patches(
            decimated_density,
            decimated_exterior,
            decimated_core,
            decimated_radius,
            decimated_height,
            wall,
            float(np.sum(exact_current)),
        )
        decimated_records.extend(
            {
                "shot": key[0],
                "frame": key[1],
                **item,
            }
            for item in detected
            if item["detectable_above_tare_floor"]
        )
        count = int(row["efit_lcfs_n"][key[1]])
        boundary = np.column_stack(
            (
                np.asarray(row["efit_lcfs_r"][key[1]][:count], dtype=float),
                np.asarray(row["efit_lcfs_z"][key[1]][:count], dtype=float),
            )
        )
        x_points.append(
            boundary_gradient_minimum(
                radius,
                height,
                np.asarray(row["efit_psirz"][key[1]], dtype=float),
                boundary,
            )
        )

    decimated_coordinates = np.asarray(
        [[item["centroid_r_m"], item["centroid_z_m"]] for item in decimated_records]
    )
    decimated_labels = cluster_centroids(decimated_coordinates)
    decimated_matches, decimated_position = _decimated_lead_matches(
        decimated_records, decimated_labels
    )
    x_point_array = np.asarray(x_points, dtype=float)
    leads = []
    for cluster, centre_values in LEAD_CLUSTER_CENTRES_RZ_M.items():
        centre = np.asarray(centre_values, dtype=float)
        leads.append(
            {
                "banked_cluster": cluster,
                "centroid_rz_m": centre.tolist(),
                "signed_current": _lead_current_summary(patch_records, labels, cluster),
                "fixed_geometry": _fixed_feature_summary(
                    centre, radius, height, wall, x_point_array
                ),
                "factor_two_decimation": decimated_matches[cluster],
            }
        )
    surviving = sum(item["factor_two_decimation"]["survives"] for item in leads)
    feature_coincidences = sum(
        item["fixed_geometry"]["coincides_with_any_declared_feature"] for item in leads
    )
    confirmed = surviving == len(leads) and feature_coincidences == 0
    receipt = {
        "source_stamp": stamp,
        "evidence_input_sha256": input_digests,
        "evidence_inputs_remained_byte_identical": bool(
            input_digests[str(origin_path)] == _sha256(origin_path)
            and input_digests[str(source_path)] == _sha256(source_path)
        ),
        "native_cell_m": NATIVE_CELL_M,
        "wall": {
            "authored_vertex_count": int(len(wall)),
            "outer_extent_r_m": float(np.max(wall[:, 0])),
        },
        "grid": {
            "native_shape_rz": [int(len(radius)), int(len(height))],
            "decimated_shape_rz": [int(len(radius[::2])), int(len(height[::2]))],
        },
        "lead_count": len(leads),
        "leads": leads,
        "decimated_population": decimated_position,
        "verdict": {
            "word": "lead-confirmed" if confirmed else "lead-dismissed",
            "rule": (
                "lead-confirmed only when both banked leads match a decimated "
                "cross-shot cluster and neither lies within one native cell of a "
                "declared fixed geometric feature"
            ),
            "surviving_lead_count": surviving,
            "both_leads_survive": surviving == len(leads),
            "fixed_feature_coincidence_count": feature_coincidences,
            "discriminating_number": {
                "name": "factor_two_decimation_surviving_lead_count",
                "value": surviving,
                "required_for_confirmation": len(leads),
            },
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    receipt_path = output / SETTLEMENT_RECEIPT_NAME
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    if not receipt["evidence_inputs_remained_byte_identical"]:
        raise RuntimeError("a banked evidence receipt changed during settlement")
    return receipt


def _magnitude_summary(
    records: list[dict[str, Any]], names: tuple[str, ...]
) -> dict[str, Any]:
    current = np.asarray([record["shipped_ampere_turns"] for record in records])
    absolute = np.abs(current)
    unclaimed = np.asarray(
        [record["native_unclaimed_ampere_turns_l1"] for record in records]
    )
    per_conductor = {
        name: {
            **_distribution(absolute[:, index]),
            "signed_median_ampere_turns": float(np.median(current[:, index])),
        }
        for index, name in enumerate(names)
    }
    maximum = np.max(absolute, axis=1)
    sum_l1 = np.sum(absolute, axis=1)
    return {
        "released_conductor_count": len(names),
        "released_conductor_names": list(names),
        "unclaimed_ampere_turns_l1": _distribution(unclaimed),
        "per_frame_maximum_absolute_released_ampere_turns": _distribution(maximum),
        "per_frame_sum_absolute_released_ampere_turns": _distribution(sum_l1),
        "median_unclaimed_to_median_frame_maximum_ratio": float(
            np.median(unclaimed) / np.median(maximum)
        ),
        "median_unclaimed_to_median_frame_sum_ratio": float(
            np.median(unclaimed) / np.median(sum_l1)
        ),
        "per_conductor_absolute_ampere_turns": per_conductor,
        "landed_fraction_of_extracted_plasma_current": LANDED_MEDIAN_UNCLAIMED_FRACTION,
        "interpretation": (
            "the comparison establishes machine-current scale only; comparable "
            "magnitude is necessary but not sufficient evidence of physical current"
        ),
    }


def _render(
    patch_records: list[dict[str, Any]],
    labels: np.ndarray,
    position: dict[str, Any],
    records: list[dict[str, Any]],
    output: Path,
) -> Path:
    """Render the four preregistered discriminators."""

    figure, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    coordinates = np.asarray(
        [[item["centroid_r_m"], item["centroid_z_m"]] for item in patch_records]
    )
    axes[0, 0].scatter(
        coordinates[:, 0], coordinates[:, 1], c=labels, cmap="tab20", s=14, alpha=0.75
    )
    axes[0, 0].set_xlabel("R [m]")
    axes[0, 0].set_ylabel("Z [m]")
    axes[0, 0].set_title("Detectable patch centroid clusters")

    largest = position["largest_clusters"][:8]
    locations = np.arange(len(largest))
    axes[0, 1].bar(
        locations - 0.18,
        [item["radial_peak_to_peak_m"] for item in largest],
        width=0.36,
        label="R spread",
    )
    axes[0, 1].bar(
        locations + 0.18,
        [item["vertical_peak_to_peak_m"] for item in largest],
        width=0.36,
        label="Z spread",
    )
    axes[0, 1].axhline(NATIVE_CELL_M, color="black", ls="--", label="0.021 m cell")
    axes[0, 1].set_xticks(locations, [str(item["cluster"]) for item in largest])
    axes[0, 1].set_ylabel("peak-to-peak spread [m]")
    axes[0, 1].set_xlabel("cluster")
    axes[0, 1].set_title("Largest cross-shot cluster stability")
    axes[0, 1].legend()

    native = np.asarray([item["native_unclaimed_ampere_turns_l1"] for item in records])
    non_gs = np.asarray([item["non_gs_apparent_ampere_turns_l1"] for item in records])
    limit = float(max(np.max(native), np.max(non_gs)))
    axes[1, 0].scatter(native / 1000.0, non_gs / 1000.0, s=20, alpha=0.75)
    axes[1, 0].plot([0.0, limit / 1000.0], [0.0, limit / 1000.0], "k--", lw=1)
    axes[1, 0].set_xlabel("native unclaimed current [kA-turn]")
    axes[1, 0].set_ylabel("non-GS apparent current [kA-turn]")
    axes[1, 0].set_title("Label inconsistency accounting")

    decimated = np.asarray(
        [item["decimated_unclaimed_ampere_turns_l1"] for item in records]
    )
    axes[1, 1].scatter(native / 1000.0, decimated / 1000.0, s=20, alpha=0.75)
    axes[1, 1].plot([0.0, limit / 1000.0], [0.0, limit / 1000.0], "k--", lw=1)
    axes[1, 1].set_xlabel("native unclaimed current [kA-turn]")
    axes[1, 1].set_ylabel("factor-two decimated current [kA-turn]")
    axes[1, 1].set_title("Resolution dependence")
    path = output / FIGURE_NAME
    figure.savefig(path, dpi=170)
    plt.close(figure)
    return path


def run(data: Path, output: Path, *, workers: int = 1) -> dict[str, Any]:
    """Execute the fixed-cohort origin discrimination."""

    preregistration_path = write_preregistration(output)
    preregistration_digest = hashlib.sha256(
        preregistration_path.read_bytes()
    ).hexdigest()
    configure_dtypes()
    source = json.loads(SOURCE_RECEIPT.read_text())
    selection = source["selection"]
    if selection["frames"] != FRAME_COUNT or selection["shots"] != SHOT_COUNT:
        raise RuntimeError(
            "landed exterior-current cohort is not 60 frames over 20 shots"
        )
    if not selection["all_selected_absent_from_polarity_population"]:
        raise RuntimeError("landed exterior-current cohort contains a polarity shot")
    patch_records, source_by_frame = _source_patch_records(source)
    if len(patch_records) != DETECTABLE_PATCH_COUNT:
        raise RuntimeError(
            "landed exterior-current cohort does not contain 391 detectable patches"
        )
    source_median = source["all_frames"]["total_unclaimed_ampere_turns_l1"]["median"]
    if not np.isclose(
        source_median, LANDED_MEDIAN_UNCLAIMED_AMPERE_TURNS, rtol=0.0, atol=1e-6
    ):
        raise RuntimeError("landed unclaimed-current median changed")

    coordinates = np.asarray(
        [[item["centroid_r_m"], item["centroid_z_m"]] for item in patch_records]
    )
    labels = cluster_centroids(coordinates)
    position = position_summary(patch_records, labels)

    affected = exact_tare.polarity_population()
    selected, limited_rows = exact_tare.select_frames(
        sorted(data.glob("*.parquet")), affected, SHOT_COUNT, FRAME_COUNT // SHOT_COUNT
    )
    rows = {name: patches._read(data / name) for name in limited_rows}
    first = rows[selected[0].path.name]
    radius, height = exact_tare.canonical_axes(first)
    mesh, geometry, width, vertical_extent = exact_tare.rectangular_geometry(
        radius, height
    )
    prepared = [
        exact_tare.prepare_frame(item, rows[item.path.name], radius, height)
        for item in selected
    ]
    source_mask = np.any(
        np.stack([item.participation_zr.reshape(-1) for item in prepared]), axis=0
    )
    source_indices = np.flatnonzero(source_mask & np.asarray(mesh.interior()))
    blocks = exact_tare.response_blocks(
        mesh, source_indices, width, vertical_extent, max(1, workers)
    )
    integrate = exact_tare.moment_integrator(mesh, geometry)
    operator = _operator(radius, height)
    registry = DiiidDescriptionRegistry()
    response_cache: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
    with np.load(patches.VESSEL_ARTIFACT) as vessel:
        wall = np.asarray(vessel["limiter_contour_rz_m"], dtype=float)
    checkpoint = output / CHECKPOINT_NAME
    checkpoint.write_text("")
    records: list[dict[str, Any]] = []
    conductor_names: tuple[str, ...] | None = None
    for prepared_frame in prepared:
        key = (prepared_frame.selected.path.name, prepared_frame.selected.frame)
        if key not in source_by_frame:
            raise RuntimeError(f"selected frame {key!r} is absent from landed receipt")
        source_record = source_by_frame[key]
        row = rows[prepared_frame.selected.path.name]

        exact_vectors = integrate(
            prepared_frame.psi_norm_zr,
            prepared_frame.participation_zr,
            prepared_frame.profile_surface,
            prepared_frame.p_prime,
            prepared_frame.ff_prime,
        )
        exact_current, exact_radial, exact_vertical, _boundary = (
            np.asarray(value) for value in jax.block_until_ready(exact_vectors)
        )
        exact_flux_zr = (
            blocks[0] @ exact_current[source_indices]
            + blocks[1] @ exact_radial[source_indices]
            + blocks[2] @ exact_vertical[source_indices]
        ).reshape(prepared_frame.label_total_zr.shape)
        tared_total_zr = prepared_frame.label_total_zr - exact_flux_zr

        profile_source, reliable, label_total_rz = _profile_source(
            row, prepared_frame.selected.frame, radius, height
        )
        fixed_solution_rz = operator.solve(profile_source, label_total_rz)
        strict_non_gs_rz = label_total_rz - fixed_solution_rz
        non_gs_delta = apply_delta_star(radius, height, strict_non_gs_rz)
        non_gs_density = np.asarray(non_gs_delta.toroidal_current_density)
        non_gs_exterior = (
            ~prepared_frame.core_rz & non_gs_delta.valid & np.isfinite(non_gs_density)
        )
        native_area = float(np.mean(np.diff(radius)) * np.mean(np.diff(height)))
        non_gs_l1 = float(np.sum(np.abs(non_gs_density[non_gs_exterior])) * native_area)

        decimated_radius = radius[::2]
        decimated_height = height[::2]
        decimated_core = prepared_frame.core_rz[::2, ::2]
        decimated_delta = apply_delta_star(
            decimated_radius, decimated_height, tared_total_zr[::2, ::2].T
        )
        decimated_density = np.asarray(decimated_delta.toroidal_current_density)
        decimated_exterior = (
            ~decimated_core & decimated_delta.valid & np.isfinite(decimated_density)
        )
        decimated_patches, decimated_metrics, _masks = patches.locate_patches(
            decimated_density,
            decimated_exterior,
            decimated_core,
            decimated_radius,
            decimated_height,
            wall,
            float(np.sum(exact_current)),
        )
        decimated_detectable = sum(
            item["detectable_above_tare_floor"] for item in decimated_patches
        )

        description = registry.ingest(row, source_row=prepared_frame.selected.path.name)
        response = response_cache.get(description.physical_digest)
        if response is None:
            response = vacuum_response(
                description, row["efit_grid_R"], row["efit_grid_Z"]
            )
            response_cache[description.physical_digest] = response
        names, _matrix = response
        if conductor_names is None:
            conductor_names = names
        elif conductor_names != names:
            raise RuntimeError("released conductor ordering changed within the cohort")
        current = _current_vector(
            row, description, names, prepared_frame.selected.time_ms
        )
        record = {
            "shot": key[0],
            "frame": key[1],
            "time_ms": prepared_frame.selected.time_ms,
            "reliable_profile_surfaces": reliable,
            "native_unclaimed_ampere_turns_l1": source_record[
                "total_unclaimed_ampere_turns_l1"
            ],
            "native_detectable_patch_count": sum(
                item["detectable_above_tare_floor"] for item in source_record["patches"]
            ),
            "non_gs_apparent_ampere_turns_l1": non_gs_l1,
            "non_gs_to_native_unclaimed_ratio": (
                non_gs_l1 / source_record["total_unclaimed_ampere_turns_l1"]
            ),
            "decimated_unclaimed_ampere_turns_l1": decimated_metrics[
                "total_unclaimed_ampere_turns_l1"
            ],
            "decimated_detectable_patch_count": decimated_detectable,
            "shipped_ampere_turns": current.tolist(),
        }
        records.append(record)
        with checkpoint.open("a") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")

    if conductor_names is None:
        raise RuntimeError("no released conductor currents were read")
    native_total = np.asarray(
        [item["native_unclaimed_ampere_turns_l1"] for item in records]
    )
    decimated_total = np.asarray(
        [item["decimated_unclaimed_ampere_turns_l1"] for item in records]
    )
    native_patch_count = int(
        sum(item["native_detectable_patch_count"] for item in records)
    )
    decimated_patch_count = int(
        sum(item["decimated_detectable_patch_count"] for item in records)
    )
    non_gs_current = np.asarray(
        [item["non_gs_apparent_ampere_turns_l1"] for item in records]
    )
    non_gs_fraction = float(np.median(non_gs_current) / np.median(native_total))
    total_relative_change = float(
        (np.median(decimated_total) - np.median(native_total)) / np.median(native_total)
    )
    patch_count_relative_change = float(
        (decimated_patch_count - native_patch_count) / native_patch_count
    )
    resolution = {
        "native_grid_shape": [len(radius), len(height)],
        "decimated_grid_shape": [len(radius[::2]), len(height[::2])],
        "native_radial_step_m": float(np.mean(np.diff(radius))),
        "native_vertical_step_m": float(np.mean(np.diff(height))),
        "native_unclaimed_ampere_turns_l1": _distribution(native_total),
        "decimated_unclaimed_ampere_turns_l1": _distribution(decimated_total),
        "per_frame_decimated_to_native_ratio": _distribution(
            decimated_total / native_total
        ),
        "median_total_relative_change": total_relative_change,
        "native_detectable_patch_count": native_patch_count,
        "decimated_detectable_patch_count": decimated_patch_count,
        "detectable_patch_count_relative_change": patch_count_relative_change,
    }
    non_gs = {
        "landed_irreducible_strict_gs_residual_fraction": NON_GS_LABEL_CONTENT_FRACTION,
        "apparent_exterior_ampere_turns_l1": _distribution(non_gs_current),
        "per_frame_apparent_to_unclaimed_ratio": _distribution(
            non_gs_current / native_total
        ),
        "median_apparent_fraction_of_landed_median_unclaimed": non_gs_fraction,
        "landed_median_unclaimed_ampere_turns": LANDED_MEDIAN_UNCLAIMED_AMPERE_TURNS,
        "construction": (
            "Delta-star of label minus fixed-border extracted-profile solution"
        ),
    }
    verdict = origin_verdict(
        non_gs_fraction,
        position["position_stable_cluster_count"],
        total_relative_change,
        patch_count_relative_change,
    )
    figure = _render(patch_records, labels, position, records, output)
    receipt = {
        "preregistration": preregistration(),
        "preregistration_sha256": preregistration_digest,
        "selection": {
            "frames": len(records),
            "shots": len({item["shot"] for item in records}),
            "detectable_patches": len(patch_records),
            "polarity_population_count": len(affected),
            "all_selected_absent_from_polarity_population": all(
                item.path.name not in affected for item in selected
            ),
        },
        "position_stability": position,
        "non_gs_accounting": non_gs,
        "magnitude_plausibility": _magnitude_summary(records, conductor_names),
        "resolution_dependence": resolution,
        "verdict": verdict,
        "records": records,
        "artifacts": {
            "preregistration": str(preregistration_path),
            "receipt": str(output / RECEIPT_NAME),
            "incremental_frame_checkpoint": str(checkpoint),
            "figure": str(figure),
            "source_patch_receipt": str(SOURCE_RECEIPT),
        },
    }
    receipt_path = output / RECEIPT_NAME
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    if len(records) != FRAME_COUNT or receipt["selection"]["shots"] != SHOT_COUNT:
        raise RuntimeError("origin measurement did not complete the 60-frame cohort")
    if native_patch_count != DETECTABLE_PATCH_COUNT:
        raise RuntimeError(
            "origin measurement did not retain all 391 detectable patches"
        )
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--settle-cluster-leads",
        action="store_true",
        help="settle the two banked position-stable leads without rewriting inputs",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    if arguments.settle_cluster_leads:
        receipt = settle_cluster_leads(
            arguments.data, arguments.output, workers=arguments.workers
        )
        summary = {
            "lead_count": receipt["lead_count"],
            "verdict": receipt["verdict"],
        }
    else:
        receipt = run(arguments.data, arguments.output, workers=arguments.workers)
        summary = {"selection": receipt["selection"], "verdict": receipt["verdict"]}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
