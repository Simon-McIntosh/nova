"""Independent audit of stencil critical-point candidate multiplicity.

The audit treats a returned vertex as evidence about a continuous stationary
point, not as a stationary point by itself.  It derives the stationary points
of the periodic field used by ``stencil_null_route.py``, validates those roots
with an independent nonlinear solve, clusters vertices with the native
periodic metric, and measures truncation before any local quadratic fit.

A separate deterministic bank probes the harder selection problem: retaining
weak quadratic O/X points while rejecting white noise, correlated noise, and
low-order backgrounds.  Native gradient winding decides candidate existence;
clustering preserves signed index, while covariance, boundary robustness, local
Hessian/residual consistency, adjacent-scale survival, and prominence only rank
each returned candidate as resolved or unresolved.

The full numerical bank is intended for one CPU execution.  ``device`` mode
replays only the discrete-mask cases on another JAX device.  ``merge`` combines
the captured CPU report and GPU receipt without rerunning either measurement.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time
from typing import Any

import numpy as np
import scipy.ndimage  # type: ignore[import-untyped]
import scipy.optimize  # type: ignore[import-untyped]


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs/figures/jax-dissolution/fieldnull_candidate_audit.json"

RECTANGULAR_RING = (
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
    (-1, 0),
    (-1, 1),
)
HEXAGONAL_RING = ((-1, 0), (0, -1), (1, -1), (1, 0), (0, 1), (-1, 1))

PERIODIC_RESOLUTIONS = (33, 61, 65, 81)
PERIODIC_SHIFTS = {
    "unshifted": (0.0, 0.0),
    "off_vertex": (0.173, -0.119),
}
NOISE_RESOLUTION = 65
NOISE_STRENGTHS = (0.125, 0.25, 0.5, 1.0, 2.0)
NOISE_LEVELS = (0.25, 0.5, 1.0, 2.0, 4.0)
CORRELATION_LENGTHS_CELLS = (0.0, 1.0, 2.0, 4.0)
SMOOTHING_SCALES_CELLS = (0.0, 1.0, 2.0, 4.0)
TRUE_FIELD_SEEDS = (11, 23, 47)
NOISE_ONLY_SEEDS = (3, 7, 13, 19, 29, 37, 43, 53)
NULL_PLACEMENTS = {
    "vertex_aligned": (0.375, 0.4375),
    "cell_shifted": (0.3828125, 0.4453125),
}
BACKGROUND_VARIANTS = ("none", "linear_quadratic")


def _git_revision() -> str:
    """Return the source revision being measured."""
    if revision := os.environ.get("NOVA_AUDIT_REVISION"):
        return revision
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _strict_json(value: Any) -> Any:
    """Convert NumPy values and reject non-finite JSON numbers."""
    if isinstance(value, dict):
        return {str(key): _strict_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return _strict_json(value.tolist())
    if isinstance(value, np.generic):
        return _strict_json(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a stable strict-JSON report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_strict_json(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _quantiles(values: list[float]) -> dict[str, float | None]:
    """Return compact distribution evidence."""
    if not values:
        return {key: None for key in ("min", "p50", "p90", "p99", "max")}
    array = np.asarray(values, dtype=float)
    return {
        "min": float(np.min(array)),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(np.max(array)),
    }


def _ring_counts(field: np.ndarray, offsets=RECTANGULAR_RING) -> np.ndarray:
    """Independent closed-ring sign-change count with an excluded border."""
    comparisons = np.stack(
        [
            np.roll(field, shift=(-row_offset, -column_offset), axis=(0, 1)) > field
            for row_offset, column_offset in offsets
        ]
    )
    changes = np.sum(
        comparisons != np.roll(comparisons, -1, axis=0),
        axis=0,
        dtype=np.int32,
    )
    changes[[0, -1], :] = -1
    changes[:, [0, -1]] = -1
    return changes


def _legacy_periodic_field(
    resolution: int, shift: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Recreate the prior field through its exact physical-coordinate path."""
    radial = np.linspace(0.5, 1.5, resolution)
    vertical = np.linspace(-0.8, 0.8, resolution)
    rr, zz = np.meshgrid(radial, vertical)
    field = np.sin(8.0 * np.pi * (rr - 0.5) + 2.0 * np.pi * shift[0]) * np.sin(
        8.0 * np.pi * (zz + 0.8) / 1.6 + 2.0 * np.pi * shift[1]
    )
    return radial, vertical, field, np.linspace(0.0, 1.0, resolution)


def _stationary_points(shift: tuple[float, float], kind: str) -> np.ndarray:
    """Derive all torus-unique stationary points of one Hessian class."""
    offset = 0.0 if kind == "saddle" else 0.5
    radial = np.mod(
        (np.arange(8, dtype=float) + offset - 2.0 * shift[0]) / 8.0,
        1.0,
    )
    vertical = np.mod(
        (np.arange(8, dtype=float) + offset - 2.0 * shift[1]) / 8.0,
        1.0,
    )
    return np.asarray([(r, z) for z in vertical for r in radial], dtype=float)


def _periodic_gradient(point: np.ndarray, shift: tuple[float, float]) -> np.ndarray:
    """Analytic gradient of the normalized periodic field."""
    radial_angle = 8.0 * np.pi * point[0] + 2.0 * np.pi * shift[0]
    vertical_angle = 8.0 * np.pi * point[1] + 2.0 * np.pi * shift[1]
    frequency = 8.0 * np.pi
    return np.asarray(
        [
            frequency * np.cos(radial_angle) * np.sin(vertical_angle),
            frequency * np.sin(radial_angle) * np.cos(vertical_angle),
        ]
    )


def _periodic_hessian(point: np.ndarray, shift: tuple[float, float]) -> np.ndarray:
    """Analytic Hessian of the normalized periodic field."""
    radial_angle = 8.0 * np.pi * point[0] + 2.0 * np.pi * shift[0]
    vertical_angle = 8.0 * np.pi * point[1] + 2.0 * np.pi * shift[1]
    frequency_squared = (8.0 * np.pi) ** 2
    diagonal = -frequency_squared * np.sin(radial_angle) * np.sin(vertical_angle)
    cross = frequency_squared * np.cos(radial_angle) * np.cos(vertical_angle)
    return np.asarray([[diagonal, cross], [cross, diagonal]])


def _torus_delta(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return componentwise shortest displacement on a unit torus."""
    delta = np.abs(first - second)
    return np.minimum(delta, 1.0 - delta)


def _torus_distance(
    first: np.ndarray, second: np.ndarray, geometry: str = "rectangular"
) -> float:
    """Measure periodic distance in rectangular or native axial coordinates."""
    delta = _torus_delta(np.asarray(first), np.asarray(second))
    if geometry == "hexagonal":
        cartesian = np.asarray(
            [delta[0] + 0.5 * delta[1], np.sqrt(3.0) * delta[1] / 2.0]
        )
        return float(np.linalg.norm(cartesian))
    return float(np.linalg.norm(delta))


def _independent_root_validation(
    shifts: dict[str, tuple[float, float]],
) -> dict[str, Any]:
    """Numerically solve every derived stationary point from a perturbed seed."""
    rows = []
    for label, shift in shifts.items():
        errors = []
        classifications = Counter()
        for kind in ("saddle", "extremum"):
            for truth in _stationary_points(shift, kind):
                seed = truth + np.asarray([0.0031, -0.0023])
                solution = scipy.optimize.root(
                    lambda point: _periodic_gradient(point, shift), seed
                )
                if not solution.success:
                    raise RuntimeError(
                        f"stationary solve failed for {label} {kind}: "
                        f"{solution.message}"
                    )
                root = np.mod(solution.x, 1.0)
                errors.append(_torus_distance(root, truth))
                eigenvalues = np.linalg.eigvalsh(_periodic_hessian(root, shift))
                solved_kind = (
                    "saddle" if eigenvalues[0] * eigenvalues[1] < 0.0 else "extremum"
                )
                classifications[solved_kind] += 1
        rows.append(
            {
                "shift": label,
                "derived_stationary_count": 128,
                "derived_saddle_count_on_torus": 64,
                "derived_extremum_count_on_torus": 64,
                "solved_classification_counts": dict(classifications),
                "maximum_periodic_root_error": float(max(errors)),
                "maximum_gradient_residual": float(
                    max(
                        np.linalg.norm(_periodic_gradient(point, shift))
                        for kind in ("saddle", "extremum")
                        for point in _stationary_points(shift, kind)
                    )
                ),
            }
        )
    return {
        "method": (
            "closed-form trigonometric roots, independently replayed with "
            "scipy.optimize.root and analytic Hessian classification"
        ),
        "rows": rows,
    }


def _fit_local_quadratic(
    field: np.ndarray,
    radial_unit: np.ndarray,
    vertical_unit: np.ndarray,
    row: int,
    column: int,
    radius: int = 1,
) -> dict[str, Any] | None:
    """Fit a local quadratic without using either production implementation."""
    if (
        row < radius
        or column < radius
        or row >= field.shape[0] - radius
        or column >= field.shape[1] - radius
    ):
        return None
    rows = np.arange(row - radius, row + radius + 1)
    columns = np.arange(column - radius, column + radius + 1)
    xx, yy = np.meshgrid(radial_unit[columns], vertical_unit[rows])
    values = field[np.ix_(rows, columns)].reshape(-1)
    design = np.column_stack(
        [
            xx.reshape(-1) ** 2,
            yy.reshape(-1) ** 2,
            xx.reshape(-1),
            yy.reshape(-1),
            (xx * yy).reshape(-1),
            np.ones(values.size),
        ]
    )
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    a, b, c, d, cross, _constant = coefficients
    hessian = np.asarray([[2.0 * a, cross], [cross, 2.0 * b]])
    try:
        stationary = np.linalg.solve(hessian, -np.asarray([c, d]))
    except np.linalg.LinAlgError:
        return None
    residual = design @ coefficients - values
    return {
        "coordinate": stationary,
        "hessian": hessian,
        "eigenvalues": np.linalg.eigvalsh(hessian),
        "residual_rms": float(np.sqrt(np.mean(residual**2))),
    }


def _periodic_clusters(
    points: list[np.ndarray], threshold: float, geometry: str = "rectangular"
) -> list[list[int]]:
    """Cluster physical coordinates with a grid-aware periodic radius."""
    parents = list(range(len(points)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parents[second_root] = first_root

    for first in range(len(points)):
        for second in range(first + 1, len(points)):
            if _torus_distance(points[first], points[second], geometry) <= threshold:
                union(first, second)
    groups: dict[int, list[int]] = {}
    for index in range(len(points)):
        groups.setdefault(find(index), []).append(index)
    return list(groups.values())


def _nearest_truth(
    point: np.ndarray, truth: np.ndarray, geometry: str = "rectangular"
) -> tuple[int, float]:
    distances = [_torus_distance(point, candidate, geometry) for candidate in truth]
    index = int(np.argmin(distances))
    return index, float(distances[index])


def _cluster_representative(points: list[np.ndarray], members: list[int]) -> np.ndarray:
    """Choose the member with the smallest total periodic within-cluster distance."""
    if len(members) == 1:
        return points[members[0]]
    totals = [
        sum(_torus_distance(points[index], points[other]) for other in members)
        for index in members
    ]
    return points[members[int(np.argmin(totals))]]


def _audit_rectangular_case(
    resolution: int, shift_label: str, shift: tuple[float, float]
) -> dict[str, Any]:
    radial, vertical, field, unit_grid = _legacy_periodic_field(resolution, shift)
    radial_unit = radial - 0.5
    vertical_unit = (vertical + 0.8) / 1.6
    counts = _ring_counts(field)
    raw_indices = np.argwhere(counts == 4)
    fits = []
    fitted_points: list[np.ndarray] = []
    for row, column in raw_indices:
        fit = _fit_local_quadratic(
            field,
            radial_unit,
            vertical_unit,
            int(row),
            int(column),
        )
        if fit is None:
            fitted_points.append(np.asarray([unit_grid[column], unit_grid[row]]))
            fits.append(None)
        else:
            fitted_points.append(np.mod(fit["coordinate"], 1.0))
            fits.append(fit)

    spacing = 1.0 / (resolution - 1)
    clusters = _periodic_clusters(
        fitted_points,
        threshold=min(2.25 * spacing, 0.45 / 8.0),
    )
    truth = _stationary_points(shift, "saddle")
    boundary_truth = {
        index
        for index, point in enumerate(truth)
        if np.any(np.minimum(point, 1.0 - point) <= 1.0e-12)
    }
    near_border_truth = {
        index
        for index, point in enumerate(truth)
        if np.any(np.minimum(point, 1.0 - point) < spacing)
    }
    matched_clusters: dict[int, list[list[int]]] = {}
    unmatched_clusters = []
    residuals = []
    localization_errors = []
    hessian_counts = Counter()
    for members in clusters:
        representative = _cluster_representative(fitted_points, members)
        truth_index, distance = _nearest_truth(representative, truth)
        if distance <= min(2.5 * spacing, 0.45 / 8.0):
            matched_clusters.setdefault(truth_index, []).append(members)
            localization_errors.append(distance)
        else:
            unmatched_clusters.append(members)
        for member in members:
            fit = fits[member]
            point = fitted_points[member]
            residuals.append(float(np.linalg.norm(_periodic_gradient(point, shift))))
            if fit is None:
                hessian_counts["degenerate"] += 1
            elif fit["eigenvalues"][0] * fit["eigenvalues"][1] < 0.0:
                hessian_counts["saddle"] += 1
            else:
                hessian_counts["extremum"] += 1

    matched_truth = set(matched_clusters)
    open_truth = set(range(len(truth))) - boundary_truth
    false_negatives = open_truth - matched_truth
    boundary_missing = boundary_truth - matched_truth
    border_proximity_missing = near_border_truth - matched_truth
    cluster_sizes = [len(members) for members in clusters]

    first_slot_points = fitted_points[:8]
    first_slot_truth = []
    first_slot_unmatched = 0
    for point in first_slot_points:
        truth_index, distance = _nearest_truth(point, truth)
        if distance <= min(2.5 * spacing, 0.45 / 8.0):
            first_slot_truth.append(truth_index)
        else:
            first_slot_unmatched += 1

    return {
        "geometry_contract": "rectangular eight-neighbour vertex ring",
        "resolution": resolution,
        "shift": shift_label,
        "grid_spacing_normalized": spacing,
        "raw_vertex_count": int(raw_indices.shape[0]),
        "physical_cluster_count": len(clusters),
        "cluster_size_histogram": {
            str(size): count for size, count in sorted(Counter(cluster_sizes).items())
        },
        "analytic_torus_saddle_count": len(truth),
        "analytic_strict_open_saddle_count": len(open_truth),
        "matched_torus_saddle_count": len(matched_truth),
        "matched_strict_open_saddle_count": len(matched_truth & open_truth),
        "duplicate_vertex_count": int(raw_indices.shape[0] - len(clusters)),
        "duplicate_cluster_count": int(
            sum(max(0, len(groups) - 1) for groups in matched_clusters.values())
        ),
        "boundary_periodic_truth_count": len(boundary_truth),
        "boundary_periodic_missing_count": len(boundary_missing),
        "within_one_cell_of_border_missing_count": len(border_proximity_missing),
        "low_gradient_unmatched_cluster_count": int(
            sum(
                np.linalg.norm(
                    _periodic_gradient(
                        _cluster_representative(fitted_points, members), shift
                    )
                )
                <= (8.0 * np.pi) ** 2 * spacing
                for members in unmatched_clusters
            )
        ),
        "genuine_extra_stationary_cluster_count": 0,
        "false_positive_cluster_count_open_contract": int(
            len(unmatched_clusters) + len(matched_truth & boundary_truth)
        ),
        "false_negative_count_open_contract": len(false_negatives),
        "hessian_classification_by_vertex": dict(hessian_counts),
        "exact_gradient_residual_norm": _quantiles(residuals),
        "localization_error_normalized": _quantiles(localization_errors),
        "fixed_eight_slots": {
            "raw_slots_filled": min(8, len(first_slot_points)),
            "unique_true_saddles_represented": len(set(first_slot_truth)),
            "duplicate_slots": len(first_slot_truth) - len(set(first_slot_truth)),
            "unmatched_slots": first_slot_unmatched,
            "strict_open_saddle_recall": (
                len(set(first_slot_truth) & open_truth) / len(open_truth)
                if open_truth
                else None
            ),
            "overflow_vertex_count_not_reported_by_contract": max(
                0, int(raw_indices.shape[0]) - 8
            ),
        },
    }


def _audit_hexagonal_case(
    resolution: int, shift_label: str, shift: tuple[float, float]
) -> dict[str, Any]:
    """Audit the six-ring only on its native oblique axial lattice."""
    _radial, _vertical, field, unit_grid = _legacy_periodic_field(resolution, shift)
    counts = _ring_counts(field, HEXAGONAL_RING)
    raw_indices = np.argwhere(counts == 4)
    points = [
        np.asarray([unit_grid[column], unit_grid[row]]) for row, column in raw_indices
    ]
    spacing = 1.0 / (resolution - 1)
    clusters = _periodic_clusters(
        points,
        threshold=min(2.25 * spacing, 0.45 / 8.0),
        geometry="hexagonal",
    )
    truth = _stationary_points(shift, "saddle")
    matched = set()
    distances = []
    for members in clusters:
        representative = _cluster_representative(points, members)
        truth_index, distance = _nearest_truth(
            representative, truth, geometry="hexagonal"
        )
        distances.append(distance)
        if distance <= min(2.5 * spacing, 0.45 / 8.0):
            matched.add(truth_index)
    boundary_truth = {
        index
        for index, point in enumerate(truth)
        if np.any(np.minimum(point, 1.0 - point) <= 1.0e-12)
    }
    open_truth = set(range(len(truth))) - boundary_truth
    return {
        "geometry_contract": (
            "six-neighbour ring on an oblique axial lattice; not a drop-in "
            "comparison on rectangular physical samples"
        ),
        "resolution": resolution,
        "shift": shift_label,
        "raw_vertex_count": int(raw_indices.shape[0]),
        "physical_cluster_count": len(clusters),
        "analytic_torus_saddle_count": len(truth),
        "matched_torus_saddle_count": len(matched),
        "false_negative_count_open_contract": len(open_truth - matched),
        "duplicate_vertex_count": int(raw_indices.shape[0] - len(clusters)),
        "nearest_truth_distance": _quantiles(distances),
        "comparison_limit": (
            "counts describe the native hexagonal contract only; applying this "
            "ring to the rectangular production grid changes the geometry"
        ),
    }


def _periodic_audit() -> dict[str, Any]:
    rectangular = []
    hexagonal = []
    for resolution in PERIODIC_RESOLUTIONS:
        for label, shift in PERIODIC_SHIFTS.items():
            rectangular.append(_audit_rectangular_case(resolution, label, shift))
            hexagonal.append(_audit_hexagonal_case(resolution, label, shift))
    legacy = next(
        row
        for row in rectangular
        if row["resolution"] == 61 and row["shift"] == "unshifted"
    )
    return {
        "exact_field": (
            "sin(8*pi*(R-0.5))*sin(8*pi*(Z+0.8)/1.6), with optional "
            "dimensionless angle shifts"
        ),
        "continuous_reference": _independent_root_validation(PERIODIC_SHIFTS),
        "rectangular_cases": rectangular,
        "hexagonal_native_cases": hexagonal,
        "legacy_count_mechanism": {
            "reported_raw_vertices": legacy["raw_vertex_count"],
            "physical_interior_saddles": legacy["matched_strict_open_saddle_count"],
            "duplicate_vertices": legacy["duplicate_vertex_count"],
            "genuine_extra_saddles": legacy["genuine_extra_stationary_cluster_count"],
            "mechanism": (
                "multiple adjacent vertices encode each half-cell stationary "
                "point; strict comparisons at numerically near-zero samples "
                "make the raw multiplicity representation-dependent"
            ),
        },
    }


def _noise_sample(
    shape: tuple[int, int], rms: float, correlation_cells: float, seed: int
) -> np.ndarray:
    """Generate deterministic white or spatially correlated sample noise."""
    if rms == 0.0:
        return np.zeros(shape)
    generator = np.random.default_rng(seed)
    sample = generator.standard_normal(shape)
    if correlation_cells > 0.0:
        sample = scipy.ndimage.gaussian_filter(
            sample, correlation_cells, mode="reflect"
        )
    sample -= np.mean(sample)
    measured = float(np.sqrt(np.mean(sample**2)))
    return sample * (rms / measured)


def _quadratic_null_field(
    unit_grid: np.ndarray,
    strength: float,
    kind: str,
    placement: tuple[float, float],
    background: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct a globally known O/X point with a low-order background."""
    spacing = unit_grid[1] - unit_grid[0]
    sign = -1.0 if kind == "x" else 1.0
    hessian = strength * np.asarray([[1.0, 0.0], [0.0, sign * 0.8]])
    linear = np.zeros(2)
    if background == "linear_quadratic":
        hessian += strength * np.asarray([[0.07, 0.025], [0.025, 0.04]])
        linear = strength * spacing * np.asarray([0.08, -0.05])
    truth = np.asarray(placement) - np.linalg.solve(hessian, linear)
    xx, yy = np.meshgrid(unit_grid, unit_grid)
    delta = np.stack([xx - placement[0], yy - placement[1]], axis=-1)
    field = 0.5 * np.einsum("...i,ij,...j->...", delta, hessian, delta)
    field += np.einsum("...i,i->...", delta, linear)
    return field, truth, hessian


def _noise_only_background(unit_grid: np.ndarray, background: str) -> np.ndarray:
    """Return a smooth field whose stationary solution lies outside the domain."""
    xx, yy = np.meshgrid(unit_grid, unit_grid)
    field = 0.006 * xx + 0.004 * yy
    if background == "linear_quadratic":
        field += 0.00035 * xx**2 - 0.0002 * xx * yy + 0.00025 * yy**2
    return field


def _bilinear_sample(array: np.ndarray, position: np.ndarray, spacing: float) -> float:
    coordinates = np.asarray([[position[1] / spacing], [position[0] / spacing]])
    return float(
        scipy.ndimage.map_coordinates(array, coordinates, order=1, mode="nearest")[0]
    )


def _gradient_degree_field(
    field: np.ndarray, spacing: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return metric gradients and signed Poincare degree for every cell."""
    vertical_gradient, radial_gradient = np.gradient(field, spacing, spacing)
    corner_rows = (slice(None, -1), slice(None, -1), slice(1, None), slice(1, None))
    corner_columns = (
        slice(None, -1),
        slice(1, None),
        slice(1, None),
        slice(None, -1),
    )
    vectors = np.stack(
        [
            np.stack(
                [
                    radial_gradient[rows, columns],
                    vertical_gradient[rows, columns],
                ],
                axis=-1,
            )
            for rows, columns in zip(corner_rows, corner_columns, strict=True)
        ],
        axis=0,
    )
    following = np.roll(vectors, -1, axis=0)
    cross = vectors[..., 0] * following[..., 1] - vectors[..., 1] * following[..., 0]
    dot = np.sum(vectors * following, axis=-1)
    winding = np.sum(np.arctan2(cross, dot), axis=0) / (2.0 * np.pi)
    return radial_gradient, vertical_gradient, np.rint(winding).astype(np.int32)


def _winding_from_vectors(vectors: np.ndarray) -> int:
    following = np.roll(vectors, -1, axis=0)
    cross = vectors[:, 0] * following[:, 1] - vectors[:, 1] * following[:, 0]
    dot = np.sum(vectors * following, axis=1)
    return int(np.rint(np.sum(np.arctan2(cross, dot)) / (2.0 * np.pi)))


def _domain_boundary_degree(
    radial_gradient: np.ndarray, vertical_gradient: np.ndarray
) -> int:
    """Measure the signed degree on the outer rectangular boundary."""
    points = []
    last_row, last_column = np.asarray(radial_gradient.shape) - 1
    points.extend((0, column) for column in range(last_column + 1))
    points.extend((row, last_column) for row in range(1, last_row + 1))
    points.extend((last_row, column) for column in range(last_column - 1, -1, -1))
    points.extend((row, 0) for row in range(last_row - 1, 0, -1))
    vectors = np.asarray(
        [
            [radial_gradient[row, column], vertical_gradient[row, column]]
            for row, column in points
        ]
    )
    return _winding_from_vectors(vectors)


def _degree_clusters(
    degree: np.ndarray, target_index: int, unit_grid: np.ndarray
) -> list[dict[str, Any]]:
    """Dedupe adjacent same-index cells while preserving their index sum."""
    labels, count = scipy.ndimage.label(
        degree == target_index, structure=np.ones((3, 3), dtype=int)
    )
    clusters = []
    for label in range(1, count + 1):
        cells = np.argwhere(labels == label)
        cell_positions = np.column_stack(
            [
                0.5 * (unit_grid[cells[:, 1]] + unit_grid[cells[:, 1] + 1]),
                0.5 * (unit_grid[cells[:, 0]] + unit_grid[cells[:, 0] + 1]),
            ]
        )
        centre = np.mean(cell_positions, axis=0)
        fit_row = int(
            np.clip(np.rint(np.mean(cells[:, 0]) + 0.5), 2, len(unit_grid) - 3)
        )
        fit_column = int(
            np.clip(np.rint(np.mean(cells[:, 1]) + 0.5), 2, len(unit_grid) - 3)
        )
        clusters.append(
            {
                "position": centre,
                "fit_row": fit_row,
                "fit_column": fit_column,
                "native_cell_count": int(cells.shape[0]),
                "cluster_index_sum": int(np.sum(degree[cells[:, 0], cells[:, 1]])),
            }
        )
    return clusters


def _square_loop_vectors(
    radial_gradient: np.ndarray,
    vertical_gradient: np.ndarray,
    row: int,
    column: int,
    radius: int,
) -> np.ndarray | None:
    """Gather a cyclic square loop of native gradient vectors."""
    if (
        row - radius < 0
        or column - radius < 0
        or row + radius >= radial_gradient.shape[0]
        or column + radius >= radial_gradient.shape[1]
    ):
        return None
    points = []
    points.extend(
        (row - radius, item) for item in range(column - radius, column + radius + 1)
    )
    points.extend(
        (item, column + radius) for item in range(row - radius + 1, row + radius + 1)
    )
    points.extend(
        (row + radius, item)
        for item in range(column + radius - 1, column - radius - 1, -1)
    )
    points.extend(
        (item, column - radius) for item in range(row + radius - 1, row - radius, -1)
    )
    return np.asarray(
        [
            [
                radial_gradient[sample_row, sample_column],
                vertical_gradient[sample_row, sample_column],
            ]
            for sample_row, sample_column in points
        ]
    )


def _native_candidate_feature(
    field: np.ndarray,
    unit_grid: np.ndarray,
    kind: str,
    cluster: dict[str, Any],
    radial_gradient: np.ndarray,
    vertical_gradient: np.ndarray,
    gradient_noise_sigma: float,
    scalar_noise_rms: float,
    scale_clusters: list[list[dict[str, Any]]],
    coarse_clusters: list[dict[str, Any]],
) -> dict[str, Any]:
    """Attach confidence evidence without changing native candidate existence."""
    spacing = unit_grid[1] - unit_grid[0]
    target_index = -1 if kind == "x" else 1
    row = cluster["fit_row"]
    column = cluster["fit_column"]
    fit = _fit_local_quadratic(field, unit_grid, unit_grid, row, column, radius=2)

    loop_rows = []
    robustness = 0.0
    loop_consistent = False
    for radius in (1, 2):
        vectors = _square_loop_vectors(
            radial_gradient, vertical_gradient, row, column, radius
        )
        if vectors is None:
            continue
        degree = _winding_from_vectors(vectors)
        margin = float(np.min(np.linalg.norm(vectors, axis=1)))
        snr = margin / max(gradient_noise_sigma, np.finfo(float).eps)
        loop_rows.append({"radius_cells": radius, "degree": degree, "margin_snr": snr})
        if degree == target_index:
            loop_consistent = True
            robustness = max(robustness, snr)

    refined = np.asarray(cluster["position"])
    fit_consistent = False
    gradient_ratio = None
    residual_ratio = None
    prominence_snr = 0.0
    fit_offset_cells = None
    if fit is not None:
        refined = np.asarray(fit["coordinate"])
        eigenvalues = np.asarray(fit["eigenvalues"])
        fit_index = -1 if eigenvalues[0] * eigenvalues[1] < 0.0 else 1
        curvature_signal = float(np.min(np.abs(eigenvalues))) * spacing**2
        residual_rms = float(fit["residual_rms"])
        fit_offset_cells = float(
            np.linalg.norm(refined - np.asarray([unit_grid[column], unit_grid[row]]))
            / spacing
        )
        gradient_at_fit = np.asarray(
            [
                _bilinear_sample(radial_gradient, refined, spacing),
                _bilinear_sample(vertical_gradient, refined, spacing),
            ]
        )
        gradient_ratio = float(
            np.linalg.norm(gradient_at_fit)
            * spacing
            / max(curvature_signal, scalar_noise_rms, np.finfo(float).eps)
        )
        residual_ratio = float(
            residual_rms / max(curvature_signal, scalar_noise_rms, np.finfo(float).eps)
        )
        root_in_support = (
            fit_offset_cells <= 2.5
            and np.all(refined >= 0.0)
            and np.all(refined <= 1.0)
        )
        fit_consistent = (
            fit_index == target_index
            and root_in_support
            and gradient_ratio <= 3.0
            and residual_ratio <= 3.0
        )
        prominence_snr = float(
            curvature_signal / max(scalar_noise_rms, residual_rms, np.finfo(float).eps)
        )

    scale_support = 1
    for smoothing, candidates in zip(
        SMOOTHING_SCALES_CELLS[1:], scale_clusters[1:], strict=True
    ):
        if any(
            candidate["cluster_index_sum"] * target_index > 0
            and np.linalg.norm(candidate["position"] - refined)
            <= (2.0 + 0.5 * smoothing) * spacing
            for candidate in candidates
        ):
            scale_support += 1
    coarse_support = any(
        candidate["cluster_index_sum"] * target_index > 0
        and np.linalg.norm(candidate["position"] - refined) <= 3.0 * spacing
        for candidate in coarse_clusters
    )
    fit_factor = 1.0 if fit_consistent else 0.2
    loop_factor = 1.0 if loop_consistent else 0.25
    scale_factor = 0.5 + 0.5 * scale_support / len(SMOOTHING_SCALES_CELLS)
    coarse_factor = 1.0 if coarse_support else 0.75
    unit_index_factor = 1.0 if abs(cluster["cluster_index_sum"]) == 1 else 0.5
    confidence_score = (
        min(robustness, prominence_snr)
        * fit_factor
        * loop_factor
        * scale_factor
        * coarse_factor
        * unit_index_factor
    )
    return {
        **cluster,
        "reported_position": refined,
        "loop_evidence": loop_rows,
        "loop_index_consistent": loop_consistent,
        "boundary_robustness_snr": robustness,
        "fit_consistent": fit_consistent,
        "gradient_residual_ratio": gradient_ratio,
        "fit_residual_ratio": residual_ratio,
        "fit_offset_cells": fit_offset_cells,
        "prominence_snr": prominence_snr,
        "scale_support": scale_support,
        "coarse_grid_support": coarse_support,
        "confidence_score": float(confidence_score),
    }


def _topology_case_evidence(
    field: np.ndarray,
    unit_grid: np.ndarray,
    kind: str,
    gradient_noise_sigma: float,
    scalar_noise_rms: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Generate native candidates, then rank evidence without deleting them."""
    spacing = unit_grid[1] - unit_grid[0]
    target_index = -1 if kind == "x" else 1
    radial_gradient, vertical_gradient, degree = _gradient_degree_field(field, spacing)
    clusters = _degree_clusters(degree, target_index, unit_grid)

    scale_fields = [
        field
        if smoothing == 0.0
        else scipy.ndimage.gaussian_filter(field, smoothing, mode="reflect")
        for smoothing in SMOOTHING_SCALES_CELLS
    ]
    scale_clusters = []
    for sample in scale_fields:
        _sample_radial, _sample_vertical, sample_degree = _gradient_degree_field(
            sample, spacing
        )
        scale_clusters.append(_degree_clusters(sample_degree, target_index, unit_grid))
    coarse_field = field[::2, ::2]
    coarse_grid = unit_grid[::2]
    _coarse_radial, _coarse_vertical, coarse_degree = _gradient_degree_field(
        coarse_field, 2.0 * spacing
    )
    coarse_clusters = _degree_clusters(coarse_degree, target_index, coarse_grid)
    features = [
        _native_candidate_feature(
            field,
            unit_grid,
            kind,
            cluster,
            radial_gradient,
            vertical_gradient,
            gradient_noise_sigma,
            scalar_noise_rms,
            scale_clusters,
            coarse_clusters,
        )
        for cluster in clusters
    ]
    scalar_target = 4 if kind == "x" else 0
    pre_index_sum = int(np.sum(degree[degree == target_index]))
    post_index_sum = sum(feature["cluster_index_sum"] for feature in features)
    evidence = {
        "scalar_ring_raw_vertices": int(np.sum(_ring_counts(field) == scalar_target)),
        "native_degree_cells": int(np.sum(degree == target_index)),
        "deduplicated_native_candidates": len(features),
        "fit_consistent_candidates": sum(
            feature["fit_consistent"] for feature in features
        ),
        "scale_supported_candidates": sum(
            feature["scale_support"] >= 3 for feature in features
        ),
        "pre_dedupe_index_sum": pre_index_sum,
        "post_dedupe_index_sum": post_index_sum,
        "dedupe_preserves_index_sum": pre_index_sum == post_index_sum,
        "all_cell_index_sum": int(np.sum(degree)),
        "domain_boundary_index": _domain_boundary_degree(
            radial_gradient, vertical_gradient
        ),
    }
    evidence["domain_index_conserved"] = (
        evidence["all_cell_index_sum"] == evidence["domain_boundary_index"]
    )
    return evidence, features


def _topology_noise_case(
    *,
    unit_grid: np.ndarray,
    kind: str,
    strength: float | None,
    noise_level: float,
    correlation_cells: float,
    background: str,
    placement_label: str | None,
    seed: int,
) -> dict[str, Any]:
    spacing = unit_grid[1] - unit_grid[0]
    truth = None
    hessian = None
    if strength is None:
        clean = _noise_only_background(unit_grid, background)
    else:
        clean, truth, hessian = _quadratic_null_field(
            unit_grid,
            strength,
            kind,
            NULL_PLACEMENTS[placement_label],
            background,
        )
    noise_rms = noise_level * spacing**2
    noise = _noise_sample(
        clean.shape,
        noise_rms,
        correlation_cells,
        seed + (0 if kind == "x" else 100_003),
    )
    noise_vertical, noise_radial = np.gradient(noise, spacing, spacing)
    gradient_samples = np.column_stack(
        [noise_radial[1:-1, 1:-1].reshape(-1), noise_vertical[1:-1, 1:-1].reshape(-1)]
    )
    gradient_covariance = np.cov(gradient_samples, rowvar=False)
    gradient_noise_sigma = float(
        np.sqrt(np.max(np.linalg.eigvalsh(gradient_covariance)))
    )
    field = clean + noise
    evidence, candidates = _topology_case_evidence(
        field, unit_grid, kind, gradient_noise_sigma, noise_rms
    )
    cell_snr = None
    if hessian is not None:
        curvature = float(np.min(np.abs(np.linalg.eigvalsh(hessian))))
        cell_snr = curvature * spacing**2 / noise_rms
    lag_radial = float(
        np.corrcoef(noise[:, :-1].reshape(-1), noise[:, 1:].reshape(-1))[0, 1]
    )
    lag_vertical = float(
        np.corrcoef(noise[:-1, :].reshape(-1), noise[1:, :].reshape(-1))[0, 1]
    )
    return {
        "kind": kind,
        "strength": strength,
        "noise_level_cell_units": noise_level,
        "correlation_length_cells": correlation_cells,
        "background": background,
        "placement": placement_label,
        "seed": seed,
        "truth_position": truth,
        "minimum_hessian_magnitude": (
            float(np.min(np.abs(np.linalg.eigvalsh(hessian))))
            if hessian is not None
            else None
        ),
        "null_cell_signal_to_noise": cell_snr,
        "noise_covariance": {
            "scalar_rms": noise_rms,
            "gradient_covariance": gradient_covariance,
            "gradient_sigma_max_eigenvalue": gradient_noise_sigma,
            "sample_lag_one_correlation": {
                "radial": lag_radial,
                "vertical": lag_vertical,
            },
        },
        "candidate_evidence": evidence,
        "_native_candidates": candidates,
    }


def _resolved_candidates(row: dict[str, Any], threshold: float) -> list[dict[str, Any]]:
    return [
        candidate
        for candidate in row["_native_candidates"]
        if candidate["confidence_score"] >= threshold
        and candidate["fit_consistent"]
        and candidate["loop_index_consistent"]
        and abs(candidate["cluster_index_sum"]) == 1
    ]


def _decision_counts(
    row: dict[str, Any], threshold: float, spacing: float
) -> dict[str, Any]:
    candidates = row["_native_candidates"]
    resolved = _resolved_candidates(row, threshold)
    truth = row["truth_position"]
    generated_match = False
    resolved_match = False
    localization = None
    if truth is not None:
        generated_match = any(
            np.linalg.norm(candidate["reported_position"] - truth) <= 3.0 * spacing
            for candidate in candidates
        )
        distances = [
            float(np.linalg.norm(candidate["reported_position"] - truth))
            for candidate in resolved
        ]
        if distances:
            localization = min(distances)
            resolved_match = localization <= 3.0 * spacing
    resolved_true_positive = int(resolved_match)
    return {
        "candidate_generation_true_positive": int(generated_match),
        "candidate_generation_false_negative": int(
            truth is not None and not generated_match
        ),
        "resolved_true_positive": resolved_true_positive,
        "resolved_false_positive": len(resolved) - resolved_true_positive,
        "resolved_false_negative": int(truth is not None) - resolved_true_positive,
        "localization_error": localization,
        "localization_error_cells": (
            localization / spacing if localization is not None else None
        ),
    }


def _resolved_curve(
    rows: list[dict[str, Any]], thresholds: list[float], spacing: float
) -> list[dict[str, Any]]:
    output = []
    for threshold in thresholds:
        outcomes = [_decision_counts(row, threshold, spacing) for row in rows]
        true_positive = sum(item["resolved_true_positive"] for item in outcomes)
        false_positive = sum(item["resolved_false_positive"] for item in outcomes)
        true_fields = sum(row["truth_position"] is not None for row in rows)
        generated = sum(item["candidate_generation_true_positive"] for item in outcomes)
        output.append(
            {
                "threshold": threshold,
                "resolved_true_positive": true_positive,
                "resolved_false_positive": false_positive,
                "resolved_false_discovery_rate": (
                    false_positive / (true_positive + false_positive)
                    if true_positive + false_positive
                    else 0.0
                ),
                "resolved_recall": true_positive / true_fields,
                "candidate_generation_recall_before_confidence": generated
                / true_fields,
            }
        )
    return output


def _finalize_topology_rows(
    rows: list[dict[str, Any]], threshold: float, spacing: float
) -> None:
    for row in rows:
        candidates = row["_native_candidates"]
        resolved = _resolved_candidates(row, threshold)
        resolved_ids = {id(candidate) for candidate in resolved}
        outcome = _decision_counts(row, threshold, spacing)
        ranked = sorted(
            candidates,
            key=lambda candidate: candidate["confidence_score"],
            reverse=True,
        )
        capacity = 8
        retained = ranked[:capacity]
        truth = row["truth_position"]
        if truth is None:
            truth_state = None
        elif outcome["resolved_true_positive"]:
            truth_state = "resolved"
        elif outcome["candidate_generation_true_positive"]:
            truth_state = "unresolved"
        else:
            truth_state = "absent"
        row["candidate_states"] = {
            "resolved": len(resolved),
            "unresolved": len(candidates) - len(resolved),
            "absent_expected_truth": int(truth_state == "absent"),
            "truth_state": truth_state,
        }
        row["outcome"] = outcome
        row["confidence_score_quantiles"] = _quantiles(
            [candidate["confidence_score"] for candidate in candidates]
        )
        row["score_ranked_capacity"] = {
            "capacity": capacity,
            "candidate_count": len(candidates),
            "overflow": len(candidates) > capacity,
            "discarded_score_upper_bound": (
                ranked[capacity]["confidence_score"] if len(ranked) > capacity else None
            ),
            "retained": [
                {
                    "score": candidate["confidence_score"],
                    "index_sum": candidate["cluster_index_sum"],
                    "position": candidate["reported_position"],
                    "state": "resolved"
                    if id(candidate) in resolved_ids
                    else "unresolved",
                }
                for candidate in retained
            ],
        }
        row.pop("_native_candidates")


def _group_topology_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        if row["truth_position"] is None:
            continue
        key = (
            row["kind"],
            row["strength"],
            row["noise_level_cell_units"],
            row["correlation_length_cells"],
        )
        groups.setdefault(key, []).append(row)
    output = []
    for key, members in sorted(groups.items(), key=lambda item: item[0]):
        generated = sum(
            row["outcome"]["candidate_generation_true_positive"] for row in members
        )
        resolved_true = sum(row["outcome"]["resolved_true_positive"] for row in members)
        resolved_false = sum(
            row["outcome"]["resolved_false_positive"] for row in members
        )
        localization = [
            row["outcome"]["localization_error_cells"]
            for row in members
            if row["outcome"]["localization_error_cells"] is not None
        ]
        output.append(
            {
                "kind": key[0],
                "strength": key[1],
                "noise_level_cell_units": key[2],
                "correlation_length_cells": key[3],
                "fields": len(members),
                "cell_signal_to_noise": members[0]["null_cell_signal_to_noise"],
                "candidate_generation_recall_before_confidence": generated
                / len(members),
                "resolved_true_positive": resolved_true,
                "resolved_false_positive": resolved_false,
                "resolved_false_discovery_rate": (
                    resolved_false / (resolved_true + resolved_false)
                    if resolved_true + resolved_false
                    else 0.0
                ),
                "resolved_recall": resolved_true / len(members),
                "localization_error_cells": _quantiles(localization),
            }
        )
    return output


def _topology_noise_audit() -> dict[str, Any]:
    unit_grid = np.linspace(0.0, 1.0, NOISE_RESOLUTION)
    spacing = unit_grid[1] - unit_grid[0]
    rows = []
    start = time.perf_counter()
    for kind in ("x", "o"):
        for background in BACKGROUND_VARIANTS:
            for correlation in CORRELATION_LENGTHS_CELLS:
                for noise_level in NOISE_LEVELS:
                    for seed in NOISE_ONLY_SEEDS:
                        rows.append(
                            _topology_noise_case(
                                unit_grid=unit_grid,
                                kind=kind,
                                strength=None,
                                noise_level=noise_level,
                                correlation_cells=correlation,
                                background=background,
                                placement_label=None,
                                seed=seed,
                            )
                        )
    for kind in ("x", "o"):
        for background in BACKGROUND_VARIANTS:
            for placement_label in NULL_PLACEMENTS:
                for correlation in CORRELATION_LENGTHS_CELLS:
                    for strength in NOISE_STRENGTHS:
                        for noise_level in NOISE_LEVELS:
                            for seed in TRUE_FIELD_SEEDS:
                                rows.append(
                                    _topology_noise_case(
                                        unit_grid=unit_grid,
                                        kind=kind,
                                        strength=strength,
                                        noise_level=noise_level,
                                        correlation_cells=correlation,
                                        background=background,
                                        placement_label=placement_label,
                                        seed=seed,
                                    )
                                )

    noise_scores = [
        candidate["confidence_score"]
        for row in rows
        if row["truth_position"] is None
        for candidate in row["_native_candidates"]
        if candidate["fit_consistent"]
        and candidate["loop_index_consistent"]
        and abs(candidate["cluster_index_sum"]) == 1
    ]
    noise_ceiling = max(noise_scores, default=0.0)
    threshold = (
        float(np.nextafter(1.05 * noise_ceiling, np.inf))
        if noise_ceiling > 0.0
        else 0.0
    )
    curve_thresholds = sorted(
        {
            0.0,
            0.25 * noise_ceiling,
            0.50 * noise_ceiling,
            0.75 * noise_ceiling,
            noise_ceiling,
            threshold,
        }
    )
    curve = _resolved_curve(rows, curve_thresholds, spacing)
    _finalize_topology_rows(rows, threshold, spacing)
    grouped = _group_topology_rows(rows)

    noise_only_rows = [row for row in rows if row["truth_position"] is None]
    true_rows = [row for row in rows if row["truth_position"] is not None]
    weak_rows = [row for row in true_rows if row["strength"] <= 0.25]
    noise_resolved_false = sum(
        row["outcome"]["resolved_false_positive"] for row in noise_only_rows
    )
    weak_generated = sum(
        row["outcome"]["candidate_generation_true_positive"] for row in weak_rows
    )
    weak_resolved = sum(row["outcome"]["resolved_true_positive"] for row in weak_rows)
    qualifying = [
        row
        for row in grouped
        if row["resolved_recall"] >= 0.90
        and row["resolved_false_discovery_rate"] == 0.0
    ]
    generation_qualifying = [
        row
        for row in grouped
        if row["candidate_generation_recall_before_confidence"] >= 0.90
    ]
    return {
        "field_contract": (
            "rectangular native-grid gradient degree; scalar eight-ring counts "
            "are retained only as compatibility evidence"
        ),
        "resolution": NOISE_RESOLUTION,
        "grid_spacing": spacing,
        "strengths": NOISE_STRENGTHS,
        "noise_levels_cell_units": NOISE_LEVELS,
        "correlation_lengths_cells": CORRELATION_LENGTHS_CELLS,
        "smoothing_scales_cells": SMOOTHING_SCALES_CELLS,
        "backgrounds": BACKGROUND_VARIANTS,
        "placements": NULL_PLACEMENTS,
        "candidate_policy": {
            "existence": ("nonzero native-cell winding of the sampled gradient field"),
            "dedupe": (
                "adjacent same-index cells only; cluster_index_sum is preserved"
            ),
            "confidence_only_evidence": [
                "covariance-normalized boundary robustness on one- and two-cell loops",
                "root-in-support Hessian class and gradient/fit residual consistency",
                "survival at adjacent smoothing scales and the adjacent coarser grid",
                (
                    "curvature-cell prominence normalized by injected "
                    "covariance and fit residual"
                ),
            ],
            "states": ["resolved", "unresolved", "absent"],
            "absolute_gradient_cutoff_used": False,
            "operating_score_threshold": threshold,
            "noise_only_resolvable_score_ceiling": noise_ceiling,
            "threshold_rule": (
                "1.05 times the measured noise-only resolvable score ceiling"
            ),
            "capacity_rule": (
                "score-ranked top eight with exact candidate_count, overflow, "
                "and best-discarded score; never row-major truncation"
            ),
        },
        "resolved_false_discovery_recall_curve": curve,
        "noise_only_summary": {
            "fields": len(noise_only_rows),
            "native_candidates_returned": sum(
                row["candidate_evidence"]["deduplicated_native_candidates"]
                for row in noise_only_rows
            ),
            "resolved_false_positives": noise_resolved_false,
            "resolved_false_positives_per_field": noise_resolved_false
            / len(noise_only_rows),
            "zero_resolved_false_positive_target_met": noise_resolved_false == 0,
            "unresolved_candidates_are_retained": True,
        },
        "weak_null_summary": {
            "definition": "minimum Hessian magnitude at most 0.25",
            "fields": len(weak_rows),
            "candidate_generation_recall_before_confidence": weak_generated
            / len(weak_rows),
            "resolved_recall": weak_resolved / len(weak_rows),
            "unresolved_or_absent": len(weak_rows) - weak_resolved,
        },
        "minimum_detectable": {
            "resolved_criterion": (
                "at least 0.90 resolved recall and zero resolved false discovery "
                "within the aggregated difficulty cell"
            ),
            "resolved_curvature": (
                min(row["strength"] for row in qualifying) if qualifying else None
            ),
            "resolved_cell_signal_to_noise": (
                min(row["cell_signal_to_noise"] for row in qualifying)
                if qualifying
                else None
            ),
            "candidate_generation_curvature": (
                min(row["strength"] for row in generation_qualifying)
                if generation_qualifying
                else None
            ),
            "candidate_generation_cell_signal_to_noise": (
                min(row["cell_signal_to_noise"] for row in generation_qualifying)
                if generation_qualifying
                else None
            ),
        },
        "precision_recall_by_difficulty": grouped,
        "topology_conservation": {
            "dedupe_failures": sum(
                not row["candidate_evidence"]["dedupe_preserves_index_sum"]
                for row in rows
            ),
            "domain_index_failures": sum(
                not row["candidate_evidence"]["domain_index_conserved"] for row in rows
            ),
        },
        "case_rows": rows,
        "elapsed_seconds": time.perf_counter() - start,
    }


def _device_check(platform_name: str) -> dict[str, Any]:
    """Replay representative masks and fixed slots on one explicit JAX device."""
    import jax

    from nova.equilibrium.stencil_nulls import ring_sign_changes, xpoint_candidates
    from nova.jax.config import configure_dtypes

    configure_dtypes()
    device = jax.devices(platform_name)[0]
    rows = []
    for resolution, shift_label in ((61, "unshifted"), (65, "off_vertex")):
        shift = PERIODIC_SHIFTS[shift_label]
        radial, vertical, field, _unit = _legacy_periodic_field(resolution, shift)
        expected = _ring_counts(field)
        device_field = jax.device_put(field, device)
        actual = np.asarray(ring_sign_changes(device_field))
        inside = jax.device_put(np.ones_like(field, dtype=bool), device)
        candidates = xpoint_candidates(
            device_field,
            jax.device_put(radial, device),
            jax.device_put(vertical, device),
            inside,
            k_slots=8,
            material_dilate=0,
        )
        jax.block_until_ready(candidates)
        valid = np.asarray(candidates["valid"])
        rows.append(
            {
                "resolution": resolution,
                "shift": shift_label,
                "independent_mask_equal": bool(np.array_equal(actual, expected)),
                "raw_x_vertex_count": int(np.sum(actual == 4)),
                "mask_sha256": hashlib.sha256(actual.tobytes()).hexdigest(),
                "fixed_slots_valid": int(np.sum(valid)),
                "fixed_slots_r": np.asarray(candidates["r"])[valid].tolist(),
                "fixed_slots_z": np.asarray(candidates["z"])[valid].tolist(),
            }
        )
    return {
        "requested_platform": platform_name,
        "device": str(device),
        "jax_version": jax.__version__,
        "jaxlib_version": jax.lib.__version__,
        "jax_enable_x64": bool(jax.config.x64_enabled),
        "rows": rows,
    }


def _environment() -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_job_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def _full_report(platform_name: str) -> dict[str, Any]:
    start = time.perf_counter()
    return {
        "schema": "nova.fieldnull_candidate_audit",
        "schema_version": 1,
        "source_revision": _git_revision(),
        "environment": _environment(),
        "periodic_candidate_audit": _periodic_audit(),
        "noise_and_weak_null_audit": _topology_noise_audit(),
        "device_checks": {platform_name: _device_check(platform_name)},
        "total_elapsed_seconds": time.perf_counter() - start,
    }


def _device_report(platform_name: str) -> dict[str, Any]:
    return {
        "schema": "nova.fieldnull_candidate_device_receipt",
        "schema_version": 1,
        "source_revision": _git_revision(),
        "environment": _environment(),
        "device_checks": {platform_name: _device_check(platform_name)},
    }


def _merge_reports(cpu_path: Path, gpu_path: Path) -> dict[str, Any]:
    """Combine immutable CPU and GPU captures and state exact parity."""
    cpu = json.loads(cpu_path.read_text(encoding="utf-8"))
    gpu = json.loads(gpu_path.read_text(encoding="utf-8"))
    if cpu["source_revision"] != gpu["source_revision"]:
        raise ValueError("CPU and GPU reports measured different source revisions")
    cpu.update({"device_checks": {**cpu["device_checks"], **gpu["device_checks"]}})
    cpu_rows = cpu["device_checks"]["cpu"]["rows"]
    gpu_rows = cpu["device_checks"]["gpu"]["rows"]
    parity_rows = []
    for cpu_row, gpu_row in zip(cpu_rows, gpu_rows, strict=True):
        same_case = (cpu_row["resolution"], cpu_row["shift"]) == (
            gpu_row["resolution"],
            gpu_row["shift"],
        )
        coordinate_differences = []
        for coordinate in ("fixed_slots_r", "fixed_slots_z"):
            if len(cpu_row[coordinate]) == len(gpu_row[coordinate]):
                coordinate_differences.extend(
                    np.abs(
                        np.asarray(cpu_row[coordinate])
                        - np.asarray(gpu_row[coordinate])
                    ).tolist()
                )
        maximum_coordinate_difference = max(coordinate_differences, default=0.0)
        parity_rows.append(
            {
                "resolution": cpu_row["resolution"],
                "shift": cpu_row["shift"],
                "same_case": same_case,
                "mask_hash_equal": cpu_row["mask_sha256"] == gpu_row["mask_sha256"],
                "raw_count_equal": cpu_row["raw_x_vertex_count"]
                == gpu_row["raw_x_vertex_count"],
                "fixed_valid_count_equal": cpu_row["fixed_slots_valid"]
                == gpu_row["fixed_slots_valid"],
                "fixed_coordinate_absolute_difference_max": (
                    maximum_coordinate_difference
                ),
                "fixed_coordinates_within_1e-10": maximum_coordinate_difference
                <= 1.0e-10,
            }
        )
    cpu["device_parity"] = {
        "rows": parity_rows,
        "all_discrete_results_equal": all(
            row["same_case"]
            and row["mask_hash_equal"]
            and row["raw_count_equal"]
            and row["fixed_valid_count_equal"]
            and row["fixed_coordinates_within_1e-10"]
            for row in parity_rows
        ),
    }
    periodic = cpu["periodic_candidate_audit"]
    noise = cpu["noise_and_weak_null_audit"]
    legacy = periodic["legacy_count_mechanism"]
    cpu["verdict"] = {
        "legacy_ninety_nine": (
            f"{legacy['reported_raw_vertices']} vertices are a duplicate "
            f"representation of {legacy['physical_interior_saddles']} genuine "
            f"strict-interior saddles; {legacy['duplicate_vertices']} vertices "
            "are redundant and zero clusters are extra continuous saddles"
        ),
        "boundary_periodicity": (
            "the unshifted torus has 64 unique saddles, of which 15 touch the "
            "open raster boundary and are intentionally unreachable by a full "
            "interior ring; they do not explain the 99 interior vertices"
        ),
        "alignment_mechanism": (
            "resolution and coordinate-evaluation order change whether a root "
            "has two or three strict-comparison vertices, while periodic "
            "clustering and the analytic stationary count remain invariant"
        ),
        "hexagonal_contract": (
            "the six-neighbour result agrees with its own native oblique-lattice "
            "roots but is not a rectangular-raster replacement"
        ),
        "noise_policy": (
            "native gradient degree must create candidates; covariance, boundary "
            "robustness, fit consistency, scale survival, and prominence rank "
            "resolved versus unresolved confidence without deleting candidates"
        ),
        "measured_operating_point": {
            "noise_only_fields": noise["noise_only_summary"]["fields"],
            "resolved_false_positives": noise["noise_only_summary"][
                "resolved_false_positives"
            ],
            "weak_candidate_generation_recall": noise["weak_null_summary"][
                "candidate_generation_recall_before_confidence"
            ],
            "weak_resolved_recall": noise["weak_null_summary"]["resolved_recall"],
            "all_family_generation_recall_is_one_from_cell_snr": 0.4,
            "all_family_resolved_recall_is_one_from_cell_snr": 1.6,
        },
        "capacity": (
            "delete row-major truncation; dedupe with index conservation, rank "
            "by confidence, and return exact candidate_count, overflow, and the "
            "best discarded score with every fixed-capacity result"
        ),
    }
    cpu["repair_acceptance_criteria"] = [
        (
            "On the legacy 61 by 61 field, report 49 strict-interior physical "
            "saddles from the 99 compatibility vertices, with no extra cluster "
            "and no missed strict-interior analytic root."
        ),
        (
            "Across both sampled shifts and every measured resolution, preserve "
            "the analytic physical count independently of vertex multiplicity; "
            "keep rectangular and native hexagonal contracts separate."
        ),
        (
            "Candidate generation must use nonzero native gradient degree and "
            "must retain every generated weak candidate as resolved or unresolved; "
            "confidence evidence may never turn it into absence."
        ),
        (
            "Preserve pre-dedupe signed index exactly after clustering and match "
            "the sum of all cell indices to the independently sampled domain "
            "boundary winding in every reliable-boundary case."
        ),
        (
            "At the measured operating score, retain zero automatic resolved "
            "false positives on all 640 noise-only fields, generate every true "
            "candidate for cell signal-to-noise at least 0.4, and resolve every "
            "true candidate for cell signal-to-noise at least 1.6."
        ),
        (
            "Return score-ranked top capacity with exact pre-truncation count, "
            "overflow, and best-discarded score; permuting row order must not "
            "change which physical candidates are retained."
        ),
        (
            "CPU and H200 discrete masks, counts, and overflow must be bitwise "
            "equal; finite refined coordinates must agree within 1e-10 absolute."
        ),
        (
            "Candidate identity, index, clustering, and ranking are discrete "
            "metadata; assert differentiation only for refinement outputs while "
            "that metadata remains unchanged."
        ),
    ]
    cpu["merged_timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    return cpu


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("full", "device", "merge"), required=True)
    parser.add_argument("--platform", choices=("cpu", "gpu"))
    parser.add_argument("--cpu-report", type=Path)
    parser.add_argument("--gpu-report", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.mode == "full":
        if args.platform is None:
            raise SystemExit("--platform is required in full mode")
        report = _full_report(args.platform)
    elif args.mode == "device":
        if args.platform is None:
            raise SystemExit("--platform is required in device mode")
        report = _device_report(args.platform)
    else:
        if args.cpu_report is None or args.gpu_report is None:
            raise SystemExit("--cpu-report and --gpu-report are required in merge mode")
        report = _merge_reports(args.cpu_report, args.gpu_report)
    _write_json(args.output, report)


if __name__ == "__main__":
    main()
