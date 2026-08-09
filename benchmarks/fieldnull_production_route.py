"""Measure the production rectangular critical-point route on CPU and GPU.

``measure`` captures one platform without modifying the repository.  ``assemble``
combines the two immutable captures into the strict JSON evidence artifact.
Candidate fields reuse the independent audit generators; only the production
detector, fit, confidence gate, and fixed-capacity result are evaluated here.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import socket
import subprocess
import time
from typing import Any

import jax
import numpy as np

from benchmarks.fieldnull_candidate_audit import (
    BACKGROUND_VARIANTS,
    CORRELATION_LENGTHS_CELLS,
    NOISE_LEVELS,
    NOISE_ONLY_SEEDS,
    NOISE_RESOLUTION,
    NOISE_STRENGTHS,
    NULL_PLACEMENTS,
    TRUE_FIELD_SEEDS,
    _legacy_periodic_field,
    _noise_only_background,
    _noise_sample,
    _quadratic_null_field,
    _ring_counts,
)
from nova.jax.stencil_nulls import (
    critical_point_candidates_batch,
    gradient_cell_degree,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATHS = (
    "nova/jax/stencil_nulls.py",
    "nova/jax/connectivity_boundary.py",
    "nova/equilibrium/profile.py",
    "benchmarks/fieldnull_production_route.py",
)
REPEATS = 7
TIMING_BATCH = 128
TIMING_SHAPE = (65, 129)
TIMING_CAPACITY = 8
AUDIT_CAPACITY = 64


def _strict(value: Any) -> Any:
    """Return JSON-safe Python values, replacing nonfinite floats with null."""
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


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_strict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _revision() -> str:
    if supplied := os.environ.get("NOVA_AUDIT_REVISION"):
        return supplied
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _source_hashes() -> dict[str, str]:
    return {
        path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        for path in SOURCE_PATHS
    }


def _block(tree):
    for leaf in jax.tree.leaves(tree):
        method = getattr(leaf, "block_until_ready", None)
        if method is not None:
            method()
    return tree


def _tree_bytes(tree) -> int:
    return int(
        sum(
            getattr(leaf, "size", 0) * getattr(leaf, "dtype", np.dtype("u1")).itemsize
            for leaf in jax.tree.leaves(tree)
        )
    )


def _timing_fields() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nz, nr = TIMING_SHAPE
    radial = np.linspace(0.2, 1.8, nr)
    vertical = np.linspace(-1.2, 1.2, nz)
    rr, zz = np.meshgrid(radial, vertical)
    fields = []
    for index in range(TIMING_BATCH):
        phase = (index + 0.5) / TIMING_BATCH
        fields.append(
            np.sin(8.0 * np.pi * (rr - 0.2) / 1.6 + 0.13 * phase)
            * np.sin(8.0 * np.pi * (zz + 1.2) / 2.4 - 0.17 * phase)
        )
    return np.asarray(fields), radial, vertical, np.ones(TIMING_SHAPE, dtype=bool)


def _run_production(
    fields, radial, vertical, inside, noise_sigma, target_index, capacity
):
    return critical_point_candidates_batch(
        fields,
        radial,
        vertical,
        inside,
        k_slots=capacity,
        material_dilate=0,
        target_index=target_index,
        noise_sigma=noise_sigma,
    )


def _timings(device) -> dict[str, Any]:
    host_fields, host_radial, host_vertical, host_inside = _timing_fields()
    fields = jax.device_put(host_fields, device)
    radial = jax.device_put(host_radial, device)
    vertical = jax.device_put(host_vertical, device)
    inside = jax.device_put(host_inside, device)
    noise = jax.device_put(np.zeros(TIMING_BATCH), device)

    start = time.perf_counter()
    compiled_result = _block(
        _run_production(fields, radial, vertical, inside, noise, -1, TIMING_CAPACITY)
    )
    compile_seconds = time.perf_counter() - start
    start = time.perf_counter()
    _block(
        _run_production(fields, radial, vertical, inside, noise, -1, TIMING_CAPACITY)
    )
    cache_probe_seconds = time.perf_counter() - start

    resident_samples = []
    for _ in range(REPEATS):
        start = time.perf_counter_ns()
        _block(
            _run_production(
                fields, radial, vertical, inside, noise, -1, TIMING_CAPACITY
            )
        )
        resident_samples.append((time.perf_counter_ns() - start) / 1.0e9)

    inclusive_samples = []
    for _ in range(REPEATS):
        start = time.perf_counter_ns()
        moved = _run_production(
            jax.device_put(host_fields, device),
            jax.device_put(host_radial, device),
            jax.device_put(host_vertical, device),
            jax.device_put(host_inside, device),
            jax.device_put(np.zeros(TIMING_BATCH), device),
            -1,
            TIMING_CAPACITY,
        )
        host = jax.device_get(_block(moved))
        if int(np.asarray(host["candidate_count"])[0]) < 0:
            raise RuntimeError("candidate count must be nonnegative")
        inclusive_samples.append((time.perf_counter_ns() - start) / 1.0e9)

    resident = float(np.median(resident_samples))
    inclusive = float(np.median(inclusive_samples))
    input_bytes = sum(
        array.nbytes for array in (host_fields, host_radial, host_vertical, host_inside)
    )
    result_bytes = _tree_bytes(compiled_result)
    double_buffered = 2 * (input_bytes + result_bytes)
    memory = device.memory_stats() or {}
    limit = memory.get("bytes_limit")
    return {
        "batch": TIMING_BATCH,
        "shape": list(TIMING_SHAPE),
        "capacity": TIMING_CAPACITY,
        "compile_seconds": compile_seconds,
        "cache_probe_seconds": cache_probe_seconds,
        "resident_seconds": resident,
        "resident_fields_per_second": TIMING_BATCH / resident,
        "inclusive_seconds": inclusive,
        "inclusive_fields_per_second": TIMING_BATCH / inclusive,
        "transfer_dispatch_fraction": max(0.0, inclusive - resident) / inclusive,
        "input_bytes": input_bytes,
        "result_bytes": result_bytes,
        "double_buffered_bytes": double_buffered,
        "device_bytes_limit": limit,
        "double_buffered_hbm_fraction": (
            double_buffered / limit if isinstance(limit, int) and limit > 0 else None
        ),
        "candidate_count_min": int(
            np.min(np.asarray(compiled_result["candidate_count"]))
        ),
        "candidate_count_max": int(
            np.max(np.asarray(compiled_result["candidate_count"]))
        ),
        "overflow_fields": int(np.sum(np.asarray(compiled_result["overflow"]))),
    }


def _legacy_check(device) -> dict[str, Any]:
    radial, vertical, field, _unit = _legacy_periodic_field(61, (0.0, 0.0))
    fields = jax.device_put(field[None], device)
    result = _block(
        _run_production(
            fields,
            jax.device_put(radial, device),
            jax.device_put(vertical, device),
            jax.device_put(np.ones_like(field, dtype=bool), device),
            jax.device_put(np.zeros(1), device),
            -1,
            64,
        )
    )
    host = jax.device_get(result)
    present = np.asarray(host["present"])[0]
    coordinates = np.column_stack(
        [np.asarray(host["r"])[0, present], np.asarray(host["z"])[0, present]]
    )
    truth = np.asarray(
        [
            (0.5 + radial_index / 8.0, -0.8 + 1.6 * vertical_index / 8.0)
            for vertical_index in range(1, 8)
            for radial_index in range(1, 8)
        ]
    )
    distances = np.linalg.norm(coordinates[:, None] - truth[None], axis=-1)
    nearest = np.min(distances, axis=1)
    spacing = min(radial[1] - radial[0], vertical[1] - vertical[0])
    fixed = _block(
        _run_production(
            fields,
            jax.device_put(radial, device),
            jax.device_put(vertical, device),
            jax.device_put(np.ones_like(field, dtype=bool), device),
            jax.device_put(np.zeros(1), device),
            -1,
            8,
        )
    )
    fixed_host = jax.device_get(fixed)
    return {
        "legacy_raw_scalar_vertices": int(np.sum(_ring_counts(field) == 4)),
        "candidate_count": int(host["candidate_count"][0]),
        "resolved_count": int(np.sum(host["resolved"][0])),
        "unresolved_count": int(np.sum(host["state"][0] == 1)),
        "candidate_index_sum": int(host["candidate_index_sum"][0]),
        "domain_signed_index": int(host["domain_signed_index"][0]),
        "matched_reference_count": int(np.sum(nearest <= spacing)),
        "extra_reference_count": int(np.sum(nearest > spacing)),
        "worst_localization_cells": float(np.max(nearest) / spacing),
        "fixed_capacity": 8,
        "fixed_present_count": int(np.sum(fixed_host["present"][0])),
        "fixed_overflow": bool(fixed_host["overflow"][0]),
        "fixed_discarded_score_upper_bound": float(
            fixed_host["discarded_score_upper_bound"][0]
        ),
        "coordinates": coordinates,
        "source_cells": np.asarray(host["source_cell"])[0, present],
    }


def _noise_cases(kind: str):
    unit_grid = np.linspace(0.0, 1.0, NOISE_RESOLUTION)
    spacing = unit_grid[1] - unit_grid[0]
    fields = []
    sigmas = []
    labels = []
    offset = 0 if kind == "x" else 100_003
    for background in BACKGROUND_VARIANTS:
        clean = _noise_only_background(unit_grid, background)
        for correlation in CORRELATION_LENGTHS_CELLS:
            for noise_level in NOISE_LEVELS:
                sigma = noise_level * spacing**2
                for seed in NOISE_ONLY_SEEDS:
                    fields.append(
                        clean
                        + _noise_sample(clean.shape, sigma, correlation, seed + offset)
                    )
                    sigmas.append(sigma)
                    labels.append((background, correlation, noise_level))
    return np.asarray(fields), np.asarray(sigmas), labels, unit_grid


def _true_cases(kind: str):
    unit_grid = np.linspace(0.0, 1.0, NOISE_RESOLUTION)
    spacing = unit_grid[1] - unit_grid[0]
    fields = []
    sigmas = []
    truth = []
    metadata = []
    offset = 0 if kind == "x" else 100_003
    for background in BACKGROUND_VARIANTS:
        for placement_name, placement in NULL_PLACEMENTS.items():
            for correlation in CORRELATION_LENGTHS_CELLS:
                for strength in NOISE_STRENGTHS:
                    clean, root, hessian = _quadratic_null_field(
                        unit_grid, strength, kind, placement, background
                    )
                    curvature = float(np.min(np.abs(np.linalg.eigvalsh(hessian))))
                    for noise_level in NOISE_LEVELS:
                        sigma = noise_level * spacing**2
                        cell_snr = curvature * spacing**2 / sigma
                        for seed in TRUE_FIELD_SEEDS:
                            fields.append(
                                clean
                                + _noise_sample(
                                    clean.shape, sigma, correlation, seed + offset
                                )
                            )
                            sigmas.append(sigma)
                            truth.append(root)
                            metadata.append(
                                {
                                    "kind": kind,
                                    "background": background,
                                    "placement": placement_name,
                                    "correlation_cells": correlation,
                                    "strength": strength,
                                    "noise_level": noise_level,
                                    "cell_snr": cell_snr,
                                }
                            )
    return (
        np.asarray(fields),
        np.asarray(sigmas),
        np.asarray(truth),
        metadata,
        unit_grid,
    )


def _match(result, truth, spacing):
    radius = np.asarray(result["r"])
    height = np.asarray(result["z"])
    positions = np.stack([radius, height], axis=-1)
    distance = np.linalg.norm(positions - truth[:, None, :], axis=-1) / spacing
    present = np.asarray(result["present"])
    resolved = np.asarray(result["resolved"])
    generated_match = np.any(present & (distance <= 3.0), axis=1)
    resolved_match = np.any(resolved & (distance <= 3.0), axis=1)
    resolved_false = resolved & (distance > 3.0)
    closest_resolved = np.min(np.where(resolved, distance, np.inf), axis=1)
    return generated_match, resolved_match, resolved_false, closest_resolved


def _native_generation_match(device, fields, unit_grid, truth, target_index):
    """Match truth against the complete pre-ranking native-degree field."""
    device_grid = jax.device_put(unit_grid, device)
    degree = jax.jit(
        lambda values: gradient_cell_degree(values, device_grid, device_grid)[0]
    )(jax.device_put(fields, device))
    degree = np.asarray(jax.device_get(_block(degree)))
    spacing = unit_grid[1] - unit_grid[0]
    columns = np.floor((truth[:, 0] - unit_grid[0]) / spacing).astype(int)
    rows = np.floor((truth[:, 1] - unit_grid[0]) / spacing).astype(int)
    generated = []
    for field_index, (row, column) in enumerate(zip(rows, columns, strict=True)):
        row_start = max(0, row - 3)
        row_stop = min(degree.shape[1], row + 4)
        column_start = max(0, column - 3)
        column_stop = min(degree.shape[2], column + 4)
        generated.append(
            np.any(
                degree[field_index, row_start:row_stop, column_start:column_stop]
                == target_index
            )
        )
    return np.asarray(generated, dtype=bool)


def _scientific_check(device) -> dict[str, Any]:
    noise_summary = {}
    true_rows = []
    localization = []
    for kind, target_index in (("x", -1), ("o", 1)):
        noise_fields, noise_sigma, _labels, unit_grid = _noise_cases(kind)
        noise_result = jax.device_get(
            _block(
                _run_production(
                    jax.device_put(noise_fields, device),
                    jax.device_put(unit_grid, device),
                    jax.device_put(unit_grid, device),
                    jax.device_put(np.ones(noise_fields.shape[1:], dtype=bool), device),
                    jax.device_put(noise_sigma, device),
                    target_index,
                    AUDIT_CAPACITY,
                )
            )
        )
        resolved_per_field = np.sum(np.asarray(noise_result["resolved"]), axis=1)
        noise_summary[kind] = {
            "fields": int(noise_fields.shape[0]),
            "raw_candidates": int(np.sum(noise_result["candidate_count"])),
            "resolved_false_positive_fields": int(np.sum(resolved_per_field > 0)),
            "resolved_false_positive_candidates": int(np.sum(resolved_per_field)),
            "overflow_fields": int(np.sum(noise_result["overflow"])),
        }

        true_fields, true_sigma, truth, metadata, unit_grid = _true_cases(kind)
        true_result = jax.device_get(
            _block(
                _run_production(
                    jax.device_put(true_fields, device),
                    jax.device_put(unit_grid, device),
                    jax.device_put(unit_grid, device),
                    jax.device_put(np.ones(true_fields.shape[1:], dtype=bool), device),
                    jax.device_put(true_sigma, device),
                    target_index,
                    AUDIT_CAPACITY,
                )
            )
        )
        spacing = unit_grid[1] - unit_grid[0]
        retained, resolved, false, closest = _match(true_result, truth, spacing)
        generated = _native_generation_match(
            device, true_fields, unit_grid, truth, target_index
        )
        for index, row in enumerate(metadata):
            true_rows.append(
                {
                    **row,
                    "generated": bool(generated[index]),
                    "retained": bool(retained[index]),
                    "resolved": bool(resolved[index]),
                    "resolved_false_candidates": int(np.sum(false[index])),
                }
            )
            if resolved[index]:
                localization.append(float(closest[index]))

    grouped = defaultdict(lambda: [0, 0, 0, 0])
    for row in true_rows:
        family = (
            row["kind"],
            row["background"],
            row["placement"],
            row["correlation_cells"],
        )
        if row["cell_snr"] >= 0.4:
            grouped[family][0] += int(row["generated"])
            grouped[family][1] += 1
        if row["cell_snr"] >= 1.6:
            grouped[family][2] += int(row["resolved"])
            grouped[family][3] += 1
    family_rows = []
    for family, counts in sorted(grouped.items()):
        family_rows.append(
            {
                "kind": family[0],
                "background": family[1],
                "placement": family[2],
                "correlation_cells": family[3],
                "generation_recall_from_cell_snr_0_4": counts[0] / counts[1],
                "resolved_recall_from_cell_snr_1_6": counts[2] / counts[3],
            }
        )
    return {
        "noise_only": noise_summary,
        "noise_only_fields": sum(row["fields"] for row in noise_summary.values()),
        "noise_only_resolved_false_positive_fields": sum(
            row["resolved_false_positive_fields"] for row in noise_summary.values()
        ),
        "true_fields": len(true_rows),
        "generation_recall_from_cell_snr_0_4_minimum_family": min(
            row["generation_recall_from_cell_snr_0_4"] for row in family_rows
        ),
        "resolved_recall_from_cell_snr_1_6_minimum_family": min(
            row["resolved_recall_from_cell_snr_1_6"] for row in family_rows
        ),
        "resolved_false_candidates": sum(
            row["resolved_false_candidates"] for row in true_rows
        ),
        "retained_generation_recall": sum(row["retained"] for row in true_rows)
        / len(true_rows),
        "resolved_localization_cells_max": max(localization, default=None),
        "family_rows": family_rows,
    }


def measure(platform_name: str) -> dict[str, Any]:
    device = jax.devices(platform_name)[0]
    return {
        "schema": "nova.fieldnull-production-route",
        "schema_version": 1,
        "captured_at": datetime.now(UTC).isoformat(),
        "revision": _revision(),
        "source_hashes": _source_hashes(),
        "platform": platform_name,
        "device": str(device),
        "environment": {
            "host": socket.gethostname(),
            "python": platform.python_version(),
            "jax": jax.__version__,
            "jaxlib": jax.lib.__version__,
            "jax_enable_x64": bool(jax.config.x64_enabled),
            "xla_flags": os.environ.get("XLA_FLAGS"),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        },
        "legacy": _legacy_check(device),
        "scientific": _scientific_check(device),
        "timing": _timings(device),
    }


def assemble(cpu_path: Path, gpu_path: Path) -> dict[str, Any]:
    cpu = json.loads(cpu_path.read_text(encoding="utf-8"))
    gpu = json.loads(gpu_path.read_text(encoding="utf-8"))
    if (
        cpu["revision"] != gpu["revision"]
        or cpu["source_hashes"] != gpu["source_hashes"]
    ):
        raise ValueError("CPU and GPU captures do not describe identical sources")
    cpu_coordinates = np.asarray(cpu["legacy"]["coordinates"])
    gpu_coordinates = np.asarray(gpu["legacy"]["coordinates"])
    parity = {
        "candidate_count_equal": cpu["legacy"]["candidate_count"]
        == gpu["legacy"]["candidate_count"],
        "overflow_equal": cpu["legacy"]["fixed_overflow"]
        == gpu["legacy"]["fixed_overflow"],
        "source_cells_equal": cpu["legacy"]["source_cells"]
        == gpu["legacy"]["source_cells"],
        "coordinate_max_absolute_difference": float(
            np.max(np.abs(cpu_coordinates - gpu_coordinates))
        ),
    }
    speedup = (
        gpu["timing"]["resident_fields_per_second"]
        / cpu["timing"]["resident_fields_per_second"]
    )
    scientific_pass = all(
        capture["legacy"]["candidate_count"] == 49
        and capture["legacy"]["matched_reference_count"] == 49
        and capture["legacy"]["extra_reference_count"] == 0
        and capture["scientific"]["noise_only_resolved_false_positive_fields"] == 0
        and capture["scientific"]["generation_recall_from_cell_snr_0_4_minimum_family"]
        == 1.0
        and capture["scientific"]["resolved_recall_from_cell_snr_1_6_minimum_family"]
        == 1.0
        and capture["scientific"]["resolved_false_candidates"] == 0
        for capture in (cpu, gpu)
    )
    return {
        "schema": "nova.fieldnull-production-route",
        "schema_version": 1,
        "assembled_at": datetime.now(UTC).isoformat(),
        "revision": cpu["revision"],
        "source_hashes": cpu["source_hashes"],
        "contract": {
            "geometry": "rectangular four-corner native gradient degree",
            "candidate_states": ["absent", "unresolved", "resolved"],
            "selection": (
                "confidence-ranked fixed top-k with exact pre-capacity metadata"
            ),
            "hexagonal_contract": "separate and unchanged",
        },
        "cpu": cpu,
        "gpu": gpu,
        "parity": parity,
        "performance": {
            "resident_gpu_over_one_core_cpu_speedup": speedup,
            "speedup_target": 3.0,
            "speedup_pass": speedup >= 3.0,
            "gpu_transfer_dispatch_fraction": gpu["timing"][
                "transfer_dispatch_fraction"
            ],
            "transfer_dispatch_target": 0.2,
            "transfer_dispatch_pass": gpu["timing"]["transfer_dispatch_fraction"] < 0.2,
            "remaining_seam": "host materialization between archive/profile batches",
            "double_buffered_hbm_fraction": gpu["timing"][
                "double_buffered_hbm_fraction"
            ],
        },
        "verdict": {
            "scientific_pass": scientific_pass,
            "cpu_gpu_parity_pass": parity["candidate_count_equal"]
            and parity["overflow_equal"]
            and parity["source_cells_equal"]
            and parity["coordinate_max_absolute_difference"] <= 1.0e-10,
            "production_route": "single rectangular implementation",
        },
    }


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    measurement = subparsers.add_parser("measure")
    measurement.add_argument("--platform", choices=("cpu", "gpu"), required=True)
    measurement.add_argument("--output", type=Path, required=True)
    assembly = subparsers.add_parser("assemble")
    assembly.add_argument("--cpu", type=Path, required=True)
    assembly.add_argument("--gpu", type=Path, required=True)
    assembly.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    if arguments.command == "measure":
        _write(arguments.output, measure(arguments.platform))
    else:
        _write(arguments.output, assemble(arguments.cpu, arguments.gpu))


if __name__ == "__main__":
    main()
