"""Measure global-spline contour geometry against closed-form level sets.

The accuracy ladder compares linear edge interpolation with fixed-iteration
roots of the global tensor spline.  Cubic Hermite controls use endpoint
gradients from that same spline.  Separate hyperbolic fields exercise both
resolved and exactly tied diagonal-corner saddle configurations.

The benchmark does not call the polygon clip integration path.  Its receipt is
completed only after the pinned transport-characterisation tests have passed.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import t as student_t

from nova.equilibrium.flux_surface_connectivity import traced_spline_contour
from nova.jax.config import configure_dtypes


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT / "docs/figures/coefficient-space-newton/higher-order-contour.json"
)
POINT_COUNTS = (17, 25, 33, 49, 65, 97)
LINEAR_ORDER_BAND = (1.7, 2.3)
SPLINE_ORDER_BAND = (3.2, 4.8)
TANGENT_JUMP_LIMIT = 1.0e-8
AGREEMENT_TOLERANCE = 2.0e-12
CHARACTERISATION_TESTS = (
    "tests/test_transport_geometry_reference.py::"
    "test_nova_fsa_matches_torax_eqdsk_reader",
    "tests/test_transport_geometry_reference.py::"
    "test_clipped_cells_match_independent_contour_geometry",
)
STEP_BOUNDS_PERCENT = {"vpr_face": 3.92, "g1_face": 6.20}
INTERIM_COARSE_SMOKE = {
    "linear_crossing_rms_error_m": 2.370674740501376e-3,
    "global_spline_crossing_rms_error_m": 9.151001514989856e-7,
    "linear_to_global_spline_error_ratio": (
        2.370674740501376e-3 / 9.151001514989856e-7
    ),
    "linear_maximum_unit_tangent_jump": 0.2258469748620969,
    "global_spline_maximum_unit_tangent_jump": 0.0,
    "global_spline_shared_edge_comparisons": 29,
    "statement": (
        "Initial isolated 16-by-16-cell smoke retained exactly as observed before "
        "the full configured-x64 receipt."
    ),
}


def _source_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _strict(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _strict(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_strict(item) for item in value]
    if isinstance(value, np.ndarray):
        return _strict(value.tolist())
    if isinstance(value, np.generic):
        return _strict(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_strict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _analytic_flux(radius: np.ndarray, height: np.ndarray) -> np.ndarray:
    radial = (radius - 1.55) / 0.58
    vertical = height / 0.73
    return radial**2 + 0.16 * radial**4 + vertical**2


def _cell_edges(
    radial: np.ndarray, vertical: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    cell_shape = (vertical.size - 1, radial.size - 1)
    radial_low = np.broadcast_to(radial[:-1][None, :], cell_shape)
    radial_high = np.broadcast_to(radial[1:][None, :], cell_shape)
    vertical_low = np.broadcast_to(vertical[:-1, None], cell_shape)
    vertical_high = np.broadcast_to(vertical[1:, None], cell_shape)
    start = np.stack(
        (
            np.stack((radial_low, vertical_low), axis=-1),
            np.stack((radial_high, vertical_low), axis=-1),
            np.stack((radial_high, vertical_high), axis=-1),
            np.stack((radial_low, vertical_high), axis=-1),
        ),
        axis=-2,
    )
    return start, np.roll(start, shift=-1, axis=-2)


def _truth_crossings(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    low = start.copy()
    high = end.copy()
    low_value = _analytic_flux(low[:, 0], low[:, 1]) - 1.0
    for _ in range(80):
        middle = 0.5 * (low + high)
        middle_value = _analytic_flux(middle[:, 0], middle[:, 1]) - 1.0
        same_side = (low_value >= 0.0) == (middle_value >= 0.0)
        low = np.where(same_side[:, None], middle, low)
        high = np.where(same_side[:, None], high, middle)
        low_value = np.where(same_side, middle_value, low_value)
    return 0.5 * (low + high)


def _linear_crossings(
    start: np.ndarray, end: np.ndarray, start_value: np.ndarray, end_value: np.ndarray
) -> np.ndarray:
    parameter = -start_value / (end_value - start_value)
    return start + parameter[:, None] * (end - start)


def _fit_order(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    spacing = np.asarray([row["maximum_cell_pitch_m"] for row in rows])
    error = np.asarray([row[key] for row in rows])
    design = np.column_stack((np.ones_like(spacing), np.log(spacing)))
    coefficient, _, _, _ = np.linalg.lstsq(design, np.log(error), rcond=None)
    residual = np.log(error) - design @ coefficient
    degrees_of_freedom = spacing.size - design.shape[1]
    variance = float(residual @ residual / degrees_of_freedom)
    covariance = variance * np.linalg.inv(design.T @ design)
    uncertainty = math.sqrt(float(covariance[1, 1]))
    critical = float(student_t.ppf(0.975, degrees_of_freedom))
    order = float(coefficient[1])
    return {
        "estimate": order,
        "one_sigma_uncertainty": uncertainty,
        "confidence_95_percent": [
            order - critical * uncertainty,
            order + critical * uncertainty,
        ],
        "rungs": int(spacing.size),
    }


def _grouped_tangent_jumps(
    endpoints: np.ndarray, tangents: np.ndarray
) -> dict[str, float | int]:
    groups: dict[tuple[float, float], list[np.ndarray]] = defaultdict(list)
    for point, tangent in zip(endpoints.reshape(-1, 2), tangents.reshape(-1, 2)):
        groups[tuple(np.round(point, decimals=11))].append(tangent)
    jumps: list[float] = []
    for group in groups.values():
        if len(group) < 2:
            continue
        for first in range(len(group) - 1):
            for second in range(first + 1, len(group)):
                direct = np.linalg.norm(group[first] - group[second])
                reversed_direction = np.linalg.norm(group[first] + group[second])
                jumps.append(float(min(direct, reversed_direction)))
    return {
        "shared_edge_comparisons": len(jumps),
        "maximum_unit_tangent_jump": max(jumps, default=0.0),
        "rms_unit_tangent_jump": (
            float(np.sqrt(np.mean(np.square(jumps)))) if jumps else 0.0
        ),
    }


def _gather_pairs(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return np.take_along_axis(values[..., None, :, :], indices[..., None], axis=-2)


def _accuracy_row(point_count: int) -> tuple[dict[str, Any], dict[str, Any]]:
    radial = np.linspace(0.78, 2.32, point_count)
    vertical = np.linspace(-0.98, 0.98, point_count)
    mesh_radial, mesh_vertical = np.meshgrid(radial, vertical)
    values = _analytic_flux(mesh_radial, mesh_vertical)
    with jax.disable_jit():
        contour = traced_spline_contour(
            jnp.asarray(values),
            jnp.asarray(radial),
            jnp.asarray(vertical),
            jnp.asarray(1.0),
        )
    contour = jax.tree.map(np.asarray, contour)
    if not bool(contour["well_formed"]):
        raise RuntimeError(f"point count {point_count} produced malformed cells")

    start, end = _cell_edges(radial, vertical)
    mask = contour["edge_crossing"]
    active_start = start[mask]
    active_end = end[mask]
    start_value = _analytic_flux(active_start[:, 0], active_start[:, 1]) - 1.0
    end_value = _analytic_flux(active_end[:, 0], active_end[:, 1]) - 1.0
    truth = _truth_crossings(active_start, active_end)
    linear = _linear_crossings(active_start, active_end, start_value, end_value)
    spline = contour["edge_crossing_rz"][mask]
    linear_error = np.linalg.norm(linear - truth, axis=-1)
    spline_error = np.linalg.norm(spline - truth, axis=-1)

    linear_edge_point = np.zeros_like(start)
    parameter = -(_analytic_flux(start[..., 0], start[..., 1]) - 1.0) / (
        _analytic_flux(end[..., 0], end[..., 1])
        - _analytic_flux(start[..., 0], start[..., 1])
    )
    linear_edge_point = start + parameter[..., None] * (end - start)
    linear_segment = _gather_pairs(linear_edge_point, contour["segment_edge_indices"])
    valid = contour["segment_valid"]
    linear_segment = linear_segment[valid]
    linear_chord = linear_segment[:, 1] - linear_segment[:, 0]
    linear_tangent = linear_chord / np.linalg.norm(linear_chord, axis=-1)[:, None]
    linear_endpoint_tangent = np.repeat(linear_tangent[:, None, :], 2, axis=1)
    spline_segment = contour["segment_endpoints_rz"][valid]
    spline_tangent = contour["segment_endpoint_tangents_rz"][valid]

    row = {
        "radial_cells": point_count - 1,
        "vertical_cells": point_count - 1,
        "total_cells": (point_count - 1) ** 2,
        "maximum_cell_pitch_m": max(
            float(np.diff(radial).max()), float(np.diff(vertical).max())
        ),
        "active_edge_slots": int(mask.sum()),
        "unique_crossings": int(
            np.unique(np.round(truth, decimals=13), axis=0).shape[0]
        ),
        "linear_crossing_rms_error_m": float(np.sqrt(np.mean(linear_error**2))),
        "linear_crossing_max_error_m": float(np.max(linear_error)),
        "spline_crossing_rms_error_m": float(np.sqrt(np.mean(spline_error**2))),
        "spline_crossing_max_error_m": float(np.max(spline_error)),
    }
    continuity = {
        "linear": _grouped_tangent_jumps(linear_segment, linear_endpoint_tangent),
        "global_spline_cubic": _grouped_tangent_jumps(spline_segment, spline_tangent),
    }
    return row, continuity


def _saddle_case(level: float) -> dict[str, Any]:
    radial = np.asarray((-0.82, -0.18, 0.32, 0.91))
    vertical = np.asarray((-0.93, -0.27, 0.23, 0.87))
    mesh_radial, mesh_vertical = np.meshgrid(radial, vertical)
    values = mesh_radial * mesh_vertical
    result = jax.tree.map(
        np.asarray,
        traced_spline_contour(
            jnp.asarray(values),
            jnp.asarray(radial),
            jnp.asarray(vertical),
            jnp.asarray(level),
        ),
    )
    ambiguous_cells = np.argwhere(result["ambiguous_saddle"])
    outcomes = []
    for row, column in ambiguous_cells:
        resolved = bool(result["ambiguous_resolved"][row, column])
        tie_broken = bool(result["ambiguous_tie_broken"][row, column])
        if resolved == tie_broken:
            raise RuntimeError("an ambiguous cell lacks one exclusive outcome")
        outcomes.append(
            {
                "cell_index_vertical_radial": [int(row), int(column)],
                "outcome": "resolved" if resolved else "tie_broken",
                "stationary_point_found": bool(
                    result["saddle_stationary"][row, column]
                ),
                "stationary_rz": result["saddle_rz"][row, column].tolist(),
                "stationary_value": float(result["saddle_value"][row, column]),
                "level": level,
                "segment_edge_pairing": result["segment_edge_indices"][
                    row, column
                ].tolist(),
            }
        )
    return {
        "truth": "psi(R,Z)=R*Z with an analytic saddle at (0,0)",
        "level": level,
        "ambiguous_cell_count": int(len(ambiguous_cells)),
        "resolved_count": int(result["ambiguous_resolved"].sum()),
        "tie_broken_count": int(result["ambiguous_tie_broken"].sum()),
        "outcomes": outcomes,
    }


def _field_differences(
    first: dict[str, Any], second: dict[str, Any]
) -> dict[str, float]:
    differences: dict[str, float] = {}
    if set(first) != set(second):
        raise RuntimeError("contour transformations returned different fields")
    for name in sorted(first):
        left = first[name]
        right = second[name]
        left_array = np.asarray(left)
        right_array = np.asarray(right)
        if left_array.dtype == bool:
            differences[name] = 0.0 if np.array_equal(left_array, right_array) else 1.0
        else:
            differences[name] = float(
                np.max(np.abs(left_array.astype(float) - right_array.astype(float)))
            )
    return differences


def _contract() -> dict[str, Any]:
    radial = jnp.linspace(0.78, 2.32, 17)
    vertical = jnp.linspace(-0.98, 0.98, 19)
    mesh_radial, mesh_vertical = jnp.meshgrid(radial, vertical)
    values = (
        ((mesh_radial - 1.55) / 0.58) ** 2
        + 0.16 * ((mesh_radial - 1.55) / 0.58) ** 4
        + (mesh_vertical / 0.73) ** 2
    )
    level = jnp.asarray(1.0)
    with jax.disable_jit():
        eager = traced_spline_contour(values, radial, vertical, level)
    compiled = jax.jit(
        lambda field, contour_level: traced_spline_contour(
            field, radial, vertical, contour_level
        )
    )(values, level)
    batch = jnp.stack((values, 1.03 * values, 0.97 * values))
    levels = jnp.asarray((1.0, 1.03, 0.97))
    per_slice = [
        traced_spline_contour(field, radial, vertical, contour_level)
        for field, contour_level in zip(batch, levels, strict=True)
    ]
    mapped = jax.vmap(
        lambda field, contour_level: traced_spline_contour(
            field, radial, vertical, contour_level
        )
    )(batch, levels)
    scalar_stacked = jax.tree.map(lambda *items: jnp.stack(items), *per_slice)
    expected_shapes = {
        "edge_crossing": [18, 16, 4],
        "segment_valid": [18, 16, 2],
        "segment_controls_rz": [18, 16, 2, 4, 2],
        "ambiguous_saddle": [18, 16],
    }
    observed_shapes = {
        key: list(np.asarray(compiled[key]).shape) for key in expected_shapes
    }
    jit_field_differences = _field_differences(eager, compiled)
    vmap_field_differences = _field_differences(scalar_stacked, mapped)
    jit_field = max(jit_field_differences, key=jit_field_differences.get)
    vmap_field = max(vmap_field_differences, key=vmap_field_differences.get)
    jit_difference = jit_field_differences[jit_field]
    vmap_difference = vmap_field_differences[vmap_field]
    return {
        "expected_shapes": expected_shapes,
        "observed_shapes": observed_shapes,
        "fixed_shapes_pass": observed_shapes == expected_shapes,
        "comparison_scope": (
            "every returned value, including canonical exact-zero inactive padding"
        ),
        "inactive_padding": "canonical exact zero",
        "jit_field_maximum_absolute_differences": jit_field_differences,
        "jit_maximum_difference_field": jit_field,
        "jit_maximum_absolute_difference": jit_difference,
        "vmap_field_maximum_absolute_differences": vmap_field_differences,
        "vmap_maximum_difference_field": vmap_field,
        "vmap_maximum_absolute_difference": vmap_difference,
        "agreement_tolerance": AGREEMENT_TOLERANCE,
        "jit_agreement_pass": jit_difference <= AGREEMENT_TOLERANCE,
        "vmap_agreement_pass": vmap_difference <= AGREEMENT_TOLERANCE,
    }


def measure(output: Path) -> dict[str, Any]:
    configure_dtypes()
    rows: list[dict[str, Any]] = []
    finest_continuity: dict[str, Any] | None = None
    for point_count in POINT_COUNTS:
        row, continuity = _accuracy_row(point_count)
        rows.append(row)
        finest_continuity = continuity
    if finest_continuity is None:
        raise RuntimeError("the accuracy ladder is empty")
    orders = {
        "linear": _fit_order(rows, "linear_crossing_rms_error_m"),
        "global_spline": _fit_order(rows, "spline_crossing_rms_error_m"),
    }
    saddle_cases = {
        "resolved_offset_level": _saddle_case(0.015),
        "exact_saddle_level": _saddle_case(0.0),
    }
    ambiguous_count = sum(
        case["ambiguous_cell_count"] for case in saddle_cases.values()
    )
    classified_count = sum(
        case["resolved_count"] + case["tie_broken_count"]
        for case in saddle_cases.values()
    )
    contract = _contract()
    clip_path = ROOT / "nova/equilibrium/separatrix_clip.py"
    clip_changed = (
        subprocess.run(
            ["git", "diff", "--quiet", "--", str(clip_path.relative_to(ROOT))],
            cwd=ROOT,
            check=False,
        ).returncode
        != 0
    )
    verdict_checks = {
        "linear_order_in_declared_band": LINEAR_ORDER_BAND[0]
        <= orders["linear"]["estimate"]
        <= LINEAR_ORDER_BAND[1],
        "spline_order_in_declared_band": SPLINE_ORDER_BAND[0]
        <= orders["global_spline"]["estimate"]
        <= SPLINE_ORDER_BAND[1],
        "spline_more_accurate_on_finest_grid": rows[-1]["spline_crossing_rms_error_m"]
        < rows[-1]["linear_crossing_rms_error_m"],
        "global_tangent_jump_below_declared_limit": finest_continuity[
            "global_spline_cubic"
        ]["maximum_unit_tangent_jump"]
        <= TANGENT_JUMP_LIMIT,
        "every_ambiguous_cell_classified": ambiguous_count > 0
        and classified_count == ambiguous_count,
        "resolved_and_tie_broken_cases_exercised": sum(
            case["resolved_count"] for case in saddle_cases.values()
        )
        > 0
        and sum(case["tie_broken_count"] for case in saddle_cases.values()) > 0,
        "fixed_shape_jit_vmap_contract": all(
            contract[key]
            for key in (
                "fixed_shapes_pass",
                "jit_agreement_pass",
                "vmap_agreement_pass",
            )
        ),
        "clip_integration_source_unchanged": not clip_changed,
    }
    payload = {
        "schema": "nova-higher-order-contour-measurement",
        "method": {
            "truth": (
                "psi=((R-1.55)/0.58)^2 + 0.16*((R-1.55)/0.58)^4 + (Z/0.73)^2 at level 1"
            ),
            "linear": "corner-value linear interpolation on each crossed edge",
            "global_spline": (
                "fixed 40-step edge roots of the global not-a-knot tensor spline "
                "with cubic Hermite controls from its gradient"
            ),
            "declared_point_counts_per_axis": list(POINT_COUNTS),
            "declared_order_bands": {
                "linear": list(LINEAR_ORDER_BAND),
                "global_spline": list(SPLINE_ORDER_BAND),
            },
        },
        "crossing_position": {
            "interim_coarse_smoke": INTERIM_COARSE_SMOKE,
            "cell_ladder": rows,
            "fitted_orders": orders,
        },
        "continuity_on_finest_grid": {
            **finest_continuity,
            "tangent_measure": (
                "sign-invariant Euclidean jump between unit tangents at each "
                "shared cell-edge crossing"
            ),
            "curvature": {
                "implemented": False,
                "measured_jump": None,
                "statement": (
                    "This extraction implements continuously tangent cubic arcs; "
                    "quintic curvature matching is not implemented."
                ),
            },
        },
        "ambiguous_saddles": {
            "ambiguous_cell_count": ambiguous_count,
            "classified_cell_count": classified_count,
            "cases": saddle_cases,
            "tie_break_rule": (
                "pair edges (0,1) and (2,3) only when the evaluated interior "
                "stationary value is indistinguishable from the requested level"
            ),
        },
        "contract": contract,
        "clip_integration": {
            "path": "nova/equilibrium/separatrix_clip.py",
            "source_sha256": _sha256(clip_path),
            "working_tree_changed": clip_changed,
            "statement": (
                "The contour output is separate from polygon clip integration; "
                "the clip implementation, fixed capacities, and exact edge "
                "antiderivatives are unchanged."
            ),
        },
        "characterisation_suite": {
            "status": "not_run",
            "tests": list(CHARACTERISATION_TESTS),
            "step_bounds_percent": STEP_BOUNDS_PERCENT,
            "statement": (
                "The pinned STEP clipped-to-contour upper bounds remain 3.92% "
                "for vpr_face and 6.20% for g1_face."
            ),
            "test_log": None,
        },
        "verdict": {
            "checks": verdict_checks,
            "measurement_pass": all(verdict_checks.values()),
            "complete": False,
        },
        "provenance": {
            "source_revision": _source_revision(),
            "hostname": socket.gethostname(),
            "jax_backend": jax.default_backend(),
            "jax_enable_x64": bool(jax.config.jax_enable_x64),
            "benchmark_sha256": _sha256(Path(__file__)),
            "connectivity_sha256": _sha256(
                ROOT / "nova/equilibrium/flux_surface_connectivity.py"
            ),
            "execution": "slurm" if os.environ.get("SLURM_JOB_ID") else "local",
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
    }
    _write_json(output, payload)
    if not payload["verdict"]["measurement_pass"]:
        failed = [name for name, passed in verdict_checks.items() if not passed]
        raise RuntimeError(f"contour measurement failed checks: {failed}")
    return payload


def confirm_characterisation(receipt: Path, test_log: Path) -> dict[str, Any]:
    report = json.loads(receipt.read_text(encoding="utf-8"))
    log = test_log.read_text(encoding="utf-8")
    passed_counts = [int(value) for value in re.findall(r"(\d+) passed", log)]
    if not passed_counts or passed_counts[-1] < 3:
        raise RuntimeError("the characterisation log does not report three passes")
    if re.search(r"\b(?:failed|error)s?\b", log, flags=re.IGNORECASE):
        raise RuntimeError("the characterisation log contains a failure or error")
    report["characterisation_suite"].update(
        {
            "status": "passed",
            "passed_tests": passed_counts[-1],
            "test_log": str(test_log),
            "step_bounds_unchanged": report["characterisation_suite"][
                "step_bounds_percent"
            ]
            == STEP_BOUNDS_PERCENT,
        }
    )
    report["verdict"]["complete"] = bool(
        report["verdict"]["measurement_pass"]
        and report["characterisation_suite"]["step_bounds_unchanged"]
    )
    if not report["verdict"]["complete"]:
        raise RuntimeError("the completed receipt does not pass")
    report["provenance"]["benchmark_sha256"] = _sha256(Path(__file__))
    report["provenance"]["connectivity_sha256"] = _sha256(
        ROOT / "nova/equilibrium/flux_surface_connectivity.py"
    )
    _write_json(receipt, report)
    return report


def check(receipt: Path) -> dict[str, Any]:
    report = json.loads(receipt.read_text(encoding="utf-8"))
    if report["method"]["declared_point_counts_per_axis"] != list(POINT_COUNTS):
        raise ValueError("the declared cell ladder changed")
    if not report["verdict"]["measurement_pass"]:
        raise ValueError("one or more contour measurement checks failed")
    if report["characterisation_suite"]["status"] != "passed":
        raise ValueError("the characterisation suite is not confirmed")
    if report["characterisation_suite"]["step_bounds_percent"] != STEP_BOUNDS_PERCENT:
        raise ValueError("the STEP bounds changed")
    if not report["verdict"]["complete"]:
        raise ValueError("the receipt is incomplete")
    if report["clip_integration"]["working_tree_changed"]:
        raise ValueError("the clip integration path changed")
    ambiguous = report["ambiguous_saddles"]
    if ambiguous["ambiguous_cell_count"] != ambiguous["classified_cell_count"]:
        raise ValueError("not every ambiguous saddle cell has an outcome")
    return report


def _summary(report: dict[str, Any]) -> str:
    orders = report["crossing_position"]["fitted_orders"]
    continuity = report["continuity_on_finest_grid"]
    saddles = report["ambiguous_saddles"]
    return (
        "HIGHER_ORDER_CONTOUR "
        f"linear_order={orders['linear']['estimate']:.6f}+/-"
        f"{orders['linear']['one_sigma_uncertainty']:.6f} "
        f"spline_order={orders['global_spline']['estimate']:.6f}+/-"
        f"{orders['global_spline']['one_sigma_uncertainty']:.6f} "
        f"linear_tangent_jump={continuity['linear']['maximum_unit_tangent_jump']:.6e} "
        "spline_tangent_jump="
        f"{continuity['global_spline_cubic']['maximum_unit_tangent_jump']:.6e} "
        f"ambiguous={saddles['ambiguous_cell_count']} "
        f"complete={str(report['verdict']['complete']).lower()}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("measure", "confirm-characterisation", "check")
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--test-log", type=Path)
    arguments = parser.parse_args()
    if arguments.mode == "measure":
        report = measure(arguments.output)
    elif arguments.mode == "confirm-characterisation":
        if arguments.test_log is None:
            parser.error("confirm-characterisation requires --test-log")
        report = confirm_characterisation(arguments.output, arguments.test_log)
    else:
        report = check(arguments.output)
    print(_summary(report))


if __name__ == "__main__":
    main()
