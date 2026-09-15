"""Measure exact-clip cold-seed current and optional static solve outcomes.

The production cold seed is evaluated through the same certificate fixture,
operator, and public solve seam as the static accuracy rows.  Seed acceptance
is the operator's analytic current normalization,
``target_current / sum(unscaled_clipped_cell_current)``, bounded to one percent
of unity.  The historical exact-clip row remains beside the current result so
the receipt records the defect this measurement could have reproduced.

Rows are persisted independently before the next build or solve begins.  A
single invocation therefore survives a later row failing without converting
completed evidence into an empty aggregate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
from time import perf_counter
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmarks import solovev_certificate as certificate
from nova.equilibrium import ForwardProfile
from nova.equilibrium.forward_operator import set_support_clip_mode
from nova.equilibrium.stencil_mesh import StencilMesh
from nova.jax.config import configure_dtypes
from nova.media import poloidal
from nova.media.ink import DEFAULT_INK, poloidal_axes
from scripts.analytic_oracle_fixtures import measure as oracle_fixture
from scripts.oracle_rebaseline import measure as recovery


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "docs/figures/cut-cell-current-attribution/exact-clip-seed"
HISTORICAL_RECEIPT = (
    ROOT / "docs/figures/cut-cell-current-attribution/gate-c-resolve/receipt.json"
)
STATIC_CASES = (
    "weak-rotation-reactor-static",
    "moderate-rotation-conventional-static",
    "strong-rotation-compact-static",
)
REFERENCE_CELLS = (-110, -300)
AMPLITUDE_BOUND = 1.0e-2


def _strict(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _strict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
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
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(_strict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _lane() -> dict[str, Any]:
    return {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "node": os.environ.get("SLURM_JOB_NODELIST"),
        "hostname": socket.gethostname(),
        "cpu_count": int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
        "jax_platform": jax.default_backend(),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "tmpdir": os.environ.get("TMPDIR"),
    }


def _historical_rows() -> dict[tuple[str, int], dict[str, Any]]:
    receipt = json.loads(HISTORICAL_RECEIPT.read_text(encoding="utf-8"))
    return {
        (row["case"], int(row["requested_cells"])): {
            "gate_c_exact_clip": row["landed"],
            "committed_whole_cell": row["committed"],
        }
        for row in receipt["rows"]
        if row["case"] in STATIC_CASES
    }


def _problem(case_name: str, requested_cells: int):
    carrier_case, source_case, exact = certificate._case(case_name)
    machine = certificate._case_machine(case_name, carrier_case, exact, requested_cells)
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    analytic = certificate._exact_state(case_name, exact, coordinates)
    empty = oracle_fixture.forward_operator(source_case, machine)
    exact_physical, exterior, _cache = oracle_fixture.cached_fixture_exterior(
        source_case, exact, machine, empty, analytic
    )
    operator = oracle_fixture.forward_operator(source_case, machine, exterior)
    profile = ForwardProfile(
        operator,
        StencilMesh(machine.node, machine.stencil, machine.area),
        newton_steps=recovery.NEWTON_STEPS,
    )
    target_current, centroid, current_receipt = certificate._closed_form_current_target(
        case_name, source_case, operator, exact_physical
    )
    return (
        machine,
        exact,
        analytic,
        operator,
        profile,
        target_current,
        centroid,
        current_receipt,
    )


def _seed_row(
    case_name: str,
    requested_cells: int,
    historical: dict[str, Any],
) -> dict[str, Any]:
    started = perf_counter()
    (
        machine,
        _exact,
        _analytic,
        operator,
        profile,
        target_current,
        centroid,
        current_receipt,
    ) = _problem(case_name, requested_cells)
    seed, requested_class, seed_receipt = certificate._production_seed(
        profile, case_name, target_current, centroid, current_receipt
    )
    moments = operator.cell_current_moments(
        jnp.asarray(seed), requested_class=requested_class
    )
    booked_current = float(jnp.sum(moments.cell_current))
    amplitude = float(target_current / booked_current)
    return {
        "case": case_name,
        "requested_cells": requested_cells,
        "realised_cells": len(machine.node),
        "target_current_a": float(target_current),
        "booked_current_at_unit_amplitude_a": booked_current,
        "seed_amplitude": amplitude,
        "amplitude_error": abs(amplitude - 1.0),
        "amplitude_bound": AMPLITUDE_BOUND,
        "accepted": abs(amplitude - 1.0) <= AMPLITUDE_BOUND,
        "seed": seed_receipt,
        "historical_exact_clip": historical,
        "elapsed_seconds": perf_counter() - started,
    }


def _difference_levels(field: np.ndarray) -> np.ndarray:
    finite = np.abs(np.asarray(field)[np.isfinite(field)])
    upper = float(np.max(finite))
    nonzero = finite[finite > 0.0]
    if nonzero.size == 0:
        return np.asarray([-1.0e-15, 1.0e-15])
    lower = max(float(np.percentile(nonzero, 10.0)), upper * 1.0e-5)
    positive = np.geomspace(lower, upper, 8) if upper > lower else np.asarray([upper])
    return np.concatenate((-positive[::-1], positive))


def _draw_solve_comparison(row: dict[str, Any], path: Path) -> dict[str, Any]:
    """Draw solved/reference contours and signed flux-span difference contours."""

    data = row["render_data"]
    coordinates = np.asarray(data["coordinates_rz_m"], dtype=np.float64)
    solved_values = np.asarray(data["terminal_flux_wb"], dtype=np.float64)
    analytic_values = np.asarray(data["analytic_flux_wb"], dtype=np.float64)
    wall = np.asarray(data["wall_units_rz_m"][0], dtype=np.float64)
    boundary = np.asarray(data["boundary_rz_m"], dtype=np.float64)
    solved_topology = data["terminal_topology"]
    analytic_topology = data["analytic_topology"]
    radial, height, solved = certificate._raster_field(coordinates, solved_values, wall)
    _, _, analytic = certificate._raster_field(coordinates, analytic_values, wall)
    span = max(abs(float(analytic_topology["flux_span_wb"])), np.finfo(np.float64).tiny)
    difference = (solved - analytic) / span
    shared_levels = poloidal.contour_levels(
        np.concatenate((solved.ravel(), analytic.ravel())), count=12
    )
    difference_levels = _difference_levels(difference)
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 5.2), constrained_layout=True)
    wall_units = (wall,)
    poloidal.draw_flux_contours(
        axes[0], radial, height, analytic, shared_levels, color="#3366cc"
    )
    poloidal.draw_flux_contours(
        axes[0], radial, height, solved, shared_levels, color="#cc7722"
    )
    poloidal.draw_boundary(axes[0], boundary[:, 0], boundary[:, 1], color="#3366cc")
    poloidal.draw_flux_contours(
        axes[1], radial, height, difference, difference_levels, color="#7a3e9d"
    )
    for axis in axes:
        poloidal.draw_wall(axis, units=wall_units)
        poloidal.draw_nulls(
            axis,
            magnetic_axis=analytic_topology["axis_rz_m"],
            x_points=analytic_topology["x_point_rz_m"],
            style=DEFAULT_INK.variant(
                axis_marker="^", axis_color="#3366cc", xpoint_color="#3366cc"
            ),
            contain=wall_units,
        )
        poloidal.draw_nulls(
            axis,
            magnetic_axis=solved_topology["axis_rz_m"],
            x_points=solved_topology["x_point_rz_m"],
            style=DEFAULT_INK.variant(
                axis_marker="^", axis_color="#cc7722", xpoint_color="#cc7722"
            ),
            contain=wall_units,
        )
        poloidal_axes(axis)
    axes[0].set_title("analytic blue / solved ochre; shared Wb levels", fontsize=9)
    axes[1].set_title("(solved - analytic) / analytic flux span", fontsize=9)
    axes[1].text(
        0.02,
        0.02,
        "levels: " + ", ".join(f"{level:.2e}" for level in difference_levels),
        transform=axes[1].transAxes,
        fontsize=6,
        va="bottom",
        bbox=DEFAULT_INK.label_bbox,
    )
    figure.suptitle(
        f"{row['case']} · {abs(int(row['requested_cells']))} requested cells"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return {
        "filesystem_path": str(path.relative_to(ROOT)),
        "project_absolute_src": f"/nova/{path.relative_to(ROOT / 'docs')}",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "difference_normalisation": "analytic flux span",
        "difference_levels": difference_levels.tolist(),
    }


def _solve_summary(
    row: dict[str, Any], comparison_figure: dict[str, Any]
) -> dict[str, Any]:
    samples = row["solver"]["lambda_amplitude_history"]["samples"]
    amplitudes = {sample["state"]: sample["amplitude"] for sample in samples}
    root_topology = row["geometry"]["root_topology"]
    exact_topology = row["geometry"]["exact_topology"]
    axis_error = row["geometry"]["magnetic_axis_position_error_m"]
    pitch = float(row["characteristic_pitch_m"])
    return {
        "seed_amplitude": amplitudes["seed"],
        "terminal_amplitude": amplitudes["terminal"],
        "terminal_residual": row["solver"]["terminal_fixed_point_residual"],
        "termination": row["solver"]["production_telemetry"]["termination"],
        "converged": row["solver"]["production_telemetry"]["converged"],
        "axis_error_m": axis_error,
        "axis_error_in_pitch": None if axis_error is None else axis_error / pitch,
        "boundary_flux_wb": root_topology["boundary_flux_wb"],
        "analytic_boundary_flux_wb": exact_topology["boundary_flux_wb"],
        "boundary_flux_error_wb": row["geometry"]["boundary_flux_error_wb"],
        "certificate_figure": row["figure"],
        "comparison_figure": comparison_figure,
        "part": str(
            certificate._part_path(row["case"], row["requested_cells"]).relative_to(
                ROOT
            )
        ),
    }


def _configure_certificate_output(output_root: Path) -> None:
    certificate.FIGURE_ROOT = output_root / "panels"
    certificate.PART_ROOT = output_root / "parts"
    certificate.DIAGNOSTIC_ROOT = output_root / "diagnostics"


def run(output_root: Path, *, solve: bool) -> dict[str, Any]:
    configure_dtypes()
    if not jax.config.jax_enable_x64:
        raise RuntimeError("exact-clip seed measurement requires extended precision")
    set_support_clip_mode("exact")
    _configure_certificate_output(output_root)
    historical = _historical_rows()
    receipt: dict[str, Any] = {
        "$id": "nova.exact-clip-seed-amplitude",
        "revision": _revision(),
        "clip_mode": "exact",
        "lane": _lane(),
        "amplitude_acceptance": {
            "rule": "absolute distance from unit amplitude at most one percent",
            "bound": AMPLITUDE_BOUND,
        },
        "solve_requested": solve,
        "rows": [],
    }
    receipt_path = output_root / "receipt.json"
    for case_name in STATIC_CASES:
        for requested_cells in REFERENCE_CELLS:
            key = (case_name, requested_cells)
            row_path = (
                output_root / "seed-parts" / f"{case_name}-{abs(requested_cells)}.json"
            )
            row = _seed_row(case_name, requested_cells, historical[key])
            _write_json(row_path, row)
            if solve:
                solved = certificate._measure(case_name, requested_cells)
                comparison_path = (
                    output_root
                    / "panels"
                    / f"{case_name}-{abs(requested_cells)}-comparison.png"
                )
                comparison = _draw_solve_comparison(solved, comparison_path)
                row["solve"] = _solve_summary(solved, comparison)
                _write_json(row_path, row)
            receipt["rows"].append(row)
            _write_json(receipt_path, receipt)
            print(
                "EXACT_CLIP_SEED_ROW "
                f"case={case_name} cells={requested_cells} "
                f"amplitude={row['seed_amplitude']:.9f} "
                f"accepted={row['accepted']}",
                flush=True,
            )
    receipt["accepted"] = all(row["accepted"] for row in receipt["rows"])
    receipt["completed_rows"] = len(receipt["rows"])
    _write_json(receipt_path, receipt)
    print(
        f"EXACT_CLIP_SEED_EXIT accepted={receipt['accepted']} "
        f"rows={receipt['completed_rows']}",
        flush=True,
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--solve", action="store_true")
    arguments = parser.parse_args()
    receipt = run(arguments.output_root, solve=arguments.solve)
    return 0 if receipt["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
