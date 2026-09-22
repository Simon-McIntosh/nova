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

Every panel this module draws states its terminal fixed-point residual and its
converged flag in the figure title, read from the persisted row, and every
poloidal panel draws both null sets.  ``render_only`` rebuilds the panels from
the committed part receipts alone -- no fixture, no operator, no solve -- and
writes ``render-receipt.json`` recording each figure's title line and the null
glyphs drawn per panel per set, so the reading is auditable without a solve.
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


ANALYTIC_INK_COLOR = "#3366cc"
SOLVED_INK_COLOR = "#cc7722"
RENDER_RECEIPT_NAME = "render-receipt.json"
NULL_GLYPH_KEYS = ("axis", "admitted_saddle", "other_qualified")


def null_glyph_tally(
    axes,
    topology: dict[str, Any],
    color: str,
    wall_units,
    other_x_points: np.ndarray | None = None,
) -> dict[str, int]:
    """Draw one topology's nulls and count the glyphs the panel actually got.

    The counts are read off the marker lines ``draw_nulls`` added rather than
    off its inputs, so a null dropped by the containment filter or carrying a
    non-finite coordinate is counted absent rather than as drawn.  The hollow
    markers of the set's remaining qualified nulls are told from the admitted
    saddle by their face, which ``draw_nulls`` leaves unfilled.
    """

    added = len(axes.lines)
    poloidal.draw_nulls(
        axes,
        magnetic_axis=topology.get("axis_rz_m"),
        x_points=topology.get("x_point_rz_m"),
        other_x_points=other_x_points,
        style=DEFAULT_INK.variant(
            axis_marker="^", axis_color=color, xpoint_color=color
        ),
        contain=wall_units,
    )
    tally = dict.fromkeys(NULL_GLYPH_KEYS, 0)
    for line in axes.lines[added:]:
        glyphs = len(np.atleast_1d(line.get_xdata()))
        if str(line.get_marker()) == str(DEFAULT_INK.axis_marker):
            tally["axis"] += glyphs
            continue
        if str(line.get_marker()) != str(DEFAULT_INK.xpoint_marker):
            continue
        face = line.get_markerfacecolor()
        hollow = isinstance(face, str) and face == "none"
        tally["other_qualified" if hollow else "admitted_saddle"] += glyphs
    return tally


def _null_sets(
    axes,
    analytic_topology: dict[str, Any],
    terminal_topology: dict[str, Any],
    wall_units,
) -> dict[str, dict[str, int]]:
    """Draw both null sets on one poloidal panel and count each set."""

    return {
        "analytic": null_glyph_tally(
            axes, analytic_topology, ANALYTIC_INK_COLOR, wall_units
        ),
        "solved": null_glyph_tally(
            axes, terminal_topology, SOLVED_INK_COLOR, wall_units
        ),
    }


def _row_terminal(part: dict[str, Any]) -> tuple[float | None, bool]:
    """Return the persisted terminal residual and converged flag of one row."""

    solver = part["solver"]
    residual = solver.get("terminal_fixed_point_residual")
    return residual, bool(solver.get("converged"))


def _null_caption(axes, *, y: float = 0.10) -> None:
    """Say which null set is which on a panel that draws both."""

    axes.text(
        0.02,
        y,
        f"nulls: analytic {ANALYTIC_INK_COLOR} / solved {SOLVED_INK_COLOR}",
        transform=axes.transAxes,
        fontsize=6,
        va="bottom",
        bbox=DEFAULT_INK.label_bbox,
    )


def _figure_title(part: dict[str, Any], suffix: str) -> str:
    """Compose the one title line that states this panel's terminal state."""

    residual, converged = _row_terminal(part)
    return (
        f"{part['case']} · {suffix} · residual={residual!r} · converged={converged}"
    )


def _draw_production_route(part: dict[str, Any], path: Path) -> dict[str, Any]:
    """Draw the 2x2 solve panel from a persisted part and receipt it.

    All four panels are poloidal, so all four draw both null sets; the title
    carries the terminal residual and converged flag of the persisted row.
    """

    data = part["render_data"]
    certificate._validate_render_data(data)
    coordinates = np.asarray(data["coordinates_rz_m"], dtype=np.float64)
    terminal_state = np.asarray(data["terminal_flux_wb"], dtype=np.float64)
    analytic_state = np.asarray(data["analytic_flux_wb"], dtype=np.float64)
    derivative_coordinates = np.asarray(
        data["derivative_coordinates_rz_m"], dtype=np.float64
    )
    errors = {
        name: np.asarray(data["error_fields"][name]) for name in certificate.NORM_FIELDS
    }
    boundary = np.asarray(data["boundary_rz_m"], dtype=np.float64)
    wall = np.asarray(data["wall_units_rz_m"][0], dtype=np.float64)
    terminal_topology = data["terminal_topology"]
    analytic_topology = data["analytic_topology"]
    wall_units = (wall,)

    figure, axes = plt.subplots(2, 2, figsize=(10.5, 9.0), constrained_layout=True)
    flux_axis = axes[0, 0]
    radial, height, solved = certificate._raster_field(
        coordinates, terminal_state, wall
    )
    _, _, analytic = certificate._raster_field(coordinates, analytic_state, wall)
    levels = poloidal.contour_levels(
        np.concatenate((solved.ravel(), analytic.ravel())), count=12
    )
    poloidal.draw_flux_contours(
        flux_axis, radial, height, analytic, levels, color=ANALYTIC_INK_COLOR
    )
    poloidal.draw_wall(flux_axis, units=wall_units)
    poloidal.draw_flux_contours(
        flux_axis, radial, height, solved, levels, color=SOLVED_INK_COLOR
    )
    poloidal.draw_boundary(
        flux_axis, boundary[:, 0], boundary[:, 1], color=ANALYTIC_INK_COLOR
    )
    poloidal_axes(flux_axis)
    flux_axis.set_title(
        "analytic blue contours and nulls / solved ochre\n"
        "contours and nulls; shared Wb levels",
        fontsize=8,
    )
    panels: list[dict[str, Any]] = [
        {
            "panel": "flux",
            "null_glyphs": _null_sets(
                flux_axis, analytic_topology, terminal_topology, wall_units
            ),
        }
    ]
    for axis, name, points in zip(
        (axes[0, 1], axes[1, 0], axes[1, 1]),
        certificate.NORM_FIELDS,
        (
            coordinates[: len(errors["psi"])],
            derivative_coordinates,
            derivative_coordinates,
        ),
        strict=True,
    ):
        certificate._draw_error_contours(
            axis, points, errors[name], wall, boundary, name
        )
        _null_caption(axis)
        panels.append(
            {
                "panel": f"{name}_error",
                "null_glyphs": _null_sets(
                    axis, analytic_topology, terminal_topology, wall_units
                ),
            }
        )
    title = _figure_title(part, certificate._slug(part["requested_cells"]))
    figure.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    residual, converged = _row_terminal(part)
    return {
        "filesystem_path": str(path.relative_to(ROOT)),
        "project_absolute_src": f"/nova/{path.relative_to(ROOT / 'docs')}",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "title": title,
        "residual": residual,
        "converged": converged,
        "poloidal_panels": panels,
    }


def _draw_solve_comparison(part: dict[str, Any], path: Path) -> dict[str, Any]:
    """Draw solved/reference contours and signed flux-span difference contours."""

    data = part["render_data"]
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
        axes[0], radial, height, analytic, shared_levels, color=ANALYTIC_INK_COLOR
    )
    poloidal.draw_flux_contours(
        axes[0], radial, height, solved, shared_levels, color=SOLVED_INK_COLOR
    )
    poloidal.draw_boundary(
        axes[0], boundary[:, 0], boundary[:, 1], color=ANALYTIC_INK_COLOR
    )
    poloidal.draw_flux_contours(
        axes[1], radial, height, difference, difference_levels, color="#7a3e9d"
    )
    panels: list[dict[str, Any]] = []
    for panel_name, axis in (("shared_levels", axes[0]), ("difference", axes[1])):
        poloidal.draw_wall(axis, units=wall_units)
        counts = _null_sets(axis, analytic_topology, solved_topology, wall_units)
        poloidal_axes(axis)
        _null_caption(axis, y=0.10 if panel_name == "difference" else 0.02)
        panels.append({"panel": panel_name, "null_glyphs": counts})
    axes[0].set_title(
        "analytic blue contours and nulls / solved ochre\n"
        "contours and nulls; shared Wb levels",
        fontsize=8,
    )
    axes[1].set_title("(solved - analytic) / analytic flux span", fontsize=8)
    axes[1].text(
        0.02,
        0.02,
        "levels: " + ", ".join(f"{level:.2e}" for level in difference_levels),
        transform=axes[1].transAxes,
        fontsize=6,
        va="bottom",
        bbox=DEFAULT_INK.label_bbox,
    )
    title = _figure_title(part, f"{abs(int(part['requested_cells']))} requested cells")
    figure.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return {
        "title": title,
        "residual": _row_terminal(part)[0],
        "converged": _row_terminal(part)[1],
        "poloidal_panels": panels,
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


def _panel_paths(
    output_root: Path, case_name: str, requested_cells: int
) -> tuple[Path, Path]:
    panels = output_root / "panels"
    slug = certificate._slug(requested_cells)
    return (
        panels / f"{case_name}-production-route-{slug}.png",
        panels / f"{case_name}-{abs(requested_cells)}-comparison.png",
    )


def _figure_record(figure: dict[str, Any]) -> dict[str, Any]:
    return {
        "figure": figure["filesystem_path"],
        "project_absolute_src": figure["project_absolute_src"],
        "sha256": figure["sha256"],
        "title": figure["title"],
        "residual": figure["residual"],
        "converged": figure["converged"],
        "poloidal_panels": figure["poloidal_panels"],
    }


def render_only(output_root: Path) -> dict[str, Any]:
    """Rebuild every panel from the committed part receipts, with no solve.

    The part receipts carry each row's complete render input and its terminal
    solver state, so this path reaches no fixture, no operator and no solver:
    it reads JSON, rasterises the stored fields and writes the panels plus the
    render receipt that records each title line and null-glyph count.
    """

    _configure_certificate_output(output_root)
    figures: list[dict[str, Any]] = []
    for case_name in STATIC_CASES:
        for requested_cells in REFERENCE_CELLS:
            part_path = certificate._part_path(case_name, requested_cells)
            part = json.loads(part_path.read_text(encoding="utf-8"))
            production_path, comparison_path = _panel_paths(
                output_root, case_name, requested_cells
            )
            production = _draw_production_route(part, production_path)
            comparison = _draw_solve_comparison(part, comparison_path)
            part["figure"] = {
                "filesystem_path": production["filesystem_path"],
                "project_absolute_src": production["project_absolute_src"],
                "sha256": production["sha256"],
                "render_source": "persisted_part_receipt",
            }
            _write_json(part_path, part)
            figures.append(_figure_record(production))
            figures.append(_figure_record(comparison))
            print(
                "EXACT_CLIP_SEED_RENDER "
                f"case={case_name} cells={requested_cells} "
                f"title={production['title']}",
                flush=True,
            )
    receipt = {
        "$id": "nova.exact-clip-seed-render-receipt",
        "revision": _revision(),
        "clip_mode": "exact",
        "figures": figures,
    }
    receipt["figure_count"] = len(figures)
    _write_json(output_root / RENDER_RECEIPT_NAME, receipt)
    return receipt


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
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="rebuild the panels from the committed part receipts, with no solve",
    )
    arguments = parser.parse_args()
    if arguments.render_only:
        receipt = render_only(arguments.output_root)
        return 0 if receipt["figure_count"] else 1
    receipt = run(arguments.output_root, solve=arguments.solve)
    return 0 if receipt["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
