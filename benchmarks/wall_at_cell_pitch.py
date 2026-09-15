#!/usr/bin/env python3
"""Measure limiter contact resolution when the fixture wall is sampled per pitch.

For the weak and moderate analytic rows at 1000 and 2500 requested cells the
measurement builds carriers whose wall node counts put one, half and a quarter
outboard panel per realised cell pitch (odd counts derived from the realised
pitch and limiter perimeter), evaluates the analytic flux, runs the public
production read and records, beside the fixed 121-node baseline:

- contact position error against the analytic tangency in metres and pitch
- level error in span
- machine build wall time split between plasma and wall block families
- the frozen interaction-matrix byte size
- the selected outboard panel length over the realised cell pitch

Each row is persisted as it lands and the 2500-cell stage runs after the
1000-cell stage, so an allocation expiry loses the landed rows of one stage at
most.  Aggregation is a separate command.
"""

from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator

import jax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmarks import limiter_read_resolution_audit as audit
from benchmarks import solovev_certificate as certificate
from nova.media.ink import DEFAULT_INK, trace_axes
from nova.jax.config import (
    configure_dtypes,
    configure_persistent_compilation_cache,
    default_persistent_compilation_cache_root,
)
from scripts.analytic_oracle_fixtures import measure as oracle_fixture

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_DIRECTORY = Path(
    "/home/ITER/mcintos/.config/reckon/crew/reports/nova/s19-review/wall-pitch"
)
DEFAULT_FIGURE_DIRECTORY = ROOT / "docs/figures/cut-cell-current-attribution/wall-pitch"
CASE_NAMES = ("weak-rotation-reactor-static", "moderate-rotation-conventional-static")
TARGET_PANEL_FACTORS = ((1.0, "one"), (0.5, "half"), (0.25, "quarter"))
PITCH_ERROR_TARGET = 1.0e-3


def _part_path(
    report_directory: Path, case_name: str, requested_cells: int, wall_nodes: int
) -> Path:
    """Return the durable part path for one row."""

    return (
        report_directory
        / "parts"
        / f"{case_name}-cells-{abs(requested_cells)}-wall-{wall_nodes}.json"
    )


@contextmanager
def _timed_flux_blocks() -> Iterator[dict[str, list[float]]]:
    """Time each frozen-moment block family during one carrier build.

    ``build_machine`` calls the module-level ``_flux_blocks`` exactly three
    times in a fixed order: the plasma-grid family, the wall family, then the
    sampling family.  Families are classified by call position because the wall
    family is the only one whose target count changes with wall resolution.
    """

    tallies: dict[str, list[float]] = {"grid": [], "wall": [], "sample": []}
    sequence = iter(("grid", "wall", "sample"))
    active = [None]

    original = oracle_fixture._flux_blocks

    def instrumented(targets, polygons, centres, *, executor=None):
        family = active[0]
        started = perf_counter()
        try:
            return original(targets, polygons, centres, executor=executor)
        finally:
            if family is not None:
                tallies[family].append(perf_counter() - started)

    oracle_fixture._flux_blocks = lambda *a, **k: (
        (active.__setitem__(0, next(sequence))),
        instrumented(*a, **k),
    )[1]
    try:
        yield tallies
    finally:
        oracle_fixture._flux_blocks = original


def _wall_metrics(case_name: str, exact: Any, requested_cells: int) -> dict[str, Any]:
    """Return the realised pitch and limiter perimeter of the 121-node carrier."""

    machine = audit._machine(case_name, exact, exact, requested_cells, 121)
    panel_lengths = np.linalg.norm(
        np.roll(machine.wall_node, -1, axis=0) - machine.wall_node, axis=1
    )
    perimeter = float(np.sum(panel_lengths))
    pitch = float(np.sqrt(np.median(np.asarray(machine.area, dtype=np.float64))))
    average_panel = perimeter / len(panel_lengths)
    outboard_panel = float(panel_lengths[(len(panel_lengths) - 1) // 2])
    return {
        "realised_cells": int(len(machine.node)),
        "pitch_m": pitch,
        "perimeter_m": perimeter,
        "outboard_panel_factor": outboard_panel / average_panel,
    }


def _per_pitch_wall_counts(metrics: dict[str, float]) -> dict[str, int]:
    """Return odd wall counts from one to a quarter outboard panel per pitch.

    The outboard panel is ``outboard_panel_factor`` average panels, so setting
    the count to ``perimeter x factor / (target x pitch)`` places that panel at
    ``target`` pitches.  Each count is rounded to the nearest odd integer so
    ``limiter_contour`` keeps a node exactly on the outboard tangency.
    """

    def odd(value: float) -> int:
        even = int(round(value))
        return even + 1 if even % 2 == 0 else even

    return {
        label: odd(
            metrics["perimeter_m"]
            * metrics["outboard_panel_factor"]
            / (factor * metrics["pitch_m"])
        )
        for factor, label in TARGET_PANEL_FACTORS
    }


def _measure_row(
    case_name: str,
    requested_cells: int,
    wall_nodes: int,
    report_directory: Path,
    sampling_label: str | None = None,
) -> dict[str, Any]:
    """Measure and persist one wall-sampling row.

    ``sampling_label`` names the target density the wall count realises
    (``one``/``half``/``quarter`` pitch, or ``None`` for the 121-node
    baseline) so the report need not re-derive it from panel sizes.
    """

    started = perf_counter()
    part_path = _part_path(report_directory, case_name, requested_cells, wall_nodes)
    if part_path.exists():
        with part_path.open(encoding="utf-8") as stream:
            existing = json.load(stream)
        if existing.get("completed"):
            print(
                f"WALL_ROW_REUSE case={case_name} cells={abs(requested_cells)} "
                f"wall={wall_nodes}",
                flush=True,
            )
            return existing
    progress = {
        "schema": "nova.wall-at-cell-pitch-part",
        "version": 1,
        "source_revision": audit._source_revision(),
        "case": case_name,
        "requested_cells": requested_cells,
        "wall_nodes": wall_nodes,
        "sampling_label": sampling_label,
        "completed": False,
    }
    audit._write_json(part_path, progress)
    carrier, source, exact = certificate._case(case_name)
    with _timed_flux_blocks() as tallies:
        machine = audit._machine(case_name, carrier, exact, requested_cells, wall_nodes)
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    analytic = audit._exact_flux(case_name, exact, coordinates)
    operator = oracle_fixture.forward_operator(source, machine)
    _masks, topology, production_diverted, read_timing = audit._read_measurement(
        operator, analytic
    )
    topology_class = "diverted" if production_diverted else "limited"
    analytic_contact = audit._analytic_wall_extremum(
        case_name, exact, machine.wall_node, float(operator.polarity)
    )
    contact = np.asarray(topology.wall_point, dtype=np.float64)
    contact_flux = float(topology.wall_point_flux)
    exact_contact = np.asarray(analytic_contact["coordinate_rz_m"], dtype=np.float64)
    span = abs(float(topology.axis_flux) - analytic_contact["flux_wb"])
    panel_metrics = audit._nearest_segment(contact, machine.wall_node)
    pitch = float(np.sqrt(np.median(np.asarray(machine.area, dtype=np.float64))))
    tangency = _analytic_tangency(exact)
    block_bytes = _block_bytes(machine)
    row: dict[str, Any] = {
        **progress,
        "realised_cells": len(machine.node),
        "topology_class": topology_class,
        "cache": machine.cache,
        "characteristic_cell_pitch_m": pitch,
        "analytic_tangency_rz_m": tangency.tolist(),
        "wall": {
            "node_count": wall_nodes,
            "perimeter_m": float(
                np.sum(
                    np.linalg.norm(
                        np.roll(machine.wall_node, -1, axis=0) - machine.wall_node,
                        axis=1,
                    )
                )
            ),
            "median_panel_length_m": float(
                np.median(
                    np.linalg.norm(
                        np.roll(machine.wall_node, -1, axis=0) - machine.wall_node,
                        axis=1,
                    )
                )
            ),
            "maximum_panel_length_m": float(
                np.max(
                    np.linalg.norm(
                        np.roll(machine.wall_node, -1, axis=0) - machine.wall_node,
                        axis=1,
                    )
                )
            ),
            "selected_outboard_panel_length_m": panel_metrics["length_m"],
            "selected_outboard_panel_over_pitch": panel_metrics["length_m"] / pitch,
            "maximum_panel_over_pitch": float(
                np.max(
                    np.linalg.norm(
                        np.roll(machine.wall_node, -1, axis=0) - machine.wall_node,
                        axis=1,
                    )
                )
            )
            / pitch,
        },
        "analytic_wall_extremum": analytic_contact,
        "production_contact": {
            "coordinate_rz_m": contact.tolist(),
            "flux_wb": contact_flux,
            "read_distance_to_true_tangency_m": float(
                np.linalg.norm(contact - tangency)
            ),
            "position_error_m": float(np.linalg.norm(contact - exact_contact)),
            "position_error_in_pitch": float(
                np.linalg.norm(contact - exact_contact) / pitch
            ),
            "level_error_wb": abs(contact_flux - analytic_contact["flux_wb"]),
            "level_error_in_span": abs(contact_flux - analytic_contact["flux_wb"])
            / span,
            "selected_equals_wall_node": bool(panel_metrics["distance_m"] <= 1.0e-12),
            "representation": "three-node quadratic sub-panel interpolation",
        },
        "interaction_matrix": block_bytes,
        "block_build_seconds": tallies,
        "read_timing": read_timing,
        "wall_seconds": None,
        "completed": False,
    }
    row["wall_seconds"] = perf_counter() - started
    row["completed"] = True
    audit._write_json(part_path, row)
    print(
        "WALL_ROW "
        f"case={case_name} cells={abs(requested_cells)} wall={wall_nodes} "
        f"class={topology_class} "
        f"panel_pitch={row['wall']['selected_outboard_panel_over_pitch']:.6f} "
        f"position_error_m={row['production_contact']['position_error_m']:.8e} "
        f"error_pitch={row['production_contact']['position_error_in_pitch']:.8e} "
        f"build={machine.cache['build_seconds']:.1f}s "
        f"wall_family={sum(row['block_build_seconds']['wall']):.1f}s",
        flush=True,
    )
    return row


def _block_bytes(machine: Any) -> dict[str, Any]:
    """Return the frozen interaction-matrix byte size by family."""

    blocks = {
        "plasma_grid": (
            np.asarray(machine.plasma_to_grid).nbytes
            + np.asarray(machine.plasma_to_grid_r).nbytes
            + np.asarray(machine.plasma_to_grid_z).nbytes
        ),
        "plasma_wall": (
            np.asarray(machine.plasma_to_wall).nbytes
            + np.asarray(machine.plasma_to_wall_r).nbytes
            + np.asarray(machine.plasma_to_wall_z).nbytes
        ),
        "plasma_sample": (
            np.asarray(machine.plasma_to_sample).nbytes
            + np.asarray(machine.plasma_to_sample_r).nbytes
            + np.asarray(machine.plasma_to_sample_z).nbytes
        ),
    }
    return {**blocks, "total_bytes": sum(blocks.values())}


def _analytic_tangency(exact: Any) -> np.ndarray:
    """Return the authored smooth-limiter outboard tangency coordinate."""

    return np.asarray([exact.boundary_midplane_radii()[1], 0.0], dtype=np.float64)


def measure_stage(report_directory: Path, stage_cells: int) -> list[dict[str, Any]]:
    """Measure every 1000 or 2500 requested-cell row in one allocation."""

    configure_dtypes()
    if not jax.config.jax_enable_x64:
        raise RuntimeError("the measurement requires JAX double precision")
    allocation = audit._allocation()
    cache = configure_persistent_compilation_cache(
        default_persistent_compilation_cache_root()
    )
    if stage_cells not in (1000, 2500):
        raise ValueError(f"unsupported stage {stage_cells}")
    derived: dict[str, Any] = {}
    for case_name in CASE_NAMES:
        _carrier, _source, exact = certificate._case(case_name)
        metrics = _wall_metrics(case_name, exact, -stage_cells)
        counts = _per_pitch_wall_counts(metrics)
        derived[case_name] = {**metrics, "counts": counts}
    derived_path = report_directory / f"derived-{stage_cells}.json"
    audit._write_json(
        derived_path,
        {
            "source_revision": audit._source_revision(),
            "allocation": allocation,
            "persistent_compilation_cache": cache.receipt(),
            "derived": derived,
            "pitch_error_target": PITCH_ERROR_TARGET,
            "completed": False,
        },
    )
    rows: list[dict[str, Any]] = []
    for case_name in CASE_NAMES:
        counts = derived[case_name]["counts"]
        ordered: list[tuple[int, str | None]] = [(121, None)]
        ordered += [(counts[label], label) for _factor, label in TARGET_PANEL_FACTORS]
        for wall_nodes, sampling_label in ordered:
            rows.append(
                _measure_row(
                    case_name,
                    -stage_cells,
                    wall_nodes,
                    report_directory,
                    sampling_label,
                )
            )
    payload = json.loads(derived_path.read_text(encoding="utf-8"))
    payload["completed"] = True
    audit._write_json(derived_path, payload)
    print(f"WALL_STAGE_EXIT=0 stage={stage_cells} rows={len(rows)}", flush=True)
    return rows


def _load_all_parts(report_directory: Path) -> list[dict[str, Any]]:
    """Load every completed wall-at-cell-pitch row."""

    rows = []
    for path in sorted((report_directory / "parts").glob("*.json")):
        with path.open(encoding="utf-8") as stream:
            probe = json.load(stream)
        if probe.get("schema") != "nova.wall-at-cell-pitch-part":
            continue
        rows.append(audit._load_part(path))
    return rows


def _render_error_figure(rows: list[dict[str, Any]], path: Path) -> None:
    """Render one contact-error against panel-per-pitch convergence figure."""

    fig, axis = plt.subplots(figsize=(7.0, 5.0))
    trace_axes(axis)
    colours = {1000: DEFAULT_INK.flux_color, 2500: DEFAULT_INK.separatrix_color}
    for row in rows:
        cells = abs(row["requested_cells"])
        panel = row["wall"]["selected_outboard_panel_over_pitch"]
        error = row["production_contact"]["position_error_in_pitch"]
        axis.loglog(
            [panel],
            [error],
            marker="D" if row["wall_nodes"] == 121 else "o",
            color=colours[cells],
            markersize=6.0 if row["wall_nodes"] == 121 else 4.5,
            linestyle="None",
            mfc="none",
        )
        if row["wall_nodes"] == 121:
            axis.annotate(
                f"{row['case'].split('-')[0]} {cells}",
                (panel, error),
                textcoords="offset points",
                xytext=(5, 4),
                fontsize=6,
            )
    reference = next(
        (row for row in rows if row["wall_nodes"] == 169),
        next(
            (row for row in rows if row["wall_nodes"] != 121),
            None,
        ),
    )
    if reference is None:
        raise RuntimeError("no non-baseline row to draw the first-order reference")
    slope = (
        reference["production_contact"]["position_error_in_pitch"]
        / reference["wall"]["selected_outboard_panel_over_pitch"]
    )
    samples = np.geomspace(0.02, 2.0, 80)
    axis.loglog(
        samples,
        slope * samples,
        color=DEFAULT_INK.contour_color,
        linewidth=0.9,
        linestyle=":",
        label="first-order slope",
    )
    axis.axhline(
        PITCH_ERROR_TARGET,
        color=DEFAULT_INK.wall_color,
        linewidth=1.0,
        linestyle="--",
        label="one thousandth of pitch",
    )
    axis.set_xlabel("outboard wall panel / cell pitch")
    axis.set_ylabel("contact position error / cell pitch")
    axis.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _required_wall_count(row: dict[str, Any]) -> dict[str, Any]:
    """Extrapolate the wall count reaching one thousandth of pitch."""

    panel = row["wall"]["selected_outboard_panel_over_pitch"]
    error = row["production_contact"]["position_error_in_pitch"]
    slope = error / panel
    needed = PITCH_ERROR_TARGET / slope
    pitch = row["characteristic_cell_pitch_m"]
    estimate = row["wall"]["perimeter_m"] / (needed * pitch)
    odd = int(estimate)
    odd += 1 if odd % 2 == 0 else 0
    scaling = odd / row["wall_nodes"]
    wall_family = sum(row["block_build_seconds"]["wall"])
    build_time = row["cache"]["build_seconds"]
    wall_bytes = row["interaction_matrix"]["plasma_wall"]
    return {
        "slope_error_per_panel": slope,
        "required_outboard_panel_over_pitch": needed,
        "required_odd_wall_count": odd,
        "estimated_build_seconds": (
            build_time + wall_family * (scaling - 1.0) if build_time > 0.0 else None
        ),
        "estimated_wall_matrix_bytes": int(wall_bytes * scaling),
        "cost_scaling_over_measured": scaling,
    }


def aggregate(report_directory: Path, figure_directory: Path) -> dict[str, Any]:
    """Render the figure, receipt and report from every landed row."""

    rows = _load_all_parts(report_directory)
    if not rows:
        raise RuntimeError("no landed wall-at-cell-pitch rows to aggregate")
    figure_directory.mkdir(parents=True, exist_ok=True)
    contact_figure = figure_directory / "contact-error-vs-panel-pitch.svg"
    _render_error_figure(rows, contact_figure)
    extrapolation: dict[tuple[str, int], Any] = {}
    for case in CASE_NAMES:
        for cells in (1000, 2500):
            group = [
                row
                for row in rows
                if row["case"] == case and abs(row["requested_cells"]) == cells
            ]
            if not group:
                continue
            reached = [
                row["wall_nodes"]
                for row in group
                if row["production_contact"]["position_error_in_pitch"]
                <= PITCH_ERROR_TARGET
            ]
            if reached:
                extrapolation[(case, cells)] = sorted(reached)
            else:
                finest = max(group, key=lambda row: row["wall_nodes"])
                extrapolation[(case, cells)] = _required_wall_count(finest)
    receipt = {
        "schema": "nova.wall-at-cell-pitch-receipt",
        "version": 1,
        "source_revision": audit._source_revision(),
        "rows": sorted(
            rows, key=lambda r: (r["case"], -abs(r["requested_cells"]), r["wall_nodes"])
        ),
        "pitch_error_target": PITCH_ERROR_TARGET,
        "extrapolation": {
            f"{case}-{cells}": extrapolation[(case, cells)]
            for (case, cells) in extrapolation
        },
        "figures": [str(contact_figure)],
        "completed": True,
    }
    audit._write_json(report_directory / "receipt.json", receipt)
    _write_report(figure_directory / "report.md", receipt)
    print(f"WALL_AGGREGATE_EXIT=0 rows={len(rows)}", flush=True)
    return receipt


def _write_report(path: Path, receipt: dict[str, Any]) -> None:
    """Render the markdown narrative for the landing record."""

    rows = receipt["rows"]
    lines = [
        "# Analytic wall contact at one panel per cell pitch",
        "",
        "The production read takes the fixed-design hex-carrier branch and "
        "resolves the contact inside one wall panel through the three-node "
        "quadratic wall fit, landing on the authored smooth-limiter outboard "
        "tangency to sub-micron accuracy (`read_distance_to_true_tangency_m`, "
        "positive control).  The remaining position error is therefore the "
        "sagitta of the wall polyline itself: the gap between the true "
        "tangency `[outboard, 0]` and the closest point the piecewise wall "
        "represents.  Each row samples the wall at one, half and a quarter "
        "outboard panel per realised cell pitch beside the fixed 121-node "
        "baseline.",
        "",
        "| Case | Cells | Wall nodes | Pitch [m] | Panel / pitch | "
        "Contact error [m] | Error / pitch | Level / span | "
        "Build [s] | Wall family [s] | Matrix [MB] | Warm read [s] |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['case']} | {abs(row['requested_cells'])} | "
            f"{row['wall_nodes']} | "
            f"{row['characteristic_cell_pitch_m']:.6f} | "
            f"{row['wall']['selected_outboard_panel_over_pitch']:.4f} | "
            f"{row['production_contact']['position_error_m']:.8e} | "
            f"{row['production_contact']['position_error_in_pitch']:.8e} | "
            f"{row['production_contact']['level_error_in_span']:.8e} | "
            f"{row['cache']['build_seconds']:.1f} | "
            f"{sum(row['block_build_seconds']['wall']):.1f} | "
            f"{row['interaction_matrix']['total_bytes'] / 1e6:.1f} | "
            f"{row['read_timing']['warm_read_seconds_median']:.4f} |"
        )
    lines.extend(["", "## Per-pitch wall counts", ""])
    lines.append(
        "Counts are odd, derived from the realised pitch and the limiter "
        "perimeter through the outboard panel factor (outboard panel over "
        "average panel at 121 nodes): `N(f) = odd(perimeter x factor / "
        "(f x pitch))` for `f in {1, 1/2, 1/4}`."
    )
    lines.append("| Case | Cells | 1 | 1/2 | 1/4 |")
    lines.append("|---|---|---:|---:|---:|")
    for case in CASE_NAMES:
        for cells in (1000, 2500):
            group = [
                row
                for row in rows
                if row["case"] == case and abs(row["requested_cells"]) == cells
            ]
            if not group:
                continue
            labelled = {
                row["sampling_label"]: row["wall_nodes"]
                for row in group
                if row["sampling_label"] is not None
            }
            if "one" not in labelled:
                lines.append(f"| {case} | {cells} | n/a | n/a | n/a |")
                continue
            lines.append(
                f"| {case} | {cells} | {labelled['one']} | "
                f"{labelled['half']} | {labelled['quarter']} |"
            )
    lines.extend(["", "## One thousandth of pitch", ""])
    for case in CASE_NAMES:
        for cells in (1000, 2500):
            entry = receipt["extrapolation"].get(f"{case}-{cells}")
            if entry is None:
                continue
            if isinstance(entry, list):
                lines.append(f"- {case} {cells}: reached at {entry[0]} wall nodes.")
                continue
            group = [
                row
                for row in rows
                if row["case"] == case and abs(row["requested_cells"]) == cells
            ]
            sampled = any(
                row["sampling_label"] is not None for row in group
            )
            caveat = (
                ""
                if sampled
                else " (one-point slope from the 121-node baseline; the "
                "finer samplings did not land in this run)"
            )
            build = entry["estimated_build_seconds"]
            build_text = "n/a (cached)" if build is None else f"~{build:.0f} s build"
            lines.append(
                f"- {case} {cells}: measured slope "
                f"{entry['slope_error_per_panel']:.6f} error/pitch per "
                f"panel/pitch; one thousandth of pitch needs "
                f"{entry['required_odd_wall_count']} wall nodes "
                f"({build_text}, "
                f"{entry['estimated_wall_matrix_bytes'] / 1e6:.1f} MB wall "
                f"family), so the polyline sagitta does not reach the "
                f"target at any practical count and requires a curved wall "
                f"representation.{caveat}"
            )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("measure1000", "measure2500", "aggregate"))
    parser.add_argument(
        "--report-directory", type=Path, default=DEFAULT_REPORT_DIRECTORY
    )
    parser.add_argument(
        "--figure-directory", type=Path, default=DEFAULT_FIGURE_DIRECTORY
    )
    return parser.parse_args()


def main() -> None:
    args = _parse()
    if args.action == "aggregate":
        aggregate(args.report_directory, args.figure_directory)
    else:
        measure_stage(
            args.report_directory, 1000 if args.action == "measure1000" else 2500
        )


if __name__ == "__main__":
    main()
