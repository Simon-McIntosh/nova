"""Re-solve the Solovev certificate rows and table them against the committed rows.

The committed absolute-accuracy certificate was banked before the profile
support clipped against the signed iterate flux.  This driver re-solves every
row on the current tree through the same per-row measure the committed rows
were produced by (``solovev_certificate._measure``), redirecting its part
receipts and contour panels into this driver's own figure directory so the
committed ``production-route`` artifacts are never touched, and tables each
landed row beside its committed counterpart.

Modes
-----
``--row CASE CELLS``
    Solve one row in the current process.  This is the per-row fresh process
    the sweep launches: the committed machinery is imported and driven with
    its output roots redirected, the panel is re-rendered with the residual
    and converged flag in the caption, and a compact scalar row is persisted
    alongside the full part receipt.  A part receipt already present is
    reloaded rather than re-solved, which is what makes an expired allocation
    resumable.
``--sweep``
    Orchestrate the measurement inside one allocation.  The sixteen rows run
    in the fixed order, cheapest first: for each resolution (110, 300, 500,
    1000 cells) the four cases in order (weak, moderate, strong, diverted),
    one fresh process each.  Rows whose part receipt already exists are
    skipped.  A sweep-state file records the job id and each landed row as it
    lands so an allocation expiry loses at most one row.
``--table``
    Build the comparison table from the landed scalar rows and the committed
    aggregate receipt without entering the solver.  Writes ``receipt.json``
    and ``receipt.md`` under the figure directory and the report directory.

The report states, per case, whether the fixed point moved toward the
analytic equilibrium (axis error and terminal residual both smaller than the
committed row) and quotes the amplitude rule: the plasma current amplitude at
the seed and terminal states is within one percent of unity for the clip in
production.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from typing import Any

import numpy as np

import benchmarks.solovev_certificate as certificate


ROOT = Path(__file__).resolve().parents[1]
FIGURE_ROOT = Path(
    os.environ.get(
        "GATE_C_CERTIFICATE_FIGURE_ROOT",
        ROOT / "docs/figures/cut-cell-current-attribution/gate-c-certificate",
    )
)
PART_ROOT = FIGURE_ROOT / "parts"
SCALAR_ROOT = FIGURE_ROOT / "scalars"
REPORT_ROOT = Path(
    os.environ.get(
        "GATE_C_CERTIFICATE_REPORT_ROOT",
        "/home/ITER/mcintos/.config/reckon/crew/reports/nova/s19-review/"
        "gate-c-certificate",
    )
)
COMMITTED_AGGREGATE_PATH = (
    "docs/figures/gs-absolute-accuracy/solovev-certificate-production-route.json"
)
COMMITTED_AGGREGATE = ROOT / COMMITTED_AGGREGATE_PATH
CASE_NAMES = certificate.CASE_NAMES
CELL_CHOICES = (-110, -300, -500, -1000)
CELL_ORDER = ("-110", "300", "500", "1000")
AMPLITUDE_RULE = (
    "the plasma current amplitude at the seed and terminal states is within one "
    "percent of unity for the clip in production"
)
AMPLITUDE_TOLERANCE = 0.01


def _slug(requested_cells: int) -> str:
    return "reduced" if requested_cells == -110 else f"cells-{abs(requested_cells)}"


def _case_key(case_name: str, requested_cells: int) -> str:
    return f"{case_name}:{requested_cells}"


def _part_path(case_name: str, requested_cells: int) -> Path:
    return PART_ROOT / f"{case_name}-production-route-{_slug(requested_cells)}.json"


def _scalar_path(case_name: str, requested_cells: int) -> Path:
    return SCALAR_ROOT / f"{case_name}-{_slug(requested_cells)}.json"


def _figure_path(case_name: str, requested_cells: int) -> Path:
    return FIGURE_ROOT / f"{case_name}-production-route-{_slug(requested_cells)}.png"


def _plot_title(case_name: str, requested_cells: int, row: dict[str, Any]) -> str:
    solver = row["solver"]
    residual = solver.get("terminal_fixed_point_residual")
    residual_text = f"{residual:.3e}" if residual is not None else "nan"
    return (
        f"{case_name} · {_slug(requested_cells)} · {solver['qualification']} · "
        f"residual {residual_text} · converged {solver.get('converged', False)}"
    )


def _render_with_caption(row: dict[str, Any]) -> None:
    """Re-render the row's panel with residual and converged flag in the caption.

    The committed renderer draws the shared-level line contours of the solved
    and analytic fields with both null sets and the wall; the committed
    ``_measure`` caption carries only case, resolution and qualification, so
    this driver overwrites the panel with the evidence the gate requires.
    """

    data = row["render_data"]
    errors = {
        name: np.asarray(data["error_fields"][name]) for name in certificate.NORM_FIELDS
    }
    geometry = row["geometry"]
    figure = _figure_path(row["case"], row["requested_cells"])
    certificate._plot(
        np.asarray(data["coordinates_rz_m"]),
        np.asarray(data["terminal_flux_wb"]),
        np.asarray(data["analytic_flux_wb"]),
        np.asarray(data["derivative_coordinates_rz_m"]),
        errors,
        np.asarray(data["boundary_rz_m"]),
        np.asarray(data["wall_units_rz_m"][0]),
        geometry["root_topology"],
        geometry["exact_topology"],
        figure,
        _plot_title(row["case"], row["requested_cells"], row),
    )
    row["figure"]["sha256"] = hashlib.sha256(figure.read_bytes()).hexdigest()
    row["figure"]["project_absolute_src"] = (
        f"/nova/figures/cut-cell-current-attribution/gate-c-certificate/{figure.name}"
    )


def _measure_row(case_name: str, requested_cells: int) -> dict[str, Any]:
    """Solve one row through the committed measure with outputs redirected."""
    certificate.PART_ROOT = PART_ROOT
    certificate.FIGURE_ROOT = FIGURE_ROOT
    row = certificate._measure(case_name, requested_cells)
    _render_with_caption(row)
    return row


def _scalar_row(row: dict[str, Any]) -> dict[str, Any]:
    solver = row["solver"]
    geometry = row["geometry"]
    samples = {
        s["state"]: s["amplitude"]
        for s in solver["lambda_amplitude_history"]["samples"]
    }
    seed_amplitude = samples.get("seed")
    terminal_amplitude = samples.get("terminal")
    within = (
        seed_amplitude is not None
        and terminal_amplitude is not None
        and abs(seed_amplitude - 1.0) <= AMPLITUDE_TOLERANCE
        and abs(terminal_amplitude - 1.0) <= AMPLITUDE_TOLERANCE
    )
    return {
        "case": row["case"],
        "requested_cells": row["requested_cells"],
        "realised_cells": row["realised_cells"],
        "terminal_fixed_point_residual": solver.get("terminal_fixed_point_residual"),
        "terminal_fixed_point_residual_status": solver.get(
            "terminal_fixed_point_residual_status"
        ),
        "converged": bool(solver.get("converged", False)),
        "qualification": solver["qualification"],
        "qualification_reason": solver.get("qualification_reason"),
        "qualification_bound": solver.get("qualification_bound"),
        "magnetic_axis_position_error_m": geometry.get(
            "magnetic_axis_position_error_m"
        ),
        "x_point_position_error_m": geometry.get("x_point_position_error_m"),
        "seed_amplitude": seed_amplitude,
        "terminal_amplitude": terminal_amplitude,
        "amplitude_within_one_percent": bool(within),
        "trip_count": solver["production_telemetry"]["trip_count"],
        "termination": solver["production_telemetry"].get("termination"),
        "figure": dict(row["figure"]),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(certificate._strict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _run_row(case_name: str, requested_cells: int) -> dict[str, Any]:
    row = _measure_row(case_name, requested_cells)
    scalar = _scalar_row(row)
    scalar["part"] = str(_part_path(case_name, requested_cells).relative_to(ROOT))
    _write_json(_part_path(case_name, requested_cells), row)
    _write_json(_scalar_path(case_name, requested_cells), scalar)
    print(
        "RUNG_A_ROW_MEASURE case=%s requested_cells=%s qualification=%s "
        "terminal_residual=%s converged=%s part=%s"
        % (
            row["case"],
            row["requested_cells"],
            row["solver"]["qualification"],
            row["solver"].get("terminal_fixed_point_residual"),
            row["solver"]["converged"],
            row["figure"]["filesystem_path"],
        ),
        flush=True,
    )
    return scalar


def _committed_rows() -> dict[str, dict[str, Any]]:
    receipt = json.loads(COMMITTED_AGGREGATE.read_text(encoding="utf-8"))
    scalars: dict[str, dict[str, Any]] = {}
    for case_name, case_receipt in receipt["cases"].items():
        for row in case_receipt["rows"]:
            scalars[_case_key(row["case"], row["requested_cells"])] = _scalar_row(row)
    return scalars


def _landed_rows() -> dict[str, dict[str, Any]]:
    scalars: dict[str, dict[str, Any]] = {}
    for case_name in CASE_NAMES:
        for requested_cells in CELL_CHOICES:
            path = _scalar_path(case_name, requested_cells)
            if path.exists():
                scalars[_case_key(case_name, requested_cells)] = json.loads(
                    path.read_text(encoding="utf-8")
                )
    return scalars


def _fmt(value: Any, width: int = 10, digits: int = 3) -> str:
    if value is None:
        return "—".rjust(width)
    if isinstance(value, bool):
        return ("True" if value else "False").ljust(width)
    if isinstance(value, (int, np.integer)):
        return f"{int(value):d}".rjust(width)
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(float(value)):
            return "nan".rjust(width)
        if abs(float(value)) >= 1e-4 and abs(float(value)) < 1e5:
            return f"{float(value):.{digits}f}".rjust(width)
        return f"{float(value):.{digits}e}".rjust(width)
    return str(value).rjust(width)


def _build_table(
    landed: dict[str, dict[str, Any]], committed: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    orders: list[dict[str, Any]] = []
    for case_name in CASE_NAMES:
        for requested_cells in CELL_CHOICES:
            key = _case_key(case_name, requested_cells)
            landed_row = landed.get(key)
            committed_row = committed.get(key)
            entry: dict[str, Any] = {
                "case": case_name,
                "requested_cells": requested_cells,
                "landed": landed_row,
                "committed": committed_row,
            }
            if landed_row is not None and committed_row is not None:
                lr = landed_row["terminal_fixed_point_residual"]
                cr = committed_row["terminal_fixed_point_residual"]
                la = landed_row["magnetic_axis_position_error_m"]
                ca = committed_row["magnetic_axis_position_error_m"]
                residual_smaller = (
                    lr is not None
                    and cr is not None
                    and np.isfinite(lr)
                    and np.isfinite(cr)
                    and lr < cr
                )
                axis_smaller = (
                    la is not None
                    and ca is not None
                    and np.isfinite(la)
                    and np.isfinite(ca)
                    and la < ca
                )
                entry["terminal_residual_smaller"] = bool(residual_smaller)
                entry["axis_error_smaller"] = bool(axis_smaller)
                entry["toward_analytic"] = bool(residual_smaller and axis_smaller)
                entry["amplitude_rule_met"] = bool(
                    landed_row["amplitude_within_one_percent"]
                    and committed_row["amplitude_within_one_percent"]
                )
            elif landed_row is None and committed_row is not None:
                entry["toward_analytic"] = None
                entry["amplitude_rule_met"] = None
            orders.append(entry)
    return {
        "driver": "rung_a_certificate_resolve",
        "committed_aggregate": str(COMMITTED_AGGREGATE.relative_to(ROOT)),
        "revision": certificate._source_revision(),
        "amplitude_rule": AMPLITUDE_RULE,
        "rows": orders,
    }


def _case_summary(table: dict[str, Any]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for case_name in CASE_NAMES:
        rows = [r for r in table["rows"] if r["case"] == case_name]
        landed = sum(1 for r in rows if r["landed"] is not None)
        toward = [r for r in rows if r["toward_analytic"]]
        residual_smaller = sum(1 for r in rows if r.get("terminal_residual_smaller"))
        axis_smaller = sum(1 for r in rows if r.get("axis_error_smaller"))
        amplitude_landed = [
            r
            for r in rows
            if r["landed"] is not None and r["landed"]["amplitude_within_one_percent"]
        ]
        amplitude_committed = [
            r
            for r in rows
            if r["committed"] is not None
            and r["committed"]["amplitude_within_one_percent"]
        ]
        summaries.append(
            {
                "case": case_name,
                "rows_landed": landed,
                "rows_of": len(rows),
                "terminal_residual_smaller": residual_smaller,
                "axis_error_smaller": axis_smaller,
                "toward_analytic_count": len(toward),
                "amplitude_rule_landed": len(amplitude_landed),
                "amplitude_rule_committed": len(amplitude_committed),
            }
        )
    return summaries


def _render_markdown(table: dict[str, Any], summaries: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append(
        "# Solov'ev certificate re-solve with the signed-flux profile-support clip"
    )
    lines.append("")
    lines.append(
        f"Sixteen certificate rows re-solved on this tree "
        f"(revision `{table['revision']}`) and tabled beside the committed rows from "
        f"`{table['committed_aggregate']}`."
    )
    lines.append("")
    lines.append(f"**Amplitude rule:** {table['amplitude_rule']}.")
    lines.append("")
    lines.append("## Per-case verdict")
    lines.append("")
    lines.append(
        "| case | rows landed | residual smaller | axis error smaller | "
        "both (toward analytic) | amp rule (landed) | amp rule (committed) |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for s in summaries:
        lines.append(
            "| {case} | {landed}/{of} | {rs} | {ax} | {tw} | {al} | {ac} |".format(
                case=s["case"]
                .replace("-rotation-reactor-static", "")
                .replace("-rotation-conventional-static", "")
                .replace("-rotation-compact-static", "")
                .replace("-single-null", ""),
                landed=s["rows_landed"],
                of=s["rows_of"],
                rs=s["terminal_residual_smaller"],
                ax=s["axis_error_smaller"],
                tw=s["toward_analytic_count"],
                al=s["amplitude_rule_landed"],
                ac=s["amplitude_rule_committed"],
            )
        )
    lines.append("")
    lines.append(
        "The fixed point moved toward the analytic equilibrium on a row when the "
        "landed terminal residual *and* the landed axis position error are both "
        "smaller than the committed row's. The amplitude rule is met when the "
        "plasma current amplitude at the seed and terminal states both lie within "
        "one percent of unity."
    )
    lines.append("")
    lines.append("## Per-row table")
    lines.append("")
    header = (
        "| row | cells | res (comm) | res (landed) | conv (comm) | conv (landed) | "
        "axis err (comm) | axis err (landed) | x err (comm) | x err (landed) | "
        "qual (comm) | qual (landed) | seed amp (comm) | seed amp (landed) | "
        "term amp (comm) | term amp (landed) | trips (comm) | trips (landed) | toward |"
    )
    lines.append(header)
    lines.append("|" + "---|" * 19)
    labels = {
        "weak-rotation-reactor-static": "weak",
        "moderate-rotation-conventional-static": "moderate",
        "strong-rotation-compact-static": "strong",
        "diverted-single-null": "diverted",
    }
    for entry in table["rows"]:
        committed_row = entry["committed"]
        landed_row = entry["landed"]
        line = (
            "| {label} | {cells} | {cr} | {lr} | {cc} | {lc} | {ca} | {la} | "
            "{cx} | {lx} | {cq} | {lq} | {csa} | {lsa} | {cta} | {lta} | "
            "{ct} | {lt} | {tw} |"
        )
        lines.append(
            line.format(
                label=labels.get(entry["case"], entry["case"]),
                cells=abs(entry["requested_cells"]),
                cr=_fmt(
                    committed_row["terminal_fixed_point_residual"]
                    if committed_row
                    else None
                ),
                lr=_fmt(
                    landed_row["terminal_fixed_point_residual"] if landed_row else None
                ),
                cc=_fmt(committed_row["converged"] if committed_row else None),
                lc=_fmt(landed_row["converged"] if landed_row else None),
                ca=_fmt(
                    committed_row["magnetic_axis_position_error_m"]
                    if committed_row
                    else None
                ),
                la=_fmt(
                    landed_row["magnetic_axis_position_error_m"] if landed_row else None
                ),
                cx=_fmt(
                    committed_row["x_point_position_error_m"] if committed_row else None
                ),
                lx=_fmt(landed_row["x_point_position_error_m"] if landed_row else None),
                cq=(committed_row["qualification"] if committed_row else None),
                lq=(landed_row["qualification"] if landed_row else None),
                csa=_fmt(committed_row["seed_amplitude"] if committed_row else None),
                lsa=_fmt(landed_row["seed_amplitude"] if landed_row else None),
                cta=_fmt(
                    committed_row["terminal_amplitude"] if committed_row else None
                ),
                lta=_fmt(landed_row["terminal_amplitude"] if landed_row else None),
                ct=_fmt(committed_row["trip_count"] if committed_row else None),
                lt=_fmt(landed_row["trip_count"] if landed_row else None),
                tw=(
                    "yes"
                    if entry["toward_analytic"] is True
                    else ("—" if entry["toward_analytic"] is None else "no")
                ),
            )
        )
    lines.append("")
    lines.append("## Amplitude rule detail (landed rows)")
    lines.append("")
    lines.append(
        "| row | cells | seed amplitude | terminal amplitude | within one percent |"
    )
    lines.append("|---|---|---|---|---|")
    for entry in table["rows"]:
        landed_row = entry["landed"]
        if landed_row is None:
            continue
        lines.append(
            "| {label} | {cells} | {sa} | {ta} | {within} |".format(
                label=labels.get(entry["case"], entry["case"]),
                cells=abs(entry["requested_cells"]),
                sa=_fmt(landed_row["seed_amplitude"]),
                ta=_fmt(landed_row["terminal_amplitude"]),
                within="yes" if landed_row["amplitude_within_one_percent"] else "no",
            )
        )
    lines.append("")
    return "\n".join(lines)


def _build_receipt() -> tuple[dict[str, Any], str]:
    landed = _landed_rows()
    committed = _committed_rows()
    table = _build_table(landed, committed)
    summaries = _case_summary(table)
    table["case_summaries"] = summaries
    markdown = _render_markdown(table, summaries)
    return table, markdown


def _table() -> None:
    table, markdown = _build_receipt()
    _write_json(FIGURE_ROOT / "receipt.json", table)
    (FIGURE_ROOT / "receipt.md").write_text(markdown, encoding="utf-8")
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    (REPORT_ROOT / "receipt.md").write_text(markdown, encoding="utf-8")
    (REPORT_ROOT / "receipt.json").write_text(
        json.dumps(table, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        "RUNG_A_TABLE rows=%d landed=%d"
        % (
            len(table["rows"]),
            sum(1 for r in table["rows"] if r["landed"] is not None),
        ),
        flush=True,
    )


def _sweep_state() -> Path:
    run_root = Path(
        os.environ.get(
            "GATE_C_CERTIFICATE_RUN_ROOT",
            "/home/ITER/mcintos/.config/reckon/crew/runs/"
            "r-20260911T025644895210-cca-gate-c-certificate-resolve",
        )
    )
    run_root.mkdir(parents=True, exist_ok=True)
    return run_root / "sweep-state.json"


def _load_sweep_state() -> dict[str, Any]:
    path = _sweep_state()
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"rows": {}, "completed": False}


def _save_sweep_state(state: dict[str, Any]) -> None:
    _write_json(_sweep_state(), state)


def _sweep() -> None:
    state = _load_sweep_state()
    state.setdefault("job_id", os.environ.get("SLURM_JOB_ID", "unknown"))
    state.setdefault("revision", certificate._source_revision())
    state.setdefault("rows", {})
    state["completed"] = False
    _save_sweep_state(state)
    python = sys.executable
    for requested_cells in CELL_CHOICES:
        for case_name in CASE_NAMES:
            key = _case_key(case_name, requested_cells)
            if _part_path(case_name, requested_cells).exists():
                state["rows"].setdefault(
                    key,
                    {"status": "landed", "exit": 0, "note": "present from a prior run"},
                )
                _save_sweep_state(state)
                continue
            log_path = (
                _sweep_state().parent / f"row-{case_name}-{_slug(requested_cells)}.log"
            )
            state["rows"][key] = {"status": "running"}
            _save_sweep_state(state)
            print(
                "RUNG_A_ROW_START %s %s -> %s" % (case_name, requested_cells, log_path),
                flush=True,
            )
            started = perf_counter()
            with open(log_path, "wb") as captured:
                result = subprocess.run(
                    [
                        python,
                        "-m",
                        "benchmarks.rung_a_certificate_resolve",
                        "--row",
                        case_name,
                        str(requested_cells),
                    ],
                    stdout=captured,
                    stderr=subprocess.STDOUT,
                )
            elapsed = perf_counter() - started
            state["rows"][key] = {
                "status": "landed" if result.returncode == 0 else "failed",
                "exit": result.returncode,
                "elapsed_seconds": round(elapsed, 1),
                "log": str(log_path),
            }
            _save_sweep_state(state)
            print(
                "RUNG_A_ROW_END %s %s exit=%d elapsed=%.1fs"
                % (case_name, requested_cells, result.returncode, elapsed),
                flush=True,
            )
            if result.returncode != 0:
                print(
                    "RUNG_A_ROW_FAILED %s %s see %s"
                    % (case_name, requested_cells, log_path),
                    flush=True,
                )
    state["completed"] = True
    _save_sweep_state(state)
    _table()


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--row",
        nargs=2,
        metavar=("CASE", "CELLS"),
        help="solve one row in this process",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="solve every unlanded row, one fresh process each",
    )
    parser.add_argument(
        "--table",
        action="store_true",
        help="build the comparison table without solving",
    )
    return parser.parse_args()


def main() -> None:
    arguments = _parse()
    if arguments.row is not None:
        case_name, cells_text = arguments.row
        if case_name not in CASE_NAMES:
            raise SystemExit(f"unknown case {case_name!r}")
        requested_cells = int(cells_text)
        if requested_cells not in CELL_CHOICES:
            raise SystemExit(f"unknown cell count {requested_cells}")
        _run_row(case_name, requested_cells)
        print(
            "RUNG_A_ROW_EXIT=0 case=%s requested_cells=%s"
            % (case_name, requested_cells),
            flush=True,
        )
        return
    if arguments.sweep:
        _sweep()
        return
    if arguments.table:
        _table()
        return
    raise SystemExit("one of --row, --sweep or --table is required")


if __name__ == "__main__":
    main()
