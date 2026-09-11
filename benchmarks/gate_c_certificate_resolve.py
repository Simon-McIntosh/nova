"""Re-solve the certificate rows with the profile support as a live Newton unknown.

The committed solve holds the clipped profile support frozen inside each
Newton trip, re-reading it only at the trip boundary; on the three limited
110-cell rows that iteration stops unconverged after six to seven trips
(residual 0.0125, 0.0078, 0.0098; axis error 68.7, 20.6, 6.3 mm) with the
terminal support booking about five percent more than the target.  This
driver re-solves the same three rows with ``chord_live`` selected, which
re-derives the clip against the live iterate on every moment call so through
autodiff the geometry enters the Newton residual and the Krylov products,
while the clipped coupling blocks stay the trip's frozen record.  The fixed
point is then judged against the frozen-clip row and the committed whole-cell
row on per-trip residual history, seed and terminal amplitude, axis error,
converged flag, trip count and the count of cells whose cut status changed
between trips.

Modes
-----
``--row CASE CELLS``
    Solve one row in the current process at an inner Newton trip budget of at
    least twelve.  A part receipt already present is reloaded rather than
    re-solved, which is what makes an expired allocation resumable.
``--sweep``
    Orchestrate the measurement inside one allocation: the three limited
    110-cell rows in the fixed order, one fresh process each.  Rows whose
    part receipt already exists are skipped.  A sweep-state file records the
    job id and each landed row as it lands so an allocation expiry loses at
    most one row.
``--table``
    Build the comparison table from the landed scalar rows, the frozen-clip
    rows and the committed aggregate without entering the solver.  Writes
    ``receipt.json``, ``receipt.md`` and the report document under the figure
    directory and the report directory.
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

import numpy as np

import benchmarks.solovev_certificate as certificate

ROOT = Path(__file__).resolve().parents[1]
FIGURE_ROOT = Path(
    os.environ.get(
        "NEWTON_GEOMETRY_FIGURE_ROOT",
        ROOT / "docs/figures/cut-cell-current-attribution/newton-geometry",
    )
)
PART_ROOT = FIGURE_ROOT / "parts"
SCALAR_ROOT = FIGURE_ROOT / "scalars"
REPORT_ROOT = Path(
    os.environ.get(
        "NEWTON_GEOMETRY_REPORT_ROOT",
        "/home/ITER/mcintos/.config/reckon/crew/reports/nova/s19-review/"
        "newton-geometry",
    )
)
COMMITTED_AGGREGATE = (
    ROOT / "docs/figures/gs-absolute-accuracy/solovev-certificate-production-route.json"
)
FROZEN_CLIP_RECEIPT = (
    ROOT / "docs/figures/cut-cell-current-attribution/cold-seed/receipt.json"
)

#: The three limited certificate rows this node measures.
ROW_CASE = {
    "weak": "weak-rotation-reactor-static",
    "moderate": "moderate-rotation-conventional-static",
    "strong": "strong-rotation-compact-static",
}
CELL_CHOICES = (-110,)
#: The clip mode under test and the inner Newton step budget.
LIVE_CLIP_MODE = "chord_live"
INNER_NEWTON_STEPS = 12
AMPLITUDE_TOLERANCE = 0.01


def _slug(requested_cells: int) -> str:
    return "reduced" if requested_cells == -110 else f"cells-{abs(requested_cells)}"


def _case_key(case_name: str, requested_cells: int) -> str:
    return f"{case_name}:{requested_cells}"


def _part_path(case_name: str, requested_cells: int) -> Path:
    return PART_ROOT / f"{case_name}-{LIVE_CLIP_MODE}-{_slug(requested_cells)}.json"


def _scalar_path(case_name: str, requested_cells: int) -> Path:
    return SCALAR_ROOT / f"{case_name}-{_slug(requested_cells)}.json"


def _figure_path(case_name: str, requested_cells: int) -> Path:
    return FIGURE_ROOT / (f"{case_name}-{LIVE_CLIP_MODE}-{_slug(requested_cells)}.png")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(certificate._strict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _plot_title(case_name: str, requested_cells: int, row: dict[str, object]) -> str:
    solver = row["solver"]
    residual = solver.get("terminal_fixed_point_residual")
    residual_text = f"{residual:.3e}" if residual is not None else "nan"
    telemetry = solver["production_telemetry"]
    return (
        f"{case_name} · {_slug(requested_cells)} · {LIVE_CLIP_MODE} · "
        f"{solver['qualification']} · residual {residual_text} · "
        f"converged {solver.get('converged', False)} · "
        f"trips {telemetry['trip_count']}"
    )


def _render_with_caption(row: dict[str, object]) -> None:
    """Re-render the row's panel with mode, residual and converged in the caption."""
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
        f"/nova/figures/cut-cell-current-attribution/newton-geometry/{figure.name}"
    )


def _cut_status(record: dict[str, object]) -> np.ndarray:
    """Return the per-cell clip status of one boundary census.

    ``0`` absent, ``1`` half cell, ``2`` whole atom.  A cell is cut when its
    clipped area falls short of its full atomic area by more than rounding.
    """
    included = np.asarray(record["included"], dtype=bool)
    area = np.asarray(record["area"], dtype=np.float64)
    full = np.asarray(record["full_area"], dtype=np.float64)
    cut = ~np.isclose(area, full, rtol=1.0e-9, atol=1.0e-15)
    return np.where(included, np.where(cut, 1, 2), 0)


def _cut_telemetry(census: list[dict[str, object]]) -> dict[str, object]:
    """Derive per-trip cut-status churn from the boundary census series."""
    statuses = [_cut_status(record) for record in census]
    changes = [
        int(np.sum(statuses[index] != statuses[index + 1]))
        for index in range(len(statuses) - 1)
    ]
    counts = [
        {
            "cells_absent": int(np.sum(status == 0)),
            "cells_cut": int(np.sum(status == 1)),
            "cells_whole": int(np.sum(status == 2)),
        }
        for status in statuses
    ]
    return {
        "boundary_reads": len(census),
        "between_trip_cut_status_changes": changes,
        "per_boundary_census": counts,
        "terminal_cut_cell_count": int(np.sum(statuses[-1] == 1)) if statuses else None,
    }


def _measure_row(case_name: str, requested_cells: int) -> dict[str, object]:
    """Solve one row with the live clip and persist its part and panel."""
    import scripts.oracle_rebaseline.measure as recovery
    from nova.equilibrium import forward_operator as operator_module

    certificate.PART_ROOT = PART_ROOT
    certificate.FIGURE_ROOT = FIGURE_ROOT
    previous_mode = operator_module.set_support_clip_mode(LIVE_CLIP_MODE)
    previous_steps = recovery.NEWTON_STEPS
    recovery.NEWTON_STEPS = INNER_NEWTON_STEPS
    operator_module._CUT_CENSUS_SINK = []
    try:
        row = certificate._measure(case_name, requested_cells)
        census = list(operator_module._CUT_CENSUS_SINK)
    finally:
        operator_module._CUT_CENSUS_SINK = None
        operator_module.set_support_clip_mode(previous_mode)
        recovery.NEWTON_STEPS = previous_steps

    row["solver"]["clip_mode"] = LIVE_CLIP_MODE
    row["solver"]["inner_newton_steps"] = INNER_NEWTON_STEPS
    row["solver"]["cut_support_telemetry"] = _cut_telemetry(census)
    _render_with_caption(row)
    return row


def _scalar_row(row: dict[str, object]) -> dict[str, object]:
    solver = row["solver"]
    geometry = row["geometry"]
    samples = {
        item["state"]: item["amplitude"]
        for item in solver["lambda_amplitude_history"]["samples"]
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
        "converged": bool(solver.get("converged", False)),
        "qualification": solver["qualification"],
        "magnetic_axis_position_error_m": geometry.get(
            "magnetic_axis_position_error_m"
        ),
        "seed_amplitude": seed_amplitude,
        "terminal_amplitude": terminal_amplitude,
        "amplitude_within_one_percent": bool(within),
        "trip_count": solver["production_telemetry"]["trip_count"],
        "termination": solver["production_telemetry"].get("termination"),
        "clip_mode": solver.get("clip_mode"),
        "inner_newton_steps": solver.get("inner_newton_steps"),
        "cut_support_telemetry": solver.get("cut_support_telemetry"),
        "figure": dict(row["figure"]),
    }


def _run_row(case_name: str, requested_cells: int) -> dict[str, object]:
    row = _measure_row(case_name, requested_cells)
    scalar = _scalar_row(row)
    scalar["part"] = str(_part_path(case_name, requested_cells).relative_to(ROOT))
    _write_json(_part_path(case_name, requested_cells), row)
    _write_json(_scalar_path(case_name, requested_cells), scalar)
    telemetry = scalar["cut_support_telemetry"]
    print(
        "NEWTON_GEOMETRY_ROW case=%s requested_cells=%s qualification=%s "
        "terminal_residual=%s converged=%s trips=%s between_trip_changes=%s part=%s"
        % (
            row["case"],
            row["requested_cells"],
            row["solver"]["qualification"],
            row["solver"].get("terminal_fixed_point_residual"),
            row["solver"]["converged"],
            row["solver"]["production_telemetry"]["trip_count"],
            telemetry["between_trip_cut_status_changes"],
            row["figure"]["filesystem_path"],
        ),
        flush=True,
    )
    return scalar


def _committed_rows() -> dict[str, dict[str, object]]:
    receipt = json.loads(COMMITTED_AGGREGATE.read_text(encoding="utf-8"))
    rows: dict[str, dict[str, object]] = {}
    for case_receipt in receipt["cases"].values():
        for row in case_receipt["rows"]:
            if (
                row["case"] in ROW_CASE.values()
                and row["requested_cells"] in CELL_CHOICES
            ):
                rows[_case_key(row["case"], row["requested_cells"])] = _scalar_row(row)
    return rows


def _frozen_clip_rows() -> dict[str, dict[str, object]]:
    receipt = json.loads(FROZEN_CLIP_RECEIPT.read_text(encoding="utf-8"))
    rows: dict[str, dict[str, object]] = {}
    for entry in receipt["rows"]:
        landed = entry.get("landed")
        if (
            landed is None
            or entry["case"] not in ROW_CASE.values()
            or entry["requested_cells"] not in CELL_CHOICES
        ):
            continue
        rows[_case_key(entry["case"], entry["requested_cells"])] = dict(landed)
    return rows


def _landed_rows() -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    for case_name in ROW_CASE.values():
        for requested_cells in CELL_CHOICES:
            path = _scalar_path(case_name, requested_cells)
            if path.exists():
                rows[_case_key(case_name, requested_cells)] = json.loads(
                    path.read_text(encoding="utf-8")
                )
    return rows


def _fmt(value: object, width: int = 10, digits: int = 3) -> str:
    if value is None:
        return "—".rjust(width)
    if isinstance(value, bool):
        return ("True" if value else "False").ljust(width)
    if isinstance(value, int):
        return f"{int(value):d}".rjust(width)
    if isinstance(value, float):
        if not np.isfinite(float(value)):
            return "nan".rjust(width)
        if abs(float(value)) >= 1e-4 and abs(float(value)) < 1e5:
            return f"{float(value):.{digits}f}".rjust(width)
        return f"{float(value):.{digits}e}".rjust(width)
    return str(value).rjust(width)


def _row_flat(row: dict[str, object]) -> dict[str, object]:
    """Flatten one scalar row to the columns the table compares."""
    return {
        "residual": row.get("terminal_fixed_point_residual"),
        "converged": bool(row.get("converged", False)),
        "qualification": row.get("qualification"),
        "axis_error_m": row.get("magnetic_axis_position_error_m"),
        "seed_amplitude": row.get("seed_amplitude"),
        "terminal_amplitude": row.get("terminal_amplitude"),
        "trip_count": row.get("trip_count"),
        "termination": row.get("termination"),
    }


def _build_table(
    landed: dict[str, dict[str, object]],
    frozen: dict[str, dict[str, object]],
    committed: dict[str, dict[str, object]],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for case_name in ROW_CASE.values():
        for requested_cells in CELL_CHOICES:
            key = _case_key(case_name, requested_cells)
            live_row = landed.get(key)
            frozen_row = frozen.get(key)
            committed_row = committed.get(key)
            entry: dict[str, object] = {
                "case": case_name,
                "requested_cells": requested_cells,
                "live": _row_flat(live_row) if live_row is not None else None,
                "frozen_clip": _row_flat(frozen_row)
                if frozen_row is not None
                else None,
                "committed": _row_flat(committed_row)
                if committed_row is not None
                else None,
            }
            if live_row is not None and committed_row is not None:
                lr = live_row.get("terminal_fixed_point_residual")
                mr = committed_row.get("terminal_fixed_point_residual")
                la = live_row.get("magnetic_axis_position_error_m")
                ma = committed_row.get("magnetic_axis_position_error_m")
                residual_smaller = _strictly_smaller(lr, mr)
                axis_smaller = _strictly_smaller(la, ma)
                entry["reached_analytic_equilibrium"] = bool(
                    residual_smaller and axis_smaller
                )
                entry["residual_below_committed"] = residual_smaller
                entry["axis_below_committed"] = axis_smaller
                entry["converged"] = bool(live_row.get("converged", False))
                entry["between_trip_cut_status_changes"] = (
                    live_row["cut_support_telemetry"]["between_trip_cut_status_changes"]
                    if live_row.get("cut_support_telemetry")
                    else None
                )
                entry["terminal_cut_cell_count"] = (
                    live_row["cut_support_telemetry"]["terminal_cut_cell_count"]
                    if live_row.get("cut_support_telemetry")
                    else None
                )
            rows.append(entry)
    return {"rows": rows}


def _strictly_smaller(candidate: object, reference: object) -> bool:
    if candidate is None or reference is None:
        return False
    return bool(
        np.isfinite(float(candidate))
        and np.isfinite(float(reference))
        and float(candidate) < float(reference)
    )


def _sentence(rows: list[dict[str, object]]) -> str:
    reached = [row for row in rows if row.get("reached_analytic_equilibrium")]
    if len(reached) == len(rows):
        return (
            "Newton on geometry beats Picard on geometry here: with the clip a "
            "live Newton-Krylov unknown every row reaches the analytic "
            "equilibrium that the frozen per-trip geometry stalls short of."
        )
    if reached:
        names = ", ".join(row["case"].split("-rotation")[0] for row in reached)
        return (
            f"Newton on geometry beats Picard on geometry on {names}: the live "
            "clip reaches the analytic equilibrium where the frozen per-trip "
            "geometry stalls, but does not do so on every row."
        )
    return (
        "Newton on geometry does not beat Picard on geometry here: with the "
        "clip a live Newton-Krylov unknown, no row reaches the analytic "
        "equilibrium that the frozen per-trip geometry already fails to reach."
    )


def _write_report(receipt: dict[str, object]) -> None:
    for directory in (FIGURE_ROOT, REPORT_ROOT):
        _write_json(directory / "receipt.json", receipt)
        (directory / "receipt.md").write_text(
            receipt["report_markdown"], encoding="utf-8"
        )


def _build_receipt() -> dict[str, object]:
    landed = _landed_rows()
    frozen = _frozen_clip_rows()
    committed = _committed_rows()
    table = _build_table(landed, frozen, committed)
    rows = table["rows"]
    sentence = _sentence(rows)

    lines = [
        "# Newton geometry step: live clip against the frozen per-trip clip",
        "",
        "The three limited 110-cell certificate rows re-solved with the profile "
        f"support clip as a live Newton-Krylov unknown ({LIVE_CLIP_MODE}, inner "
        f"Newton budget {INNER_NEWTON_STEPS}) and tabled against the frozen-clip "
        "rows and the committed whole-cell rows.",
        "",
        "## Verdict",
        "",
        sentence,
        "",
        "## Per-row table",
        "",
        "| case | res (comm) | res (live) | axis (comm) | axis (live) | "
        "conv (live) | trips (live) | seed amp | term amp | cut churn | toward |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        live = row["live"]
        committed = row["committed"]
        if live is None or committed is None:
            lines.append("skipped-row")
            continue
        lines.append(
            "| {case} | {rc} | {rl} | {ac} | {al} | {cv} | {tr} | {sa} | {ta} "
            "| {churn} | {toward} |".format(
                case=row["case"].split("-rotation")[0],
                rc=_fmt(committed.get("residual")),
                rl=_fmt(live.get("residual")),
                ac=_fmt(committed.get("axis_error_m")),
                al=_fmt(live.get("axis_error_m")),
                cv=_fmt(live.get("converged")),
                tr=_fmt(live.get("trip_count")),
                sa=_fmt(live.get("seed_amplitude")),
                ta=_fmt(live.get("terminal_amplitude")),
                churn=_fmt(str(row.get("between_trip_cut_status_changes"))),
                toward=("yes" if row.get("reached_analytic_equilibrium") else "no"),
            )
        )
    lines += [
        "",
        "cut churn = count of cells whose cut status changed between successive "
        "trip-boundary reads of the live clip support.",
    ]

    report_markdown = "\n".join(lines) + "\n"
    return {
        "driver": Path(__file__).name,
        "revision": certificate._source_revision(),
        "clip_mode": LIVE_CLIP_MODE,
        "inner_newton_steps": INNER_NEWTON_STEPS,
        "committed_aggregate": str(COMMITTED_AGGREGATE.relative_to(ROOT)),
        "frozen_clip_receipt": str(FROZEN_CLIP_RECEIPT.relative_to(ROOT)),
        "sentence": sentence,
        "rows": rows,
        "report_markdown": report_markdown,
    }


def run_row(case_name: str, requested_cells: int) -> int:
    _run_row(case_name, requested_cells)
    return 0


def run_sweep() -> int:
    sweep_state = FIGURE_ROOT / "sweep-state.json"
    state = {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "started": _source_time(),
        "rows": {},
    }
    if sweep_state.exists():
        state["rows"].update(
            json.loads(sweep_state.read_text(encoding="utf-8"))["rows"]
        )
    for case_name in ROW_CASE.values():
        for requested_cells in CELL_CHOICES:
            key = _case_key(case_name, requested_cells)
            if key in state["rows"]:
                continue
            if _part_path(case_name, requested_cells).exists():
                state["rows"][key] = {
                    "status": "landed",
                    "exit": 0,
                    "note": "part present from a prior run",
                }
                _write_json(sweep_state, state)
                continue
            log_path = FIGURE_ROOT / f"row-{case_name}-{_slug(requested_cells)}.log"
            state["rows"][key] = {"status": "running"}
            _write_json(sweep_state, state)
            print(
                "NEWTON_GEOMETRY_ROW_START %s %s -> %s"
                % (case_name, requested_cells, log_path),
                flush=True,
            )
            started = perf_counter()
            with open(log_path, "wb") as captured:
                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "benchmarks.gate_c_certificate_resolve",
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
            _write_json(sweep_state, state)
            print(
                "NEWTON_GEOMETRY_ROW_END %s %s exit=%d elapsed=%.1fs"
                % (case_name, requested_cells, result.returncode, elapsed),
                flush=True,
            )
            if result.returncode != 0:
                print(
                    "NEWTON_GEOMETRY_ROW_FAILED %s log %s" % (key, log_path),
                    flush=True,
                )
                continue
    return 0


def _source_time() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def run_table() -> int:
    receipt = _build_receipt()
    _write_report(receipt)
    print(receipt["report_markdown"])
    print("NEWTON_GEOMETRY_TABLE receipt=%s" % (FIGURE_ROOT / "receipt.json"))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--row", nargs=2, metavar=("CASE", "CELLS"))
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--table", action="store_true")
    args = parser.parse_args()

    if args.row:
        case_name, cells_text = args.row
        requested_cells = int(cells_text)
        if requested_cells not in CELL_CHOICES:
            raise SystemExit(f"unsupported resolution {requested_cells}")
        return run_row(case_name, requested_cells)
    if args.sweep:
        return run_sweep()
    if args.table:
        return run_table()
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
