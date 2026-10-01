"""Count how often each recovery-ladder path runs per certificate solve.

The solver source is not modified.  Each counted path is wrapped at trace time
so the traced program emits an ordered ``jax.debug.callback`` receipt when it
runs that path, making the count a runtime count rather than a trace-time one.
Instruction shares come from the committed census, not a new compile.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import hashlib
import json
import tempfile
import threading
from pathlib import Path

import jax

from nova.equilibrium import fixed_point

ROOT = Path(__file__).resolve().parents[1]
SHARED = ROOT / "docs/figures/forward-solver-route-integrity/operator-sharing"
CALLERS = SHARED / "callers-candidate-300.json"
CENSUS = SHARED / "census-candidate-300.json"
OUTPUT_ROOT = ROOT / "docs/figures/forward-solver-route-integrity/recovery-ladders"
TRIP = "_newton_krylov_inner"
COUNTED = (
    "_rebuilt_model_promotion",
    "_steepest_descent_promotion",
)
PATHS = (TRIP, *COUNTED)

_EVENTS: list[str] = []
_LOCK = threading.Lock()


def _emit(name):
    with _LOCK:
        _EVENTS.append(name)


def _wrap(name, function):
    emit = functools.partial(_emit, name)

    def counted(*args, **kwargs):
        jax.debug.callback(emit, ordered=True)
        return function(*args, **kwargs)

    return counted


@contextlib.contextmanager
def instrumented():
    saved = {}
    for name in PATHS:
        function = getattr(fixed_point, name, None)
        if function is None:
            continue
        saved[name] = function
        setattr(fixed_point, name, _wrap(name, function))
    try:
        yield
    finally:
        for name, function in saved.items():
            setattr(fixed_point, name, function)


def _attribute(events):
    trip = 0
    counts = {}
    trips = {}
    for name in events:
        if name == TRIP:
            trip += 1
            continue
        counts[name] = counts.get(name, 0) + 1
        trips.setdefault(name, []).append(trip)
    return counts, trips, trip


def _instruction_share():
    total = json.loads(CENSUS.read_text())["optimized_instructions"]
    by_owner = {}
    for row in json.loads(CALLERS.read_text()):
        who = None
        for chain, _weight in row.get("callers", []):
            for frame in chain:
                owner = frame.split(":")[0].split(".")[0]
                if owner in COUNTED:
                    who = owner
                    break
            if who is not None:
                break
        if who is None:
            continue
        value = int(row.get("instructions", 0))
        by_owner[who] = by_owner.get(who, 0) + value
    instructions = {}
    share = {}
    for name in COUNTED:
        value = by_owner.get(name, 0)
        instructions[name] = value
        share[name] = value / total
    return {
        "optimized_instructions": total,
        "instructions": instructions,
        "share": share,
        "by_owner": by_owner,
    }


def _redirect(solovev, scratch, figures):
    saved = {}
    for name in ("FIGURE_ROOT", "PART_ROOT", "DIAGNOSTIC_ROOT"):
        saved[name] = getattr(solovev, name)
    # The rendered panel is receipted as a URL under docs/, so the figure root
    # has to stay inside docs/ even when the surrounding run is redirected.
    setattr(solovev, "FIGURE_ROOT", figures)
    setattr(solovev, "PART_ROOT", scratch / "parts")
    setattr(solovev, "DIAGNOSTIC_ROOT", scratch / "diagnostics")
    return saved


def _restore(solovev, saved):
    for name, value in saved.items():
        setattr(solovev, name, value)


def run(rows, output, scratch_root):

    from benchmarks import solovev_certificate

    output.mkdir(parents=True, exist_ok=True)
    scratch = scratch_root
    scratch.mkdir(parents=True, exist_ok=True)
    figures = output / "scratch-figures"
    figures.mkdir(parents=True, exist_ok=True)
    saved = _redirect(solovev_certificate, scratch, figures)
    receipts = []
    try:
        for case, cells in rows:
            _EVENTS[:] = []
            with instrumented():
                row = solovev_certificate._measure(case, cells)
            events = list(_EVENTS)
            counts, trips, trip_total = _attribute(events)
            solver = row.get("solver", {}) if isinstance(row, dict) else {}
            recorded = solver.get("production_telemetry") or {}
            converged = solver.get("converged")
            if converged is None:
                converged = recorded.get("converged")
            per_solve = {}
            per_trip = {}
            for name in COUNTED:
                if name == TRIP:
                    continue
                per_solve[name] = counts.get(name, 0)
                per_trip[name] = (
                    counts.get(name, 0) / trip_total if trip_total else None
                )
            receipts.append(
                {
                    "case": case,
                    "cells": cells,
                    "requested_cells": cells,
                    "converged": converged,
                    "termination": solver.get("termination"),
                    "residual_status": solver.get(
                        "terminal_fixed_point_residual_status"
                    ),
                    "trips": trip_total,
                    "executions": per_solve,
                    "executions_per_trip": per_trip,
                    "fired_in_trips": trips,
                    "terminal_fixed_point_residual": solver.get(
                        "terminal_fixed_point_residual"
                    ),
                }
            )
            _store(output, receipts[-1])
            _merged(output, rows)
    finally:
        _restore(solovev_certificate, saved)
    payload = _merged(output, rows)
    return payload


FENCE_BASE = "52bfefc0a0f02412d8910"


def _row_file(output, case, cells):
    return output / "rows" / ("%s-cells%d.json" % (case, cells))


def _store(output, receipt):
    path = _row_file(output, receipt["case"], receipt["cells"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


def _load(output):
    store = output / "rows"
    receipts = []
    if store.is_dir():
        for path in sorted(store.glob("*.json")):
            receipts.append(json.loads(path.read_text()))
    return receipts


def _merged(output, requested):
    receipts = _load(output)
    have = {(row["case"], row["cells"]) for row in receipts}
    missing = [
        "%s:%d" % (case, cells)
        for case, cells in requested
        if (case, cells) not in have
    ]
    payload = {
        "fence_base_sha": FENCE_BASE,
        "base_sha": _base_sha(),
        "fixed_point_digest": _digest(),
        "instruction_census": _instruction_share(),
        "complete": not missing,
        "missing": missing,
        "verdict": _verdict(receipts),
        "family_verdict": _family_verdict(receipts),
        "rows": receipts,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    (output / "report.md").write_text(_markdown(payload))
    return payload


def _verdict(rows):
    converged = [row for row in rows if row.get("converged")]
    fired = [row for row in converged if sum(row.get("executions", {}).values()) > 0]
    total = sum(sum(row["executions"].values()) for row in rows)
    if not rows:
        return "no certificate row was measured"
    if not converged:
        return (
            "no measured row converged, so ladder activity is reported on "
            "non-converging rows only"
        )
    if fired:
        return (
            "%d of %d converging rows enter a recovery ladder (%d rung "
            "executions over those rows), so the ladders are a hot path on "
            "converging rows"
            % (
                len(fired),
                len(converged),
                sum(sum(r["executions"].values()) for r in fired),
            )
        )
    return (
        "none of the %d converging rows enters a recovery ladder (%d rung "
        "executions across all measured rows), so the ladders are a rare "
        "fallback reached only off the converging path" % (len(converged), total)
    )


def _family_verdict(rows):
    ratios = []
    for row in rows:
        trips = row.get("trips") or 0
        if not trips:
            continue
        for fired in row.get("fired_in_trips", {}).values():
            ratios.append(len(set(fired)) / trips)
    if not ratios:
        return (
            "no ladder rung was observed in the measured rows, so no row "
            "indicates whether the ladders are a fallback or a hot path"
        )
    hot = sum(1 for value in ratios if value >= 0.5)
    converged = [row for row in rows if row.get("converged")]
    if hot == len(ratios):
        where = (
            "converging rows"
            if converged
            else "the measured rows, none of which converged"
        )
        return (
            "every counted ladder rung fires in at least half the trips on %s, "
            "so the ladders are a hot path rather than a rare fallback" % where
        )
    return (
        "%d of %d counted ladder rungs fire in at least half the trips on the "
        "measured rows, so the ladders are part of the working path rather "
        "than a rare fallback" % (hot, len(ratios))
    )


def _base_sha():
    import subprocess

    out = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


def _digest():
    data = (ROOT / "nova/equilibrium/fixed_point.py").read_bytes()
    return hashlib.sha256(data).hexdigest()


def _markdown(payload):
    census = payload["instruction_census"]
    total = census["optimized_instructions"]
    rows = payload["rows"]
    lines = []
    lines.append("# Recovery-ladder executions per certificate solve")
    lines.append("")
    lines.append("Fence base revision `%s`." % payload["fence_base_sha"])
    lines.append("")
    lines.append("Worktree head `%s`." % payload["base_sha"])
    lines.append("")
    lines.append(
        "Complete: `%s`; missing rows: `%s`."
        % (payload["complete"], ", ".join(payload["missing"]) or "none")
    )
    lines.append("")
    lines.append("**Verdict.** %s" % payload["verdict"])
    lines.append("")
    lines.append("**Family verdict.** %s" % payload["family_verdict"])
    lines.append("")
    header = "| path | instructions | share of %d |" % total
    lines.append(header)
    lines.append("| --- | --- | --- |")
    for name in COUNTED:
        if name == TRIP:
            continue
        value = census["instructions"].get(name, 0)
        share = census["share"].get(name, 0.0)
        lines.append("| `%s` | %d | %.4f%% |" % (name, value, share * 100.0))
    lines.append("")
    for row in rows:
        lines.append("## `%s` at %d requested cells" % (row["case"], row["cells"]))
        lines.append("")
        lines.append("Trips: %d." % row["trips"])
        lines.append(
            "Converged `%s`; termination `%s`."
            % (row.get("converged"), row.get("termination"))
        )
        lines.append("")
        lines.append("| path | executions | per trip | trips |")
        lines.append("| --- | --- | --- | --- |")
        for name in COUNTED:
            if name == TRIP:
                continue
            fired = row["fired_in_trips"].get(name, [])
            per = row["executions_per_trip"].get(name)
            trips = ", ".join(str(t) for t in fired) or "none"
            shown = "n/a" if per is None else "%.3f" % per
            lines.append(
                "| `%s` | %d | %s | %s |"
                % (name, row["executions"].get(name, 0), shown, trips)
            )
        lines.append("")
        lines.append(
            "Terminal fixed-point residual: `%s`."
            % row["terminal_fixed_point_residual"]
        )
        lines.append("")
    return "\n".join(lines) + "\n"


def parse_row(value):
    case, _, cells = value.rpartition(":")
    return (case, int(cells))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--row",
        action="append",
        default=[],
        metavar="CASE:CELLS",
        help="certificate row, qualified by requested cells",
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="rewrite the merged report from the stored receipts, solving nothing",
    )
    parser.add_argument("--output", default=str(OUTPUT_ROOT))
    parser.add_argument("--scratch", default="")
    args = parser.parse_args(argv)
    rows = [parse_row(value) for value in args.row]
    scratch = (
        Path(args.scratch)
        if args.scratch
        else Path(tempfile.gettempdir()) / "recovery-ladder-census"
    )
    if args.merge_only and not rows:
        parser.error("--merge-only needs the --row list it should expect")
    payload = (
        _merged(Path(args.output), rows)
        if args.merge_only
        else run(rows, Path(args.output), scratch)
    )
    for row in payload["rows"]:
        print("ROW %s trips=%d" % (row["case"], row["trips"]))
        for name, count in sorted(row["executions"].items()):
            print("  %s executions=%d" % (name, count))
    print("VERDICT %s" % payload["verdict"])
    if payload["complete"]:
        print("CENSUS-OK rows=%d" % len(payload["rows"]))
    else:
        print("CENSUS-PARTIAL rows=%d" % len(payload["rows"]))
        print("MISSING %s" % " ".join(payload["missing"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
