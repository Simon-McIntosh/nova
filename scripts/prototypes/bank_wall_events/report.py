"""Check paired precision receipts and table direct and attributed solve costs."""

import argparse
from html import escape
import json
from pathlib import Path

import numpy as np

from scripts.prototypes.bank_solve_cycle.report import pair_gaps, PAIR_LIMITS
from scripts.prototypes.bank_solve_cycle.receipts import assemble
from scripts.prototypes.bank_wall_events.suite import BASE_FILES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("fragment", type=Path)
    args = parser.parse_args()
    captures = {
        arm: json.loads((args.run / arm / "capture.json").read_text())
        for arm in ("base", "head")
    }
    traces = {
        arm: json.loads((args.run / arm / "attribution.json").read_text())
        for arm in ("base", "head")
    }
    suites = {}
    for arm in ("base", "head"):
        names = (
            BASE_FILES if arm == "base" else ("test_wall_events_batched", *BASE_FILES)
        )
        first_log = args.run / f"{arm}-{names[0]}.log"
        revision = first_log.read_text().splitlines()[0].split()[2]
        suites[arm] = assemble(
            args.run, [f"{arm}-{name}" for name in names], revision, f"{arm}-suite.log"
        )
    added = set(suites["head"]["failure_ids"]) - set(suites["base"]["failure_ids"])
    assert not added, added
    negative = (args.run / "negative-test_wall_events_batched.log").read_text()
    assert negative.rstrip().endswith("EXIT=1")
    assert (
        "FAILED tests/test_wall_events_batched.py::"
        "test_seed_brackets_and_candidate_order"
        in negative
    )
    assert (
        "tests/test_wall_events_batched.py::test_seed_brackets_and_candidate_order"
        not in suites["head"]["failure_ids"]
    )
    (args.run / "suites.json").write_text(json.dumps(suites, indent=2))
    rows, measurements = [], []
    for index, lanes in enumerate((1, 16)):
        before, after = (captures[arm][str(lanes)] for arm in ("base", "head"))
        gaps, _ = pair_gaps(before, after)
        maxima = {name: float(np.max(gaps[name])) for name in PAIR_LIMITS}
        assert all(
            np.isfinite(maxima[name]) and maxima[name] <= limit
            for name, limit in PAIR_LIMITS.items()
        ), maxima
        times = {}
        for arm in ("base", "head"):
            capture = captures[arm][str(lanes)]
            trace = traces[arm]["solves"][index]
            assert trace["lanes"] == lanes
            assert all(capture["converged"])
            assert all(np.isfinite(capture["residual"]))
            assert max(capture["residual"]) <= 1e-10
            wall = float(np.median(capture["unprofiled_ms"]))
            group = trace["groups"]["read wall events"]
            assert group["launches"] > 0 and group["device_ms"] > 0
            times[arm] = dict(wall_ms=wall, **group)
            steps = f"{min(capture['steps'])}–{max(capture['steps'])}"
            reads = f"{min(capture['reads'])}–{max(capture['reads'])}"
            precision = (
                "reference"
                if arm == "base"
                else "/".join(f"{maxima[name]:.3g}" for name in PAIR_LIMITS)
            )
            rows.append(
                f"<tr><td>{lanes}</td><td>{arm}</td>"
                f"<td>{group['device_ms']:.3f}</td>"
                f"<td>{group['launches']:,}</td><td>{wall:.3f}</td>"
                f"<td>{wall / lanes:.3f}</td><td>{steps}</td><td>{reads}</td>"
                f"<td>{max(capture['residual']):.3g}</td><td>{precision}</td></tr>"
            )
        row = dict(
            lanes=lanes,
            pair_gaps=maxima,
            times=times,
            whole_solve_saved_ms=times["base"]["wall_ms"] - times["head"]["wall_ms"],
            attributed_saved_ms=times["base"]["device_ms"] - times["head"]["device_ms"],
        )
        measurements.append(row)
        print("PAIRED", json.dumps(row), flush=True)
    (args.run / "paired.json").write_text(json.dumps(measurements, indent=2))
    deltas = " ".join(
        f"At {row['lanes']} slices, direct whole-solve saving is "
        f"{row['whole_solve_saved_ms']:.3f} ms and attributed wall-event saving is "
        f"{row['attributed_saved_ms']:.3f} ms."
        for row in measurements
    )
    base_revision = captures["base"]["1"]["revision"]
    head_revision = captures["head"]["1"]["revision"]
    args.fragment.write_text(
        '<section id="bank-wall-events-landing">\n'
        "<h2>Batched wall-edge polishing</h2>\n"
        "<p>The outer edge map in <code>topology._wall_events</code> is vectorized. "
        "The nine seeds, seed-bracket clip, Newton polish, stationarity test and "
        "flattened candidate order are unchanged.</p>\n"
        f"<p>Archived base <code>{base_revision}</code> and head "
        f"<code>{head_revision}</code> ran in fresh processes in one H200 allocation. "
        "The fixture is <code>bank_solve_profile.fixture.build(16)</code>; the "
        "one-slice case takes its first lane. Each shape is compiled and warmed, "
        "then measured five times without profiling and once with a device trace. "
        "The direct wall is the median of the five blocked whole-solve calls. "
        "All operands remain runtime arguments and x64 is asserted "
        "before construction.</p>\n"
        '<table id="bank-wall-events-measurement"><thead><tr>'
        "<th>Slices</th><th>Arm</th><th>Wall events [device ms]</th><th>Launches</th>"
        "<th>Whole solve [ms]</th><th>Per slice [ms]</th><th>Updates/lane</th>"
        "<th>Reads/lane</th><th>Max fresh residual</th>"
        "<th>Max paired gaps: flux / axis / X-point / boundary</th>"
        "</tr></thead><tbody>" + "\n".join(rows) + "</tbody></table>\n"
        f"<p>{deltas} The wall-event attribution uses the existing "
        "<code>bank_solve_trace.analyse</code> and <code>groups.py</code> "
        "source bindings. "
        "CUDA graph events can inherit the enclosing compiled loop, so attributed "
        "device time is an estimate of group ownership. The direct paired whole-solve "
        "wall measures the effect on elapsed time, including any change in iteration "
        "trajectory. Neither measure isolates an arithmetic-only saving.</p>\n"
        "<p>Every lane satisfies a fresh residual at most 1e-10. "
        "<code>bank_solve_cycle.report.pair_gaps</code> checks flux against 1e-8 "
        "of the base span, axis and X-point against 5e-4 of minor radius, and "
        "boundary flux against 1e-4 of span. Per-lane values and both arms’ oracle "
        "observations remain in the raw receipts. Each trace instrument sees its "
        "known matrix-multiply kernel and 2,097,152-byte host-to-device control.</p>\n"
        '<p id="bank-wall-events-tests">The limited and diverted 550-cell trial and '
        "converged states match an independent edge-by-edge polish within 1e-13 "
        "relative for points and finite values, with identical nonfinite masks and "
        "candidate ordering. The controlled quadratic case matches exactly; removing "
        "the first edge’s bracket clip in a scratch archive fails "
        "<code>test_seed_brackets_and_candidate_order</code> with point error 0.675 "
        "(negative log EXIT=1). The structural check fails at base with "
        "“wall edges are evaluated by a sequential scan” and passes at head. "
        "Paired Sirius suites run each named file in a fresh process and its own "
        f"basetemp: base failure count {suites['base']['failure_count']}, head "
        f"{suites['head']['failure_count']}, added failures {len(added)}. "
        "The head adds the wall-event file to the four common files: "
        "<code>test_support_read</code>, <code>test_tracked_read</code>, "
        "<code>test_bank_solve_driver</code> and "
        "<code>test_bank_solve_cycle</code>.</p>\n"
        "<p>Raw receipts, traces, compiler provenance, kernel ledgers and logs: "
        f"<code>{escape(str(args.run))}</code>.</p>\n</section>\n"
    )


if __name__ == "__main__":
    main()
