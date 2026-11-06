"""Table paired solve timings, device ownership and every lane's precision."""

import argparse
from html import escape
import itertools
import json
from pathlib import Path

import numpy as np

from scripts.prototypes.bank_solve_cycle.report import PAIR_LIMITS, pair_gaps


def table(columns, rows):
    return (
        "<table><thead><tr>"
        + "".join(f"<th>{escape(c)}</th>" for c in columns)
        + "</tr></thead><tbody>"
        + "".join(
            "<tr>" + "".join(f"<td>{escape(str(cell))}</td>" for cell in row) + "</tr>"
            for row in rows
        )
        + "</tbody></table>\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("fragment", type=Path)
    args = parser.parse_args()
    captures = {
        (arm, process): json.loads(
            (args.run / f"{arm}-{process}/capture.json").read_text()
        )
        for arm in ("base", "head")
        for process in (1, 2)
    }
    suites = json.loads((args.run / "suites.json").read_text())
    added = set(suites["head"]["failure_ids"]) - set(suites["base"]["failure_ids"])
    assert not added, added
    negative = (args.run / "negative-test_candidate_inspection_batched.log").read_text()
    assert negative.rstrip().endswith("EXIT=1")
    assert (
        "FAILED "
        "tests/test_candidate_inspection_batched.py::test_candidate_slots_match_individual_evaluation"
        in negative
    )
    physical = (args.run / "physical.log").read_text()
    assert physical.rstrip().endswith("EXIT=0") and "4 passed" in physical
    for kind, state in itertools.product(
        ("limited", "diverted"), ("trial", "converged")
    ):
        assert f"CERTIFICATE {kind} {state}" in physical
    cache_keys = {}
    for arm, process in captures:
        log = (args.run / f"device-{arm}-{process}.log").read_text()
        assert "XLA_FLAGS=unset" in log
        lookups = [
            json.loads(line.split(" ", 1)[1])
            for line in log.splitlines()
            if line.startswith("CACHE_LOOKUP ")
        ]
        solves = [row for row in lookups if row["module"] == "jit_solve"]
        assert len(solves) == 2 and all(row["outcome"] == "hit" for row in solves)
        cache_keys[arm, process] = [row["key"] for row in solves]
    for arm in ("base", "head"):
        assert cache_keys[arm, 1] == cache_keys[arm, 2]
    walls, devices, precisions, retained, summaries = [], [], [], [], []
    for lanes in (1, 16):
        for (arm, process), capture in captures.items():
            row = capture[str(lanes)]
            assert all(row["converged"]) and max(row["residual"]) <= 1e-10
            wall = float(np.median(row["unprofiled_ms"]))
            reads = row["reads"]
            walls.append(
                [
                    lanes,
                    arm,
                    process,
                    ", ".join(map(str, reads)),
                    f"{wall:.3f}",
                    f"{wall / max(reads):.3f}",
                    f"{max(row['residual']):.6g}",
                ]
            )
        for arm in ("base", "head"):
            trace = json.loads((args.run / f"{arm}-1/attribution.json").read_text())
            control = trace["control"]
            assert control["h2d_bytes"] == 2097152 and control["kernels"] > 0
            row = next(row for row in trace["solves"] if row["lanes"] == lanes)
            group = row["groups"]["read candidate loop"]
            assert group["launches"] > 0 and group["device_ms"] > 0
            devices.append(
                [
                    lanes,
                    arm,
                    f"{group['device_ms']:.3f}",
                    group["launches"],
                    ", ".join(map(str, captures[arm, 1][str(lanes)]["reads"])),
                ]
            )
        per_read = {
            arm: [
                float(np.median(captures[arm, process][str(lanes)]["unprofiled_ms"]))
                / max(captures[arm, process][str(lanes)]["reads"])
                for process in (1, 2)
            ]
            for arm in ("base", "head")
        }
        reduction = 100 * (1 - np.mean(per_read["head"]) / np.mean(per_read["base"]))
        summaries.append(
            f"At {lanes} slices, mean wall per batch read falls {reduction:.2f}%: "
            f"base {min(per_read['base']):.2f}–{max(per_read['base']):.2f} ms, "
            f"head {min(per_read['head']):.2f}–{max(per_read['head']):.2f} ms."
        )
        maxima = {key: 0.0 for key in PAIR_LIMITS}
        for left, right in itertools.product((1, 2), repeat=2):
            gaps, _ = pair_gaps(
                captures["base", left][str(lanes)], captures["head", right][str(lanes)]
            )
            retained.append(
                dict(
                    lanes=lanes,
                    base_process=left,
                    head_process=right,
                    gaps={
                        key: np.asarray(value).tolist() for key, value in gaps.items()
                    },
                )
            )
            for key, limit in PAIR_LIMITS.items():
                value = float(np.max(gaps[key]))
                assert np.isfinite(value) and value <= limit, (lanes, key, value, limit)
                maxima[key] = max(maxima[key], value)
        for key, value in maxima.items():
            precisions.append([lanes, key, f"{value:.9g}", PAIR_LIMITS[key]])
    (args.run / "precision-pairs.json").write_text(json.dumps(retained, indent=2))
    base, head = (captures[arm, 1]["1"]["revision"] for arm in ("base", "head"))
    args.fragment.write_text(
        '<section id="bank-candidate-loop-landing">\n<h2>Batched candidate '
        "inspection</h2>\n"
        "<p>The read evaluates candidate support and normal forms with "
        "one vmap over slots. "
        "Admission is masked by the live saddle flags; forms retain only "
        "saddle slots and slot 0. "
        "The selected index still maximizes signed flux over admitted "
        "candidates and falls back to slot 0.</p>\n"
        f"<p>Base <code>{base}</code> and head <code>{head}</code> "
        "ran in one H200 allocation. "
        "Each revision uses its own persistent compilation cache, warmed by process 0. "
        "Processes 1 and 2 are fresh interpreters, both with recorded "
        "solve-cache hits and XLA_FLAGS unset. "
        "Each process warms once and measures five blocked calls per "
        "width. The fixture is "
        "<code>bank_solve_profile.fixture.build(16)</code>; width 1 takes "
        "the first lane.</p>\n"
        + "<p>"
        + " ".join(summaries)
        + "</p>\n"
        + table(
            [
                "Slices",
                "Arm",
                "Process",
                "Per-lane reads",
                "Whole solve median [ms]",
                "Wall / batch read [ms]",
                "Max fresh residual",
            ],
            walls,
        )
        + "<p>Wall per read divides the whole-solve wall by the largest "
        "per-lane read count, "
        "the number of batch read traversals. It includes all solver "
        "work; it is not standalone read latency.</p>\n"
        + table(
            [
                "Slices",
                "Arm (process 1)",
                "Candidate device [ms]",
                "Launches",
                "Per-lane reads",
            ],
            devices,
        )
        + "<p>Source attribution uses the unchanged "
        "<code>bank_solve_trace.groups</code> classifier. "
        "It identifies the nested inspect function and retains separate "
        "curvature, normal-form and fragment-selection groups. "
        "The instrument observes its matrix-multiply kernel and 2,097,152 "
        "host-to-device bytes in both controls. "
        "These are summed device durations, not elapsed stage timings; "
        "shared and unbound ownership remain in the raw ledger.</p>\n"
        + table(
            ["Slices", "Quantity / base scale", "Largest paired gap", "Budget"],
            precisions,
        )
        + "<p>Every head process is compared against every base process "
        "through <code>pair_gaps</code>, "
        "with every lane retained in precision-pairs.json. Flux and "
        "boundary use the base span; "
        "axis and X-point use the minor radius. All solves satisfy the "
        "fresh residual ceiling of 1e-10.</p>\n"
        + "<p>The four 550-cell cases (limited and diverted, trial and "
        "converged) pass on the H200. "
        "Admission flags and selected index agree exactly with "
        "independent slot evaluation; "
        "normal-form leaves agree at rtol 1e-12 and atol 1e-13 with "
        "identical finite masks. "
        "The small CPU fixture includes a positively admitted saddle. "
        "Swapping slots 0 and 1 support results "
        "in a scratch archive fails that fixture with EXIT=1. The "
        "structural regression reproduces at base: "
        "“candidate slots are evaluated by a sequential scan”; the head passes.</p>\n"
        + table(
            ["Suite arm", "Failure count", "Added failures", "Complete"],
            [
                [
                    arm,
                    suites[arm]["failure_count"],
                    len(added),
                    suites[arm]["completed"],
                ]
                for arm in ("base", "head")
            ],
        )
        + "<p>The paired sirius suites run one fresh pytest process per file "
        "with separate basetemps. "
        "Common files cover support reads, tracked reads, the bank solve "
        "driver and the bank solve cycle; "
        "head also runs the light candidate cases. No slow certificate "
        "build runs on CPU.</p>\n"
        + "<p>Raw logs, per-process reads and walls, traces, compiler source "
        "bindings and precision receipts: "
        f"<code>{escape(str(args.run))}</code>.</p>\n</section>\n"
    )
    print(
        "PAIRED_COMPLETE",
        json.dumps(dict(walls=walls, devices=devices, precision=precisions)),
        flush=True,
    )


if __name__ == "__main__":
    main()
