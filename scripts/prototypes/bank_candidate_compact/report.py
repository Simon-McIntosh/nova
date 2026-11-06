"""Render measured support compaction, paired solves, and guard evidence."""

import argparse
from html import escape
from itertools import product
import json
from pathlib import Path
import re

import numpy as np

from scripts.prototypes.bank_candidate_loop.report import table
from scripts.prototypes.bank_solve_cycle.report import PAIR_LIMITS, pair_gaps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("fragment", type=Path)
    args = parser.parse_args()
    captures = {
        (arm, process): json.loads(
            (args.run / f"{arm}-{process}/capture.json").read_text()
        )
        for arm, process in product(("base", "head"), (1, 2))
    }
    keys = {}
    for arm, process in captures:
        log = (args.run / f"device-{arm}-{process}.log").read_text()
        assert log.rstrip().endswith("EXIT=0") and "XLA_FLAGS=unset" in log
        events = [
            json.loads(line.split(" ", 1)[1])
            for line in log.splitlines()
            if line.startswith("CACHE_LOOKUP ")
        ]
        solves = [event for event in events if event["module"] == "jit_solve"]
        assert len(solves) == 2 and all(event["outcome"] == "hit" for event in solves)
        keys[arm, process] = [event["key"] for event in solves]
    assert all(keys[arm, 1] == keys[arm, 2] for arm in ("base", "head"))
    walls, devices, precision, retained, summaries = [], [], [], [], []
    for lanes in (1, 16):
        for (arm, process), capture in captures.items():
            row = capture[str(lanes)]
            assert all(row["converged"]) and max(row["residual"]) <= 1e-10
            wall = float(np.median(row["unprofiled_ms"]))
            walls.append(
                [
                    lanes,
                    arm,
                    process,
                    ", ".join(map(str, row["reads"])),
                    f"{wall:.3f}",
                    f"{wall / lanes:.3f}",
                    f"{wall / max(row['reads']):.3f}",
                    f"{max(row['residual']):.6g}",
                ]
            )
        for arm in ("base", "head"):
            trace = json.loads((args.run / f"{arm}-1/attribution.json").read_text())
            assert trace["control"]["h2d_bytes"] == 2097152
            assert trace["control"]["kernels"] > 0
            row = next(row for row in trace["solves"] if row["lanes"] == lanes)
            group = row["groups"]["read candidate loop"]
            assert group["launches"] > 0 and group["device_ms"] > 0
            reads = max(captures[arm, 1][str(lanes)]["reads"])
            devices.append(
                [
                    lanes,
                    arm,
                    f"{group['device_ms']:.3f}",
                    group["launches"],
                    reads,
                    f"{group['device_ms'] / reads:.3f}",
                    group["launches"] / reads,
                ]
            )
        maxima = {key: 0.0 for key in PAIR_LIMITS}
        for left, right in product((1, 2), repeat=2):
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
        precision.extend(
            [lanes, key, f"{value:.9g}", PAIR_LIMITS[key]]
            for key, value in maxima.items()
        )
        per_read = {
            arm: np.mean(
                [
                    np.median(captures[arm, process][str(lanes)]["unprofiled_ms"])
                    / max(captures[arm, process][str(lanes)]["reads"])
                    for process in (1, 2)
                ]
            )
            for arm in ("base", "head")
        }
        summaries.append(
            f"At {lanes} {'slice' if lanes == 1 else 'slices'}, "
            "mean wall per batch read falls "
            f"{100 * (1 - per_read['head'] / per_read['base']):.2f}% "
            f"({per_read['base']:.3f} to {per_read['head']:.3f} ms)."
        )
    (args.run / "precision-pairs.json").write_text(json.dumps(retained, indent=2))
    suites = json.loads((args.run / "suites.json").read_text())
    assert all(arm["completed"] for arm in suites.values())
    added = set(suites["head"]["failure_ids"]) - set(suites["base"]["failure_ids"])
    assert not added, added
    suite_rows = []
    for arm, receipt in suites.items():
        text = Path(receipt["log_path"]).read_text()
        counts = {
            name: sum(map(int, re.findall(r"(\d+) " + name, text)))
            for name in ("passed", "skipped")
        }
        suite_rows.append(
            [
                arm,
                counts["passed"],
                counts["skipped"],
                receipt["failure_count"],
                len(added),
                receipt["completed"],
            ]
        )
    controls = []
    for label, identity in (
        ("overflow", "test_candidate_overflow_matches_full_evaluation"),
        ("unrolled", "test_candidate_slots_share_one_program"),
    ):
        log = (args.run / f"{label}-test_candidate_inspection_batched.log").read_text()
        assert log.rstrip().endswith("EXIT=1")
        failures = re.findall(r"^FAILED\s+(\S+::\S+)", log, re.MULTILINE)
        assert any(name.endswith("::" + identity) for name in failures)
        assert not any(
            name.endswith("::" + identity) for name in suites["head"]["failure_ids"]
        )
        controls.append([label, identity, 1])
    physical = (args.run / "physical.log").read_text()
    assert physical.rstrip().endswith("EXIT=0") and "4 passed" in physical
    rows = re.findall(
        r"CERTIFICATE (limited|diverted) (trial|converged)\n"
        r"CANDIDATES (\[[^\n]+\]) SELECTED (\d+) FORM_MAX_ABSOLUTE ([^\n]+)\n"
        r"SADDLE_COUNT (\d+)",
        physical,
    )
    assert len(rows) == 4
    certificates = []
    for kind, state, flags, selected, error, count in rows:
        admitted = [i for i, flag in enumerate(json.loads(flags.lower())) if flag]
        certificates.append(
            [
                kind,
                state,
                count,
                ", ".join(map(str, admitted)) or "none",
                selected,
                f"{float(error):.6g}",
            ]
        )
    base, head = (captures[arm, 1]["1"]["revision"] for arm in ("base", "head"))
    args.fragment.write_text(
        '<section id="bank-candidate-compact-landing">\n'
        "<h2>Compact saddle inspection</h2>\n"
        "<p>The candidate read inspects slot 0 and live saddle slots at static width "
        "<strong>K = 3</strong>, covering slot 0 plus two saddles. "
        "Padding reuses slot 0 "
        "and its duplicate results are dropped on scatter. When any lane needs more "
        "than three retained slots, one scalar conditional selects full evaluation "
        "for the whole batch. A custom batching rule reduces overflow across lanes "
        "and marks the predicate unbatched, keeping the conditional outside the lane "
        "map. Admission, selected slot, normal forms and the slot-0 fallback retain "
        "their full-evaluation semantics.</p>\n"
        + table(
            [
                "Geometry",
                "State",
                "Live saddles",
                "Admitted slots",
                "Selected slot",
                "Max form absolute gap",
            ],
            certificates,
        )
        + "<p>All four 550-cell states pass on the H200 against independent slot "
        "evaluation. Admission and selection agree exactly; forms match at rtol "
        "1e-12 and atol 1e-13 with identical finite masks. The CPU overflow fixture "
        "has four live saddle slots in one lane and a sparse companion lane whose "
        "only saddle is slot 4. Both match the full evaluation; the traced batch "
        "retains exactly one scalar conditional.</p>\n"
        + f"<p>Base <code>{base}</code> and head <code>{head}</code> ran in one "
        "H200 allocation. Each revision has its own cache, warmed by process 0. "
        "Processes 1 and 2 are fresh interpreters with recorded solve-cache hits "
        "and XLA_FLAGS unset. Each warms once and measures five blocked calls at "
        "each width. The bank_solve_profile fixture supplies sixteen diverted "
        "550-cell slices; width 1 takes its first lane.</p>\n"
        + "<p>"
        + " ".join(summaries)
        + "</p>\n"
        + table(
            [
                "Slices",
                "Arm",
                "Process",
                "Per-lane reads",
                "Whole solve [ms]",
                "Wall per slice [ms]",
                "Wall per batch read [ms]",
                "Max fresh residual",
            ],
            walls,
        )
        + "<p>Wall per batch read divides whole-solve median wall by the largest "
        "lane read count. It includes all solver work. Equal-read processes or this "
        "normalisation support timing comparisons across compilation outcomes.</p>\n"
        + table(
            [
                "Slices",
                "Arm (process 1)",
                "Candidate device [ms]",
                "Launches",
                "Batch reads",
                "Device per read [ms]",
                "Launches per read",
            ],
            devices,
        )
        + "<p>The unchanged bank_solve_trace/groups.py classifier attributes "
        "candidate support by the inspect source frame; curvature, normal form "
        "and fragment selection retain their separate groups. Device durations "
        "are summed kernel times. The control sees matrix-multiply kernels and "
        "2,097,152 host-to-device bytes in both arms; shared and unbound "
        "ownership remains in the raw ledger.</p>\n"
        + table(
            ["Slices", "Quantity / base scale", "Largest paired gap", "Budget"],
            precision,
        )
        + "<p>Every head process is compared with every base process through "
        "pair_gaps. Every lane is retained in precision-pairs.json. Flux and "
        "boundary use base flux span; axis and saddle use base minor radius. "
        "Every terminal fresh residual is at most 1e-10.</p>\n"
        + table(
            ["Suite", "Passed", "Skipped", "Failures", "Added failures", "Complete"],
            suite_rows,
        )
        + "<p>Both Sirius arms run all five named files, with only the light "
        "candidate cases selected on CPU. Every file has a fresh pytest process "
        "and separate basetemp. At base, removing the extractor anchor prints "
        "<code>MISSING_ASSIGNMENT StopIteration ''</code>, and unrolling the "
        "candidate map prints <code>UNROLLED_GUARD_PASSED</code>. The head names "
        "the missing saddle_index assignment and requires exactly two support "
        "flood programs, one in each conditional branch.</p>\n"
        + table(["Scratch mutation", "Failing test (passes at head)", "Exit"], controls)
        + "<p>The overflow mutation forces the compact branch, silently skipping "
        "overflow slots; the unrolling mutation replaces the full candidate map "
        "with a Python loop. Both are retained as diffs and failing logs.</p>\n"
        + f"<p>Raw receipts, logs, cache decisions and source-bound traces: "
        f"<code>{escape(str(args.run))}</code>.</p>\n</section>\n"
    )
    print(
        "PAIRED_COMPLETE",
        json.dumps(
            dict(
                walls=walls,
                devices=devices,
                precision=precision,
                added_failures=sorted(added),
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
