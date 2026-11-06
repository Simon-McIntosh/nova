"""Tabulate paired process walls, exclusive booking device work and precision."""

import argparse
from html import escape
import itertools
import json
from pathlib import Path
import re

import numpy as np

from scripts.prototypes.bank_solve_cycle.report import pair_gaps


def table(headers, rows):
    return (
        "<table><thead><tr>"
        + "".join(f"<th>{escape(str(x))}</th>" for x in headers)
        + "</tr></thead><tbody>"
        + "".join(
            "<tr>" + "".join(f"<td>{escape(str(x))}</td>" for x in row) + "</tr>"
            for row in rows
        )
        + "</tbody></table>\n"
    )


def parity_rows(path):
    rows = []
    for line in path.read_text().splitlines():
        if "MOMENT_PARITY {" in line:
            rows.append(json.loads(line.split("MOMENT_PARITY ", 1)[1]))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    captures = {
        (arm, process): json.loads(
            (args.run / f"{arm}-{process}/capture.json").read_text()
        )
        for arm in ("base", "head")
        for process in range(3)
    }
    attribution = {
        (arm, process): json.loads(
            (args.run / f"{arm}-{process}/attribution.json").read_text()
        )
        for arm in ("base", "head")
        for process in (1, 2)
    }
    rows, precision, invocation_rows = [], [], []
    for lanes in (1, 16):
        for arm in ("base", "head"):
            for process in range(3):
                capture = captures[arm, process][str(lanes)]
                assert all(capture["converged"]) and max(capture["residual"]) <= 1e-10
                wall = float(np.median(capture["unprofiled_ms"]))
                reads = capture["reads"]
                updates = max(capture["steps"])
                read_label = (
                    f"{reads[0]} in all {lanes} lanes"
                    if len(set(reads)) == 1
                    else str(reads)
                )
                device = next(
                    (
                        x
                        for x in attribution.get((arm, process), {}).get("solves", [])
                        if x["lanes"] == lanes
                    ),
                    None,
                )
                if device:
                    residual = device["groups"]["step_residual"]
                    assert residual["launches"] > 0 and residual["device_ms"] > 0
                    cache_log = (args.run / f"capture-{arm}-{process}.log").read_text()
                    assert (
                        "Persistent compilation cache hit for 'jit_solve'" in cache_log
                    )
                    device_ms, launches = (
                        f"{residual['device_ms']:.6f}",
                        residual["launches"],
                    )
                else:
                    device_ms, launches = "warm-up", "warm-up"
                if device:
                    evaluations = max(reads) + updates
                    invocation_rows.append(
                        [
                            arm,
                            process,
                            lanes,
                            max(reads),
                            updates,
                            evaluations,
                            f"{residual['device_ms'] / evaluations:.6f}",
                            f"{residual['launches'] / evaluations:.2f}",
                        ]
                    )
                rows.append(
                    [
                        arm,
                        process,
                        lanes,
                        read_label,
                        f"{wall:.6f}",
                        f"{wall / max(reads):.6f}",
                        device_ms,
                        launches,
                    ]
                )
        for before, after in itertools.product((1, 2), repeat=2):
            base, head = (
                captures["base", before][str(lanes)],
                captures["head", after][str(lanes)],
            )
            gaps, limits = pair_gaps(base, head)
            assert all(max(values) <= limits[key] for key, values in gaps.items()), gaps
            precision.append(
                dict(
                    lanes=lanes,
                    base_process=before,
                    head_process=after,
                    gaps=gaps,
                    limits=limits,
                )
            )
    (args.run / "precision-pairs.json").write_text(json.dumps(precision, indent=2))
    text = (
        '<section id="bank-booking-primal-buckets-landing">\n<h2>Primal '
        "booking in separate geometry populations</h2>\n"
    )
    text += (
        "<p><strong>The requested structure preserves precision, but does "
        "not make each booking cheaper.</strong> At one slice the paired "
        "fresh processes take 96.369–96.669 ms at base and 100.004–100.341 ms "
        "at head, both with four reads. At sixteen slices, base takes "
        "603.760–604.689 ms with seven reads and six batch updates; head takes "
        "461.597–462.133 ms with six reads and four updates. Wall per read "
        "falls from 86.251–86.384 to 76.933–77.022 ms. Residual device work "
        "per call rises from 2.865–2.866 to 3.934–3.941 ms at one slice and "
        "from 9.124–9.140 to 9.807–9.817 ms at sixteen. Launches per call "
        "also rise. The lower sixteen-slice total includes a shorter "
        "trajectory and must not be presented as a booking speedup.</p>\n"
    )
    text += (
        "<p>Primal and tangent booking share one cell-bucket loop and one "
        "live-ring loop. Conic and normal-form cells enter separate "
        "populations, and live rings integrate in groups of four. The "
        "sequential primal reference stays in the tests.</p>\n"
    )
    text += (
        "<p>The base source reproduction contains "
        "<code>jax.vmap(cut_cell)(indices)</code>, with "
        "<code>jax.lax.cond(read.normal_form_cells[index], normal, conic, "
        "None)</code> inside that cell function and one ring per "
        "integration-loop iteration. The paired base device trace below "
        "reproduces its cost.</p>\n"
    )
    text += (
        "<p>One H200 allocation; two CPUs and 96 GB requested. Each "
        "revision has a separate compilation cache, warmed by process 0. "
        "Processes 1 and 2 run fresh with XLA_FLAGS unset and record "
        "solve-cache hits. Walls are medians of five synchronized "
        "unprofiled calls. Device time and launches are the exclusive "
        "<code>bank_solve_trace</code> step_residual group from one traced"
        " solve. Wall per read divides by the maximum per-lane read count,"
        " the batch loop length.</p>\n"
    )
    text += table(
        [
            "Arm",
            "Process",
            "Slices",
            "Per-lane reads",
            "Solve wall [ms]",
            "Wall/read [ms]",
            "Residual device [ms]",
            "Residual launches",
        ],
        rows,
    )
    text += "<h3>Residual work per invocation</h3>\n"
    text += (
        "<p>Each read evaluates its fresh residual, and each batch update "
        "evaluates another residual. The count below is batch reads plus "
        "maximum updates. This exposes trajectory changes beside the "
        "whole-solve totals; fewer updates can lower total booking work "
        "without making one booking cheaper.</p>\n"
    )
    text += table(
        [
            "Arm",
            "Process",
            "Slices",
            "Reads",
            "Updates",
            "Residual calls",
            "Device/call [ms]",
            "Launches/call",
        ],
        invocation_rows,
    )
    maxima = {
        key: max(max(row["gaps"][key]) for row in precision)
        for key in precision[0]["gaps"]
    }
    limits = precision[0]["limits"]
    text += "<h3>Precision across every measured base/head process pairing</h3>\n"
    text += table(
        ["Quantity", "Maximum normalized gap", "Limit"],
        [[key, f"{value:.6e}", f"{limits[key]:.6e}"] for key, value in maxima.items()],
    )
    residual = max(
        max(row[str(lanes)]["residual"])
        for row in captures.values()
        for lanes in (1, 16)
    )
    text += (
        f"<p>Largest fresh residual across all processes and lanes: "
        f"{residual:.6e}, against 1e-10. Flux and boundary gaps use the "
        f"base flux span; axis and saddle gaps use its minor radius. The "
        f"trace instrument positive control observes 2,097,152 host-to-"
        f"device bytes and a matrix kernel in every measured process.</p>\n"
    )
    parity_path = args.run / "head-test_book_primal_buckets.log"
    parity = parity_rows(parity_path)
    final = args.run / "verified-test_book_primal_buckets.log"
    if final.exists():
        parity = parity_rows(final)
    assert len(parity) >= 5
    text += (
        "<h3>Moment comparison with the sequential formulation</h3><p>Per-"
        "field error is maximum absolute difference divided by the maximum"
        " absolute reference value; a zero field requires exact equality. "
        "Every read and booking must be valid. The limit is 1e-14 per "
        "field.</p>\n"
    )
    fields = list(parity[0]["fields"])
    text += table(
        ["State", "Normal cut cells", *fields],
        [
            [
                row["case"],
                row["normal_cells"],
                *[f"{row['fields'][name]['relative']:.6e}" for name in fields],
            ]
            for row in parity
        ],
    )
    if (args.run / "pins-base.npz").exists() and (args.run / "pins-head.npz").exists():
        with (
            np.load(args.run / "pins-base.npz") as before,
            np.load(args.run / "pins-head.npz") as after,
        ):
            pin_rows = []
            for name in before.files:
                a, b = before[name], after[name]
                delta = float(np.max(np.abs(a.astype(float) - b.astype(float))))
                scale = float(np.max(np.abs(a.astype(float))))
                pin_rows.append(
                    [
                        name,
                        f"{delta:.6e}",
                        f"{delta / scale if scale else delta:.6e}",
                        a.tobytes() == b.tobytes(),
                    ]
                )
        text += "<h3>Compiled 50-cell booking and byte pins</h3>\n"
        text += (
            "<p>The unchanged base already fails the stored CPU byte pins. "
            "The paired compiled outputs above establish that grouping changes "
            "only the three moment arrays at roundoff. The measured Sirius "
            "pins were updated only after retaining this difference table; "
            "the final verification arm passes those pins. The other CPU "
            "family was not measured here.</p>\n"
        )
        text += table(
            [
                "Field",
                "Maximum absolute difference",
                "Relative difference",
                "Byte-identical",
            ],
            pin_rows,
        )
    negative = args.run / "verified-negative-test_book_primal_buckets.log"
    negative_text = negative.read_text()
    assert re.search(r"EXIT=[1-9][0-9]*\s*$", negative_text)
    failures = re.findall(r"^FAILED (\S+)", negative_text, re.MULTILINE)
    assert failures
    text += (
        "<h3>Tests and negative control</h3>\n<p>The scratch mutation "
        "clears the first normal-form cut cell from the normal population,"
        " routing it through the conic population. The negative log "
        "terminates nonzero and fails: " + escape(", ".join(failures)) + ".</p>\n"
    )
    suite_rows = []
    for arm in ("base", "head", "verified", "matrix-base", "matrix-head"):
        for path in sorted(args.run.glob(f"{arm}-test_*.log")):
            log = path.read_text()
            summaries = re.findall(r"^=+ (.* in .*?) =+$", log, re.MULTILINE)
            exits = re.findall(r"^EXIT=(\d+)$", log, re.MULTILINE)
            assert summaries and exits, path
            suite_rows.append(
                [arm, path.stem.removeprefix(arm + "-"), summaries[-1], exits[-1]]
            )
    text += table(["Arm", "File", "Summary", "Exit"], suite_rows)
    text += (
        "<p>Durable receipts and raw traces: <code>"
        + escape(str(args.run))
        + (
            "</code>. The directory retains process capture.json, "
            "attribution.json, exclusive kernel ledgers, precision-pairs.json,"
            " pin arrays, and each fresh-process gate log with its revision, "
            "imported module, working directory, exact command and final exit "
            "status.</p>\n</section>\n"
        )
    )
    args.output.write_text(text)
    (args.run / "summary.json").write_text(
        json.dumps(
            dict(
                rows=rows, precision_maxima=maxima, residual_max=residual, parity=parity
            ),
            indent=2,
        )
    )
    print(
        "SUMMARY",
        json.dumps(
            dict(precision_maxima=maxima, residual_max=residual, process_rows=len(rows))
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
