"""Attribute solve trajectory differences and compare booking tile alternatives."""

import argparse
import itertools
import json
import re
from pathlib import Path

import numpy as np

from scripts.prototypes.bank_booking_primal_buckets.report import table
from scripts.prototypes.bank_solve_cycle.report import pair_gaps

PREFIX = "bank-booking-primal-buckets"


def read_processes(run, arms):
    result = {}
    for arm, process in itertools.product(arms, (1, 2)):
        directory = run / f"{arm}-{process}"
        log = (run / f"capture-{arm}-{process}.log").read_text()
        hits = re.findall(
            r"Persistent compilation cache hit for 'jit_solve' with key '([^']+)'",
            log,
        )
        assert len(set(hits)) == 2 and "XLA_FLAGS=unset" in log
        assert "PERSISTENT COMPILATION CACHE MISS for 'jit_solve'" not in log
        assert log.rstrip().endswith("EXIT=0")
        attribution = json.loads((directory / "attribution.json").read_text())
        control = attribution["control"]
        assert control["h2d_bytes"] == 2097152 and control["kernels"] > 0
        captures = json.loads((directory / "capture.json").read_text())
        for device in attribution["solves"]:
            lanes = device["lanes"]
            capture = captures[str(lanes)]
            assert all(capture["converged"])
            assert max(capture["residual"]) <= 1e-10
            reads, updates = max(capture["reads"]), max(capture["steps"])
            result[arm, process, lanes] = dict(
                capture=capture,
                device=device,
                reads=reads,
                updates=updates,
                calls=reads + updates,
                wall=float(np.median(capture["unprofiled_ms"])),
            )
    return result


def mean_value(records, arm, function, lanes=16):
    return float(np.mean([function(records[arm, p, lanes]) for p in (1, 2)]))


def count_range(values):
    low, high = min(values), max(values)
    return str(low) if low == high else f"{low}–{high}"


def group_attribution(run):
    records = read_processes(run, ("base", "head"))
    groups = sorted(
        set().union(
            *(
                records[arm, p, 16]["device"]["groups"]
                for arm, p in itertools.product(("base", "head"), (1, 2))
            )
        )
    )
    rows, reductions = [], {}
    for group in groups:
        row = [group]
        per_read = {}
        for arm in ("base", "head"):
            values = [records[arm, p, 16] for p in (1, 2)]
            owners = [
                v["device"]["groups"].get(group, dict(device_ms=0.0, launches=0))
                for v in values
            ]
            row.extend(
                [
                    f"{np.mean([v['device_ms'] for v in owners]):.6f}",
                    count_range([v["launches"] for v in owners]),
                ]
            )
            per_read[arm] = float(
                np.mean(
                    [
                        owner["device_ms"] / v["reads"]
                        for owner, v in zip(owners, values, strict=True)
                    ]
                )
            )
        reductions[group] = per_read["base"] - per_read["head"]
        rows.append(
            row
            + [
                f"{per_read['base']:.6f}",
                f"{per_read['head']:.6f}",
                f"{reductions[group]:+.6f}",
            ]
        )
    closure = {}
    for arm in ("base", "head"):
        closure[arm] = {
            key: mean_value(
                records, arm, lambda r, key=key: r["device"][key] / r["reads"]
            )
            for key in ("device_ms", "gap_ms", "overlap_ms", "profiled_ms")
        }
        closure[arm]["wall"] = mean_value(
            records, arm, lambda r: r["wall"] / r["reads"]
        )
        closure[arm]["copies"] = mean_value(
            records,
            arm,
            lambda r: (
                sum(v["device_ms"] for v in r["device"]["copies"].values()) / r["reads"]
            ),
        )
    reduction = {
        key: closure["base"][key] - closure["head"][key] for key in closure["base"]
    }
    calls = []
    for arm, process, stage in itertools.product(
        ("base", "head"), (1, 2), ("step_residual", "step_jacobian")
    ):
        record = records[arm, process, 16]
        count = record["calls"] if stage == "step_residual" else record["updates"]
        group = record["device"]["groups"][stage]
        calls.append(
            [
                arm,
                process,
                stage,
                count,
                f"{group['device_ms'] / count:.6f}",
                f"{group['launches'] / count:.2f}",
            ]
        )
    html = f'<section id="{PREFIX}-attribution">\n'
    html += "<h2>Where the sixteen-slice wall per read went</h2>\n"
    html += (
        "<p>This table reuses the two fresh-process traces from H200 job "
        "1307054. Device times are the mean of those two processes; launch "
        "counts agree between them. Every exclusive and shared source group "
        "is retained. A zero means that neither trace attributes a kernel "
        "to that group, with the transfer/matmul positive control passing. "
        "Positive deltas below are base minus head milliseconds per read.</p>\n"
    )
    html += table(
        [
            "Source group",
            "Base ms/solve",
            "Base launches",
            "Head ms/solve",
            "Head launches",
            "Base ms/read",
            "Head ms/read",
            "Reduction ms/read",
        ],
        rows,
    )
    html += "<h3>Residual and Jacobian work per invocation</h3>\n"
    html += table(
        ["Arm", "Process", "Stage", "Calls/solve", "Device ms/call", "Launches/call"],
        calls,
    )
    html += (
        "<p>Residual calls are reads plus batch updates; Jacobian calls are "
        "batch updates. Base has seven reads and six updates; head has six "
        "reads and four updates. Thus updates per read fall from 6/7 to 4/6. "
        "The Jacobian itself is nearly unchanged per call, but it runs less "
        "often per read. This accounts for most of the apparent benefit.</p>\n"
    )
    decomposition = []
    for stage in ("step_residual", "step_jacobian", "lu_solve + update"):
        rates, costs = {}, {}
        for arm in ("base", "head"):
            values = [records[arm, process, 16] for process in (1, 2)]
            counts = [
                v["calls"] if stage == "step_residual" else v["updates"] for v in values
            ]
            rates[arm] = float(
                np.mean([n / v["reads"] for n, v in zip(counts, values, strict=True)])
            )
            costs[arm] = float(
                np.mean(
                    [
                        v["device"]["groups"][stage]["device_ms"] / n
                        for n, v in zip(counts, values, strict=True)
                    ]
                )
            )
        count_effect = costs["base"] * (rates["base"] - rates["head"])
        cost_effect = rates["head"] * (costs["base"] - costs["head"])
        assert abs(count_effect + cost_effect - reductions[stage]) < 1e-9
        decomposition.append(
            [
                stage,
                f"{rates['base']:.6f}",
                f"{rates['head']:.6f}",
                f"{count_effect:+.6f}",
                f"{cost_effect:+.6f}",
                f"{reductions[stage]:+.6f}",
            ]
        )
    html += table(
        [
            "Stage",
            "Base calls/read",
            "Head calls/read",
            "Fewer calls [ms/read]",
            "Changed cost/call [ms/read]",
            "Total reduction [ms/read]",
        ],
        decomposition,
    )
    html += (
        "<p>The count contribution holds base cost per call fixed; the cost "
        "contribution uses the head call rate. Their sum equals the measured "
        "group reduction. The residual's reduced invocation rate hides its "
        "higher per-call cost; the Jacobian and linear solve benefits are "
        "almost entirely fewer invocations.</p>\n"
    )
    html += "<h3>Device contribution and observed host gaps</h3>\n"
    html += table(
        ["Quantity [ms/read]", "Base", "Head", "Base − head"],
        [
            [
                key,
                f"{closure['base'][key]:.6f}",
                f"{closure['head'][key]:.6f}",
                f"{reduction[key]:+.6f}",
            ]
            for key in closure["base"]
        ],
    )
    html += (
        f"<p>The unprofiled wall loses {reduction['wall']:.6f} ms/read. "
        f"The measured device groups lose {reduction['device_ms']:.6f} ms/read: "
        f"Jacobian {reductions['step_jacobian']:.6f}, linear solve/update "
        f"{reductions['lu_solve + update']:.6f}, residual "
        f"{reductions['step_residual']:.6f}, other solve work "
        f"{reductions['solve_batch (other)']:.6f}, and equilibration "
        f"{reductions['equilibrate + lu_factor']:.6f} are the largest "
        "contributions. The table retains all remaining contributions, "
        "including copies and fills in the closure table.</p>\n"
    )
    html += (
        f"<p>The traced host gaps shrink {reduction['gap_ms']:.6f} ms/read "
        f"and overlap increases {-reduction['overlap_ms']:.6f} ms/read. "
        "Device minus overlap plus gaps closes the profiled wall, whose "
        f"reduction is {reduction['profiled_ms']:.6f} ms/read. The "
        f"{reduction['wall'] - reduction['device_ms']:.6f} ms/read difference "
        "between the unprofiled wall reduction and profiled device reduction "
        "is a cross-measurement remainder, not a measured unprofiled host "
        "gap. Profiling changes the host timeline, so those two walls are "
        "not equated. The benefit is predominantly fewer updates per read, "
        "not faster booking or faster read geometry.</p>\n"
    )
    propagation = {}
    for arm in ("base", "head"):
        propagation[arm] = []
        for process in (1, 2):
            kernels = json.loads((run / f"{arm}-{process}/kernels-16.json").read_text())
            launches = sum(
                row["launches"]
                for row in kernels
                if row["group"] == "step_residual"
                and any(
                    "quadratic_cell_fragments" in binding
                    for binding in row["source_bindings"]
                )
            )
            propagation[arm].append(launches / records[arm, process, 16]["calls"])
    html += (
        "<p>The residual kernel ledgers also explain the launch regression: "
        "conic strip propagation at <code>quadratic_cell_fragments</code> "
        f"runs {count_range(propagation['base'])} times per call at base and "
        f"{count_range(propagation['head'])} at the first head. The primal "
        "tile changed from 32 cells to the tangent tile's 16 cells. The "
        "candidate restores the primal tile to 32 cells; the two shared "
        "helpers, separate normal population, and four-ring groups stay.</p>\n"
        "</section>\n"
    )
    return html, dict(
        group_reductions=reductions,
        closure=closure,
        reduction=reduction,
        group_rows=rows,
        call_rows=calls,
        invocation_decomposition=decomposition,
        propagation_per_call=propagation,
    )


def comparison(run):
    arms = ("base", "prior", "head")
    records = read_processes(run, arms)
    rows, calls, pairs = [], [], []
    for lanes, arm, process in itertools.product((1, 16), arms, (1, 2)):
        r = records[arm, process, lanes]
        stage = r["device"]["groups"]["step_residual"]
        jac = r["device"]["groups"]["step_jacobian"]
        rows.append(
            [
                arm,
                process,
                lanes,
                (
                    f"{r['reads']} in all {lanes} lanes"
                    if len(set(r["capture"]["reads"])) == 1
                    else str(r["capture"]["reads"])
                ),
                r["updates"],
                f"{r['wall']:.6f}",
                f"{r['wall'] / r['reads']:.6f}",
            ]
        )
        calls.append(
            [
                arm,
                process,
                lanes,
                r["calls"],
                f"{stage['device_ms']:.6f}",
                stage["launches"],
                f"{stage['device_ms'] / r['calls']:.6f}",
                f"{stage['launches'] / r['calls']:.2f}",
                f"{jac['device_ms'] / r['updates']:.6f}",
            ]
        )
    for lanes, arm, first, second in itertools.product(
        (1, 16), ("prior", "head"), (1, 2), (1, 2)
    ):
        gaps, limits = pair_gaps(
            records["base", first, lanes]["capture"],
            records[arm, second, lanes]["capture"],
        )
        assert all(max(v) <= limits[k] for k, v in gaps.items()), gaps
        pairs.append(
            dict(
                lanes=lanes,
                arm=arm,
                base_process=first,
                process=second,
                gaps=gaps,
                limits=limits,
            )
        )
    verdict = {}
    for lanes in (1, 16):
        metrics = {}
        for arm in arms:
            values = [records[arm, process, lanes] for process in (1, 2)]
            metrics[arm] = dict(
                device_per_call=[
                    r["device"]["groups"]["step_residual"]["device_ms"] / r["calls"]
                    for r in values
                ],
                launches_per_call=[
                    r["device"]["groups"]["step_residual"]["launches"] / r["calls"]
                    for r in values
                ],
                wall_per_read=[r["wall"] / r["reads"] for r in values],
            )
        verdict[lanes] = dict(
            metrics=metrics,
            faster=max(metrics["head"]["device_per_call"])
            < min(metrics["base"]["device_per_call"]),
            launches=max(metrics["head"]["launches_per_call"])
            <= min(metrics["base"]["launches_per_call"]),
        )
    retain = all(v["faster"] and v["launches"] for v in verdict.values())
    html = f'<section id="{PREFIX}-three-arm">\n'
    html += "<h2>Three-arm comparison in one further H200 allocation</h2>\n"
    html += (
        "<p>H200 job 1307085 requests one GPU, two CPUs and 96 GB. "
        "Base is the sequential formulation; prior is the first shared-helper "
        "head with 16-cell primal tiles; head is the 32-cell primal candidate. "
        "Every arm has its own compilation cache, one warm-up process and two "
        "fresh measured processes with XLA_FLAGS unset. All precision budgets "
        "pass against base for every pairing of measured processes. The "
        "positive transfer/matmul control passes in every measured process. The "
        "candidate is retained only if both widths beat base residual device "
        "time per call and do not exceed base launches per call.</p>\n"
    )
    html += table(
        [
            "Arm",
            "Process",
            "Slices",
            "Per-lane reads",
            "Batch updates",
            "Solve ms",
            "Wall/read ms",
        ],
        rows,
    )
    html += table(
        [
            "Arm",
            "Process",
            "Slices",
            "Residual calls",
            "Residual ms/solve",
            "Residual launches",
            "Residual ms/call",
            "Residual launches/call",
            "Jacobian ms/call",
        ],
        calls,
    )
    html += (
        "<p>Measured revisions: "
        + "; ".join(
            f"{arm} <code>{records[arm, 1, 16]['capture']['revision']}</code>"
            for arm in arms
        )
        + ".</p>\n"
    )
    precision = {
        key: max(max(pair["gaps"][key]) for pair in pairs) for key in pairs[0]["gaps"]
    }
    residual = max(max(record["capture"]["residual"]) for record in records.values())
    html += "<h3>Precision across all measured process pairings</h3>\n"
    html += table(
        ["Quantity", "Maximum normalized gap", "Limit"],
        [
            [key, f"{value:.6e}", f"{pairs[0]['limits'][key]:.6e}"]
            for key, value in precision.items()
        ]
        + [["Fresh residual", f"{residual:.6e}", "1.000000e-10"]],
    )
    html += (
        "<p>The candidate also passes all six focused Sirius files in fresh "
        "processes: 25 passed, zero failed, three optional skips. Its five "
        "moment-reference comparisons retain a maximum relative difference "
        "of 1.708597e-16 against the 1e-14 limit, with the first head's pins "
        "unchanged. Routing one normal-form cell through the conic bucket "
        "in a scratch archive fails the same three diverted tests and ends "
        "EXIT=1; the two limited cases pass. Receipts are in the run's "
        "<code>restructure</code> directory.</p>\n"
    )
    conclusions = []
    for lanes, v in verdict.items():
        for arm, m in v["metrics"].items():
            d, w = m["device_per_call"], m["wall_per_read"]
            conclusions.append(
                f"{arm}, {lanes} slices: booking {min(d):.3f}–{max(d):.3f} "
                f"ms/call and whole-solve wall {min(w):.3f}–{max(w):.3f} ms/read"
            )
    decision = (
        "The wider primal tile meets both cost and launch conditions and is retained."
        if retain
        else (
            "The candidate does not beat base under both conditions; "
            "the first head is retained."
        )
    )
    html += "<p>" + "; ".join(conclusions) + ". " + decision + "</p>\n</section>\n"
    return html, dict(
        rows=rows,
        calls=calls,
        pairs=pairs,
        verdict=verdict,
        precision_maxima=precision,
        max_fresh_residual=residual,
        retain_candidate=retain,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--comparison", action="store_true")
    args = parser.parse_args()
    html, receipt = group_attribution(args.run)
    (args.run / "group-attribution.json").write_text(json.dumps(receipt, indent=2))
    if args.comparison:
        added, receipt = comparison(args.run / "restructure")
        html += added
        (args.run / "restructure/comparison.json").write_text(
            json.dumps(receipt, indent=2)
        )
        print(
            "COMPARISON",
            json.dumps(
                {
                    "retain_candidate": receipt["retain_candidate"],
                    "verdict": receipt["verdict"],
                }
            ),
            flush=True,
        )
    prior = args.output.read_text().split(f'<section id="{PREFIX}-attribution">')[0]
    args.output.write_text(prior + html)
    print("ATTRIBUTED", json.dumps(receipt.get("reduction", {})), flush=True)


if __name__ == "__main__":
    main()
