"""Render stationary latency and audit receipts without rerunning measurements."""

import argparse
from html import escape
import json
from pathlib import Path
import subprocess

import matplotlib.pyplot as plt
import numpy as np


def table(anchor, headings, rows):
    return (
        '<table id="fixed-point-production-'
        + anchor
        + '"><thead><tr>'
        + "".join("<th>" + escape(str(value)) + "</th>" for value in headings)
        + "</tr></thead><tbody>"
        + "".join(
            "<tr>"
            + "".join("<td>" + escape(str(value)) + "</td>" for value in row)
            + "</tr>"
            for row in rows
        )
        + "</tbody></table>"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--base", required=True)
    args = parser.parse_args()
    rows = [
        json.loads((args.run / "rows" / f"{kind}-{cells}.json").read_text())
        for kind in ("diverted", "limited")
        for cells in (2000, 5000, 10000)
    ]
    assert all(
        row["complete"] and row["p99_seconds"] < 0.001 and row["failures"] == 0
        for row in rows
    )
    tail = json.loads((args.run / "tail-rows/diverted-2000.json").read_text())
    assert tail["complete"]
    head = json.loads((args.run / "contracts-complete/head.json").read_text())
    negative = json.loads((args.run / "contracts-complete/negative.json").read_text())
    assert head["completed"] and head["exit_status"] == 0
    assert negative["completed"] and negative["exit_status"] != 0
    commits = subprocess.check_output(
        ["git", "rev-list", "--reverse", args.base + "..HEAD"], text=True
    ).split()
    plt.style.use("data-ink")
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=100, sharey=True)
    for axis, kind in zip(axes, ("diverted", "limited")):
        selected = [row for row in rows if row["kind"] == kind]
        x = [row["cells"] for row in selected]
        for key, label, color, style in (
            ("prototype_p99_seconds", "prototype", "0.45", "--"),
            ("p99_seconds", "production", "0.15", "-"),
        ):
            y = [row[key] * 1000 for row in selected]
            axis.plot(
                x,
                y,
                color=color,
                linestyle=style,
                linewidth=3 if label == "production" else 2.6,
                marker="o",
            )
            axis.annotate(
                label,
                (x[-1], y[-1]),
                xytext=(-4, -24 if label == "production" else 10),
                textcoords="offset points",
                ha="right",
                color=color,
                fontsize=20,
            )
        axis.axhline(1.0, color="0.5", linewidth=1.2, linestyle=":")
        axis.text(2000, 1.025, "1 ms limit", color="0.4", fontsize=20)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(False)
        axis.set_xticks((2000, 5000, 10000))
        axis.set_xlabel("Cells")
        axis.set_ylim(0, 1.2)
    axes[0].set_ylabel("Fixed-point p99 [ms]")
    figure.tight_layout()
    image = Path(
        "docs/figures/converged-forward-solve/fixed-point-production/latency.svg"
    )
    image.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(image)
    plt.close(figure)
    latency = [
        [
            row["kind"],
            row["cells"],
            f"{row['prototype_median_seconds'] * 1000:.6f}",
            f"{row['median_seconds'] * 1000:.6f}",
            f"{row['prototype_p99_seconds'] * 1000:.6f}",
            f"{row['p99_seconds'] * 1000:.6f}",
            row["calls"],
            row["failures"],
        ]
        for row in rows
    ]
    maintenance = []
    for row in rows:
        slide = [state["maintenance"]["seconds"] for state in row["states"][1:]]
        maintenance.append(
            [
                row["kind"],
                row["cells"],
                f"{row['states'][0]['maintenance']['seconds']:.6f}",
                f"{np.median(slide):.6f}",
                f"{max(slide):.6f}",
                row["refresh_count"],
                f"{min(row['travel_pitches']):.6f}",
                row["states"][0]["maintenance"]["cache_bytes"],
            ]
        )
    tail_rows = [
        [
            row["state_kind"],
            row["calls"],
            f"{row['median_seconds'] * 1000:.6f}",
            f"{row['production_median_seconds'] * 1000:.6f}",
            f"{row['p99_seconds'] * 1000:.6f}",
            f"{row['production_p99_seconds'] * 1000:.6f}",
        ]
        for row in tail["rows"]
        if row["arm"] == "cached-adaptive"
    ]
    body = (
        '<h1 id="fixed-point-production-landing">Production stationary-point solve</h1>'
    )
    body += table(
        "latency",
        [
            "Case",
            "Cells",
            "Prototype median ms",
            "Production median ms",
            "Prototype p99 ms",
            "Production p99 ms",
            "Calls",
            "Refusals / wrong roots",
        ],
        latency,
    )
    body += (
        '<figure id="fixed-point-production-figure"><img src="/nova/figur'
        'es/converged-forward-solve/fixed-point-production/latency.svg" a'
        'lt="Production and prototype p99 latency across three mesh sizes'
        '"><figcaption>Left: diverted. Right: limited. Each point is 420 '
        "synchronized calls over 21 moving states with nonzero booked pla"
        "sma moments; each sweep travels about eight pitches and slides i"
        "ts bounded row window. Row maintenance and compilation are outsi"
        "de the timed calls.</figcaption></figure>"
    )
    body += table(
        "maintenance",
        [
            "Case",
            "Cells",
            "Initial build s",
            "Slide median s",
            "Slide max s",
            "Slides adding rows",
            "Minimum travel / pitch",
            "Row bytes",
        ],
        maintenance,
    )
    body += table(
        "tail",
        [
            "Original workload",
            "Calls / arm",
            "Prototype median ms",
            "Production median ms",
            "Prototype p99 ms",
            "Production p99 ms",
        ],
        tail_rows,
    )
    body += table(
        "contracts",
        ["Contract", "Evidence"],
        [
            [
                "Public surface",
                (
                    "solve(cache, moments, seeds, previous, pitch, axis_sign); mainta"
                    "in_rows(geometry, coupling, exterior, previous, motion_margin). "
                    "No production wiring in this change."
                ),
            ],
            [
                "Root policy",
                (
                    "Axis is definite with requested sign; X-points are indefinite. W"
                    "rong roots retry previous through topology.stationary_read and r"
                    "emain refused if still wrong."
                ),
            ],
            [
                "Row-window miss",
                (
                    "Typed NEEDS_ROWS; solve performs no row builds. Maintenance adds"
                    " intersecting cells around previous points plus the motion margi"
                    "n and retains overlap within a fixed 512-node capacity."
                ),
            ],
            [
                "Coarse reproduction",
                (
                    "H200 job 1284243 reproduced valid saddle axes at states 0,16,17,"
                    "18. State 16 eigenvalues: -3.28191563527506, +4.187683936036714."
                ),
            ],
            [
                "Fast CPU head",
                (
                    "15 passed with root conftest, including quartic jets and limited"
                    "/diverted analytic 50-cell states."
                ),
            ],
            [
                "Base delta",
                (
                    "Base has no production fixed_points module; identical added cont"
                    "racts stop at collection with one missing-module error. Head has"
                    " no failing ids."
                ),
            ],
            [
                "Negative control",
                (
                    "Disabling _root_type: 7 failed, including test_coarse_replay_nev"
                    "er_accepts_saddle_as_axis[0], [16], [17], [18]; EXIT=1. Coarse r"
                    "eplay deliberately retains the wrong previous point to prove a r"
                    "etry cannot authorize it."
                ),
            ],
            [
                "Fusion",
                (
                    "One polynomial derivative contraction replaces three differentia"
                    "ted Bell evaluations inside batched adaptive Newton. Tables comp"
                    "are the complete guarded production route with the unguarded ada"
                    "ptive prototype."
                ),
            ],
            [
                "Earlier 1.638275 ms tail",
                (
                    "Original Picard receipt has 100 calls, all three Newton updates."
                    " Adjacent calls 58 and 59 took 6.135 and 1.593 ms; one sam"
                    "pled device trace totals 91.424 microseconds. This rules out var"
                    "iable iteration count; external scheduling is an inference, not "
                    "an attributed cause. The exact-workload replay above measures 10"
                    "00 interleaved calls per arm."
                ),
            ],
            [
                "Scale workload",
                (
                    "Analytic exterior shifted through eight pitches plus nonzero pla"
                    "sma moments scaled by 0.002 + 0.001 sin(state). The separate ful"
                    "l-current replay uses the original analytic and Picard moment wo"
                    "rkloads at 2000 cells."
                ),
            ],
            [
                "Scope limit",
                (
                    "Latency qualifies prepared cache calls on H200. Maintenance rema"
                    "ins a separate host operation; it is not included in the sub-mil"
                    "lisecond claim. No Grad-Shafranov terminal solve is produced."
                ),
            ],
            [
                "Source",
                (
                    "nova/equilibrium/fixed_points.py; tests/test_fixed_points.py; sc"
                    "ripts/prototypes/fixed_points"
                ),
            ],
            ["Durable receipts", str(args.run.resolve())],
            ["CPU head log", head["log_path"]],
            ["Guard-disabled log", negative["log_path"]],
        ],
    )
    html = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta'
        ' name="viewport" content="width=device-width, initial-scale=1">'
    )
    metadata = {
        "docs-project": "nova",
        "reckon-type": "evidence",
        "plan-slug": "fixed-point-production",
        "plan-title": "Production stationary-point solve",
        "plan-summary": (
            "Bounded Bell rows with adaptive guarded Newton, maintenance and "
            "latency qualification"
        ),
        "plan-evidence-for": "converged-forward-solve",
        "plan-verifies": "converged-forward-solve#s2",
        "plan-recorded-at": "2026-10-13",
        "plan-verdict": "pass",
        "plan-commits": ",".join(commits),
        "plan-environment": "H200 x64 for latency; CPU x64 for contracts",
    }
    html += "".join(
        '<meta name="' + key + '" content="' + escape(value, quote=True) + '">'
        for key, value in metadata.items()
    )
    html += (
        (
            "<title>Production stationary-point solve | nova</title><link rel"
            '="stylesheet" href="/_shared/foundation.css"><link rel="styleshe'
            'et" href="/_shared/dashboard.css"></head><body><main class="plan'
            '-doc">'
        )
        + body
        + "</main></body></html>\n"
    )
    destination = Path(
        "docs/evidence/fragments/converged-forward-solve/fixed-point-production.html"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(html)
    print(
        json.dumps(
            dict(
                rows=len(rows),
                calls=sum(row["calls"] for row in rows),
                maximum_p99_ms=max(row["p99_seconds"] for row in rows) * 1000,
                fragment=str(destination),
                figure=str(image),
            )
        )
    )


if __name__ == "__main__":
    main()
