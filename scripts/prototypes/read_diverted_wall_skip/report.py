"""Validate paired CPU receipts and render the read-wall comparison."""

# ruff: noqa: E501

import argparse
import hashlib
from html import escape
import json
from pathlib import Path
import re
import statistics
import subprocess


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--measurements", type=Path)
    args = parser.parse_args()
    measurement_run = args.measurements or args.run
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from scripts.prototypes.bank_solve_profile.plot import save_checked

    pairs = {
        arm: [
            json.loads((measurement_run / f"{arm}-{i}.json").read_text())
            for i in (1, 2)
        ]
        for arm in ("base", "head")
    }
    for receipts in pairs.values():
        assert all(receipt["complete"] for receipt in receipts)
        assert receipts[0]["revision"] == receipts[1]["revision"]
        assert receipts[0]["instrument_sha256"] == receipts[1]["instrument_sha256"]
        assert receipts[0]["helper_sha256"] == receipts[1]["helper_sha256"]
        for receipt in receipts:
            for path, key in (
                (
                    "scripts/prototypes/read_diverted_wall_skip/measure.py",
                    "instrument_sha256",
                ),
                ("tests/test_read_wall_skip.py", "helper_sha256"),
            ):
                committed = subprocess.check_output(
                    [
                        "git",
                        "-C",
                        str(args.root),
                        "show",
                        f"{receipt['instrument_revision']}:{path}",
                    ]
                )
                assert hashlib.sha256(committed).hexdigest() == receipt[key]
            if receipt["revision"] == pairs["head"][0]["revision"]:
                for path in (
                    "nova/equilibrium/topology.py",
                    "nova/equilibrium/support_read.py",
                ):
                    committed = subprocess.check_output(
                        [
                            "git",
                            "-C",
                            str(args.root),
                            "show",
                            f"{receipt['revision']}:{path}",
                        ]
                    )
                    assert committed == (args.root / path).read_bytes()
    summary = []
    max_error = 0.0
    for index, row in enumerate(pairs["base"][0]["rows"]):
        key = f"{row['population']}-{row['width']}-{row['kind']}"
        values = {}
        for arm in ("base", "head"):
            copies = [receipt["rows"][index] for receipt in pairs[arm]]
            for field in (
                "hlo_operations",
                "stablehlo_operations",
                "while_sites",
                "fusion_sites",
            ):
                assert copies[0][field] == copies[1][field], (key, field)
            medians = [copy["median_ms"] for copy in copies]
            spread = (max(medians) - min(medians)) / statistics.mean(medians)
            assert spread <= 0.10, (key, arm, "process spread", spread)
            values[arm] = dict(
                mean_ms=statistics.mean(medians),
                medians_ms=medians,
                spread=spread,
                counts={
                    field: copies[0][field]
                    for field in (
                        "hlo_operations",
                        "stablehlo_operations",
                        "while_sites",
                        "fusion_sites",
                    )
                },
            )
        for process in (1, 2):
            with (
                np.load(measurement_run / f"base-{process}-{key}.npz") as baseline,
                np.load(measurement_run / f"head-{process}-{key}.npz") as head,
            ):
                assert baseline.files == head.files
                for field in baseline.files:
                    left, right = baseline[field], head[field]
                    if left.dtype.kind in "bui":
                        np.testing.assert_array_equal(left, right)
                    else:
                        np.testing.assert_allclose(left, right, rtol=1e-10, atol=1e-12)
                        finite = np.isfinite(left) & np.isfinite(right)
                        if np.any(finite):
                            max_error = max(
                                max_error,
                                float(np.max(np.abs(left[finite] - right[finite]))),
                            )
        summary.append(dict(**row, **values))

    plt.style.use("data-ink")
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.2), dpi=100, sharey=True)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.17, top=0.82, wspace=0.25)
    handles = []
    for ax, population in zip(axes, ("diverted", "mixed"), strict=True):
        for arm, color in (("base", "#929292"), ("head", "#3366cc")):
            for kind, dash in (("carried", "-"), ("full", "--")):
                rows = [
                    row
                    for row in summary
                    if row["population"] == population and row["kind"] == kind
                ]
                (line,) = ax.plot(
                    [row["width"] for row in rows],
                    [row[arm]["mean_ms"] / 1000 for row in rows],
                    dash,
                    marker="o",
                    color=color,
                    lw=3 if arm == "head" else 2.6,
                    zorder=3 if arm == "head" else 2,
                    label=f"{arm} {kind}",
                )
                if population == "diverted":
                    handles.append(line)
        ax.set_xlabel("batch width")
        ax.set_ylabel("read per batch [s]")
        ax.set_xticks([1, 16])
        ax.set_xlim(0, 17)
        ax.set_ylim(bottom=0)
        ax.text(
            0.5,
            1.06,
            "all diverted" if population == "diverted" else "mixed / limited",
            transform=ax.transAxes,
            ha="center",
            fontsize=16,
        )
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(False)
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=4,
        frameon=False,
    )
    figure = (
        args.root
        / "docs/figures/sub-millisecond-forward-path/read-diverted-wall-skip/read-wall.svg"
    )
    save_checked(fig, figure)
    fig.savefig(args.run / "read-wall.png", dpi=100)
    cells = pairs["head"][0]["cells"]
    reproduction = (args.run / "reproduce.log").read_text()
    counter = re.search(r"WALL_EXECUTIONS[^\n]+", reproduction)
    assert counter is not None and "EXIT=1" in reproduction
    controls = (args.run / "wall-controls.log").read_text()
    assert controls.rstrip().endswith("EXIT=0")
    execution_rows = re.findall(
        r"WALL_EXECUTIONS \((\d), (\d)\) certificate (True|False) (\d+)", controls
    )
    assert len(execution_rows) == 4
    for left, right, certificate, count in execution_rows:
        expected = certificate == "True" or left == "1" or right == "1"
        assert (int(count) > 0) == expected
    suite_rows = []
    for arm in ("base", "head"):
        for filename in (
            "test_tracked_read.py",
            "test_support_read.py",
            "test_read_wall_skip.py",
        ):
            if arm == "base" and filename == "test_read_wall_skip.py":
                continue
            log = (args.run / f"{arm}-{filename}.log").read_text()
            assert log.rstrip().endswith("EXIT=0"), (arm, filename)
            passed = re.search(r"(\d+) passed", log)
            assert passed is not None
            suite_rows.append(
                f"<tr><td>{arm}</td><td><code>{filename}</code></td><td>{passed[1]} passed</td><td>0</td></tr>"
            )
    mutant = (args.run / "negative-control.log").read_text()
    assert mutant.rstrip().endswith("EXIT=1")
    failures = re.findall(r"^FAILED (\S+)", mutant, re.M)
    assert any("classes2" in failure for failure in failures), failures
    verification = f"""<h3 id="read-diverted-wall-skip-tests">Regression and refusal evidence</h3>
<p>One seven-CPU Sirius job runs each suite file in a fresh interpreter at base and head. The node file is absent at base and is run at head only. No added failures occur.</p>
<table><thead><tr><th>Arm</th><th>File</th><th>Result</th><th>Failures</th></tr></thead><tbody>{"".join(suite_rows)}</tbody></table>
<p>The all-to-any scratch mutation fails <code>{escape(", ".join(failures))}</code>, with <code>EXIT=1</code>; the head passes those same test ids. The forced stale limited contact rebuilds the entire mixed batch while leaving stationary cells unchanged.</p>
<p>Explicit execution counters (diverted selector 0, limited selector 1): <code>{escape(str(execution_rows))}</code>. The all-diverted carry sees zero wall calls; limited and mixed carries and the all-diverted certificate see positive counts. These observations are in <code>wall-controls.log</code>; the mutation is in <code>negative-control.log</code>.</p>"""
    table_rows = []
    count_rows = []
    for row in summary:
        table_rows.append(
            "<tr>"
            + "".join(
                f"<td>{escape(str(value))}</td>"
                for value in (
                    row["population"],
                    row["width"],
                    row["kind"],
                    *(f"{v:.3f}" for v in row["base"]["medians_ms"]),
                    *(f"{v:.3f}" for v in row["head"]["medians_ms"]),
                    f"{100 * (1 - row['head']['mean_ms'] / row['base']['mean_ms']):.2f}%",
                    f"{100 * max(row['base']['spread'], row['head']['spread']):.2f}%",
                )
            )
            + "</tr>"
        )
        for arm in ("base", "head"):
            count_rows.append(
                "<tr>"
                + "".join(
                    f"<td>{escape(str(value))}</td>"
                    for value in (
                        row["population"],
                        row["width"],
                        row["kind"],
                        arm,
                        *row[arm]["counts"].values(),
                    )
                )
                + "</tr>"
            )
    headers = "".join(
        f"<th>{text}</th>"
        for text in (
            "Population",
            "Width",
            "Read",
            "Base 1 [ms]",
            "Base 2 [ms]",
            "Head 1 [ms]",
            "Head 2 [ms]",
            "Reduction",
            "Largest spread",
        )
    )
    fragment = f"""<section id="read-diverted-wall-skip-record">
<h2>One wall decision per carried batch</h2>
<p>The carried refresh selects a scalar all-diverted branch before mapping over lanes. That branch omits wall evaluation. Limited and mixed batches retain the lane wall competition; full reads and terminal certificates retain the full topology owner.</p>
<p>The pre-fix execution counter observed <code>{escape(counter[0])}</code>. Head execution controls require zero calls for an all-diverted carry and positive counts for limited, mixed and certified reads. A scratch mutation using any diverted lane instead of every diverted lane is refused by the mixed parity check.</p>
<figure id="read-diverted-wall-skip-figure"><img src="/nova/figures/sub-millisecond-forward-path/read-diverted-wall-skip/read-wall.svg" alt="Paired CPU read time by batch width for diverted and mixed batches"><figcaption>Native CPU read over {cells} cells on the shared single-null fixture vessel. Two independent physical fields share one operator bank: the closed-form single-null diverted field and a wall-limited polynomial with an internal saddle. Full reads are dashed; carried reads are solid. Blue is head; grey is base. Each plotted value averages two fresh-process medians of twelve warm calls after five untimed warm-ups. Width one of the mixed family is its limited member, because one lane cannot be mixed. These CPU measurements are not H200 measurements and do not establish the plan's per-slice device bound.</figcaption></figure>
<h3 id="read-diverted-wall-skip-times">Paired CPU reads</h3>
<table><thead><tr>{headers}</tr></thead><tbody>{"".join(table_rows)}</tbody></table>
<p>Every returned numeric field agrees across base/head at rtol 1e-10 and atol 1e-12; boolean and integer fields agree exactly, including validity and rebuild flags. Maximum finite absolute difference: {max_error:.6g}. All process-pair spreads are at most 10% and both processes reproduce the lowered counts.</p>
<h3 id="read-diverted-wall-skip-counts">Lowered native programs</h3>
<p>Counts include both dynamic branches, including the untaken full-read fallback. They describe static optimized HLO operations, StableHLO operations, while sites and fusion sites, respectively, rather than dynamic executions. The wall execution counter above establishes which branch actually runs.</p>
<table><thead><tr><th>Population</th><th>Width</th><th>Read</th><th>Arm</th><th>HLO ops</th><th>StableHLO ops</th><th>While sites</th><th>Fusion sites</th></tr></thead><tbody>{"".join(count_rows)}</tbody></table>
<h3 id="read-diverted-wall-skip-provenance">Provenance and verification</h3>
{verification}
<p>Base <code>{pairs["base"][0]["revision"]}</code>; measured head <code>{pairs["head"][0]["revision"]}</code>. Instrument SHA-256 <code>{pairs["head"][0]["instrument_sha256"]}</code>; fixture SHA-256 <code>{pairs["head"][0]["helper_sha256"]}</code>. Instrument and fixture were committed before measurement. Raw receipts and complete logs are under <code>{measurement_run}</code>; regression logs are under <code>{args.run}</code>. Each receipt records the imported module, scratch working directory, backend, device, compile time and individual warm samples. Each process log ends in its captured exit status.</p>
</section>
"""
    destination = (
        args.root
        / "docs/evidence/fragments/sub-millisecond-forward-path/read-diverted-wall-skip.html"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(fragment)
    (args.run / "summary.json").write_text(
        json.dumps(dict(rows=summary, max_absolute_error=max_error), indent=2)
    )
    print("PAIRED_RECEIPTS_VALIDATED", len(summary), "rows", flush=True)


if __name__ == "__main__":
    main()
