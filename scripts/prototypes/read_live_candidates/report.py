"""Compare paired CPU receipts and publish the candidate-inspection evidence."""

import argparse
import hashlib
from html import escape
import json
from pathlib import Path
import re
import statistics


def table(headers, rows):
    def cells(tag, values):
        return (
            "<tr>"
            + "".join(f"<{tag}>{escape(str(value))}</{tag}>" for value in values)
            + "</tr>"
        )

    return (
        "<table><thead>"
        + cells("th", headers)
        + "</thead><tbody>"
        + "".join(cells("td", row) for row in rows)
        + "</tbody></table>"
    )


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("run", type=Path)
    parser.add_argument("--suites", type=Path)
    args = parser.parse_args()
    suites = args.suites or args.run
    rows, payloads = {}, {}
    timings, stages, lowered, native_counts = [], [], [], []
    for arm in ("base", "head"):
        for kind in ("full", "carried"):
            for process in (1, 2):
                name = f"{arm}-{kind}-{process}"
                payload = json.loads((args.run / f"{name}.json").read_text())
                assert payload["complete"] and payload["backend"] == "cpu"
                assert (
                    (args.run / f"{name}.log").read_text().rstrip().endswith("EXIT=0")
                )
                payloads[arm, kind, process] = payload
                for row in payload["rows"]:
                    assert row["valid"] and row["negative_control"]["passed"]
                    assert abs(row["coverage_gap"]) <= 0.10
                    rows[arm, kind, process, row["width"]] = row
                    for owner, stage in row["stages"].items():
                        counts = {
                            key: sum(
                                r[key] for r in row["regions"] if r["owner"] == owner
                            )
                            for key in (
                                "optimized_hlo_ops",
                                "stablehlo_ops",
                                "fusion_sites",
                                "while_sites",
                                "custom_call_sites",
                            )
                        }
                        stages.append(
                            [
                                arm,
                                kind,
                                row["width"],
                                process,
                                owner,
                                f"{stage['median_ms'] / 1000:.7f}",
                            ]
                        )
                        lowered.append(
                            [arm, kind, row["width"], process, owner, *counts.values()]
                        )
    assert len({p["job"] for p in payloads.values()}) == 1
    assert len({p["instrument_sha256"] for p in payloads.values()}) == 1
    summary = []
    for width in (1, 16):
        for kind in ("full", "carried"):
            entry = dict(width=width, kind=kind)
            for arm in ("base", "head"):
                pair = [rows[arm, kind, p, width] for p in (1, 2)]
                values = [row["native_ms"] for row in pair]
                spread = 100 * abs(values[1] - values[0]) / statistics.mean(values)
                assert spread <= 10, (arm, kind, width, spread)
                assert pair[0]["native_counts"] == pair[1]["native_counts"]
                assert pair[0]["regions"] == pair[1]["regions"]
                native_counts.append(
                    [arm, kind, width, *pair[0]["native_counts"].values()]
                )
                entry[arm] = dict(ms=values, spread_percent=spread)
                timings.append(
                    [
                        arm,
                        kind,
                        width,
                        *(f"{value / 1000:.6f}" for value in values),
                        f"{spread:.2f}%",
                        *(f"{100 * row['coverage_gap']:+.2f}%" for row in pair),
                    ]
                )
            entry["reduction_percent"] = 100 * (
                1
                - statistics.mean(entry["head"]["ms"])
                / statistics.mean(entry["base"]["ms"])
            )
            summary.append(entry)
    (args.run / "comparison.json").write_text(json.dumps(summary, indent=2))

    import matplotlib.pyplot as plt
    from scripts.prototypes.bank_solve_profile.plot import save_checked

    plt.style.use("data-ink")
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.2), dpi=100)
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.17, top=0.84, wspace=0.32)
    for ax, width in zip(axes, (1, 16), strict=True):
        for offset, arm, colour, marker in (
            (-0.12, "base", "#929292", "s"),
            (0.12, "head", "#1f5d92", "o"),
        ):
            for index, kind in enumerate(("full", "carried")):
                values = [rows[arm, kind, p, width]["native_ms"] / 1000 for p in (1, 2)]
                ax.plot([index + offset] * 2, values, color=colour, linewidth=2.6)
                ax.plot(
                    index + offset,
                    statistics.mean(values),
                    marker=marker,
                    color=colour,
                    markersize=9,
                    label=arm if index == 0 and width == 1 else None,
                    zorder=3 if arm == "head" else 2,
                )
        ax.set_xticks((0, 1), ("full", "carried"))
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(bottom=0)
        ax.set_ylabel("CPU wall per batch [s]")
        ax.set_xlabel(f"{width} slice" + ("s" if width > 1 else ""))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    figure = args.root / (
        "docs/figures/sub-millisecond-forward-path/"
        "read-live-candidates/paired-cpu-wall.svg"
    )
    save_checked(fig, figure)
    fig.savefig(args.run / "paired-cpu-wall.png", dpi=100)
    plt.close(fig)

    base = payloads["base", "full", 1]["revision"]
    head = payloads["head", "full", 1]["revision"]
    wide = next(row for row in summary if row["width"] == 16 and row["kind"] == "full")
    instrument_hash = payloads["base", "full", 1]["instrument_sha256"]
    direction = "falls" if wide["reduction_percent"] >= 0 else "rises"
    fragment = args.root / (
        "docs/evidence/fragments/sub-millisecond-forward-path/read-live-candidates.html"
    )
    fragment.parent.mkdir(parents=True, exist_ok=True)
    suite_rows = []
    for arm, files in (
        ("base", ("tracked_read", "support_read")),
        ("head", ("tracked_read", "support_read", "read_live_candidates")),
    ):
        for name in files:
            text = (suites / f"{arm}-test_{name}.log").read_text()
            assert text.rstrip().endswith("EXIT=0")
            summaries = re.findall(r"^.*\d+ passed.*$", text, re.M)
            assert summaries, (arm, name)
            suite_rows.append([arm, "tests/test_" + name + ".py", summaries[-1]])
    negative = (suites / "negative-control.log").read_text()
    assert negative.rstrip().endswith("EXIT=1") and "FAILED " in negative
    fragment.write_text(
        '<section id="read-live-candidates-result">\n'
        "<h2>Compact support inspection follows the live saddle set</h2>\n"
        "<p>The compact branch constructs the fallback normal form separately "
        "and inspects support only for live saddles. It retains the full-census "
        "overflow branch unchanged. The unchanged-base runtime control recorded "
        "<code>assert 5 == (1 + 2)</code> for one saddle: three candidate "
        "inspections plus initial and terminal support. The head guard requires "
        "zero, one and two live saddles to execute two, three and four support "
        "inspections respectively.</p>\n"
        f"<p>Paired full-read width-sixteen CPU wall {direction} "
        f"{abs(wide['reduction_percent']):.2f}%. This is a CPU comparison, "
        "not an H200 latency verdict. Both fresh processes agree within 10% "
        "on every native arm; complete process spreads and stage-sum gaps are "
        "retained below. Width-one and width-sixteen full and carried reads "
        "include current booking.</p>\n"
        '<figure id="read-live-candidates-figure">'
        '<img src="/nova/figures/sub-millisecond-forward-path/'
        'read-live-candidates/paired-cpu-wall.svg" '
        'alt="Paired CPU full and carried read walls at widths one and sixteen">'
        "<figcaption>Squares show base and circles head. Each point is the "
        "mean of two fresh-process medians; vertical segments join those "
        "medians. Twelve synchronized native samples bracket the attribution "
        "chain in each process. Both panels use seconds per batch, with "
        "independent vertical scales. The shared save function checks text "
        "bounds and collisions; the produced PNG is inspected before delivery."
        "</figcaption></figure>\n"
        '<h3 id="read-live-candidates-parity">Parity and adversarial controls</h3>'
        "<p>Every floating leaf of the topology and booking receipts is compared "
        "with the unchanged full-census branch at rtol 1e-10, atol 1e-12; "
        "integer and validity leaves are exact. Cases cover no saddles, one "
        "saddle, repeated candidate slots, two distinct unequal saddles, a "
        "limited field and compact overflow. A scratch mutation drops the "
        "real saddle from the live set and must fail the one-saddle parity "
        "test. Paired tracked-read and support-read suites additionally cover "
        "rebuild behavior.</p>\n"
        + table(["Arm", "Test file", "Runner summary"], suite_rows)
        + "<p>The new test file is absent from the base tree. The paired "
        "suite job runs each existing file in its own fresh process; the "
        "final committed head is checked again with the armed candidate "
        "test file. The mutation log names the one-saddle parity failure "
        "and ends with EXIT=1.</p>"
        '<h3 id="read-live-candidates-native">Native CPU walls</h3>'
        + table(
            [
                "Arm",
                "Read",
                "Width",
                "Process 1 [s]",
                "Process 2 [s]",
                "Spread",
                "Stage gap 1",
                "Stage gap 2",
            ],
            timings,
        )
        + '<h3 id="read-live-candidates-native-counts">Native lowered counts</h3>'
        + table(
            [
                "Arm",
                "Read",
                "Width",
                "Optimized HLO",
                "StableHLO",
                "Fusion",
                "While",
                "Custom call",
            ],
            native_counts,
        )
        + '<h3 id="read-live-candidates-stages">Exclusive stage walls</h3>'
        + table(["Arm", "Read", "Width", "Process", "Owner", "Median [s]"], stages)
        + '<h3 id="read-live-candidates-counts">Lowered stage counts</h3>'
        + table(
            [
                "Arm",
                "Read",
                "Width",
                "Process",
                "Owner",
                "Optimized HLO",
                "StableHLO",
                "Fusion",
                "While",
                "Custom call",
            ],
            lowered,
        )
        + '<h3 id="read-live-candidates-provenance">Provenance and limits</h3>'
        f"<p>Base <code>{base}</code>; measured source head <code>{head}</code>. "
        "One all_debug job runs two fresh processes per revision and read kind, "
        "each at widths one and sixteen. Base and head run on disjoint "
        "eight-CPU affinity sets, swapped for the second process pair. "
        "The preceding sequential attempt was refused when the carried "
        "width-sixteen stage sum exceeded native wall by 10.435662%; its "
        "receipts are retained and are excluded from these tables. "
        "The original committed instrument is "
        "<code>scripts/prototypes/bank_read_attribution/measure.py</code> from "
        "<code>d3d17c064bd10c209aa4adec67c0ce9e5d1a3785</code>, extracted "
        f"without changes; SHA-256 <code>{instrument_hash}</code>. "
        "Its native-parity and 10% additive-coverage guards remain enabled. "
        "The fragment does not claim independently timed clipping and "
        "connectivity for full reads: they remain under the joint rebuilt "
        "reading owner. The live candidate loop is one joint region under "
        "that owner, including its candidate normal form; the separate "
        "curvature row covers only independently lowered work. Stage labels "
        "therefore describe compilation-boundary ownership. "
        "Lowered counts are static program sites, including "
        "untaken fallbacks, and are not dynamic GPU launch counts. "
        "Native counts include the preserved overflow branch and the live "
        "candidate loop; stage counts cover the specialized diagnostic partition. "
        f"Raw timing receipts are under <code>{escape(str(args.run))}</code>; "
        f"suite and negative-control logs are under "
        f"<code>{escape(str(suites))}</code>. The reproduction and refused "
        "attempts remain in their parent run directory.</p>\n</section>\n"
    )
    print("COMPARISON", json.dumps(summary), flush=True)
    print("REPORT_SHA256", hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
