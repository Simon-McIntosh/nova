"""Validate paired attribution receipts and draw their width-sixteen costs."""

import argparse
import hashlib
from html import escape
import json
import math
from pathlib import Path
import re
import statistics
import subprocess


def table(headers, rows):
    def cells(tag, values):
        return (
            "<tr>"
            + "".join(f"<{tag}>{escape(str(v))}</{tag}>" for v in values)
            + "</tr>"
        )

    return "\n".join(
        [
            "<table><thead>" + cells("th", headers) + "</thead><tbody>",
            *(cells("td", row) for row in rows),
            "</tbody></table>",
        ]
    )


def spread(values):
    return 100 * (max(values) - min(values)) / statistics.mean(values)


def instrument_provenance(root, run):
    """Bind each recorded source digest to identical committed file bytes."""
    path = "scripts/prototypes/bank_read_attribution/measure.py"
    history = subprocess.check_output(
        ["git", "-C", str(root), "log", "--full-history", "--format=%H", "--", path],
        text=True,
    ).splitlines()
    revisions = {}
    for revision in history:
        source = subprocess.check_output(
            ["git", "-C", str(root), "show", f"{revision}:{path}"]
        )
        revisions.setdefault(hashlib.sha256(source).hexdigest(), revision)
    rows = []
    for kind in ("full", "carried"):
        for process in (1, 2):
            name = f"{kind}-{process}.json"
            receipt = json.loads((run / name).read_text())
            digest = receipt["instrument_sha256"]
            assert digest in revisions, f"uncommitted instrument: {name} {digest}"
            rows.append([name, receipt["revision"], digest, revisions[digest]])
            print("INSTRUMENT_SOURCE", json.dumps(rows[-1]), flush=True)
    return table(
        [
            "Receipt",
            "Recorded revision",
            "Instrument SHA-256",
            "Matching file revision",
        ],
        rows,
    )


def fragment(root, run, summary, tables, job, provenance):
    """Bind the measured table to a bounded, explicitly estimated next brief."""
    wide = {row["kind"]: row for row in summary if row["width"] == 16}

    def cost(kind, owner):
        return wide[kind]["stages"].get(owner, {"mean_ms": 0})["mean_ms"]

    booking = statistics.mean(cost(k, "current.book") for k in wide)
    candidates = 0.35 * (
        cost("full", "support_read._rebuilt_reading")
        + cost("full", "topology._fragments_at_level")
        + cost("full", "topology._support_at_level/connectivity")
    )
    wall = cost("carried", "topology._wall_events")
    changes = [
        (
            booking * 0.25,
            "Widen scalar booking's conic tile",
            f"Target <code>current.book</code> and "
            f"<code>current._SCALAR_CUT_BATCH_WIDTH</code>: try 128 instead of 32. "
            f"The 113 conic cut cells then need one bucket instead of four; the "
            f"four normal-form cut cells keep their independent bucket. "
            f"Expected CPU saving: {0.25 * booking:.1f}–{0.5 * booking:.1f} ms "
            f"per width-sixteen read, a 25–50% reduction of the measured "
            f"{booking:.1f} ms booking stage. This range is an engineering "
            f"estimate informed by the analogous tangent-booking tile result, "
            f"not a scalar-booking ablation. Verify both process pairs, all "
            f"booked moments and normalization against the unchanged scalar "
            f"owner, trial and converged residuals at the existing tolerance, "
            f"and dynamic conic-bucket counts on the H200. Include fewer than "
            f"32 and more than 128 cut cells so the population bound is tested.",
        ),
        (
            candidates,
            "Inspect only live saddle candidates",
            f"Target <code>topology._read_at_nulls.compact_candidates</code>, "
            f"<code>_fragments_at_level</code> and "
            f"<code>_support_at_level</code>. This single-saddle fixture packs "
            f"three slots: the axis, the saddle and a padded axis duplicate. "
            f"Keep the axis normal form for the limited fallback, but avoid "
            f"the axis and padding's unnecessary support inspections. The "
            f"full read makes three candidate support evaluations plus initial "
            f"and terminal support evaluations; removing two of these five "
            f"gives a first CPU estimate of {candidates:.1f} ms per full read "
            f"(a discounted 35% budget of its joint field/support stage). "
            f"Vectorization and fusion make this a scaling estimate, not a "
            f"timed saving. Verify admitted saddles, boundary ownership, "
            f"membership and qualification against the full owner, including "
            f"zero, one and multiple saddles and the overflow fallback; then "
            f"measure the candidate dimension and device time on the H200.",
        ),
        (
            wall,
            "Move diverted wall skipping outside the lane map",
            f"Target <code>topology.refresh_at_stationary_points.wall_contact</code> "
            f"and its batched caller. The per-lane conditional becomes a select "
            f"under vmap, so this all-diverted carried batch still executes "
            f"<code>_wall_events</code>. A batch-uniform diverted branch can "
            f"remove approximately the measured {wall:.1f} ms wall stage per "
            f"carried width-sixteen read, less dispatch overhead. Retain wall "
            f"competition in a full certificate and wall evaluation for limited "
            f"and mixed batches. Verify all-diverted, all-limited and mixed "
            f"batches against the native read; force a limited contact change "
            f"to prove the wall path still runs and requests a rebuild. "
            f"Confirm the missing wall executions with a device trace whose "
            f"limited positive control sees the known wall kernels.",
        ),
    ]
    changes.sort(key=lambda row: -row[0])
    native = "; ".join(
        f"{kind}: "
        + "/".join(f"{v:.1f}" for v in wide[kind]["native_ms"])
        + " ms per batch"
        for kind in ("carried", "full")
    )
    largest_spread = max(row["native_spread_percent"] for row in summary)
    max_stage_spread = max(
        stage["spread_percent"] for row in summary for stage in row["stages"].values()
    )
    destination = root / (
        "docs/evidence/fragments/sub-millisecond-forward-path/"
        "bank-read-attribution.html"
    )
    destination.write_text(
        '<section id="bank-read-attribution-result">\n'
        "<h2>Read cost belongs to booking and refreshed geometry</h2>\n"
        f"<p>Two fresh CPU processes per read kind, at widths one and sixteen, "
        f"measure the diverted 550-cell bank without changing product source. "
        f"At width sixteen, {native}. The exclusive stage sums meet the "
        f"10% native-wall bound in every process, kind and width. Maximum "
        f"end-to-end process spread is {largest_spread:.2f}%; the maximum "
        f"individual-stage spread is {max_stage_spread:.2f}%. The tables retain "
        f"every spread rather than hiding small-stage variability. These "
        f"are paired CPU measurements, not an H200 speed verdict.</p>\n"
        '<figure id="bank-read-attribution-figure">'
        '<img src="/nova/figures/sub-millisecond-forward-path/'
        'bank-read-attribution/stage-cost.svg" '
        'alt="Function-owned CPU costs of full and carried sixteen-lane reads">'
        "<figcaption>Width sixteen, synchronized warm CPU wall divided by "
        "sixteen. Squares show full reads and circles carried reads; horizontal "
        "segments join the two process medians. Separate linear axes show "
        "the five largest stages on the left and the smaller stages on the "
        "right, both in milliseconds per slice. Tables use total "
        "batch milliseconds. The single 1400 by 720 canvas passes the shared "
        "text-bound and collision checks; the rendered artifact is inspected "
        "before delivery.</figcaption></figure>\n"
        '<h3 id="bank-read-attribution-method">What one read includes</h3>\n'
        "<p>The end-to-end owner calls native <code>support_read.read_batch</code> "
        "then <code>support_read.book</code>, using the booking owner of "
        "<code>read_and_book</code>. The full arm uses <code>tracking=None</code>: "
        "its <code>whole_mesh</code> is the same full geometry and qualification "
        "work used by <code>full_verdict</code>, with an initial counter instead "
        "of incrementing a prior counter. It does not include a preceding "
        "carried trial. The carried arm supplies tracking from a full read of "
        "the identical state; every lane is valid and reports no rebuild. "
        "Both arms include current quadrature and normalization, but exclude "
        "Newton updates and residual-norm reductions.</p>\n"
        '<p>The fixture is <code>bank_newton_step.measure.machine("diverted")</code>, '
        "with its cached 2048-node, 550-cell, three-moment bank. Its content "
        "digest is <code>sha256:"
        "6ebe02e02ae9ed8646077b3116596fa08ff0f18cb7f17d2566f91f779b4c17be</code>. "
        "Both widths take a prefix of the same sixteen perturbations from "
        "2e-5 to 6e-5; warm seeds are offset by 0.03 median cell pitch. "
        "Each lane admits one saddle and has 117 cut cells. "
        "<code>configure_dtypes()</code> precedes array construction and x64 "
        f"is asserted. One <code>all_debug</code> job, {job}, runs the four "
        "fresh processes serially on one node with eight CPUs and 96 GB.</p>\n"
        "<p>The prototype exposes the measured steady branch and the compact "
        "candidate branch with anchored source extraction. Overflow and "
        "incidence-change predicates remain live through the validity output; "
        "a violated branch assumption fails native parity. Function-source "
        "metadata partitions the straight-line JAX graph after sharing pure "
        "common expressions, retaining every live "
        "crossing value as a runtime operand. The stationary solve, strip loops, "
        "connectivity propagation and booking population loops remain intact. "
        "Twelve warm dependency-chain repetitions time each synchronized "
        "exclusive region; regions with the same owner are added within each "
        "repetition before taking a median. Six native calls before and "
        "six after bracket the chain, twelve native samples in total. "
        "Every returned leaf is checked against "
        "the unspecialized native executable at rtol 1e-10, atol 1e-12, with "
        "validity and rebuild flags asserted. Compilation-boundary effects "
        "are retained as the reported stage-sum gap, without normalization.</p>\n"
        "<p><code>_fragments_at_level</code> includes quadratic clipping, "
        "strip-area quadrature and normal-form fragment geometry. "
        "On full reads these calculations and connected-label propagation "
        "remain with field construction in one "
        "<code>support_read._rebuilt_reading</code> stage, so the "
        "initial and terminal support evaluations keep XLA's geometry sharing. "
        "The carried arm separates clipping from the incidence comparison. "
        "The finer full split is diagnostic only: its +12.644% control failed "
        "the coverage bound even after graph sharing and is retained under "
        "the run's full-control receipt rather than added to the accepted "
        "table. Consequently full clipping and connectivity are covered "
        "jointly, not independently timed in the additive partition. "
        "Missing points and not-separate table entries mean work included "
        "under another owner, never a zero execution cost. "
        "<code>_support_at_level/connectivity</code> includes component "
        "propagation and axis ownership on full reads, and the local incidence "
        "comparison on carried reads. <code>current.book</code> includes "
        "whole-cell, conic and normal-form current quadrature and normalization. "
        "Curvature and saddle normal forms are separated from the axis/saddle "
        "qualification arithmetic. The remaining dispatch and validity masks "
        "have their own row.</p>\n"
        '<h3 id="bank-read-attribution-tables">'
        "Paired stages and lowered programs</h3>\n"
        "<p>Process spread is the absolute difference of the two medians "
        "divided by their mean. Counts describe independently compiled stage "
        "regions: optimized HLO operations, StableHLO operations and static "
        "fusion, while and custom-call sites. These are lowered launch sites, "
        "not dynamic GPU kernel counts; while-body executions and custom calls "
        "can execute repeatedly. Both fresh processes must reproduce the "
        "same counts. Native conditional programs also contain the untaken "
        "fallback branch; their counts are retained in each raw receipt.</p>\n"
        + tables
        + '\n<h3 id="bank-read-attribution-controls">Controls and provenance</h3>\n'
        "<p>The coverage guard is exercised by dropping the largest stage "
        "from each actual receipt; every omission is refused. The final "
        "report repeats the omission on the full width-sixteen receipt. "
        "A pilot first reproduced a +11.817% coverage refusal despite exact "
        "native outputs: splitting <code>step_residual</code> materialized "
        "zero-direction field work that the native compiler removed. The "
        "accepted instrument uses the requested direct booking owner. "
        "A subsequent full-read pilot was also refused at +19.490% because "
        "independently compiled cuts repeated common normal-form geometry and "
        "bank contractions. The accepted graph shares these pure expressions "
        "before partitioning. A common-expression control merges identical "
        "sines but keeps cosine and opposite signed-zero products distinct. "
        "A separate profile-argument error and a native-trace-cache pilot are "
        "retained as instrumentation failures; none changed product code.</p>\n"
        "<p>Raw receipts and complete per-process logs are in "
        f"<code>{escape(str(run))}</code>. "
        "Their first lines carry the revision, tree and actual command and "
        "their last lines are <code>EXIT=0</code>. The measured source digest "
        "and imported product module are printed in each receipt. "
        "Product-equivalence checks bind all five read owners to the base "
        "bytes; only prototype and evidence commits advance during the run. "
        "The product base is <code>7ca1b11a36f16779d583edb6ddbfb264218d24cb</code>. "
        "The committed root-argument driver and <code>h200.sh</code> can be "
        "submitted from main with <code>TMPDIR=/tmp</code> in the submit "
        "environment. No test suite pair is armed for this measurement node.</p>\n"
        "<p>The following bindings compare each receipt's instrument hash "
        "with the exact committed <code>measure.py</code> bytes. The first "
        "full process captured an uncommitted instrument; its exact bytes "
        "were recovered from the recorded edit sequence and committed. "
        "The other three processes used the retained canonical instrument. "
        "Original receipt revisions and hashes remain unchanged; a matching "
        "file revision identifies source bytes, not a new measurement.</p>\n"
        + provenance
        + "\n"
        "<p>Context: the earlier "
        '<a href="/nova/evidence/fragments/sub-millisecond-forward-path/'
        'bank-solve-trace-four-read.html">four-read H200 trace</a> measured '
        "35.692 ms per width-sixteen read plus fresh booking. "
        '<a href="/nova/evidence/fragments/sub-millisecond-forward-path/'
        'bank-solve-trace-head.html">The preceding trace</a> supplies '
        "the older call-level attribution. "
        '<a href="/nova/evidence/fragments/sub-millisecond-forward-path/'
        'bank-carried-decision.html">Carried-decision evidence</a> defines '
        "the fresh geometry and certificate contract; "
        '<a href="/nova/evidence/fragments/sub-millisecond-forward-path/'
        'bank-booking-cost.html">booking-cost evidence</a> motivates the '
        "scalar-tile experiment. Their GPU numbers are not rescaled from "
        "this CPU table.</p>\n"
        '<h3 id="bank-read-attribution-next">Ranked read-cost changes</h3>\n'
        "<p>Ranked by the conservative estimated CPU saving on this fixture. "
        "The following savings are proposals inferred from measured stage "
        "costs and source execution shape, not implemented speedups. "
        "They do not establish a per-slice H200 wall.</p>\n<ol>\n"
        + "\n".join(
            f"<li><strong>{title}.</strong> {body}</li>" for _, title, body in changes
        )
        + "\n</ol>\n</section>\n"
    )
    print("FRAGMENT", destination, flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    import sys

    sys.path.insert(0, str(args.root.resolve()))
    import matplotlib.pyplot as plt
    from scripts.prototypes.bank_solve_profile.plot import save_checked
    from scripts.prototypes.bank_read_attribution.measure import coverage

    receipts = {}
    provenance = instrument_provenance(args.root, args.run)
    jobs = set()
    stages = set()
    for kind in ("carried", "full"):
        for process in (1, 2):
            receipt = json.loads((args.run / f"{kind}-{process}.json").read_text())
            assert receipt["complete"] and receipt["backend"] == "cpu"
            jobs.add(receipt["job"])
            assert (
                (args.run / f"{kind}-{process}.log")
                .read_text()
                .rstrip()
                .endswith("EXIT=0")
            )
            shared = re.findall(
                r"^SHARED_EXPRESSIONS (\d+) (\d+)$",
                (args.run / f"{kind}-{process}.log").read_text(),
                re.M,
            )
            assert {int(width) for width, _ in shared} == {1, 16}
            assert all(int(count) > 0 for _, count in shared)
            assert sorted(row["width"] for row in receipt["rows"]) == [1, 16]
            for row in receipt["rows"]:
                assert row["valid"] and all(row["cut_cells"])
                assert row["negative_control"]["passed"]
                coverage(
                    {name: value["median_ms"] for name, value in row["stages"].items()},
                    row["native_ms"],
                )
                assert all(len(v["samples_ms"]) >= 12 for v in row["stages"].values())
                receipts[kind, process, row["width"]] = row
                stages.update(row["stages"])
    stages = sorted(
        stages,
        key=lambda name: (
            -max(
                receipts[k, p, 16]["stages"].get(name, {"median_ms": 0})["median_ms"]
                for k in ("carried", "full")
                for p in (1, 2)
            )
        ),
    )
    assert len(jobs) == 1, jobs
    timing_rows, count_rows, closure_rows = [], [], []
    summary = []
    for width in (1, 16):
        for kind in ("carried", "full"):
            pair = [receipts[kind, p, width] for p in (1, 2)]
            native = [row["native_ms"] for row in pair]
            closure_rows.append(
                [
                    width,
                    kind,
                    *(f"{v:.3f}" for v in native),
                    f"{spread(native):.2f}%",
                    *(f"{100 * row['coverage_gap']:+.2f}%" for row in pair),
                ]
            )
            per_stage = {}
            for name in stages:
                values = [
                    row["stages"].get(name, {"median_ms": 0})["median_ms"]
                    for row in pair
                ]
                change = spread(values) if max(values) else 0
                separate = name in pair[0]["stages"]
                timing_rows.append(
                    [
                        width,
                        kind,
                        name,
                        *(f"{v:.4f}" if separate else "not separate" for v in values),
                        f"{change:.2f}%" if separate else "—",
                    ]
                )
                if separate:
                    per_stage[name] = dict(
                        process_ms=values,
                        mean_ms=statistics.mean(values),
                        spread_percent=change,
                    )
                counts = []
                for row in pair:
                    counts.append(
                        {
                            key: sum(
                                region[key]
                                for region in row["regions"]
                                if region["owner"] == name
                            )
                            for key in (
                                "optimized_hlo_ops",
                                "stablehlo_ops",
                                "fusion_sites",
                                "while_sites",
                                "custom_call_sites",
                            )
                        }
                    )
                assert counts[0] == counts[1], (kind, width, name, counts)
                count_rows.append(
                    [
                        width,
                        kind,
                        name,
                        *(counts[0].values() if separate else ["not separate"] * 5),
                    ]
                )
            summary.append(
                dict(
                    width=width,
                    kind=kind,
                    native_ms=native,
                    native_spread_percent=spread(native),
                    stages=per_stage,
                )
            )
    # This measured omission exercises the same guard used to admit the table.
    damaged = dict(receipts["full", 1, 16]["stages"])
    omitted = max(damaged, key=lambda name: damaged[name]["median_ms"])
    del damaged[omitted]
    try:
        coverage(
            {name: value["median_ms"] for name, value in damaged.items()},
            receipts["full", 1, 16]["native_ms"],
        )
    except AssertionError as error:
        print("OMISSION_REFUSED", omitted, str(error), flush=True)
    else:
        raise AssertionError("the actual receipt's omitted stage passed coverage")
    (args.run / "summary.json").write_text(json.dumps(summary, indent=2))
    tables = "\n".join(
        [
            table(
                [
                    "Width",
                    "Read",
                    "Native process 1 [ms]",
                    "Native process 2 [ms]",
                    "Process spread",
                    "Stage sum gap 1",
                    "Stage sum gap 2",
                ],
                closure_rows,
            ),
            table(
                [
                    "Width",
                    "Read",
                    "Owning function",
                    "Process 1 [ms]",
                    "Process 2 [ms]",
                    "Spread",
                ],
                timing_rows,
            ),
            table(
                [
                    "Width",
                    "Read",
                    "Owning function",
                    "Optimized HLO ops",
                    "StableHLO ops",
                    "Fusion sites",
                    "While sites",
                    "Custom-call sites",
                ],
                count_rows,
            ),
        ]
    )
    (args.run / "tables.html").write_text(tables)
    labels = {
        "current.book": "booking + clipped quadrature",
        "fixed_points.solve": "stationary-point search + polish",
        "topology._wall_events": "wall contact",
        "topology._fragments_at_level": "cut-cell clipping + area quadrature",
        "topology._support_at_level/connectivity": "topology + connectivity",
        "support_read.nodal_jets/field_from_jets": "nodal contraction + Bell field",
        "topology._curvature_derivatives/saddle_normal_form": (
            "saddle curvature + normal form"
        ),
        "topology._read_at_nulls/qualification": "axis + saddle qualification",
        "support_read._rebuilt_reading": "field + clipping + support decision",
        "support_read.read_batch/verdict": "read dispatch + booking verdict",
    }
    plt.style.use("data-ink")
    fig = plt.figure(figsize=(14, 7.2), dpi=100)
    for position, subset, tick_step in (
        ((0.30, 0.15, 0.19, 0.72), stages[:5], 100.0),
        ((0.78, 0.15, 0.19, 0.72), stages[5:], 0.5),
    ):
        ax = fig.add_axes(position)
        maximum = 0.0
        for offset, kind, colour, marker in (
            (-0.13, "full", "#929292", "s"),
            (0.13, "carried", "#1f5d92", "o"),
        ):
            for index, name in enumerate(subset):
                values = [
                    receipts[kind, p, 16]["stages"].get(name, {"median_ms": 0})[
                        "median_ms"
                    ]
                    / 16
                    for p in (1, 2)
                ]
                maximum = max(maximum, *values)
                if max(values):
                    ax.plot(
                        values,
                        [index + offset] * 2,
                        color=colour,
                        linewidth=2.6,
                        zorder=2,
                    )
                    ax.plot(
                        statistics.mean(values),
                        index + offset,
                        marker=marker,
                        color=colour,
                        markersize=7,
                        label=kind if name == "current.book" else None,
                        zorder=3 if kind == "carried" else 2,
                    )
        upper = math.ceil(1.05 * maximum / tick_step) * tick_step
        ax.set_xlim(-0.03 * upper, upper)
        ax.set_xticks(
            [tick_step * index for index in range(round(upper / tick_step) + 1)]
        )
        ax.ticklabel_format(axis="x", style="plain", useOffset=False)
        ax.set_yticks(range(len(subset)), [labels[name] for name in subset])
        ax.invert_yaxis()
        ax.set_xlabel("CPU wall per slice [ms]")
        ax.spines[["left", "top", "right"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.grid(False)
    handles, legend_labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=2, frameon=False)
    destination = (
        args.root
        / "docs/figures/sub-millisecond-forward-path/bank-read-attribution"
        / "stage-cost.svg"
    )
    save_checked(fig, destination)
    fig.savefig(args.run / "stage-cost.png", dpi=100)
    plt.close(fig)
    fragment(args.root, args.run, summary, tables, next(iter(jobs)), provenance)
    print("SUMMARY", json.dumps(summary), flush=True)
    print(
        "GATE_SUMMARY",
        json.dumps(
            dict(
                passed=True,
                process_width_rows=len(receipts),
                maximum_coverage_gap_percent=max(
                    100 * abs(row["coverage_gap"]) for row in receipts.values()
                ),
                maximum_native_spread_percent=max(
                    row["native_spread_percent"] for row in summary
                ),
                maximum_stage_spread_percent=max(
                    stage["spread_percent"]
                    for row in summary
                    for stage in row["stages"].values()
                ),
                maximum_absolute_difference=max(
                    row["maximum_absolute_difference"] for row in receipts.values()
                ),
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
