"""Publish selected fresh-process walls without merging disagreeing pairs."""

import argparse
import html
import json
from pathlib import Path
import statistics


def load(run):
    paths = [run / "h200-levers-corrected/summary.json"]
    inventory = json.loads((run / "measurement-jobs.json").read_text())
    paths += [Path(job["output"]) / "summary.json" for job in inventory["jobs"]]
    paths += sorted(run.glob("measurement-repair-*/summary.json"))
    comparisons = {}
    headline = None
    for path in paths:
        value = json.loads(path.read_text())
        if "repair" in path.parent.name and not value["completed"]:
            continue
        for row in value["comparisons"]:
            key = (
                row["name"],
                row["width"],
                row.get("step", 1e-5),
                row.get("population", "diverted"),
            )
            comparisons[key] = row
        if "headline" in value:
            headline = value["headline"]
    return list(comparisons.values()), headline, inventory


def arms(comparison):
    return comparison.get("tiles") or [comparison["off"], comparison["on"]]


def profile_step(comparison):
    """Name physical profile increments only for continuation cases."""
    if comparison["name"] == "factor-carry":
        return f"{comparison['step']:g}"
    if comparison["name"] == "lane-drive":
        return "1e-5" if comparison["width"] == 1 else "1e-5 / 1e-3"
    return "—"


def population(comparison):
    """Identify the one-lane limited counterpart of the mixed wall batch."""
    value = comparison.get("population", "diverted")
    if value == "mixed" and comparison["width"] == 1:
        return "limited"
    return value


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    run, root = args.run.resolve(), args.root.resolve()
    comparisons, headline, inventory = load(run)
    audit = json.loads((run / "audit-summary.json").read_text())
    assert audit["completed"] and audit["input_mismatches"] == 0
    table = []
    compile_table = []
    mismatches = set()
    max_spread = 0.0
    for comparison in comparisons:
        for arm in arms(comparison):
            records = arm["records"]
            assert len(records) == 2
            first, second = records
            reference = arms(comparison)[0]["records"]
            if any(
                a["input_sha256"] != b["input_sha256"]
                for a, b in zip(reference, records, strict=True)
            ):
                mismatches.add(comparison["name"])
            rows = {row["name"]: row for row in second["rows"]}
            for row in first["rows"]:
                other = rows[row["name"]]
                identity = [
                    comparison["name"],
                    str(comparison["width"]),
                    profile_step(comparison),
                    population(comparison),
                    str(arm["arm"]),
                    row["name"],
                ]
                if "compile_seconds" in row:
                    compile_table.append(
                        identity
                        + [
                            f"{row['compile_seconds']:.6f}",
                            f"{other['compile_seconds']:.6f}",
                        ]
                    )
                if "median_ms" not in row:
                    continue
                spread = arm["spreads"][row["name"]]
                assert spread <= 0.05
                max_spread = max(max_spread, spread)
                table.append(
                    identity
                    + [
                        f"{row['median_ms']:.6f}",
                        f"{other['median_ms']:.6f}",
                        f"{100 * spread:.3f}%",
                    ]
                )
    per_slice = [
        [
            row["per_slice_ms"]
            for row in record["rows"]
            if row["name"].startswith("slice-") and row["name"] != "slice-1"
        ]
        for record in headline["records"]
    ]
    warm = [statistics.median(values) for values in per_slice]
    jobs = ", ".join(job["job_id"] for job in inventory["jobs"])

    def render_table(identifier, headers, values):
        return (
            f'<table id="h200-levers-{identifier}"><thead><tr>'
            + "".join(f"<th>{html.escape(x)}</th>" for x in headers)
            + "</tr></thead><tbody>\n"
            + "\n".join(
                "<tr>" + "".join(f"<td>{html.escape(x)}</td>" for x in row) + "</tr>"
                for row in values
            )
            + "\n</tbody></table>\n"
        )

    assert not mismatches
    verdicts = []
    for comparison in comparisons:
        name, width = comparison["name"], comparison["width"]
        options = arms(comparison)
        left, right = options[0], options[-1]
        stages = (
            ["full-to-carried"]
            if name == "carried-decisions"
            else ["warm-slices"]
            if name == "factor-carry"
            else list(left["metrics"])
        )
        for stage in stages:

            def values(arm, baseline):
                if stage == "warm-slices":
                    return [
                        statistics.median(
                            v
                            for k, v in metrics.items()
                            if k.startswith("slice-") and k != "slice-1"
                        )
                        for metrics in (arm["metrics"], arm["second_metrics"])
                    ]
                key = (
                    ("full" if baseline else "carried")
                    if stage == "full-to-carried"
                    else stage
                )
                return [arm["metrics"][key], arm["second_metrics"][key]]

            a, b = values(left, True), values(right, False)
            spread = max(abs(a[0] - a[1]) / min(a), abs(b[0] - b[1]) / min(b))
            assert spread <= 0.05
            margin = max(0.03, spread)
            change = 100 * (b[0] / a[0] - 1)
            if b[0] > a[0] * (1 + margin):
                verdict = "off: on is slower beyond margin"
            elif a[0] > b[0] * (1 + margin):
                verdict = "on: faster beyond margin"
            else:
                verdict = "retain incumbent: difference unresolved"
            if name == "scalar-tile":
                assert a[0] > b[0] * (1 + margin)
                verdict = "128: fastest measured tile; change default from 32"
            elif name == "lane-drive":
                assert b[0] > a[0] * (1 + margin)
                verdict = "shared scalar default; independent arrays remain opt-in"
            elif name == "wall-skip" and stage == "full":
                verdict = "attribution only: skip is in carried branch"
            elif name == "factor-carry":
                assert b[0] > a[0] * (1 + margin)
                verdict = "off; factor carry remains opt-in"
            elif name in ("booking", "live-candidates", "wall-skip"):
                assert not verdict.startswith("off:")
            verdicts.append(
                [
                    name,
                    str(width),
                    profile_step(comparison),
                    population(comparison),
                    stage,
                    ", ".join(f"{x:.6f}" for x in a),
                    ", ".join(f"{x:.6f}" for x in b),
                    f"{change:+.3f}%",
                    f"{100 * margin:.3f}%",
                    verdict,
                ]
            )
    repairs = json.loads((run / "repair-jobs.json").read_text())["jobs"]
    repair_ids = ", ".join(x["job_id"] for x in repairs if not x.get("superseded_by"))
    for left, right in zip(
        headline["records"][0]["rows"], headline["records"][1]["rows"], strict=True
    ):
        if "compile_seconds" in left:
            compile_table.append(
                [
                    "secant-headline",
                    "16",
                    "1e-05",
                    "diverted",
                    "off",
                    left["name"],
                    f"{left['compile_seconds']:.6f}",
                    f"{right['compile_seconds']:.6f}",
                ]
            )
    residual = audit["max_solved_residual"]
    errors = json.loads((run / "precision-audit.json").read_text())
    position_error = max(
        value
        for entry in errors
        for key, value in entry["errors"].items()
        if key.endswith(("-axis", "-saddle"))
    )
    flux_error = max(
        value
        for entry in errors
        for key, value in entry["errors"].items()
        if key.endswith("-boundary")
    )
    text = f"""<section id="h200-levers-measurements">
<h2>H200 speed-lever verdicts</h2>
<p>The seven original jobs ({jobs}) completed. Canonical-input repair jobs
{repair_ids} complete the paired comparisons. Every original completed receipt,
including carried-decision widths 1 and 16 and booking-off width 1, is retained
unchanged. The selected fresh-process pairs agree within {100 * max_spread:.3f}%.
Earlier disagreeing pairs are excluded, never averaged.</p>
<p>The inventory audit passes {audit["comparisons"]} paired comparisons and
{audit["paired_geometry_records"]} geometry checks with no input fingerprint
mismatches. Position differences use the analytic plasma minor radius
({audit["minor_radius"]:.6g} m) and the locked 5e-4 position and 1e-4 flux-span
budgets; these qualify the lever changes against the historical reference,
whose analytic qualification is inherited. Maximum paired position error is
{position_error:.6g} of the minor radius and boundary-flux error is {flux_error:.6g}
of the span. Recorded terminal residuals are at most {residual:.6g}; all solved
lanes are certified at 1e-10. Receipts assert x64 on NVIDIA H200 NVL.
Compilation/cache-load
and synchronized warm execution walls are reported separately.</p>
<p>The historical carried branch returns a frozen reading and is retained for
attribution only. The carried-decision verdict compares the historical full
read with the current refreshed carried read. An identical-program arm pair is
refused. Every accepted pair has different lowered programs or executed-work
counts. The negative control admits distinct fingerprints and then refuses
identical programs; the added test also checks the refusal, and its deliberate
bypass is measured separately in the paired CPU suite job.</p>
<p>The scalar conic default changes from 32 to 128 cells. Factor carry remains
off by default. Shared scalar continuation remains the scalar-input default;
independent per-lane continuation is selected explicitly by array inputs.
Carried decisions, booking tiling, live candidates and the all-diverted carried
wall skip stay enabled. Normal-form tiles remain eight cells.</p>
<p>Each decision below uses the larger of 3% and the larger process spread.
Inside that margin the incumbent stays. Scalar rows compare 32 with 128; 64 is
included in the complete execution table. Width-16 full-read 128 versus 64 is
inside 3%, but 128 beats the incumbent 32 beyond the margin at both widths and
has the smallest measured wall. Each process is retained separately.
The width-16 lane-drive batch alternates 1e-5 and 1e-3 profile steps; its
width-1 case uses 1e-5. The width-1 counterpart of the mixed wall batch is
a single limited lane. A dash marks cases with no profile continuation.</p>
"""
    text += render_table(
        "verdicts",
        [
            "Lever",
            "Width",
            "Profile step",
            "Population",
            "Stage",
            "Off / tile 32 [ms], P1,P2",
            "On / tile 128 [ms], P1,P2",
            "On change",
            "Margin",
            "Verdict",
        ],
        verdicts,
    )
    text += f"""<figure id="h200-levers-walls-figure">
<img src="/nova/figures/sub-millisecond-forward-path/h200-levers/arm-walls.svg"
alt="Per-lever H200 walls at widths one and sixteen and a secant flattop trace."/>
<figcaption>First selected process per arm; the tables include both processes.
Read panels show carried reads, except the decision-free reference uses its
full read. Booking shows assembly; factor carry shows the median across warm
flattop slices 2–7 within one process; wall skip shows the all-diverted batch.
The final panel shows per-slice width-16 wall, including prediction; other
panels show whole-batch walls. Colours consistently identify batch width.
No disagreeing process pair contributes to a plotted point.</figcaption></figure>
<p>The width-16 secant flattop stream has warm per-slice medians over slices
2–7 of {warm[0]:.6f} and {warm[1]:.6f} ms in the two fresh processes.
The 1 ms target is not met. Slice 1 precedes the two-slice secant history and
is tabled separately. This headline was measured on revision
<code>{headline["records"][0]["revision"]}</code>, before changing the scalar
default; it is the recorded main-arm wall, not a post-change projection.</p>
"""
    headers = ["Lever", "Width", "Profile step", "Population", "Arm", "Stage"]
    text += render_table(
        "execute-walls", headers + ["Process 1 [ms]", "Process 2 [ms]", "Spread"], table
    )
    text += render_table(
        "compile-walls",
        headers + ["Process 1 compile/load [s]", "Process 2 compile/load [s]"],
        compile_table,
    )
    headline_rows = []
    for left, right in zip(
        headline["records"][0]["rows"], headline["records"][1]["rows"], strict=True
    ):
        if "per_slice_ms" in left:
            headline_rows.append(
                [
                    left["name"],
                    f"{left['per_slice_ms']:.6f}",
                    f"{right['per_slice_ms']:.6f}",
                ]
            )
    text += render_table(
        "headline",
        ["Slice", "Process 1 [ms/slice]", "Process 2 [ms/slice]"],
        headline_rows,
    )
    text += f"""<p>Durable inputs, receipts, lowered fingerprints and logs:
<code>{run}</code>. Inventory: <code>measurement-jobs.json</code>;
preserved original hashes: <code>retained-receipts.json</code>;
audit diagnostics: <code>precision-audit.json</code> and
<code>input-provenance-errors.json</code>. The successful inventory audit is
recorded in <code>audit-summary.json</code>.</p></section>"""
    fragment = (
        root / "docs/evidence/fragments/sub-millisecond-forward-path/h200-levers.html"
    )
    fragment.write_text(text)

    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    from scripts.prototypes.bank_solve_profile.plot import save_checked

    plt.style.use("data-ink")
    fig, axes = plt.subplots(2, 4, figsize=(14, 7.2), dpi=100)
    fig.subplots_adjust(
        left=0.08, right=0.985, bottom=0.12, top=0.89, wspace=0.57, hspace=0.65
    )
    names = [
        ("carried-decisions", "Carried decisions", "carried"),
        ("booking", "Booking", "assembly"),
        ("factor-carry", "Factor carry", None),
        ("scalar-tile", "Scalar tile", "carried"),
        ("lane-drive", "Lane drive", "mixed-drive"),
        ("live-candidates", "Live candidates", "carried"),
        ("wall-skip", "Wall skip", "carried"),
    ]
    for ax, (name, label, stage) in zip(axes.flat, names):
        for width, colour in [(1, "#929292"), (16, "#1f5d92")]:
            c = next(
                c
                for c in comparisons
                if c["name"] == name
                and c["width"] == width
                and c.get("step", 1e-5) == 1e-5
                and c.get("population", "diverted") == "diverted"
            )
            values = []
            labels = []
            for arm in arms(c):
                metrics = arm["metrics"]
                value = (
                    statistics.median(
                        v
                        for k, v in metrics.items()
                        if k.startswith("slice-") and k != "slice-1"
                    )
                    if stage is None
                    else metrics[
                        "full"
                        if name == "carried-decisions" and arm["arm"] == "off"
                        else stage
                    ]
                )
                values.append(value)
                labels.append(str(arm["arm"]))
            ax.plot(
                range(len(values)),
                values,
                "o-",
                color=colour,
                linewidth=2.6,
                label=f"Width {width}",
                zorder=3 if width == 16 else 2,
            )
            ax.set_xticks(range(len(values)), labels)
        ax.set_ylabel(label + "\nBatch wall [ms]", fontsize=13)
        ax.tick_params(labelsize=11)
        ax.yaxis.set_major_locator(MaxNLocator(3))
        ax.spines[["top", "right"]].set_visible(False)
        ax.margins(x=0.18, y=0.15)
    ax = axes.flat[-1]
    record = headline["records"][0]
    values = [r["per_slice_ms"] for r in record["rows"] if "per_slice_ms" in r]
    ax.plot(
        range(1, len(values) + 1), values, "o-", color="#1f5d92", linewidth=3, zorder=3
    )
    ax.set(
        xlabel="Stream slice",
        ylabel="Secant flattop\nWall [ms/slice]",
        xticks=[1, 4, 7],
    )
    ax.xaxis.label.set_fontsize(13)
    ax.yaxis.label.set_fontsize(13)
    ax.tick_params(labelsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.set_major_locator(MaxNLocator(3))
    fig.legend(
        *axes.flat[0].get_legend_handles_labels(),
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=13,
    )
    output = (
        root / "docs/figures/sub-millisecond-forward-path/h200-levers/arm-walls.svg"
    )
    for axis in axes.flat:
        lower, upper = axis.get_ylim()
        axis.set_yticks([tick for tick in axis.get_yticks() if lower <= tick <= upper])
    save_checked(fig, output)
    fig.savefig(run / "arm-walls.png", dpi=100)
    print(
        f"SUMMARY {len(table)} selected execution rows; "
        f"warm per-slice {warm[0]:.6f}, {warm[1]:.6f} ms; "
        "all verdicts adjudicated; scalar default 128"
    )


if __name__ == "__main__":
    main()
