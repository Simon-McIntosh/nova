"""Report the paired native slice walls and inspectable decision margin."""

import argparse
import hashlib
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("figure", type=Path)
    parser.add_argument("fragment", type=Path)
    args = parser.parse_args()
    summary = json.loads((args.run / "summary.json").read_text())
    assert summary["complete"] and summary["inert_control"].startswith("inert lever")
    from scripts.prototypes.geometry_reuse.run import compare

    paired = {}
    for entry in summary["arms"]:
        processes = summary["decisions"][str(entry["width"])]["processes"]
        if entry["process"] not in processes:
            continue
        receipt = json.loads(Path(entry["receipt"]).read_text())
        assert receipt["complete"] and receipt["revision"] == summary["revision"]
        assert receipt["device"].find("H200") >= 0 and receipt["x64"]
        assert max(receipt["fresh_residual"]) <= 1e-10
        assert max(receipt["precision"]["axis"]) <= 5e-4
        assert max(receipt["precision"]["saddle"]) <= 5e-4
        assert max(receipt["precision"]["boundary"]) <= 1e-4
        assert (
            hashlib.sha256(Path(receipt["inputs_path"]).read_bytes()).hexdigest()
            == receipt["inputs_sha256"]
        )
        paired.setdefault(entry["width"], {}).setdefault(entry["arm"], []).append(
            receipt
        )
    assert set(paired) == {1, 16, 64}
    decisions = {
        width: compare(arms["off"], arms["on"]) for width, arms in paired.items()
    }
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter, NullLocator
    from scripts.prototypes.bank_solve_profile.plot import save_checked

    plt.style.use("data-ink")
    figure, axis = plt.subplots(figsize=(14, 7.2), dpi=100)
    widths = sorted(paired)
    for arm, colour, style, marker in (
        ("off", "#666666", "--", "s"),
        ("on", "#1f5d92", "-", "o"),
    ):
        values = [
            [row["per_slice_ms"] for row in paired[width][arm]] for width in widths
        ]
        means = [sum(row) / len(row) for row in values]
        axis.errorbar(
            widths,
            means,
            yerr=[
                [mean - min(row) for mean, row in zip(means, values, strict=True)],
                [max(row) - mean for mean, row in zip(means, values, strict=True)],
            ],
            color=colour,
            linestyle=style,
            marker=marker,
            markersize=8,
            linewidth=3 if arm == "on" else 2.6,
            capsize=5,
            label=f"geometry reuse {arm}",
            zorder=3 if arm == "on" else 2,
        )
    axis.set_xscale("log", base=2)
    axis.set_xticks(widths, [str(width) for width in widths])
    axis.set_yscale("log")
    axis.yaxis.set_major_formatter(ScalarFormatter())
    axis.yaxis.set_minor_locator(NullLocator())
    all_walls = [
        row["per_slice_ms"]
        for arms in paired.values()
        for rows in arms.values()
        for row in rows
    ]
    lower, upper = min(all_walls) * 0.8, max(all_walls) * 1.25
    axis.set_yticks(
        [
            multiple * 10.0**power
            for power in range(
                math.floor(math.log10(lower)), math.ceil(math.log10(upper)) + 1
            )
            for multiple in (1, 2, 5)
            if lower <= multiple * 10.0**power <= upper
        ]
    )
    axis.set_ylim(lower, upper)
    axis.set_xlim(0.85, 75)
    axis.set_xlabel("Concurrent independent pulses")
    axis.set_ylabel("Throughput cost [ms/slice]")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(False)
    figure.legend(
        *axis.get_legend_handles_labels(), loc="upper center", ncol=2, frameon=False
    )
    figure.subplots_adjust(left=0.12, right=0.96, bottom=0.16, top=0.86)
    save_checked(figure, args.figure)
    figure.savefig(args.run / "figure-review.png", dpi=100)
    plt.close(figure)

    rows = []
    for width in widths:
        decision = decisions[width]
        cells = [str(width)]
        for arm in ("off", "on"):
            cells += [
                " / ".join(f"{row['per_slice_ms']:.4f}" for row in paired[width][arm])
            ]
        cells += [
            f"{100 * decision['improvement']:.2f}%",
            f"{100 * decision['margin']:.2f}%",
            decision["verdict"],
        ]
        rows.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")
    revision = summary["revision"]
    image = (
        "/nova/figures/sub-millisecond-forward-path/geometry-reuse/per-slice-wall.svg"
    )
    args.fragment.parent.mkdir(parents=True, exist_ok=True)
    args.fragment.write_text(
        f'''<section id="geometry-reuse-landing">
<h2>Clipped geometry carried across sequential slices</h2>
<p>The existing topology receipt carries conic strips, fragment validity and
capacity, saddle fragments and exact clipping keys. Unchanged records are
retained; changed cells are compacted into device tiles. A conservative box
bound gives wholly positive or negative cells a constant clipping representation.
Connected qualification still runs in full before terminal acceptance.
The preceding slice’s support enters <code>solve_batch(initial_support=...)</code>
through continuation’s existing <code>solve_options</code>.</p>
<p>Measured revision <code>{revision}</code>.
One H200 job pairs off/on arms at widths 1, 16 and 64, with one warming process
and two fresh measured processes per arm. Each uses the committed slice-profile
driver’s three sequential drive steps and eleven synchronized warm samples.
Width counts independent pulses; slices of each pulse remain sequential.</p>
<table><thead><tr><th>Width</th><th>Off processes [ms/slice]</th>
<th>On processes [ms/slice]</th><th>Reduction</th><th>Decision margin</th>
<th>Verdict</th></tr></thead><tbody>
{"".join(rows)}
</tbody></table>
<p>The decision margin is the greater of 3% and either arm’s process spread;
every reported pair agrees within 5%. A positive reduction favours reuse.
Every lane passes the independent fresh residual bound of 1e-10, axis and saddle
bounds of 5e-4 minor radii, and boundary-flux bound of 1e-4 of the flux span.
Off/on lowered program digests differ; comparing an arm with itself is refused
as an inert lever.</p>
<figure id="geometry-reuse-wall"><img src="{image}"
alt="Paired geometry reuse walls across independent pulse widths">
<figcaption>Mean of two fresh-process medians; whiskers span the two medians.
Both axes are logarithmic. Width 1 is causal slice latency; larger widths show
aggregate throughput cost, not lower latency within one pulse. Each arm uses
the same input stream and revision. Compilation is recorded separately and
excluded from these warm walls.</figcaption></figure>
<p>Durable per-arm receipts, compilation walls, lowered digests and gate logs:
<code>{args.run}</code>. The reproduction at the pinned base rebuilt all 30 cells
on each of two identical support refreshes. The selective-reuse check observes
no conic clip execution on the unchanged support and observes execution when
a cut moves. The moved-support mutation forces all clipping keys to compare
unchanged and must fail the head-passing tests; its log and the CPU file
receipts are named in the worker manifest.</p>
</section>
'''
    )
    print("SUMMARY", json.dumps(decisions, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
