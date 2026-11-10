"""Validate paired predictor receipts and draw reads against slice index."""

import argparse
from html import escape
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.prototypes.bank_solve_profile.plot import save_checked

ARMS = ("plain", "secant", "tangent")
STEPS = (1e-5, 1e-3)
BOUNDS = dict(flux=1e-8, axis=5e-4, saddle=5e-4, boundary=1e-4)


def precision(reference, candidate, key, radius):
    span = np.abs(reference[key + "-span"])
    assert np.all(np.isfinite(span)) and np.all(span > 0)
    position = (
        np.linalg.norm(
            candidate[key + "-position"] - reference[key + "-position"], axis=-1
        )
        / radius
    )
    return dict(
        flux=np.max(
            np.abs(candidate[key + "-flux"] - reference[key + "-flux"]), axis=-1
        )
        / span,
        axis=position[:, 0],
        saddle=position[:, 1],
        boundary=np.abs(candidate[key + "-boundary"] - reference[key + "-boundary"])
        / span,
    )


def require_budget(values):
    for name, bound in BOUNDS.items():
        assert np.all(np.isfinite(values[name])) and np.max(values[name]) <= bound, (
            name,
            values[name],
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--fragment", type=Path, required=True)
    args = parser.parse_args()
    data, arrays = {}, {}
    for arm in ARMS:
        for repeat in ("first", "second"):
            stem = args.run / f"runtime-{arm}-{repeat}"
            data[arm, repeat] = json.loads(stem.with_suffix(".json").read_text())
            arrays[arm, repeat] = dict(
                np.load(stem.with_suffix(".npz"), allow_pickle=False)
            )
            assert len(data[arm, repeat]) == 14
            for row in data[arm, repeat]:
                assert len(row["residual"]) == 8 and all(row["converged"])
                assert np.max(row["residual"]) <= 1e-10
        for first, second in zip(data[arm, "first"], data[arm, "second"], strict=True):
            for name in (
                "step",
                "slice",
                "reads",
                "updates",
            ):
                np.testing.assert_array_equal(first[name], second[name])
        for row in data[arm, "first"]:
            key = f"{row['step']:g}-{row['slice']}"
            require_budget(
                precision(
                    arrays[arm, "first"],
                    arrays[arm, "second"],
                    key,
                    row["minor_radius"],
                )
            )
    maxima = {name: 0.0 for name in BOUNDS}
    for arm in ARMS:
        for repeat in ("first", "second"):
            for row in data[arm, repeat]:
                key = f"{row['step']:g}-{row['slice']}"
                values = precision(
                    arrays["plain", repeat],
                    arrays[arm, repeat],
                    key,
                    row["minor_radius"],
                )
                require_budget(values)
                for name in maxima:
                    maxima[name] = max(maxima[name], float(np.max(values[name])))
    reference = arrays["plain", "first"]
    key = "1e-05-1"
    radius = data["plain", "first"][0]["minor_radius"]
    refused = []
    for quantity in BOUNDS:
        mutant = {name: value.copy() for name, value in reference.items()}
        if quantity == "flux":
            mutant[key + "-flux"][0, 0] += 2e-8 * abs(reference[key + "-span"][0])
        elif quantity == "boundary":
            mutant[key + "-boundary"][0] += 2e-4 * abs(reference[key + "-span"][0])
        else:
            mutant[key + "-position"][0, 0 if quantity == "axis" else 1, 0] += (
                1e-3 * radius
            )
        try:
            require_budget(precision(reference, mutant, key, radius))
        except AssertionError:
            refused.append(quantity)
    assert set(refused) == set(BOUNDS)
    plt.style.use("data-ink")
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.2), dpi=100, sharey=True)
    styles = dict(
        plain=("#929292", "--", "s"),
        secant=("#3366cc", "-", "o"),
        tangent=("#222222", ":", "^"),
    )
    for ax, step in zip(axes, STEPS, strict=True):
        for arm in ARMS:
            rows = [row for row in data[arm, "first"] if row["step"] == step]
            colour, line, marker = styles[arm]
            ax.plot(
                [
                    row["slice"] + {"plain": -0.08, "secant": 0, "tangent": 0.08}[arm]
                    for row in rows
                ],
                [max(row["reads"]) + int(arm == "tangent") for row in rows],
                color=colour,
                linestyle=line,
                marker=marker,
                fillstyle="none",
                linewidth=3 if arm == "secant" else 2.6,
                markersize=9,
                label=arm,
                zorder=4 if arm == "secant" else 2,
            )
        ax.set_xlabel("slice index")
        ax.set_xticks(range(1, 8))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(False)
        ax.text(
            0.5,
            1.04,
            f"drive step {step:g}",
            transform=ax.transAxes,
            ha="center",
            fontsize=14,
        )
    top = max(
        max(row["reads"]) + int(arm == "tangent")
        for arm in ARMS
        for row in data[arm, "first"]
    )
    axes[0].set_yticks(range(1, top + 1))
    axes[0].set_ylim(0.7, top + 0.8)
    axes[0].set_ylabel("fresh reads per slice [count]")
    fig.legend(
        *axes[0].get_legend_handles_labels(), ncol=3, loc="upper center", frameon=False
    )
    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.14, top=0.78, wspace=0.16)
    save_checked(fig, args.figure)
    fig.savefig(args.run / "reads-per-slice-preview.png", dpi=100)
    table = []
    summary = []
    for step in STEPS:
        for arm in ARMS:
            rows = [row for row in data[arm, "first"] if row["step"] == step]
            walls = [
                row["per_slice_ms"]
                for repeat in ("first", "second")
                for row in data[arm, repeat]
                if row["step"] == step
            ]
            summary.append(
                dict(
                    arm=arm,
                    step=step,
                    reads=sorted(
                        {
                            value + int(arm == "tangent")
                            for row in rows
                            for value in row["reads"]
                        }
                    ),
                    solve_reads=sorted(
                        {value for row in rows for value in row["reads"]}
                    ),
                    updates=sorted({value for row in rows for value in row["updates"]}),
                    seed_residual=max(max(row["seed_residual"]) for row in rows),
                    terminal_residual=max(max(row["residual"]) for row in rows),
                    per_slice_ms=float(np.mean(walls)),
                )
            )
            for row in rows:
                other = next(
                    r
                    for r in data[arm, "second"]
                    if r["step"] == step and r["slice"] == row["slice"]
                )
                table.append(
                    f"<tr><td>{arm}</td><td>{step:g}</td><td>{row['slice']}</td>"
                    f"<td>{max(row['seed_residual']):.3e}</td>"
                    f"<td>{max(row['reads']) + int(arm == 'tangent')}</td>"
                    f"<td>{max(row['reads'])}</td>"
                    f"<td>{max(row['updates'])}</td><td>{max(row['residual']):.3e}</td>"
                    f"<td>{row['per_slice_ms']:.3f}, {other['per_slice_ms']:.3f}</td>"
                    f"<td>{row['prediction_ms'] / 8:.3f}, "
                    f"{other['prediction_ms'] / 8:.3f}</td></tr>"
                )
    report = dict(
        summary=summary,
        precision_maxima=maxima,
        precision_mutants_refused=refused,
        processes_agree_within_budget=True,
        wall_label="titan P100, width 8",
    )
    (args.run / "predictor-report.json").write_text(json.dumps(report, indent=2))
    fragment = f"""<section data-reckon="fragment" id="bank-warm-predictor-fragment"
data-plan="sub-millisecond-forward-path" data-node="bank-warm-predictor">
<h2>Warm drive predictor: titan P100, width 8</h2>
<p>Plain, secant and assembled-Jacobian tangent seeds use the first eight lanes of
the saved warm-start streams, at drive steps 1e-5 and 1e-3 over slices 1–7. Each arm
has one cache warmup followed by two fresh interpreters. Read and update counts agree
exactly; both processes certify every lane and their terminal flux/geometry agree
within the locked precision budget. Slice
1 has only one converged history state, so secant uses the plain warm-start
fallback.</p>
<p>The secant extrapolates two accepted states in drive. The tangent assembles the
bank Jacobian at the converged state and uses its equilibrated block-GMRES solve
against the drive derivative. Its own read, assembly and solve are included in its
predictor cost. Neither predictor changes terminal certification: every lane has a
fresh residual at most 1e-10.</p>
<figure id="bank-warm-predictor-reads"><img
src="/nova/figures/sub-millisecond-forward-path/bank-warm-predictor/reads-per-slice.svg"
alt="Batch reads against slice index for plain, secant and tangent seeds at two drive
steps" width="1400"><figcaption>Reads per slice, titan P100, width 8. Each point is
the largest lane count in its batch; both fresh processes agree exactly. Secant is
blue, plain grey and tangent dotted near-black. Predictor work is tabled separately
and total reads include the tangent predictor’s one fresh read before the
solve. Small horizontal offsets separate coincident counts.</figcaption></figure>
<table id="bank-warm-predictor-table"><thead><tr><th>arm</th><th>drive
step</th><th>slice</th><th>seed fresh residual, max</th>
<th>total fresh reads, max</th><th>solve reads, max</th><th>updates, max</th>
<th>terminal fresh residual, max</th><th>total wall /
slice, two processes [ms]; titan P100, width 8</th><th>predictor wall / slice, two
processes [ms]; titan P100, width
8</th></tr></thead><tbody>{"".join(table)}</tbody></table>
<p>Every lane in both measured processes is checked against its same-process plain
arm: flux ≤1e-8 of its span, axis and X-point ≤5e-4 of the minor radius, boundary
flux ≤1e-4 of span. Observed maxima: {escape(json.dumps(maxima))}. Synthetic
over-budget changes in each of those four quantities were refused by the same
check.</p>
<p>Width 16 failed on titan P100: jit_one requested 15,120,805,120 bytes (14.08 GiB).
Width 8 is the memory fallback. These walls are titan P100, width 8;
the H200 width-16 headline remains pending a free card.</p>
</section>"""
    args.fragment.write_text(fragment + "\n")
    print("SUMMARY " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
