"""Render each lane's accepted drive and table counted batch work."""

import argparse
from html import escape
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.prototypes.bank_solve_profile.plot import save_checked


def accepted_drives(record):
    current = np.ones(8)
    rows = [current.copy()]
    for row in record["rows"]:
        accepted = np.asarray(row["converged"]) & (
            np.asarray(row["residual"]) <= np.asarray(row["tolerance"])
        )
        if record["arm"] == "main":
            accepted[:] = np.all(accepted)
        current = np.where(accepted, row["drive"], current)
        rows.append(current.copy())
    np.testing.assert_allclose(current, record["target"], atol=1e-15, rtol=0)
    return np.asarray(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--fragment", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.run / "count-report.json").read_text())
    allocation = (args.run / "counts-allocation.log").read_text()
    assert allocation.splitlines()[-1] == "EXIT=0"
    suites = json.loads((args.run / "suite-receipts.json").read_text())
    assert all(
        value["completed"] and value["exit_status"] == 0
        for key, value in suites.items()
        if key != "shared-step-mutant"
    )
    mutant = suites["shared-step-mutant"]
    assert mutant["exit_status"] != 0 and mutant["failure_ids"]
    records = {
        arm: json.loads((args.run / f"counts-{arm}-measured.json").read_text())
        for arm in ("main", "head")
    }
    plt.style.use("data-ink")
    fig, axes = plt.subplots(2, 4, figsize=(14, 7.2), dpi=100)
    colours = {"main": "#929292", "head": "#1f5d92"}
    labels = {"main": "shared step", "head": "per-lane step"}
    paths = {arm: accepted_drives(record) for arm, record in records.items()}
    for lane, ax in enumerate(axes.flat):
        for arm in ("main", "head"):
            values = paths[arm][:, lane]
            ax.step(
                np.arange(len(values)),
                (values - 1.0) * 1000,
                where="post",
                color=colours[arm],
                lw=3.0 if arm == "head" else 2.6,
                linestyle="-" if arm == "head" else "--",
                label=labels[arm],
                zorder=3 if arm == "head" else 2,
            )
        ax.set_ylabel(f"lane {lane + 1} drive change [‰]", fontsize=13)
        ax.tick_params(labelsize=11)
        ax.set_xticks(
            np.arange(max(record["subsolves"] for record in records.values()) + 1)
        )
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(False)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=14,
        bbox_to_anchor=(0.5, 0.99),
    )
    fig.supxlabel("batch sub-solve", fontsize=16, y=0.025)
    fig.subplots_adjust(
        left=0.085, right=0.98, bottom=0.12, top=0.87, wspace=0.62, hspace=0.35
    )
    save_checked(fig, args.figure)
    plt.close(fig)
    count_rows = "".join(
        f"<tr><td>{arm}</td><td>{record['reads']}</td>"
        f"<td>{record['updates']}</td><td>{record['subsolves']}</td>"
        f"<td>{record['seconds']:.3f}</td></tr>"
        for arm, record in records.items()
    )
    precision_rows = "".join(
        f"<tr><td>{lane + 1}</td><td>{records['main']['residual'][lane]:.3e}</td>"
        f"<td>{records['head']['residual'][lane]:.3e}</td>"
        + "".join(
            f"<td>{report['precision'][key][lane]:.3e}</td>"
            for key in ("flux", "axis", "saddle", "boundary")
        )
        + "</tr>"
        for lane in range(8)
    )
    cpu_rows = "".join(
        f"<tr><td>{escape(key)}</td><td>{value['exit_status']}</td>"
        f"<td>{value['failure_count']}</td></tr>"
        for key, value in suites.items()
    )
    fragment = f"""<section data-reckon="fragment" id="bank-lane-drive-landing"
 data-plan="sub-millisecond-forward-path" data-node="bank-lane-drive">
<h2>Independent adaptive drive lanes</h2>
<p>Each array-drive lane retains its own accepted state, drive, adaptive step and
secant history. Rejected lanes retry from their incumbent; certified lanes keep
their target state and take zero further Newton updates. Scalar calls retain the
shared-step loop. A callable lane-profile selector is mapped inside both the
residual and Newton booking in <code>support_read.solve_batch</code>.</p>
<p>The main arm loads the shared-step driver at
<code>{records["main"]["revision"]}</code>; the head driver is
<code>{records["head"]["revision"]}</code>. Both use the head numerical solve and
its per-lane profile adapter, because the main solver cannot express mixed
profiles. Thus these counts isolate driver scheduling; they do not claim a
whole-source main-versus-head numerical comparison. The scalar main coordinate
parameterises both physical ramps. First four lanes advance by 1e-5 and last four
by 1e-3 per nominal slice, through seven slices from drive 1.0. Initial seeds are
the predictor's saved first eight lanes. Common input hash:
<code>{report["inputs"]["hash"]}</code>.</p>
<table id="bank-lane-drive-counts"><thead><tr><th>driver</th><th>batch reads</th>
<th>batch updates</th><th>sub-solves</th><th>batch wall [s]</th></tr></thead>
<tbody>{count_rows}</tbody></table>
<p>Each batch read or update counts the largest lane count, summed over sub-solves.
One Titan P100 job ran one warmup and one measured process per arm, at width 8.
Walls are Titan P100 walls and carry no speed verdict. Allocation log ends
<code>EXIT=0</code>; all eight lanes certify in both measured arms.</p>
<figure id="bank-lane-drive-trajectories"><img width="1400"
 src="/nova/figures/sub-millisecond-forward-path/bank-lane-drive/drive-trajectories.svg"
 alt="Accepted per-lane drive against batch sub-solve for both drivers">
<figcaption>Accepted drive relative to the initial amplitude, in parts per thousand.
Shared-step progress is dashed grey; independent lane progress is blue. The
horizontal portion shows a rejected attempt or a lane held at its certified
target. Each panel follows one saved lane.</figcaption></figure>
<table id="bank-lane-drive-precision"><thead><tr><th>lane</th>
<th>main fresh residual</th><th>head fresh residual</th><th>flux/span</th>
<th>axis/minor radius</th><th>X-point/minor radius</th>
<th>boundary/span</th></tr></thead>
<tbody>{precision_rows}</tbody></table>
<p>Every residual is at most 1e-10; every flux difference is at most 1e-8 of
span; axis and X-point displacement are at most 5e-4 of minor radius; boundary
flux difference is at most 1e-4 of span. The precision checker was also made to
refuse a synthetic violation of each of these four physical bounds.</p>
<table id="bank-lane-drive-cpu"><thead><tr><th>fresh-process file arm</th>
<th>exit</th><th>failed tests</th></tr></thead><tbody>{cpu_rows}</tbody></table>
<p>The paired Sirius files add zero failures. Sharing one sub-step again in a
scratch copy fails <code>test_stiff_lane_does_not_shrink_easy_lane</code> while
the head passes it. Scalar/uniform-lane equivalence, distinct targets, stationary
lanes and both Newton profile paths are covered in the new test file.</p>
<p>Receipts, full logs, module provenance and per-lane arrays are retained under
<code>{escape(str(args.run))}</code>. The shared saver checks canvas bounds and
text overlap. The SVG is rendered and inspected before delivery.</p>
</section>"""
    args.fragment.parent.mkdir(parents=True, exist_ok=True)
    args.fragment.write_text(fragment + "\n")
    print("SUMMARY fragment and checked SVG written", flush=True)


if __name__ == "__main__":
    main()
