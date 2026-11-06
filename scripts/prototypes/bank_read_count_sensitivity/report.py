"""Tabulate each observed lane and locate its first discrete path split."""

import argparse
from html import escape
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.prototypes.bank_solve_profile.plot import save_checked

DECISIONS = ("cut", "normal", "fragments")


def updates(receipt):
    rows = []
    for event in receipt["events"]:
        kind = event["kind"]
        if kind == "begin":
            row = {"begin": event, "trials": []}
        elif kind == "trial_norm":
            trial = {"norm": event}
        elif kind == "trial_decisions":
            trial["decisions"] = event
        elif kind == "trial_rule":
            trial["rule"] = event
            row["trials"].append(trial)
        else:
            row[kind] = event
            if kind == "after":
                rows.append(row)
    assert len(rows) == max(receipt["observed"]["steps"])
    for lane in range(16):
        assert len(rows[0]["before"]["cut"]) == 16
        assert rows[0]["before"]["cut_count"][lane] > 0
        assert rows[0]["before"]["normal_count"][lane] > 0
        assert rows[0]["before"]["fragments_count"][lane] > 0
    return rows


def first_split(left, right):
    for update, (a, b) in enumerate(zip(left, right), 1):
        for lane in range(16):
            reasons = []
            for key in DECISIONS:
                if a["after"][key][lane] != b["after"][key][lane]:
                    reasons.append(key)
            if a["end"]["done"][lane] != b["end"]["done"][lane]:
                reasons.append("tolerance/acceptance")
            if a["search"]["scale"][lane] != b["search"]["scale"][lane]:
                reasons.append("selected scale")
            for x, y in zip(a["trials"], b["trials"]):
                for key in DECISIONS:
                    if x["decisions"][key][lane] != y["decisions"][key][lane]:
                        reasons.append("trial " + key)
                if x["rule"]["pending_after"][lane] != y["rule"]["pending_after"][lane]:
                    reasons.append("trial acceptance")
            if reasons:
                return dict(
                    update=update,
                    lane=lane,
                    reasons=sorted(set(reasons)),
                    base_residual=a["end"]["residual"][lane],
                    head_residual=b["end"]["residual"][lane],
                    absolute_gap=abs(
                        a["end"]["residual"][lane] - b["end"]["residual"][lane]
                    ),
                )
    raise ValueError("No differing decision observed")


def table(headers, rows):
    return (
        "<table><thead><tr>"
        + "".join(f"<th>{escape(x)}</th>" for x in headers)
        + "</tr></thead><tbody>"
        + "".join(
            "<tr>" + "".join(f"<td>{escape(str(x))}</td>" for x in row) + "</tr>\n"
            for row in rows
        )
        + "</tbody></table>\n"
    )


def lane_rows(label, rows, lanes):
    result = []
    for lane in lanes:
        for update, row in enumerate(rows, 1):
            trials = []
            for trial in row["trials"]:
                if trial["rule"]["pending"][lane]:
                    state = (
                        "rejected"
                        if trial["rule"]["pending_after"][lane]
                        else "accepted"
                    )
                    changed = [
                        key
                        for key in DECISIONS
                        if row["before"][key][lane] != trial["decisions"][key][lane]
                    ]
                    trials.append(
                        f"{trial['rule']['scale'][lane]:g}: "
                        f"{trial['norm']['norm'][lane]:.16e} ({state}; "
                        f"sets: {', '.join(changed) or 'unchanged'})"
                    )
            changes = [
                key
                for key in DECISIONS
                if row["before"][key][lane] != row["after"][key][lane]
            ]
            result.append(
                [
                    label,
                    lane,
                    update,
                    f"{row['begin']['residual'][lane]:.16e}",
                    f"{row['end']['residual'][lane]:.16e}",
                    row["search"]["scale"][lane],
                    "; ".join(trials) or "companion only",
                    row["end"]["refactors"][lane],
                    ", ".join(changes) or "none",
                    row["end"]["releases"][lane],
                    row["end"]["done"][lane],
                    row["end"]["reads"][lane],
                ]
            )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("fragment", type=Path)
    parser.add_argument("figure", type=Path)
    args = parser.parse_args()
    receipts = {
        (arm, process): json.loads((args.run / f"{arm}-{process}.json").read_text())
        for arm in ("base", "head")
        for process in (0, 1, 2)
    }
    rows = {key: updates(value) for key, value in receipts.items()}
    for receipt in receipts.values():
        assert receipt["same_path"] and receipt["same_reads"]
        assert receipt["largest_moment_gap"] == 0
        assert all(receipt["plain"]["converged"])
        assert max(receipt["plain"]["residual"]) <= 1e-10
    assert (
        receipts["base", 1]["plain"]["reads"] != receipts["head", 1]["plain"]["reads"]
    )
    split = first_split(rows["base", 1], rows["head", 1])
    for arm in ("base", "head"):
        assert len({receipts[arm, p]["initial_hash"] for p in (0, 1, 2)}) == 1
        assert receipts[arm, 0]["events"] == receipts[arm, 1]["events"]
        assert receipts[arm, 1]["events"] == receipts[arm, 2]["events"]
        for field in receipts[arm, 1]["plain"]:
            assert np.array_equal(
                receipts[arm, 1]["plain"][field],
                receipts[arm, 2]["plain"][field],
                equal_nan=True,
            )
        for process in (1, 2):
            log = (args.run / f"{arm}-{process}.log").read_text()
            for name in ("solve", "observed"):
                assert f"Persistent compilation cache hit for 'jit_{name}'" in log
    summary = dict(
        first_split=split,
        same_initial=len({r["initial_hash"] for r in receipts.values()}) == 1,
        rows=[
            dict(
                arm=arm,
                process=p,
                reads=r["plain"]["reads"],
                observed_reads=r["observed"]["reads"],
                same_path=r["same_path"],
                median_ms=float(np.median(r["walls_ms"])),
                ms_per_read=float(np.median(r["walls_ms"])) / max(r["plain"]["reads"]),
            )
            for (arm, p), r in receipts.items()
        ],
    )
    for field in ("initial_moments", "terminal_moments"):
        left = np.asarray(receipts["base", 2][field])
        right = np.asarray(receipts["head", 2][field])
        gap = np.max(np.abs(left - right), axis=(1, 2))
        summary[field] = dict(
            maximum_absolute=gap.tolist(),
            maximum_relative=(gap / np.max(np.abs(left), axis=(1, 2))).tolist(),
        )
    summary["initial_residual_max_gap"] = float(
        np.max(
            np.abs(
                np.asarray(rows["base", 1][0]["begin"]["residual"])
                - np.asarray(rows["head", 1][0]["begin"]["residual"])
            )
        )
    )
    (args.run / "summary.json").write_text(json.dumps(summary, indent=2))
    print("SUMMARY", json.dumps(summary), flush=True)
    prefix = args.fragment.stem
    html = [
        f'<section id="{prefix}-landing">',
        "<h2>Read-count sensitivity to numerical perturbations</h2>",
    ]
    html.append(
        f"<p>First observed discrete split: update {split['update']}, "
        f"zero-based lane {split['lane']}; "
        f"{escape(', '.join(split['reasons']))}. "
        f"Accepted fresh residuals differ by {split['absolute_gap']:.16e}.</p>"
    )
    html.append(
        table(
            [
                "Arm",
                "Process",
                "Reads in lane order",
                "Observed reads",
                "Median solve [ms]",
                "Wall/read [ms]",
                "Identical observed path",
            ],
            [
                [
                    r["arm"],
                    "warm-up" if r["process"] == 0 else r["process"],
                    r["reads"],
                    r["observed_reads"],
                    f"{r['median_ms']:.6f}",
                    f"{r['ms_per_read']:.6f}",
                    r["same_path"],
                ]
                for r in summary["rows"]
            ],
        )
    )
    headers = [
        "Arm/process",
        "Lane",
        "Update",
        "Fresh before",
        "Fresh after",
        "Selected scale",
        "Pending trials: scale, fresh norm, decision",
        "Active refactor",
        "Decision changes",
        "Lock releases",
        "Done",
        "Cumulative reads",
    ]
    html.append(f'<h3 id="{prefix}-first-lane">First differing lane</h3>')
    html.append(
        table(
            headers,
            sum(
                [
                    lane_rows(arm + "/1", rows[arm, 1], [split["lane"]])
                    for arm in ("base", "head")
                ],
                [],
            ),
        )
    )
    html.append(
        f'<figure id="{prefix}-history"><img '
        f'src="/nova/figures/sub-millisecond-forward-path/{prefix}/{args.figure.name}" '
        'alt="Fresh residual against update for both revisions in the first '
        'differing lane"><figcaption>'
        f"Zero-based lane {split['lane']}, both warm-cache processes per revision "
        "have identical observed histories. The dotted line is the unchanged "
        "1e-10 acceptance tolerance. Values after a lane has converged include "
        "companion reads. Logarithmic vertical axis.</figcaption></figure>"
    )
    html.append(
        f'<details id="{prefix}-all-lanes"><summary>All lanes and updates, '
        "both fresh processes</summary>"
    )
    html.append(
        table(
            headers,
            sum(
                [
                    lane_rows(f"{arm}/{p}", values, range(16))
                    for (arm, p), values in rows.items()
                    if p
                ],
                [],
            ),
        )
    )
    html.extend(["</details>", "</section>"])
    args.fragment.write_text("\n".join(html))
    plt.style.use("data-ink")
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.2), dpi=100)
    for arm, colour, dash, marker in (
        ("base", "#929292", "--", "o"),
        ("head", "#3366cc", "-", "s"),
    ):
        values = rows[arm, 1]
        path = [values[0]["begin"]["residual"][split["lane"]]] + [
            row["end"]["residual"][split["lane"]] for row in values
        ]
        for ax in axes:
            ax.plot(
                range(len(path)),
                path,
                color=colour,
                linestyle=dash,
                marker=marker,
                linewidth=3 if arm == "head" else 2.6,
                label=arm,
                zorder=3 if arm == "head" else 2,
            )
    for ax in axes:
        ax.axhline(
            1e-10,
            color="#666666",
            linestyle=":",
            linewidth=1.2,
            label="tolerance",
            zorder=1,
        )
        ax.set_yscale("log")
        ax.set_xlabel("Batch update")
        ax.grid(False)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Fresh residual")
    axes[0].set_xticks(range(max(map(len, rows.values())) + 1))
    axes[1].set_xlim(2.85, 6.15)
    axes[1].set_ylim(4e-11, 2e-10)
    axes[1].set_xticks([3, 4, 5, 6])
    axes[1].set_yticks(
        [5e-11, 1e-10, 2e-10],
        [r"$5\times10^{-11}$", r"$10^{-10}$", r"$2\times10^{-10}$"],
    )
    axes[1].minorticks_off()
    fig.legend(
        *axes[0].get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False
    )
    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.14, top=0.86, wspace=0.32)
    save_checked(fig, args.figure)
    args.figure.write_text(
        "\n".join(line.rstrip() for line in args.figure.read_text().splitlines()) + "\n"
    )
    fig.savefig(args.run / "residual-history.png", dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    main()
