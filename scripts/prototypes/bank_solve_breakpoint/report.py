"""Validate paired precision and plot the actual restart residual receipts."""

import argparse
from html import escape
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nova.media.ink import InkStyle
from scripts.prototypes.bank_solve_read.plot import save_checked


def restart(log):
    for line in log.read_text().splitlines():
        if line.startswith("RESULT "):
            row = json.loads(line.partition(" ")[2])
            if row["phase"] == "restart":
                return row
    raise ValueError(f"no completed restart receipt: {log}")


def precision(before, after):
    gaps = {}
    for key, unit in (
        ("flux", before["span"]),
        ("axis", before["minor_radius"]),
        ("saddle", before["minor_radius"]),
        ("boundary_flux", before["span"]),
    ):
        difference = np.asarray(after[key]) - np.asarray(before[key])
        if key in ("axis", "saddle"):
            difference = np.linalg.norm(difference, axis=-1)
        if key == "flux":
            difference = np.max(np.abs(difference), axis=-1)
        gaps[key] = (np.abs(difference) / unit).tolist()
    for arm, row in [("base", before), ("head", after)]:
        roots = np.asarray(row["oracle_roots"])
        for key, point in [("axis", 0), ("saddle", 1)]:
            gaps[f"{arm}_oracle_{key}"] = (
                np.linalg.norm(np.asarray(row[key]) - roots[point], axis=-1)
                / row["minor_radius"]
            ).tolist()
        # The first-lane absolute oracle error plus its companion difference
        # bounds every lane without cancellation between signed errors.
        gaps[f"{arm}_oracle_boundary"] = (
            row["oracle"]["boundary_ratio"]
            + np.abs(np.asarray(row["boundary_flux"]) - row["boundary_flux"][0])
            / row["span"]
        ).tolist()
    limits = {"flux": 1e-8, "axis": 5e-4, "saddle": 5e-4, "boundary_flux": 1e-4}
    limits.update(
        {
            key: 1e-4 if key.endswith("boundary") else 5e-4
            for key in gaps
            if "oracle" in key
        }
    )
    passed = {
        key: bool(np.all(np.asarray(value) <= limits[key]))
        for key, value in gaps.items()
    }
    for arm, row in [("base", before), ("head", after)]:
        passed[arm + "_fresh_residual"] = (
            all(row["converged"]) and max(row["residual"]) <= 1e-10
        )
    return dict(
        lanes=after["lanes"],
        call=after["call"],
        gaps=gaps,
        limits=limits,
        passed=passed,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()
    arms = {
        name: json.loads((args.run / f"{name}-solve.json").read_text())
        for name in ("base", "head")
    }
    comparisons = []
    for before, after in zip(arms["base"]["rows"], arms["head"]["rows"], strict=True):
        assert (before["lanes"], before["call"]) == (after["lanes"], after["call"])
        comparisons.append(precision(before, after))
    (args.run / "precision.json").write_text(json.dumps(comparisons, indent=2))
    print("PRECISION", json.dumps(comparisons), flush=True)
    rows = []
    for name, receipt in arms.items():
        timings = json.loads((args.run / f"{name}-timings.json").read_text())
        for row in receipt["rows"]:
            if row["call"] != 1:
                continue
            values = [
                name,
                str(row["lanes"]),
                ", ".join(map(str, row["steps"])),
                ", ".join(map(str, row["reads"])),
                f"{timings[str(row['lanes']) + ':assembly']:.3f}",
                f"{row['wall_ms']:.3f}",
                f"{row['wall_ms'] / max(row['steps']):.3f}",
                f"{row['per_slice_ms']:.3f}",
                f"{max(row['residual']):.6e}",
            ]
            rows.append(
                "<tr>" + "".join("<td>" + escape(v) + "</td>" for v in values) + "</tr>"
            )
    (args.run / "measurement-table.html").write_text(
        '<table id="bank-solve-breakpoint-measurement"><thead><tr>'
        "<th>Arm</th><th>Slices</th><th>Updates, lane order</th>"
        "<th>Fresh reads, lane order</th><th>Jacobian [ms/batch]</th>"
        "<th>Whole solve [ms]</th><th>Per iterate [ms]</th>"
        "<th>Converged [ms/slice]</th><th>Largest fresh residual</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )
    plt.style.use("data-ink")
    fig, ax = plt.subplots(figsize=(14, 7.2), dpi=100)
    histories = {
        name: restart(args.run / f"cpu-{name}-test_bank_solve_cycle.log")
        for name in arms
    }
    for name, colour, dash, layer in [
        ("base", "#929292", "--", 2),
        ("head", InkStyle().flux_color, "-", 3),
    ]:
        row = histories[name]
        count = row["steps"][0]
        ax.plot(
            np.arange(count + 1),
            row["residual_path"][0][: count + 1],
            color=colour,
            linestyle=dash,
            marker="o",
            linewidth=3 if name == "head" else 2.6,
            label=name,
            zorder=layer,
        )
    ax.axhline(
        1e-10,
        color="#666666",
        linestyle=":",
        linewidth=1.2,
        label="acceptance limit",
        zorder=1,
    )
    ax.set_yscale("log")
    ax.set_xlabel("Restart update")
    ax.set_ylabel("Fresh residual")
    ax.set_xticks(np.arange(9))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(False)
    fig.legend(loc="upper center", ncol=3, frameon=False)
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.14, top=0.86)
    save_checked(fig, args.figure)
    fig.savefig(args.run / "residual-history.png", dpi=100)
    plt.close(fig)
    failed = [
        (row["lanes"], row["call"], key)
        for row in comparisons
        for key, value in row["passed"].items()
        if not value
    ]
    print("PRECISION_FAILURES", json.dumps(failed), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
