"""Validate width receipts and draw the converged wall per slice."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator, ScalarFormatter
import numpy as np

from scripts.prototypes.bank_solve_profile.plot import save_checked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("figure", type=Path)
    args = parser.parse_args()
    rows = []
    cells = []
    digests = set()
    for width in (16, 32, 64, 128, 256):
        if (args.run / f"width-{width}-unfit.json").exists():
            missing = json.loads((args.run / f"width-{width}-unfit.json").read_text())
            assert width == 256
            assert (
                missing["memory_requirement"]["temporary_bytes"]
                > missing["physical_bytes_estimate"]
            )
            continue
        cold = json.loads((args.run / f"width-{width}-process-0.json").read_text())
        for process in (1, 2):
            stem = args.run / f"width-{width}-process-{process}"
            row = json.loads(stem.with_suffix(".json").read_text())
            assert stem.with_suffix(".log").read_text().splitlines()[-1] == "EXIT=0"
            assert cold["cache"][0]["outcome"] == "miss"
            assert row["cache"][0]["outcome"] == "hit"
            assert cold["cache"][0]["key"] == row["cache"][0]["key"]
            assert all(row["converged"]) and max(row["residual"]) <= 1e-10
            assert len(row["reads"]) == len(row["converged"]) == width
            for name, bound in (
                ("axis_ratio", 5e-4),
                ("saddle_ratio", 5e-4),
                ("boundary_ratio", 1e-4),
            ):
                values = row["precision"][name]
                assert len(values) == width and np.all(np.isfinite(values))
                assert max(values) <= bound
            digests.add(row["common_sha256"])
            row["cold_seconds"] = cold["compile_seconds"]
            row["per_slice_ms"] = row["wall_ms"] / width
            row["per_slice_read_ms"] = row["per_slice_ms"] / max(row["reads"])
            rows.append(row)
            cells.append(
                f"<tr><td>{width}</td><td>{process}</td>"
                f"<td>{min(row['reads'])}–{max(row['reads'])}</td>"
                f"<td>{min(row['steps'])}–{max(row['steps'])}</td>"
                f"<td>{row['wall_ms']:.3f}</td><td>{row['per_slice_ms']:.3f}</td>"
                f"<td>{row['per_slice_read_ms']:.3f}</td>"
                f"<td>{cold['compile_seconds']:.2f}</td>"
                f"<td>{row['compile_seconds']:.2f}</td>"
                f"<td>{row['executable_bytes'] / 1e6:.3f}</td>"
                f"<td>{row['memory']['peak_bytes_in_use'] / 2**30:.3f}</td></tr>"
            )
    assert len(digests) == 1
    common = args.run / "common-seeds.npz"
    assert hashlib.sha256(common.read_bytes()).hexdigest() == next(iter(digests))
    with np.load(common) as saved:
        assert len(np.unique(saved["scales"])) == 256
        assert len(np.unique(saved["state_0"], axis=0)) == 256
    (args.run / "summary.json").write_text(json.dumps(rows, indent=2))
    (args.run / "table.html").write_text("\n".join(cells))
    plt.style.use("data-ink")
    fig, ax = plt.subplots(figsize=(14, 7.2), dpi=100)
    for process, marker, style in ((1, "o", "-"), (2, "s", "--")):
        selected = [row for row in rows if row["process"] == process]
        ax.plot(
            [row["width"] for row in selected],
            [row["per_slice_ms"] for row in selected],
            color="#2e8b57",
            marker=marker,
            fillstyle="none",
            linestyle=style,
            linewidth=3,
            markersize=9,
            label=f"process {process}",
            zorder=3 + process,
        )
    ax.axhline(
        1, color="#666666", linestyle=":", linewidth=1.2, label="target: 1 ms", zorder=1
    )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    widths = sorted({row["width"] for row in rows})
    ax.set_xticks(widths)
    top = max(row["per_slice_ms"] for row in rows) * 1.4
    ax.set_yticks([value for value in (1, 2, 4, 8, 16, 32, 64) if value <= top])
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlim(13, max(widths) * 1.25)
    ax.set_ylim(0.8, top)
    ax.set_xlabel("Batch width [slices]")
    ax.set_ylabel("Converged solve wall per slice [ms]")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(False)
    fig.legend(loc="upper center", ncol=3, frameon=False)
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.14, top=0.85)
    save_checked(fig, args.figure)
    args.figure.write_text(
        "\n".join(line.rstrip() for line in args.figure.read_text().splitlines()) + "\n"
    )
    fig.savefig(args.run / "width-wall.png", dpi=100)
    best = min(rows, key=lambda row: row["per_slice_ms"])
    print(
        "REPORT "
        + json.dumps(
            dict(
                best_width=best["width"],
                best_ms=best["per_slice_ms"],
                max_residual=max(max(row["residual"]) for row in rows),
                precision={
                    name: max(max(row["precision"][name]) for row in rows)
                    for name in rows[0]["precision"]
                },
                samples=len(rows),
                common_sha256=next(iter(digests)),
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
