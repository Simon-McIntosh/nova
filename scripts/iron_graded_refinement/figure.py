"""Draw shell error against cell count for the uniform and graded meshes.

    python scripts/iron_graded_refinement/figure.py --run <run-dir> --output <svg>

Each mesh is one point at its cell count, against the worst and best of the
three probes of the analytic mu_r = 1000 closed shell. The uniform meshes are
the dashed reference, the graded meshes the feature colour, and the 0.5 percent
accuracy target a dotted limit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as ticker  # noqa: E402
import numpy as np  # noqa: E402

from scripts.edge_sheet.figure import (  # noqa: E402
    HEIGHT_INCHES,
    WIDTH_INCHES,
    save_checked,
)

GRADED = "#2e6fb7"
UNIFORM = "#1a1a1a"
LIMIT = "#929292"
TARGET = 0.5


def collect(record: dict) -> dict:
    """Group the shell arms by mesh family, ordered by cell count."""
    series = {"uniform": [], "graded": []}
    for section in ("shell", "sweep"):
        for label, row in record.get(section, {}).items():
            if not row.get("converged") or "relative_error" not in row:
                continue
            kind = "uniform" if label.startswith("uniform") else "graded"
            series[kind].append(
                dict(
                    label=label,
                    cells=int(row["cells"]),
                    error=np.asarray(row["relative_error"], dtype=float) * 100,
                )
            )
    for rows in series.values():
        rows.sort(key=lambda row: row["cells"])
    return series


def render(run, destination: Path, png: Path | None = None):
    record = json.loads(Path(Path(run) / "measurement.json").read_text())
    series = collect(record)
    assert series["uniform"] and series["graded"], series.keys()
    plt.style.use("data-ink")
    plt.rcParams["svg.fonttype"] = "none"
    fig, axis = plt.subplots(figsize=(WIDTH_INCHES, HEIGHT_INCHES), dpi=100)
    for kind, colour, style, marker, width in (
        ("uniform", UNIFORM, (0, (5, 3)), "o", 2.6),
        ("graded", GRADED, "-", "D", 3.0),
    ):
        rows = series[kind]
        cells = np.array([row["cells"] for row in rows], dtype=float)
        worst = np.array([row["error"].max() for row in rows])
        best = np.array([row["error"].min() for row in rows])
        axis.plot(
            cells,
            worst,
            color=colour,
            linestyle=style,
            marker=marker,
            markersize=7,
            markerfacecolor=colour,
            markeredgecolor=colour,
            linewidth=width,
            label=f"{kind} mesh, worst probe",
            zorder=4 if kind == "graded" else 3,
        )
        axis.plot(
            cells,
            best,
            color=colour,
            linestyle=style,
            linewidth=1.2,
            alpha=0.6,
            label=f"{kind} mesh, best probe",
            zorder=axis.get_zorder(),
        )
    axis.axhline(TARGET, color=LIMIT, linestyle=(0, (1, 3)), linewidth=1.2, zorder=1)
    axis.annotate(
        "0.5 % target",
        xy=(0.02, TARGET),
        xycoords=("axes fraction", "data"),
        xytext=(0, 6),
        textcoords="offset points",
        color=LIMIT,
        fontsize=14,
    )
    axis.set_xscale("log")
    counts = [row["cells"] for kind in ("uniform", "graded") for row in series[kind]]
    low, high = min(counts) * 0.8, max(counts) * 1.25
    axis.set_xlim(low, high)
    ladder = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
    axis.xaxis.set_major_locator(
        ticker.FixedLocator([value for value in ladder if low <= value <= high])
    )
    axis.xaxis.set_minor_locator(ticker.NullLocator())
    worst_error = max(row["error"].max() for kind in series for row in series[kind])
    top = float(np.ceil(worst_error * 1.1))
    axis.set_ylim(0, top)
    step = 2.0 if top <= 12 else 5.0
    axis.yaxis.set_major_locator(ticker.FixedLocator(np.arange(0.0, top + step, step)))
    axis.set_xlabel("cells")
    axis.set_ylabel("closed-shell error [%]")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(labelsize=14)
    fig.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=4,
        frameon=False,
        fontsize=14,
        handlelength=2.4,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    save_checked(fig, str(destination), str(png) if png else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--png", type=Path, default=None)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    render(args.run, args.output, args.png)


if __name__ == "__main__":
    main()
