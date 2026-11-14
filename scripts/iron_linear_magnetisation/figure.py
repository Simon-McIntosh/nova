"""Plot closed-shell worst-probe error against magnetic unknown count."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as ticker  # noqa: E402
import numpy as np  # noqa: E402

from scripts.edge_sheet.figure import WIDTH_INCHES, HEIGHT_INCHES, save_checked  # noqa: E402
from scripts.iron_graded_refinement.figure import (
    GRADED as LINEAR,
    UNIFORM as CONSTANT,
    LIMIT,
)  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--linear", type=Path, required=True)
    parser.add_argument("--constant", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--png", type=Path, required=True)
    args = parser.parse_args()
    reference = json.loads(args.constant.read_text())
    constant = [
        (2 * row["cells"], max(row["relative_error"]) * 100)
        for label, row in reference["shell"].items()
        if label.startswith("uniform") and row["converged"]
    ]
    linear = []
    for path in args.linear.glob("linear-*.json"):
        row = json.loads(path.read_text())
        if row.get("completed") and row["converged"]:
            linear.append((row["unknowns"], row["worst_error"] * 100))
    if not constant or not linear:
        raise SystemExit("both measured basis families are required")
    plt.style.use(
        Path(__file__).parents[1]
        / "magnetisation_high_permeability"
        / "data-ink.mplstyle"
    )
    plt.rcParams["svg.fonttype"] = "none"
    fig, axis = plt.subplots(figsize=(WIDTH_INCHES, HEIGHT_INCHES), dpi=100)
    for label, values, colour, dash, marker, width, layer in (
        ("constant", constant, CONSTANT, "--", "o", 2.6, 3),
        ("linear", linear, LINEAR, "-", "D", 3.0, 4),
    ):
        values = np.asarray(sorted(values))
        axis.plot(
            values[:, 0],
            values[:, 1],
            color=colour,
            linestyle=dash,
            marker=marker,
            markersize=7,
            linewidth=width,
            label=label,
            zorder=layer,
        )
    axis.axhline(
        0.5, color=LIMIT, linestyle=":", linewidth=1.2, label="0.5 % target", zorder=1
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    all_values = np.array(constant + linear)
    low, high = all_values[:, 0].min() / 1.25, all_values[:, 0].max() * 1.25
    bottom, top = min(0.5, all_values[:, 1].min()) / 1.7, all_values[:, 1].max() * 1.6
    axis.set_xlim(low, high)
    axis.set_ylim(bottom, top)
    ladder = [1, 2, 5]

    def ticks_between(lower, upper):
        return [
            v * 10.0**power
            for power in range(-8, 9)
            for v in ladder
            if lower <= v * 10.0**power <= upper
        ]

    for scale, lower, upper in ((axis.xaxis, low, high), (axis.yaxis, bottom, top)):
        ticks = ticks_between(lower, upper)
        scale.set_major_locator(ticker.FixedLocator(ticks))
        scale.set_major_formatter(
            ticker.FixedFormatter([f"{value:g}" for value in ticks])
        )
        scale.set_minor_locator(ticker.NullLocator())
    axis.set_xlabel("magnetisation unknowns")
    axis.set_ylabel("worst-probe shielding error [%]")
    axis.spines[["top", "right"]].set_visible(False)
    fig.legend(loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.99))
    fig.tight_layout(rect=(0.01, 0.01, 0.99, 0.91))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.png.parent.mkdir(parents=True, exist_ok=True)
    names = save_checked(fig, str(args.output), str(args.png))
    # Physical dimensions preserve the slide size independently of SVG pixels.
    svg = args.output.read_text()
    svg = re.sub(
        r'width="[^"]+" height="[^"]+"', 'width="31.111cm" height="16cm"', svg, count=1
    )
    args.output.write_text(svg)
    print(
        f"checked {len(names)} text artists; every major tick labelled; {args.output}"
    )


if __name__ == "__main__":
    main()
