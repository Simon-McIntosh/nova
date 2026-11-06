"""Error against strip width, one curve per slope, from the sweep receipt.

Renders the measured curve from ``strip-sweep.json``. The figure is written
through :func:`save_checked`, which refuses to write when a text artist runs off
the canvas or two of them overlap, so a clipped label cannot reach a reader.

    python scripts/edge_sheet/figure.py --json <receipt> --output <svg>
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# one hue in lightness steps: slope is an ordered quantity, so the ramp carries
# the ordering. Adjacent steps are below the colour-blind separation threshold on
# their own, so each curve also carries its own dash pattern.
COLOURS = ("#08306b", "#1b5fa0", "#3b83c0", "#6aa6d6", "#94c0e3")
DASHES = ("solid", (0, (7, 2)), (0, (4, 2)), (0, (2, 2)), (0, (1, 1)))
LABELS = ("0", "π/8", "π/4", "3π/8", "π/2")

WIDTH_INCHES = 14.0
HEIGHT_INCHES = 7.2


def _text_artists(fig):
    artists = []
    for axis in fig.axes:
        artists.extend(axis.get_xticklabels())
        artists.extend(axis.get_yticklabels())
        artists.append(axis.xaxis.label)
        artists.append(axis.yaxis.label)
        artists.extend(axis.texts)
    artists.extend(fig.texts)
    legend = fig.legends[0] if fig.legends else None
    if legend is not None:
        artists.extend(legend.get_texts())
    return [artist for artist in artists if artist.get_text().strip()]


def save_checked(fig, path: str, png: str | None = None) -> list[str]:
    """Write ``fig`` to ``path`` only if no text is clipped or overlapping."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.bbox
    problems = []
    boxes = []
    for artist in _text_artists(fig):
        box = artist.get_window_extent(renderer)
        if (
            box.x0 < canvas.x0 - 0.5
            or box.y0 < canvas.y0 - 0.5
            or box.x1 > canvas.x1 + 0.5
            or box.y1 > canvas.y1 + 0.5
        ):
            problems.append("clipped: %r" % artist.get_text())
        boxes.append((artist.get_text(), box))
    for index, (name, box) in enumerate(boxes):
        for other_name, other in boxes[index + 1 :]:
            if box.overlaps(other):
                problems.append("overlap: %r with %r" % (name, other_name))
    if problems:
        raise SystemExit("figure refused:\n  " + "\n  ".join(problems))
    fig.savefig(path, format="svg")
    if png:
        fig.savefig(png, dpi=200)
    return [name for name, _ in boxes]


def render(receipt: dict, output: str, png: str | None) -> None:
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.2,
            "xtick.major.width": 1.2,
            "ytick.major.width": 1.2,
            "lines.linewidth": 2.6,
            "svg.fonttype": "none",
        }
    )
    fractions = np.array(receipt["geometry"]["fractions"])
    tolerance = receipt["geometry"]["tolerance"]
    fig, axis = plt.subplots(figsize=(WIDTH_INCHES, HEIGHT_INCHES), dpi=100)
    for index, slope in enumerate(receipt["slopes"]):
        axis.plot(
            fractions,
            slope["curve"],
            color=COLOURS[index],
            linestyle=DASHES[index],
            linewidth=3.0 if index == 0 else 2.6,
            zorder=3 + index,
            label="slope %s" % LABELS[index],
        )
        axis.plot(
            [slope["best_fraction"]],
            [slope["best_error"]],
            marker="o",
            markersize=5.0,
            markeredgecolor="white",
            markeredgewidth=0.8,
            color=COLOURS[index],
            zorder=10,
        )
    axis.axhline(
        tolerance, color="#929292", linewidth=1.2, linestyle=(0, (1, 2)), zorder=1
    )
    axis.text(
        fractions[-1] * 1.12,
        tolerance * 1.25,
        "1e-10",
        color="#6b6b6b",
        fontsize=14,
        ha="right",
        va="bottom",
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("strip width / segment length")
    axis.set_ylabel("worst relative error over the target set")
    axis.set_xlim(fractions.min() / 1.8, fractions.max() * 1.9)
    # explicit ticks keep the locator from asking for labels outside the view,
    # which would be drawn off the canvas
    axis.set_xticks([1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2])
    axis.set_yticks([1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3])
    axis.set_ylim(5e-11, 4e-3)
    axis.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=5,
        frameon=False,
        handlelength=2.4,
        columnspacing=1.8,
    )
    fig.tight_layout(rect=(0.01, 0.01, 0.99, 0.92))
    names = save_checked(fig, output, png)
    print("checked %d text artists; wrote %s" % (len(names), output))
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--png", default=None)
    arguments = parser.parse_args()
    with open(arguments.json) as handle:
        receipt = json.load(handle)
    render(receipt, arguments.output, arguments.png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
