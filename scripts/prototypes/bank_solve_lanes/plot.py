"""Draw the per-iterate residual history of the sixteen-lane bank solve.

The history is ``SolveResult.residual_path`` and ``update_path`` as the driver
itself returns them, so no reconstructed trajectory enters the figure.
"""

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
FIGURE = (
    ROOT
    / "docs/figures/sub-millisecond-forward-path/bank-solve-lanes/residual-history.svg"
)

SLOW = "#c0442b"
SLOW_SECOND = "#d98a2b"
MEDIAN = "#1f5d92"
POPULATION = "#929292"
FREEZE_THRESHOLD = 1e-6


def covered_by(box, points, margin=2.0):
    """How many plotted samples a text box covers, counting a clearance margin."""
    if not len(points):
        return 0
    return int(
        (
            (points[:, 0] >= box.x0 - margin)
            & (points[:, 0] <= box.x1 + margin)
            & (points[:, 1] >= box.y0 - margin)
            & (points[:, 1] <= box.y1 + margin)
        ).sum()
    )


def plotted_points(fig, samples=48):
    """Every plotted vertex and the interior of every plotted segment.

    Sampling the segments, not only the vertices, is what makes a label that
    sits in the gap between two vertices visible to the check: a polyline
    crosses the gap whether or not a vertex lands in it. The sampling runs in
    pixels, not in data coordinates, because that is what the renderer draws: a
    segment on a log axis is a straight chord between two pixels, not the image
    of a straight line in data space, and interpolating the data and then
    transforming leaves the chord crossing a label the samples report as clear.
    """
    points = []
    for axis in fig.axes:
        transform = axis.transData
        for line in axis.lines:
            xy = np.asarray(line.get_xydata(), dtype=float)
            if xy.size == 0:
                continue
            pixels = transform.transform(xy[:, :2])
            if len(pixels) == 1:
                points.append(pixels)
                continue
            starts, ends = pixels[:-1], pixels[1:]
            weight = np.linspace(0.0, 1.0, samples)[:, None, None]
            points.append(
                (starts[None] * (1.0 - weight) + ends[None] * weight).reshape(-1, 2)
            )
    return np.concatenate(points) if points else np.zeros((0, 2))


def save_checked(fig, output):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.bbox
    texts = [
        item for item in fig.findobj(Text) if item.get_visible() and item.get_text()
    ]
    boxes = [(item, item.get_window_extent(renderer)) for item in texts]
    for item, box in boxes:
        if not all(canvas.contains(*corner) for corner in box.get_points()):
            raise ValueError(f"text outside canvas: {item.get_text()}")
    for index, (left, left_box) in enumerate(boxes):
        for right, right_box in boxes[index + 1 :]:
            if left_box.overlaps(right_box):
                raise ValueError(
                    f"text overlap: {left.get_text()} / {right.get_text()}"
                )
    points = plotted_points(fig)
    if len(points):
        for item, box in boxes:
            covered = covered_by(box, points)
            if covered:
                raise ValueError(
                    f"text over plotted data: {item.get_text()!r} covers "
                    f"{covered} data point(s)"
                )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    root = ET.parse(output)
    root.getroot().set("width", "31.1cm")
    root.getroot().set("height", "16cm")
    root.write(output, encoding="unicode", xml_declaration=True)
    print(f"FIGURE={output} TEXT_CHECK=passed CANVAS=1400x720", flush=True)


def histories(payload):
    """Each lane's fresh residual per iterate, truncated at its own step count."""
    residual = np.asarray(payload["paths"]["residual"])
    steps = np.asarray(payload["solve_batch"]["steps"])
    return [residual[lane, : int(steps[lane]) + 1] for lane in range(len(steps))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--output", type=Path, default=FIGURE)
    args = parser.parse_args()
    payload = json.loads(args.receipt.read_text())
    steps = np.asarray(payload["solve_batch"]["steps"])
    curves = histories(payload)
    order = np.argsort(steps)
    slowest, second = int(order[-1]), int(order[-2])
    median_lane = int(order[len(order) // 2])
    colour = {slowest: SLOW, second: SLOW_SECOND, median_lane: MEDIAN}
    report = {
        "steps": steps.tolist(),
        "slowest": slowest,
        "second": second,
        "median": median_lane,
        "median_steps": int(steps[median_lane]),
    }
    print("REPORT " + json.dumps(report, sort_keys=True), flush=True)

    plt.style.use("data-ink")
    fig, ax = plt.subplots(figsize=(14, 7.2), dpi=100)
    for lane, curve in enumerate(curves):
        if lane in colour:
            continue
        ax.plot(range(len(curve)), curve, color=POPULATION, linewidth=1.4, zorder=2)
    for lane in (median_lane, second, slowest):
        curve = curves[lane]
        iterate = list(range(len(curve)))
        ax.plot(iterate, curve, color=colour[lane], linewidth=3.0, zorder=5)
        locked = [at for at, value in enumerate(curve) if value < FREEZE_THRESHOLD]
        if locked:
            ax.plot(
                [locked[0]],
                [curve[locked[0]]],
                linestyle="none",
                marker="o",
                markersize=8,
                markerfacecolor="white",
                markeredgecolor=colour[lane],
                markeredgewidth=2.0,
                zorder=7,
            )
        ax.axhline(
            FREEZE_THRESHOLD, color="#222222", linewidth=1.2, linestyle=":", zorder=3
        )
    ax.annotate(
        "freeze threshold, 1e-6",
        (1.0, FREEZE_THRESHOLD),
        xycoords=("axes fraction", "data"),
        textcoords="offset points",
        xytext=(0, 5),
        ha="right",
        fontsize=14,
        color="#222222",
        zorder=9,
    )
    ax.set_yscale("log")
    ax.set_xlabel("iterate")
    ax.set_ylabel("fresh residual")
    ax.set_xlim(0, max(len(curve) for curve in curves) - 1 + 9)
    ax.set_ylim(1e-13, 1.0)
    ax.tick_params(axis="both", labelsize=14)
    ax.xaxis.label.set_size(16)
    ax.yaxis.label.set_size(16)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(False)
    handles = [
        Line2D(
            [],
            [],
            color=SLOW,
            linewidth=3.0,
            label=f"slowest lane {slowest + 1}, {int(steps[slowest])} steps",
        ),
        Line2D(
            [],
            [],
            color=SLOW_SECOND,
            linewidth=3.0,
            label=f"second slowest lane {second + 1}, {int(steps[second])} steps",
        ),
        Line2D(
            [],
            [],
            color=MEDIAN,
            linewidth=3.0,
            label=f"median lane {median_lane + 1}, {int(steps[median_lane])} steps",
        ),
        Line2D([], [], color=POPULATION, linewidth=1.4, label="other 13 lanes"),
        Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markerfacecolor="white",
            markeredgecolor="#333333",
            markersize=8,
            markeredgewidth=2.0,
            label="active set locked",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
        fontsize=14,
    )
    fig.subplots_adjust(left=0.08, right=0.86, bottom=0.11, top=0.84)
    save_checked(fig, args.output)
    plt.close(fig)


if __name__ == "__main__":
    main()
