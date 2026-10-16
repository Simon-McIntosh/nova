"""Render the measured cell-integral and solve comparisons."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter, NullLocator

from nova.media.ink import DEFAULT_INK, poloidal_axes
from nova.media.poloidal import draw_nulls, draw_scattered_contours, draw_wall


GREY = "#929292"
BLUE = "#326c88"
REFERENCE = "#666666"


def _save(figure, output):
    figure.tight_layout(rect=(0, 0, 1, 0.88))
    figure.canvas.draw()
    canvas = figure.bbox
    artists = [
        artist
        for artist in figure.findobj(plt.Text)
        if artist.get_visible() and artist.get_text()
    ]
    bounds = [(artist.get_text(), artist.get_window_extent()) for artist in artists]
    for label, box in bounds:
        if not canvas.contains(box.x0, box.y0) or not canvas.contains(box.x1, box.y1):
            raise ValueError(f"text outside canvas: {label}")
    for position, (left_name, left) in enumerate(bounds):
        for right_name, right in bounds[position + 1 :]:
            overlap = left.intersection(left, right)
            if overlap is not None and overlap.width > 1 and overlap.height > 1:
                raise ValueError(f"text overlap: {left_name} / {right_name}")
    figure.savefig(output, format="svg", metadata={"Date": None})
    plt.close(figure)


def agreement(receipts, output):
    plt.style.use("data-ink")
    plt.rcParams["axes.grid"] = False
    figure, axes = plt.subplots(1, 3, figsize=(14, 5.5), dpi=100, sharey=True)
    for axis, receipt in zip(axes, receipts, strict=True):
        rows = receipt["rows"]
        ids = np.asarray([row["cell"] for row in rows])
        positions = np.arange(len(ids))
        errors = np.asarray([max(row["relative_error"]) for row in rows])
        errors = np.maximum(errors, 1e-18)
        largest = int(np.argmax(errors))
        axis.scatter(positions, errors, color=GREY, s=45, zorder=2)
        axis.scatter(positions[largest], errors[largest], color=BLUE, s=70, zorder=3)
        axis.axhline(1e-12, color=REFERENCE, linestyle=":", linewidth=1.2, zorder=1)
        axis.set_yscale("log")
        axis.set_ylim(3e-19, 3e-11)
        axis.set_xticks(positions, labels=ids)
        axis.set_xlabel(f"cell ({receipt['cells']:,} total)")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0].set_ylabel("scaled relative integral difference")
    figure.legend(
        handles=[
            Line2D([], [], marker="o", linestyle="", color=GREY, label="cell"),
            Line2D(
                [], [], marker="o", linestyle="", color=BLUE, label="largest difference"
            ),
            Line2D([], [], linestyle=":", color=REFERENCE, label="10⁻¹² limit"),
        ],
        loc="upper center",
        ncol=3,
        frameon=False,
    )
    _save(figure, output)


def map_error(receipts, output, connected=None):
    plt.style.use("data-ink")
    figure, axis = plt.subplots(figsize=(14, 5.5), dpi=100)
    cells = np.asarray([receipt["cells"] for receipt in receipts])
    observed = np.asarray([receipt["centre_map_flux_error"] for receipt in receipts])
    baseline = np.asarray((0.00108657917135, 0.000487271823, 0.0000866659))
    axis.loglog(
        cells,
        baseline,
        color=REFERENCE,
        linestyle="--",
        linewidth=2.6,
        marker="o",
        label="inherited map",
    )
    axis.loglog(
        cells,
        observed,
        color=BLUE,
        linewidth=3.0,
        marker="o",
        label="axis-facing Bell",
    )
    if connected:
        connected_error = np.asarray(
            [receipt["centre_map_flux_error"] for receipt in connected]
        )
        axis.loglog(
            cells,
            connected_error,
            color=BLUE,
            linestyle=":",
            linewidth=2.6,
            marker="s",
            label="connected Bell",
        )
    axis.set_xticks(cells, labels=[f"{number:,}" for number in cells])
    axis.xaxis.set_minor_locator(NullLocator())
    axis.xaxis.set_minor_formatter(NullFormatter())
    axis.set_xlim(cells[0] * 0.85, cells[-1] * 1.15)
    axis.set_ylim(5e-5, 0.12 if connected else 2e-3)
    axis.set_yticks((1e-4, 1e-3, 1e-2) if connected else (1e-4, 1e-3))
    axis.yaxis.set_minor_locator(NullLocator())
    axis.yaxis.set_minor_formatter(NullFormatter())
    axis.set_xlabel("carrier cells")
    axis.set_ylabel("normalized analytic map flux error")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    figure.legend(loc="upper center", ncol=2, frameon=False)
    _save(figure, output)


def residual_histories(receipts, output):
    plt.style.use("data-ink")
    figure, axis = plt.subplots(figsize=(14, 5.5), dpi=100)
    values = []
    for receipt, color in zip(receipts, (REFERENCE, BLUE, "#9a5f47"), strict=True):
        history = [receipt["centre_residual"]] + [
            row["residual"] for row in receipt["history"]
        ]
        values.extend(history)
        axis.semilogy(
            np.arange(len(history)),
            history,
            color=color,
            linewidth=3.0,
            marker="o",
            markersize=5,
            label=f"{receipt['cells']:,} cells",
        )
    axis.axhline(1e-12, color=REFERENCE, linestyle=":", linewidth=1.2)
    axis.set_xlim(-0.15, max(len(receipt["history"]) for receipt in receipts) + 0.15)
    axis.set_xticks(np.arange(max(len(receipt["history"]) for receipt in receipts) + 1))
    finite = np.asarray(values)
    finite = finite[np.isfinite(finite) & (finite > 0)]
    axis.set_ylim(1e-14, max(1e-1, np.max(finite) * 3))
    axis.set_yticks((1e-12, 1e-9, 1e-6, 1e-3))
    axis.yaxis.set_minor_locator(NullLocator())
    axis.yaxis.set_minor_formatter(NullFormatter())
    axis.set_xlabel("Newton correction")
    axis.set_ylabel("scaled moment residual")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    figure.legend(loc="upper center", ncol=3, frameon=False)
    _save(figure, output)


def bell_contact(receipts, output):
    plt.style.use("data-ink")
    figure, axes = plt.subplots(1, 3, figsize=(14, 5.5), dpi=100, sharey=True)
    for axis, receipt in zip(axes, receipts, strict=True):
        rows = receipt["rows"]
        above = [row for row in rows if row["signed_critical_offset"] > 0]
        below = [row for row in rows if row["signed_critical_offset"] < 0]
        for series, color, label, width in (
            (above, BLUE, "connected through saddle", 3.0),
            (below, GREY, "axis branch only", 2.6),
        ):
            x = -np.log10([abs(row["signed_critical_offset"]) for row in series])
            y = [row["area_fraction"] for row in series]
            axis.plot(x, y, marker="o", color=color, linewidth=width, label=label)
        axis.set_xlim(8.8, 11.2)
        axis.set_xticks((9, 10, 11), labels=("10⁻⁹", "10⁻¹⁰", "10⁻¹¹"))
        axis.set_ylim(-0.025, 0.575)
        axis.set_yticks((0, 0.2, 0.4))
        axis.set_xlabel(f"|saddle offset| [Wb] ({receipt['cells']:,} cells)")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0].set_ylabel("connected area / cell area")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    _save(figure, output)


def terminal_panel(receipt, output):
    data = receipt["panel"]
    points = np.asarray(data["coordinates"])
    wall = np.asarray(data["wall"])
    solved = np.asarray(data["solved_flux"])
    analytic = np.asarray(data["analytic_flux"])
    lower = max(np.min(solved), np.min(analytic))
    upper = min(np.max(solved), np.max(analytic))
    levels = np.linspace(lower, upper, 9)[1:-1]
    figure, axis = plt.subplots(figsize=(14, 8), dpi=100)
    poloidal_axes(axis)
    draw_scattered_contours(
        axis,
        points[:, 0],
        points[:, 1],
        solved,
        levels,
        wall,
        color=DEFAULT_INK.flux_color,
        linewidth=3.0,
    )
    reference_lines = draw_scattered_contours(
        axis,
        points[:, 0],
        points[:, 1],
        analytic,
        levels,
        wall,
        color="#444444",
        linewidth=2.6,
    )
    reference_lines.set_linestyles("dashed")
    draw_wall(axis, wall[:, 0], wall[:, 1])
    reference_style = DEFAULT_INK.variant(
        axis_color="#444444",
        xpoint_color="#444444",
        axis_markersize=10,
        xpoint_markersize=11,
    )
    solved_style = DEFAULT_INK.variant(
        axis_color=DEFAULT_INK.flux_color,
        xpoint_color=DEFAULT_INK.flux_color,
    )
    analytic_roots = np.asarray(data["analytic_roots"])
    solved_roots = np.asarray(data["solved_roots"])
    draw_nulls(
        axis,
        analytic_roots[0],
        analytic_roots[1:],
        style=reference_style,
        contain=wall,
    )
    draw_nulls(
        axis,
        solved_roots[0],
        solved_roots[1:],
        style=solved_style,
        contain=wall,
    )
    axis.autoscale_view()
    figure.canvas.draw()
    figure.savefig(output, format="svg", metadata={"Date": None})
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agreement", type=Path, action="append", default=[])
    parser.add_argument("--solve", type=Path, action="append", default=[])
    parser.add_argument("--bell-map", type=Path, action="append", default=[])
    parser.add_argument("--panel", type=Path, action="append", default=[])
    parser.add_argument("--contact", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.agreement:
        if len(args.agreement) != 3:
            parser.error("three carrier receipts are required")
        agreement(
            [json.loads(path.read_text()) for path in args.agreement],
            args.output / "per-cell-agreement.svg",
        )
    if args.solve:
        if len(args.solve) != 3:
            parser.error("three carrier solve receipts are required")
        if args.bell_map and len(args.bell_map) != 3:
            parser.error("three connected Bell map receipts are required")
        receipts = [json.loads(path.read_text()) for path in args.solve]
        connected = [json.loads(path.read_text()) for path in args.bell_map]
        map_error(receipts, args.output / "map-error.svg", connected)
        residual_histories(receipts, args.output / "residual-histories.svg")
    elif args.bell_map:
        parser.error("connected Bell maps require the solve receipts")
    for path in args.panel:
        receipt = json.loads(path.read_text())
        terminal_panel(
            receipt,
            args.output / f"terminal-{receipt['cells']}.svg",
        )
    if args.contact:
        if len(args.contact) != 3:
            parser.error("three carrier contact receipts are required")
        bell_contact(
            [json.loads(path.read_text()) for path in args.contact],
            args.output / "bell-contact.svg",
        )


if __name__ == "__main__":
    main()
