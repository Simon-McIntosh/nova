"""Draw paired line-contour poloidal panels from measured terminal states."""

# ruff: noqa: E402

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "poloidal-contour"
import matplotlib.pyplot as plt
import numpy as np

from nova.media.ink import DEFAULT_INK, poloidal_axes
from nova.media.poloidal import draw_nulls, draw_scattered_contours, draw_wall


def panel(data, output):
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
        analytic,
        levels,
        wall,
        color="#444444",
        linewidth=2.6,
    )
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
    draw_wall(axis, wall[:, 0], wall[:, 1])
    solved_roots = np.asarray(data["solved_roots"])
    analytic_roots = np.asarray(data["analytic_roots"])
    reference_style = DEFAULT_INK.variant(axis_color="#444444", xpoint_color="#444444")
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
        style=DEFAULT_INK,
        contain=wall,
    )
    axis.text(
        0.02,
        0.06,
        "solved flux and nulls",
        transform=axis.transAxes,
        color=DEFAULT_INK.flux_color,
        fontsize=20,
    )
    axis.text(
        0.02,
        0.01,
        "analytic flux and nulls",
        transform=axis.transAxes,
        color="#444444",
        fontsize=20,
    )
    axis.autoscale_view()
    figure.savefig(output, format="svg", metadata={"Date": None})
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = json.loads(args.input.read_text())
    for label, payload in data["panels"].items():
        name = f"{args.prefix}-{label}.svg"
        panel(payload, args.output / name)
        print(f"PANEL {name} residual={max(payload['analytic_root_residuals'])}")


if __name__ == "__main__":
    main()
