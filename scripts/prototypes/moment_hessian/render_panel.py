"""Draw the terminal limited flux beside its analytic reference."""

from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
import re  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402

from nova.media.ink import DEFAULT_INK, poloidal_axes  # noqa: E402
from nova.media.poloidal import (  # noqa: E402
    draw_nulls,
    draw_scattered_contours,
    draw_wall,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("receipt", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = json.loads(args.receipt.read_text())
    panel = data["panel"]
    coordinates = np.asarray(panel["coordinates"])
    wall = np.asarray(panel["wall"])
    analytic = np.asarray(panel["analytic_flux"])
    solved = np.asarray(panel["solved_flux"])
    reference_roots = np.asarray(panel["analytic_roots"])
    solved_roots = np.asarray(panel["solved_roots"])
    span = float(np.ptp(analytic))
    levels = np.linspace(
        float(np.min(analytic)) + 0.1 * span, float(np.max(analytic)) - 0.1 * span, 7
    )
    figure, axes = plt.subplots(figsize=(14, 7.2), dpi=100)
    poloidal_axes(axes)
    draw_scattered_contours(
        axes,
        coordinates[:, 0],
        coordinates[:, 1],
        analytic,
        levels,
        wall,
        color="#929292",
        linewidth=2.6,
        zorder=2,
    )
    draw_scattered_contours(
        axes,
        coordinates[:, 0],
        coordinates[:, 1],
        solved,
        levels,
        wall,
        color=DEFAULT_INK.flux_color,
        linewidth=3.0,
        zorder=3,
    )
    draw_wall(axes, wall[:, 0], wall[:, 1])
    draw_nulls(
        axes,
        reference_roots[0],
        reference_roots[1:],
        style=DEFAULT_INK.variant(axis_color="#929292", xpoint_color="#929292"),
        contain=wall,
    )
    draw_nulls(
        axes,
        solved_roots[0],
        solved_roots[1:],
        style=DEFAULT_INK.variant(
            axis_color=DEFAULT_INK.flux_color, xpoint_color=DEFAULT_INK.flux_color
        ),
        contain=wall,
    )
    axes.set_aspect("equal")
    figure.legend(
        handles=[
            Line2D([], [], color="#929292", lw=2.6, label="analytic flux"),
            Line2D([], [], color=DEFAULT_INK.flux_color, lw=3.0, label="solved flux"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=2,
        frameon=False,
        fontsize=14,
    )
    figure.subplots_adjust(left=0.03, right=0.97, top=0.9, bottom=0.03)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, format="svg")
    plt.close(figure)
    image = args.output.read_text()
    image, count = re.subn(
        r'width="[^"]+" height="[^"]+" viewBox=',
        'width="31.1cm" height="16cm" viewBox=',
        image,
        count=1,
    )
    assert count == 1
    args.output.write_text(image)
    print(
        f"panel={args.output} residual={panel['terminal_residual']} "
        f"converged={panel['converged']} levels={levels.tolist()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
