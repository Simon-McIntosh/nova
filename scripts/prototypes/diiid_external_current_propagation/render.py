"""Draw the base and positive-control terminal flux of frame 137D together.

Both panels share one absolute level array so the two terminal states are read
on the same contour values. Each panel carries the vessel, its own nulls in the
solved vocabulary and the base arm's nulls as the reference set, so a movement
is read against a mark that does not depend on which panel is open.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from nova.media.poloidal import (  # noqa: E402
    draw_flux_contours,
    draw_nulls,
    draw_wall,
)
from scripts.prototypes.diiid_xpoint_offset.style import (  # noqa: E402
    REFERENCE,
    SOLVED,
    WALL,
)

CANVAS = (14, 7.2)
LEVEL_COUNT = 22
PANELS = (
    ("base", "measured currents"),
    ("row", "coil currents scaled x1.1"),
)


def _load(folder: Path, arm: str):
    data = np.load(folder / f"arm-{arm}.npz")
    return {name: np.asarray(data[name]) for name in data.files}


def _levels(base_flux, wall, radius, height):
    """Return one absolute level array over the base field inside the vessel."""

    from nova.media.poloidal import inside_wall_units

    grid_r, grid_z = np.meshgrid(radius, height)
    inside = inside_wall_units(
        np.column_stack((grid_r.ravel(), grid_z.ravel())), wall
    ).reshape(base_flux.shape)
    values = base_flux[inside]
    low, high = float(np.min(values)), float(np.max(values))
    return np.linspace(low, high, LEVEL_COUNT)[1:-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previews", type=Path, required=True)
    args = parser.parse_args()

    plt.style.use(Path.home() / ".config/matplotlib/stylelib/data-ink.mplstyle")
    plt.rcParams.update({"svg.fonttype": "none", "font.size": 14})

    arms = {arm: _load(args.input, arm) for arm, _ in PANELS}
    base = arms["base"]
    radius, height = base["radius"], base["height"]
    wall = base["wall"]
    # Saved flux is (radius, height); the painter wants (height, radius).
    levels = _levels(base["flux"].T, wall, radius, height)
    reference = dict(
        axis=base["axis"],
        x_point=base["lower_x"][None, :],
    )

    figure = plt.figure(figsize=(14, 7.2), dpi=100)
    axes = [
        figure.add_axes(box)
        for box in ((0.03, 0.10, 0.44, 0.80), (0.51, 0.10, 0.44, 0.80))
    ]
    notes = []
    for axes_item, (arm, label) in zip(axes, PANELS, strict=True):
        data = arms[arm]
        axes_item.set_axis_off()
        draw_wall(
            axes_item, units=None, radius=wall[:, 0], height=wall[:, 1], color=WALL
        )
        draw_flux_contours(
            axes_item,
            radius,
            height,
            data["flux"].T,
            levels,
            color=SOLVED,
            wall=wall,
            linewidth=1.2,
        )
        draw_nulls(
            axes_item,
            magnetic_axis=data["axis"],
            x_points=data["lower_x"][None, :],
            other_x_points=reference["x_point"] if arm != "base" else None,
        )
        notes.append(
            axes_item.text(
                0.02,
                0.98,
                label,
                transform=axes_item.transAxes,
                va="top",
                ha="left",
                fontsize=14,
                color=SOLVED,
            )
        )
        record = json.loads(
            (args.input / f"arm-{arm}.json").read_text(encoding="utf-8")
        )
        notes.append(
            axes_item.text(
                0.02,
                0.02,
                "lower X offset %.3f / %.3f cm, residual %.2e"
                % (
                    record["lower_x_offset_cm"][0],
                    record["lower_x_offset_cm"][1],
                    record["residual"],
                ),
                transform=axes_item.transAxes,
                va="bottom",
                ha="left",
                fontsize=12,
                color=REFERENCE,
            )
        )

    from scripts.prototypes.diiid_xpoint_offset.style import save

    args.output.mkdir(parents=True, exist_ok=True)
    args.previews.mkdir(parents=True, exist_ok=True)
    save(
        figure,
        axes,
        notes,
        args.output / "frame-137-terminal-base-versus-scaled.svg",
        args.previews / "frame-137-terminal-base-versus-scaled.png",
    )


if __name__ == "__main__":
    main()
