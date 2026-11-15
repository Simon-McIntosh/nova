"""Draw the base and positive-control terminal flux of frame 137D together.

Both panels share one absolute level array, so the two terminal states are read
on the same contour values. Each draws the vessel, its own nulls and, on the
scaled arm, the base arm's lower null as a reference that does not move with
the panel. The flux grid is 33 by 33 over the canonical axes, so a lower-null
displacement of a few centimetres is far below one cell; the nulls are drawn
from the terminal topology, which resolves them independently of the grid.
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

LEVEL_COUNT = 22
PANELS = (
    ("base", "measured currents"),
    ("row", "coil currents scaled x1.1"),
)
COLUMN_LEFT = (0.06, 0.56)
COLUMN_WIDTH = 0.36
PANEL_BOTTOM = 0.085
PANEL_HEIGHT = 0.775


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
    return np.linspace(float(np.min(values)), float(np.max(values)), LEVEL_COUNT)[1:-1]


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
    reference = base["lower_x"][None, :]

    figure = plt.figure(figsize=(14, 7.2), dpi=100)
    axes: list = []
    notes: list = []
    for index, (arm, label) in enumerate(PANELS):
        data = arms[arm]
        record = json.loads(
            (args.input / f"arm-{arm}.json").read_text(encoding="utf-8")
        )
        # Offsets come from the stored null coordinates in metres, never from
        # the arm record's centimetre field.
        offset_cm = 100.0 * (
            np.asarray(data["lower_x"]) - np.asarray(data["efit_lower_x"])
        )
        left = COLUMN_LEFT[index]
        item = figure.add_axes((left, PANEL_BOTTOM, COLUMN_WIDTH, PANEL_HEIGHT))
        axes.append(item)
        item.set_axis_off()
        draw_wall(item, units=None, radius=wall[:, 0], height=wall[:, 1], color=WALL)
        draw_flux_contours(
            item,
            radius,
            height,
            data["flux"].T,
            levels,
            color=SOLVED,
            wall=wall,
            linewidth=1.2,
        )
        draw_nulls(
            item,
            magnetic_axis=data["axis"],
            x_points=data["lower_x"][None, :],
            other_x_points=None if arm == "base" else reference,
        )
        item.set_xlim(radius.min() - 0.04, radius.max() + 0.04)
        item.set_ylim(height.min() - 0.04, height.max() + 0.04)
        item.set_aspect("equal")
        item.set_axis_off()
        notes.append(
            figure.text(
                left, 0.945, label, va="bottom", ha="left", fontsize=14, color=SOLVED
            )
        )
        notes.append(
            figure.text(
                left,
                0.025,
                "offset %.3f / %+.3f cm, %.3f cm from EFIT null, residual %.1e"
                % (
                    offset_cm[0],
                    offset_cm[1],
                    float(np.linalg.norm(offset_cm)),
                    record["residual"],
                ),
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
