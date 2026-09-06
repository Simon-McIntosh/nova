"""Render the styled poloidal view record figure from the channel data.

The playable app draws its fitted frame through Bokeh, which needs a browser
to rasterise; this driver renders the same styled channel data through
matplotlib so the committed record figure reflects exactly the channels the
poloidal view binds.  Run on the CPU lane from the repository root:

    UV_PROJECT_ENVIRONMENT=~/Code/nova/.venv PYTHONPATH="$PWD" \
      JAX_PLATFORMS=cpu uv run --no-sync python \
        docs/figures/playable-forward-solve/view/render_styled_poloidal_view.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from apps.playable.machines import build_machine
from apps.pulsedesign.poloidal_view import (
    CONTOUR_COLOR,
    O_POINT_COLOR,
    PLASMA_FILL_ALPHA,
    PLASMA_FILL_COLOR,
    PLASMA_LINE_COLOR,
    WALL_COLOR,
    X_POINT_COLOR,
    X_POINT_SECONDARY_COLOR,
    poloidal_channels,
)

OUTPUT = Path(__file__).with_name("styled-poloidal-view.png")


def main() -> None:
    """Solve the Solov'ev frame, assemble the styled channels, and render."""
    machine = build_machine("solovev")
    equilibrium = machine.profile.solve(machine.seed, route="host")
    channels = poloidal_channels(
        equilibrium,
        machine.profile,
        wall=machine.wall,
        coils=machine.coils,
    )

    _, axis = plt.subplots(figsize=(7, 7))
    axis.set_aspect("equal")
    axis.axis("off")

    wall = np.c_[channels["wall"]["x"], channels["wall"]["z"]]
    axis.plot(wall[:, 0], wall[:, 1], color=WALL_COLOR, linewidth=1.5)

    for x_row, z_row in zip(channels["coil"]["x"], channels["coil"]["z"], strict=True):
        ring = np.c_[x_row[0], z_row[0]]
        ring = np.vstack([ring, ring[:1]])
        axis.plot(ring[:, 0], ring[:, 1], color="#777777", linewidth=0.8)

    for x_row, z_row in zip(
        channels["plasma"]["x"], channels["plasma"]["z"], strict=True
    ):
        cell = np.c_[x_row[0], z_row[0]]
        axis.fill(
            cell[:, 0],
            cell[:, 1],
            color=PLASMA_FILL_COLOR,
            alpha=PLASMA_FILL_ALPHA,
            edgecolor=PLASMA_LINE_COLOR,
            linewidth=0.5,
        )

    for x_line, z_line in zip(
        channels["levelset"]["x"], channels["levelset"]["z"], strict=True
    ):
        axis.plot(x_line, z_line, color=CONTOUR_COLOR, linewidth=0.9)

    for x, z in zip(channels["o_points"]["x"], channels["o_points"]["z"], strict=True):
        axis.plot(x, z, "o", color=O_POINT_COLOR, ms=7)
    for x, z in zip(channels["x_points"]["x"], channels["x_points"]["z"], strict=True):
        axis.plot(x, z, "x", color=X_POINT_COLOR, mew=1.6, ms=12)
    for x, z in zip(
        channels["x_points_secondary"]["x"],
        channels["x_points_secondary"]["z"],
        strict=True,
    ):
        axis.plot(x, z, "x", color=X_POINT_SECONDARY_COLOR, mew=1.2, ms=8)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
