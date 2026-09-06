"""Render the two-panel playable page record figure from the channel data.

The playable app draws its frames through Bokeh, which needs a browser to
rasterise; this driver renders the same channel data through matplotlib so the
committed record figure reflects exactly the channels the page binds: the
styled poloidal view on the left, the decoded camera panel on the right (the
grey placeholder carrying the frame index), and below, the compensating
current bars with the commanded key legend and the keyframe wall/trips
sparkline.  Run on the CPU lane from the repository root:

    UV_PROJECT_ENVIRONMENT=~/Code/nova/.venv PYTHONPATH="$PWD" \
      JAX_PLATFORMS=cpu uv run --no-sync python \
        docs/figures/playable-forward-solve/view/render_two_panel_page.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from apps.playable.camera import PlaceholderDecoder, camera_push
from apps.playable.machines import build_machine
from apps.playable.production import ProductionSolver
from apps.playable.session import PlayableSession
from apps.playable.shape import PlasmaShape
from apps.pulsedesign.poloidal_view import (
    O_POINT_COLOR,
    PLASMA_FILL_ALPHA,
    PLASMA_FILL_COLOR,
    PLASMA_LINE_COLOR,
    WALL_COLOR,
    X_POINT_COLOR,
    X_POINT_SECONDARY_COLOR,
    poloidal_channels,
)

OUTPUT = Path(__file__).with_name("two-panel-page.png")

#: The held key instructions the record figure notes under the sparkline.
MOVES = ("bulk_r+", "elongation+", "x_point_z+", "inner_gap+")


def main() -> None:
    """Drive the playable session on the Solov'ev fixture and render the page."""
    machine = build_machine("solovev")
    session = PlayableSession(
        solver=ProductionSolver(machine),
        shape=PlasmaShape(),
        machine="solovev",
        wall=machine.wall,
    )
    session.decoder = PlaceholderDecoder()

    session.prime()
    decoded = session.decode_frame()
    assert decoded is not None
    for key in MOVES:
        session.step(key)
        decoded = session.decode_frame()
        assert decoded is not None
    camera = np.asarray(decoded.image)
    pushed = camera_push(session, decoded)
    sparkline = pushed["sparkline"]
    compensation = frame_push_compensation(session)

    channels = poloidal_channels(
        session.equilibrium,
        machine.profile,
        wall=machine.wall,
        coils=machine.coils,
    )

    figure = plt.figure(figsize=(15, 9))
    grid = figure.add_gridspec(2, 2, height_ratios=(4, 1.4), hspace=0.28, wspace=0.18)

    # --- left: the styled poloidal view -----------------------------------
    axis = figure.add_subplot(grid[0, 0])
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
        axis.plot(x_line, z_line, color="#3f3f3f", linewidth=0.9)
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
    axis.set_title("poloidal view (no axes)", fontsize=11)

    # --- right: the decoded camera panel ----------------------------------
    axis = figure.add_subplot(grid[0, 1])
    axis.imshow(camera, aspect="equal")
    axis.axis("off")
    axis.add_patch(
        Rectangle(
            (0, 0),
            camera.shape[1],
            camera.shape[0],
            fill=False,
            edgecolor="#333",
        )
    )
    axis.set_title("decoded MAST camera (rbb) — placeholder", fontsize=11)

    # --- bottom left: commanded legend and compensating currents ----------
    axis = figure.add_subplot(grid[1, 0])
    circuits = np.asarray(compensation["circuit"], dtype=float)
    currents = np.asarray(compensation["current"], dtype=float)
    width = 0.7 if circuits.size else 1.0
    if circuits.size:
        axis.bar(circuits, currents, width=width, color="#8a6fb5")
        axis.set_xlabel("circuit")
        axis.set_ylabel("compensating current / A", fontsize=9)
    else:
        axis.text(
            0.5, 0.5, "compensating current per circuit", ha="center", va="center"
        )
    axis.tick_params(labelsize=8)
    shape = session.shape
    command = (
        f"R {shape.axis_r:.3f}  Z {shape.axis_z:+.3f}  κ {shape.elongation:.2f}  "
        f"δu {shape.triangularity_upper:+.2f}  δl {shape.triangularity_lower:+.2f}"
    )
    axis.set_title(
        f"keys R Z κ δu δl Xr Xz gap-in gap-out · commanded: {command}", fontsize=9
    )

    # --- bottom right: keyframe wall and trips sparkline ------------------
    axis = figure.add_subplot(grid[1, 1])
    indices = sparkline["index"]
    axis.plot(
        indices, sparkline["wall"], color="#3f3f3f", linewidth=1.5, label="wall / s"
    )
    right = axis.twinx()
    right.plot(
        indices,
        sparkline["trips"],
        color="#a98fd0",
        linewidth=1.5,
        label="trips",
    )
    right.set_ylim(0, max(8, int(sparkline["trips"].max()) + 1))
    axis.set_xlabel("keyframe")
    axis.set_ylabel("wall / s", fontsize=9)
    right.set_ylabel("trips", fontsize=9)
    axis.tick_params(labelsize=8)
    right.tick_params(labelsize=8)
    axis.set_title(
        f"keyframe wall and trips · record ● · session frame "
        f"{session.frame_index:04d}/{session.frame_index:04d}",
        fontsize=9,
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes)")


def frame_push_compensation(session) -> dict[str, np.ndarray]:
    """Return the per-circuit compensation channel the page draws."""
    from apps.playable.session import compensating_currents

    return compensating_currents(session.equilibrium)


if __name__ == "__main__":
    main()
