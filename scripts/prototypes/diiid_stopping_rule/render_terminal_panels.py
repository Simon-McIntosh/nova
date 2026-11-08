"""Draw a measured terminal state as a line-contour poloidal panel.

The panel carries the governed limiter, the bounding surface traced from the
terminal flux, the EFIT label boundary, the terminal flux contours on stated
levels and the nulls the arm recorded, so a reader can see where the state
actually sat relative to the machine.  The limiter supplies the scale: the
poloidal panels carry no axes, and the wall is what maps a position on the page
to a place in the vessel.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from nova.media.ink import DEFAULT_INK, poloidal_axes
from nova.media.poloidal import draw_nulls, draw_wall

LIMITER = "#1f1f1f"
BOUNDING = "#929292"


def _on_bounding_box(ring: np.ndarray, tolerance: float = 1.0e-6) -> int:
    """Count ring points lying on the ring's own bounding rectangle.

    The probe's artifact wall is that rectangle, so a ring reported wholly on
    its edges is the pseudo wall rather than the machine.
    """

    lower, upper = ring.min(axis=0), ring.max(axis=0)
    on_edge = np.isclose(ring, lower, atol=tolerance) | np.isclose(
        ring, upper, atol=tolerance
    )
    return int(np.count_nonzero(on_edge.any(axis=1)))


def _wall_gap(points: np.ndarray, ring: np.ndarray) -> float:
    """Shortest distance from points to the limiter's segments [m].

    Measured to the outline's segments rather than its vertices, since the ring
    spans up to a metre of wall between vertices.
    """

    start = ring
    edge = np.roll(ring, -1, axis=0) - start
    length2 = np.einsum("ij,ij->i", edge, edge)
    safe = np.where(length2 > 0.0, length2, 1.0)
    offset = points[:, None, :] - start[None, :, :]
    fraction = np.clip(np.einsum("ijk,jk->ij", offset, edge) / safe[None, :], 0.0, 1.0)
    onto = start[None, :, :] + fraction[..., None] * edge[None, :, :]
    return float(np.min(np.linalg.norm(points[:, None, :] - onto, axis=2)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--arm", default="default")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    data = np.load(arguments.artifact)
    flux = data[
        f"frames.{arguments.frame}.branches.{arguments.branch}"
        f".arms.{arguments.arm}._flux_grid"
    ]
    radius, height = data["radius"], data["height"]
    label_boundary = data["label_boundary"]
    geometry = np.load(arguments.geometry)
    limiter = geometry["limiter"]
    surface = geometry["surface"]
    record = json.loads(arguments.record.read_text())
    arm = record["frames"][str(arguments.frame)]["branches"][arguments.branch]["arms"][
        arguments.arm
    ]
    axis = np.asarray(arm["presence"]["axis_rz_m"], dtype=float)
    saddle = arm.get("x_point_rz_m")
    saddles = (
        np.asarray(saddle, dtype=float).reshape(1, 2) if saddle else np.zeros((0, 2))
    )
    frame_edge_points = _on_bounding_box(limiter)
    if frame_edge_points == len(limiter):
        raise SystemExit(
            "the recorded limiter is the flux-grid bounding box, not the machine "
            "wall; rerun record_machine_geometry.py"
        )

    finite = flux[np.isfinite(flux)]
    span = float(np.ptp(finite))
    levels = np.linspace(finite.min() + 0.1 * span, finite.max() - 0.1 * span, 7)

    figure, axes = plt.subplots(figsize=(14, 7.2), dpi=100)
    poloidal_axes(axes)
    axes.contour(
        radius,
        height,
        flux,
        levels=levels,
        colors=DEFAULT_INK.flux_color,
        linewidths=2.6,
        linestyles="solid",
        zorder=4,
    )
    if len(surface):
        axes.plot(
            surface[:, 0], surface[:, 1], color=BOUNDING, linewidth=1.6, zorder=3
        )
    axes.plot(
        label_boundary[:, 0],
        label_boundary[:, 1],
        color=BOUNDING,
        linestyle="--",
        linewidth=2.4,
        zorder=3,
    )
    draw_wall(axes, limiter[:, 0], limiter[:, 1], color=LIMITER, linewidth=2.2)
    draw_nulls(axes, axis, saddles, style=DEFAULT_INK, contain=limiter)
    axes.annotate(
        f"axis {axis[0]:.2f} m, {axis[1]:.2f} m",
        xy=(axis[0], axis[1]),
        xytext=(12, -14),
        textcoords="offset points",
        fontsize=14,
        color=DEFAULT_INK.axis_color,
    )
    axes.set_aspect("equal")
    lower, upper = limiter.min(axis=0), limiter.max(axis=0)
    margin = 0.05 * (upper - lower)
    axes.set_xlim(lower[0] - margin[0], upper[0] + margin[0])
    axes.set_ylim(lower[1] - margin[1], upper[1] + margin[1])
    handles = [
        Line2D([], [], color=DEFAULT_INK.flux_color, lw=2.6, label="solved flux"),
        Line2D([], [], color=LIMITER, lw=1.4, label="limiter"),
    ]
    if len(surface):
        handles.append(
            Line2D([], [], color=BOUNDING, lw=1.6, label="bounding surface")
        )
    handles += [
        Line2D([], [], color=BOUNDING, ls="--", lw=2.4, label="EFIT boundary"),
    ]
    axes.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
        fontsize=14,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output, bbox_inches="tight")
    print(
        f"WROTE {arguments.output} residual={arm['terminal_residual']:.6g} "
        f"limiter_points={len(limiter)} on_bounding_box={frame_edge_points} "
        f"surface_points={len(surface)} "
        f"surface_gap_m={_wall_gap(surface, limiter) if len(surface) else float('nan'):.4f} "
        f"axis_gap_m={_wall_gap(axis[None, :], limiter):.4f}"
    )


if __name__ == "__main__":
    main()