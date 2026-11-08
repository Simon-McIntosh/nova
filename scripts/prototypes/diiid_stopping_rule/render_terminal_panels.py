"""Draw a measured terminal state as a line-contour poloidal panel.

The panel carries the governed limiter, the frame's EFIT label boundary with its
own axis and X-point, the terminal flux contours on stated levels, and both null
sets -- the solved arm's axis and its recorded X-point, and the reference axis
and X-point -- so a reader can compare the solved topology against the reference
one and see where the state sat in the machine.  The limiter supplies the scale:
the poloidal panels carry no axes, and the wall is what maps a position on the
page to a place in the vessel.

Every figure is checked before it is written.  The margin notes must lie outside
the poloidal field, no two text artists may overlap, and the axis note names the
coordinates it marks, so a label never sits on the data it describes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from nova.media.ink import DEFAULT_INK, poloidal_axes
from nova.media.poloidal import draw_nulls, draw_wall

PROBE = Path(__file__).resolve().parent.parent / "diiid_forward_convergence"
sys.path.insert(0, str(PROBE))

from boundary_competition import _wall_distance  # noqa: E402

LIMITER = "#1f1f1f"
REFERENCE = "#929292"


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


def _gap(points: np.ndarray, ring: np.ndarray) -> float:
    """Shortest distance from points to the limiter's segments [m]."""

    distances, _onto = _wall_distance(
        np.atleast_2d(np.asarray(points, dtype=float)), ring
    )
    return float(distances.min())


def _text_boxes(figure, artists) -> list[tuple[str, object]]:
    renderer = figure.canvas.get_renderer()
    return [(name, artist.get_window_extent(renderer)) for name, artist in artists]


def _check_layout(figure, axes, artists) -> list[str]:
    """Refuse a figure whose labels sit on the field or on each other."""

    figure.canvas.draw()
    boxes = _text_boxes(figure, artists)
    axes_box = axes.get_window_extent(figure.canvas.get_renderer())
    report = []
    for name, box in boxes:
        outside = not (
            box.x0 < axes_box.x1
            and box.x1 > axes_box.x0
            and box.y0 < axes_box.y1
            and box.y1 > axes_box.y0
        )
        report.append(f"{name}_outside_field={outside}")
        if not outside:
            raise SystemExit(f"label {name} lies inside the poloidal field")
    for index, (name, box) in enumerate(boxes):
        for other_name, other in boxes[index + 1 :]:
            overlap = not (
                box.x1 <= other.x0
                or other.x1 <= box.x0
                or box.y1 <= other.y0
                or other.y1 <= box.y0
            )
            report.append(f"{name}_overlaps_{other_name}={overlap}")
            if overlap:
                raise SystemExit(f"labels {name} and {other_name} overlap")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--arm", default="default")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--branch-label", default="")
    arguments = parser.parse_args()

    data = np.load(arguments.artifact)
    flux = data[
        f"frames.{arguments.frame}.branches.{arguments.branch}"
        f".arms.{arguments.arm}._flux_grid"
    ]
    radius, height = data["radius"], data["height"]
    geometry = np.load(arguments.geometry)
    limiter = geometry["limiter"]
    surface = geometry["surface"]
    surface_found = bool(geometry["surface_found"])
    surface_margin = float(geometry["surface_class_margin"])
    surface_refusal = str(geometry["surface_refusal"])
    efit_boundary = geometry["efit_boundary"]
    efit_axis = np.asarray(geometry["efit_axis"], dtype=float)
    efit_x_point = np.asarray(geometry["efit_x_point"], dtype=float)
    record = json.loads(arguments.record.read_text())
    arm = record["frames"][str(arguments.frame)]["branches"][arguments.branch]["arms"][
        arguments.arm
    ]
    axis = np.asarray(arm["presence"]["axis_rz_m"], dtype=float)
    solved_x_point = arm.get("x_point_rz_m")
    qualified = bool(arm.get("qualified"))
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
        ring = np.vstack((surface, surface[:1])) if surface_found else surface
        axes.plot(
            ring[:, 0],
            ring[:, 1],
            color=REFERENCE,
            linewidth=1.6,
            linestyle="-" if surface_found else "--",
            zorder=3,
        )
    axes.plot(
        efit_boundary[:, 0],
        efit_boundary[:, 1],
        color=REFERENCE,
        linestyle="--",
        linewidth=2.4,
        zorder=3,
    )
    draw_wall(axes, limiter[:, 0], limiter[:, 1], color=LIMITER, linewidth=2.2)
    axes.set_aspect("equal")
    lower, upper = limiter.min(axis=0).astype(float), limiter.max(axis=0).astype(float)
    margin = 0.05 * (upper - lower)
    axes.set_xlim(lower[0] - margin[0], upper[0] + margin[0])
    axes.set_ylim(lower[1] - margin[1], upper[1] + margin[1])
    x_lower, x_upper = axes.get_xlim()
    y_lower, y_upper = axes.get_ylim()

    def _inside(point: np.ndarray | None) -> bool:
        if point is None:
            return False
        return bool(x_lower <= point[0] <= x_upper and y_lower <= point[1] <= y_upper)

    solved_point = (
        None if solved_x_point is None else np.asarray(solved_x_point, dtype=float)
    )
    solved_visible = _inside(solved_point)
    if solved_visible and not qualified:
        draw_nulls(
            axes,
            axis,
            None,
            other_x_points=solved_point.reshape(1, 2),
            style=DEFAULT_INK,
            contain=limiter,
        )
    else:
        draw_nulls(axes, axis, None, style=DEFAULT_INK, contain=limiter)
    if solved_visible and qualified:
        axes.plot(
            solved_point[0],
            solved_point[1],
            marker=DEFAULT_INK.xpoint_marker,
            markersize=DEFAULT_INK.xpoint_markersize,
            color=DEFAULT_INK.xpoint_color,
            linestyle="none",
            zorder=7,
        )

    axes.plot(
        efit_axis[0],
        efit_axis[1],
        marker="o",
        markersize=15,
        markerfacecolor="none",
        markeredgecolor=REFERENCE,
        markeredgewidth=2.0,
        linestyle="none",
        zorder=5,
    )
    axes.plot(
        efit_x_point[0],
        efit_x_point[1],
        marker="X",
        markersize=12,
        markerfacecolor="none",
        markeredgecolor=REFERENCE,
        markeredgewidth=2.2,
        linestyle="none",
        zorder=5,
    )

    axis_fraction = (axis[0] - x_lower) / (x_upper - x_lower)
    axis_note = axes.text(
        axis_fraction,
        -0.115,
        f"axis R={axis[0]:.2f} m, Z={axis[1]:.2f} m",
        transform=axes.transAxes,
        fontsize=14,
        color=DEFAULT_INK.axis_color,
        ha="center",
        va="top",
        clip_on=False,
        zorder=8,
    )
    axes.annotate(
        "",
        xy=(axis[0], axis[1]),
        xytext=(axis_fraction, -0.09),
        textcoords="axes fraction",
        zorder=8,
        arrowprops={
            "arrowstyle": "-",
            "color": DEFAULT_INK.axis_color,
            "linewidth": 1.4,
            "shrinkA": 0,
            "shrinkB": 8,
        },
    )
    if surface_found and len(surface):
        if np.isfinite(surface_margin):
            surface_text = (
                f"bounding surface: {len(surface)} vertices, "
                f"class margin {surface_margin:.3f}"
            )
        else:
            surface_text = (
                f"bounding surface: {len(surface)} vertices, class margin undefined"
            )
    else:
        surface_text = (
            f"no bounding surface: {surface_refusal or 'the read returned none'}"
        )

    handles = [
        Line2D([], [], color=DEFAULT_INK.flux_color, lw=2.6, label="solved flux"),
        Line2D([], [], color=LIMITER, lw=2.2, label="limiter"),
        Line2D([], [], color=REFERENCE, lw=1.6, label="bounding surface"),
        Line2D([], [], color=REFERENCE, ls="--", lw=2.4, label="EFIT boundary"),
        Line2D(
            [],
            [],
            color=DEFAULT_INK.axis_color,
            marker=DEFAULT_INK.axis_marker,
            markersize=DEFAULT_INK.axis_markersize,
            linestyle="none",
            label="solved axis",
        ),
    ]
    if solved_visible:
        handles.append(
            Line2D(
                [],
                [],
                color=DEFAULT_INK.xpoint_color,
                marker=DEFAULT_INK.xpoint_marker,
                markersize=DEFAULT_INK.xpoint_markersize,
                markerfacecolor="none",
                linestyle="none",
                label="solved X-point" if qualified else "solved X-point (unqualified)",
            )
        )
    handles += [
        Line2D(
            [],
            [],
            color=REFERENCE,
            marker="o",
            markersize=15,
            markerfacecolor="none",
            markeredgewidth=2.0,
            linestyle="none",
            label="EFIT axis",
        ),
        Line2D(
            [],
            [],
            color=REFERENCE,
            marker="X",
            markersize=12,
            markerfacecolor="none",
            markeredgewidth=2.2,
            linestyle="none",
            label="EFIT X-point",
        ),
    ]
    axes.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
        fontsize=13,
    )
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    legend = axes.get_legend()
    legend_box = legend.get_window_extent(renderer)
    axes_box = axes.get_window_extent(renderer)
    legend_bottom = (legend_box.y0 - axes_box.y0) / axes_box.height
    surface_note = axes.text(
        1.01,
        legend_bottom - 0.07,
        surface_text,
        transform=axes.transAxes,
        fontsize=13,
        color=REFERENCE,
        ha="left",
        va="top",
        clip_on=False,
        wrap=True,
        zorder=8,
    )
    report = _check_layout(
        figure,
        axes,
        [
            ("axis_note", axis_note),
            ("surface_note", surface_note),
            ("legend", legend),
        ],
    )
    report.append(f"legend_entries={len(handles)}")

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output, bbox_inches="tight")
    print(
        f"WROTE {arguments.output} residual={arm['terminal_residual']:.6g} "
        f"limiter_points={len(limiter)} on_bounding_box={frame_edge_points} "
        f"surface_points={len(surface)} surface_found={surface_found} "
        f"axis_gap_m={_gap(axis[None, :], limiter):.4f} "
        f"surface_gap_m="
        f"{_gap(surface, limiter) if len(surface) else float('nan'):.4f} "
        f"efit_axis={efit_axis[0]:.3f},{efit_axis[1]:.3f} "
        f"ef_xpoint={efit_x_point[0]:.3f},{efit_x_point[1]:.3f} "
        f"surface_note={surface_text!r} " + " ".join(report)
    )


if __name__ == "__main__":
    main()
