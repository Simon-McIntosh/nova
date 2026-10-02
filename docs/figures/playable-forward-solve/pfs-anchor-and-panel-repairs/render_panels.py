"""Re-render the limited-anchor census and early-frame-placement panels.

Every panel is drawn from a committed receipt (and, for the limited-anchor
census, the atlas-supplement flux raster persisted beside its own panel). No
solve runs: the receipt carries the state, and the panel states the state it
draws.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from nova.media import poloidal  # noqa: E402
from nova.media.ink import poloidal_axes  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
LIMITED = ROOT / "docs/figures/playable-forward-solve/limited-anchor"
SUPPLEMENT = (
    ROOT / "docs/figures/null-identification-authority/convergence-atlas/supplement"
)
EARLY = ROOT / "docs/figures/playable-forward-solve/early-frame-placement"

GREY = "#b8c2cc"
REFUSED = "#996a1f"
FREE = "#cc3344"
COLD = "#008b72"
WARM = "#7d3cff"


def _normalised_level(level: float, axis_flux: float, x_flux: float) -> float:
    """Fraction of the axis-to-X flux span at which a drawn level sits."""
    return float((level - axis_flux) / (x_flux - axis_flux))


def render_limited_anchor() -> list[Path]:
    receipt = json.loads((LIMITED / "limited-anchor-containment.json").read_text())
    shot = receipt["shot"]
    written: list[Path] = []
    for frame in sorted(receipt["frames"], key=lambda item: item["manifest_row"]):
        row = int(frame["manifest_row"])
        operands = np.load(SUPPLEMENT / f"27079-{row:02d}-operands.npz")
        radius = operands["radius"]
        height = operands["height"]
        raster = operands["raster"]
        wall = operands["wall"]
        public = frame["public_read"]
        axis_flux = float(public["axis_flux_wb"])
        boundary_flux = float(public["boundary_flux_wb"])
        x_flux = float(public["x_point_flux_wb"])
        converged = bool(frame["converged"])
        closed = bool(public["contour_closed"])
        anchor_position = np.asarray(
            public["selected_anchor_node_position_m"], dtype=float
        )
        boundary_position = np.asarray(public["boundary_position_m"], dtype=float)

        figure, axes = plt.subplots(figsize=(4.8, 4.8))
        poloidal_axes(axes)
        levels = np.linspace(x_flux, axis_flux, 11)[1:-1]
        poloidal.draw_flux_contours(
            axes, radius, height, raster, levels, color=GREY, linewidth=0.4
        )
        poloidal.draw_wall(axes, wall[:, 0], wall[:, 1], color="#202020")
        poloidal.draw_nulls(
            axes,
            np.asarray(public["axis_position_m"], dtype=float),
            np.atleast_2d(np.asarray(public["x_point_position_m"], dtype=float)),
            contain=wall,
        )
        candidates = np.asarray(
            [candidate["node_position_m"] for candidate in frame["candidates"]],
            dtype=float,
        )
        axes.scatter(
            candidates[:, 0],
            candidates[:, 1],
            marker="D",
            s=24,
            facecolors="none",
            edgecolors="#3366cc",
            linewidths=0.9,
            zorder=8,
        )

        if converged and closed:
            axes.scatter(
                *anchor_position,
                marker="o",
                s=60,
                facecolors="none",
                edgecolors=FREE,
                linewidths=1.4,
                zorder=9,
            )
            poloidal.draw_flux_contours(
                axes,
                radius,
                height,
                raster,
                [boundary_flux],
                color=FREE,
                linewidth=2.2,
            )
            separation_mm = 1e3 * float(
                np.hypot(*(anchor_position - boundary_position))
            )
            level_fraction = _normalised_level(boundary_flux, axis_flux, x_flux)
            caption = (
                f"MAST {shot}, row {row}, t = {1e3 * frame['time_s']:.0f} ms — "
                f"converged\n"
                f"public contour closed; drawn boundary level psi_N = "
                f"{level_fraction:.4f} of axis->X\n"
                f"anchor->boundary separation {separation_mm:.1f} mm"
            )
        else:
            selected = public["selected_anchor_node"]
            trace_error = next(
                (
                    candidate["trace_error"]
                    for candidate in frame["candidates"]
                    if candidate["node"] == selected and candidate.get("trace_error")
                ),
                "no closed contour",
            )
            axes.scatter(
                *anchor_position,
                marker="x",
                s=95,
                color=REFUSED,
                linewidths=2.2,
                zorder=10,
            )
            axes.text(
                0.5,
                0.04,
                "REFUSED: no closed solved contour",
                transform=axes.transAxes,
                ha="center",
                color=REFUSED,
                fontsize=9,
                fontweight="bold",
            )
            caption = (
                f"MAST {shot}, row {row}, t = {1e3 * frame['time_s']:.1f} ms — "
                f"converged=False\n"
                f"contour closed=False; max containment "
                f"{frame['maximum_containment_fraction']:.3f}\n"
                f"{trace_error[:56]}"
            )
        axes.set_title(caption, fontsize=8.5)
        axes.set_xlim(
            float(np.min(wall[:, 0]) - 0.08), float(np.max(wall[:, 0]) + 0.08)
        )
        axes.set_ylim(
            float(np.min(wall[:, 1]) - 0.08), float(np.max(wall[:, 1]) + 0.08)
        )

        panel = LIMITED / f"row-{row:02d}-{1e3 * frame['time_s']:.0f}ms.png"
        figure.savefig(panel, dpi=180, bbox_inches="tight")
        plt.close(figure)
        written.append(panel)
    return written


def _arm_summary(arm: str, entry: dict) -> str:
    ratio = entry.get("enclosed_area_ratio")
    ratio_text = "—" if ratio is None else f"{ratio:.3f}"
    return f"{_ARM_CODE[arm]} n{entry.get('first_contact_node')} r{ratio_text}"


_ARM_CODE = {"free": "free", "conditioned": "cold", "conditioned_warm": "warm"}


def render_early_frame_placement() -> Path:
    receipt = json.loads((EARLY / "early-frame-placement.json").read_text())
    rows = sorted(receipt["rows"], key=lambda item: item["manifest_row"])
    columns = 4
    counts = len(rows)
    grid_rows = int(np.ceil(counts / columns))
    figure, axes = plt.subplots(
        grid_rows, columns, figsize=(4.6 * columns, 4.6 * grid_rows)
    )
    figure.subplots_adjust(hspace=0.75, wspace=0.05, top=0.95, bottom=0.02)
    flat = np.atleast_1d(axes).ravel()
    for slot, frame in enumerate(rows):
        panel = flat[slot]
        poloidal_axes(panel)
        wall = np.asarray(frame["geometry"]["wall"], dtype=float)
        poloidal.draw_wall(panel, wall[:, 0], wall[:, 1], color="#202020")
        reconstruction = np.asarray(
            frame["geometry"]["reconstruction_boundary"], dtype=float
        )
        poloidal.draw_boundary(
            panel,
            reconstruction[:, 0],
            reconstruction[:, 1],
            color="#222222",
            linestyle="--",
            linewidth=1.1,
        )
        for arm, color in (
            ("free", FREE),
            ("conditioned", COLD),
            ("conditioned_warm", WARM),
        ):
            entry = frame.get(arm)
            if entry is None or entry.get("solve_error"):
                continue
            if entry.get("converged") and entry.get("contour_closed"):
                poloidal.draw_boundary(
                    panel,
                    entry["contour_r"],
                    entry["contour_z"],
                    color=color,
                    linewidth=2.0,
                )
                panel.plot(
                    entry["axis_r_m"],
                    entry["axis_z_m"],
                    marker="o",
                    markersize=6,
                    color=color,
                    markeredgecolor="black",
                    markeredgewidth=0.5,
                    zorder=9,
                )
            elif entry.get("contour_closed"):
                poloidal.draw_boundary(
                    panel,
                    entry["contour_r"],
                    entry["contour_z"],
                    color=REFUSED,
                    linestyle=(0, (4, 3)),
                    linewidth=2.0,
                )
                panel.plot(
                    entry["axis_r_m"],
                    entry["axis_z_m"],
                    marker="x",
                    markersize=9,
                    color=REFUSED,
                    zorder=10,
                )
            else:
                panel.plot(
                    entry["axis_r_m"],
                    entry["axis_z_m"],
                    marker="x",
                    markersize=8,
                    color=REFUSED,
                    zorder=10,
                )
        panel.plot(
            frame["efit_rmaxis_m"],
            frame["efit_zmaxis_m"],
            marker="+",
            markersize=11,
            color="#202020",
            markeredgewidth=1.6,
            zorder=9,
        )
        arms = [
            arm
            for arm in ("free", "conditioned", "conditioned_warm")
            if frame.get(arm) is not None
        ]
        unconverged = [
            _ARM_CODE[arm] for arm in arms if not frame[arm].get("converged", False)
        ]
        summary = "  ".join(_arm_summary(arm, frame[arm]) for arm in arms)
        refused_line = (
            "unconverged arms: " + ", ".join(unconverged)
            if unconverged
            else "all arms converged"
        )
        panel.set_title(
            f"row {frame['manifest_row']}, t = {1e3 * frame['time_s']:.0f} ms, "
            f"{frame['requested_class']}\n{summary}\n{refused_line}",
            fontsize=7.8,
        )
        panel.set_xlim(
            float(np.min(wall[:, 0]) - 0.10), float(np.max(wall[:, 0]) + 0.10)
        )
        panel.set_ylim(
            float(np.min(wall[:, 1]) - 0.10), float(np.max(wall[:, 1]) + 0.10)
        )
    for slot in range(counts, flat.size):
        flat[slot].set_visible(False)
    path = EARLY / "early-frame-placement.png"
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)
    return path


if __name__ == "__main__":
    for produced in render_limited_anchor():
        print("wrote", produced.relative_to(ROOT))
    print("wrote", render_early_frame_placement().relative_to(ROOT))
