"""Draw the measured-current and scaled-current terminal flux of frame 137D.

The view spans the union of the physical first wall and the registered
pseudo-wall, so the physical vessel is drawn closed through draw_wall and the
pseudo-wall is drawn as a dotted reference. Both panels share one absolute level
array over the pseudo-wall, so the two terminal states are read on the same
contour values. Each panel draws its own magnetic axis and lower X-point, the
contour through that X-point as the arm's separatrix, the EFIT boundary and EFIT
lower X-point as the reference every offset is measured from, and its own
residual read from the stored arm record.
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
    inside_wall_units,
)
from scripts.prototypes.diiid_xpoint_offset.style import (  # noqa: E402
    REFERENCE,
    SOLVED,
    WALL,
)

FRAME = 137
LEVEL_COUNT = 22
PANELS = (
    ("base", "measured currents"),
    ("row", "coil currents scaled x1.1"),
)
FIGURE_WIDTH_IN = 14.0
FIGURE_HEIGHT_IN = 7.2
PANEL_HEIGHT_FRACTION = 0.74
VIEW_MARGIN_M = 0.06
COLUMN_GAP = 0.14
BOTTOM = 0.12


def _corpus_row():
    from benchmarks import diiid_forward_gs_match as bench

    paths = sorted(bench.DEFAULT_DATA.glob("*.parquet"))
    selected = bench.select_frames(
        paths, bench.EXECUTION_FRAME_COUNT, bench.polarity_population()
    )
    match = [item for item in selected if item.frame == FRAME]
    if len(match) != 1:
        raise RuntimeError(f"frame {FRAME} did not select exactly one corpus row")
    columns = (
        bench._LABEL_COLUMNS
        + bench._GEOMETRY_COLUMNS
        + bench._CURRENT_COLUMNS
        + bench._PLASMA_CURRENT_COLUMNS
    )
    return bench, bench._read(match[0].path, columns)


def _efit_boundary(row):
    count = int(row["efit_lcfs_n"][FRAME])
    return np.c_[
        np.asarray(row["efit_lcfs_r"][FRAME][:count], dtype=float),
        np.asarray(row["efit_lcfs_z"][FRAME][:count], dtype=float),
    ]


def _load(folder: Path, arm: str):
    data = np.load(folder / f"arm-{arm}.npz")
    return {name: np.asarray(data[name]) for name in data.files}


def _levels(base_flux, wall, radius, height):
    """Return one absolute level array over the base field inside the vessel.

    The saved flux is shaped (radius, height) and the painter wants
    (height, radius), so the caller transposes before asking for levels.
    """

    grid_r, grid_z = np.meshgrid(radius, height)
    inside = inside_wall_units(
        np.column_stack((grid_r.ravel(), grid_z.ravel())),
        wall,
    ).reshape(base_flux.shape)
    values = base_flux[inside]
    return np.linspace(float(np.min(values)), float(np.max(values)), LEVEL_COUNT)[1:-1]


def _flux_at(flux, radius, height, point):
    """Bilinear flux at a point, from a field shaped (radius, height)."""

    r = float(point[0])
    z = float(point[1])
    j = int(np.clip(np.searchsorted(radius, r) - 1, 0, radius.size - 2))
    i = int(np.clip(np.searchsorted(height, z) - 1, 0, height.size - 2))
    tr = (r - radius[j]) / (radius[j + 1] - radius[j])
    tz = (z - height[i]) / (height[i + 1] - height[i])
    f00, f01 = flux[j, i], flux[j + 1, i]
    f10, f11 = flux[j, i + 1], flux[j + 1, i + 1]
    return float(
        (1.0 - tz) * ((1.0 - tr) * f00 + tr * f01) + tz * ((1.0 - tr) * f10 + tr * f11)
    )


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
    radius = base["radius"]
    height = base["height"]
    pseudo_wall = base["wall"]
    levels = _levels(base["flux"].T, pseudo_wall, radius, height)
    bench, row = _corpus_row()
    boundary = _efit_boundary(row)
    vessel, _vessel_receipt = bench._physical_wall_ring(
        bench.DEFAULT_MACHINE_ARTIFACT_CACHE, bench.DEFAULT_MACHINE_ARTIFACT_DIGEST
    )
    extent_r = np.r_[vessel[:, 0], pseudo_wall[:, 0]]
    extent_z = np.r_[vessel[:, 1], pseudo_wall[:, 1]]
    r_lo = float(extent_r.min()) - VIEW_MARGIN_M
    r_hi = float(extent_r.max()) + VIEW_MARGIN_M
    z_lo = float(extent_z.min()) - VIEW_MARGIN_M
    z_hi = float(extent_z.max()) + VIEW_MARGIN_M
    span_ratio = (r_hi - r_lo) / (z_hi - z_lo)
    height_in = PANEL_HEIGHT_FRACTION * FIGURE_HEIGHT_IN
    width_fraction = height_in * span_ratio / FIGURE_WIDTH_IN
    height_fraction = PANEL_HEIGHT_FRACTION
    left = (0.5 - COLUMN_GAP / 2 - width_fraction, 0.5 + COLUMN_GAP / 2)

    figure = plt.figure(figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN), dpi=100)
    notes: list = [
        figure.text(
            0.5,
            0.99,
            "solid dark: physical first wall;  dotted grey: registered pseudo-wall",
            ha="center",
            va="top",
            fontsize=12,
            color=REFERENCE,
        )
    ]
    axes: list = []
    for index, (arm, label) in enumerate(PANELS):
        data = arms[arm]
        # The residual is the solve's own record of the arm, read from the
        # stored arm JSON so the footer, the table and the caption agree
        record = json.loads((args.input / f"arm-{arm}.json").read_text())
        # Offsets come from the stored null coordinates in metres, never from
        # the arm record's centimetre field, which reads ten times too large
        offset_cm = 100.0 * (
            np.asarray(data["lower_x"]) - np.asarray(data["efit_lower_x"])
        )
        separatrix = _flux_at(data["flux"], radius, height, data["lower_x"])
        item = figure.add_axes((left[index], BOTTOM, width_fraction, height_fraction))
        axes.append(item)
        draw_wall(
            item,
            units=None,
            radius=vessel[:, 0],
            height=vessel[:, 1],
            color=REFERENCE,
            linewidth=1.4,
        )
        item.plot(
            pseudo_wall[:, 0],
            pseudo_wall[:, 1],
            color=WALL,
            linewidth=1.0,
            linestyle=":",
            zorder=1,
        )
        item.plot(boundary[:, 0], boundary[:, 1], color=REFERENCE, linewidth=1.2)
        draw_flux_contours(
            item,
            radius,
            height,
            data["flux"].T,
            levels,
            color=SOLVED,
            wall=pseudo_wall,
            linewidth=1.1,
            alpha=0.55,
        )
        draw_flux_contours(
            item,
            radius,
            height,
            data["flux"].T,
            [separatrix],
            color=SOLVED,
            wall=pseudo_wall,
            linewidth=2.6,
        )
        draw_nulls(
            item,
            magnetic_axis=data["axis"],
            x_points=np.asarray(data["lower_x"])[None, :],
        )
        item.plot(
            float(data["efit_lower_x"][0]),
            float(data["efit_lower_x"][1]),
            marker="o",
            markersize=9,
            markerfacecolor="none",
            markeredgecolor=REFERENCE,
            markeredgewidth=1.3,
            linestyle="none",
        )
        item.set_xlim(r_lo, r_hi)
        item.set_ylim(z_lo, z_hi)
        item.set_aspect("equal")
        item.set_axis_off()
        notes.append(
            figure.text(
                left[index],
                BOTTOM + height_fraction + 0.02,
                label,
                va="bottom",
                ha="left",
                fontsize=14,
                color=SOLVED,
            )
        )
        notes.append(
            figure.text(
                left[index],
                BOTTOM - 0.055,
                "offset %.3f / %+.3f cm, %.3f cm from the EFIT null"
                % (
                    offset_cm[0],
                    offset_cm[1],
                    float(np.linalg.norm(offset_cm)),
                ),
                va="bottom",
                ha="left",
                fontsize=12,
                color=REFERENCE,
            )
        )
        notes.append(
            figure.text(
                left[index],
                BOTTOM - 0.100,
                "residual %.3e, from arm-%s.json" % (record["residual"], arm),
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
