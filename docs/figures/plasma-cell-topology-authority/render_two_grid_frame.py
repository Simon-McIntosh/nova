"""Draw the two-grid MAST frame under the adopted production topology read.

One panel per grid, both on ONE physical level array spanning the two maps.
Independent levels would let two maps of the same field be made to look like
anything, and the question here is whether the read lands on the same saddle
at both resolutions.

The nulls drawn are the read's own output, not the archived pre-adoption set:
the magnetic axis and the admitted saddle come from the replay receipt, so a
panel shows what the read admitted rather than what an earlier read stored.
The reference X-point of the store is drawn hollow in the shared vocabulary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from nova.equilibrium.wall_mask import WallUnit
from nova.media import poloidal
from nova.media.ink import DEFAULT_INK, poloidal_axes


HERE = Path(__file__).resolve().parent
RECEIPT = HERE / "two-grid-frame.json"
EDIT_INDEX = 10
PANELS = ("stride_33x33", "full_axes_65x65")
LEVEL_COUNT = 26


def wall_units(data) -> tuple[WallUnit, ...]:
    offsets = np.asarray(data["wall_offsets"], dtype=int)
    coordinate = np.asarray(data["wall_coordinate"], dtype=float).reshape(-1, 2)
    closed = np.asarray(data["wall_closed"], dtype=bool)
    kinds = [str(value) for value in np.asarray(data["wall_kinds"])]
    return tuple(
        WallUnit(
            coordinate[start:stop, 0],
            coordinate[start:stop, 1],
            kind=kinds[index],
            closed=bool(closed[index]),
        )
        for index, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:]))
    )


def panel(axes, data, row, levels) -> None:
    poloidal_axes(axes)
    units = wall_units(data)
    radius = np.asarray(data["radius"], dtype=float)
    height = np.asarray(data["height"], dtype=float)
    flux = np.asarray(data[f"psi_{EDIT_INDEX}"], dtype=float)
    poloidal.draw_wall(axes, units=units)
    poloidal.draw_flux_contours(axes, radius, height, flux, levels)
    poloidal.draw_nulls(
        axes,
        other_x_points=np.asarray(row["reference_x_point"], dtype=float)[None, :],
        contain=units,
    )
    axis = np.asarray(row["admitted_axis"], dtype=float).reshape(-1)[:2]
    saddle = np.asarray(row["admitted_x_point"], dtype=float).reshape(-1)[:2]
    poloidal.draw_nulls(
        axes,
        magnetic_axis=axis[None, :] if np.all(np.isfinite(axis)) else None,
        x_points=saddle[None, :],
        contain=units,
    )
    axes.set_title(
        "%s, %d cells" % (row["grid"], row["realised_cells"]), fontsize=9, pad=6
    )
    axes.set_xlim(float(radius.min()), float(radius.max()))
    axes.set_ylim(float(height.min()), float(height.max()))


def render(figure_path: Path) -> None:
    document = json.loads(RECEIPT.read_text(encoding="utf-8"))
    rows = [row for row in document["rows"] if row["grid"] in PANELS]
    states = [np.load(row["state_path"], allow_pickle=True) for row in rows]
    maps = [np.asarray(state[f"psi_{EDIT_INDEX}"], dtype=float) for state in states]
    levels = poloidal.contour_levels(
        np.concatenate([item.ravel() for item in maps]), LEVEL_COUNT
    )
    figure, axes = plt.subplots(
        1, 2, figsize=(9.6, 6.6), facecolor=DEFAULT_INK.figure_facecolor
    )
    for axis, state, row in zip(axes, states, rows, strict=True):
        panel(axis, state, row, levels)
    caption = "\n".join(
        "%s: admitted saddle (%.5f, %.5f) at %.1f cell pitches from the "
        "reference x-point"
        % (
            row["grid"],
            row["admitted_x_point"][0],
            row["admitted_x_point"][1],
            row["admitted_distance_to_reference_pitch"],
        )
        for row in rows
    )
    figure.suptitle(
        "the adopted topology read on one matched edit at two resolutions, "
        "%d shared levels\nsolid triangle: admitted axis.   filled cross: the "
        "admitted saddle.   hollow cross: the reference x-point." % levels.size,
        fontsize=9,
    )
    figure.text(
        0.5,
        0.005,
        "%s\nthe same reference field on both lattices is separately receipted: the"
        " read\n"
        "lands the saddle 0.25 and 0.12 of a cell pitch from the reference X-point,\n"
        "so the grid does not by itself move the admission." % caption,
        ha="center",
        va="bottom",
        fontsize=8,
    )
    figure.subplots_adjust(top=0.88, bottom=0.16)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, dpi=200, bbox_inches="tight")
    figure.savefig(figure_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)
    print("TWO_GRID_FIGURE=%s levels=%d" % (figure_path, levels.size), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", type=Path, default=HERE / "two-grid-frame.png")
    render(parser.parse_args().figure)


if __name__ == "__main__":
    main()
