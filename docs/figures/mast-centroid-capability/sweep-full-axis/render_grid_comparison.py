"""Draw one matched sweep edit at both grids, from the persisted panel states.

Edit index 10 is the matched pair: the same coil and the same two-percent
fraction converges on the full stored axes and fails at the stored-axis
stride. It sits closest to the base frame of the three matched pairs, so what
differs between the panels is the grid rather than the size of the excursion.

Both panels are contoured on ONE physical level array spanning the two maps.
Two grids drawn on independently chosen levels can be made to look like
anything, and the whole point of the pair is that the field is nearly the same
while the solve closes on one grid and not on the other.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import jax.numpy as jnp

from nova.equilibrium.separatrix_branches import assemble_separatrix_branches
from nova.equilibrium.wall_mask import WallUnit
from nova.media import poloidal
from nova.media.ink import DEFAULT_INK, poloidal_axes


ROOT = Path(__file__).resolve().parents[4]
STRIDE_STATES = (
    ROOT / "docs/figures/forward-solve-api/coil-edit-nonconvergence/panel-states.npz"
)
STRIDE_RECEIPT = (
    ROOT / "docs/figures/forward-solve-api/coil-edit-latency/coil-edit-latency.json"
)
FULL_STATES = Path(__file__).resolve().parent / "panel-states.npz"
FULL_RECEIPT = Path(__file__).resolve().parent / "coil-edit-latency.json"
DEFAULT_FIGURE = Path(__file__).resolve().parent / "matched-edit-both-grids.png"
EDIT_INDEX = 10
LEVEL_COUNT = 26


def _wall_units(data) -> tuple[WallUnit, ...]:
    """Rebuild the vessel units the archive stored flat."""
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


def _edit_row(receipt: Path, index: int) -> dict:
    """Return the receipt row of one edit, for its residual and flag."""
    document = json.loads(receipt.read_text(encoding="utf-8"))
    for row in document["edits"]:
        if int(row["edit_index"]) == index:
            return row
    raise KeyError(f"edit {index} is absent from {receipt}")


def _assembled_boundary(data, index: int) -> np.ndarray:
    """Return the axis-enclosing lobe of the map at its admitted saddle.

    The level is read from the map itself at the admitted saddle position
    rather than from a stored scalar, so the two archives are treated
    identically even though only one of them persisted a saddle flux.
    """
    radius = np.asarray(data["radius"], dtype=float)
    height = np.asarray(data["height"], dtype=float)
    flux = np.asarray(data[f"psi_{index}"], dtype=float)
    axis = np.asarray(data[f"axis_{index}"], dtype=float).reshape(-1)[:2]
    solved = np.asarray(data[f"xpoints_{index}"], dtype=float).reshape(-1, 2)
    saddle = (
        int(np.asarray(data[f"saddle_index_{index}"]))
        if f"saddle_index_{index}" in data.files
        else 0
    )
    if not solved.size or not np.all(np.isfinite(solved[saddle])):
        return np.empty((0, 2))
    point = solved[saddle]
    # Bilinear read of the map at the saddle: the level that lobe sits on.
    level = float(
        np.interp(
            point[0],
            radius,
            [
                np.interp(point[1], height, flux[:, column])
                for column in range(radius.size)
            ],
        )
    )
    assembled = assemble_separatrix_branches(
        jnp.asarray(flux),
        jnp.asarray(radius),
        jnp.asarray(height),
        jnp.asarray(level),
        jnp.asarray(axis),
    )
    return poloidal.sample_cubic_controls(
        np.asarray(assembled["closed_controls_rz"], float),
        np.asarray(assembled["closed_valid"], bool),
        12,
    )


def _panel(axes, data, index: int, title: str) -> None:
    """Draw one grid's state with its own and its reference landmarks."""
    poloidal_axes(axes)
    units = _wall_units(data)
    radius = np.asarray(data["radius"], dtype=float)
    height = np.asarray(data["height"], dtype=float)
    flux = np.asarray(data[f"psi_{index}"], dtype=float)
    poloidal.draw_wall(axes, units=units)
    poloidal.draw_flux_contours(axes, radius, height, flux, _panel.levels)
    # The archived ``separatrix_N`` is the RAW level set: unordered segments,
    # unsplit and unbounded. Drawing it as a polyline sprays chords across the
    # panel. Assemble the map's own level set at the admitted saddle instead,
    # which is the curve the painter was written for.
    boundary = _assembled_boundary(data, index)
    if boundary.shape[0] >= 3:
        poloidal.draw_boundary(axes, boundary[:, 0], boundary[:, 1])
    # The reference nulls first and hollow, so the solved set reads as the
    # answer and the reference as what it is being compared against.
    poloidal.draw_nulls(
        axes,
        other_x_points=np.asarray(data["reference_xpoints"], dtype=float).reshape(
            -1, 2
        ),
        contain=units,
    )
    solved = np.asarray(data[f"xpoints_{index}"], dtype=float).reshape(-1, 2)
    saddle = (
        int(np.asarray(data[f"saddle_index_{index}"]))
        if (f"saddle_index_{index}" in data.files)
        else 0
    )
    poloidal.draw_nulls(
        axes,
        magnetic_axis=np.asarray(data[f"axis_{index}"], dtype=float),
        x_points=solved[saddle : saddle + 1] if solved.size else None,
        other_x_points=(
            np.delete(solved, saddle, axis=0) if solved.shape[0] > 1 else None
        ),
        contain=units,
    )
    axes.set_title(title, fontsize=8)
    axes.set_xlim(float(radius.min()), float(radius.max()))
    axes.set_ylim(float(height.min()), float(height.max()))


def render(figure_path: Path) -> None:
    """Draw the matched edit at both grids on one shared level array."""
    stride = np.load(STRIDE_STATES, allow_pickle=True)
    full = np.load(FULL_STATES, allow_pickle=True)
    stride_row = _edit_row(STRIDE_RECEIPT, EDIT_INDEX)
    full_row = _edit_row(FULL_RECEIPT, EDIT_INDEX)
    maps = (
        np.asarray(stride[f"psi_{EDIT_INDEX}"], dtype=float),
        np.asarray(full[f"psi_{EDIT_INDEX}"], dtype=float),
    )
    _panel.levels = poloidal.contour_levels(
        np.concatenate([item.ravel() for item in maps]), LEVEL_COUNT
    )
    figure, axes = plt.subplots(
        1, 2, figsize=(9.2, 6.4), facecolor=DEFAULT_INK.figure_facecolor
    )
    _panel(
        axes[0],
        stride,
        EDIT_INDEX,
        "stored-axis stride, 33 x 33\n"
        f"residual {stride_row['terminal_residual']:.3e}, "
        f"converged {bool(stride_row['converged'])}",
    )
    _panel(
        axes[1],
        full,
        EDIT_INDEX,
        "full stored axes, 65 x 65\n"
        f"residual {full_row['terminal_residual']:.3e}, "
        f"converged {bool(full_row['converged'])}",
    )
    figure.suptitle(
        f"one matched edit, index {EDIT_INDEX} at "
        f"{float(stride['edit_fraction'][EDIT_INDEX]):+.2f} of the p4_upper "
        f"current, on {_panel.levels.size} shared levels",
        fontsize=9,
    )
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, dpi=200, bbox_inches="tight")
    figure.savefig(figure_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)
    print(
        "MATCHED_FIGURE=%s levels=%d stride_converged=%s full_converged=%s"
        % (
            figure_path,
            _panel.levels.size,
            bool(stride_row["converged"]),
            bool(full_row["converged"]),
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    render(parser.parse_args().figure)


if __name__ == "__main__":
    main()
