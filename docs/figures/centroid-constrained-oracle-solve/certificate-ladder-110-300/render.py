"""Draw stored constrained Solovev states on shared physical contour levels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmarks import centroid_constrained_fixture_receipt as fixture
from benchmarks import solovev_certificate as certificate
from nova.jax.config import configure_dtypes
from nova.media import poloidal
from nova.media.ink import DEFAULT_INK, poloidal_axes


CASES = (
    "weak-rotation-reactor-static",
    "moderate-rotation-conventional-static",
    "strong-rotation-compact-static",
)
CELLS = (110, 300)
SUPPORTS = ("exact", "whole-cell")
FIGURES = Path(__file__).resolve().parents[1] / "cco-certificate-ladder-110-300"


def _draw(context: dict, receipt: dict, state: np.ndarray, path: Path) -> None:
    wall = np.asarray(context["machine"].wall_node, dtype=np.float64)
    radial, height, analytic = certificate._raster_field(
        context["coordinates"], context["analytic"], wall
    )
    _, _, solved = certificate._raster_field(context["coordinates"], state, wall)
    _, _, difference = certificate._raster_field(
        context["coordinates"], state - context["analytic"], wall
    )
    levels = poloidal.contour_levels(analytic, count=12)
    difference_extent = float(np.nanmax(np.abs(difference)))
    difference_levels = np.linspace(-difference_extent, difference_extent, 11)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    poloidal.draw_flux_contours(
        axes[0],
        radial,
        height,
        analytic,
        levels,
        color=fixture.ANALYTIC_INK,
        linewidth=2.4,
        wall=(wall,),
    )
    poloidal.draw_flux_contours(
        axes[0],
        radial,
        height,
        solved,
        levels,
        color=fixture.TERMINAL_INK,
        linewidth=3.0,
        wall=(wall,),
    )
    poloidal.draw_flux_contours(
        axes[1],
        radial,
        height,
        difference,
        difference_levels,
        color=fixture.TERMINAL_INK,
        linewidth=2.6,
        wall=(wall,),
    )
    analytic_axis = np.asarray(context["exact"].magnetic_axis, dtype=float)
    analytic_x = getattr(context["exact"], "x_point", None)
    solved_nulls = receipt["solve"]["topology"]
    for axis in axes:
        poloidal.draw_wall(axis, units=(wall,))
        poloidal.draw_nulls(
            axis,
            magnetic_axis=analytic_axis,
            x_points=analytic_x,
            style=DEFAULT_INK.variant(
                axis_marker="^",
                axis_color=fixture.ANALYTIC_INK,
                xpoint_color=fixture.ANALYTIC_INK,
            ),
            contain=(wall,),
        )
        poloidal.draw_nulls(
            axis,
            magnetic_axis=solved_nulls["axis_rz_m"],
            x_points=solved_nulls["x_point_rz_m"],
            style=DEFAULT_INK.variant(
                axis_marker="^",
                axis_color=fixture.TERMINAL_INK,
                xpoint_color=fixture.TERMINAL_INK,
            ),
            contain=(wall,),
        )
        poloidal_axes(axis)
    axes[0].text(
        0.03,
        0.95,
        "analytic blue · solved orange",
        transform=axes[0].transAxes,
        va="top",
        fontsize=20,
    )
    axes[1].text(
        0.03,
        0.95,
        "solved − analytic [Wb]",
        transform=axes[1].transAxes,
        va="top",
        fontsize=20,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipts", type=Path, required=True)
    args = parser.parse_args()
    configure_dtypes()
    for case in CASES:
        for cells in CELLS:
            carrier, _, exact = certificate._case(case, clip_mode="exact")
            machine = certificate._case_machine(
                case, carrier, exact, -cells, clip_mode="exact"
            )
            coordinates = np.vstack(
                (machine.node, machine.wall_node, machine.sample_coordinates)
            )
            context = {
                "machine": machine,
                "coordinates": coordinates,
                "analytic": certificate._exact_state(case, exact, coordinates),
                "exact": exact,
            }
            for support in SUPPORTS:
                stem = f"{case}-{cells}-{support}"
                receipt = json.loads((args.receipts / f"{stem}.json").read_text())
                state = np.load(args.receipts / f"{stem}.npy")
                assert state.shape == context["analytic"].shape
                assert receipt["realised_cells"] == len(context["machine"].node)
                _draw(context, receipt, state, FIGURES / f"{stem}.png")
                print(f"PANEL {stem}", flush=True)
    print("PANELS_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
