"""Overlay Nova's solved flux map on the EFIT reconstruction it reproduces.

One poloidal panel: both maps drawn as unfilled contours at ONE shared array
of absolute flux levels, both boundaries drawn, over the coils and first wall.
The two right-hand panels quantify the agreement the left panel asserts --
the signed difference along the midplane, and its distribution over every grid
point inside the limiter.

Three choices make the comparison honest rather than flattering.

The level array is computed once from the reference map and handed to both,
so neither is renormalised to suit the other. Independent per-map levels can
make any two maps look alike, which is why
:func:`nova.media.poloidal.draw_flux_contours` refuses to default them.

The grid is wide enough to enclose the conductors -- R 0.02 to 2.20 m and Z
plus or minus 2.60 m at 150 by 350 -- because a map cropped to the
reconstruction's own 65 by 65 domain cannot show whether the solve holds where
the coils are. Nova's map is evaluated there through an ordinary CoilSet grid
solve, and EFIT's is bicubically interpolated onto the same points.

Neither map is regauged. Both carry Nova's total poloidal flux in Wb, so the
difference is a physical disagreement rather than an offset, and the receipt
records the maximum and RMS over the limiter interior alongside the flux span
they should be judged against.

Two things the comparison must not do, both of which it did on the first
attempt. It must not compare outside the reference's OWN domain: EFIT's grid
stops at R 2.00 m and Z plus or minus 2.00 m, and the bicubic interpolation
happily extrapolates past that, which produced an 18 mWb excursion at the
machine axis and a 13 mWb one past the outboard wall that were entirely
artefacts of the spline. Everything quantified here is masked to the stored
grid, and the limiter interior sits well inside it.

And it must not draw Nova's boundary from the solve's separatrix vertex list.
That list is the separatrix INCLUDING its open legs traced to the domain edge
-- 138 vertices with a median step of 0.15 m but a maximum of 1.11 m, reaching
the grid's own Z limit -- so joining it as one closed polyline draws a zigzag
rather than a boundary. Nova's boundary is instead contoured from Nova's own
map at the boundary flux level, by the same marching squares that draws every
other level, so it cannot disagree with the surfaces around it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from nova.media import poloidal as pol, traces as tr
from nova.media.ink import DEFAULT_INK
from nova.media.layout import three_view
from nova.media.sources.mast_efit import read_pulse


def _reference_span(shot: int) -> tuple[float, float, float, float]:
    """Return the stored EFIT grid's own bounds, outside which it is no data."""
    import zarr

    from nova.imas.mast_vacuum_cohort import SHOT_STORE

    group = zarr.open_group(str(Path(SHOT_STORE) / f"{shot}.zarr"), mode="r")["efm"]
    radius = np.asarray(group["gridr"], dtype=float)
    height = np.asarray(group["gridz"], dtype=float)
    return (
        float(radius.min()),
        float(radius.max()),
        float(height.min()),
        float(height.max()),
    )


DEFAULT_SOLVE = (
    Path.home()
    / ".config/reckon/crew/reports/nova/presentation-media/mast-21978-wide-solve.npz"
)
DEFAULT_RECEIPT = DEFAULT_SOLVE.with_name("wide-solve-receipt.json")
DEFAULT_OUTPUT = Path("docs/figures/presentation-media")

#: Sienna, imas-ink's reference-underlay colour: the reconstruction is the
#: reference and Nova's map is the thing under test, so they must not share a
#: hue even when they agree.
REFERENCE_COLOR = "#b85c38"


def render_overlay(
    solve: Path = DEFAULT_SOLVE,
    receipt_path: Path = DEFAULT_RECEIPT,
    output: Path = DEFAULT_OUTPUT,
    levels: int = 28,
    height: float = 7.0,
) -> dict[str, object]:
    """Write the overlay figure and return its receipt."""
    style = DEFAULT_INK
    carrier = np.load(solve, allow_pickle=True)
    solved = json.loads(Path(receipt_path).read_text())

    radius = np.asarray(carrier["radial_m"], dtype=float)
    vertical = np.asarray(carrier["vertical_m"], dtype=float)
    # The carrier stores (radial, vertical); a media frame wants (height, radius).
    nova_flux = np.asarray(carrier["total_poloidal_flux_wb_rz"], dtype=float).T
    efit_flux = np.asarray(carrier["stored_efit_flux_wb_rz"], dtype=float).T
    inside = np.asarray(carrier["limiter_interior_mask_rz"], dtype=bool).T
    axis = np.asarray(carrier["magnetic_axis_rz_m"], dtype=float)
    x_points = np.atleast_2d(np.asarray(carrier["x_points_rz_m"], dtype=float))

    shot = int(solved["shot"])
    time = float(solved["selected_time_s"])
    pulse = read_pulse(shot, times=[time])
    machine = pulse.geometry
    reference = pulse.frames[0]

    boundary_flux = float(solved["topology"]["boundary_flux_wb"])
    # One level array, from the reference, handed to both maps.
    level_array = pol.contour_levels(efit_flux, levels, boundary=boundary_flux)

    # The reference is only data inside its own stored grid; beyond it the
    # interpolation extrapolates, so nothing outside is compared.
    reference_span = _reference_span(shot)
    within = (
        (radius[None, :] >= reference_span[0])
        & (radius[None, :] <= reference_span[1])
        & (vertical[:, None] >= reference_span[2])
        & (vertical[:, None] <= reference_span[3])
    )

    view = three_view(extent=pulse.extent(), height=height, style=style)
    view.clear()

    pol.draw_flux_contours(
        view.poloidal,
        radius,
        vertical,
        efit_flux,
        level_array,
        color=REFERENCE_COLOR,
        linewidth=0.6,
    )
    pol.draw_flux_contours(
        view.poloidal, radius, vertical, nova_flux, level_array, color=style.flux_color
    )
    pol.draw_coils(view.poloidal, machine.coils)
    pol.draw_wall(view.poloidal, *machine.limiter.T)
    pol.draw_boundary(
        view.poloidal,
        *reference.boundary.T,
        color=REFERENCE_COLOR,
        linestyle="dashed",
        linewidth=1.6,
    )
    # Nova's boundary as a contour of Nova's own map at the boundary level,
    # not as the separatrix vertex list, which carries open legs.
    pol.draw_flux_contours(
        view.poloidal,
        radius,
        vertical,
        nova_flux,
        [boundary_flux],
        color=style.separatrix_color,
        linewidth=style.separatrix_linewidth,
        zorder=style.zorder_separatrix,
    )
    nulls = pol.draw_nulls(
        view.poloidal,
        magnetic_axis=axis,
        x_points=x_points,
        contain=machine.limiter,
    )

    difference = np.where(within, nova_flux - efit_flux, np.nan)
    midplane = int(np.argmin(np.abs(vertical)))
    span = abs(
        float(solved["topology"]["axis_flux_wb"])
        - float(solved["topology"]["boundary_flux_wb"])
    )

    tr.draw_trace(
        view.upper, radius, 1e3 * difference[midplane], color=style.flux_color
    )
    view.upper.axhline(0.0, color=style.contour_color, linewidth=0.5)
    tr.label_axes(view.upper, None, r"nova $-$ EFIT at $Z\approx0$  [mWb]", style=style)
    view.upper.set_xlim(reference_span[0], reference_span[1])

    interior = difference[inside & within]
    if not np.all(np.isfinite(interior)):
        raise ValueError("the limiter interior must lie inside the reference grid")
    view.lower.hist(
        1e3 * interior, bins=60, color=style.flux_color, alpha=0.85, log=True
    )
    view.lower.axvline(0.0, color=style.contour_color, linewidth=0.5)
    tr.label_axes(view.lower, r"nova $-$ EFIT inside the limiter  [mWb]", "grid points")

    tr.annotate_time(
        view.upper,
        f"MAST {shot}   t = {1e3 * time:.0f} ms\n"
        f"free solve, residual {float(solved['solve']['terminal_residual']):.1e}\n"
        f"max |$\\Delta$| = {1e3 * np.max(np.abs(interior)):.1f} mWb "
        f"({100 * np.max(np.abs(interior)) / span:.1f}% of span)",
    )

    output.mkdir(parents=True, exist_ok=True)
    name = f"mast-{shot}-efit-nova-overlay"
    figure_path = output / f"{name}.png"
    view.figure.savefig(figure_path, dpi=style.figure_dpi)

    receipt = {
        "figure": str(figure_path),
        "shot": shot,
        "time_s": time,
        "grid_shape_zr": list(nova_flux.shape),
        "grid_radial_span_m": [float(radius.min()), float(radius.max())],
        "grid_vertical_span_m": [float(vertical.min()), float(vertical.max())],
        "coil_sections_enclosed": len(machine.coils),
        "contour_levels": len(level_array),
        "levels_from": "the EFIT reference map, shared with the nova map",
        "level_span_wb": [float(level_array[0]), float(level_array[-1])],
        "flux_span_axis_to_boundary_wb": span,
        "interior_point_count": int(inside.sum()),
        "maximum_absolute_difference_wb": float(np.max(np.abs(interior))),
        "rms_difference_wb": float(np.sqrt(np.mean(interior**2))),
        "maximum_difference_fraction_of_span": float(np.max(np.abs(interior)) / span),
        "regauged": False,
        "reference_grid_span_rz_m": list(reference_span),
        "comparison_masked_to_reference_grid": True,
        "nova_boundary_drawn_as": (
            "contour of the nova map at the boundary flux level; the solve's "
            "separatrix vertex list carries open legs and is not a polyline"
        ),
        "solve": {
            key: solved["solve"][key]
            for key in ("qualified", "converged", "terminal_residual", "conditioned")
        },
        "boundary_source_nova": solved["topology"]["boundary_source"],
        "boundary_source_reference": "efm/lcfs",
        "nulls": nulls,
        "carrier": str(solve),
        "carrier_sha256": solved.get("array_sha256"),
    }
    (output / f"{name}-receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True, default=str)
    )
    return receipt


def main(argv: list[str] | None = None) -> int:
    """Render the EFIT-versus-Nova overlay from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solve", type=Path, default=DEFAULT_SOLVE)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--levels", type=int, default=28)
    parser.add_argument("--height", type=float, default=7.0)
    arguments = parser.parse_args(argv)
    receipt = render_overlay(
        solve=arguments.solve,
        receipt_path=arguments.receipt,
        output=arguments.output,
        levels=arguments.levels,
        height=arguments.height,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
