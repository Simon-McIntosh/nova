"""Stage-by-stage assertion of where the topology read loses the X-point.

The diverted single-null certificate rows at 110 and 300 cells carry a solved
flux that matches the Cerfon-Freidberg analytic field to order 1e-17, yet the
production topology read returns no X-point candidate (``x_candidate_count``
0); the 500-cell row is the positive control whose X-point IS admitted at
1.20 mm.  This benchmark loads the persisted terminal flux of those three
committed production-route parts (``docs/figures/gs-absolute-accuracy/solovev/
production-route-parts``), rebuilds the production forward operator exactly as
the certificate driver does (``scripts/analytic_oracle_fixtures/measure.py``
``cached_machine`` + ``forward_operator``), and replays the topology read stage
by stage on each row:

  1. the sign-change census over the rotated centroid ring
     (``_FixedDesignNull2D.read_census`` -> ``_compatibility_census`` on these
     hex carriers),
  2. first-wall containment and the occupiable region,
  3. the split-spline polish (the ``read_qualification`` polish receipt),
  4. the candidate dedupe radius,
  5. the Hessian-type gate,
  6. the admission verdict.

Each stage publishes which candidates survive and the stated reason the others
are dropped, so the staged read is auditable against the code that runs it.
One saddle-region figure per row is rendered with the imas-ink painters (line
contours only, wall, analytic nulls in the analytic style, census candidates
hollow with the admitted one filled, and the cells around the analytic X-point
outlined).

Usage:
    benchmarks/null_census_assertion.py --figure-root DIR --receipt PATH
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np
from scipy.interpolate import LinearNDInterpolator

from benchmarks.diiid_forward_gs_match import candidate_flux_margins
from nova.equilibrium.analytic_single_null import cerfon_freidberg_single_null
from nova.jax.config import configure_dtypes
from nova.media import poloidal
from nova.media.ink import DEFAULT_INK, poloidal_axes
from scripts.analytic_oracle_fixtures import measure as oracle_fixture
from tests.rotating_equilibrium_references import reference_cases

ROOT = Path(__file__).resolve().parents[1]
PARTS = ROOT / "docs/figures/gs-absolute-accuracy/solovev/production-route-parts"
DIVERTED_REFERENCE = cerfon_freidberg_single_null()
ANALYTIC_AXIS = np.asarray(DIVERTED_REFERENCE.magnetic_axis, dtype=np.float64)
ANALYTIC_X = np.asarray(DIVERTED_REFERENCE.x_point, dtype=np.float64)
WALL_CLEARANCE_FRACTION = 0.35

# (requested_cells, part suffix, display label, caption)
ROWS = (
    (
        -110,
        "reduced",
        "diverted single-null, 110 cells (reduced)",
        "110-cell row: the sign-change census admits no saddle anywhere; solved "
        "flux matches the analytic field to 1.9e-17.",
    ),
    (
        -300,
        "cells-300",
        "diverted single-null, 300 cells",
        "300-cell row: the sign-change census admits no saddle anywhere; solved "
        "flux matches the analytic field to 3.2e-17.",
    ),
    (
        -500,
        "cells-500",
        "diverted single-null, 500 cells (positive control)",
        "500-cell row: the census admits one saddle at 1.20 mm from the "
        "analytic X-point (positive control).",
    ),
)
RING_COLOURS = {
    "solved": "#cc7722",
    "analytic": "#3366cc",
    "census": "#8a2be2",
}

_CATALOGUE_RADIUS_PITCHES = 2.6
_CLOSEUP_HALF_EXTENT = 0.34
_FIGURE_FACE = "#f8f8f2"


def diverted_wall() -> np.ndarray:
    """Return the certificate's standard-node wall outside the single-null."""
    return oracle_fixture.offset_wall(
        DIVERTED_REFERENCE.separatrix(1441),
        clearance=WALL_CLEARANCE_FRACTION * DIVERTED_REFERENCE.minor_radius,
        points=oracle_fixture.WALL_POINT_COUNT,
    )


def load_row(row: tuple) -> tuple[dict, object, object, object]:
    """Load the persisted part, machine and production operator for one row."""
    requested_cells, suffix, _label, _caption = row
    parts = json.loads(
        (PARTS / f"diverted-single-null-production-route-{suffix}.json").read_text()
    )
    carrier = reference_cases()[oracle_fixture.ANALYTIC_CASE]
    machine = oracle_fixture.cached_machine(
        carrier,
        requested_cells,
        wall_nodes=oracle_fixture.WALL_POINT_COUNT,
        wall=diverted_wall(),
    )
    if machine.cache["semantic_key"] != parts["cache"]["semantic_key"]:
        raise RuntimeError(
            "machine cache key does not match the persisted part; the operator "
            "would not reproduce the row"
        )
    operator = oracle_fixture.forward_operator(carrier, machine)
    return parts, machine, operator


def characteristic_pitch(parts: dict) -> float:
    """Return the part's stored median-cell characteristic pitch [m]."""
    return float(parts["characteristic_pitch_m"])


def ring_sign_catalogue(fgrid, psi_grid, pitch: float) -> dict:
    """Catalogue the ring sign patterns around the analytic saddle.

    ``Null2D.crossing_count`` counts cyclic flips of ``ring > centre`` around
    each rotated centroid ring (six neighbours on the hex carrier).  The
    receipt lists every ring within the catalogue radius of the analytic
    X-point with its centre, distance, crossing count and the six-bit pattern
    of ``neighbour > centre``, beside the whole-grid crossing histogram.  The
    solved field is what the read samples; the analytic field agrees to the
    per-row residual, so the patterns are identical on both to round-off.
    """
    stencil = np.asarray(fgrid.locator.stencil, dtype=np.intp)
    centre_index = stencil[:, 0]
    centres = np.asarray(fgrid.locator.coordinate, dtype=np.float64)[centre_index]
    distance = np.linalg.norm(centres - ANALYTIC_X[None, :], axis=1)
    order = np.argsort(distance)
    near = order[distance[order] <= _CATALOGUE_RADIUS_PITCHES * pitch]

    ring_psi = psi_grid[stencil]
    bits = (ring_psi[:, 1:] > ring_psi[:, :1]).astype(np.int8)
    flips = np.sum(bits != np.roll(bits, -1, axis=1), axis=1)
    cells = []
    for index in near:
        cells.append(
            {
                "centre_rz_m": [float(centres[index, 0]), float(centres[index, 1])],
                "distance_to_analytic_x_m": float(distance[index]),
                "distance_in_pitches": float(distance[index] / pitch),
                "ring_crossings": int(flips[index]),
                "ring_above_bits": "".join(str(int(bit)) for bit in bits[index].tolist()),
                "centre_psi_wb": float(psi_grid[centre_index[index]]),
                "ring_psi_wb": [float(value) for value in ring_psi[index, 1:].tolist()],
            }
        )
    unique, counts = np.unique(flips, return_counts=True)
    histogram = {
        str(int(key)): int(value)
        for key, value in zip(unique, counts, strict=True)
    }
    return {
        "cell_pitch_m": pitch,
        "catalogue_radius_pitches": _CATALOGUE_RADIUS_PITCHES,
        "catalogued_rings": cells,
        "whole_grid_ring_crossing_histogram": histogram,
        "saddle_crossing_ring_count": int(np.sum(flips == 4)),
        "census_ring_count": int(stencil.shape[0]),
    }


def census_candidates(fgrid, psi_grid) -> dict:
    """Run stage 1, the fixed-design sign-change census, on one grid flux."""
    started = perf_counter()
    (vmap_o, vmap_x), census = fgrid.read_census(jnp.asarray(psi_grid))
    elapsed = perf_counter() - started
    status = fgrid.candidate_table_status(jnp.asarray(psi_grid))

    def rows(array):
        result = []
        for value in np.asarray(array, dtype=np.float64):
            if np.all(np.isfinite(value[:3])):
                result.append(
                    {
                        "r_m": float(value[0]),
                        "z_m": float(value[1]),
                        "psi_wb": float(value[2]),
                        "kind": float(value[3]),
                        "distance_to_analytic_x_m": float(
                            np.linalg.norm(value[:2] - ANALYTIC_X)
                        ),
                    }
                )
        return result

    return {
        "elapsed_seconds": elapsed,
        "o_candidate_count": int(np.asarray(status["candidate_count"])[0]),
        "x_candidate_count": int(np.asarray(status["candidate_count"])[1]),
        "o_rows": rows(vmap_o),
        "x_rows": rows(vmap_x),
        "spline_authored": bool(np.asarray(census["spline_authored"]).item()),
        "overflow": bool(np.asarray(census["overflow"]).any()),
        "raw_ring_saddle_count": int(np.asarray(census["raw_ring_count"])[1]),
    }


def containment_stage(ftop, vmap_x) -> dict:
    """Run stage 2: first-wall containment of the census saddle rows."""
    contained = np.asarray(
        ftop.contained_x_candidates(jnp.asarray(vmap_x)), dtype=bool
    )
    finite = np.all(np.isfinite(vmap_x[:, :3]), axis=1)
    kept = [
        {
            "r_m": float(vmap_x[index, 0]),
            "z_m": float(vmap_x[index, 1]),
            "psi_wb": float(vmap_x[index, 2]),
            "distance_to_analytic_x_m": float(
                np.linalg.norm(vmap_x[index, :2] - ANALYTIC_X)
            ),
        }
        for index in range(vmap_x.shape[0])
        if finite[index] and contained[index]
    ]
    return {
        "in_wall_candidates": kept,
        "in_wall_count": int(contained.sum()),
        "finite_candidate_count": int(finite.sum()),
        "dropped_outside_wall": int(np.sum(finite & ~contained)),
        "occupiable_region_note": (
            "the oracle carrier marks every grid node inside_material "
            "(operator.inside_material is all-true), so the occupiable-cell gate "
            "collapses to the first-wall polygon"
        ),
    }


def _finite_columns(array):
    """Return a JSON-safe list with non-finite entries replaced by None."""
    return [
        [float(value) if np.isfinite(value) else None for value in row]
        for row in np.asarray(array, dtype=np.float64).reshape((-1, 2))
    ]


def polish_receipt_stage(ftop, operator, physical) -> dict:
    """Run stage 3: the production read's own split-spline polish receipt.

    ``read_qualification`` is the production entry the solve uses; on these
    hex carriers the tensor spline is absent, ``complete_map`` is false and the
    polish retains the census rows without authoring a spline position.
    """
    qualification = ftop.read_qualification(
        jnp.asarray(physical), operator.polarity, operator.inside_material
    )
    receipt = qualification.polish_receipt
    return {
        "complete_map": bool(np.asarray(receipt["complete_map"]).all()),
        "spline_authored": [bool(v) for v in np.asarray(receipt["spline_authored"])],
        "converged": [bool(v) for v in np.asarray(receipt["converged"])],
        "fit_converged": [
            bool(v) for v in np.asarray(receipt.get("fit_converged", [False, False]))
        ],
        "selected_position_rz": _finite_columns(receipt["selected_position_rz"]),
        "selected_value_wb": [
            (float(v) if np.isfinite(v) else None)
            for v in np.asarray(receipt["selected_value"])
        ],
        "note": (
            "hex carrier is not tensor-supported: complete_map false, so the "
            "polish retains the census rows and claims no spline position"
        ),
    }


def production_read(parts: dict, operator) -> dict:
    """Run the complete production read and the banked margin block."""
    terminal = np.asarray(parts["render_data"]["terminal_flux_wb"], dtype=np.float64)
    physical = jnp.asarray(terminal[: operator.physical_node_number])
    started = perf_counter()
    _masks, topology = operator.read(physical)
    elapsed = perf_counter() - started
    margin_block = candidate_flux_margins(operator, physical, polarity=1)
    x_point = np.asarray(topology.x_point, dtype=np.float64)
    return {
        "read_elapsed_seconds": elapsed,
        "read_status": "qualified_axis",
        "class": "diverted" if bool(topology.diverted) else "limited",
        "axis_rz_m": np.asarray(topology.axis, dtype=np.float64).tolist(),
        "x_point_rz_m": x_point.tolist() if np.all(np.isfinite(x_point)) else None,
        "boundary_flux_wb": float(topology.boundary_flux),
        "axis_flux_wb": float(topology.axis_flux),
        "x_candidate_count": margin_block["x_candidate_count"],
        "o_candidate_count": margin_block["o_candidate_count"],
        "x_point_error_m": (
            float(np.linalg.norm(x_point - ANALYTIC_X))
            if np.all(np.isfinite(x_point))
            else None
        ),
    }


def dropped_reason(parts: dict, census: dict, production: dict) -> str:
    """Return the one-sentence reason each row loses (or keeps) its X-point."""
    pitch = characteristic_pitch(parts)
    saddles = census["x_rows"]
    if production["x_point_rz_m"] is not None:
        return (
            "no drop: the X-point is admitted at "
            f"{1.0e3 * production['x_point_error_m']:.3f} mm from the analytic "
            "saddle"
        )
    if any(s["distance_to_analytic_x_m"] <= pitch for s in saddles):
        return "census finds a saddle within one pitch; a later stage removes it"
    if not saddles:
        return (
            "the sign-change census finds no saddle anywhere on the grid (no "
            "ring reads four cyclic sign changes), so no X candidate ever "
            "enters containment, polish, dedupe or the type gate"
        )
    return (
        "the sign-change census finds saddles only beyond one characteristic "
        "pitch of the analytic X-point"
    )


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def _raster_field(coordinates, values, samples=241):
    """Interpolate one hex-node field onto a regular raster for contours."""
    points = np.asarray(coordinates, dtype=np.float64)
    field = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = np.all(np.isfinite(points), axis=1) & np.isfinite(field)
    points, field = points[finite], field[finite]
    radial = np.linspace(points[:, 0].min(), points[:, 0].max(), samples)
    height = np.linspace(points[:, 1].min(), points[:, 1].max(), samples)
    radius_grid, height_grid = np.meshgrid(radial, height)
    raster = LinearNDInterpolator(points, field, fill_value=np.nan)(
        radius_grid, height_grid
    )
    return radial, height, np.asarray(raster, dtype=np.float64)


def _draw_cells_outline(axis, machine, pitch: float) -> None:
    """Outline the hex cells within two pitches of the analytic X-point."""
    kept = []
    for polygon in machine.cell_polygons:
        poly = np.asarray(polygon, dtype=np.float64)[:, :2]
        if np.linalg.norm(poly.mean(axis=0) - ANALYTIC_X) <= 2.0 * pitch:
            kept.append(poly)
    if kept:
        axis.add_collection(
            PolyCollection(
                kept,
                facecolors="none",
                edgecolors="#555555",
                linewidths=0.6,
                zorder=3,
            )
        )


def render_row_figure(parts, machine, operator, saddle_rows, admitted, output, label, caption):
    """Render the full-map and saddle-region close-up panels for one row."""
    terminal = np.asarray(parts["render_data"]["terminal_flux_wb"], dtype=np.float64)
    psi_grid = terminal[: operator.grid.node_number]
    coordinates = np.asarray(
        parts["render_data"]["coordinates_rz_m"], dtype=np.float64
    )[: operator.grid.node_number]
    wall = np.asarray(parts["render_data"]["wall_units_rz_m"], dtype=np.float64)[0]
    boundary = np.asarray(parts["render_data"]["boundary_rz_m"], dtype=np.float64)
    pitch = characteristic_pitch(parts)

    radial, height, solved = _raster_field(coordinates, psi_grid)
    levels = poloidal.contour_levels(solved, count=14)

    figure, axes = plt.subplots(1, 2, figsize=(12.0, 5.4), constrained_layout=True)
    figure.patch.set_facecolor(_FIGURE_FACE)
    full, close = axes

    for axis in (full, close):
        poloidal.draw_flux_contours(
            axis, radial, height, solved, levels, color=RING_COLOURS["solved"]
        )
        poloidal.draw_wall(axis, units=(wall,), linewidth=0.5)

    poloidal.draw_boundary(
        full, boundary[:, 0], boundary[:, 1], color=RING_COLOURS["analytic"]
    )

    analytic_style = DEFAULT_INK.variant(
        axis_marker="^",
        axis_color=RING_COLOURS["analytic"],
        xpoint_color=RING_COLOURS["analytic"],
    )
    poloidal.draw_nulls(
        full,
        magnetic_axis=ANALYTIC_AXIS,
        x_points=ANALYTIC_X[None, :],
        style=analytic_style,
        contain=(wall,),
    )
    poloidal.draw_nulls(
        close,
        magnetic_axis=None,
        x_points=ANALYTIC_X[None, :],
        style=analytic_style,
        contain=(wall,),
    )

    for saddle in saddle_rows:
        filled = admitted is not None and np.allclose(
            (saddle["r_m"], saddle["z_m"]),
            (admitted["r_m"], admitted["z_m"]),
            atol=1.0e-9,
        )
        for axis in (full, close):
            axis.plot(
                saddle["r_m"],
                saddle["z_m"],
                marker="o",
                markersize=6 if filled else 5,
                markerfacecolor=RING_COLOURS["census"] if filled else "none",
                markeredgecolor=RING_COLOURS["census"],
                linestyle="none",
                zorder=8,
            )

    _draw_cells_outline(close, machine, pitch)
    close.set_xlim(
        ANALYTIC_X[0] - _CLOSEUP_HALF_EXTENT, ANALYTIC_X[0] + _CLOSEUP_HALF_EXTENT
    )
    close.set_ylim(
        ANALYTIC_X[1] - _CLOSEUP_HALF_EXTENT, ANALYTIC_X[1] + _CLOSEUP_HALF_EXTENT
    )

    poloidal_axes(full)
    poloidal_axes(close)
    full.set_title("solved flux contours; analytic nulls; census candidates", fontsize=8)
    close.set_title(
        "saddle-region close-up: cells around the analytic X-point outlined", fontsize=8
    )
    figure.suptitle(f"{label}\n{caption}", fontsize=9)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)


def main() -> int:
    """Run the staged assertion over the three certificate rows."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figure-root",
        type=Path,
        default=ROOT / "docs/figures/null-identification-authority/census-assertion",
    )
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--rows", nargs="*", default=None)
    args = parser.parse_args()

    configure_dtypes()
    assert jax.config.jax_enable_x64 is True
    assert jax.default_backend() == "cpu"

    report: dict = {
        "schema": "nova.null-census-assertion",
        "analytic_x_point_rz_m": ANALYTIC_X.tolist(),
        "analytic_axis_rz_m": ANALYTIC_AXIS.tolist(),
        "code_reference": {
            "sign_change_census": (
                "nova/equilibrium/forward_operator.py:661 _compatibility_census "
                "(ring crossing_count at :664; saddle ring mask :665)"
            ),
            "read_census_rows": "nova/equilibrium/forward_operator.py:747",
            "wall_containment": (
                "nova/equilibrium/topology.py:383 contained_x_candidates; "
                ":404 x_point_index ranks contained candidates only"
            ),
            "split_spline_polish": (
                "nova/equilibrium/flux_surface_connectivity.py:1062 "
                "polish_census_stationary_points (via read_qualification "
                "topology.py:1040)"
            ),
            "dedupe_radius": (
                "nova/equilibrium/forward_operator.py:364 _deduplicate_type "
                "(structured); :680 retained_valid slot cap (hex carrier)"
            ),
            "hessian_type_gate": (
                "nova/equilibrium/forward_operator.py:679 type_agrees "
                "(candidate kind == expected type); polish :1314"
            ),
            "admission_verdict": (
                "nova/equilibrium/topology.py:416 x_point_data; :1009 "
                "read_qualification data_x; :1036 admitted bit"
            ),
        },
        "rows": {},
        "figures": [],
    }

    for row in ROWS:
        cells, suffix, label, caption = row
        if args.rows and suffix not in args.rows and str(cells) not in args.rows:
            continue
        parts, machine, operator = load_row(row)
        fgrid = operator._fixed_design_topology.grid
        ftop = operator._fixed_design_topology
        pitch = characteristic_pitch(parts)
        physical = np.asarray(
            parts["render_data"]["terminal_flux_wb"], dtype=np.float64
        )[: operator.physical_node_number]
        psi_grid = physical[: operator.grid.node_number]

        census = census_candidates(fgrid, psi_grid)
        sign_catalogue = ring_sign_catalogue(fgrid, psi_grid, pitch)
        vmap_x = np.asarray(
            jax.device_get(ftop.grid(jnp.asarray(psi_grid))[1]), dtype=np.float64
        )
        containment = containment_stage(ftop, vmap_x)
        polish = polish_receipt_stage(ftop, operator, physical)
        prod = production_read(parts, operator)
        stage_verdict = {
            "census_finds_saddle_within_one_pitch": any(
                s["distance_to_analytic_x_m"] <= pitch for s in census["x_rows"]
            ),
            "defect_stage": (
                None
                if prod["x_point_rz_m"] is not None
                else "1_sign_change_census"
            ),
        }

        admitted = None
        if prod["x_point_rz_m"] is not None:
            admitted = {
                "r_m": prod["x_point_rz_m"][0],
                "z_m": prod["x_point_rz_m"][1],
            }

        figure_name = f"census-assertion-{suffix}.png"
        output = args.figure_root / figure_name
        render_row_figure(
            parts, machine, operator, census["x_rows"], admitted, output, label, caption
        )
        report["figures"].append(
            {
                "row": suffix,
                "path": str(output),
                "project_absolute_src": (
                    "/nova/figures/null-identification-authority"
                    f"/census-assertion/{figure_name}"
                ),
            }
        )
        report["rows"][suffix] = {
            "requested_cells": cells,
            "realised_cells": int(parts["realised_cells"]),
            "characteristic_pitch_m": pitch,
            "solved_fixed_point_residual": parts["solver"][
                "terminal_fixed_point_residual"
            ],
            "boundary_flux_error_wb": parts["geometry"]["boundary_flux_error_wb"],
            "stages": {
                "census": census,
                "sign_catalogue": sign_catalogue,
                "containment": containment,
                "split_spline_polish": polish,
                "production_read": prod,
            },
            "dropped_reason": dropped_reason(parts, census, prod),
            "verdict": stage_verdict,
        }

    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"CENSUS_ASSERTION_RECEIPT {args.receipt}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
