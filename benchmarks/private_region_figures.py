#!/usr/bin/env python3
"""Render the private-region mask figures from the measured receipt.

Two panels, drawn with the committed nova.media painters:

  private-mask-cost-cells.svg     per-state milliseconds against realised cell
                                  count on log axes for the four mask
                                  formulations, beside the production read's
                                  dominant private-exclusion stage from the
                                  census receipt.
  private-mask-poloidal-1000.svg  the poloidal scene at the 1000-cell rung:
                                  analytic flux line contours with the
                                  boundary level pinned, the first wall, the
                                  analytic nulls, the private-region cells
                                  hatched, and the cells where a formulation
                                  disagrees with the production mask outlined.

The 1000-cell rung is rebuilt and all four masks recomputed as the positive
control; the per-rung disagreement counts must equal the measured receipt or
the script raises, so the figure cannot drift from what was timed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch

import nova.media.poloidal as media
from nova.media.ink import DEFAULT_INK, poloidal_axes
from nova.jax.config import configure_dtypes
from benchmarks import limiter_read_resolution_audit as limiter_audit
from benchmarks import solovev_certificate as certificate
from benchmarks import private_region_parallel_kernel as prk

configure_dtypes()

_DIVERTED = prk.DIVERTED
_RECEIPT = Path(__file__).parent / "private_region_results.json"
_CENSUS_PATTERN = (
    Path(__file__).parents[1]
    / "docs"
    / "figures"
    / "cut-cell-current-attribution"
    / "census-kernel"
    / "parts"
    / "census-kernel-gpu-cells-{requested}.json"
)
_OUT_DIR = (
    Path(__file__).parents[1]
    / "docs"
    / "figures"
    / "cut-cell-current-attribution"
    / "private-region"
)
_STYLE = DEFAULT_INK.variant(axis_marker="^", axis_markersize=8)


def _case_scene(requested_cells: int):
    """The machine, analytic flux and read state the benchmark builds a rung on."""
    carrier, source, exact = certificate._case(_DIVERTED)
    machine = limiter_audit._machine(
        _DIVERTED, carrier, exact, -requested_cells, prk.WALL_NODE_COUNT
    )
    return machine, exact


def _census_reference_ms() -> dict[int, dict[str, float]]:
    """Per-requested-rung production-read stage costs from the census receipt."""
    reference: dict[int, dict[str, float]] = {}
    for requested in (500, 1000, 2500):
        record = json.loads(
            Path(str(_CENSUS_PATTERN).format(requested=requested)).read_text()
        )
        marginal = record["reference"]["stage_marginal_seconds"]["private_exclusion"]
        reference[requested] = {
            "private_exclusion_ms": marginal / 16.0 * 1e3,
            "read_ms": record["reference"]["batch_seconds_per_state_ms"],
        }
    return reference


def _hexagon_vertices(
    centre: np.ndarray, neighbours: np.ndarray, pitch: float
) -> np.ndarray:
    """The regular hexagon around ``centre`` snapped to the mesh lattice.

    The mesh is a locally exact hexagonal packing whose edge directions lie
    on thirty-degree multiples, so the centre-to-neighbour axis selects one
    lattice direction and the cell's six Voronoi corners sit half a step off
    it, at the corner radius ``pitch / sqrt(3)``.  A cell with fewer than six
    at-pitch neighbours lies on the packing edge; its footprint is still the
    full lattice hexagon, oriented from the neighbours it does have.
    """
    offset = neighbours - centre
    mean_angle = float(np.arctan2(np.mean(offset[:, 1]), np.mean(offset[:, 0])))
    lattice_step = int(np.rint(mean_angle / (np.pi / 6.0)))
    vertex_angle = (lattice_step + 0.5) * np.pi / 6.0
    phi = vertex_angle + np.arange(6) * (np.pi / 3.0)
    radius = pitch / np.sqrt(3.0)
    return centre + radius * np.stack((np.cos(phi), np.sin(phi)), axis=-1)


def _cell_polygons(
    coordinate: np.ndarray, rings: np.ndarray, pitch: float, selected: np.ndarray
) -> list[np.ndarray]:
    """Hexagon polygons for the selected cells from at-pitch neighbours.

    The saddle-aware ``connectivity_rings`` table excludes the cut cells that
    sit on the separatrix legs, and the private-region cells are exactly those
    cells, so the ring table cannot close their hexagons.  The underlying
    mesh is an exact-pitch hexagonal packing, so the neighbours that do close
    a cell's hexagon are the coordinate points one pitch away, found from
    geometry rather than from the filtered edge table.  The ``rings`` argument
    keeps the call site's vocabulary; the neighbours come from the mesh.
    """
    positions = np.asarray(coordinate, dtype=np.float64)
    radius = 1.05 * float(pitch)
    radius2 = radius * radius
    polygons: list[np.ndarray] = []
    for cell in np.nonzero(selected)[0]:
        cell = int(cell)
        dist2 = np.sum((positions - positions[cell]) ** 2, axis=1)
        nids = np.nonzero((dist2 > 1e-9) & (dist2 <= radius2))[0]
        if nids.size == 0:
            continue
        polygons.append(_hexagon_vertices(positions[cell], positions[nids], pitch))
    return polygons


def figure_cost() -> None:
    """Log-log per-state cost against realised cells for all four masks."""
    receipt = json.loads(_RECEIPT.read_text())
    cells = [r["realised"] for r in receipt["rungs"]]
    series = {key: [r["per_state_ms"][key] for r in receipt["rungs"]] for key in "abcd"}
    census = _census_reference_ms()

    fig, axes = plt.subplots(figsize=(7.0, 4.6), layout="constrained")
    styles = {
        "a": ("#3366cc", "o", "production hex flood"),
        "b": ("#8c564b", "s", "raster doubling"),
        "c": ("#2ca02c", "^", "pointer jumping"),
        "d": ("#e377c2", "v", "saddle-wedge level"),
    }
    for key in "abcd":
        color, marker, label = styles[key]
        axes.plot(
            cells,
            series[key],
            marker=marker,
            color=color,
            linestyle="-" if key == "a" else "--",
            linewidth=1.2 if key == "a" else 1.0,
            markersize=5,
            label=label,
        )
    ref = [
        (r["realised"], census[r["requested"]]["private_exclusion_ms"])
        for r in receipt["rungs"]
        if r["requested"] in census
    ]
    axes.plot(
        [r for r, _ in ref],
        [ms for _, ms in ref],
        marker="*",
        markersize=13,
        color="#cc0000",
        linestyle="none",
        label="production read\n(private-exclusion stage)",
    )
    axes.set_xscale("log")
    axes.set_yscale("log")
    axes.set_xlabel("realised cells")
    axes.set_ylabel("per-state milliseconds")
    axes.grid(True, which="both", alpha=0.25)
    axes.legend(fontsize=8, loc="best")
    fig.savefig(_OUT_DIR / "private-mask-cost-cells.svg")
    plt.close(fig)


def figure_poloidal() -> None:
    """Poloidal panel at the 1000-cell rung: private cells and disagreements."""
    bundle = prk._build_rung(1000)
    machine, _exact = _case_scene(1000)
    receipt = json.loads(_RECEIPT.read_text())
    row = next(r for r in receipt["rungs"] if r["requested"] == 1000)

    masks: dict[str, np.ndarray] = {}
    for key in "abcd":
        single = lambda state: prk._FORMULATIONS[key](bundle, state)  # noqa: E731
        mask = np.asarray(jax.jit(single)(jnp.asarray(bundle.psi_grid)))
        masks[key] = mask
        if int(np.count_nonzero(mask)) != row["private"][key]:
            raise RuntimeError(
                f"formulation {key} at 1000 cells: measured {row['private'][key]} "
                f"private cells, rebuilt {int(np.count_nonzero(mask))}"
            )
    counts = {
        key: int(np.count_nonzero(masks[key] != bundle.production_private))
        for key in "abcd"
    }
    for key in "abcd":
        if counts[key] != row["differing_cells"][key]:
            raise RuntimeError(
                f"disagreement {key} at 1000 cells: measured "
                f"{row['differing_cells'][key]}, rebuilt {counts[key]}"
            )
    if counts["a"] != 0 or counts["c"] != 0:
        raise RuntimeError("a/c disagree with production at 1000 cells")

    coordinate = np.asarray(bundle.topo.connectivity_coordinate, dtype=np.float64)
    rings = np.asarray(bundle.topo.connectivity_rings, dtype=np.int64)
    pitch = float(bundle.raster["pitch"])

    r_span = (float(coordinate[:, 0].min()), float(coordinate[:, 0].max()))
    z_span = (float(coordinate[:, 1].min()), float(coordinate[:, 1].max()))
    margin = 0.04 * max(r_span[1] - r_span[0], z_span[1] - z_span[0])
    radius = np.linspace(r_span[0] - margin, r_span[1] + margin, 220)
    height = np.linspace(z_span[0] - margin, z_span[1] + margin, 240)
    rr, zz = np.meshgrid(radius, height, indexing="ij")
    flux2d = limiter_audit._exact_flux(
        _DIVERTED, _exact, np.stack((rr.ravel(), zz.ravel()), axis=-1)
    ).reshape(radius.size, height.size)

    fig, axes = plt.subplots(figsize=(6.4, 6.0))
    poloidal_axes(axes, _STYLE)
    levels = media.contour_levels(
        flux2d, 20, boundary=bundle.boundary_flux, axis=bundle.axis_flux
    )
    media.draw_flux_contours(axes, radius, height, flux2d.T, levels, style=_STYLE)
    wall = np.asarray(machine.wall_node, dtype=np.float64)
    media.draw_wall(axes, radius=wall[:, 0], height=wall[:, 1], style=_STYLE)

    production = bundle.production_private
    # At this rung the raster formulation (b) returns no private cells at all,
    # so its disagreement is the whole private set; the disagreements with
    # spatial structure are the saddle-wedge test's (d) five misses, and those
    # are the cells worth drawing against the hatched private set.
    disagreement = masks["d"] != production
    axes.add_collection(
        PolyCollection(
            _cell_polygons(coordinate, rings, pitch, production),
            facecolors="none",
            edgecolors="#cc0000",
            linewidths=0.7,
            hatch="///",
            zorder=_STYLE.zorder_plasma + 1,
        )
    )
    axes.add_collection(
        PolyCollection(
            _cell_polygons(coordinate, rings, pitch, disagreement),
            facecolors="none",
            edgecolors="#ff7f0e",
            linewidths=1.6,
            zorder=_STYLE.zorder_plasma + 1,
        )
    )
    media.draw_nulls(
        axes,
        magnetic_axis=bundle.axis,
        x_points=bundle.x_point[None, :],
        contain=wall,
        style=_STYLE,
    )
    handles = [
        Patch(
            facecolor="none",
            edgecolor="#cc0000",
            hatch="///",
            label="private-flux cells",
        ),
        Patch(facecolor="none", edgecolor="#ff7f0e", label="saddle-wedge misses"),
    ]
    axes.legend(handles=handles, fontsize=8, loc="upper right")
    fig.savefig(_OUT_DIR / "private-mask-poloidal-1000.svg")
    plt.close(fig)


def main() -> None:
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    figure_cost()
    figure_poloidal()
    print("wrote", _OUT_DIR / "private-mask-cost-cells.svg")
    print("wrote", _OUT_DIR / "private-mask-poloidal-1000.svg")


if __name__ == "__main__":
    main()
