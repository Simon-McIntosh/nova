#!/usr/bin/env python3
"""Render the private-region mask figures from the measured receipt.

Two panels, drawn with the committed nova.media painters:

  private-mask-cost-cells.svg     per-state milliseconds against realised cell
                                  count on log axes for the four mask
                                  formulations, beside the production read's
                                  dominant private-exclusion stage from the
                                  census receipt.
  private-mask-poloidal-1074.svg  the poloidal scene at 1074 realised cells.
  private-mask-poloidal-2616.svg  the poloidal scene at 2616 realised cells.

Both poloidal panels use each cached oracle cell's own polygon vertices.  The
production private mask is hatched, the pointer-jumping mask is outlined, and
the saddle-wedge disagreements carry a third line style.  Analytic separatrix
legs and the Hessian eigenvector directions make the wedge convention visible.

Both rungs are rebuilt and all four masks recomputed as the positive control;
the per-rung disagreement counts must equal the measured receipt or the script
raises, so the figure cannot drift from what was timed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import nova.media.poloidal as media
from nova.media.ink import DEFAULT_INK, poloidal_axes
from nova.jax.config import configure_dtypes
from benchmarks import limiter_read_resolution_audit as limiter_audit
from benchmarks import solovev_certificate as certificate
from benchmarks import private_region_parallel_kernel as prk
from benchmarks import private_region_wedge_audit as wedge_audit
from benchmarks import xpoint_cell_allocation_rca as allocation_rca

configure_dtypes()

_DIVERTED = prk.DIVERTED
_RECEIPT = (
    Path(__file__).parents[1]
    / "docs"
    / "figures"
    / "cut-cell-current-attribution"
    / "private-region"
    / "receipt.json"
)
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
_REFERENCE_STYLE = _STYLE.variant(
    axis_color="#3366cc", xpoint_color="#3366cc", xpoint_marker="x"
)
_ADMITTED_STYLE = _STYLE.variant(
    axis_color="#ff7f0e", xpoint_color="#ff7f0e", xpoint_marker="+"
)


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


def _selected_polygons(
    polygons: list[np.ndarray], selected: np.ndarray
) -> list[np.ndarray]:
    """Return authored polygons for selected cells without synthesising geometry."""
    return [polygons[int(cell)] for cell in np.flatnonzero(selected)]


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


def figure_poloidal(requested_cells: int) -> Path:
    """Poloidal panel at one rung: authored cells, masks, legs, and directions."""
    bundle = prk._build_rung(requested_cells)
    machine, exact = _case_scene(requested_cells)
    receipt = json.loads(_RECEIPT.read_text())
    row = next(r for r in receipt["rungs"] if r["requested"] == requested_cells)

    masks: dict[str, np.ndarray] = {}
    for key in "abcd":
        single = lambda state: prk._FORMULATIONS[key](bundle, state)  # noqa: E731
        mask = np.asarray(jax.jit(single)(jnp.asarray(bundle.psi_grid)))
        masks[key] = mask
        if int(np.count_nonzero(mask)) != row["private"][key]:
            raise RuntimeError(
                f"formulation {key} at {requested_cells} cells: measured "
                f"{row['private'][key]} "
                f"private cells, rebuilt {int(np.count_nonzero(mask))}"
            )
    counts = {
        key: int(np.count_nonzero(masks[key] != bundle.production_private))
        for key in "abcd"
    }
    for key in "abcd":
        if counts[key] != row["differing_cells"][key]:
            raise RuntimeError(
                f"disagreement {key} at {requested_cells} cells: measured "
                f"{row['differing_cells'][key]}, rebuilt {counts[key]}"
            )
    if counts["a"] != 0 or counts["c"] != 0:
        raise RuntimeError(f"a/c disagree with production at {requested_cells} cells")

    coordinate = np.asarray(bundle.topo.connectivity_coordinate, dtype=np.float64)
    polygons = wedge_audit.aligned_cell_polygons(bundle, machine)

    r_span = (float(coordinate[:, 0].min()), float(coordinate[:, 0].max()))
    z_span = (float(coordinate[:, 1].min()), float(coordinate[:, 1].max()))
    margin = 0.04 * max(r_span[1] - r_span[0], z_span[1] - z_span[0])
    radius = np.linspace(r_span[0] - margin, r_span[1] + margin, 220)
    height = np.linspace(z_span[0] - margin, z_span[1] + margin, 240)
    rr, zz = np.meshgrid(radius, height, indexing="ij")
    flux2d = limiter_audit._exact_flux(
        _DIVERTED, exact, np.stack((rr.ravel(), zz.ravel()), axis=-1)
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
    disagreement = masks["d"] != production
    axes.add_collection(
        PolyCollection(
            _selected_polygons(polygons, production),
            facecolors="none",
            edgecolors="#cc0000",
            linewidths=0.7,
            hatch="///",
            zorder=_STYLE.zorder_plasma + 1,
        )
    )
    axes.add_collection(
        PolyCollection(
            _selected_polygons(polygons, masks["c"]),
            facecolors="none",
            edgecolors="#2ca02c",
            linewidths=1.2,
            linestyles="--",
            zorder=_STYLE.zorder_plasma + 2,
        )
    )
    axes.add_collection(
        PolyCollection(
            _selected_polygons(polygons, disagreement),
            facecolors="none",
            edgecolors="#ff7f0e",
            linewidths=1.6,
            linestyles=":",
            zorder=_STYLE.zorder_plasma + 3,
        )
    )

    branches = allocation_rca._analytic_separatrix_branches(_DIVERTED, exact)
    for position, leg in enumerate(branches["legs"]):
        leg = np.asarray(leg, dtype=np.float64)
        axes.plot(
            leg[:, 0],
            leg[:, 1],
            color=_STYLE.separatrix_color,
            linewidth=_STYLE.separatrix_linewidth,
            linestyle="-",
            label="analytic separatrix legs" if position == 0 else None,
            zorder=_STYLE.zorder_separatrix,
        )

    geometry = wedge_audit.saddle_geometry(exact, bundle.x_point)
    ray_length = 0.42 * (z_span[1] - z_span[0])
    for position, direction in enumerate(geometry.eigenvector_rays):
        endpoint = np.asarray(bundle.x_point) + ray_length * direction
        axes.plot(
            [bundle.x_point[0], endpoint[0]],
            [bundle.x_point[1], endpoint[1]],
            color="#6a3d9a",
            linewidth=1.1,
            linestyle="-.",
            label="Hessian eigenvectors" if position == 0 else None,
            zorder=_STYLE.zorder_plasma + 1,
        )
    media.draw_nulls(
        axes,
        magnetic_axis=np.asarray(certificate.AXIS_M),
        x_points=np.asarray(certificate.X_POINT_M)[None, :],
        contain=wall,
        style=_REFERENCE_STYLE,
    )
    media.draw_nulls(
        axes,
        magnetic_axis=bundle.axis,
        x_points=bundle.x_point[None, :],
        contain=wall,
        style=_ADMITTED_STYLE,
    )
    handles = [
        Patch(
            facecolor="none",
            edgecolor="#cc0000",
            hatch="///",
            label="production private mask",
        ),
        Patch(
            facecolor="none",
            edgecolor="#2ca02c",
            linestyle="--",
            label="pointer-jumping mask",
        ),
        Patch(
            facecolor="none",
            edgecolor="#ff7f0e",
            linestyle=":",
            label="wedge disagreements",
        ),
        Line2D(
            [],
            [],
            color=_STYLE.separatrix_color,
            linewidth=_STYLE.separatrix_linewidth,
            label="analytic separatrix legs",
        ),
        Line2D([], [], color="#6a3d9a", linestyle="-.", label="Hessian eigenvectors"),
        Line2D(
            [],
            [],
            color=_REFERENCE_STYLE.xpoint_color,
            marker="x",
            linestyle="none",
            label="analytic nulls",
        ),
        Line2D(
            [],
            [],
            color=_ADMITTED_STYLE.xpoint_color,
            marker="+",
            linestyle="none",
            label="admitted nulls",
        ),
    ]
    axes.legend(handles=handles, fontsize=7, loc="upper right")
    output = _OUT_DIR / f"private-mask-poloidal-{bundle.realised}.svg"
    fig.savefig(output)
    if requested_cells == 1000:
        fig.savefig(_OUT_DIR / "private-mask-poloidal-1000.svg")
    plt.close(fig)
    return output


def main() -> None:
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    figure_cost()
    outputs = [figure_poloidal(requested) for requested in (1000, 2500)]
    print("wrote", _OUT_DIR / "private-mask-cost-cells.svg")
    for output in outputs:
        print("wrote", output)


if __name__ == "__main__":
    main()
