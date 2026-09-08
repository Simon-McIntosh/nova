#!/usr/bin/env python3
"""Render the poloidal convergence atlas from committed topology operands.

Each panel draws one frame of the committed MAST twelve-row bank or the
five-frame DIII-D demonstration route as a single poloidal scene: full-map
flux contours, the governed wall, the magnetic axis, every qualified X-point
with out-of-vessel candidates drawn distinctly rather than dropped, the chosen
boundary (Nova closed surface) beside the EFIT reference surface, and the
hysteretic wall-shadow mask drawn as shaded wall segments.

The purpose is to make the class of defect that the limited-boundary
investigation spent the whole sprint chasing visible in one glance: the atlas
shows every committed frame rather than a curated set, including the
non-converged rows, the frames whose read returned no closed boundary, and the
frame whose recognised wall anchor sits on the lower centre column. See the
planning document "the poloidal convergence atlas" for the lead request.

Nothing here solves or reads an image: the flux maps, walls and stationary
points are the committed operands of docs/figures/topology-visual-corroboration,
and the only writes are PNGs and a machine-readable receipt.

The painter set is nova.media.poloidal (draw_wall, draw_flux_contours,
draw_boundary, contour_levels) together with one purpose-built null painter.
nova.media.poloidal.draw_nulls is NOT reused for the X-points because it drops
out-of-vessel candidates as a matter of design (its containment filter exists
to remove markers that mislead), and showing the out-of-vessel population is
precisely the content the atlas is for. The null painter here therefore keeps
every finite candidate and draws the outside-wall subset with a distinct
marker and a per-panel tally. The wall shadow is the production
wall_height_shadow_mask hysteretic rule run on the committed axis, admitted
saddle and finite X candidates; the connectivity-private term is passed as an
empty mask because the committed operands do not persist the per-wall-node
private flux classification, so the drawn shadow is the saddle-height band
only. Both deviations are stated in the evidence document.

Run on a debug partition with a per-job TMPDIR so the compilation cache cannot
collide (the render itself is matplotlib-only; the shadow rule is one eager
kernel):
    srun --partition=<machine>_debug --time=00:59:00 --cpus-per-task=2 \
        bash -lc 'export TMPDIR=/tmp; \
          UV_PROJECT_ENVIRONMENT=~/Code/nova/.venv PYTHONPATH=$PWD \
          uv run --no-sync python benchmarks/poloidal_convergence_atlas.py' \
        > /tmp/atlas-render.log 2>&1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from nova.equilibrium.connectivity_boundary import wall_height_shadow_mask
from nova.equilibrium.domain import PlasmaDomain
from nova.equilibrium.wall_mask import inside_polygon
from nova.media import poloidal
from nova.media.ink import DEFAULT_INK

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_DIR = ROOT / "docs/figures/topology-visual-corroboration"
OUT_DIR = ROOT / "docs/figures/null-identification-authority/convergence-atlas"
RECEIPT = OUT_DIR / "convergence-atlas.json"
DIIID_SOURCE_ROOT = Path("/work/projects/imas_gpu/sophelio/raw/data/diii_d_train")

MAST_TOPOLOGY = EVIDENCE_DIR / "mast-topology-operands.npz"
MAST_METADATA = EVIDENCE_DIR / "mast-topology-operands.metadata.json"
DIIID_TOPOLOGY = EVIDENCE_DIR / "diiid-topology-operands.npz"
DIIID_METADATA = EVIDENCE_DIR / "diiid-topology-operands.metadata.json"

MAST_CURRENT_BY_SHOT = {
    21978: 801.5e3,
    21983: 813.7e3,
    21985: 934.4e3,
    21986: 874.3e3,
    21989: 928.9e3,
    22086: 933.0e3,
}
DIIID_CURRENT_MA_BY_IDENTITY = {
    "d3d_shot_00000c4a7b:179": 1.28,
    "d3d_shot_0003ff34e7:44": 1.22,
    "d3d_shot_001554e054:144": 1.02,
    "d3d_shot_002495e835:146": 1.63,
    "d3d_shot_0040ca9bdc:137": 1.60,
}
CONTOUR_COUNT = 14
PLASMA_CURRENT_BAND_MIDPOINT_MA = 1.3


def _git_revision() -> str:
    """Return the tree revision running the renderer."""
    return subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _resolve_raster(
    cells: np.ndarray, per_cell: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (radius, height, flux) with flux shaped (height, radius).

    The committed operand grid is a 33 by 33 radius-major latitude (cells
    ordered R then Z), and draw_flux_contours requires (nz, nr), so the per
    cell values are reshaped and transposed.
    """
    radius = np.unique(np.asarray(cells[:, 0]))
    height = np.unique(np.asarray(cells[:, 1]))
    if radius.size * height.size != per_cell.size:
        raise ValueError("operand cell grid is not a full radius-major lattice")
    values = np.asarray(per_cell).reshape(radius.size, height.size)
    return radius, height, values.T


def _geometry_hysteresis(
    wall: np.ndarray, radius: np.ndarray, height: np.ndarray
) -> tuple[float, float]:
    """Return the geometry-derived wall-height band and X qualification radius.

    These mirror the operator constants in nova.equilibrium.forward_operator:
    the hysteresis is a quarter of the smallest positive wall-height step and
    the qualification distance is 1.5 times the largest grid step.
    """
    wall_heights = np.unique(np.asarray(wall[:, 1]))
    steps = np.diff(wall_heights)
    positive = steps[steps > 0.0]
    hysteresis = 0.25 * float(np.min(positive)) if positive.size else float("inf")
    radial = np.diff(np.unique(np.asarray(radius)))
    vertical = np.diff(np.unique(np.asarray(height)))
    grid_steps = np.concatenate((radial[radial > 0.0], vertical[vertical > 0.0]))
    if not grid_steps.size:
        return hysteresis, 0.0
    return hysteresis, 1.5 * float(np.max(grid_steps))


def _distinct_candidates(
    x_candidates: np.ndarray, spacing: float = 0.015
) -> np.ndarray:
    """Return the deduplicated finite X candidates at detector spacing."""
    array = x_candidates.reshape(-1, 2)
    finite = array[np.all(np.isfinite(array[:, :2]), axis=1)]
    if finite.size == 0:
        return finite
    kept: list[np.ndarray] = []
    occupied: list[np.ndarray] = []
    for point in np.asarray(finite, dtype=float):
        if (
            not kept
            or min(np.linalg.norm(point - prior) for prior in occupied) > spacing
        ):
            kept.append(point)
            occupied.append(point)
    return np.asarray(kept, dtype=float)


def _qualified_x_points(
    candidates: np.ndarray,
    admitted_x: np.ndarray,
    efit_x: np.ndarray,
    qualification_distance: float,
) -> np.ndarray:
    """Recover the qualified X set by the operator's qualification radius.

    The committed store persists only the raw 30-slot candidate table and the
    single admitted saddle, so the qualified population is recovered by the
    visual proxy of lying within the operator's X-qualification distance of the
    admitted saddle or an EFIT saddle. This is stated as a proxy in the
    evidence document; the raw table is not the typed set.
    """
    distinct = _distinct_candidates(candidates)
    if distinct.size == 0:
        return distinct
    radius = max(qualification_distance, 0.05)
    recognised: list[np.ndarray] = []
    references = [np.asarray(admitted_x[:2], dtype=float)]
    for point in np.asarray(efit_x, dtype=float).reshape(-1, 2):
        if np.all(np.isfinite(point)):
            references.append(point)
    for point in distinct:
        if any(np.linalg.norm(point - ref) <= radius for ref in references):
            recognised.append(point)
    return np.asarray(recognised, dtype=float)


def _wall_shadow(
    wall: np.ndarray,
    axis: np.ndarray,
    admitted_x: np.ndarray,
    qualified_x: np.ndarray,
    hysteresis: float,
    qualification_distance: float,
    cells: np.ndarray,
    domain_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the per-wall-node masks (height bracket, production shadow).

    The production shadow rule is ``proposed & private_wall``, and the
    committed operands persist neither the per-wall-node private flux nor a
    raster carrying the private-flux cells on the MAST rows, so two honest
    quantities are drawn. The dark band is the production mask with the
    connectivity-private term recovered from the committed domain labels where
    they exist (a wall node whose nearest grid cell is labelled PRIVATE_FLUX);
    wherever that term is absent the production shadow reads empty, which is the
    sprint-wide census result the atlas exists to show. The light band is the
    visual X-point height bracket --- the wall arc beyond the lower and upper
    qualified saddle heights --- labelled as a footprint, not a shadowed-node
    claim: it shows where the excluded band would lie without pretending the
    per-node private mask survived into the store.
    """
    if qualified_x.shape[0] == 0 or not np.all(np.isfinite(axis[:2])):
        return np.zeros(wall.shape[0], dtype=bool), np.zeros(wall.shape[0], dtype=bool)
    if domain_labels is not None and cells is not None:
        nearest = _nearest_cell(domain_labels, cells, wall)
        private = nearest == int(PlasmaDomain.PRIVATE_FLUX)
    else:
        private = np.zeros(wall.shape[0], dtype=bool)
    primary = (
        np.asarray(admitted_x[:2], dtype=float)
        if np.all(np.isfinite(admitted_x[:2]))
        else qualified_x[0][:2]
    )
    shadow = wall_height_shadow_mask(
        wall[:, 1],
        axis[1],
        primary,
        qualified_x[:, :2],
        jnp.asarray(private, dtype=bool),
        jnp.zeros(wall.shape[0], dtype=bool),
        hysteresis,
        qualification_distance,
    )
    finite_heights = np.asarray(qualified_x[:, 1], dtype=float)[
        np.all(np.isfinite(qualified_x[:, :2]), axis=1)
    ]
    if finite_heights.size:
        extreme = float(np.max(np.abs(finite_heights - axis[1])))
    else:
        extreme = 0.0
    band = (wall[:, 1] < axis[1] - extreme - hysteresis) | (
        wall[:, 1] > axis[1] + extreme + hysteresis
    )
    return band, np.asarray(shadow, dtype=bool)


def _nearest_cell(
    domain_labels: np.ndarray, cells: np.ndarray, wall: np.ndarray
) -> np.ndarray:
    """Return each wall node's nearest grid-cell domain label."""
    labels = np.asarray(domain_labels).reshape(-1)
    centres = np.asarray(cells, dtype=float).reshape(-1, 2)
    nearest: list[int] = []
    for node in np.asarray(wall, dtype=float).reshape(-1, 2):
        offset = centres - node
        nearest.append(
            int(labels[int(np.argmin(np.einsum("ij,ij->i", offset, offset)))])
        )
    return np.asarray(nearest, dtype=int)


class NullPainter:
    """Draw the axis and the qualified X-points with the out-of-vessel split.

    ``draw_nulls`` in nova.media.poloidal drops every candidate outside the
    containment polygon, which is the opposite of what the atlas has to show,
    so this painter keeps the qualified set and draws the wall-exterior
    members with a distinct hollow square while the in-vessel members are
    hollow circles; the admitted saddle is drawn filled on top by the caller.
    """

    def __init__(self, axes, style=DEFAULT_INK) -> None:
        self.axes = axes
        self.style = style

    def draw(self, magnetic_axis, x_points, wall) -> dict[str, int]:
        tally = {
            "qualified_x_count": 0,
            "qualified_x_outside_wall": 0,
        }
        if magnetic_axis is not None:
            point = np.asarray(magnetic_axis, dtype=float).reshape(-1)[:2]
            if np.all(np.isfinite(point)):
                self.axes.plot(
                    point[0],
                    point[1],
                    marker=self.style.axis_marker,
                    markersize=self.style.axis_markersize,
                    color=self.style.axis_color,
                    linestyle="none",
                    zorder=self.style.zorder_markers,
                )
        if x_points is None:
            return tally
        array = np.atleast_2d(np.asarray(x_points, dtype=float))
        array = array[np.all(np.isfinite(array[:, :2]), axis=1)]
        if array.size == 0:
            return tally
        tally["qualified_x_count"] = int(array.shape[0])
        wall2 = np.asarray(wall, dtype=float).reshape(-1, 2)
        inside = inside_polygon(array[:, 0], array[:, 1], wall2[:, 0], wall2[:, 1])
        tally["qualified_x_outside_wall"] = int(np.sum(~np.asarray(inside, dtype=bool)))
        # Outside-wall qualified X-points: hollow squares in the outboard accent.
        self.axes.plot(
            array[~inside, 0],
            array[~inside, 1],
            marker="s",
            markersize=self.style.xpoint_markersize,
            markerfacecolor="none",
            markeredgecolor="#b35806",
            markeredgewidth=self.style.xpoint_markeredgewidth,
            linestyle="none",
            zorder=self.style.zorder_markers,
        )
        # Inside-wall qualified X-points: hollow circles.
        self.axes.plot(
            array[inside, 0],
            array[inside, 1],
            marker=self.style.xpoint_marker,
            markersize=self.style.xpoint_markersize,
            markerfacecolor="none",
            markeredgecolor=self.style.xpoint_color,
            markeredgewidth=self.style.xpoint_markeredgewidth,
            linestyle="none",
            zorder=self.style.zorder_markers,
        )
        return tally


def _draw_shadowed_wall(
    axes, wall: np.ndarray, shadow: np.ndarray, alpha: float = 0.22, width: float = 7.0
) -> int:
    """Stroke the shadowed wall segments as a translucent broad band."""
    edges = np.stack((wall[:-1], wall[1:]), axis=1)
    masked_edge = edges[shadow[:-1] | shadow[1:]]
    if masked_edge.size == 0:
        return 0
    segments = [np.asarray(edge, dtype=float) for edge in masked_edge]
    axes.add_collection(
        LineCollection(
            segments,
            colors=[DEFAULT_INK.wall_color],
            linewidths=width,
            alpha=alpha,
            zorder=DEFAULT_INK.zorder_wall - 1,
        )
    )
    return len(segments)


def _load_metadata(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))["rows"]


def _panel(
    store: Any,
    index: int,
    record: dict[str, Any],
    *,
    radius: np.ndarray,
    height: np.ndarray,
    flux: np.ndarray,
    current_ma: float,
    current_band: str,
    revision: str,
) -> dict[str, int | float | str]:
    prefix = f"row_{index:02d}_"
    wall = np.asarray(store[f"{prefix}wall"], dtype=float).reshape(-1, 2)
    axis = np.asarray(store[f"{prefix}selected_o"], dtype=float).reshape(-1)[:2]
    admitted_x = np.asarray(store[f"{prefix}selected_x"], dtype=float).reshape(-1)[:2]
    wall_point = np.asarray(store[f"{prefix}wall_point"], dtype=float).reshape(-1)[:2]
    x_candidates = np.asarray(store[f"{prefix}x_candidates"], dtype=float)
    x_candidates = (
        x_candidates.reshape(-1, 2) if x_candidates.ndim == 1 else x_candidates
    )
    nova_boundary = np.asarray(store[f"{prefix}nova_boundary"], dtype=float)
    nova_boundary = (
        nova_boundary.reshape(-1, 2) if nova_boundary.ndim == 1 else nova_boundary
    )
    efit_lcfs = np.asarray(store[f"{prefix}efit_lcfs"], dtype=float).reshape(-1, 2)
    efit_axis = np.asarray(store[f"{prefix}efit_axis"], dtype=float).reshape(-1)[:2]
    efit_x = np.asarray(store[f"{prefix}efit_x"], dtype=float).reshape(-1, 2)
    hysteresis, qualification = _geometry_hysteresis(wall, radius, height)
    qualified_x = _qualified_x_points(x_candidates, admitted_x, efit_x, qualification)
    label = (
        np.asarray(store[f"{prefix}domain_labels"])
        if f"{prefix}domain_labels" in store.files
        else None
    )
    cells = (
        np.asarray(store[f"{prefix}cell_rz"])
        if f"{prefix}cell_rz" in store.files
        else None
    )

    started = time.perf_counter()
    figure, axes = plt.subplots(figsize=(5.2, 5.2))
    axes.set_aspect("equal")
    poloidal.draw_flux_contours(
        axes,
        radius,
        height,
        flux,
        poloidal.contour_levels(flux, CONTOUR_COUNT),
        style=DEFAULT_INK,
        linewidth=0.5,
    )
    band, shadow = _wall_shadow(
        wall, axis, admitted_x, qualified_x, hysteresis, qualification, cells, label
    )
    band_segments = _draw_shadowed_wall(axes, wall, band, alpha=0.14, width=9.0)
    shadow_segments = _draw_shadowed_wall(axes, wall, shadow, alpha=0.45, width=6.0)
    poloidal.draw_wall(axes, wall[:, 0], wall[:, 1], style=DEFAULT_INK, linewidth=1.6)
    if np.all(np.isfinite(efit_lcfs)) and efit_lcfs.size >= 3:
        axes.plot(
            efit_lcfs[:, 0],
            efit_lcfs[:, 1],
            color="#7f7f7f",
            linewidth=1.1,
            linestyle="--",
            zorder=3,
        )
    if np.all(np.isfinite(nova_boundary)) and nova_boundary.size >= 3:
        poloidal.draw_boundary(
            axes, nova_boundary[:, 0], nova_boundary[:, 1], style=DEFAULT_INK
        )
    painter = NullPainter(axes)
    tally = painter.draw(axis, qualified_x, wall)
    # The admitted saddle is drawn on top of the qualified population.
    if np.all(np.isfinite(admitted_x)):
        axes.plot(
            admitted_x[0],
            admitted_x[1],
            marker="X",
            markersize=DEFAULT_INK.xpoint_markersize + 2,
            markerfacecolor=DEFAULT_INK.xpoint_color,
            markeredgecolor="white",
            markeredgewidth=0.8,
            linestyle="none",
            zorder=DEFAULT_INK.zorder_markers + 1,
        )
    if np.all(np.isfinite(efit_axis)):
        axes.plot(
            efit_axis[0],
            efit_axis[1],
            marker="+",
            markersize=8,
            color="#555555",
            linestyle="none",
            zorder=3,
        )
    if np.all(np.isfinite(efit_x[:, :2])):
        axes.plot(
            efit_x[:, 0],
            efit_x[:, 1],
            marker="+",
            markersize=10,
            color="#555555",
            linestyle="none",
            zorder=3,
        )
    axes.set_xlabel("R [m]")
    axes.set_ylabel("Z [m]")
    axes.set_xlim(float(np.min(wall[:, 0])) - 0.06, float(np.max(wall[:, 0])) + 0.06)
    axes.set_ylim(float(np.min(wall[:, 1])) - 0.06, float(np.max(wall[:, 1])) + 0.06)

    identity = record["identity"]
    slug = identity.replace("/", "-").replace(" ", "-").replace(":", "-")
    out_path = OUT_DIR / f"{slug}.png"
    figure.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    elapsed = time.perf_counter() - started

    raw_time = record.get("time")
    time_s = (
        float(raw_time) / 1000.0
        if isinstance(raw_time, (int, float)) and abs(raw_time) > 100
        else raw_time
    )
    class_label = "diverted" if nova_boundary.shape[0] >= 3 else "unclassified"
    return {
        "machine": record["machine"],
        "identity": identity,
        "shot": record.get("shot"),
        "frame": record.get("frame", record.get("slice_index")),
        "time_s": time_s,
        "current_band": current_band,
        "current_ma": round(current_ma, 4),
        "class": class_label,
        "converged": bool(record.get("converged", False)),
        "terminal_residual": record.get("terminal_residual"),
        "qualification": record.get("qualification"),
        "boundary_point_count": int(nova_boundary.shape[0]),
        "axis": [float(v) for v in axis],
        "admitted_x": [float(v) for v in admitted_x],
        "wall_point": [float(v) for v in wall_point],
        "efit_axis": [float(v) for v in efit_axis],
        "efit_x_count": int(np.sum(np.all(np.isfinite(efit_x[:, :2]), axis=1))),
        **_host_tally(tally),
        "shadow_band_wall_nodes": int(np.sum(band)),
        "shadow_band_drawn_segments": band_segments,
        "shadow_private_wall_nodes": int(np.sum(shadow)),
        "shadow_private_drawn_segments": shadow_segments,
        "revision": revision,
        "render_seconds": elapsed,
        "png_path": (
            f"/nova/figures/null-identification-authority/convergence-atlas/{out_path.name}"
        ),
    }


def _host_tally(tally: dict[str, int]) -> dict[str, int]:
    return {key: int(value) for key, value in tally.items()}


def _mast_panels(store: Any) -> list[dict[str, int | float | str]]:
    records = _load_metadata(MAST_METADATA)
    revision = "e6a83802b"
    cells = np.asarray(store["row_00_cell_rz"])
    radius = np.unique(cells[:, 0])
    height = np.unique(cells[:, 1])
    panels = []
    for index, record in enumerate(records):
        per_cell = np.asarray(store[f"row_{index:02d}_per_cell_flux_values"])
        _, _, flux = _resolve_raster(store[f"row_{index:02d}_cell_rz"], per_cell)
        current = MAST_CURRENT_BY_SHOT.get(int(record["shot"]), 0.0)
        band = "high" if current / 1e3 > 850 else "low"
        panels.append(
            _panel(
                store,
                index,
                record,
                radius=radius,
                height=height,
                flux=flux,
                current_ma=current / 1e6,
                current_band=band,
                revision=revision,
            )
        )
    return panels


def _diiid_frame_flux(
    source_parquet, frame: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the EFIT reference (radius, height, flux) at one DIII-D frame.

    The committed DIII-D topology operands persist geometry and nulls but not
    the flux field, so the polar contours of a DIII-D panel are the EFIT
    reference map read from the source parquet's 65 by 65 psirz grid. Nova's
    read (axis, X-points, wall point, boundary) is drawn over that reference,
    which is the comparison the atlas exists to expose.
    """
    import pandas as pd

    data = pd.read_parquet(source_parquet).iloc[0]
    radius = np.asarray(data["efit_grid_R"], dtype=float)
    height = np.asarray(data["efit_grid_Z"], dtype=float)
    rows = np.asarray(data["efit_psirz"][frame])
    matrix = np.stack([np.asarray(item, dtype=float) for item in rows])
    return radius, height, np.asarray(matrix.T, dtype=float)


def _diiid_panels(store: Any) -> list[dict[str, int | float | str]]:
    records = _load_metadata(DIIID_METADATA)
    panels = []
    for index, record in enumerate(records):
        source = record.get("shot", "")
        if not source:
            raise ValueError(f"DIII-D row {index} carries no source parquet name")
        source_parquet = DIIID_SOURCE_ROOT / source
        frame = int(record["frame"])
        radius, height, flux = _diiid_frame_flux(source_parquet, frame)
        identity = record["identity"]
        current_ma = DIIID_CURRENT_MA_BY_IDENTITY.get(identity, 0.0)
        band = "high" if current_ma > PLASMA_CURRENT_BAND_MIDPOINT_MA else "low"
        revision = f"diiid:{Path(source).stem}"
        panels.append(
            _panel(
                store,
                index,
                record,
                radius=radius,
                height=height,
                flux=flux,
                current_ma=current_ma,
                current_band=band,
                revision=revision,
            )
        )
    return panels


def run(output: Path) -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mast = np.load(MAST_TOPOLOGY, allow_pickle=False)
    diiid = np.load(DIIID_TOPOLOGY, allow_pickle=False)
    panels = _mast_panels(mast) + _diiid_panels(diiid)
    payload = {
        "schema": "nova-poloidal-convergence-atlas",
        "renderer_revision": _git_revision(),
        "panel_count": len(panels),
        "per_frame_render_cost": None,
        "panels": panels,
    }
    if panels:
        seconds = [float(panel["render_seconds"]) for panel in panels]
        payload["per_frame_render_cost"] = {
            "mean_seconds": float(np.mean(seconds)),
            "max_seconds": float(np.max(seconds)),
            "count": len(seconds),
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print("ATLAS_PANELS " + json.dumps({"count": len(panels), "files": str(OUT_DIR)}))
    for panel in panels:
        print("ATLAS_PANEL " + json.dumps(panel, sort_keys=True, default=str))
    return payload


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RECEIPT)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    run(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
