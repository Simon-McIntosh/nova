"""Draw wall reachability from axis-connected flux components.

The panels combine governed machine-wall polygons with analytic topology
fixtures.  Classification is entirely discrete: confined cells receive the
canonical labels from Nova's accelerator-native four-connected component
kernel, and every positive label unequal to the magnetic-axis label is private.
No vertical band or wall-height proxy enters the calculation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath
import numpy as np
from scipy.optimize import root
from scipy.spatial import cKDTree

from nova.equilibrium.flux_surface_connectivity import (
    label_connected_components,
    private_flux_mask,
)
from nova.imas.diiid_machine_ids import repair_limiter_ring
from nova.jax.config import configure_dtypes


HERE = Path(__file__).resolve().parent
MAST_SOURCE = Path("nova/catalog/mast_geometry.json")
DIIID_SOURCE = Path(
    "docs/figures/diiid-forward-onboarding/ids-description/ids_description_receipt.json"
)
OUTPUT_PNG = HERE / "wall-reachability-topology-grid.png"
OUTPUT_JSON = HERE / "wall-reachability-topology-grid.json"
GRID_SHAPE = (181, 181)
WALL_SAMPLES = 420


@dataclass(frozen=True)
class Blob:
    """One anisotropic Gaussian contribution to a poloidal-flux fixture."""

    radius: float
    height: float
    amplitude: float
    radial_scale: float
    vertical_scale: float


@dataclass(frozen=True)
class PanelSpec:
    """Machine wall and analytic field parameters for one topology panel."""

    panel: str
    machine: str
    topology: str
    wall: np.ndarray
    blobs: tuple[Blob, ...]
    axis_seed: tuple[float, float]
    saddle_seeds: tuple[tuple[float, float], ...]
    source: str


def _closed_wall(wall: np.ndarray) -> np.ndarray:
    """Return a finite wall polygon with one explicit closing vertex."""
    finite = np.asarray(wall, dtype=float)[np.isfinite(wall).all(axis=1)]
    if not np.array_equal(finite[0], finite[-1]):
        finite = np.vstack((finite, finite[0]))
    return finite


def _machine_walls() -> tuple[np.ndarray, np.ndarray]:
    """Read the governed MAST and DIII-D wall contours from banked receipts."""
    mast_payload = json.loads(MAST_SOURCE.read_text())
    configurations = mast_payload["configurations"]
    mast_key = sorted(configurations)[0]
    mast_wall = _closed_wall(configurations[mast_key]["geometry"]["limiter"])

    diiid_payload = json.loads(DIIID_SOURCE.read_text())
    source_entries = diiid_payload["source_entries"]
    source = next(row for row in source_entries if row["shot"] == 133221)
    contour = source["contour"]
    raw_wall = np.column_stack((contour["r"], contour["z"]))
    diiid_wall, _receipt = repair_limiter_ring(raw_wall)
    return mast_wall, _closed_wall(diiid_wall)


def _flux(blobs: tuple[Blob, ...], radius, height):
    """Evaluate a smooth multi-O-point fixture field."""
    value = np.zeros(np.broadcast(radius, height).shape, dtype=float)
    for blob in blobs:
        exponent = -(
            ((radius - blob.radius) / blob.radial_scale) ** 2
            + ((height - blob.height) / blob.vertical_scale) ** 2
        )
        value += blob.amplitude * np.exp(exponent)
    return value


def _gradient(blobs: tuple[Blob, ...], coordinate: np.ndarray) -> np.ndarray:
    """Return the analytic field gradient at one coordinate."""
    radius, height = coordinate
    radial = 0.0
    vertical = 0.0
    for blob in blobs:
        value = float(_flux((blob,), radius, height))
        radial += value * -2.0 * (radius - blob.radius) / blob.radial_scale**2
        vertical += value * -2.0 * (height - blob.height) / blob.vertical_scale**2
    return np.asarray([radial, vertical])


def _hessian(blobs: tuple[Blob, ...], coordinate: np.ndarray) -> np.ndarray:
    """Return the analytic field Hessian at one coordinate."""
    radius, height = coordinate
    rr = 0.0
    zz = 0.0
    rz = 0.0
    for blob in blobs:
        value = float(_flux((blob,), radius, height))
        dr = radius - blob.radius
        dz = height - blob.height
        rr += value * (4.0 * dr**2 / blob.radial_scale**4 - 2.0 / blob.radial_scale**2)
        zz += value * (
            4.0 * dz**2 / blob.vertical_scale**4 - 2.0 / blob.vertical_scale**2
        )
        rz += value * 4.0 * dr * dz / (blob.radial_scale**2 * blob.vertical_scale**2)
    return np.asarray([[rr, rz], [rz, zz]])


def _stationary_point(
    blobs: tuple[Blob, ...], seed: tuple[float, float], expected: str
) -> np.ndarray:
    """Refine and type-check an O-point or saddle from a fixture seed."""
    solved = root(lambda point: _gradient(blobs, point), np.asarray(seed, dtype=float))
    if not solved.success or np.linalg.norm(_gradient(blobs, solved.x)) > 1.0e-8:
        raise RuntimeError(f"stationary-point refinement failed for {expected}")
    eigenvalues = np.linalg.eigvalsh(_hessian(blobs, solved.x))
    is_saddle = bool(eigenvalues[0] < 0.0 < eigenvalues[1])
    is_peak = bool(np.all(eigenvalues < 0.0))
    if (expected == "saddle" and not is_saddle) or (expected == "axis" and not is_peak):
        raise RuntimeError(
            f"stationary point has wrong type for {expected}: {eigenvalues.tolist()}"
        )
    return solved.x


def _densify_wall(wall: np.ndarray, count: int = WALL_SAMPLES) -> np.ndarray:
    """Sample a closed wall at uniform arc-length intervals."""
    segment = np.hypot(np.diff(wall[:, 0]), np.diff(wall[:, 1]))
    distance = np.concatenate(([0.0], np.cumsum(segment)))
    query = np.linspace(0.0, distance[-1], count + 1)
    return np.column_stack(
        (np.interp(query, distance, wall[:, 0]), np.interp(query, distance, wall[:, 1]))
    )


def _panel_geometry(spec: PanelSpec) -> tuple[dict, dict]:
    """Label the first pre-saddle region and classify every wall segment."""
    wall = spec.wall
    radial_span = float(np.ptp(wall[:, 0]))
    vertical_span = float(np.ptp(wall[:, 1]))
    radial = np.linspace(
        wall[:, 0].min() - 0.03 * radial_span,
        wall[:, 0].max() + 0.03 * radial_span,
        GRID_SHAPE[1],
    )
    height = np.linspace(
        wall[:, 1].min() - 0.03 * vertical_span,
        wall[:, 1].max() + 0.03 * vertical_span,
        GRID_SHAPE[0],
    )
    rr, zz = np.meshgrid(radial, height)
    inside = (
        MplPath(wall)
        .contains_points(np.column_stack((rr.ravel(), zz.ravel())), radius=1.0e-12)
        .reshape(rr.shape)
    )

    axis = _stationary_point(spec.blobs, spec.axis_seed, "axis")
    saddles = np.stack(
        [_stationary_point(spec.blobs, seed, "saddle") for seed in spec.saddle_seeds]
    )
    saddle_inside_wall = MplPath(wall).contains_points(saddles, radius=1.0e-12)
    if not np.all(saddle_inside_wall):
        raise RuntimeError(f"{spec.panel} has an out-of-vessel saddle candidate")
    axis_flux = float(_flux(spec.blobs, *axis))
    field = _flux(spec.blobs, rr, zz)
    outward = axis_flux - field
    saddle_levels = axis_flux - _flux(spec.blobs, saddles[:, 0], saddles[:, 1])
    selected_index = int(np.argmin(saddle_levels))
    selected_x = saddles[selected_index]
    saddle_binding = float(saddle_levels[selected_index])

    dense_wall = _densify_wall(wall)
    wall_levels = axis_flux - _flux(spec.blobs, dense_wall[:, 0], dense_wall[:, 1])
    wall_extremum_index = int(np.argmin(wall_levels[:-1]))
    wall_extremum_level = float(wall_levels[wall_extremum_index])
    wall_extremum = dense_wall[wall_extremum_index]

    grid_step_level = max(
        float(np.nanmedian(np.abs(np.diff(outward, axis=0)))),
        float(np.nanmedian(np.abs(np.diff(outward, axis=1)))),
        np.finfo(float).eps,
    )
    inward_offset = 0.5 * grid_step_level
    axis_row = int(np.argmin(np.abs(height - axis[1])))
    axis_column = int(np.argmin(np.abs(radial - axis[0])))
    seed = np.zeros_like(inside)
    seed[axis_row, axis_column] = True
    inside_coordinates = np.column_stack((rr[inside], zz[inside]))
    nearest = cKDTree(inside_coordinates).query(dense_wall[:-1], k=1)[1]

    def classify(level: float):
        confined_at_level = inside & (outward <= level)
        labels_at_level = np.asarray(
            label_connected_components(
                jnp.asarray(confined_at_level), sum(confined_at_level.shape)
            )
        )
        axis_label_at_level = int(labels_at_level[axis_row, axis_column])
        if axis_label_at_level <= 0:
            raise RuntimeError(f"{spec.panel} axis seed is outside its confined region")
        private_at_level = np.asarray(
            private_flux_mask(jnp.asarray(labels_at_level), jnp.asarray(seed))
        )
        public_at_level = labels_at_level == axis_label_at_level
        segment_labels_at_level = labels_at_level[inside][nearest]
        reachable_at_level = segment_labels_at_level == axis_label_at_level
        return (
            confined_at_level,
            labels_at_level,
            axis_label_at_level,
            private_at_level,
            public_at_level,
            segment_labels_at_level,
            reachable_at_level,
        )

    pre_saddle = classify(saddle_binding - inward_offset)
    pre_saddle_reachable = pre_saddle[-1]
    if np.any(pre_saddle_reachable):
        derived_topology = "limited"
        binding_level = float(np.min(wall_levels[:-1][pre_saddle_reachable]))
        classification = classify(binding_level + inward_offset)
    else:
        derived_topology = "double-null" if len(saddles) == 2 else "diverted"
        binding_level = saddle_binding
        classification = pre_saddle
    if derived_topology != spec.topology:
        raise RuntimeError(
            f"{spec.panel} connectivity derived {derived_topology}, "
            f"expected {spec.topology}"
        )

    _confined, labels, axis_label, private, public, segment_labels, reachable = (
        classification
    )
    classified_level = (
        binding_level + inward_offset
        if derived_topology == "limited"
        else binding_level - inward_offset
    )
    shadowed = ~reachable
    private_segments = (segment_labels > 0) & (segment_labels != axis_label)

    cell_area = float(np.mean(np.diff(radial)) * np.mean(np.diff(height)))
    private_labels = np.unique(labels[private])
    record = {
        "panel": spec.panel,
        "machine": spec.machine,
        "topology": derived_topology,
        "wall_source": spec.source,
        "field_source": (
            "analytic topology fixture evaluated inside the governed machine wall"
        ),
        "grid_shape_height_by_radius": list(GRID_SHAPE),
        "connectivity": 4,
        "axis_component_label": axis_label,
        "binding_kind": "wall" if derived_topology == "limited" else "saddle",
        "binding_level_normalized_outward": binding_level,
        "classification_level_normalized_outward": classified_level,
        "classification_inward_offset": inward_offset,
        "reachable_wall_segment_count": int(np.count_nonzero(reachable)),
        "shadowed_wall_segment_count": int(np.count_nonzero(shadowed)),
        "private_touching_wall_segment_count": int(np.count_nonzero(private_segments)),
        "wall_segment_count": int(reachable.size),
        "public_region_area_m2": float(np.count_nonzero(public) * cell_area),
        "private_region_count": int(private_labels.size),
        "private_region_area_m2": float(np.count_nonzero(private) * cell_area),
        "axis_coordinate_m": axis.tolist(),
        "selected_x_coordinate_m": selected_x.tolist(),
        "finite_x_candidate_coordinates_m": saddles.tolist(),
        "finite_x_candidates_inside_wall": saddle_inside_wall.tolist(),
        "wall_extremum_coordinate_m": wall_extremum.tolist(),
        "wall_extremum_level_normalized_outward": wall_extremum_level,
        "selected_x_binding_level_normalized_outward": saddle_binding,
        "post_repair_selection": True,
    }
    plot = {
        "record": record,
        "radial": radial,
        "height": height,
        "outward": outward,
        "inside": inside,
        "public": public,
        "private": private,
        "wall": dense_wall,
        "reachable": reachable,
        "axis": axis,
        "saddles": saddles,
        "selected_x": selected_x,
        "wall_extremum": wall_extremum,
        "binding_level": binding_level,
    }
    return record, plot


def _specifications() -> list[PanelSpec]:
    """Return visibly different machine/topology combinations."""
    mast_wall, diiid_wall = _machine_walls()
    return [
        PanelSpec(
            panel="A",
            machine="DIII-D",
            topology="limited",
            wall=diiid_wall,
            blobs=(
                Blob(2.15, 0.02, 1.00, 0.38, 0.52),
                Blob(1.34, -0.18, 0.76, 0.28, 0.42),
            ),
            axis_seed=(2.15, 0.02),
            saddle_seeds=((1.75, -0.08),),
            source=str(DIIID_SOURCE),
        ),
        PanelSpec(
            panel="B",
            machine="MAST",
            topology="diverted",
            wall=mast_wall,
            blobs=(
                Blob(1.08, 0.18, 1.00, 0.43, 0.48),
                Blob(0.62, -0.90, 0.92, 0.36, 0.48),
            ),
            axis_seed=(1.08, 0.18),
            saddle_seeds=((0.81, -0.36),),
            source=str(MAST_SOURCE),
        ),
        PanelSpec(
            panel="C",
            machine="MAST",
            topology="double-null",
            wall=mast_wall,
            blobs=(
                Blob(1.08, 0.00, 1.00, 0.44, 0.47),
                Blob(0.62, -0.90, 0.82, 0.34, 0.44),
                Blob(0.62, 0.90, 0.81, 0.34, 0.44),
            ),
            axis_seed=(1.08, 0.00),
            saddle_seeds=((0.80, -0.47), (0.80, 0.47)),
            source=str(MAST_SOURCE),
        ),
    ]


def _draw_panel(axis_plot, plot: dict) -> None:
    """Draw one topology panel with direct geometry encodings."""
    radial = plot["radial"]
    height = plot["height"]
    region_code = np.zeros(plot["inside"].shape, dtype=int)
    region_code[plot["public"]] = 1
    region_code[plot["private"]] = 2
    masked = np.ma.masked_where(region_code == 0, region_code)
    axis_plot.pcolormesh(
        radial,
        height,
        masked,
        cmap=ListedColormap(["#85c6d4", "#d89b55"]),
        vmin=1,
        vmax=2,
        shading="nearest",
        alpha=0.62,
        rasterized=True,
    )
    axis_plot.contour(
        radial,
        height,
        np.where(plot["inside"], plot["outward"], np.nan),
        levels=[plot["binding_level"]],
        colors=["#232323"],
        linewidths=1.25,
    )

    wall = plot["wall"]
    for index, is_reachable in enumerate(plot["reachable"]):
        axis_plot.plot(
            wall[index : index + 2, 0],
            wall[index : index + 2, 1],
            color="#198754" if is_reachable else "#b43b3b",
            linewidth=2.7,
            solid_capstyle="butt",
            zorder=5,
        )
    axis_plot.scatter(
        *plot["axis"],
        marker="o",
        s=34,
        facecolor="white",
        edgecolor="#135d6a",
        linewidth=1.4,
        zorder=7,
    )
    axis_plot.scatter(
        plot["saddles"][:, 0],
        plot["saddles"][:, 1],
        marker="x",
        s=46,
        color="#512b81",
        linewidth=1.7,
        zorder=7,
    )
    axis_plot.scatter(
        *plot["selected_x"],
        marker="X",
        s=52,
        facecolor="#512b81",
        edgecolor="white",
        linewidth=0.7,
        zorder=8,
    )
    axis_plot.scatter(
        *plot["wall_extremum"],
        marker="D",
        s=34,
        facecolor="#f2c14e",
        edgecolor="#4a3a00",
        linewidth=0.8,
        zorder=8,
    )

    record = plot["record"]
    axis_plot.set_title(
        f"{record['panel']}  {record['machine']} · {record['topology']}",
        loc="left",
        fontsize=11,
        fontweight="semibold",
    )
    wall_summary = (
        f"wall: {record['reachable_wall_segment_count']} reachable / "
        f"{record['shadowed_wall_segment_count']} shadowed\n"
    )
    region_summary = (
        f"public {record['public_region_area_m2']:.3f} m² · "
        f"private {record['private_region_count']} / "
        f"{record['private_region_area_m2']:.3f} m²"
    )
    axis_plot.text(
        0.02,
        0.02,
        wall_summary + region_summary,
        transform=axis_plot.transAxes,
        fontsize=8.2,
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 2.5},
        zorder=10,
    )
    axis_plot.set_aspect("equal", adjustable="box")
    axis_plot.set_xlabel("R [m]")
    axis_plot.set_ylabel("Z [m]")
    axis_plot.spines[["top", "right"]].set_visible(False)


def run() -> dict:
    """Generate the figure and its checkable per-panel measurement bank."""
    configure_dtypes()
    records: list[dict] = []
    plots: list[dict] = []
    for specification in _specifications():
        record, plot = _panel_geometry(specification)
        records.append(record)
        plots.append(plot)

    fig, axes = plt.subplots(1, 3, figsize=(13.4, 5.2), constrained_layout=True)
    for axis_plot, plot in zip(axes, plots, strict=True):
        _draw_panel(axis_plot, plot)

    legend = [
        Line2D([0], [0], color="#198754", lw=3, label="reachable wall"),
        Line2D([0], [0], color="#b43b3b", lw=3, label="shadowed wall"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#85c6d4",
            markeredgecolor="none",
            markersize=9,
            label="O-connected public",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#d89b55",
            markeredgecolor="none",
            markersize=9,
            label="private component",
        ),
        Line2D([0], [0], color="#232323", lw=1.3, label="binding contour"),
        Line2D(
            [0],
            [0],
            marker="X",
            color="none",
            markerfacecolor="#512b81",
            markeredgecolor="white",
            markersize=8,
            label="selected X",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="none",
            markerfacecolor="#f2c14e",
            markeredgecolor="#4a3a00",
            markersize=7,
            label="wall extremum",
        ),
    ]
    fig.legend(
        handles=legend, loc="outside lower center", ncol=7, frameon=False, fontsize=8.5
    )
    fig.suptitle(
        "Wall reachability is component topology, not height",
        fontsize=14,
        fontweight="semibold",
    )
    fig.text(
        0.5,
        0.055,
        "A: DIII-D wall, limited fixture.  B: MAST wall, diverted fixture.  "
        "C: MAST wall, double-null fixture.  Green wall cells touch the first "
        "O-connected region without crossing a saddle; red cells do not.",
        ha="center",
        fontsize=8.6,
    )
    fig.savefig(OUTPUT_PNG, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    payload = {
        "artifact": "machine-and-topology wall reachability grid",
        "project_absolute_src": (
            "/nova/figures/primary-xpoint-evidence/wall-reachability-topology-grid.png"
        ),
        "caption": (
            "A: DIII-D wall, limited fixture. B: MAST wall, diverted fixture. "
            "C: MAST wall, double-null fixture. Wall reachability is derived "
            "from the first O-point-connected component, never from a height band."
        ),
        "method": {
            "definition": (
                "The public region is the positive four-connected component "
                "containing the magnetic-axis seed immediately before the first "
                "saddle crossing. Every other positive component label is private. "
                "A wall segment is reachable only when its nearest in-vessel grid "
                "cell carries the axis component label."
            ),
            "component_kernel": (
                "nova.equilibrium.flux_surface_connectivity."
                "label_connected_components and private_flux_mask"
            ),
            "private_predicate": (
                "positive component label unequal to the axis component label"
            ),
            "selection": (
                "finite analytic saddles are refined, Hessian-typed, checked inside "
                "the governed wall, and reduced by the smallest outward level; the "
                "limited panel binds at the wall before that selected saddle"
            ),
            "selection_repair_source_commit": "5acfe07b",
            "component_labelling_source_commit": "7a7bd365",
            "height_band_used": False,
            "wall_segment_sampling": WALL_SAMPLES,
        },
        "coverage": {
            "real_machine_wall_count": 2,
            "machines": sorted({row["machine"] for row in records}),
            "topologies": [row["topology"] for row in records],
            "limited_panel_count": sum(row["topology"] == "limited" for row in records),
            "diverted_panel_count": sum(
                row["topology"] == "diverted" for row in records
            ),
            "double_null_panel_count": sum(
                row["topology"] == "double-null" for row in records
            ),
        },
        "panels": records,
    }
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


if __name__ == "__main__":
    result = run()
    for panel in result["panels"]:
        print(
            panel["panel"],
            panel["machine"],
            panel["topology"],
            panel["reachable_wall_segment_count"],
            panel["shadowed_wall_segment_count"],
            panel["private_region_count"],
        )
