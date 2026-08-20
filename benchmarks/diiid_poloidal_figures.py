"""Publish DIII-D poloidal figures in the imas-ink house style.

The visual constants below were read from
``/home/ITER/mcintos/Code/imas-ink/imas_ink/style.py`` on 2026-08-20 and are
copied deliberately so nova does not acquire an imas-ink dependency.  The
rendered categories reproduce ``InkStyle``: grey open-flux contours
``#999999`` at 0.35 pt, separatrices ``#cc0000`` at 1.5 pt, walls ``#000000``
at 1.0 pt, unfilled coil sections edged ``#888888`` at 0.4 pt, magnetic probes
``#888888`` at 2.5 pt, flux loops ``#666666`` at 3.0 pt, labels at 8 pt, and
figures at 120 dpi.  Confined-flux blue ``#3366cc`` at 0.7 pt is reused for
Thomson scattering positions and the geometry-derived sightlines.

The netCDF entry is opened through IMAS-Python with ``autoconvert=False``.
Only wall, active-coil, and diagnostic-position geometry leaves are
dereferenced: neither magnetics measurement arrays nor the equilibrium IDS are
opened.  Competition flux labels and actuator currents are read separately
from the released parquet row through an explicit column allow-list.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Polygon as PolygonPatch


NETCDF_SOURCE = Path("/home/ITER/tribolp/Public/imasdb/DIII-D/200000.nc")
COMPETITION_DATA = Path("/work/projects/imas_gpu/sophelio/raw/data/diii_d_train")
SUBTRACTION_RECEIPT = Path(
    "/work/projects/imas_gpu/sophelio/vacuum-gate/diiid_plasma_subtraction_gate.json"
)
DEFAULT_OUTPUT = Path("docs/figures/diiid-forward-onboarding/poloidal")


@dataclass(frozen=True)
class HouseStyle:
    """Copied subset of imas-ink ``InkStyle`` used by these figures."""

    flux_color: str = "#3366cc"
    flux_linewidth: float = 0.7
    contour_color: str = "#999999"
    contour_linewidth: float = 0.35
    separatrix_color: str = "#cc0000"
    separatrix_linewidth: float = 1.5
    wall_color: str = "#000000"
    wall_linewidth: float = 1.0
    coil_edgecolor: str = "#888888"
    coil_facecolor: str = "none"
    coil_linewidth: float = 0.4
    probe_color: str = "#888888"
    probe_markersize: float = 2.5
    flux_loop_color: str = "#666666"
    flux_loop_markersize: float = 3.0
    label_fontsize: float = 8.0
    figure_dpi: int = 120


STYLE = HouseStyle()


@dataclass(frozen=True)
class CoilGeometry:
    """One named active coil containing one or more physical sections."""

    name: str
    elements: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class StaticGeometry:
    """Admissible poloidal geometry read from the netCDF description."""

    limiter: np.ndarray
    coils: tuple[CoilGeometry, ...]
    probe_positions: np.ndarray
    flux_loop_positions: np.ndarray
    dd_versions: dict[str, str]
    source_path: str


@dataclass(frozen=True)
class CompetitionFrame:
    """One labelled competition frame and its static Thomson coordinates."""

    shot: str
    frame: int
    time_ms: float
    recorded_r2: float
    radius: np.ndarray
    height: np.ndarray
    label: np.ndarray
    separatrix: np.ndarray
    thomson_names: tuple[str, ...]
    thomson_positions: np.ndarray
    row: dict[str, Any]
    additive_gauge: float


def _plain_text(value: Any) -> str:
    return str(value).strip()


def _canonical_axis(values: Any) -> np.ndarray:
    """Preserve stored endpoints while removing float32 spacing jitter."""

    axis = np.asarray(values, dtype=float)
    if axis.ndim != 1 or axis.size < 2 or not np.all(np.diff(axis) > 0.0):
        raise ValueError("a physical grid axis must be increasing and one-dimensional")
    return np.linspace(axis[0], axis[-1], axis.size)


def _element_outline(element: Any) -> np.ndarray:
    """Return one stored outline or an exact rectangle expansion."""

    geometry = element.geometry
    geometry_type = int(geometry.geometry_type)
    if geometry_type == 1:
        return np.column_stack(
            [
                np.asarray(geometry.outline.r, dtype=float),
                np.asarray(geometry.outline.z, dtype=float),
            ]
        )
    if geometry_type == 2:
        rectangle = geometry.rectangle
        radius = float(rectangle.r)
        height = float(rectangle.z)
        half_width = float(rectangle.width) / 2.0
        half_height = float(rectangle.height) / 2.0
        return np.asarray(
            [
                [radius - half_width, height - half_height],
                [radius + half_width, height - half_height],
                [radius + half_width, height + half_height],
                [radius - half_width, height + half_height],
            ]
        )
    raise ValueError(f"unsupported active-coil geometry type {geometry_type}")


def read_static_geometry(source_path: Path = NETCDF_SOURCE) -> StaticGeometry:
    """Read only wall, active-coil, and magnetic-diagnostic geometry leaves."""

    import imas

    with imas.DBEntry(source_path, "r") as entry:
        wall = entry.get("wall", 0, autoconvert=False)
        active = entry.get("pf_active", 0, autoconvert=False)
        magnetics = entry.get("magnetics", 0, lazy=True, autoconvert=False)

        descriptions = wall.description_2d
        if len(descriptions) != 1 or len(descriptions[0].limiter.unit) != 1:
            raise ValueError("expected one wall description with one limiter unit")
        outline = descriptions[0].limiter.unit[0].outline
        limiter = np.column_stack(
            [np.asarray(outline.r, dtype=float), np.asarray(outline.z, dtype=float)]
        )

        coils = []
        for coil_index in range(len(active.coil)):
            coil = active.coil[coil_index]
            name = (
                _plain_text(coil.identifier)
                or _plain_text(coil.name)
                or f"coil_{coil_index}"
            )
            elements = tuple(
                _element_outline(coil.element[element_index])
                for element_index in range(len(coil.element))
            )
            coils.append(CoilGeometry(name=name, elements=elements))

        flux_loop_positions = []
        for loop_index in range(len(magnetics.flux_loop)):
            positions = magnetics.flux_loop[loop_index].position
            if len(positions):
                flux_loop_positions.append(
                    [float(positions[0].r), float(positions[0].z)]
                )
        probe_positions = []
        for probe_index in range(len(magnetics.b_field_pol_probe)):
            position = magnetics.b_field_pol_probe[probe_index].position
            probe_positions.append([float(position.r), float(position.z)])

        return StaticGeometry(
            limiter=limiter,
            coils=tuple(coils),
            probe_positions=np.asarray(probe_positions, dtype=float),
            flux_loop_positions=np.asarray(flux_loop_positions, dtype=float),
            dd_versions={
                "wall": _plain_text(wall.ids_properties.version_put.data_dictionary),
                "pf_active": _plain_text(
                    active.ids_properties.version_put.data_dictionary
                ),
                "magnetics": _plain_text(
                    magnetics.ids_properties.version_put.data_dictionary
                ),
            },
            source_path=str(source_path),
        )


def _competition_columns() -> list[str]:
    """Return the explicit parquet allow-list needed for the plotted frame."""

    from nova.imas.diiid_description import F_COILS

    return [
        "efit_times",
        "efit_grid_R",
        "efit_grid_Z",
        "efit_psirz",
        "efit_r_axis",
        "efit_z_axis",
        "efit_lcfs_n",
        "efit_lcfs_r",
        "efit_lcfs_z",
        "thomson_chord_name",
        "thomson_chord_R",
        "thomson_chord_Z",
        "coil_name",
        "coil_input_column",
        "coil_R",
        "coil_Z",
        "coil_width",
        "coil_height",
        "coil_angle1",
        "coil_angle2",
        "magnetics_time",
        *(f"magnetics_{name}" for name in (*F_COILS, "ECOILA")),
    ]


def read_competition_frame(
    data_path: Path = COMPETITION_DATA,
    subtraction_receipt: Path = SUBTRACTION_RECEIPT,
) -> CompetitionFrame:
    """Read the first banked quiescent frame through an explicit column fence."""

    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError(
            "the live figure run needs a project environment carrying pyarrow"
        ) from error

    result = json.loads(subtraction_receipt.read_text())
    record = next(
        item
        for item in result["score"]["frame_records"]
        if item["population"] == "quiescent"
    )
    shot_path = data_path / record["shot"]
    table = parquet.read_table(shot_path, columns=_competition_columns())
    row = {name: table[name][0].as_py() for name in table.column_names}
    frame = int(record["frame"])
    count = int(row["efit_lcfs_n"][frame])
    separatrix = np.column_stack(
        [
            np.asarray(row["efit_lcfs_r"][frame][:count], dtype=float),
            np.asarray(row["efit_lcfs_z"][frame][:count], dtype=float),
        ]
    )
    thomson_names = tuple(str(value) for value in row["thomson_chord_name"])
    thomson_positions = np.column_stack(
        [
            np.asarray(row["thomson_chord_R"], dtype=float),
            np.asarray(row["thomson_chord_Z"], dtype=float),
        ]
    )
    radius = _canonical_axis(row["efit_grid_R"])
    height = _canonical_axis(row["efit_grid_Z"])
    return CompetitionFrame(
        shot=shot_path.name,
        frame=frame,
        time_ms=float(record["time_ms"]),
        recorded_r2=float(record["r2"]),
        radius=radius,
        height=height,
        label=np.asarray(row["efit_psirz"][frame], dtype=float),
        separatrix=separatrix,
        thomson_names=thomson_names,
        thomson_positions=thomson_positions,
        row=row,
        additive_gauge=float(record["additive_gauge_wb_per_rad"]),
    )


def build_subtraction_fields(frame: CompetitionFrame) -> dict[str, np.ndarray]:
    """Recreate the banked label-to-vacuum sequence for one frame."""

    from benchmarks.diiid_corpus_conventions import nova_total_flux_to_corpus
    from benchmarks.diiid_plasma_subtraction_gate import (
        filament_response,
        project_label_plasma,
    )
    from nova.imas.diiid_description import (
        DiiidDescriptionRegistry,
        vacuum_psi,
        vacuum_response,
    )

    response = filament_response(frame.radius, frame.height)
    plasma = project_label_plasma(
        frame.row,
        frame.frame,
        frame.radius,
        frame.height,
        response,
    )
    registry = DiiidDescriptionRegistry()
    description = registry.ingest(frame.row, source_row=frame.shot)
    coil_response = vacuum_response(description, frame.radius, frame.height)
    coil = nova_total_flux_to_corpus(
        vacuum_psi(frame.row, description, coil_response)[frame.frame]
    )
    remainder = frame.label - plasma.plasma_flux_per_radian
    aligned_coil = coil + frame.additive_gauge
    return {
        "label": frame.label,
        "plasma": plasma.plasma_flux_per_radian,
        "remainder": remainder,
        "coil": aligned_coil,
        "residual": remainder - aligned_coil,
    }


def _closed(vertices: np.ndarray) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=float)
    if np.array_equal(vertices[0], vertices[-1]):
        return vertices
    return np.vstack([vertices, vertices[0]])


def _draw_wall(axis: Axes, geometry: StaticGeometry) -> None:
    limiter = _closed(geometry.limiter)
    axis.plot(
        limiter[:, 0],
        limiter[:, 1],
        color=STYLE.wall_color,
        linewidth=STYLE.wall_linewidth,
        zorder=4,
    )


def _draw_coils(axis: Axes, geometry: StaticGeometry) -> None:
    for coil in geometry.coils:
        for element in coil.elements:
            axis.add_patch(
                PolygonPatch(
                    element,
                    closed=True,
                    facecolor=STYLE.coil_facecolor,
                    edgecolor=STYLE.coil_edgecolor,
                    linewidth=STYLE.coil_linewidth,
                    zorder=3,
                )
            )


def _coil_centre(coil: CoilGeometry) -> np.ndarray:
    return np.mean(np.vstack(coil.elements), axis=0)


def _coil_label_position(coil: CoilGeometry, centre: np.ndarray) -> np.ndarray:
    fixed = {
        "ECOILA": (0.54, 0.16),
        "ECOILB": (0.54, -0.16),
        "E567UP": (2.76, 1.55),
        "E567DN": (2.76, -1.55),
        "E89UP": (1.95, 2.06),
        "E89DN": (1.95, -2.06),
    }
    if coil.name in fixed:
        return np.asarray(fixed[coil.name], dtype=float)
    radial_offset = -0.10 if centre[0] < 1.2 else 0.10
    vertical_offset = 0.055 if centre[1] >= 0.0 else -0.055
    return centre + np.asarray([radial_offset, vertical_offset])


def _label_coils(axis: Axes, geometry: StaticGeometry) -> None:
    for coil in geometry.coils:
        centre = _coil_centre(coil)
        label = _coil_label_position(coil, centre)
        axis.annotate(
            coil.name,
            xy=centre,
            xytext=label,
            fontsize=STYLE.label_fontsize,
            ha="center",
            va="center",
            arrowprops={"arrowstyle": "-", "color": STYLE.coil_edgecolor, "lw": 0.4},
            bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none", "pad": 1},
            zorder=7,
        )


def _format_poloidal_axis(axis: Axes, title: str) -> None:
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("R [m]")
    axis.set_ylabel("Z [m]")
    axis.set_title(title)


def _contour_levels(values: np.ndarray, count: int = 11) -> np.ndarray:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        raise ValueError("contour field contains no finite values")
    lower, upper = np.quantile(finite, [0.03, 0.97])
    if np.isclose(lower, upper):
        lower, upper = float(np.min(finite)), float(np.max(finite))
    if np.isclose(lower, upper):
        upper = lower + 1.0
    return np.linspace(lower, upper, count)


def _draw_flux_contours(
    axis: Axes,
    radius: np.ndarray,
    height: np.ndarray,
    field: np.ndarray,
) -> None:
    axis.contour(
        radius,
        height,
        np.asarray(field, dtype=float),
        levels=_contour_levels(field),
        colors=STYLE.contour_color,
        linewidths=STYLE.contour_linewidth,
        linestyles="solid",
        zorder=2,
    )


def _draw_separatrix(axis: Axes, separatrix: np.ndarray) -> None:
    axis.plot(
        separatrix[:, 0],
        separatrix[:, 1],
        color=STYLE.separatrix_color,
        linewidth=STYLE.separatrix_linewidth,
        zorder=5,
    )


def _fit_collinear_line(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit a total-least-squares line and report maximum orthogonal residual."""

    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] != 2:
        raise ValueError("a coordinate string needs at least two R-Z points")
    anchor = np.mean(points, axis=0)
    _, _, vectors = np.linalg.svd(points - anchor, full_matrices=False)
    direction = vectors[0]
    direction /= np.linalg.norm(direction)
    offsets = points - anchor
    residual = np.abs(direction[0] * offsets[:, 1] - direction[1] * offsets[:, 0])
    return anchor, direction, float(np.max(residual))


def _extend_line_to_limiter(
    anchor: np.ndarray,
    direction: np.ndarray,
    limiter: np.ndarray,
) -> np.ndarray:
    """Intersect an infinite R-Z line with the outermost limiter crossings."""

    intersections: list[tuple[float, np.ndarray]] = []
    polygon = _closed(limiter)
    for start, stop in zip(polygon[:-1], polygon[1:], strict=True):
        edge = stop - start
        matrix = np.column_stack([direction, -edge])
        determinant = float(np.linalg.det(matrix))
        if abs(determinant) < 1e-12:
            continue
        line_parameter, edge_parameter = np.linalg.solve(matrix, start - anchor)
        if -1e-10 <= edge_parameter <= 1.0 + 1e-10:
            point = anchor + line_parameter * direction
            if not any(
                np.linalg.norm(point - existing[1]) < 1e-8 for existing in intersections
            ):
                intersections.append((float(line_parameter), point))
    if len(intersections) < 2:
        raise ValueError("coordinate line does not cross the limiter twice")
    intersections.sort(key=lambda item: item[0])
    return np.vstack([intersections[0][1], intersections[-1][1]])


def _thomson_groups(frame: CompetitionFrame) -> dict[str, np.ndarray]:
    groups: dict[str, list[np.ndarray]] = {"core": [], "divertor": [], "tangential": []}
    for name, position in zip(
        frame.thomson_names, frame.thomson_positions, strict=True
    ):
        matched = next((key for key in groups if name.startswith(f"TS_{key}_")), None)
        if matched is None:
            raise ValueError(f"unrecognised Thomson subsystem {name!r}")
        groups[matched].append(position)
    return {key: np.asarray(values, dtype=float) for key, values in groups.items()}


def _plot_machine_geometry(
    geometry: StaticGeometry, output: Path
) -> tuple[Path, dict[str, Any]]:
    figure, axis = plt.subplots(figsize=(9, 9), constrained_layout=True)
    _draw_wall(axis, geometry)
    _draw_coils(axis, geometry)
    _label_coils(axis, geometry)
    _format_poloidal_axis(axis, "DIII-D limiter and active coils")
    path = output / "machine_geometry.png"
    figure.savefig(path, dpi=STYLE.figure_dpi)
    plt.close(figure)
    return path, {
        "coil_labels": len(geometry.coils),
        "label_unit": "one label per coil",
    }


def _plot_competition_flux(
    geometry: StaticGeometry, frame: CompetitionFrame, output: Path
) -> Path:
    figure, axis = plt.subplots(figsize=(8, 9), constrained_layout=True)
    _draw_flux_contours(axis, frame.radius, frame.height, frame.label)
    _draw_wall(axis, geometry)
    _draw_coils(axis, geometry)
    _draw_separatrix(axis, frame.separatrix)
    _format_poloidal_axis(
        axis,
        f"Competition flux map — {frame.shot}, {frame.time_ms:g} ms",
    )
    path = output / "competition_psi.png"
    figure.savefig(path, dpi=STYLE.figure_dpi)
    plt.close(figure)
    return path


def _plot_subtraction_sequence(
    geometry: StaticGeometry,
    frame: CompetitionFrame,
    fields: dict[str, np.ndarray],
    output: Path,
) -> Path:
    figure, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    panels = (
        ("label", "Competition label"),
        ("plasma", "Projected plasma"),
        ("remainder", "Vacuum remainder"),
        ("coil", "Coil model + gauge"),
        ("residual", "Remainder - coil"),
    )
    for axis, (name, title) in zip(axes.flat, panels, strict=False):
        _draw_flux_contours(axis, frame.radius, frame.height, fields[name])
        _draw_wall(axis, geometry)
        _draw_separatrix(axis, frame.separatrix)
        _format_poloidal_axis(axis, title)
    note = axes.flat[-1]
    note.axis("off")
    note.text(
        0.02,
        0.98,
        (
            "Grey unfilled contours on physical R-Z axes\n"
            f"shot: {frame.shot}\n"
            f"frame: {frame.frame} at {frame.time_ms:g} ms\n"
            f"banked whole-grid R²: {frame.recorded_r2:.6f}\n"
            "units: Wb/rad"
        ),
        va="top",
        fontsize=STYLE.label_fontsize,
    )
    path = output / "plasma_subtraction_sequence.png"
    figure.savefig(path, dpi=STYLE.figure_dpi)
    plt.close(figure)
    return path


def _plot_thomson_and_magnetics(
    geometry: StaticGeometry, frame: CompetitionFrame, output: Path
) -> tuple[Path, dict[str, Any]]:
    figure, axis = plt.subplots(figsize=(9, 9), constrained_layout=True)
    _draw_wall(axis, geometry)
    _draw_coils(axis, geometry)
    if geometry.probe_positions.size:
        axis.plot(
            geometry.probe_positions[:, 0],
            geometry.probe_positions[:, 1],
            linestyle="none",
            marker="|",
            color=STYLE.probe_color,
            markersize=STYLE.probe_markersize,
            label="B-pol probe positions",
            zorder=6,
        )
    if geometry.flux_loop_positions.size:
        axis.plot(
            geometry.flux_loop_positions[:, 0],
            geometry.flux_loop_positions[:, 1],
            linestyle="none",
            marker="o",
            markerfacecolor="none",
            color=STYLE.flux_loop_color,
            markersize=STYLE.flux_loop_markersize,
            label="flux-loop positions",
            zorder=6,
        )

    groups = _thomson_groups(frame)
    residuals = {}
    for group_name in ("core", "divertor"):
        points = groups[group_name]
        anchor, direction, residual = _fit_collinear_line(points)
        if residual > 1e-5:
            message = (
                f"{group_name} Thomson coordinate string is not collinear: "
                f"{residual:g} m"
            )
            raise ValueError(message)
        line = _extend_line_to_limiter(anchor, direction, geometry.limiter)
        axis.plot(
            line[:, 0],
            line[:, 1],
            color=STYLE.flux_color,
            linewidth=STYLE.flux_linewidth,
            label=f"{group_name} LOS — derived from collinearity",
            zorder=5,
        )
        residuals[group_name] = residual

    for group_name, marker in (("core", "."), ("divertor", "."), ("tangential", "x")):
        points = groups[group_name]
        axis.plot(
            points[:, 0],
            points[:, 1],
            linestyle="none",
            marker=marker,
            color=STYLE.flux_color,
            markersize=STYLE.label_fontsize,
            zorder=6,
        )
    tangential_centre = np.mean(groups["tangential"], axis=0)
    axis.annotate(
        "midplane tangential row — LOS not recoverable from R-Z collinearity",
        xy=tangential_centre,
        xytext=(1.05, -0.42),
        fontsize=STYLE.label_fontsize,
        arrowprops={"arrowstyle": "-", "color": STYLE.flux_color, "lw": 0.7},
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none", "pad": 2},
        zorder=7,
    )
    axis.legend(frameon=False, fontsize=STYLE.label_fontsize, loc="upper right")
    _format_poloidal_axis(axis, "DIII-D Thomson and admissible magnetic geometry")
    path = output / "thomson_and_magnetics_geometry.png"
    figure.savefig(path, dpi=STYLE.figure_dpi)
    plt.close(figure)
    return path, {
        "thomson_points": sum(len(values) for values in groups.values()),
        "derived_sightlines": 2,
        "derived_from": "collinearity of core and divertor coordinate strings",
        "maximum_collinearity_residual_m": max(residuals.values()),
        "tangential_sightline": "not recoverable from projected coordinate row",
    }


def write_figures(
    geometry: StaticGeometry,
    frame: CompetitionFrame,
    fields: dict[str, np.ndarray],
    output: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    """Write the four publication figures and their quantitative receipt."""

    output.mkdir(parents=True, exist_ok=True)
    machine_path, label_receipt = _plot_machine_geometry(geometry, output)
    flux_path = _plot_competition_flux(geometry, frame, output)
    subtraction_path = _plot_subtraction_sequence(geometry, frame, fields, output)
    thomson_path, thomson_receipt = _plot_thomson_and_magnetics(geometry, frame, output)
    paths = (machine_path, flux_path, subtraction_path, thomson_path)
    element_count = sum(len(coil.elements) for coil in geometry.coils)
    receipt = {
        "figures": [str(path) for path in paths],
        "figure_count": len(paths),
        "physical_axes": "R-Z in metres with equal aspect",
        "rendering": {
            "flux": "unfilled contour lines via colors and linewidths",
            "contourf_used": False,
            "imshow_used": False,
            "colormap_used": False,
            "style_source": "/home/ITER/mcintos/Code/imas-ink/imas_ink/style.py",
            "style": STYLE.__dict__,
        },
        "machine_geometry": {
            "source": geometry.source_path,
            "dd_versions": geometry.dd_versions,
            "limiter_vertices": len(geometry.limiter),
            "active_coils": len(geometry.coils),
            "active_elements": element_count,
            **label_receipt,
        },
        "competition_frame": {
            "shot": frame.shot,
            "frame": frame.frame,
            "time_ms": frame.time_ms,
            "banked_whole_grid_r2": frame.recorded_r2,
        },
        "diagnostic_geometry": {
            "bpol_probe_positions": len(geometry.probe_positions),
            "flux_loop_positions": len(geometry.flux_loop_positions),
            **thomson_receipt,
        },
        "content_fence": {
            "netcdf_magnetics_signal_arrays_read": [],
            "netcdf_equilibrium_ids_opened": False,
            "competition_columns": _competition_columns(),
        },
    }
    receipt_path = output / "poloidal_figure_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    receipt["receipt"] = str(receipt_path)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--netcdf", type=Path, default=NETCDF_SOURCE)
    parser.add_argument("--data", type=Path, default=COMPETITION_DATA)
    parser.add_argument("--subtraction-receipt", type=Path, default=SUBTRACTION_RECEIPT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    geometry = read_static_geometry(args.netcdf)
    frame = read_competition_frame(args.data, args.subtraction_receipt)
    fields = build_subtraction_fields(frame)
    receipt = write_figures(geometry, frame, fields, args.output)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
