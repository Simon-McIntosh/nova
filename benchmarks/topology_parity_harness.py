"""Replay committed MAST topology operands through the cell authority."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from nova.biot.null import Null1D
from nova.equilibrium.cell_partition import cell_partition_geometry
from nova.equilibrium.connectivity_boundary import (
    _PRE_SADDLE_OFFSET_FRACTION,
    _canonicalize_reciprocal_hex_edges,
    _points_inside_polygon,
    traced_boundary_read,
)
from nova.equilibrium.domain import (
    PlasmaDomain,
    axis_connected_component,
    classify_domains,
)
from nova.equilibrium.flux_surface_connectivity import (
    fit_tensor_spline,
    hex_edge_admissibility,
)
from nova.geometry.hexstencil import hex_stencil
from nova.jax.config import configure_dtypes


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = (
    ROOT / "docs/figures/topology-visual-corroboration/mast-topology-operands.npz"
)
DEFAULT_QUALIFICATION = (
    ROOT / "docs/figures/gs-absolute-accuracy/efit-reproduction.json"
)
DEFAULT_OUTPUT = ROOT / "docs/figures/hex-cell-single-grid/topology-parity.json"
DEFAULT_FIGURE = ROOT / "docs/figures/hex-cell-single-grid/topology-parity.png"
GENERATOR = (
    ROOT / "docs/figures/topology-visual-corroboration/generate_topology_visuals.py"
)


def _generator_module():
    spec = importlib.util.spec_from_file_location(
        "topology_operand_generator", GENERATOR
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load topology operand generator {GENERATOR}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _qualification_rows(path: Path) -> dict[str, dict[str, object]]:
    """Return solver qualification metadata indexed by operand identity."""
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    rows = payload.get("data", {}).get("rows", [])
    return {
        str(row.get("frame_identity", {}).get("label")): row["solver_qualification"]
        for row in rows
        if row.get("frame_identity", {}).get("label")
        and isinstance(row.get("solver_qualification"), dict)
    }


def marginal_status(
    row: dict[str, object], qualification: dict[str, object] | None
) -> tuple[bool, str]:
    """Read the governed marginal flag, with the declared interim fallback."""
    if qualification is not None and "marginal_solver_basin" in qualification:
        return bool(qualification["marginal_solver_basin"]), "solver_qualification"
    return not bool(row["converged"]), "interim converged=false rule"


def adjudicate_difference(
    flux: float,
    axis_flux: float,
    census_boundary_flux: float,
    raster_boundary_flux: float,
) -> dict[str, object]:
    """Explain one label difference under the two boundary authorities."""
    census_norm = (flux - axis_flux) / (census_boundary_flux - axis_flux)
    raster_norm = (flux - axis_flux) / (raster_boundary_flux - axis_flux)
    census_closed = bool(census_norm <= 1.0)
    raster_closed = bool(raster_norm <= 1.0)
    return {
        "psi_norm_census_saddle": float(census_norm),
        "psi_norm_raster_binding": float(raster_norm),
        "census_closed": census_closed,
        "raster_closed": raster_closed,
        "adjudication": (
            "binding-level difference"
            if census_closed != raster_closed
            else "connectivity-cut difference"
        ),
    }


def receipt_errors(receipt: dict[str, object]) -> list[str]:
    """Return schema and parity violations without suppressing row evidence."""
    errors: list[str] = []
    if receipt.get("schema") != "nova.topology-cell-parity":
        errors.append("unexpected schema")
    rows = receipt.get("rows")
    if not isinstance(rows, list):
        return [*errors, "rows must be a list"]
    required = {"identity", "replayable", "marginal_solver_basin"}
    for row in rows:
        missing = required - row.keys()
        if missing:
            errors.append(
                f"{row.get('identity', '<unknown>')}: missing {sorted(missing)}"
            )
            continue
        if not row["replayable"]:
            if not row.get("not_replayable_reason"):
                errors.append(f"{row['identity']}: missing not-replayable reason")
            continue
        for key in (
            "compared_cell_count",
            "differing_cell_count",
            "differing_cells",
            "selected_primaries",
            "classification",
            "wall_node_census",
        ):
            if key not in row:
                errors.append(f"{row['identity']}: missing {key}")
        if not row["marginal_solver_basin"]:
            if row.get("differing_cell_count") != 0:
                errors.append(f"{row['identity']}: non-marginal labels differ")
            if not all(
                primary.get("matches", False)
                for primary in row.get("selected_primaries", {}).values()
            ):
                errors.append(f"{row['identity']}: non-marginal primary differs")
            if not row.get("classification", {}).get("matches", False):
                errors.append(f"{row['identity']}: non-marginal classification differs")
        for cell in row.get("differing_cells", []):
            if cell.get("adjudication") not in {
                "binding-level difference",
                "connectivity-cut difference",
            }:
                errors.append(
                    f"{row['identity']}: unadjudicated cell {cell.get('index')}"
                )
    return errors


def _tensor_geometry(coordinate: np.ndarray):
    radius = np.unique(coordinate[:, 0])
    height = np.unique(coordinate[:, 1])
    expected = np.c_[
        np.repeat(radius, height.size),
        np.tile(height, radius.size),
    ]
    if coordinate.shape != expected.shape or not np.array_equal(coordinate, expected):
        raise RuntimeError("cached topology cells are not a tensor carrier")
    shape = (radius.size, height.size)
    rings, shared_edges = cell_partition_geometry(
        coordinate, hex_stencil(shape), np.empty((len(coordinate), 0, 2))
    )
    return radius, height, shape, rings, shared_edges


def _axis_component_labels(
    values: np.ndarray,
    coordinate: np.ndarray,
    radius: np.ndarray,
    height: np.ndarray,
    rings,
    shared_edges,
    inside: np.ndarray,
    axis: np.ndarray,
    axis_flux: float,
    boundary: np.ndarray,
    boundary_flux: float,
) -> np.ndarray:
    """Apply the production saddle-neighbourhood cut on the cached carrier."""
    shape = (radius.size, height.size)
    field = jnp.asarray(values.reshape(shape).T)
    inside_grid = jnp.asarray(inside.reshape(shape).T)
    closed = field >= boundary_flux
    confined = closed & inside_grid
    inside_flux = jnp.where(confined, field, jnp.nan)
    inward = _PRE_SADDLE_OFFSET_FRACTION * (
        jnp.nanmax(inside_flux) - jnp.nanmin(inside_flux)
    )
    component_flux = (
        boundary_flux + jnp.where(axis_flux >= boundary_flux, 1.0, -1.0) * inward
    )
    exact_links = hex_edge_admissibility(
        field,
        jnp.asarray(radius),
        jnp.asarray(height),
        jnp.asarray(boundary_flux),
        jnp.asarray(axis_flux),
        shared_edges,
    )
    inward_links = hex_edge_admissibility(
        field,
        jnp.asarray(radius),
        jnp.asarray(height),
        component_flux,
        jnp.asarray(axis_flux),
        shared_edges,
    )
    flat_coordinate = jnp.asarray(coordinate)
    centres = flat_coordinate[rings[:, :1]]
    neighbours = flat_coordinate[rings]
    pitch = jnp.linalg.norm(neighbours - centres, axis=-1)
    midpoint = jnp.mean(shared_edges, axis=-2)
    near_saddle = jnp.linalg.norm(midpoint - boundary, axis=-1) <= 3.0 * pitch
    admissible = _canonicalize_reciprocal_hex_edges(
        rings, exact_links & (inward_links | ~near_saddle)
    )
    grid_coordinate = flat_coordinate.reshape((*shape, 2)).transpose((1, 0, 2))
    distance2 = jnp.sum((grid_coordinate - axis) ** 2, axis=-1)
    seed_index = jnp.argmin(jnp.where(confined, distance2, jnp.inf))
    seed = (
        jnp.zeros(confined.size, dtype=bool)
        .at[seed_index]
        .set(True)
        .reshape(confined.shape)
    )
    connected = axis_connected_component(confined, rings, admissible, seed)
    psi_norm = (field - axis_flux) / (boundary_flux - axis_flux)
    return np.asarray(
        classify_domains(psi_norm, closed, connected, inside_grid).label.T
    ).reshape(-1)


def _retained_binding_read(
    values: np.ndarray,
    radius: np.ndarray,
    height: np.ndarray,
    inside: np.ndarray,
    axis: np.ndarray,
    x_candidates: np.ndarray,
    x_flux: np.ndarray,
    wall: np.ndarray,
) -> dict[str, object]:
    """Read the retained raster binding using its global-surface wall fallback."""
    shape = (radius.size, height.size)
    field = jnp.asarray(values.reshape(shape).T)
    surface = fit_tensor_spline(jnp.asarray(radius), jnp.asarray(height), field)
    wall_flux = surface(jnp.asarray(wall[:, 0]), jnp.asarray(wall[:, 1]))
    wall_state = Null1D(jnp.asarray(wall))(wall_flux, 1)
    candidate_state = jnp.c_[
        jnp.asarray(x_candidates), jnp.asarray(x_flux), jnp.zeros(len(x_candidates))
    ]
    read = traced_boundary_read(
        field,
        jnp.asarray(radius),
        jnp.asarray(height),
        jnp.asarray(inside.reshape(shape).T),
        axis[0],
        axis[1],
        96,
        18,
        2,
        jnp.empty((0,), dtype=field.dtype),
        jnp.asarray(1.0, dtype=field.dtype),
        jnp.asarray(wall[:, 0]),
        jnp.asarray(wall[:, 1]),
        jnp.full((1,), jnp.nan, dtype=field.dtype),
        classification_x=candidate_state,
        classification_wall=wall_state[:3],
    )
    return {
        "boundary_flux": float(read["psi_bnd"]),
        "classification": "diverted"
        if float(read["class_margin"]) >= 0.0
        else "limited",
        "class_margin": float(read["class_margin"]),
        "wall_point": np.asarray(wall_state[:2], dtype=float),
        "wall_point_flux": float(wall_state[2]),
    }


def _replay_row(
    row: dict[str, object], qualification: dict[str, object] | None
) -> tuple[dict[str, object], dict[str, np.ndarray] | None]:
    identity = str(row["identity"])
    marginal, marginal_source = marginal_status(row, qualification)
    record: dict[str, object] = {
        "identity": identity,
        "shot": int(row["shot"]),
        "frame": int(row["frame"]),
        "arm": str(row["arm"]),
        "marginal_solver_basin": marginal,
        "marginal_flag_source": marginal_source,
    }
    coordinate = np.asarray(row["cell_rz"], dtype=float)
    values = np.asarray(row["per_cell_flux_values"], dtype=float)
    committed = np.asarray(row["domain_labels"], dtype=np.int8)
    if values.shape != (len(coordinate),) or not len(coordinate):
        record.update(
            replayable=False,
            not_replayable_reason="no cached per-cell flux for this bank row",
        )
        return record, None

    record["replayable"] = True
    radius, height, _shape, rings, shared_edges = _tensor_geometry(coordinate)
    surface = fit_tensor_spline(
        jnp.asarray(radius),
        jnp.asarray(height),
        jnp.asarray(values.reshape((radius.size, height.size)).T),
    )
    o_candidates = np.asarray(row["o_candidates"], dtype=float)
    x_candidates = np.asarray(row["x_candidates"], dtype=float)
    o_present = np.all(np.isfinite(o_candidates), axis=1)
    x_present = np.all(np.isfinite(x_candidates), axis=1)
    o_flux = np.asarray(surface(o_candidates[:, 0], o_candidates[:, 1]))
    x_flux = np.asarray(surface(x_candidates[:, 0], x_candidates[:, 1]))
    replay_o_index = int(np.argmax(np.where(o_present, o_flux, -np.inf)))
    replay_x_index = int(np.argmax(np.where(x_present, x_flux, -np.inf)))
    committed_o = np.asarray(row["selected_o"], dtype=float)[0]
    committed_x = np.asarray(row["selected_x"], dtype=float)[0]
    committed_o_index = int(
        np.argmin(np.linalg.norm(o_candidates - committed_o, axis=1))
    )
    committed_x_index = int(
        np.argmin(np.linalg.norm(x_candidates - committed_x, axis=1))
    )
    axis = o_candidates[replay_o_index]
    saddle = x_candidates[replay_x_index]
    axis_flux = float(o_flux[replay_o_index])
    saddle_flux = float(x_flux[replay_x_index])
    wall = np.asarray(row["wall"], dtype=float)
    inside = np.asarray(
        _points_inside_polygon(
            jnp.asarray(coordinate[:, 0]),
            jnp.asarray(coordinate[:, 1]),
            jnp.asarray(wall[:, 0]),
            jnp.asarray(wall[:, 1]),
        ),
        dtype=bool,
    )
    replayed = _axis_component_labels(
        values,
        coordinate,
        radius,
        height,
        rings,
        shared_edges,
        inside,
        axis,
        axis_flux,
        saddle,
        saddle_flux,
    )
    retained = _retained_binding_read(
        values,
        radius,
        height,
        inside,
        axis,
        x_candidates[x_present],
        x_flux[x_present],
        wall,
    )
    raster_flux = float(retained["boundary_flux"])
    differing_indices = np.flatnonzero(committed != replayed)
    differing_cells = []
    for index in differing_indices:
        differing_cells.append(
            {
                "index": int(index),
                "centroid_m": coordinate[index].tolist(),
                "committed_label": int(committed[index]),
                "committed_label_name": PlasmaDomain(int(committed[index])).name,
                "replayed_label": int(replayed[index]),
                "replayed_label_name": PlasmaDomain(int(replayed[index])).name,
                **adjudicate_difference(
                    float(values[index]), axis_flux, saddle_flux, raster_flux
                ),
            }
        )

    raster_labels = _axis_component_labels(
        values,
        coordinate,
        radius,
        height,
        rings,
        shared_edges,
        inside,
        axis,
        axis_flux,
        np.asarray(retained["wall_point"]),
        raster_flux,
    )
    wall_owner = np.argmin(
        np.sum((wall[:, None, :] - coordinate[None, :, :]) ** 2, axis=-1), axis=1
    )
    census_private = replayed[wall_owner] == int(PlasmaDomain.PRIVATE_FLUX)
    raster_private = raster_labels[wall_owner] == int(PlasmaDomain.PRIVATE_FLUX)
    wall_differing = np.flatnonzero(census_private != raster_private)
    wall_rows = [
        {
            "index": int(index),
            "position_m": wall[index].tolist(),
            "nearest_cell_index": int(wall_owner[index]),
            "nearest_cell_label_census": int(replayed[wall_owner[index]]),
            "nearest_cell_label_raster": int(raster_labels[wall_owner[index]]),
        }
        for index in wall_differing
    ]

    committed_classification = (
        str(qualification.get("achieved_class"))
        if qualification is not None and qualification.get("achieved_class")
        else ("diverted" if len(committed_x) else "limited")
    )
    replayed_classification = "diverted"
    raster_margin = float(retained["class_margin"])
    record.update(
        compared_cell_count=int(len(coordinate)),
        differing_cell_count=int(len(differing_indices)),
        differing_cells=differing_cells,
        census_axis_flux=axis_flux,
        census_saddle_flux=saddle_flux,
        raster_binding_flux=raster_flux,
        selected_primaries={
            "axis": {
                "committed_candidate_index": committed_o_index,
                "replayed_candidate_index": replay_o_index,
                "matches": replay_o_index == committed_o_index,
                "position_residual_m": float(np.linalg.norm(axis - committed_o)),
            },
            "x_point": {
                "committed_candidate_index": committed_x_index,
                "replayed_candidate_index": replay_x_index,
                "matches": replay_x_index == committed_x_index,
                "position_residual_m": float(np.linalg.norm(saddle - committed_x)),
            },
        },
        classification={
            "committed": committed_classification,
            "replayed": replayed_classification,
            "matches": committed_classification == replayed_classification,
            "retained_raster": retained["classification"],
            "raster_class_margin": raster_margin
            if np.isfinite(raster_margin)
            else None,
            "adjudication": (
                "matches"
                if committed_classification == replayed_classification
                else "cell-authority classification difference"
            ),
        },
        wall_node_census={
            "node_count": int(len(wall)),
            "census_private_count": int(np.count_nonzero(census_private)),
            "raster_private_count": int(np.count_nonzero(raster_private)),
            "differing_node_count": int(len(wall_differing)),
            "differing_nodes": wall_rows,
        },
    )
    plot = {
        "coordinate": coordinate,
        "values": values,
        "labels": replayed,
        "wall": wall,
        "differing_cells": differing_indices,
        "differing_wall": wall_differing,
        "census_level": np.asarray(saddle_flux),
        "raster_level": np.asarray(raster_flux),
    }
    return record, plot


def _plot(rows: list[tuple[dict[str, object], dict[str, np.ndarray]]], path: Path):
    """Show label and wall differences beside both binding contours."""
    figure, axes = plt.subplots(
        1,
        len(rows),
        figsize=(7.0 * len(rows), 6.2),
        squeeze=False,
        constrained_layout=True,
    )
    for axis, (record, data) in zip(axes[0], rows, strict=True):
        coordinate = data["coordinate"]
        radius = np.unique(coordinate[:, 0])
        height = np.unique(coordinate[:, 1])
        labels = data["labels"].reshape((radius.size, height.size)).T
        field = data["values"].reshape((radius.size, height.size)).T
        axis.pcolormesh(radius, height, labels, shading="nearest", cmap="viridis")
        for level, colour, name in (
            (float(data["census_level"]), "white", "census saddle"),
            (float(data["raster_level"]), "orange", "raster binding"),
        ):
            if np.nanmin(field) <= level <= np.nanmax(field):
                axis.contour(
                    radius,
                    height,
                    field,
                    levels=[level],
                    colors=[colour],
                    linewidths=1.5,
                )
            axis.plot([], [], color=colour, label=name)
        differing = data["differing_cells"]
        if len(differing):
            axis.scatter(
                coordinate[differing, 0],
                coordinate[differing, 1],
                marker="x",
                s=55,
                color="red",
                label="differing cell",
            )
        wall = data["wall"]
        axis.plot(wall[:, 0], wall[:, 1], color="black", linewidth=0.8)
        wall_differing = data["differing_wall"]
        if len(wall_differing):
            axis.scatter(
                wall[wall_differing, 0],
                wall[wall_differing, 1],
                facecolors="none",
                edgecolors="magenta",
                s=65,
                label="differing wall node",
            )
        axis.set(
            title=str(record["identity"]),
            xlabel="R [m]",
            ylabel="Z [m]",
            aspect="equal",
        )
        axis.legend(loc="best", fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run(
    cache: Path = DEFAULT_CACHE,
    qualification_path: Path = DEFAULT_QUALIFICATION,
    output: Path = DEFAULT_OUTPUT,
    figure: Path = DEFAULT_FIGURE,
) -> dict[str, object]:
    """Replay every schema-capable row and persist a fail-closed receipt."""
    configure_dtypes()
    generator = _generator_module()
    source_identity = generator._source_authority(generator.MAST_AUTHORITY)[
        "source_identity"
    ]
    bank_rows = generator._read_cache(cache, source_identity)
    qualifications = _qualification_rows(qualification_path)
    records = []
    plot_rows = []
    for row in bank_rows:
        identity = str(row["identity"])
        record, plot = _replay_row(row, qualifications.get(identity))
        records.append(record)
        if plot is not None:
            plot_rows.append((record, plot))
    receipt: dict[str, object] = {
        "schema": "nova.topology-cell-parity",
        "cache": str(cache.resolve()),
        "cache_source_identity": source_identity,
        "qualification_metadata": str(qualification_path.resolve()),
        "marginal_rule": (
            "use solver_qualification.marginal_solver_basin when present; "
            "otherwise converged=false is the interim marginal flag"
        ),
        "row_count": len(records),
        "replayable_row_count": sum(bool(row["replayable"]) for row in records),
        "not_replayable_row_count": sum(not bool(row["replayable"]) for row in records),
        "rows": records,
    }
    errors = receipt_errors(receipt)
    receipt["validation_errors"] = errors
    receipt["passes"] = not errors and all(
        (not row["replayable"])
        or (
            row["marginal_solver_basin"]
            or (
                row["selected_primaries"]["axis"]["matches"]
                and row["selected_primaries"]["x_point"]["matches"]
                and row["classification"]["matches"]
                and row["differing_cell_count"] == 0
            )
        )
        for row in records
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    if plot_rows:
        _plot(plot_rows, figure)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--qualification", type=Path, default=DEFAULT_QUALIFICATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    arguments = parser.parse_args()
    receipt = run(
        arguments.cache, arguments.qualification, arguments.output, arguments.figure
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if not receipt["passes"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
