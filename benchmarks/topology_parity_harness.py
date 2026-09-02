"""Replay committed MAST topology operands through the production cell read."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from nova.biot.null import Null1D, Null2D
from nova.biot.target import FluxTarget
from nova.equilibrium.connectivity_boundary import (
    _points_inside_polygon,
    traced_boundary_read,
)
from nova.equilibrium.domain import PlasmaDomain
from nova.equilibrium.flux_surface_connectivity import fit_tensor_spline
from nova.equilibrium.forward_operator import ForwardFluxOperator
from nova.equilibrium.source import DomainProfile, ForwardSource
from nova.equilibrium.stencil_mesh import MomentGeometry, StencilMesh
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
    if not path.exists():
        return {}
    rows = json.loads(path.read_text()).get("data", {}).get("rows", [])
    return {
        str(row.get("frame_identity", {}).get("label")): row["solver_qualification"]
        for row in rows
        if row.get("frame_identity", {}).get("label")
        and isinstance(row.get("solver_qualification"), dict)
    }


def marginal_status(qualification: dict[str, object] | None) -> tuple[bool | None, str]:
    """Read the governed flag and fail closed when it is absent."""
    if qualification is not None and "marginal_solver_basin" in qualification:
        return bool(
            qualification["marginal_solver_basin"]
        ), "solver_qualification.marginal_solver_basin"
    return None, "missing solver_qualification.marginal_solver_basin"


def parity_disposition(row: dict[str, object]) -> str:
    if not row.get("replayable", False):
        return "not replayable"
    if row.get("marginal_solver_basin") is None:
        return "pending marginal qualification"
    if row["marginal_solver_basin"]:
        return "marginal finding"
    return "exact non-marginal parity"


def adjudicate_difference(flux, axis_flux, census_boundary_flux, raster_boundary_flux):
    """Explain one cell under the two boundary authorities."""
    census_norm = (flux - axis_flux) / (census_boundary_flux - axis_flux)
    raster_norm = (flux - axis_flux) / (raster_boundary_flux - axis_flux)
    census_closed = bool(census_norm <= 1.0)
    raster_closed = bool(raster_norm <= 1.0)
    return {
        "psi_norm_census_saddle": float(census_norm),
        "psi_norm_raster_binding": float(raster_norm),
        "census_closed": census_closed,
        "raster_closed": raster_closed,
        "adjudication": "binding-level difference"
        if census_closed != raster_closed
        else "connectivity-cut difference",
    }


def _margin_value(value: float) -> float | str:
    if np.isposinf(value):
        return "+Infinity"
    if np.isneginf(value):
        return "-Infinity"
    return float(value)


def adjudicate_classification(
    *,
    committed,
    cell_authority,
    retained_raster,
    cell_class_margin,
    retained_raster_class_margin,
    cell_boundary_flux,
    retained_raster_boundary_flux,
    marginal,
):
    """Record the two class margins and boundary authorities."""
    committed_match = committed == cell_authority
    authority_match = cell_authority == retained_raster
    finding = not (committed_match and authority_match)
    if finding:
        text = (
            f"production cell authority is {cell_authority} at boundary flux "
            f"{cell_boundary_flux:.17g} with class margin "
            f"{_margin_value(cell_class_margin)}; "
            f"retained raster is {retained_raster} at binding flux "
            f"{retained_raster_boundary_flux:.17g} with class margin "
            f"{_margin_value(retained_raster_class_margin)}"
        )
    else:
        text = "committed, cell-authority, and retained raster classes agree"
    return {
        "committed": committed,
        "replayed_cell_authority": cell_authority,
        "retained_raster": retained_raster,
        "matches_committed": committed_match,
        "cell_raster_matches": authority_match,
        "finding": finding,
        "cell_class_margin": _margin_value(cell_class_margin),
        "retained_raster_class_margin": _margin_value(retained_raster_class_margin),
        "cell_boundary_flux": float(cell_boundary_flux),
        "retained_raster_boundary_flux": float(retained_raster_boundary_flux),
        "gate": "marginal finding"
        if marginal is True and finding
        else "pending marginal qualification"
        if marginal is None
        else "exact non-marginal parity",
        "adjudication": text,
    }


def receipt_errors(receipt: dict[str, object]) -> list[str]:
    """Validate evidence while leaving unknown marginal rows pending."""
    errors = []
    if receipt.get("schema") != "nova.topology-cell-parity":
        errors.append("unexpected schema")
    rows = receipt.get("rows")
    if not isinstance(rows, list):
        return [*errors, "rows must be a list"]
    required = {
        "identity",
        "replayable",
        "marginal_solver_basin",
        "marginal_flag_source",
        "disposition",
    }
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
        if row["marginal_solver_basin"] is False:
            if row.get("differing_cell_count") != 0:
                errors.append(f"{row['identity']}: non-marginal labels differ")
            if not all(
                item.get("matches", False)
                for item in row.get("selected_primaries", {}).values()
            ):
                errors.append(f"{row['identity']}: non-marginal primary differs")
            if row.get("classification", {}).get("finding", True):
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


def _zero_profile(psi_norm):
    return jnp.zeros_like(psi_norm)


def _tensor_axes(coordinate):
    radius = np.unique(coordinate[:, 0])
    height = np.unique(coordinate[:, 1])
    expected = np.c_[np.repeat(radius, height.size), np.tile(height, radius.size)]
    if coordinate.shape != expected.shape or not np.array_equal(coordinate, expected):
        raise RuntimeError("cached topology cells are not a tensor carrier")
    return radius, height


def _cell_polygons(value, count):
    padded = np.asarray(value, dtype=float)
    if padded.ndim != 3 or padded.shape[0] != count or padded.shape[2] != 2:
        raise RuntimeError("cached cell polygons do not match the flux carrier")
    polygons = tuple(cell[np.all(np.isfinite(cell), axis=1)] for cell in padded)
    if any(len(cell) < 3 for cell in polygons):
        raise RuntimeError("cached cell polygons contain a degenerate cell")
    return polygons


def _polygon_area(polygons):
    return np.asarray(
        [
            0.5
            * abs(
                np.dot(cell[:, 0], np.roll(cell[:, 1], -1))
                - np.dot(cell[:, 1], np.roll(cell[:, 0], -1))
            )
            for cell in polygons
        ]
    )


def _production_operator(coordinate, polygons, wall, inside):
    """Construct production topology from the row's committed polygons."""
    radius, height = _tensor_axes(coordinate)
    stencil = hex_stencil((radius.size, height.size))
    area = _polygon_area(polygons)
    geometry = MomentGeometry.from_cells(
        StencilMesh(coordinate, stencil, area), polygons
    )
    operator = ForwardFluxOperator(
        grid=FluxTarget(
            jnp.zeros((len(coordinate), 1)),
            jnp.zeros((len(coordinate), 1)),
            Null2D.from_coordinates(coordinate, stencil, maxsize=12),
        ),
        wall=FluxTarget(
            jnp.zeros((len(wall), 1)),
            jnp.zeros((len(wall), 1)),
            Null1D(jnp.asarray(wall)),
        ),
        source=ForwardSource(
            core=DomainProfile(p_prime=_zero_profile, ff_prime=_zero_profile)
        ),
        external_current=jnp.zeros(1),
        area=jnp.asarray(area),
        polarity=1,
        inside_material=jnp.asarray(inside),
        moment_geometry=geometry,
        use_linear_moments=False,
    )
    return operator


def _nearest_candidate(candidates, point):
    finite = np.all(np.isfinite(candidates), axis=1)
    distance = np.linalg.norm(candidates - point, axis=1)
    return int(np.argmin(np.where(finite, distance, np.inf)))


def _retained_raster_read(values, radius, height, inside, axis, x_candidates, wall):
    """Evaluate the retained raster boundary and class diagnostic."""
    shape = (radius.size, height.size)
    field = jnp.asarray(values.reshape(shape).T)
    surface = fit_tensor_spline(jnp.asarray(radius), jnp.asarray(height), field)
    wall_flux = surface(jnp.asarray(wall[:, 0]), jnp.asarray(wall[:, 1]))
    wall_state = Null1D(jnp.asarray(wall))(wall_flux, 1)
    finite_x = x_candidates[np.all(np.isfinite(x_candidates), axis=1)]
    x_flux = surface(jnp.asarray(finite_x[:, 0]), jnp.asarray(finite_x[:, 1]))
    candidate_state = jnp.c_[jnp.asarray(finite_x), x_flux, jnp.zeros(len(finite_x))]
    return traced_boundary_read(
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
        wall_flux,
        classification_x=candidate_state,
        classification_wall=wall_state[:3],
    )


def _replay_row(row, qualification):
    identity = str(row["identity"])
    marginal, source = marginal_status(qualification)
    record = {
        "identity": identity,
        "shot": int(row["shot"]),
        "frame": int(row["frame"]),
        "arm": str(row["arm"]),
        "marginal_solver_basin": marginal,
        "marginal_flag_source": source,
    }
    coordinate = np.asarray(row["cell_rz"], dtype=float)
    values = np.asarray(row["per_cell_flux_values"], dtype=float)
    committed = np.asarray(row["domain_labels"], dtype=np.int8)
    if values.shape != (len(coordinate),) or not len(coordinate):
        record.update(
            replayable=False,
            not_replayable_reason="no cached per-cell flux for this bank row",
        )
        record["disposition"] = parity_disposition(record)
        return record, None

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
    operator = _production_operator(
        coordinate,
        _cell_polygons(row["current_cell_polygons"], len(coordinate)),
        wall,
        inside,
    )
    radius, height = _tensor_axes(coordinate)
    surface = fit_tensor_spline(
        jnp.asarray(radius),
        jnp.asarray(height),
        jnp.asarray(values.reshape((radius.size, height.size)).T),
    )
    wall_flux = np.asarray(surface(jnp.asarray(wall[:, 0]), jnp.asarray(wall[:, 1])))
    physical = jnp.asarray(np.r_[values, wall_flux])

    # Production owns candidate selection, edge reads, flood connectivity, and labels.
    masks, state, _connected, admitted = operator._fixed_design_read(physical)
    if not bool(admitted):
        raise RuntimeError(f"{identity}: production topology read rejected its axis")
    replayed = np.asarray(masks.label, dtype=np.int8)
    committed_o = np.asarray(row["selected_o"], dtype=float)[0]
    committed_x = np.asarray(row["selected_x"], dtype=float)[0]
    o_candidates = np.asarray(row["o_candidates"], dtype=float)
    x_candidates = np.asarray(row["x_candidates"], dtype=float)
    retained = _retained_raster_read(
        values, radius, height, inside, committed_o, x_candidates, wall
    )
    raster_flux = float(retained["psi_bnd"])
    raster_margin = float(retained["class_margin"])
    cell_margin = float(operator._connectivity_class_margin(physical, state))
    differing = np.flatnonzero(committed != replayed)
    cells = [
        {
            "index": int(index),
            "centroid_m": coordinate[index].tolist(),
            "committed_label": int(committed[index]),
            "committed_label_name": PlasmaDomain(int(committed[index])).name,
            "replayed_label": int(replayed[index]),
            "replayed_label_name": PlasmaDomain(int(replayed[index])).name,
            **adjudicate_difference(
                float(values[index]),
                float(state.axis_flux),
                float(state.boundary_flux),
                raster_flux,
            ),
        }
        for index in differing
    ]

    replay_o = np.asarray(state.axis, dtype=float)
    replay_x = np.asarray(state.x_point, dtype=float)
    primaries = {
        "axis": {
            "committed_candidate_index": _nearest_candidate(o_candidates, committed_o),
            "replayed_candidate_index": _nearest_candidate(o_candidates, replay_o),
            "matches": _nearest_candidate(o_candidates, committed_o)
            == _nearest_candidate(o_candidates, replay_o),
            "position_residual_m": float(np.linalg.norm(replay_o - committed_o)),
        },
        "x_point": {
            "committed_candidate_index": _nearest_candidate(x_candidates, committed_x),
            "replayed_candidate_index": _nearest_candidate(x_candidates, replay_x),
            "matches": _nearest_candidate(x_candidates, committed_x)
            == _nearest_candidate(x_candidates, replay_x),
            "position_residual_m": float(np.linalg.norm(replay_x - committed_x)),
        },
    }
    cell_class = "diverted" if bool(state.diverted) else "limited"
    raster_class = "diverted" if raster_margin >= 0 else "limited"
    committed_class = (
        str(qualification["achieved_class"])
        if qualification and qualification.get("achieved_class")
        else "diverted"
        if np.all(np.isfinite(committed_x))
        else "limited"
    )
    classification = adjudicate_classification(
        committed=committed_class,
        cell_authority=cell_class,
        retained_raster=raster_class,
        cell_class_margin=cell_margin,
        retained_raster_class_margin=raster_margin,
        cell_boundary_flux=float(state.boundary_flux),
        retained_raster_boundary_flux=raster_flux,
        marginal=marginal,
    )
    cell_private = np.asarray(
        operator._carrier_shadow_read(physical, masks)["private_wall_node_mask"],
        dtype=bool,
    )
    raster_private = np.asarray(retained["private_wall_node_mask"], dtype=bool)
    owner = np.asarray(operator._wall_carrier_index, dtype=int)
    wall_differing = np.flatnonzero(cell_private != raster_private)
    wall_rows = [
        {
            "index": int(index),
            "position_m": wall[index].tolist(),
            "nearest_cell_index": int(owner[index]),
            "nearest_cell_label": int(replayed[owner[index]]),
            "cell_authority_private": bool(cell_private[index]),
            "retained_raster_private": bool(raster_private[index]),
        }
        for index in wall_differing
    ]
    record.update(
        replayable=True,
        production_replay_call=(
            "ForwardFluxOperator._fixed_design_read on cached cell polygons"
        ),
        compared_cell_count=int(len(coordinate)),
        differing_cell_count=int(len(differing)),
        differing_cells=cells,
        census_axis_flux=float(state.axis_flux),
        census_saddle_flux=float(state.boundary_flux),
        raster_binding_flux=raster_flux,
        selected_primaries=primaries,
        classification=classification,
        wall_node_census={
            "node_count": int(len(wall)),
            "cell_authority_private_count": int(np.count_nonzero(cell_private)),
            "retained_raster_private_count": int(np.count_nonzero(raster_private)),
            "differing_node_count": int(len(wall_differing)),
            "differing_nodes": wall_rows,
        },
    )
    record["disposition"] = parity_disposition(record)
    plot = {
        "coordinate": coordinate,
        "values": values,
        "labels": replayed,
        "wall": wall,
        "differing_cells": differing,
        "differing_wall": wall_differing,
        "census_level": np.asarray(float(state.boundary_flux)),
        "raster_level": np.asarray(raster_flux),
    }
    return record, plot


def _plot(rows, path):
    figure, axes = plt.subplots(
        1,
        len(rows),
        figsize=(7 * len(rows), 6.2),
        squeeze=False,
        constrained_layout=True,
    )
    for axis, (record, data) in zip(axes[0], rows, strict=True):
        coordinate = data["coordinate"]
        radius, height = _tensor_axes(coordinate)
        labels = data["labels"].reshape((radius.size, height.size)).T
        field = data["values"].reshape((radius.size, height.size)).T
        axis.pcolormesh(radius, height, labels, shading="nearest", cmap="viridis")
        specs = (
            (float(data["raster_level"]), "#ff8c00", "raster binding", "--", 2.6),
            (float(data["census_level"]), "cyan", "census saddle", "-", 1.5),
        )
        for level, colour, name, style, width in specs:
            if np.nanmin(field) <= level <= np.nanmax(field):
                axis.contour(
                    radius,
                    height,
                    field,
                    levels=[level],
                    colors=[colour],
                    linestyles=[style],
                    linewidths=width,
                )
            axis.plot([], [], color=colour, linestyle=style, label=name)
        if len(data["differing_cells"]):
            point = coordinate[data["differing_cells"]]
            axis.scatter(
                point[:, 0],
                point[:, 1],
                marker="x",
                s=55,
                color="red",
                label="differing cell",
            )
        wall = data["wall"]
        axis.plot(wall[:, 0], wall[:, 1], color="black", linewidth=0.8)
        if len(data["differing_wall"]):
            point = wall[data["differing_wall"]]
            axis.scatter(
                point[:, 0],
                point[:, 1],
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
    cache=DEFAULT_CACHE,
    qualification_path=DEFAULT_QUALIFICATION,
    output=DEFAULT_OUTPUT,
    figure=DEFAULT_FIGURE,
):
    configure_dtypes()
    generator = _generator_module()
    source_identity = generator._source_authority(generator.MAST_AUTHORITY)[
        "source_identity"
    ]
    bank_rows = generator._read_cache(cache, source_identity)
    qualifications = _qualification_rows(qualification_path)
    records, plots = [], []
    for row in bank_rows:
        record, plot = _replay_row(row, qualifications.get(str(row["identity"])))
        records.append(record)
        if plot is not None:
            plots.append((record, plot))
    replayable = [row for row in records if row["replayable"]]
    exact = [
        row
        for row in replayable
        if row["marginal_solver_basin"] is False
        and row["differing_cell_count"] == 0
        and all(item["matches"] for item in row["selected_primaries"].values())
        and not row["classification"]["finding"]
    ]
    pending = [row for row in replayable if row["marginal_solver_basin"] is None]
    receipt = {
        "schema": "nova.topology-cell-parity",
        "cache": str(cache.resolve()),
        "cache_source_identity": source_identity,
        "qualification_metadata": str(qualification_path.resolve()),
        "marginal_rule": (
            "only solver_qualification.marginal_solver_basin is authoritative; "
            "a missing flag is marginal-unknown and pending"
        ),
        "production_replay": (
            "ForwardFluxOperator._fixed_design_read with MomentGeometry built "
            "from the row's committed cell polygons"
        ),
        "row_count": len(records),
        "replayable_row_count": len(replayable),
        "not_replayable_row_count": len(records) - len(replayable),
        "marginal_unknown_row_count": sum(
            row["marginal_solver_basin"] is None for row in records
        ),
        "pending_replayable_row_count": len(pending),
        "exact_non_marginal_row_count": sum(
            row["marginal_solver_basin"] is False for row in replayable
        ),
        "exact_non_marginal_pass_count": len(exact),
        "rows": records,
    }
    errors = receipt_errors(receipt)
    receipt.update(
        validation_errors=errors,
        status="pending" if pending and not errors else "complete",
        passes=not errors,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    if plots:
        _plot(plots, figure)
    return receipt


def main():
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
