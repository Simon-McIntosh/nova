"""Replay committed MAST topology operands through the production cell read."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import time
from collections.abc import Iterable, Mapping
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
from nova.equilibrium.topology import NoQualifiedAxisError
from nova.geometry.hexstencil import hex_stencil
from nova.jax.config import configure_dtypes
from nova.media.ink import DEFAULT_INK, poloidal_axes
from nova.media.poloidal import draw_nulls, draw_wall

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = (
    ROOT / "docs/figures/topology-visual-corroboration/mast-topology-operands.npz"
)
# Governed marginal flags are read from every receipt that carries them; a row
# covered by more than one receipt must agree across all of them.
DEFAULT_QUALIFICATION = (
    ROOT / "docs/figures/gs-absolute-accuracy/efit-reproduction.json",
    ROOT / "docs/figures/solver-convergence-regression/bank-rebaseline-regen.json",
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
    """Index a receipt's governed solver-qualification rows by arm identity.

    A receipt states its rows either at the root or under ``data``; a row names
    its arm through ``frame_identity.label`` or through ``identity`` plus
    ``arm``. A row without an explicit ``solver_qualification`` mapping is not
    a governed flag and is skipped rather than inferred from convergence.
    """
    if not path.exists():
        raise FileNotFoundError(f"governed qualification receipt absent: {path}")
    payload = json.loads(path.read_text())
    rows = payload.get("rows", payload.get("data", {}).get("rows", []))
    indexed = {}
    for row in rows:
        qualification = row.get("solver_qualification")
        if not isinstance(qualification, dict):
            continue
        label = row.get("frame_identity", {}).get("label")
        if label is None and row.get("identity") and row.get("arm"):
            label = f"{row['identity']} {row['arm']}"
        if label is not None:
            indexed[str(label)] = qualification
    return indexed


def governed_qualifications(paths: Iterable[Path]) -> dict[str, dict[str, object]]:
    """Merge governed qualification rows, failing closed on a disagreement.

    The marginal flag is the governed verdict and must agree wherever two
    receipts both state it. The achieved class is not governed and receipts are
    observed to differ on it, so every value is kept per receipt for the row to
    report rather than one of them being chosen by argument order.
    """
    merged: dict[str, dict[str, object]] = {}
    for path in paths:
        for label, qualification in _qualification_rows(path).items():
            marginal = qualification.get("marginal_solver_basin")
            if label in merged:
                stated = merged[label]["qualification"].get("marginal_solver_basin")
                if stated != marginal:
                    raise RuntimeError(
                        f"{label}: governed receipts disagree on "
                        f"marginal_solver_basin between "
                        f"{merged[label]['source']} and {path}"
                    )
            else:
                merged[label] = {
                    "qualification": qualification,
                    "source": f"{path}:solver_qualification",
                    "variants": {},
                }
            merged[label]["variants"][Path(path).as_posix()] = {
                "marginal_solver_basin": marginal,
                "achieved_class": qualification.get("achieved_class"),
            }
    return merged


def cache_authority(cache: Path) -> dict[str, object]:
    """Name the operand cache by digest and operand base revision."""
    resolved = cache.resolve()
    metadata = json.loads(resolved.with_suffix(".metadata.json").read_text())
    return {
        "cache": str(resolved),
        "cache_sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
        "cache_source_identity": metadata.get("authority", {}).get("source_identity"),
        "operand_base_revision": metadata.get("base_revision"),
        "operand_producer_commit": metadata.get("producer_commit"),
    }


def committed_class_authority(
    row: dict[str, object],
    qualification: dict[str, object] | None,
    variants: Mapping[str, Mapping[str, object]],
    committed_x,
) -> tuple[str, str, dict[str, object], bool]:
    """The committed class, who stated it, every stated value, and any conflict.

    The committed operand's own recorded class is the authority because the
    operand is what is being replayed. Governed receipts' ``achieved_class``
    values are retained beside it rather than discarded: where two receipts
    state different classes, choosing one hides the disagreement, and a parity
    verdict that turns on which receipt was merged first is not a verdict.
    """
    operand_class = row.get("class") or row.get("solve_topology_class")
    governed_class = qualification.get("achieved_class") if qualification else None
    if operand_class:
        committed, source = str(operand_class), "operand bank metadata class"
    elif governed_class:
        committed, source = str(governed_class), "governed receipt achieved_class"
    elif np.all(np.isfinite(np.asarray(committed_x, dtype=float))):
        committed, source = "diverted", "committed selected X is finite"
    else:
        committed, source = "limited", "committed selected X is not finite"
    records: dict[str, object] = {
        "operand_metadata": str(operand_class) if operand_class else None,
    }
    records.update(
        {path: variant.get("achieved_class") for path, variant in variants.items()}
    )
    stated = {value for value in records.values() if value is not None}
    return committed, source, records, len(stated) > 1


def marginal_status(qualification: dict[str, object] | None) -> tuple[bool | None, str]:
    """Read the governed flag, keeping an absent or null flag unknown.

    Both an absent key and an explicit JSON null mean the receipt has stated no
    verdict, so neither may be coerced to False: a ``bool(None)`` here would
    publish a non-marginal verdict that the receipt never gave, and a panel
    resting on it would report exact parity it never established.
    """
    if qualification is None or "marginal_solver_basin" not in qualification:
        return None, "missing solver_qualification.marginal_solver_basin"
    value = qualification["marginal_solver_basin"]
    if value is None:
        return None, "null solver_qualification.marginal_solver_basin"
    return bool(value), "solver_qualification.marginal_solver_basin"


def differing_cell_indices(committed_labels, replayed_labels) -> np.ndarray:
    """Return every cell index whose committed and replayed labels differ."""
    committed = np.asarray(committed_labels)
    replayed = np.asarray(replayed_labels)
    if committed.shape != replayed.shape:
        raise ValueError("committed and replayed labels must share a shape")
    return np.flatnonzero(committed != replayed)


def differing_cell_records(
    indices, coordinate, committed_labels, replayed_labels, values, adjudicate
) -> list[dict[str, object]]:
    """Build one adjudicated record per differing cell, absorbing none."""
    records = []
    for index in np.asarray(indices).reshape(-1):
        index = int(index)
        record = {
            "index": index,
            "centroid_m": np.asarray(coordinate)[index].tolist(),
            "committed_label": int(np.asarray(committed_labels)[index]),
            "committed_label_name": PlasmaDomain(
                int(np.asarray(committed_labels)[index])
            ).name,
            "replayed_label": int(np.asarray(replayed_labels)[index]),
            "replayed_label_name": PlasmaDomain(
                int(np.asarray(replayed_labels)[index])
            ).name,
        }
        record.update(adjudicate(float(np.asarray(values)[index])))
        records.append(record)
    return records


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
    coverage_keys = (
        "row_count",
        "replayed_row_count",
        "unavailable_row_count",
        "not_replayable_row_count",
    )
    if all(key in receipt for key in coverage_keys):
        if receipt["row_count"] != len(rows):
            errors.append("row_count does not match the declared rows")
        accounted = (
            receipt["replayed_row_count"]
            + receipt["unavailable_row_count"]
            + receipt["not_replayable_row_count"]
        )
        if accounted != receipt["row_count"]:
            errors.append("every declared row must be replayed or named unavailable")
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
        if row["marginal_solver_basin"] is None:
            errors.append(
                f"{row['identity']}: {row['marginal_flag_source']} — an absent or "
                "null governed flag is not a non-marginal verdict"
            )
        if not row["replayable"]:
            if not row.get("not_replayable_reason"):
                errors.append(f"{row['identity']}: missing not-replayable reason")
            continue
        if row.get("replay_completed") is False:
            if not row.get("replay_exception") or not row.get("unavailable_reason"):
                errors.append(f"{row['identity']}: unavailable replay lacks evidence")
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


def _replay_row(row, governed):
    started = time.monotonic()
    identity = str(row["identity"])
    if governed is None:
        qualification, variants, (marginal, source) = None, {}, marginal_status(None)
    else:
        qualification = governed["qualification"]
        variants = governed["variants"]
        marginal, _ = marginal_status(qualification)
        source = str(governed["source"])
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
        record["figure_panel"] = "unavailable: no cached per-cell flux"
        record["wall_seconds"] = time.monotonic() - started
        return record, _annotation_payload(record)

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
    try:
        masks, state, _connected, admitted = operator._fixed_design_read(physical)
        if not bool(admitted):
            raise NoQualifiedAxisError(
                f"{identity}: production topology read found no qualified axis"
            )
    except NoQualifiedAxisError as error:
        record.update(
            replayable=True,
            replay_completed=False,
            replay_exception=type(error).__name__,
            unavailable_reason=str(error),
            disposition="unavailable production topology read",
            figure_panel=f"unavailable: {type(error).__name__}",
            wall_seconds=time.monotonic() - started,
        )
        return record, _unavailable_payload(record, coordinate, values, committed, wall)
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
    differing = differing_cell_indices(committed, replayed)
    cells = differing_cell_records(
        differing,
        coordinate,
        committed,
        replayed,
        values,
        adjudicate=lambda flux: adjudicate_difference(
            flux,
            float(state.axis_flux),
            float(state.boundary_flux),
            raster_flux,
        ),
    )

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
    committed_class, class_source, class_records, class_disagreement = (
        committed_class_authority(row, qualification, variants, committed_x)
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
        committed_class=committed_class,
        committed_class_source=class_source,
        committed_class_records=class_records,
        committed_class_disagreement=class_disagreement,
        wall_node_census={
            "node_count": int(len(wall)),
            "cell_authority_private_count": int(np.count_nonzero(cell_private)),
            "retained_raster_private_count": int(np.count_nonzero(raster_private)),
            "differing_node_count": int(len(wall_differing)),
            "differing_nodes": wall_rows,
        },
    )
    record.update(
        replay_completed=True,
        figure_panel="replayed",
        wall_seconds=time.monotonic() - started,
    )
    record["disposition"] = parity_disposition(record)
    plot = {
        "coordinate": coordinate,
        "values": values,
        "wall": wall,
        "differing_cells": differing,
        "differing_wall": wall_differing,
        "census_level": np.asarray(float(state.boundary_flux)),
        "raster_level": np.asarray(raster_flux),
        "census_nulls": {
            "magnetic_axis": np.asarray(state.axis, dtype=float),
            "x_points": np.atleast_2d(np.asarray(state.x_point, dtype=float)),
        },
        "committed_nulls": {
            "magnetic_axis": np.asarray(committed_o, dtype=float),
            "x_points": np.atleast_2d(np.asarray(committed_x, dtype=float)),
        },
    }
    return record, plot


def _unavailable_payload(record, coordinate, values, committed, wall):
    """Geometry for an unavailable row: committed operands, no replay result."""
    return {
        "coordinate": coordinate,
        "values": values,
        "wall": wall,
        "differing_cells": np.empty(0, dtype=int),
        "differing_wall": np.empty(0, dtype=int),
        "census_level": np.asarray(np.nan),
        "raster_level": np.asarray(np.nan),
        "census_nulls": {},
        "committed_nulls": {},
        "annotation": (
            f"{record['identity']}: {record['replay_exception']} — "
            f"{record['unavailable_reason']}"
        ),
    }


def _annotation_payload(record):
    """A declared row with no drawable geometry still gets its own panel."""
    return {
        "coordinate": None,
        "values": None,
        "wall": None,
        "differing_cells": np.empty(0, dtype=int),
        "differing_wall": np.empty(0, dtype=int),
        "census_level": np.asarray(np.nan),
        "raster_level": np.asarray(np.nan),
        "census_nulls": {},
        "committed_nulls": {},
        "annotation": f"{record['identity']}: {record['not_replayable_reason']}",
    }


def panel_labels(rows) -> list[str]:
    """One visible panel label per declared row, replayed or annotated."""
    labels = []
    for record, data in rows:
        label = str(record["identity"])
        if data.get("annotation"):
            label = f"{label} — {data['annotation']}"
        labels.append(label)
    return labels


def _panel_levels(field: np.ndarray, count: int = 12) -> np.ndarray:
    """Stated interior contour levels spanning one panel's own flux range."""
    values = np.asarray(field, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return np.empty(0)
    low, high = float(np.min(values)), float(np.max(values))
    if not high > low:
        return np.asarray([low])
    return np.linspace(low, high, count + 2)[1:-1]


def _finite_null(point) -> np.ndarray | None:
    """The first two coordinates of a finite null, or nothing to draw."""
    if point is None:
        return None
    value = np.asarray(point, dtype=float).reshape(-1)
    if value.size < 2:
        return None
    value = value[:2]
    return value if bool(np.all(np.isfinite(value))) else None


def draw_topology_nulls(
    axis, census_nulls, committed_nulls, style=DEFAULT_INK
) -> dict[str, int]:
    """Draw the replayed null set filled and the committed set hollow.

    Both sets reach the canvas, because a parity panel whose replayed axis has
    moved must show that it moved. The replayed axis and saddle are drawn
    through :func:`draw_nulls`; the committed saddle arrives as that painter's
    ``other_x_points``, and the committed axis is drawn here, hollow, in the
    same style, because the painter takes a single axis and the second set is
    distinguished from the first by fill rather than by absence.

    Returns the drawn counts so a coverage check can assert per panel that the
    committed axis was drawn rather than only named in a caption.
    """
    tally = {
        "replayed_axis_drawn": 0,
        "committed_axis_drawn": 0,
        "replayed_x_points_drawn": 0,
        "committed_x_points_drawn": 0,
    }
    if census_nulls:
        drawn = draw_nulls(
            axis,
            magnetic_axis=census_nulls.get("magnetic_axis"),
            x_points=census_nulls.get("x_points"),
            other_x_points=committed_nulls.get("x_points"),
            style=style,
        )
        tally["replayed_x_points_drawn"] = int(drawn["x_points_drawn"])
        tally["committed_x_points_drawn"] = int(drawn["other_x_points_drawn"])
        tally["replayed_axis_drawn"] = int(
            _finite_null(census_nulls.get("magnetic_axis")) is not None
        )
    committed_axis = (
        _finite_null(committed_nulls.get("magnetic_axis")) if committed_nulls else None
    )
    if committed_axis is not None:
        axis.plot(
            committed_axis[0],
            committed_axis[1],
            marker=style.axis_marker,
            markersize=style.axis_markersize,
            color=style.axis_color,
            markerfacecolor="none",
            linestyle="none",
            zorder=style.zorder_markers,
        )
        tally["committed_axis_drawn"] = 1
    return tally


def _plot(rows, path):
    """One poloidal panel per declared row: line contours, nulls, wall, no axes."""
    column_count = min(4, len(rows))
    row_count = int(np.ceil(len(rows) / column_count))
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(4.3 * column_count, 5.2 * row_count),
        squeeze=False,
        constrained_layout=True,
        facecolor=DEFAULT_INK.figure_facecolor,
    )
    flat_axes = axes.reshape(-1)
    for axis, (record, data) in zip(flat_axes, rows, strict=False):
        poloidal_axes(axis)
        wall = data["wall"]
        if data["coordinate"] is None:
            axis.text(
                0.5,
                0.5,
                data["annotation"],
                ha="center",
                va="center",
                wrap=True,
                color=DEFAULT_INK.wall_color,
            )
        else:
            coordinate = data["coordinate"]
            radius, height = _tensor_axes(coordinate)
            field = np.asarray(data["values"]).reshape((radius.size, height.size)).T
            axis.contour(
                radius,
                height,
                field,
                levels=_panel_levels(field),
                colors=[DEFAULT_INK.contour_color],
                linewidths=DEFAULT_INK.contour_linewidth,
            )
            levels = (
                (float(data["census_level"]), "cyan", "-", 1.5),
                (float(data["raster_level"]), "#ff8c00", "--", 1.2),
            )
            for level, colour, style, width in levels:
                if np.isfinite(level) and np.nanmin(field) <= level <= np.nanmax(field):
                    axis.contour(
                        radius,
                        height,
                        field,
                        levels=[level],
                        colors=[colour],
                        linestyles=[style],
                        linewidths=width,
                    )
                axis.plot([], [], color=colour, linestyle=style, label=f"{level:.4f}")
            if len(data["differing_cells"]):
                point = coordinate[data["differing_cells"]]
                axis.scatter(
                    point[:, 0],
                    point[:, 1],
                    marker="x",
                    s=55,
                    color="#cc0000",
                    label="differing cell",
                )
            draw_wall(axis, units=wall)
            census = data["census_nulls"]
            if census:
                draw_topology_nulls(
                    axis, census, data["committed_nulls"], style=DEFAULT_INK
                )
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
            if data.get("annotation"):
                axis.set_title(data["annotation"], fontsize=7, color="#cc0000")
        axis.set_title(str(record["identity"]), fontsize=9)
        axis.legend(loc="upper right", fontsize=6, frameon=False)
    for axis in flat_axes[len(rows) :]:
        axis.set_visible(False)
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
    qualifications = governed_qualifications(qualification_path)
    records, plots = [], []
    for row in bank_rows:
        record, plot = _replay_row(row, qualifications.get(str(row["identity"])))
        records.append(record)
        plots.append((record, plot))
    replayable = [row for row in records if row["replayable"]]
    completed = [row for row in replayable if row.get("replay_completed") is True]
    missing_operands = [row for row in records if not row["replayable"]]
    unavailable = [row for row in replayable if row.get("replay_completed") is False]
    exact = [
        row
        for row in completed
        if row["marginal_solver_basin"] is False
        and row["differing_cell_count"] == 0
        and all(item["matches"] for item in row["selected_primaries"].values())
        and not row["classification"]["finding"]
    ]
    pending = [row for row in replayable if row["marginal_solver_basin"] is None]
    receipt = {
        "schema": "nova.topology-cell-parity",
        **cache_authority(cache),
        "qualification_receipts": [
            Path(path).resolve().as_posix() for path in qualification_path
        ],
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
        "replayed_row_count": len(completed),
        "unavailable_row_count": len(unavailable),
        "not_replayable_row_count": len(missing_operands),
        "marginal_unknown_row_count": sum(
            row["marginal_solver_basin"] is None for row in records
        ),
        "pending_replayable_row_count": len(pending),
        "exact_non_marginal_row_count": sum(
            row["marginal_solver_basin"] is False for row in replayable
        ),
        "exact_non_marginal_pass_count": len(exact),
        "committed_class_authority": (
            "the operand bank's own recorded class; governed receipts' "
            "achieved_class values are kept per row as a cross-check"
        ),
        "committed_class_disagreement_count": sum(
            row.get("committed_class_disagreement") is True for row in completed
        ),
        "committed_class_disagreements": [
            {
                "identity": row["identity"],
                "committed_class": row["committed_class"],
                "records": row["committed_class_records"],
            }
            for row in completed
            if row.get("committed_class_disagreement") is True
        ],
        "rows": records,
    }
    finalize_receipt(receipt, pending)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    if plots:
        _plot(plots, figure)
    return receipt


def finalize_receipt(receipt: dict[str, object], pending: bool) -> dict[str, object]:
    """Attach the verdict a receipt is published under.

    ``passes`` is false whenever any validation error stands, so a run that
    leaves a governed row unknown cannot publish a passing receipt however
    many rows it replayed; ``status`` records whether the replay itself ran to
    the end, which is a different fact and is not a substitute for it.
    """
    errors = receipt_errors(receipt)
    receipt.update(
        validation_errors=errors,
        status="pending" if pending and not errors else "complete",
        passes=not errors,
    )
    return receipt


def exit_code(receipt: dict[str, object]) -> int:
    """Zero only for a complete receipt that carries no validation error."""
    if receipt.get("passes") is True and not receipt.get("validation_errors"):
        return 0 if receipt.get("status") == "complete" else 1
    return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--qualification",
        type=Path,
        nargs="+",
        default=list(DEFAULT_QUALIFICATION),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    arguments = parser.parse_args()
    receipt = run(
        arguments.cache, arguments.qualification, arguments.output, arguments.figure
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    status = exit_code(receipt)
    if status:
        raise SystemExit(status)


if __name__ == "__main__":
    main()
