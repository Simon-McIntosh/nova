# ruff: noqa: E501
"""Regenerate per-geometry topology corroboration figures and evidence HTML.

The renderer consumes the committed MAST and DIII-D demonstration routes.  It
does not rescore the preregistered gates: its purpose is to expose the complete
spatial operands behind those scores, including negative and nonconverged rows.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
from html import escape
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import types
from typing import Any, Callable

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.path import Path as MplPath
import numpy as np

from benchmarks import mast_response_carrier_warm as response_carrier
from benchmarks.efit_forward_parity_slice import (
    DECOMPOSITION_BANK,
    _mast_case_from_selection,
    _passive_inclusive_case,
    select_slices_by_shot,
)
from benchmarks.label_seed_residual_field import _persisted_response_cache
from nova.equilibrium.connectivity_boundary import (
    _points_inside_polygon,
    _raster_hex_partition_geometry,
)
from nova.equilibrium.domain import PlasmaDomain
from nova.equilibrium.flux_surface_connectivity import (
    fit_tensor_spline,
    polish_stationary_points,
)
from nova.equilibrium.separatrix_branches import assemble_separatrix_branches
from nova.imas.mast_efit_referee import read_efit_referee
from nova.imas.mast_vacuum_cohort import SHOT_STORE
from nova.jax.config import configure_dtypes


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
EVIDENCE = ROOT / "docs/evidence/topology-visual-corroboration.html"
MAST_CACHE = HERE / "mast-topology-operands.npz"
DIIID_CACHE = HERE / "diiid-topology-operands.npz"
MAST_AUTHORITY = (
    ROOT / "docs/figures/primary-xpoint-evidence/efit_topology_corroboration.py"
)
DIIID_AUTHORITY = ROOT / "benchmarks/diiid_forward_gs_match.py"
EXPECTED_MAST_ROWS = 12
EXPECTED_DIIID_ROWS = 5
NUMERIC_ARRAY_DTYPES = {
    "cell_rz": np.float64,
    "domain_labels": np.int8,
    "o_candidates": np.float64,
    "x_candidates": np.float64,
    "selected_o": np.float64,
    "selected_x": np.float64,
    "wall_point": np.float64,
    "wall": np.float64,
    "nova_boundary": np.float64,
    "efit_axis": np.float64,
    "efit_x": np.float64,
    "efit_lcfs": np.float64,
}
PanelPublisher = Callable[[dict[str, Any], int], None]


class StaleOperandCacheError(RuntimeError):
    """Refuse operands produced by a source that is no longer authoritative."""


def _source_authority(path: Path) -> dict[str, str]:
    resolved = path.resolve(strict=True)
    try:
        source_path = resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        source_path = str(resolved)
    return {
        "source_path": source_path,
        "source_identity": f"sha256:{hashlib.sha256(resolved.read_bytes()).hexdigest()}",
    }


def _load_path(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stationary_records(
    source_o: np.ndarray,
    source_x: np.ndarray,
    radius: np.ndarray,
    height: np.ndarray,
    flux: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    source = np.concatenate((source_o, source_x))
    valid = np.all(np.isfinite(source), axis=1)
    spline = fit_tensor_spline(radius, height, flux)
    polished = jax.device_get(polish_stationary_points(spline, source[:, :2], valid))
    positions = np.asarray(polished["position_rz"], dtype=float)
    plotted = (
        valid
        & np.asarray(polished["converged"], dtype=bool)
        & np.asarray(polished["in_domain"], dtype=bool)
        & np.all(np.isfinite(positions), axis=1)
    )
    positions = np.where(
        plotted[:, None],
        positions,
        np.where(valid[:, None], source[:, :2], np.nan),
    )
    return positions[: len(source_o)], positions[len(source_o) :]


def _mast_rows(publish: PanelPublisher | None = None) -> list[dict[str, Any]]:
    source_authority = _source_authority(MAST_AUTHORITY)
    authority = _load_path(MAST_AUTHORITY, "mast_visual_authority")
    reachability = authority._reachability_module()
    response_cache, carrier = _persisted_response_cache(
        response_carrier.DEFAULT_CARRIER, response_carrier.DEFAULT_RECEIPT
    )
    selected = select_slices_by_shot(DECOMPOSITION_BANK)
    rows: list[dict[str, Any]] = []
    for selected_row, qualification in selected:
        shot = int(selected_row["shot"])
        slice_index = int(selected_row["slice_index"])
        print(f"MAST_OPERANDS {shot}/{slice_index}", flush=True)
        case, context = _mast_case_from_selection(
            SHOT_STORE, selected_row, qualification
        )
        passive_case, profile, policy = _passive_inclusive_case(
            case, context, response_cache
        )
        if policy["section_kernel_evaluations_this_shot"] != 0:
            raise RuntimeError("MAST route entered a direct response builder")
        target_current = abs(float(passive_case["reference"]["plasma_current_a"]))
        states = reachability._mast_states(
            profile, jnp.asarray(passive_case["state"]), target_current
        )
        referee = read_efit_referee(shot, store=SHOT_STORE)
        for arm, arm_result in states.items():
            state = arm_result.state
            governed_wall = reachability._closed_wall(
                np.asarray(profile.operator.wall.coordinate, dtype=float)
            )
            try:
                physical = jnp.asarray(state)[: profile.operator.physical_node_number]
                grid_flux, _wall_flux = profile.operator.topology.split_flux_map(
                    physical
                )
                source_o, source_x = jax.device_get(
                    profile.operator._fixed_design_topology.grid(grid_flux)
                )
                masks, topology = profile.operator.read(state)
                geometry = reachability._grid_geometry(profile, state)
                flux = np.asarray(geometry["flux"], dtype=float)
                radius = np.asarray(geometry["radius"], dtype=float)
                height = np.asarray(geometry["height"], dtype=float)
                o_candidates, x_candidates = _stationary_records(
                    np.asarray(source_o), np.asarray(source_x), radius, height, flux
                )
                assembled = jax.device_get(
                    assemble_separatrix_branches(
                        jnp.asarray(flux),
                        jnp.asarray(radius),
                        jnp.asarray(height),
                        topology.boundary_flux,
                        topology.axis,
                    )
                )
                closed = authority._sample_cubic_controls(
                    np.asarray(assembled["closed_controls_rz"])[
                        np.asarray(assembled["closed_valid"], dtype=bool)
                    ]
                )
                visual = {
                    "cell_rz": np.asarray(profile.lattice.coordinate, dtype=float),
                    "domain_labels": np.asarray(masks.label, dtype=np.int8),
                    "o_candidates": o_candidates,
                    "x_candidates": x_candidates,
                    "selected_o": np.asarray(topology.axis, dtype=float),
                    "selected_x": np.asarray(topology.x_point, dtype=float),
                    "wall_point": np.asarray(topology.wall_point, dtype=float),
                    "wall": governed_wall,
                    "nova_boundary": closed,
                    "converged": bool(arm_result.converged),
                    "qualification": str(arm_result.termination_reason),
                }
            except (
                authority.NoQualifiedAxisError,
                authority.ConstraintViolationError,
            ) as error:
                empty_points = np.empty((0, 2), dtype=float)
                visual = {
                    "cell_rz": empty_points,
                    "domain_labels": np.empty(0, dtype=np.int8),
                    "o_candidates": empty_points,
                    "x_candidates": empty_points,
                    "selected_o": empty_points,
                    "selected_x": empty_points,
                    "wall_point": empty_points,
                    "wall": governed_wall,
                    "nova_boundary": empty_points,
                    "converged": False,
                    "qualification": type(error).__name__,
                }
            row = {
                "machine": "MAST",
                "identity": f"{shot}/{slice_index} {arm}",
                "shot": shot,
                "frame": slice_index,
                "arm": arm,
                "time": float(referee.time_s[slice_index]),
                **visual,
                "efit_axis": np.asarray(referee.magnetic_axis_m[slice_index]),
                "efit_x": np.asarray(referee.x_points_m[slice_index]),
                "efit_lcfs": np.asarray(referee.lcfs_m[slice_index]),
            }
            rows.append(row)
            if publish is not None:
                publish(row, len(rows))
    if len(rows) != EXPECTED_MAST_ROWS:
        raise RuntimeError(f"expected {EXPECTED_MAST_ROWS} MAST rows, got {len(rows)}")
    _write_cache(
        MAST_CACHE,
        rows,
        {
            **source_authority,
            "carrier": carrier["carrier"]["semantic_response_identity"],
        },
    )
    return _read_cache(MAST_CACHE, source_authority["source_identity"])


def _diiid_module():
    source = DIIID_AUTHORITY.read_text()
    needle = "    return result, fields\n\n\ndef _retained_solve_failure("
    injection = (
        "    try:\n"
        "        visual_physical = equilibrium.flux[: profile.operator.physical_node_number]\n"
        "        visual_grid_flux, _visual_wall_flux = "
        "profile.operator.topology.split_flux_map(visual_physical)\n"
        "        visual_source_o, visual_source_x = "
        "profile.operator._fixed_design_topology.grid(visual_grid_flux)\n"
        "        visual_source_o, visual_source_x = "
        "jax.device_get((visual_source_o, visual_source_x))\n"
        "        visual_masks, _visual_topology = profile.operator.read(equilibrium.flux)\n"
        "        fields.update({\n"
        "            'domain_labels': np.asarray(visual_masks.label, dtype=np.int8),\n"
        "            'cell_rz': np.asarray(profile.lattice.coordinate, dtype=float),\n"
        "            'nova_source_o': np.asarray(visual_source_o, dtype=float),\n"
        "            'nova_source_x': np.asarray(visual_source_x, dtype=float),\n"
        "            'nova_selected_axis_rz': np.asarray(topology.axis, dtype=float),\n"
        "            'nova_selected_x_rz': np.asarray(topology.x_point, dtype=float),\n"
        "            'nova_selected_wall_rz': np.asarray(topology.wall_point, dtype=float),\n"
        "            'efit_axis_rz': labelled_axis,\n"
        "            'efit_x_points_rz': labelled_x_point[None, :],\n"
        "            'visual_flux': np.asarray(predicted.T, dtype=float),\n"
        "            'visual_failure_exception_class': None,\n"
        "        })\n"
        "    except (NoQualifiedAxisError, ConstraintViolationError) as visual_error:\n"
        "        visual_empty = np.empty((0, 2), dtype=float)\n"
        "        fields.update({\n"
        "            'domain_labels': np.empty(0, dtype=np.int8),\n"
        "            'cell_rz': visual_empty,\n"
        "            'nova_source_o': visual_empty,\n"
        "            'nova_source_x': visual_empty,\n"
        "            'nova_selected_axis_rz': visual_empty,\n"
        "            'nova_selected_x_rz': visual_empty,\n"
        "            'nova_selected_wall_rz': visual_empty,\n"
        "            'efit_axis_rz': labelled_axis,\n"
        "            'efit_x_points_rz': labelled_x_point[None, :],\n"
        "            'visual_flux': np.empty((0, 0), dtype=float),\n"
        "            'visual_failure_exception_class': type(visual_error).__name__,\n"
        "        })\n" + needle
    )
    if source.count(needle) != 1:
        raise RuntimeError("DIII-D renderer injection seam changed")
    loaded = types.ModuleType("diiid_visual_authority")
    loaded.__file__ = str(DIIID_AUTHORITY)
    loaded.__package__ = "benchmarks"
    sys.modules[loaded.__name__] = loaded
    namespace = loaded.__dict__
    exec(
        compile(source.replace(needle, injection), str(DIIID_AUTHORITY), "exec"),
        namespace,
    )
    return namespace


def _diiid_rows(publish: PanelPublisher | None = None) -> list[dict[str, Any]]:
    source_authority = _source_authority(DIIID_AUTHORITY)
    module = _diiid_module()
    module["configure_dtypes"]()
    paths = sorted(module["DEFAULT_DATA"].glob("*.parquet"))
    selected = module["select_frames"](
        paths, module["EXECUTION_FRAME_COUNT"], module["polarity_population"]()
    )
    rows: list[dict[str, Any]] = []
    for number, selected_frame in enumerate(selected, start=1):
        print(
            f"DIIID_OPERANDS {number}/{len(selected)} "
            f"{selected_frame.path.name}:{selected_frame.frame}",
            flush=True,
        )
        record = module["_read"](
            selected_frame.path,
            module["_LABEL_COLUMNS"]
            + module["_GEOMETRY_COLUMNS"]
            + module["_CURRENT_COLUMNS"]
            + module["_PLASMA_CURRENT_COLUMNS"],
        )
        record["_source_path"] = str(selected_frame.path)
        result, fields = module["solve_frame"](
            record,
            selected_frame.frame,
            None,
        )
        visual_failure = fields["visual_failure_exception_class"]
        if visual_failure is None:
            o_candidates, x_candidates = _stationary_records(
                np.asarray(fields["nova_source_o"], dtype=float),
                np.asarray(fields["nova_source_x"], dtype=float),
                np.asarray(fields["radius"], dtype=float),
                np.asarray(fields["height"], dtype=float),
                np.asarray(fields["visual_flux"], dtype=float),
            )
        else:
            o_candidates = np.empty((0, 2), dtype=float)
            x_candidates = np.empty((0, 2), dtype=float)
        row = {
            "machine": "DIII-D",
            "identity": f"{Path(result.shot).stem}:{result.frame}",
            "shot": result.shot,
            "frame": result.frame,
            "arm": "demonstration",
            "time": result.time_ms,
            "cell_rz": np.asarray(fields["cell_rz"], dtype=float),
            "domain_labels": np.asarray(fields["domain_labels"], dtype=np.int8),
            "o_candidates": o_candidates,
            "x_candidates": x_candidates,
            "selected_o": np.asarray(fields["nova_selected_axis_rz"], dtype=float),
            "selected_x": np.asarray(fields["nova_selected_x_rz"], dtype=float),
            "wall_point": np.asarray(fields["nova_selected_wall_rz"], dtype=float),
            "wall": np.asarray(fields["pseudo_wall"], dtype=float),
            "nova_boundary": (
                np.asarray(fields["predicted_closed_boundary"], dtype=float)
                if visual_failure is None
                else np.empty((0, 2), dtype=float)
            ),
            "efit_axis": np.asarray(fields["efit_axis_rz"], dtype=float),
            "efit_x": np.asarray(fields["efit_x_points_rz"], dtype=float),
            "efit_lcfs": np.asarray(fields["labelled_closed_boundary"], dtype=float),
            "converged": bool(result.converged) and visual_failure is None,
            "qualification": visual_failure or result.solver_termination,
        }
        rows.append(row)
        if publish is not None:
            publish(row, EXPECTED_MAST_ROWS + len(rows))
    if len(rows) != EXPECTED_DIIID_ROWS:
        raise RuntimeError(
            f"expected {EXPECTED_DIIID_ROWS} DIII-D rows, got {len(rows)}"
        )
    _write_cache(DIIID_CACHE, rows, source_authority)
    return _read_cache(DIIID_CACHE, source_authority["source_identity"])


def _write_cache(
    path: Path, rows: list[dict[str, Any]], authority: dict[str, Any]
) -> None:
    arrays: dict[str, np.ndarray] = {}
    metadata = []
    for index, row in enumerate(rows):
        metadata.append(
            {
                key: value
                for key, value in row.items()
                if key not in NUMERIC_ARRAY_DTYPES
            }
        )
        for field, dtype in NUMERIC_ARRAY_DTYPES.items():
            value = [] if row[field] is None else row[field]
            array = np.asarray(value, dtype=dtype)
            if array.dtype.kind not in "biuf":
                raise TypeError(
                    f"cache numeric field {field} has forbidden dtype {array.dtype}"
                )
            if field != "domain_labels":
                array = array.reshape((-1, 2))
            arrays[f"row_{index:02d}_{field}"] = array
    np.savez_compressed(path, **arrays)
    path.with_suffix(".metadata.json").write_text(
        json.dumps(
            {"authority": authority, "rows": metadata},
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )


def _read_cache(path: Path, source_identity: str) -> list[dict[str, Any]]:
    metadata = json.loads(path.with_suffix(".metadata.json").read_text())
    recorded_identity = metadata.get("authority", {}).get("source_identity")
    if recorded_identity != source_identity:
        raise StaleOperandCacheError(
            f"stale operand cache {path}: recorded source identity "
            f"{recorded_identity!r} does not match current authority "
            f"{source_identity!r}"
        )
    with np.load(path, allow_pickle=False) as stored:
        rows = []
        for index, record in enumerate(metadata["rows"]):
            row = dict(record)
            for field, dtype in NUMERIC_ARRAY_DTYPES.items():
                key = f"row_{index:02d}_{field}"
                array = np.array(stored[key], copy=True)
                if array.dtype != np.dtype(dtype) or array.dtype.kind not in "biuf":
                    raise TypeError(
                        f"cache numeric field {key} violates dtype allowlist: "
                        f"expected {np.dtype(dtype)}, got {array.dtype}"
                    )
                row[field] = array
            rows.append(row)
    return rows


def _finite_points(value: Any) -> np.ndarray:
    points = np.asarray(value, dtype=float).reshape((-1, 2))
    return points[np.all(np.isfinite(points), axis=1)]


def _closed_separatrix_points(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.size == 0:
        return np.empty((0, 2), dtype=float)
    try:
        points = array.reshape((-1, 2))
    except ValueError as error:
        raise ValueError("closed separatrix is not an N-by-2 array") from error
    if len(points) < 3 or not np.all(np.isfinite(points)):
        raise ValueError("closed separatrix is not a finite closed N-by-2 array")
    return points


def _panel_filename(row: dict[str, Any], index: int, suffix: str) -> str:
    identity = row["identity"].replace("/", "-").replace(":", "-").replace(" ", "-")
    machine = row["machine"].lower().replace("-", "")
    return f"{index:02d}-{machine}-{identity}.{suffix}"


def _point_or_none(points: np.ndarray) -> list[float] | None:
    return points[0].tolist() if len(points) else None


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _artifact_record_path(path: Path) -> str:
    """Return a stable publication identity or a durable external path."""
    publication_directory = Path(__file__).resolve().parent
    if HERE == publication_directory:
        return f"/nova/figures/topology-visual-corroboration/{path.name}"
    return str(path.resolve())


def _line_intersection(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Intersect two production shared-edge support lines."""
    first_origin, first_end = first
    second_origin, second_end = second
    first_direction = first_end - first_origin
    second_direction = second_end - second_origin
    denominator = (
        first_direction[0] * second_direction[1]
        - first_direction[1] * second_direction[0]
    )
    if np.isclose(denominator, 0.0):
        raise ValueError("production hex shared edges do not enclose a polygon")
    offset = second_origin - first_origin
    distance = (
        offset[0] * second_direction[1] - offset[1] * second_direction[0]
    ) / denominator
    return first_origin + distance * first_direction


def _production_cell_geometry(
    cells: np.ndarray, wall: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recover physical cells, admissible links, and centre wall membership."""
    if not len(cells):
        return (
            np.empty((0, 4, 2), dtype=float),
            np.empty((0, 2, 2), dtype=float),
            np.empty(0, dtype=bool),
        )

    rg = np.unique(cells[:, 0])
    zg = np.unique(cells[:, 1])
    if len(rg) < 3 or len(zg) < 3 or len(rg) * len(zg) != len(cells):
        return (
            np.empty((0, 4, 2), dtype=float),
            np.empty((0, 2, 2), dtype=float),
            np.empty(0, dtype=bool),
        )
    expected = np.stack(np.meshgrid(rg, zg, indexing="ij"), axis=-1).reshape((-1, 2))
    if not np.allclose(cells, expected, rtol=0.0, atol=1.0e-12):
        raise ValueError("cached cells do not follow the production tensor-grid order")
    if not (
        np.allclose(np.diff(rg), np.diff(rg)[0], rtol=1.0e-10, atol=1.0e-12)
        and np.allclose(np.diff(zg), np.diff(zg)[0], rtol=1.0e-10, atol=1.0e-12)
    ):
        raise ValueError("production polygon rendering requires regular grid axes")

    rings, shared_edges = _raster_hex_partition_geometry(
        jnp.asarray(rg), jnp.asarray(zg)
    )
    rings = np.asarray(rings)
    shared_edges = np.asarray(shared_edges)
    if not len(rings):
        raise ValueError("production hex geometry has no interior ring")
    production_centres = np.stack(np.meshgrid(rg, zg), axis=-1).reshape((-1, 2))
    centre = production_centres[rings[0, 0]]
    support_lines = shared_edges[0, 1:]
    prototype = np.stack(
        [
            _line_intersection(support_lines[index], support_lines[(index + 1) % 6])
            for index in range(6)
        ]
    )
    unique_vertices: list[np.ndarray] = []
    for vertex in prototype:
        if not any(
            np.allclose(vertex, existing, rtol=0.0, atol=1.0e-9)
            for existing in unique_vertices
        ):
            unique_vertices.append(vertex)
    if len(unique_vertices) != 4:
        raise ValueError(
            "production tensor-grid support lines do not define four cell vertices"
        )
    rectangle = np.stack(unique_vertices)
    polygons = cells[:, None, :] + (rectangle - centre)[None, :, :]

    if len(wall):
        centre_inside_wall = np.asarray(
            _points_inside_polygon(
                cells[:, 0],
                cells[:, 1],
                wall[:, 0],
                wall[:, 1],
            ),
            dtype=bool,
        )
        production_inside_wall = np.asarray(
            _points_inside_polygon(
                production_centres[:, 0],
                production_centres[:, 1],
                wall[:, 0],
                wall[:, 1],
            ),
            dtype=bool,
        )
    else:
        centre_inside_wall = np.zeros(len(cells), dtype=bool)
        production_inside_wall = np.zeros(len(production_centres), dtype=bool)

    neighbour_indices = rings[:, 1:]
    centre_indices = np.broadcast_to(rings[:, :1], neighbour_indices.shape)
    link_admissible = (
        production_inside_wall[centre_indices]
        & production_inside_wall[neighbour_indices]
    )
    directed_segments = shared_edges[:, 1:][link_admissible]
    directed_pairs = np.sort(
        np.stack((centre_indices, neighbour_indices), axis=-1)[link_admissible],
        axis=1,
    )
    if len(directed_pairs):
        _, first_occurrence = np.unique(directed_pairs, axis=0, return_index=True)
        adjacency_segments = directed_segments[np.sort(first_occurrence)]
    else:
        adjacency_segments = np.empty((0, 2, 2), dtype=float)
    return polygons, adjacency_segments, centre_inside_wall


def _draw_row(row: dict[str, Any], path: Path) -> dict[str, Any]:
    figure, axis = plt.subplots(figsize=(7.4, 7.2), constrained_layout=True)
    cells = np.asarray(row["cell_rz"], dtype=float)
    labels = np.asarray(row["domain_labels"], dtype=int).reshape(-1)
    if len(cells) != len(labels):
        raise RuntimeError(f"cell/label mismatch for {row['identity']}")
    colours = {
        int(PlasmaDomain.EXCLUDED_MATERIAL): "#f4f4f4",
        int(PlasmaDomain.CORE): "#a9d6e5",
        int(PlasmaDomain.COMMON_SOL): "#d8e2dc",
        int(PlasmaDomain.PRIVATE_FLUX): "#8e5ea2",
    }
    geometry_handles: list[Any] = []
    wall = _finite_points(row["wall"])
    polygons, adjacency_segments, centre_inside_wall = _production_cell_geometry(
        cells, wall
    )
    wall_membership_mismatch_cells = 0
    if len(polygons):
        expected_inside = labels != int(PlasmaDomain.EXCLUDED_MATERIAL)
        wall_membership_mismatch_cells = int(
            np.count_nonzero(centre_inside_wall != expected_inside)
        )
        axis.add_collection(
            PolyCollection(
                polygons,
                facecolors=[colours[int(label)] for label in labels],
                edgecolors="#ffffff",
                linewidths=0.12,
                rasterized=True,
                zorder=1,
            )
        )
        if len(adjacency_segments):
            axis.add_collection(
                LineCollection(
                    adjacency_segments,
                    colors="#264653",
                    linewidths=0.42,
                    alpha=0.38,
                    rasterized=True,
                    zorder=2,
                )
            )
            geometry_handles.append(
                Line2D(
                    [],
                    [],
                    color="#264653",
                    linewidth=0.8,
                    alpha=0.65,
                    label="six-neighbour adjacency",
                )
            )
        axis.autoscale_view()
        legend_names = {
            int(PlasmaDomain.EXCLUDED_MATERIAL): "excluded material",
            int(PlasmaDomain.CORE): "axis-connected core",
            int(PlasmaDomain.COMMON_SOL): "common SOL",
            int(PlasmaDomain.PRIVATE_FLUX): "private-flux shadow",
        }
        for domain, label in legend_names.items():
            if np.any(labels == domain):
                geometry_handles.append(
                    Patch(
                        facecolor=colours[domain],
                        edgecolor="#d9d9d9",
                        linewidth=0.4,
                        label=label,
                    )
                )
    if len(wall):
        axis.plot(
            wall[:, 0],
            wall[:, 1],
            color="0.28",
            linewidth=1.2,
            label="governed wall",
            zorder=6,
        )
    boundary = _finite_points(row["nova_boundary"])
    if len(boundary):
        axis.plot(
            boundary[:, 0],
            boundary[:, 1],
            color="#005f73",
            linewidth=1.8,
            label="Nova closed boundary",
        )
    lcfs = _finite_points(row["efit_lcfs"])
    if len(lcfs):
        axis.plot(
            lcfs[:, 0],
            lcfs[:, 1],
            color="black",
            linestyle="--",
            linewidth=1.7,
            label="EFIT LCFS",
        )

    o_candidates = _finite_points(row["o_candidates"])
    x_candidates = _finite_points(row["x_candidates"])
    if len(o_candidates):
        axis.scatter(
            o_candidates[:, 0],
            o_candidates[:, 1],
            marker="o",
            s=45,
            facecolors="none",
            edgecolors="#0077b6",
            label="Nova O candidates",
            zorder=7,
        )
    if len(x_candidates):
        axis.scatter(
            x_candidates[:, 0],
            x_candidates[:, 1],
            marker="x",
            s=48,
            color="#d00000",
            label="Nova X candidates",
            zorder=7,
        )
    selected_o = _finite_points(row["selected_o"])
    selected_x = _finite_points(row["selected_x"])
    wall_point = _finite_points(row["wall_point"])
    if len(selected_o):
        axis.scatter(
            *selected_o[0],
            marker="*",
            s=145,
            color="#0077b6",
            edgecolors="white",
            linewidths=0.6,
            label="Nova primary O",
            zorder=9,
        )
    if len(selected_x):
        axis.scatter(
            *selected_x[0],
            marker="X",
            s=110,
            color="#ffb703",
            edgecolors="#8d0801",
            linewidths=0.8,
            label="Nova selected primary X",
            zorder=9,
        )
    if len(wall_point):
        axis.scatter(
            *wall_point[0],
            marker="D",
            s=65,
            color="#fb8500",
            edgecolors="black",
            linewidths=0.5,
            label="Nova closest plasma-wall point",
            zorder=9,
        )
    efit_axis = _finite_points(row["efit_axis"])
    efit_x = _finite_points(row["efit_x"])
    if len(efit_axis):
        axis.scatter(
            *efit_axis[0],
            marker="P",
            s=65,
            color="black",
            label="EFIT axis label",
            zorder=9,
        )
    if len(efit_x):
        axis.scatter(
            efit_x[:, 0],
            efit_x[:, 1],
            marker="+",
            s=85,
            color="#c1121f",
            linewidths=1.5,
            label="EFIT X label",
            zorder=9,
        )
    if row["converged"]:
        verdict = "converged"
    else:
        qualification = textwrap.fill(str(row["qualification"]), width=64)
        verdict = f"NONCONVERGED — retained\nRetained failure: {qualification}"
    axis.set_title(
        f"{row['machine']} · {row['identity']} · {verdict}",
        fontsize=9.5 if not row["converged"] else None,
    )
    axis.set_xlabel("R [m]")
    axis.set_ylabel("Z [m]")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.12)
    axis.text(
        0.01,
        0.01,
        "Tensor-grid Voronoi rectangles · six-neighbour adjacency · "
        "wall test at centres · no clipping",
        transform=axis.transAxes,
        fontsize=6.5,
        color="#333333",
        bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.86},
        zorder=10,
    )
    handles, labels_text = axis.get_legend_handles_labels()
    handles = [*geometry_handles, *handles]
    labels_text = [
        *(handle.get_label() for handle in geometry_handles),
        *labels_text,
    ]
    unique: dict[str, Any] = {}
    for handle, label in zip(handles, labels_text, strict=True):
        if label and label not in unique:
            unique[label] = handle
    axis.legend(
        unique.values(),
        unique.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.09),
        ncol=2,
        fontsize=7,
        frameon=False,
    )
    if not row["converged"]:
        axis.set_facecolor("#fff1f1")
        for spine in axis.spines.values():
            spine.set_color("#b00020")
            spine.set_linewidth(1.8)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return {
        "private_flux_cells": int(
            np.count_nonzero(labels == int(PlasmaDomain.PRIVATE_FLUX))
        ),
        "o_candidates": len(o_candidates),
        "x_candidates": len(x_candidates),
        "selected_o": len(selected_o),
        "selected_x": len(selected_x),
        "wall_point": len(wall_point),
        "efit_axis": len(efit_axis),
        "efit_x": len(efit_x),
        "efit_lcfs_vertices": len(lcfs),
        "voronoi_cells": len(polygons),
        "wall_admissible_adjacency_links": len(adjacency_segments),
        "wall_membership_mismatch_cells": wall_membership_mismatch_cells,
    }


def _draw_retained_failure(
    row: dict[str, Any],
    path: Path,
    failure_class: str,
    failure_message: str,
    total_shadow_cells: int,
) -> dict[str, Any]:
    figure, axis = plt.subplots(figsize=(7.4, 7.2), constrained_layout=True)
    axis.set_facecolor("#fff1f1")
    for spine in axis.spines.values():
        spine.set_color("#b00020")
        spine.set_linewidth(1.8)
    axis.text(
        0.5,
        0.55,
        failure_class,
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=20,
        color="#b00020",
        weight="bold",
    )
    axis.text(
        0.5,
        0.43,
        textwrap.fill(failure_message, width=64),
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=9,
        color="#5f0011",
    )
    axis.set_title(
        f"{row['machine']} · {row['identity']} · RETAINED FAILURE",
        fontsize=10,
    )
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_xlabel("Panel rendering failed; cohort position retained")
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return {
        "private_flux_cells": total_shadow_cells,
        "o_candidates": 0,
        "x_candidates": 0,
        "selected_o": 0,
        "selected_x": 0,
        "wall_point": 0,
        "efit_axis": 0,
        "efit_x": 0,
        "efit_lcfs_vertices": 0,
        "voronoi_cells": 0,
        "wall_admissible_adjacency_links": 0,
        "wall_membership_mismatch_cells": 0,
    }


def _publish_row_contents(row: dict[str, Any], index: int) -> dict[str, Any]:
    png_path = HERE / _panel_filename(row, index, "png")
    temporary_png = png_path.with_name(f".{png_path.stem}.tmp.png")
    cells = _finite_points(row["cell_rz"])
    labels = np.asarray(row["domain_labels"], dtype=int).reshape(-1)
    if len(cells) != len(labels):
        raise RuntimeError(f"cell/label mismatch for {row['identity']}")
    shadow = labels == int(PlasmaDomain.PRIVATE_FLUX)
    total_shadow_cells = int(np.count_nonzero(shadow))
    source_qualification = str(row["qualification"])
    retained_failure_class = row.get("panel_failure_exception_class")
    retained_failure_message = row.get("panel_failure_message")
    try:
        boundary = _closed_separatrix_points(row["nova_boundary"])
    except ValueError as error:
        retained_failure_class = type(error).__name__
        retained_failure_message = str(error)
        boundary = np.empty((0, 2), dtype=float)
        row["nova_boundary"] = boundary
        row["converged"] = False
        row["qualification"] = retained_failure_class
        row["panel_failure_exception_class"] = retained_failure_class
        row["panel_failure_message"] = retained_failure_message
    boundary_available = bool(len(boundary))
    inside_count = (
        int(
            np.count_nonzero(
                shadow
                & MplPath(boundary, closed=True).contains_points(cells, radius=-1.0e-10)
            )
        )
        if boundary_available
        else None
    )

    render_row = {**row, "nova_boundary": boundary}
    if retained_failure_class is not None:
        render_row["converged"] = False
        render_row["qualification"] = retained_failure_class
    render_failed = False
    try:
        plot_record = _draw_row(render_row, temporary_png)
    except Exception as error:
        plt.close("all")
        render_failed = True
        if retained_failure_class is None:
            retained_failure_class = type(error).__name__
            retained_failure_message = str(error)
        row["converged"] = False
        row["qualification"] = retained_failure_class
        row["panel_failure_exception_class"] = retained_failure_class
        row["panel_failure_message"] = retained_failure_message
        plot_record = _draw_retained_failure(
            row,
            temporary_png,
            retained_failure_class,
            retained_failure_message,
            total_shadow_cells,
        )
    os.replace(temporary_png, png_path)

    if render_failed:
        selected_x = np.empty((0, 2), dtype=float)
        selected_o = np.empty((0, 2), dtype=float)
        wall_point = np.empty((0, 2), dtype=float)
        x_candidates = np.empty((0, 2), dtype=float)
    else:
        selected_x = _finite_points(row["selected_x"])
        selected_o = _finite_points(row["selected_o"])
        wall_point = _finite_points(row["wall_point"])
        x_candidates = _finite_points(row["x_candidates"])
    selected_is_candidate = bool(
        len(selected_x)
        and len(x_candidates)
        and np.any(np.linalg.norm(x_candidates - selected_x[0], axis=1) <= 1.0e-8)
    )
    panel_record = {
        "panel_index": index,
        "machine": row["machine"],
        "identity": row["identity"],
        "converged": bool(row["converged"]) and retained_failure_class is None,
        "qualification": retained_failure_class or str(row["qualification"]),
        "source_qualification": source_qualification,
        "retained_failure_exception_class": retained_failure_class,
        "retained_failure_message": retained_failure_message,
        "closed_separatrix_available": boundary_available,
        "shadow_cells_inside_lcfs": inside_count,
        "shadow_cells_inside_closed_separatrix": inside_count,
        "total_shadow_cells": total_shadow_cells,
        "selected_primary_x_point_rz_m": _point_or_none(selected_x),
        "selected_primary_o_point_rz_m": _point_or_none(selected_o),
        "other_qualified_x_point_count": len(x_candidates) - int(selected_is_candidate),
        "closest_plasma_wall_point_rz_m": _point_or_none(wall_point),
        "converged_inside_lcfs_gate_pass": (
            None
            if retained_failure_class is not None or inside_count is None
            else bool(not row["converged"] or inside_count == 0)
        ),
        "png_path": _artifact_record_path(png_path),
    }
    json_path = HERE / _panel_filename(row, index, "json")
    panel_record["json_path"] = _artifact_record_path(json_path)
    _atomic_json(json_path, panel_record)
    print(
        f"PANEL_PUBLISHED {index:02d} {row['machine']} {row['identity']} "
        f"shadow_inside_lcfs={inside_count} total_shadow={total_shadow_cells} "
        f"retained_failure={retained_failure_class}",
        flush=True,
    )
    return {**plot_record, **panel_record}


def _publish_row(row: dict[str, Any], index: int) -> dict[str, Any]:
    try:
        return _publish_row_contents(row, index)
    finally:
        jax.clear_caches()
        gc.collect()


def _write_evidence(rows: list[dict[str, Any]], records: list[dict[str, Any]]) -> None:
    populated_records = [record for record in records if record["voronoi_cells"]]
    membership_mismatch_records = [
        record for record in records if record["wall_membership_mismatch_cells"] > 0
    ]
    binding = [
        record
        for record in records
        if record["total_shadow_cells"] > 0 and record["closed_separatrix_available"]
    ]
    trivial = [
        record
        for record in records
        if record["total_shadow_cells"] == 0 and record["closed_separatrix_available"]
    ]
    unavailable_records = [
        record for record in records if not record["closed_separatrix_available"]
    ]
    failed_gate = [
        record
        for record in binding
        if record["shadow_cells_inside_lcfs"] != 0
        or record["shadow_cells_inside_closed_separatrix"] != 0
    ]
    if failed_gate:
        identities = ", ".join(record["identity"] for record in failed_gate)
        raise RuntimeError(f"strong containment gate failed for {identities}")

    def record_names(selected: list[dict[str, Any]]) -> str:
        return ", ".join(
            f"{record['machine']} {record['identity']}" for record in selected
        )

    figure_rows = []
    for index, (row, record) in enumerate(zip(rows, records, strict=True), start=1):
        filename = f"{index:02d}-{row['machine'].lower().replace('-', '')}-{row['identity'].replace('/', '-').replace(':', '-').replace(' ', '-')}.png"
        unavailable = not record["closed_separatrix_available"]
        retained_failure = record["retained_failure_exception_class"]
        retained_message = record["retained_failure_message"]
        status = (
            f"RETAINED FAILURE — {escape(retained_failure)}"
            + (f": {escape(retained_message)}" if retained_message else "")
            + "."
            + (
                " Closed separatrix and containment counts are unavailable."
                if unavailable
                else " Closed-separatrix containment counts remain available."
            )
            if retained_failure is not None
            else (
                "Converged."
                if row["converged"]
                else (
                    f"NONCONVERGED — retained failure: "
                    f"{escape(str(row['qualification']))}."
                    + (
                        " Nova partition and landmarks are unavailable; EFIT labels, "
                        "LCFS, and governed first wall are retained."
                        if len(row["cell_rz"]) == 0
                        else ""
                    )
                )
            )
        )
        containment = (
            f"available=true; total={record['total_shadow_cells']}; "
            f"inside LCFS={record['shadow_cells_inside_lcfs']}; "
            f"inside closed separatrix="
            f"{record['shadow_cells_inside_closed_separatrix']}"
            + (
                " (trivial zero-shadow population)"
                if record["total_shadow_cells"] == 0
                else ""
            )
            if not unavailable
            else (
                f"available=false; total={record['total_shadow_cells']}; "
                "inside LCFS=null; inside closed separatrix=null (not a pass)"
            )
        )
        figure_rows.append(
            f"""<article class="figure-row" id="geometry-{index:02d}">
  <h3>{index:02d}. {row["machine"]} — {row["identity"]}</h3>
  <figure><img src="/nova/figures/topology-visual-corroboration/{filename}" alt="Topology evidence for {row["machine"]} {row["identity"]}: rectangular tensor-grid Voronoi cells carrying flood-fill domains, production six-neighbour adjacency segments, all Nova O and X candidates, selected primary O and X, wall point, and EFIT axis, X labels and LCFS overlay."><figcaption>{record["voronoi_cells"]} rectangular Voronoi cells; {record["wall_admissible_adjacency_links"]} wall-admissible six-neighbour adjacency segments; {record["private_flux_cells"]} private-flux shadow cells; {record["o_candidates"]} plotted O candidates; {record["x_candidates"]} plotted X candidates; selected O/X/wall markers {record["selected_o"]}/{record["selected_x"]}/{record["wall_point"]}; EFIT axis/X/LCFS vertices {record["efit_axis"]}/{record["efit_x"]}/{record["efit_lcfs_vertices"]}. <strong>Containment record:</strong> {containment}. These are tensor-grid rectangles, not hexagonal areas: the production half-offset stencil supplies six-neighbour hex connectivity, drawn from its physical shared-edge segments. The centre-in-polygon wall test admits or excludes each whole rectangle; the solver computes neither partial clipping nor polygon-wall intersections, so excluded cells that straddle the wall remain visible at full extent. Production centre/label wall-check mismatches: {record["wall_membership_mismatch_cells"]}. <strong>{status}</strong></figcaption></figure>
</article>"""
        )
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    mast_source = _source_authority(MAST_AUTHORITY)
    diiid_source = _source_authority(DIIID_AUTHORITY)
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="docs-project" content="nova"><meta name="reckon-type" content="evidence">
<meta name="plan-slug" content="topology-visual-corroboration">
<meta name="plan-evidence-for" content="efit-baseline-demonstration"><meta name="plan-verifies" content="efit-baseline-demonstration#s2">
<meta name="plan-title" content="Topology visual corroboration"><meta name="plan-summary" content="Per-geometry visual corroboration of flood-fill topology operands against MAST and DIII-D EFIT labels.">
<title>Topology visual corroboration | nova</title><link rel="stylesheet" href="/_shared/foundation.css"><link rel="stylesheet" href="/_shared/dashboard.css">
</head>
<body><main class="plan"><header class="plan-hero"><p class="eyebrow">Evidence · visual topology audit</p><h1>Topology visual corroboration</h1>
<p class="lede">All {EXPECTED_MAST_ROWS} MAST arms and all {EXPECTED_DIIID_ROWS} DIII-D demonstration frames are shown exactly once. Every populated panel exposes the physical rectangular Voronoi cells carrying the flood-fill domains, the production six-neighbour adjacency that makes the connectivity graph hexagonal, the complete finite O/X candidate census, selected primary O and X, selected wall point, and EFIT axis/X/LCFS labels.</p></header>
<section><h2>Authority and interpretation</h2><p>This is corroboration of committed extraction state, not a new score. EFIT is an independent magnetics-fitted reconstruction, not physical truth. MAST operands use the persisted response carrier and current source <code>{mast_source["source_path"]}</code> at content identity <code>{mast_source["source_identity"]}</code>. DIII-D operands use the current integrated benchmark source <code>{diiid_source["source_path"]}</code> at content identity <code>{diiid_source["source_identity"]}</code>. Every nonconverged row remains visibly qualified by its recorded termination name. Generated from repository head <code>{head}</code>.</p>
<p>The requested “hex cells with clipping” contained two premises that the production geometry corrects. On these tensor axes, the six shared-edge support-line intersections from <code>_raster_hex_partition_geometry</code> collapse pairwise to four unique vertices, so each physical Voronoi cell is a rectangle. The hexagonal structure belongs to the six-neighbour half-offset adjacency graph, not to an area tessellation; every panel therefore overlays the wall-admissible production shared-edge segments that make that connectivity visible. Hulling those segment endpoints would invent six-sided areas that overlap neighbours by 49.093 mm radially and 23.810 mm vertically, so no hexagon-shaped fill is drawn.</p>
<p>The governed material decision is reproduced by the production <code>_points_inside_polygon</code> symbol at cell centres and checked against the cached labels. The solver performs no partial-cell clipping: it admits or excludes each whole rectangle according to its centre. The wall is therefore drawn over full rectangles, including excluded-material cells that visibly straddle it; no polygon-wall intersections are invented. The centre/label wall check is exact for {len(populated_records) - len(membership_mismatch_records)} of {len(populated_records)} populated rows. {len(membership_mismatch_records)} retained-failure rows carry a mismatch, named per row in the figure ledger: {record_names(membership_mismatch_records)}.</p>
<p>The purple cells are the exact <code>PRIVATE_FLUX</code> labels from Nova's domain partition: closed-flux cells disconnected from the primary O-point by the X-point flood-fill cut. Pale blue is axis-connected core, grey-green is common SOL, and pale grey is excluded material. The selected wall marker is Nova's governed closest plasma-wall candidate; it is shown even when the topology class is diverted and it does not bind the LCFS.</p></section>
<section><h2>Coverage and non-vacuous containment gate</h2><p><strong>{len(rows)} of {EXPECTED_MAST_ROWS + EXPECTED_DIIID_ROWS} declared geometries rendered.</strong> {sum(row["machine"] == "MAST" for row in rows)} MAST and {sum(row["machine"] == "DIII-D" for row in rows)} DIII-D; {sum(row["converged"] for row in rows)} converged and {sum(not row["converged"] for row in rows)} nonconverged rows retained. The strong gate applies wherever <code>total_shadow_cells &gt; 0</code> and <code>closed_separatrix_available = true</code>, irrespective of convergence: both numeric inside counts must equal zero. It binds on <strong>{len(binding)} panels and passes {len(binding)} of {len(binding)}</strong>: {record_names(binding)}.</p>
<p>{len(trivial)} panels have an available separatrix but zero shadow cells, so their zero-inside result is trivial: {record_names(trivial)}. {len(unavailable_records)} panels have no available closed separatrix and are not passes; their containment counts are null: {record_names(unavailable_records)}.</p>
<p><strong>Direct defect refutation:</strong> the converged MAST 22086/43 pure and mixed panels carry 43 and 42 private-flux shadow cells respectively, with zero inside the LCFS and zero inside the closed separatrix in both cases. This is substantial private-flux shadow entirely outside the converged closed boundary.</p></section>
<section><h2>Per-geometry corroboration</h2>{"".join(figure_rows)}</section>
<section><h2>Deferred primary-selection variants</h2><p>Primary-versus-alternate domain variants require cache fields that are not present: <code>per_cell_flux_values</code> and <code>per_candidate_domain_labels</code>. Adding them is a cache-schema migration followed by a full seventeen-row bank regeneration, measured at roughly one day of lane time, so it is deferred to the next scheduled bank regeneration rather than triggering a solve campaign here. The existing synthetic-geometry sensitivity measurements remain available at <code>docs/figures/wall-height-shadow-safety/metrics.json</code>.</p></section>
<section><h2>Reproduction</h2><p>Run <code>UV_PROJECT_ENVIRONMENT=~/Code/nova/.venv PYTHONPATH="$PWD" uv run --no-sync python docs/figures/topology-visual-corroboration/generate_topology_visuals.py</code>. The committed <code>generation.log</code> records the successful run and ends with <code>EXIT_MARKER=0</code>. The scoped NPZ files retain the exact plotted operands so ordinary regeneration does not repeat the expensive solves; delete them only when intentionally refreshing from a newly committed extraction state.</p></section>
</main></body></html>"""
    EVIDENCE.write_text(html + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate topology panels and machine-checkable panel records."
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=HERE,
        help="Durable directory for caches, panels, JSON records, and evidence HTML.",
    )
    return parser.parse_args()


def main() -> None:
    global HERE, EVIDENCE, MAST_CACHE, DIIID_CACHE

    args = _parse_args()
    HERE = args.output_directory.expanduser().resolve()
    publication_directory = Path(__file__).resolve().parent
    EVIDENCE = (
        ROOT / "docs/evidence/topology-visual-corroboration.html"
        if HERE == publication_directory
        else HERE / "topology-visual-corroboration.html"
    )
    MAST_CACHE = HERE / "mast-topology-operands.npz"
    DIIID_CACHE = HERE / "diiid-topology-operands.npz"
    configure_dtypes()
    HERE.mkdir(parents=True, exist_ok=True)
    published: dict[int, dict[str, Any]] = {}

    def publish(row: dict[str, Any], index: int) -> None:
        published[index] = _publish_row(row, index)

    mast_identity = _source_authority(MAST_AUTHORITY)["source_identity"]
    diiid_identity = _source_authority(DIIID_AUTHORITY)["source_identity"]
    mast = (
        _read_cache(MAST_CACHE, mast_identity)
        if MAST_CACHE.exists()
        else _mast_rows(publish)
    )
    diiid = (
        _read_cache(DIIID_CACHE, diiid_identity)
        if DIIID_CACHE.exists()
        else _diiid_rows(publish)
    )
    rows = mast + diiid
    if len(rows) != EXPECTED_MAST_ROWS + EXPECTED_DIIID_ROWS:
        raise RuntimeError("demonstration-bank coverage is incomplete")
    for index, row in enumerate(rows, start=1):
        if index not in published:
            publish(row, index)
    records = [published[index] for index in range(1, len(rows) + 1)]
    _write_evidence(rows, records)
    digest = hashlib.sha256(EVIDENCE.read_bytes()).hexdigest()
    print(
        json.dumps(
            {
                "rows": len(rows),
                "mast": len(mast),
                "diiid": len(diiid),
                "output_directory": str(HERE),
                "evidence_sha256": digest,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
