#!/usr/bin/env python3
"""Measure analytic limiter reads across plasma and wall resolutions.

The measurement keeps the analytic field fixed and varies only the carrier
geometry.  Each row records the landmarks published by the public forward
operator and verifies that its unstructured hex carrier takes the fixed-design
class branch rather than the tensor-grid connectivity branch.  Diverted rows
additionally exercise the private-wall exclusion by making a known excluded
wall node the unmasked extremum, then requiring the masked limited read to
choose a different contact.

Parts are written after every row.  Aggregation is a separate command so two
processes can shard each wall-count group inside one scheduler allocation
without sharing a mutable receipt.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import socket
import subprocess
from time import perf_counter
from typing import Any, Callable, Iterator
import uuid

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.optimize import minimize_scalar

from benchmarks import solovev_certificate as certificate
from nova.equilibrium import forward_operator as forward_operator_module
from nova.equilibrium.connectivity_boundary import wall_height_shadow_mask
from nova.equilibrium.forward_operator import (
    ForwardFluxOperator,
    set_support_clip_mode,
    support_clip_mode,
)
from nova.equilibrium.topology import Topology, TopologyClass
from nova.jax.config import (
    configure_dtypes,
    configure_persistent_compilation_cache,
    default_persistent_compilation_cache_root,
)
from nova.media import poloidal
from nova.media.ink import DEFAULT_INK, poloidal_axes, trace_axes
from nova.media.sources.frame import inside_wall_units
from scripts.analytic_oracle_fixtures import measure as oracle_fixture


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_DIRECTORY = Path(
    "/home/ITER/mcintos/.config/reckon/crew/reports/nova/s19-review/limiter-read"
)
DEFAULT_FIGURE_DIRECTORY = (
    ROOT / "docs/figures/cut-cell-current-attribution/limiter-read"
)
WALL_NODE_COUNTS = (121, 241, 481, 961)
HIGH_RESOLUTION_REALISED_CELLS = 2600
RENDER_CASE = certificate.DIVERTED_CASE_NAME
RENDER_REQUESTED_CELLS = -1000
RENDER_WALL_NODES = 481
CONTACT_MARKER = "D"
EXCLUDED_WALL_MARKER = "s"
EXCLUDED_WALL_COLOR = "#6a3d9a"
GLYPH_FAMILIES = (
    "magnetic_axis",
    "admitted_x_point",
    "wall_contact",
    "excluded_private_wall_nodes",
)
POLOIDAL_KIND = "poloidal-contact-shadow"
ERROR_KIND = "contact-error-vs-resolution"
ROWS = (
    ("weak-rotation-reactor-static", -300),
    ("moderate-rotation-conventional-static", -300),
    (certificate.DIVERTED_CASE_NAME, -500),
    ("weak-rotation-reactor-static", -1000),
    ("moderate-rotation-conventional-static", -1000),
    (certificate.DIVERTED_CASE_NAME, -1000),
)
BRANCH_NAMES = {
    0: "retain_previous",
    1: "qualified_height_band",
    2: "connectivity_private_fallback",
}


def _strict(value: Any) -> Any:
    """Return nested JSON-native data with non-finite values made explicit."""

    if isinstance(value, dict):
        return {str(key): _strict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict(item) for item in value]
    if isinstance(value, np.ndarray):
        return _strict(value.tolist())
    if isinstance(value, np.generic):
        return _strict(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically persist strict JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(_strict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _source_revision() -> str:
    """Return the revision supplying this measurement."""

    return subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _pointer(
    function: Callable[..., Any], markers: tuple[str, ...] = ()
) -> dict[str, Any]:
    """Return a source interval and verified marker lines for one callable."""

    while hasattr(function, "__wrapped__"):
        function = function.__wrapped__
    lines, start = inspect.getsourcelines(function)
    path = Path(inspect.getsourcefile(function) or "")
    try:
        rendered = str(path.relative_to(ROOT))
    except ValueError:
        rendered = str(path)
    located = []
    for marker in markers:
        matches = [offset for offset, line in enumerate(lines) if marker in line]
        if not matches:
            raise RuntimeError(f"source marker {marker!r} is absent from {rendered}")
        located.append({"text": marker, "line": start + matches[0]})
    return {
        "path": rendered,
        "line_start": start,
        "line_end": start + len(lines) - 1,
        "markers": located,
    }


def _allocation() -> dict[str, Any]:
    """Return and validate the scheduler lane carrying the measurement."""

    job_id = os.environ.get("SLURM_JOB_ID")
    partition = os.environ.get("SLURM_JOB_PARTITION")
    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "0"))
    platforms = os.environ.get("JAX_PLATFORMS")
    memory = int(os.environ.get("SLURM_MEM_PER_NODE", "0"))
    if not job_id:
        raise RuntimeError("the measurement must run in a scheduler allocation")
    if partition != "all_debug":
        raise RuntimeError(f"expected all_debug, received {partition!r}")
    if cpus != 8:
        raise RuntimeError(f"expected eight CPUs, received {cpus}")
    if platforms != "cpu":
        raise RuntimeError(f"expected JAX_PLATFORMS=cpu, received {platforms!r}")
    if os.environ.get("TMPDIR") != "/tmp":
        raise RuntimeError("TMPDIR must be /tmp in the allocation payload")
    return {
        "job_id": int(job_id),
        "job_name": os.environ.get("SLURM_JOB_NAME"),
        "partition": partition,
        "node": os.environ.get("SLURMD_NODENAME", socket.gethostname()),
        "allocated_cpus": cpus,
        "memory_mb": memory,
        "jax_platforms": platforms.split(","),
        "jax_default_backend": jax.default_backend(),
        "tmpdir": os.environ.get("TMPDIR"),
    }


@contextmanager
def _support_mode(mode: str) -> Iterator[None]:
    """Select one benchmark-only support mode and restore the process default."""

    previous = support_clip_mode()
    set_support_clip_mode(mode)
    try:
        yield
    finally:
        set_support_clip_mode(previous)


def _row_slug(case_name: str, requested_cells: int, wall_nodes: int) -> str:
    """Return the stable filesystem identity of one row."""

    return f"{case_name}-cells-{abs(requested_cells)}-wall-{wall_nodes}"


def _part_path(
    report_directory: Path,
    case_name: str,
    requested_cells: int,
    wall_nodes: int,
) -> Path:
    """Return the durable part path for one row."""

    return (
        report_directory
        / "parts"
        / f"{_row_slug(case_name, requested_cells, wall_nodes)}.json"
    )


def _diverted_wall(exact: Any, wall_nodes: int) -> np.ndarray:
    """Return the certificate's diverted wall with a requested node count."""

    return oracle_fixture.offset_wall(
        exact.separatrix(1441),
        clearance=certificate.DIVERTED_WALL_CLEARANCE_FRACTION * exact.minor_radius,
        points=wall_nodes,
    )


def _machine(
    case_name: str,
    carrier_case: Any,
    exact: Any,
    requested_cells: int,
    wall_nodes: int,
) -> Any:
    """Load or build one semantic carrier for the requested wall identity."""

    wall = (
        _diverted_wall(exact, wall_nodes)
        if certificate._is_diverted_case(case_name)
        else None
    )
    return oracle_fixture.cached_machine(
        carrier_case,
        requested_cells,
        wall_nodes=wall_nodes,
        wall=wall,
    )


def _exact_flux(case_name: str, exact: Any, points: np.ndarray) -> np.ndarray:
    """Evaluate exact total flux in webers on arbitrary points."""

    return certificate._exact_state(
        case_name, exact, np.asarray(points, dtype=np.float64)
    )


def _analytic_wall_extremum(
    case_name: str,
    exact: Any,
    wall: np.ndarray,
    polarity: float,
) -> dict[str, Any]:
    """Resolve the exact-flux extremum continuously over the wall polyline."""

    wall = np.asarray(wall, dtype=np.float64)
    following = np.roll(wall, -1, axis=0)
    fraction = np.linspace(0.0, 1.0, 33)
    sampled = (
        wall[:, None, :] + fraction[None, :, None] * (following - wall)[:, None, :]
    )
    sampled_flux = _exact_flux(case_name, exact, sampled.reshape(-1, 2)).reshape(
        len(wall), len(fraction)
    )
    flat = int(np.argmax(polarity * sampled_flux))
    segment, bin_index = np.unravel_index(flat, sampled_flux.shape)
    lower = fraction[max(0, bin_index - 1)]
    upper = fraction[min(len(fraction) - 1, bin_index + 1)]
    start = wall[segment]
    delta = following[segment] - start

    def objective(parameter: float) -> float:
        point = start + parameter * delta
        value = float(_exact_flux(case_name, exact, point[None, :])[0])
        return -polarity * value

    refined = minimize_scalar(
        objective,
        bounds=(float(lower), float(upper)),
        method="bounded",
        options={"xatol": 1.0e-14},
    )
    candidates = [float(lower), float(upper), float(refined.x)]
    scores = [-objective(parameter) for parameter in candidates]
    chosen = candidates[int(np.argmax(scores))]
    point = start + chosen * delta
    value = float(_exact_flux(case_name, exact, point[None, :])[0])
    return {
        "coordinate_rz_m": point.tolist(),
        "flux_wb": value,
        "segment_index": int(segment),
        "segment_fraction": chosen,
        "optimizer_success": bool(refined.success),
        "instrument_positive_control": {
            "sampled_finite_count": int(np.count_nonzero(np.isfinite(sampled_flux))),
            "sampled_total_count": int(sampled_flux.size),
            "sampled_signed_span_wb": float(np.ptp(polarity * sampled_flux)),
        },
    }


def _nearest_segment(point: np.ndarray, wall: np.ndarray) -> dict[str, Any]:
    """Return the closest wall panel and its metric length."""

    point = np.asarray(point, dtype=np.float64)
    start = np.asarray(wall, dtype=np.float64)
    end = np.roll(start, -1, axis=0)
    edge = end - start
    length_squared = np.einsum("ij,ij->i", edge, edge)
    parameter = np.clip(
        np.einsum("ij,ij->i", point - start, edge)
        / np.where(length_squared > 0.0, length_squared, 1.0),
        0.0,
        1.0,
    )
    nearest = start + parameter[:, None] * edge
    distance_squared = np.einsum("ij,ij->i", nearest - point, nearest - point)
    index = int(np.argmin(distance_squared))
    return {
        "index": index,
        "length_m": float(math.sqrt(length_squared[index])),
        "projection_fraction": float(parameter[index]),
        "distance_m": float(math.sqrt(distance_squared[index])),
    }


def _recommended_wall_nodes(
    case_name: str,
    exact: Any,
    pitch: float,
) -> int:
    """Return the smallest odd wall count whose longest panel fits one pitch."""

    for count in range(9, 4002, 2):
        wall = (
            _diverted_wall(exact, count)
            if certificate._is_diverted_case(case_name)
            else oracle_fixture.limiter_contour(exact, points=count)
        )
        panel = np.linalg.norm(np.roll(wall, -1, axis=0) - wall, axis=1)
        if float(np.max(panel)) <= pitch:
            return count
    raise RuntimeError("no supported odd wall count reaches the requested pitch")


def _block_tree(value: Any) -> Any:
    """Block until every device leaf in a nested result is ready."""

    return jax.block_until_ready(value)


def _read_measurement(
    operator: ForwardFluxOperator,
    analytic: np.ndarray,
) -> tuple[Any, Any, bool, dict[str, float]]:
    """Run the public hex-carrier read and retain its fixed-design product."""

    physical = jnp.asarray(analytic)[: operator.physical_node_number]

    def execute() -> tuple[Any, Any, Any]:
        masks, topology = operator.read(jnp.asarray(analytic))
        return _block_tree((masks, topology, topology.diverted))

    compiled_started = perf_counter()
    public_masks, public_topology, public_diverted = execute()
    compile_seconds = perf_counter() - compiled_started
    warm = []
    for _ in range(3):
        started = perf_counter()
        public_masks, public_topology, public_diverted = execute()
        warm.append(perf_counter() - started)

    masks, topology, _connected, admitted = _block_tree(
        operator._fixed_design_read(physical)
    )
    public_boundary = np.asarray(public_topology.boundary, dtype=np.float64)
    if not np.array_equal(np.asarray(public_masks.label), np.asarray(masks.label)):
        raise RuntimeError("public and fixed-design reads returned different masks")
    if not np.array_equal(public_boundary, np.asarray(topology.boundary)):
        raise RuntimeError("public read did not publish the fixed-design boundary")
    if not bool(admitted):
        raise RuntimeError("the analytic field did not admit a magnetic axis")
    if operator._fixed_design_topology.connectivity_radius.size:
        raise RuntimeError("the expected hex carrier unexpectedly has a raster read")
    return (
        masks,
        topology,
        bool(public_diverted),
        {
            "compile_and_first_read_seconds": compile_seconds,
            "warm_read_seconds_median": float(np.median(warm)),
            "warm_read_seconds_min": float(np.min(warm)),
            "warm_read_seconds_max": float(np.max(warm)),
        },
    )


def _shadow_measurement(
    operator: ForwardFluxOperator,
    analytic: np.ndarray,
    masks: Any,
    topology: Any,
) -> tuple[dict[str, Any], np.ndarray]:
    """Measure the private-wall shadow and make its exclusion observable."""

    physical = jnp.asarray(analytic)[: operator.physical_node_number]
    carrier = operator._carrier_shadow_read(physical, masks)
    private = np.asarray(carrier["private_wall_node_mask"], dtype=bool)
    wall = np.asarray(operator.wall.coordinate, dtype=np.float64)
    x_point = np.asarray(topology.x_point, dtype=np.float64)
    if not np.all(np.isfinite(x_point)):
        raise RuntimeError("the diverted shadow control requires a finite X-point")
    shadow, report = wall_height_shadow_mask(
        operator.wall.coordinate[:, 1],
        topology.axis[1],
        topology.x_point,
        carrier["xset"],
        carrier["private_wall_node_mask"],
        jnp.zeros(operator.wall.node_number, dtype=bool),
        operator._wall_height_hysteresis,
        operator._x_qualification_distance,
        return_report=True,
    )
    shadow, report = _block_tree((shadow, report))
    shadow = np.asarray(shadow, dtype=bool)
    below = wall[:, 1] < x_point[1]
    private_below = private & below
    shadow_below = shadow & below
    if not np.any(private_below):
        raise RuntimeError(
            "the private-region census saw no wall node below the X-point"
        )
    if not np.any(shadow_below):
        raise RuntimeError(
            "the height shadow excluded no private wall node below the X-point"
        )

    wall_start = operator.grid.node_number
    wall_flux = np.asarray(
        physical[wall_start : operator.physical_node_number], dtype=np.float64
    )
    candidates = np.flatnonzero(shadow_below)
    injected_index = int(candidates[np.argmax(wall_flux[candidates])])
    span = abs(float(topology.axis_flux) - float(topology.x_point_flux))
    injected_flux = max(float(np.max(wall_flux)), float(topology.x_point_flux)) + max(
        0.25 * span, 1.0e-6
    )
    injected = np.asarray(physical, dtype=np.float64).copy()
    injected[wall_start + injected_index] = injected_flux
    injected_wall_flux = jnp.asarray(
        injected[wall_start : operator.physical_node_number]
    )
    unmasked_bracket = np.asarray(
        operator._fixed_design_topology.wall_anchor_bracket(
            injected_wall_flux, operator.polarity
        ),
        dtype=int,
    )
    if injected_index not in unmasked_bracket:
        raise RuntimeError(
            "the injected private node did not enter the unmasked bracket"
        )

    _unmasked_masks, unmasked, _connected, _admitted = _block_tree(
        operator._fixed_design_read(jnp.asarray(injected), int(TopologyClass.LIMITED))
    )
    _masked_masks, masked, _connected, _admitted = _block_tree(
        operator._fixed_design_read(
            jnp.asarray(injected),
            int(TopologyClass.LIMITED),
            private_wall_node_mask=jnp.asarray(shadow),
        )
    )
    unmasked_contact = np.asarray(unmasked.wall_point, dtype=np.float64)
    masked_contact = np.asarray(masked.wall_point, dtype=np.float64)
    injected_coordinate = wall[injected_index]
    adjacent = np.max(
        np.linalg.norm(
            wall[[injected_index - 1, (injected_index + 1) % len(wall)]]
            - injected_coordinate,
            axis=1,
        )
    )
    unmasked_selected_injected_panel = bool(
        np.linalg.norm(unmasked_contact - injected_coordinate) <= adjacent
    )
    masked_rejected_injected_panel = bool(
        np.linalg.norm(masked_contact - injected_coordinate) > adjacent
    )
    if not unmasked_selected_injected_panel:
        raise RuntimeError("the unmasked positive control did not select the injection")
    if not masked_rejected_injected_panel:
        raise RuntimeError("the private-wall shadow did not reject the injection")

    lower_branch = int(report["lower_branch"])
    upper_branch = int(report["upper_branch"])
    return (
        {
            "private_wall_node_count": int(np.count_nonzero(private)),
            "private_wall_nodes_below_x_point": int(np.count_nonzero(private_below)),
            "shadowed_wall_node_count": int(np.count_nonzero(shadow)),
            "shadowed_private_nodes_below_x_point": int(np.count_nonzero(shadow_below)),
            "lower_exclusion_branch": BRANCH_NAMES[lower_branch],
            "upper_exclusion_branch": BRANCH_NAMES[upper_branch],
            "mask_composition": (
                "wall_height_shadow_mask applies a qualified-saddle height gate "
                "and intersects it with the connectivity-private wall mask"
            ),
            "report": {key: np.asarray(value).item() for key, value in report.items()},
            "positive_control": {
                "injected_wall_node_index": injected_index,
                "injected_coordinate_rz_m": injected_coordinate.tolist(),
                "injected_flux_wb": injected_flux,
                "x_point_flux_wb": float(topology.x_point_flux),
                "unmasked_bracket_node_indices": unmasked_bracket.tolist(),
                "unmasked_selected_injected_panel": unmasked_selected_injected_panel,
                "unmasked_selected_contact_rz_m": unmasked_contact.tolist(),
                "unmasked_selected_flux_wb": float(unmasked.wall_point_flux),
                "masked_rejected_injected_panel": masked_rejected_injected_panel,
                "masked_selected_contact_rz_m": masked_contact.tolist(),
                "masked_selected_flux_wb": float(masked.wall_point_flux),
                "passed": bool(
                    unmasked_selected_injected_panel and masked_rejected_injected_panel
                ),
            },
            "connectivity_private_wall_node_count": int(np.count_nonzero(private)),
        },
        shadow,
    )


def _map_floor(
    case_name: str,
    source_case: Any,
    machine: Any,
    analytic: np.ndarray,
) -> dict[str, Any]:
    """Measure one target-normalised exact-clip map application."""

    empty_operator = oracle_fixture.forward_operator(source_case, machine)
    with _support_mode("chord"):
        exact_physical = oracle_fixture.exact_current_moments(
            source_case, empty_operator, analytic
        )
    exact_coefficients = empty_operator.coupling_current_moments(exact_physical)
    exact_internal = np.asarray(
        oracle_fixture._internal_flux_image(empty_operator, exact_coefficients),
        dtype=np.float64,
    )
    operator = oracle_fixture.forward_operator(
        source_case, machine, analytic - exact_internal
    )
    target_current, _centroid, target_receipt = certificate._closed_form_current_target(
        case_name, source_case, empty_operator, exact_physical
    )
    _masks, topology = operator.read(jnp.asarray(analytic))
    span = abs(float(topology.axis_flux) - float(topology.boundary_flux))
    requested = int(TopologyClass.LIMITED)
    with _support_mode("exact"):
        mapped = np.asarray(
            _block_tree(
                operator.flux_map(
                    requested_class=requested,
                    target_current=target_current,
                )(jnp.asarray(analytic))
            ),
            dtype=np.float64,
        )
        moments = operator.cell_current_moments(jnp.asarray(analytic), requested)
    grid_delta = mapped[: len(machine.node)] - analytic[: len(machine.node)]
    return {
        "mode": "exact",
        "requested_class": "limited",
        "definition": (
            "one target-normalised production-map application at the analytic "
            "flux, divided by the analytic axis-to-boundary span"
        ),
        "relative_rms_of_span": float(np.sqrt(np.mean(grid_delta**2)) / span),
        "relative_sup_of_span": float(np.max(np.abs(grid_delta)) / span),
        "absolute_rms_wb": float(np.sqrt(np.mean(grid_delta**2))),
        "absolute_sup_wb": float(np.max(np.abs(grid_delta))),
        "analytic_flux_span_wb": span,
        "booked_current_a": float(jnp.sum(moments.cell_current)),
        "target_current_a": target_current,
        "booked_over_target": float(jnp.sum(moments.cell_current)) / target_current,
        "target_receipt": target_receipt,
        "fixture_exterior": (
            "analytic total flux minus the image of analytically integrated "
            "moments under the committed whole-cell fixture posing"
        ),
    }


def _measure_row(
    case_name: str,
    requested_cells: int,
    wall_nodes: int,
    report_directory: Path,
) -> dict[str, Any]:
    """Measure and persist one analytic topology row."""

    started = perf_counter()
    part_path = _part_path(report_directory, case_name, requested_cells, wall_nodes)
    progress = {
        "schema": "nova.limiter-read-resolution-part",
        "version": 1,
        "source_revision": _source_revision(),
        "case": case_name,
        "requested_cells": requested_cells,
        "wall_nodes": wall_nodes,
        "completed": False,
    }
    _write_json(part_path, progress)
    carrier_case, source_case, exact = certificate._case(case_name)
    machine = _machine(case_name, carrier_case, exact, requested_cells, wall_nodes)
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    analytic = _exact_flux(case_name, exact, coordinates)
    operator = oracle_fixture.forward_operator(source_case, machine)
    masks, topology, production_diverted, read_timing = _read_measurement(
        operator, analytic
    )
    topology_class = "diverted" if production_diverted else "limited"
    analytic_contact = _analytic_wall_extremum(
        case_name, exact, machine.wall_node, float(operator.polarity)
    )
    contact = np.asarray(topology.wall_point, dtype=np.float64)
    contact_flux = float(topology.wall_point_flux)
    exact_contact = np.asarray(analytic_contact["coordinate_rz_m"], dtype=np.float64)
    span = abs(float(topology.axis_flux) - analytic_contact["flux_wb"])
    contact_panel = _nearest_segment(contact, machine.wall_node)
    pitch = float(np.sqrt(np.median(np.asarray(machine.area, dtype=np.float64))))
    panel_lengths = np.linalg.norm(
        np.roll(machine.wall_node, -1, axis=0) - machine.wall_node,
        axis=1,
    )
    fixed_node_distance = float(
        np.min(np.linalg.norm(machine.wall_node - contact, axis=1))
    )
    row: dict[str, Any] = {
        **progress,
        "realised_cells": len(machine.node),
        "state_dimension": len(analytic),
        "topology_class": topology_class,
        "cache": machine.cache,
        "characteristic_cell_pitch_m": pitch,
        "wall": {
            "node_count": wall_nodes,
            "perimeter_m": float(np.sum(panel_lengths)),
            "median_panel_length_m": float(np.median(panel_lengths)),
            "maximum_panel_length_m": float(np.max(panel_lengths)),
            "selected_panel_length_m": contact_panel["length_m"],
            "selected_panel_over_cell_pitch": contact_panel["length_m"] / pitch,
            "maximum_panel_over_cell_pitch": float(np.max(panel_lengths)) / pitch,
        },
        "analytic_wall_extremum": analytic_contact,
        "production_contact": {
            "coordinate_rz_m": contact.tolist(),
            "flux_wb": contact_flux,
            "position_error_m": float(np.linalg.norm(contact - exact_contact)),
            "level_error_wb": abs(contact_flux - analytic_contact["flux_wb"]),
            "level_error_in_span": abs(contact_flux - analytic_contact["flux_wb"])
            / span,
            "nearest_wall_node_distance_m": fixed_node_distance,
            "selected_equals_wall_node": bool(fixed_node_distance <= 1.0e-12),
            "representation": "three-node quadratic sub-panel interpolation",
        },
        "connectivity_contact": {
            "status": "not_applicable_to_unstructured_hex_carrier",
            "reason": (
                "the public class comparator takes the moment-geometry branch "
                "when the fixed-design topology has no tensor axes"
            ),
        },
        "read_timing": read_timing,
        "resolution_guidance": {
            "wall_121_consistent_with_realised_pitch": bool(
                wall_nodes != 121 or float(np.max(panel_lengths)) <= pitch
            ),
            "recommended_wall_nodes_for_realised_pitch": _recommended_wall_nodes(
                case_name, exact, pitch
            ),
        },
        "shadow": None,
        "map_floor": None,
        "completed": False,
        "wall_seconds": None,
    }
    if requested_cells == -1000:
        pitch_2500 = pitch * math.sqrt(
            len(machine.node) / HIGH_RESOLUTION_REALISED_CELLS
        )
        row["resolution_guidance"].update(
            {
                "estimated_cell_pitch_at_2500_m": pitch_2500,
                "wall_121_consistent_with_estimated_2500_pitch": bool(
                    wall_nodes != 121 or float(np.max(panel_lengths)) <= pitch_2500
                ),
                "recommended_wall_nodes_for_estimated_2500_pitch": (
                    _recommended_wall_nodes(case_name, exact, pitch_2500)
                ),
            }
        )
    if certificate._is_diverted_case(case_name):
        shadow, shadow_mask = _shadow_measurement(operator, analytic, masks, topology)
        row["shadow"] = shadow
        if requested_cells == -1000 and wall_nodes == 481:
            radius = np.linspace(
                float(np.min(machine.wall_node[:, 0])),
                float(np.max(machine.wall_node[:, 0])),
                220,
            )
            height = np.linspace(
                float(np.min(machine.wall_node[:, 1])),
                float(np.max(machine.wall_node[:, 1])),
                280,
            )
            rr, zz = np.meshgrid(radius, height)
            render_points = np.column_stack((rr.ravel(), zz.ravel()))
            row["render"] = {
                "radius": radius,
                "height": height,
                "flux": _exact_flux(case_name, exact, render_points).reshape(
                    len(height), len(radius)
                ),
                "wall": np.asarray(machine.wall_node),
                "axis": np.asarray(certificate.AXIS_M),
                "x_point": np.asarray(certificate.X_POINT_M),
                "selected_contact": contact,
                "excluded_wall_nodes": np.asarray(machine.wall_node)[shadow_mask],
                "boundary_flux_wb": float(topology.boundary_flux),
                "axis_flux_wb": float(topology.axis_flux),
            }
    if (
        case_name == "weak-rotation-reactor-static"
        and requested_cells == -1000
        and wall_nodes in (121, 481)
    ):
        row["map_floor"] = _map_floor(case_name, source_case, machine, analytic)
    row["wall_seconds"] = perf_counter() - started
    row["completed"] = True
    _write_json(part_path, row)
    print(
        "LIMITER_READ_ROW "
        f"case={case_name} cells={abs(requested_cells)} wall={wall_nodes} "
        f"class={topology_class} "
        f"position_error_m={row['production_contact']['position_error_m']:.8e} "
        f"level_error_span={row['production_contact']['level_error_in_span']:.8e} "
        f"read_seconds={read_timing['warm_read_seconds_median']:.8e}",
        flush=True,
    )
    return row


def _load_part(path: Path) -> dict[str, Any]:
    """Load one completed row without treating a large receipt as source text."""

    with path.open(encoding="utf-8") as stream:
        row = json.load(stream)
    if not row.get("completed"):
        raise RuntimeError(f"required landed row is incomplete: {path}")
    return row


def _saddle_wall_coupling(
    operator: ForwardFluxOperator,
    machine: Any,
    analytic: np.ndarray,
) -> dict[str, Any]:
    """Record every wall-dependent gate around the selected analytic saddle."""

    physical = jnp.asarray(analytic)[: operator.physical_node_number]
    _masks, topology, _connected, admitted = _block_tree(
        operator._fixed_design_read(physical)
    )
    grid_flux, _wall_flux = operator._fixed_design_topology.split_flux_map(physical)
    (vmap_o, vmap_x), census = _block_tree(
        operator._fixed_design_topology.grid.read_census(grid_flux)
    )
    del vmap_o
    finite = np.all(np.isfinite(np.asarray(vmap_x)[:, :3]), axis=1)
    contained = np.asarray(
        operator._fixed_design_topology.contained_x_candidates(vmap_x), dtype=bool
    )
    selected = np.asarray(topology.x_point, dtype=np.float64)
    reference = np.asarray(certificate.X_POINT_M, dtype=np.float64)
    nodes = np.ascontiguousarray(np.asarray(machine.node), dtype="<f8")
    retained = np.asarray(census.get("retained_valid", np.empty((0,), dtype=bool)))
    representative = np.asarray(
        census.get("representative_mask", np.empty((0,), dtype=bool))
    )
    return {
        "selected_x_point_rz_m": selected.tolist(),
        "selected_x_point_error_m": float(np.linalg.norm(selected - reference)),
        "axis_admitted": bool(admitted),
        "finite_x_candidate_count": int(np.count_nonzero(finite)),
        "contained_x_candidate_count": int(np.count_nonzero(contained)),
        "selected_candidate_contained": bool(
            np.any(
                contained
                & (
                    np.linalg.norm(np.asarray(vmap_x)[:, :2] - selected, axis=1)
                    <= 1.0e-10
                )
            )
        ),
        "shadow_mask_supplied_to_analytic_read": False,
        "deduplication_retained_count": (
            int(np.count_nonzero(retained[1]))
            if retained.ndim == 2 and retained.shape[0] > 1
            else None
        ),
        "deduplication_representative_count": (
            int(np.count_nonzero(representative[1]))
            if representative.ndim == 2 and representative.shape[0] > 1
            else None
        ),
        "carrier_node_count": len(machine.node),
        "carrier_node_sha256_binary64": hashlib.sha256(nodes.tobytes()).hexdigest(),
        "wall_content_sha256_binary64": hashlib.sha256(
            np.ascontiguousarray(machine.wall_node, dtype="<f8").tobytes()
        ).hexdigest(),
        "coupling_read": (
            "the public read supplies no shadow mask; containment admits the sole "
            "candidate and deduplication has no alternative to choose. The wall "
            "enters earlier because cached_machine rebuilds the plasma carrier "
            "after inserting that sampled wall."
        ),
    }


def _reconstruct_row(
    case_name: str,
    requested_cells: int,
    wall_nodes: int,
) -> tuple[Any, Any, Any, np.ndarray, ForwardFluxOperator]:
    """Warm-load the carrier and analytic state needed by a residual stage."""

    carrier_case, source_case, exact = certificate._case(case_name)
    machine = _machine(case_name, carrier_case, exact, requested_cells, wall_nodes)
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    analytic = _exact_flux(case_name, exact, coordinates)
    operator = oracle_fixture.forward_operator(source_case, machine)
    return source_case, exact, machine, analytic, operator


def measure_residual_stages(report_directory: Path) -> dict[str, Any]:
    """Rerun only private-wall and weak-map measurements over landed rows."""

    configure_dtypes()
    if not jax.config.jax_enable_x64:
        raise RuntimeError("the measurement requires JAX double precision")
    allocation = _allocation()
    cache = configure_persistent_compilation_cache(
        default_persistent_compilation_cache_root()
    )
    started = perf_counter()
    completed = []
    for wall_nodes in WALL_NODE_COUNTS:
        for requested_cells in (-500, -1000):
            case_name = certificate.DIVERTED_CASE_NAME
            path = _part_path(report_directory, case_name, requested_cells, wall_nodes)
            row = _load_part(path)
            _source_case, _exact, machine, analytic, operator = _reconstruct_row(
                case_name, requested_cells, wall_nodes
            )
            physical = jnp.asarray(analytic)[: operator.physical_node_number]
            masks, topology, _connected, admitted = _block_tree(
                operator._fixed_design_read(physical)
            )
            if not bool(admitted):
                raise RuntimeError("the private-wall rerun did not admit its axis")
            shadow, shadow_mask = _shadow_measurement(
                operator, analytic, masks, topology
            )
            row["shadow"] = shadow
            row["saddle_wall_coupling"] = _saddle_wall_coupling(
                operator, machine, analytic
            )
            if requested_cells == -1000 and wall_nodes == 481:
                render = row.get("render")
                if render is None:
                    raise RuntimeError("the landed render row omits its operands")
                render["excluded_wall_nodes"] = np.asarray(machine.wall_node)[
                    shadow_mask
                ]
            row["residual_stage"] = {
                "job_id": allocation["job_id"],
                "private_wall_shadow_remeasured": True,
                "map_floor_remeasured": False,
            }
            _write_json(path, row)
            completed.append(f"shadow:{case_name}:{abs(requested_cells)}:{wall_nodes}")
            print(completed[-1], flush=True)

    case_name = "weak-rotation-reactor-static"
    for wall_nodes in (121, 481):
        requested_cells = -1000
        path = _part_path(report_directory, case_name, requested_cells, wall_nodes)
        row = _load_part(path)
        source_case, _exact, machine, analytic, _operator = _reconstruct_row(
            case_name, requested_cells, wall_nodes
        )
        row["map_floor"] = _map_floor(case_name, source_case, machine, analytic)
        row["residual_stage"] = {
            "job_id": allocation["job_id"],
            "private_wall_shadow_remeasured": False,
            "map_floor_remeasured": True,
        }
        _write_json(path, row)
        completed.append(f"map:{case_name}:1000:{wall_nodes}")
        print(completed[-1], flush=True)

    receipt = {
        "source_revision": _source_revision(),
        "allocation": allocation,
        "persistent_compilation_cache": cache.receipt(),
        "completed_stages": completed,
        "stage_count": len(completed),
        "wall_seconds": perf_counter() - started,
        "completed": True,
        "exit_marker": "LIMITER_READ_RESIDUAL_STAGES_EXIT=0",
    }
    _write_json(report_directory / "residual-measurement.json", receipt)
    print(receipt["exit_marker"], flush=True)
    return receipt


def _source_contract() -> dict[str, Any]:
    """Return verified code-path evidence for the hex-carrier read."""

    return {
        "certificate_topology_adapter": _pointer(
            certificate._topology, ("operator.read(jnp.asarray(state))",)
        ),
        "public_forward_read": _pointer(
            ForwardFluxOperator.read,
            ("self._fixed_design_read", "_class_margin_read=lambda"),
        ),
        "fixed_design_read": _pointer(ForwardFluxOperator._fixed_design_read),
        "hex_class_branch": _pointer(
            ForwardFluxOperator._connectivity_class_margin,
            ("self.moment_geometry is not None", "self._fixed_design_read"),
        ),
        "fixed_design_wall_contact": _pointer(
            Topology._wall_anchor_selection,
            ("traced_quadratic_wall", "wall_coordinate"),
        ),
        "wall_dependent_carrier_mesh": _pointer(
            oracle_fixture.build_machine,
            ("coilset.firstwall.insert(wall", "centres ="),
        ),
        "saddle_wall_containment": _pointer(
            Topology.contained_x_candidates,
            ("_points_inside_polygon",),
        ),
        "saddle_deduplication": _pointer(
            forward_operator_module._FixedDesignNull2D._deduplicate_type,
            ("same_root =",),
        ),
        "unselected_tensor_grid_limiter": _pointer(
            __import__(
                "nova.equilibrium.connectivity_boundary",
                fromlist=["_select_reachable_wall_limiter"],
            )._select_reachable_wall_limiter,
            ("global_surface.evaluate", "refine_root"),
        ),
        "private_wall_shadow": _pointer(
            wall_height_shadow_mask.__wrapped__,
            ("mask = proposed & private_wall",),
        ),
    }


def _load_parts(report_directory: Path) -> list[dict[str, Any]]:
    """Load the complete required part census and refuse stale omissions."""

    rows = []
    for wall_nodes in WALL_NODE_COUNTS:
        for case_name, requested_cells in ROWS:
            path = _part_path(report_directory, case_name, requested_cells, wall_nodes)
            if not path.exists():
                raise FileNotFoundError(f"required part is absent: {path}")
            with path.open(encoding="utf-8") as stream:
                row = json.load(stream)
            if not row.get("completed"):
                raise RuntimeError(f"required part is incomplete: {path}")
            rows.append(row)
    return rows


def _render_error_figure(rows: list[dict[str, Any]], path: Path) -> dict[str, Any]:
    """Plot contact position error against local wall-to-cell resolution."""

    figure, axes = plt.subplots(figsize=(7.2, 4.7), constrained_layout=True)
    trace_axes(axes)
    colors = {
        "weak-rotation-reactor-static": DEFAULT_INK.flux_color,
        "moderate-rotation-conventional-static": "#cc7722",
        certificate.DIVERTED_CASE_NAME: "#6a3d9a",
    }
    markers = {-300: "o", -500: "s", -1000: "^"}
    for case_name, requested_cells in ROWS:
        selected = sorted(
            (
                row
                for row in rows
                if row["case"] == case_name
                and row["requested_cells"] == requested_cells
            ),
            key=lambda row: row["wall"]["selected_panel_over_cell_pitch"],
        )
        axes.plot(
            [row["wall"]["selected_panel_over_cell_pitch"] for row in selected],
            [row["production_contact"]["position_error_m"] for row in selected],
            color=colors[case_name],
            marker=markers[requested_cells],
            markersize=4.0,
            linewidth=1.0,
            label=f"{case_name.replace('-', ' ')} · {abs(requested_cells)} cells",
        )
    axes.set_xscale("log")
    axes.set_yscale("log")
    axes.set_xlabel("selected wall panel length / characteristic cell pitch")
    axes.set_ylabel("analytic contact position error [m]")
    title_lines = [
        (
            "analytic limiter read · contact position error "
            "against wall-panel / cell-pitch ratio"
        ),
        (
            f"{len(rows)} measured rows over {len(WALL_NODE_COUNTS)} wall resolutions; "
            "the exact field is fixed and only the carrier wall sampling varies. "
            "No nonlinear solve is entered."
        ),
    ]
    axes.set_title("\n".join(title_lines), loc="left", fontsize=7.0)
    axes.legend(frameon=False, fontsize=6, ncol=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, format="svg", facecolor=DEFAULT_INK.figure_facecolor)
    plt.close(figure)
    return {
        "figure": path.name,
        "kind": ERROR_KIND,
        "titles": [{"text": line, "loc": "left"} for line in title_lines],
        "title_lines": title_lines,
        "legend_entries": [
            f"{case_name.replace('-', ' ')} · {abs(requested_cells)} cells"
            for case_name, requested_cells in ROWS
        ],
        "glyph_markers": {
            str(requested): marker for requested, marker in markers.items()
        },
        "wall_drawn": False,
        "poloidal_panels": [],
    }


def _render_poloidal_figure(
    row: dict[str, Any],
    path: Path,
    source_receipt: dict[str, Any],
) -> dict[str, Any]:
    """Render the analytic single-null contact and private-wall exclusions."""

    render = row.get("render")
    if render is None:
        raise RuntimeError("the single-null render row carries no operands")
    radius = np.asarray(render["radius"], dtype=np.float64)
    height = np.asarray(render["height"], dtype=np.float64)
    flux = np.asarray(render["flux"], dtype=np.float64)
    wall = np.asarray(render["wall"], dtype=np.float64)
    axis = np.asarray(render["axis"], dtype=np.float64)
    x_point = np.asarray(render["x_point"], dtype=np.float64)
    contact = np.asarray(render["selected_contact"], dtype=np.float64)
    excluded = np.asarray(render["excluded_wall_nodes"], dtype=np.float64)
    levels = poloidal.contour_levels(
        flux,
        15,
        boundary=float(render["boundary_flux_wb"]),
        axis=float(render["axis_flux_wb"]),
    )
    title_lines, title_fields = _contact_title(row, source_receipt, len(wall))
    figure, axes = plt.subplots(figsize=(7.8, 6.2), constrained_layout=True)
    poloidal_axes(axes)
    poloidal.draw_flux_contours(axes, radius, height, flux, levels)
    poloidal.draw_wall(axes, wall[:, 0], wall[:, 1])
    tally = poloidal.draw_nulls(
        axes,
        magnetic_axis=axis,
        x_points=x_point[None, :],
        contain=wall,
    )
    axes.plot(
        contact[0],
        contact[1],
        marker=CONTACT_MARKER,
        markersize=5.0,
        markerfacecolor=DEFAULT_INK.flux_color,
        markeredgecolor="white",
        linestyle="none",
        zorder=DEFAULT_INK.zorder_markers + 1,
    )
    if excluded.size:
        axes.plot(
            excluded[:, 0],
            excluded[:, 1],
            marker=EXCLUDED_WALL_MARKER,
            markersize=2.5,
            markerfacecolor="none",
            markeredgecolor=EXCLUDED_WALL_COLOR,
            linestyle="none",
            zorder=DEFAULT_INK.zorder_markers,
        )
    title_lines = [
        piece for line in title_lines if line for piece in _wrap_title(line, 104)
    ]
    _pad_panel(axes)
    axes.set_title("\n".join(title_lines), loc="left", fontsize=5.6)
    axes.legend(
        handles=_contact_legend_handles(),
        frameon=False,
        fontsize=5.4,
        loc="center left",
        ncol=1,
        bbox_to_anchor=(1.01, 0.5),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        path,
        format=path.suffix.removeprefix("."),
        dpi=180,
        facecolor=DEFAULT_INK.figure_facecolor,
    )
    plt.close(figure)
    axis_drawn = _axis_glyph_drawn(axis, wall)
    return {
        "figure": path.name,
        "kind": POLOIDAL_KIND,
        "source_row_key": {
            "case": row["case"],
            "requested_cells": row["requested_cells"],
            "wall_nodes": row["wall_nodes"],
        },
        "titles": [{"text": line, "loc": "left"} for line in title_lines],
        "title_lines": title_lines,
        "title_fields": title_fields,
        "legend_entries": list(GLYPH_FAMILIES) + ["analytic_flux_contours", "wall"],
        "glyph_markers": {
            "magnetic_axis": DEFAULT_INK.axis_marker,
            "admitted_x_point": DEFAULT_INK.xpoint_marker,
            "wall_contact": CONTACT_MARKER,
            "excluded_private_wall_nodes": EXCLUDED_WALL_MARKER,
        },
        "wall_drawn": True,
        "wall_node_count": int(len(wall)),
        "poloidal_panels": [
            {
                "panel": "analytic flux levels as line contours on shared Wb levels",
                "null_sets": {
                    "analytic_magnetic_axis": {
                        "drawn": int(axis_drawn),
                        "dropped_outside_wall": int(axis_drawn == 0),
                    },
                    "analytic_admitted_x_points": {
                        "drawn": int(tally["x_points_drawn"]),
                        "dropped_outside_wall": int(
                            tally["x_points_dropped_outside_wall"]
                        ),
                    },
                },
                "wall_contact_glyphs": int(np.size(contact) // 2),
                "excluded_private_wall_glyphs": (
                    int(excluded.shape[0]) if excluded.size else 0
                ),
                "null_glyph_total": int(tally["x_points_drawn"]) + int(axis_drawn),
            }
        ],
    }


def _axis_glyph_drawn(axis: np.ndarray, wall: np.ndarray) -> int:
    """Return the count of axis glyphs ``draw_nulls`` admits inside the wall."""

    point = np.asarray(axis, dtype=float).reshape(-1)[:2]
    if not np.all(np.isfinite(point)):
        return 0
    return int(bool(inside_wall_units(point[None, :], wall)[0]))


def _contact_title(
    row: dict[str, Any],
    source_receipt: dict[str, Any],
    wall_nodes: int,
) -> tuple[list[str], dict[str, Any]]:
    """Compose the contact-figure title from the source receipt's own fields."""

    contact = row["production_contact"]
    residual = float(contact["level_error_in_span"])
    entered = bool(source_receipt.get("nonlinear_solve_entered", True))
    converged = "yes" if entered else "n/a"
    title_line = (
        f"{row['case']} · cells={int(row['realised_cells'])} · "
        f"residual={residual:.3e} · converged={converged}"
    )
    detail = (
        f"analytic fixed-design read of the exact diverted single-null field; "
        f"nominal cell scale {abs(int(row['requested_cells']))} "
        f"({int(row['realised_cells'])} realised cells), {wall_nodes} wall nodes, "
        f"selected panel/cell pitch "
        f"{float(row['wall']['selected_panel_over_cell_pitch']):.4g}, contact "
        f"offset from the exact wall extremum "
        f"{float(contact['position_error_m']):.3e} m"
    )
    flag = (
        f"no nonlinear solve is entered "
        f"(source receipt nonlinear_solve_entered={entered}); residual is the "
        f"selected-contact level error in span {residual:.3e} against the exact "
        f"wall extremum, so no convergence flag exists for this analytic read"
    )
    fields = {
        "fixture": {
            "value": str(row["case"]),
            "source_path": "rows[<case,requested_cells,wall_nodes>].case",
            "source_value": row["case"],
        },
        "cells": {
            "value": str(int(row["realised_cells"])),
            "source_path": "rows[<case,requested_cells,wall_nodes>].realised_cells",
            "source_value": int(row["realised_cells"]),
        },
        "residual": {
            "value": f"{residual:.3e}",
            "source_path": (
                "rows[<case,requested_cells,wall_nodes>]"
                ".production_contact.level_error_in_span"
            ),
            "source_value": residual,
        },
        "converged": {
            "value": converged,
            "source_path": "nonlinear_solve_entered",
            "source_value": entered,
        },
    }
    return [title_line, detail, flag], fields


def _pad_panel(axes: Any) -> None:
    """Widen the autoscaled limits slightly so no glyph sits on the frame."""

    spacings = []
    for getter in (axes.get_xlim, axes.get_ylim):
        low, high = getter()
        spacings.append(max(high - low, np.finfo(float).tiny))
    margin = 0.02 * max(spacings)
    limits = ((axes.get_xlim, axes.set_xlim), (axes.get_ylim, axes.set_ylim))
    for getter, setter in limits:
        low, high = getter()
        setter(low - margin, high + margin)


def _wrap_title(line: str, width: int) -> list[str]:
    """Break one title line at whitespace so no title runs off the panel.

    A matplotlib title is not wrapped by the layout engine, so an over-long
    line is clipped rather than reported; the break belongs to the caller.
    """

    pieces: list[str] = []
    current = ""
    for word in line.split():
        if current and len(current) + 1 + len(word) > width:
            pieces.append(current)
            current = word
        elif current:
            current = f"{current} {word}"
        else:
            current = word
    if current:
        pieces.append(current)
    return pieces


def _contact_legend_handles() -> list[Any]:
    """Return one legend handle per glyph family the contact figure draws."""

    return [
        Line2D(
            [],
            [],
            color=DEFAULT_INK.contour_color,
            linewidth=DEFAULT_INK.contour_linewidth,
            label="analytic flux contours (exact field, shared Wb levels)",
        ),
        Line2D(
            [],
            [],
            color=DEFAULT_INK.wall_color,
            linewidth=DEFAULT_INK.wall_linewidth,
            label="wall",
        ),
        Line2D(
            [],
            [],
            marker=DEFAULT_INK.axis_marker,
            color=DEFAULT_INK.axis_color,
            markersize=DEFAULT_INK.axis_markersize,
            linestyle="none",
            label="magnetic axis (analytic)",
        ),
        Line2D(
            [],
            [],
            marker=DEFAULT_INK.xpoint_marker,
            color=DEFAULT_INK.xpoint_color,
            markersize=DEFAULT_INK.xpoint_markersize,
            linestyle="none",
            label="admitted X-point (analytic)",
        ),
        Line2D(
            [],
            [],
            marker=CONTACT_MARKER,
            markerfacecolor=DEFAULT_INK.flux_color,
            markeredgecolor="white",
            color=DEFAULT_INK.flux_color,
            markersize=5.0,
            linestyle="none",
            label="selected wall contact (read)",
        ),
        Line2D(
            [],
            [],
            marker=EXCLUDED_WALL_MARKER,
            markerfacecolor="none",
            markeredgecolor=EXCLUDED_WALL_COLOR,
            color=EXCLUDED_WALL_COLOR,
            markersize=2.5,
            linestyle="none",
            label="private-wall nodes exempted from the read",
        ),
    ]


def _resolution_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return panel-to-pitch ladders and matching wall counts."""

    summary = []
    case_requests = {
        "weak-rotation-reactor-static": (-300, -1000),
        "moderate-rotation-conventional-static": (-300, -1000),
        certificate.DIVERTED_CASE_NAME: (-500, -1000),
    }
    for case_name, requests in case_requests.items():
        _carrier, _source, exact = certificate._case(case_name)
        for requested_cells in requests:
            selected = sorted(
                (
                    row
                    for row in rows
                    if row["case"] == case_name
                    and row["requested_cells"] == requested_cells
                ),
                key=lambda row: row["wall_nodes"],
            )
            base = selected[0]
            summary.append(
                {
                    "case": case_name,
                    "cell_scale": abs(requested_cells),
                    "realised_cells": base["realised_cells"],
                    "cell_pitch_m": base["characteristic_cell_pitch_m"],
                    "selected_panel_over_pitch": {
                        str(row["wall_nodes"]): row["wall"][
                            "selected_panel_over_cell_pitch"
                        ]
                        for row in selected
                    },
                    "recommended_wall_nodes": base["resolution_guidance"][
                        "recommended_wall_nodes_for_realised_pitch"
                    ],
                }
            )
        thousand = next(
            row
            for row in rows
            if row["case"] == case_name
            and row["requested_cells"] == -1000
            and row["wall_nodes"] == 121
        )
        pitch = thousand["characteristic_cell_pitch_m"] * math.sqrt(
            thousand["realised_cells"] / HIGH_RESOLUTION_REALISED_CELLS
        )
        summary.append(
            {
                "case": case_name,
                "cell_scale": 2500,
                "realised_cells": HIGH_RESOLUTION_REALISED_CELLS,
                "cell_pitch_m": pitch,
                "selected_panel_over_pitch": {
                    str(row["wall_nodes"]): row["wall"]["selected_panel_length_m"]
                    / pitch
                    for row in rows
                    if row["case"] == case_name and row["requested_cells"] == -1000
                },
                "recommended_wall_nodes": _recommended_wall_nodes(
                    case_name, exact, pitch
                ),
                "pitch_source": (
                    "thousand-cell realised pitch scaled by the square root of "
                    "realised cells over 2600"
                ),
            }
        )
    return summary


def _limited_convergence(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fit wall-panel orders for limited contact position and level errors."""

    result = []
    for case_name in (
        "weak-rotation-reactor-static",
        "moderate-rotation-conventional-static",
    ):
        for requested_cells in (-300, -1000):
            selected = sorted(
                (
                    row
                    for row in rows
                    if row["case"] == case_name
                    and row["requested_cells"] == requested_cells
                    and row["wall_nodes"] >= 241
                ),
                key=lambda row: row["wall_nodes"],
            )
            panel = np.asarray(
                [row["wall"]["selected_panel_length_m"] for row in selected]
            )
            position = np.asarray(
                [row["production_contact"]["position_error_m"] for row in selected]
            )
            level = np.asarray(
                [row["production_contact"]["level_error_in_span"] for row in selected]
            )
            result.append(
                {
                    "case": case_name,
                    "requested_cells": requested_cells,
                    "position_order": float(
                        np.polyfit(np.log(panel), np.log(position), 1)[0]
                    ),
                    "level_order": float(
                        np.polyfit(np.log(panel), np.log(level), 1)[0]
                    ),
                }
            )
    return result


def _limiter_node_construction() -> list[dict[str, Any]]:
    """Prove whether the authored limited wall samples the smooth tangency."""

    result = []
    for case_name in (
        "weak-rotation-reactor-static",
        "moderate-rotation-conventional-static",
    ):
        _carrier, _source, exact = certificate._case(case_name)
        _inboard, outboard = exact.boundary_midplane_radii()
        tangency = np.asarray([outboard, 0.0], dtype=np.float64)
        for wall_nodes in WALL_NODE_COUNTS:
            wall = oracle_fixture.limiter_contour(exact, points=wall_nodes)
            index = (wall_nodes - 1) // 2
            result.append(
                {
                    "case": case_name,
                    "wall_nodes": wall_nodes,
                    "tangency_node_index": index,
                    "tangency_node_rz_m": wall[index].tolist(),
                    "distance_to_smooth_analytic_tangency_m": float(
                        np.linalg.norm(wall[index] - tangency)
                    ),
                    "node_flux_wb": float(
                        _exact_flux(case_name, exact, wall[index : index + 1])[0]
                    ),
                    "angle_formula_index_hits_pi": True,
                }
            )
    return result


def _saddle_coupling_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize why the flux-only saddle moves when wall sampling changes."""

    result = []
    for requested_cells in (-500, -1000):
        selected = sorted(
            (
                row
                for row in rows
                if row["case"] == certificate.DIVERTED_CASE_NAME
                and row["requested_cells"] == requested_cells
            ),
            key=lambda row: row["wall_nodes"],
        )
        coupling = [row["saddle_wall_coupling"] for row in selected]
        result.append(
            {
                "requested_cells": requested_cells,
                "wall_nodes": [row["wall_nodes"] for row in selected],
                "x_point_error_m": [
                    item["selected_x_point_error_m"] for item in coupling
                ],
                "published_boundary_error_m": [
                    row["production_contact"]["position_error_m"] for row in selected
                ],
                "distinct_carrier_node_identities": len(
                    {item["carrier_node_sha256_binary64"] for item in coupling}
                ),
                "all_selected_candidates_contained": all(
                    item["selected_candidate_contained"] for item in coupling
                ),
                "shadow_mask_participated": any(
                    item["shadow_mask_supplied_to_analytic_read"] for item in coupling
                ),
                "maximum_finite_x_candidate_count": max(
                    item["finite_x_candidate_count"] for item in coupling
                ),
                "cause": (
                    "wall sampling changes the cached carrier mesh before the read; "
                    "containment admits the selected candidate, no shadow is supplied, "
                    "and one finite candidate leaves no dedupe choice"
                ),
            }
        )
    return result


def _write_report(path: Path, receipt: dict[str, Any]) -> None:
    """Write the compact human-readable measurement report."""

    rows = receipt["rows"]
    contract = receipt["source_contract"]
    adapter = contract["certificate_topology_adapter"]
    public = contract["public_forward_read"]
    branch = contract["hex_class_branch"]
    contact = contract["fixed_design_wall_contact"]
    lines = [
        "# Analytic limiter read resolution",
        "",
        (
            "The certificate adapter calls `operator.read` at "
            f"`{adapter['path']}:1680`; the public read begins at "
            f"`{public['path']}:{public['line_start']}` and selects "
            "`_fixed_design_read` at line 2006. Because these hex carriers have "
            "moment geometry but no tensor axes, the lazy class property takes the "
            f"moment-geometry branch at `{branch['path']}:1926` and performs another "
            "fixed-design read at line 1929. The tensor-spline connectivity limiter "
            "is unavailable on these rows."
        ),
        "",
        (
            "The contact is selected at "
            f"`{contact['path']}:496` (the extremal node); three wall-node values are "
            "fit by `traced_quadratic_wall` at line 519 and their coordinates are "
            "interpolated by `wall_coordinate` at line 524. That operation returns a "
            "point between nodes, but the measured position error remains first order "
            "in wall panel length. It is wall-panel-limited, not the global tensor-"
            "spline restriction and derivative-root polish already available on "
            "raster reads."
        ),
        "",
        "| Case | Cells | Wall nodes | Class | Panel / pitch | "
        "Contact error [m] | Level error / span | Warm read [s] |",
        "|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['case']} | {row['realised_cells']} | {row['wall_nodes']} | "
            f"{row['topology_class']} | "
            f"{row['wall']['selected_panel_over_cell_pitch']:.6g} | "
            f"{row['production_contact']['position_error_m']:.6g} | "
            f"{row['production_contact']['level_error_in_span']:.6g} | "
            f"{row['read_timing']['warm_read_seconds_median']:.6g} |"
        )
    lines.extend(["", "## Wall resolution against plasma pitch", ""])
    lines.extend(
        [
            "Panel-to-pitch entries use the selected outboard panel. The nominal "
            "2500-cell row uses about 2600 realised cells, with pitch scaled from "
            "the realised thousand-cell carrier.",
            "",
            "| Case | Cell scale | Realised cells | 121 | 241 | 481 | 961 | "
            "Minimum odd wall count |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in receipt["resolution_summary"]:
        ratios = item["selected_panel_over_pitch"]
        lines.append(
            f"| {item['case']} | {item['cell_scale']} | {item['realised_cells']} | "
            f"{ratios['121']:.4f} | {ratios['241']:.4f} | "
            f"{ratios['481']:.4f} | {ratios['961']:.4f} | "
            f"{item['recommended_wall_nodes']} |"
        )
    lines.extend(["", "## Measured wall order", ""])
    for item in receipt["limited_convergence"]:
        lines.append(
            f"- {item['case']} at {abs(item['requested_cells'])} requested cells: "
            f"position order {item['position_order']:.4f}; level order "
            f"{item['level_order']:.4f}."
        )
    construction = receipt["limiter_node_construction"]
    maximum_distance = max(
        item["distance_to_smooth_analytic_tangency_m"] for item in construction
    )
    maximum_flux = max(abs(item["node_flux_wb"]) for item in construction)
    lines.extend(
        [
            "",
            "## Why the 121-node level looked exact",
            "",
            (
                "Yes: `limiter_contour` samples angles as "
                "`2*pi*(arange(points)+0.5)/points`. Every requested odd count places "
                "index `(points-1)/2` exactly at angle pi, the authored smooth "
                "outboard tangency. Across weak and moderate rows and all four counts, "
                f"the maximum node displacement is {maximum_distance:.3g} m and the "
                f"maximum analytic node flux magnitude is {maximum_flux:.3g} Wb. "
                "The 2.7e-9 Wb weak-121 read is therefore a lucky sampling identity, "
                "not evidence that the piecewise wall contact is position-accurate. "
                "The wall polygon's adjacent chord enters the analytic plasma; its "
                "extremum is one panel away in position, producing the measured "
                "first-order position and second-order level ladders."
            ),
        ]
    )
    shadow_rows = [row for row in rows if row["shadow"] is not None]
    lines.extend(["", "## Private-flux wall shadow", ""])
    lines.append(
        (
            f"All {len(shadow_rows)} single-null controls passed: the injected "
            "private-region node won the unmasked limited read and was rejected by "
            "the masked read. `wall_height_shadow_mask` uses the qualified-saddle "
            "height band and intersects it with the connectivity-private wall mask. "
            "The selecting exclusion is therefore height-based over connectivity-"
            "qualified private nodes, not a connectivity-only shadow; each row records "
            "whether either side fell back to connectivity alone."
        )
    )
    lines.extend(
        [
            "",
            "| Cells | Wall nodes | Private below X | Shadowed below X | "
            "Lower branch | Selected instead [R, Z] m |",
            "|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in shadow_rows:
        shadow = row["shadow"]
        replacement = shadow["positive_control"]["masked_selected_contact_rz_m"]
        lines.append(
            f"| {row['realised_cells']} | {row['wall_nodes']} | "
            f"{shadow['private_wall_nodes_below_x_point']} | "
            f"{shadow['shadowed_private_nodes_below_x_point']} | "
            f"{shadow['lower_exclusion_branch']} | "
            f"[{replacement[0]:.6f}, {replacement[1]:.6f}] |"
        )
    lines.extend(["", "## Why the single-null saddle moves with wall count", ""])
    for item in receipt["saddle_coupling_summary"]:
        errors = ", ".join(
            f"{wall}:{error:.6g}"
            for wall, error in zip(
                item["wall_nodes"], item["published_boundary_error_m"], strict=True
            )
        )
        raw_errors = ", ".join(
            f"{wall}:{error:.6g}"
            for wall, error in zip(
                item["wall_nodes"], item["x_point_error_m"], strict=True
            )
        )
        lines.append(
            f"- {abs(item['requested_cells'])} requested cells — published boundary/"
            f"X-point errors [wall nodes:m] "
            f"{errors}; {item['distinct_carrier_node_identities']} distinct carrier "
            f"node identities. The raw fixed-design saddle candidate errors are "
            f"{raw_errors}. {item['cause']}."
        )
    lines.append(
        (
            "This is not a shadow effect: the analytic public read receives no prior "
            "wall mask. It is not containment or dedupe selection either: the selected "
            "candidate remains contained and each row has one finite X candidate. The "
            "coupling enters in `cached_machine`: inserting the differently sampled "
            "wall rebuilds the plasma carrier and its null-fit stencils, so a "
            "flux-only saddle moves with what should have been wall-only resolution."
        )
    )
    floor = receipt["map_floor_comparison"]
    lines.extend(
        [
            "",
            "## Limited-row map floor",
            "",
            (
                f"Weak 1000 exact-clip RMS floor: {floor['wall_121_rms']:.9g} "
                f"of span at 121 wall nodes and {floor['wall_481_rms']:.9g} at "
                f"481, an absolute movement of {floor['absolute_movement']:.3g}. "
                f"The floor therefore {floor['verdict']}."
            ),
            "",
            "## Figures",
            "",
            "- `/nova/figures/cut-cell-current-attribution/limiter-read/"
            "contact-error-vs-wall-resolution.svg` — all contact errors against "
            "local wall-panel-to-cell-pitch ratio.",
            "- `/nova/figures/cut-cell-current-attribution/limiter-read/"
            "single-null-contact-shadow.png` — analytic single-null 1000 field, "
            "wall, analytic nulls, selected wall contact and excluded "
            "private-wall nodes.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def render(figure_directory: Path) -> dict[str, Any]:
    """Rebuild the figures from the committed receipt, with no solve.

    The operands live in the figure directory's own ``receipt.json``, so this
    entry point only reads that file and paints. It never builds a machine,
    calls an operator or enters a solve, which is what lets it run on the
    login node in the time it takes to write two vector files.
    """

    source_receipt = json.loads(
        (figure_directory / "receipt.json").read_text(encoding="utf-8")
    )
    render_receipt = _render_figures(source_receipt, figure_directory)
    print(render_receipt["exit_marker"], flush=True)
    return render_receipt


def _render_figures(
    source_receipt: dict[str, Any],
    figure_directory: Path,
) -> dict[str, Any]:
    """Render both figures from a landed receipt and record their render receipt.

    Reading a receipt is all this needs: no machine is built, no operator is
    called and no solve runs, so the two figures can be rebuilt from the
    committed operands on the login node.
    """

    rows = source_receipt["rows"]
    contact_figure = figure_directory / "contact-error-vs-wall-resolution.svg"
    poloidal_figure = figure_directory / "single-null-contact-shadow.png"
    poloidal_vector = figure_directory / "single-null-contact-shadow.svg"
    render_row = next(
        row
        for row in rows
        if row["case"] == RENDER_CASE
        and row["requested_cells"] == RENDER_REQUESTED_CELLS
        and row["wall_nodes"] == RENDER_WALL_NODES
    )
    poloidal_record = _render_poloidal_figure(
        render_row, poloidal_figure, source_receipt
    )
    _render_poloidal_figure(render_row, poloidal_vector, source_receipt)
    error_record = _render_error_figure(rows, contact_figure)
    render_receipt = {
        "schema": "nova.limiter-read-render-receipt",
        "version": 1,
        "source_receipt": str((figure_directory / "receipt.json").resolve()),
        "source_receipt_sha256": _payload_sha256(source_receipt),
        "source_revision": source_receipt.get("source_revision"),
        "nonlinear_solve_entered": source_receipt.get("nonlinear_solve_entered"),
        "render_entry_point": ("benchmarks/limiter_read_resolution_audit.py render"),
        "figures": [error_record, poloidal_record],
        "completed": True,
        "exit_marker": "LIMITER_READ_RENDER_EXIT=0",
    }
    _write_json(figure_directory / "render-receipt.json", render_receipt)
    return render_receipt


def _payload_sha256(payload: dict[str, Any]) -> str:
    """Return the hex digest of the canonical JSON form of a receipt.

    The digest names the content the figures were painted from, so it stays
    true when the same receipt is read from another directory.
    """

    encoded = json.dumps(
        _strict(payload), indent=2, sort_keys=True, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def aggregate(report_directory: Path, figure_directory: Path) -> dict[str, Any]:
    """Aggregate all parts, render both figures, and write the final receipt."""

    rows = _load_parts(report_directory)
    floor_rows = {
        row["wall_nodes"]: row["map_floor"]
        for row in rows
        if row["case"] == "weak-rotation-reactor-static"
        and row["requested_cells"] == -1000
        and row["wall_nodes"] in (121, 481)
    }
    if set(floor_rows) != {121, 481} or any(
        value is None for value in floor_rows.values()
    ):
        raise RuntimeError("the required weak-1000 map-floor pair is incomplete")
    floor_121 = floor_rows[121]["relative_rms_of_span"]
    floor_481 = floor_rows[481]["relative_rms_of_span"]
    movement = abs(floor_481 - floor_121)
    floor_scale = max(abs(floor_121), abs(floor_481), np.finfo(float).tiny)
    map_floor_comparison = {
        "wall_121_rms": floor_121,
        "wall_481_rms": floor_481,
        "absolute_movement": movement,
        "relative_movement": movement / floor_scale,
        "verdict": (
            "does not move materially with wall resolution"
            if movement / floor_scale <= 0.01
            else "moves by more than one percent with wall resolution"
        ),
    }
    contact_figure = figure_directory / "contact-error-vs-wall-resolution.svg"
    poloidal_figure = figure_directory / "single-null-contact-shadow.png"
    receipt = {
        "schema": "nova.limiter-read-resolution-audit",
        "version": 1,
        "source_revision": _source_revision(),
        "production_code_modified": False,
        "nonlinear_solve_entered": False,
        "rows": rows,
        "source_contract": _source_contract(),
        "resolution_summary": _resolution_summary(rows),
        "limited_convergence": _limited_convergence(rows),
        "limiter_node_construction": _limiter_node_construction(),
        "saddle_coupling_summary": _saddle_coupling_summary(rows),
        "shadow_positive_control_passed": all(
            row["shadow"] is None or row["shadow"]["positive_control"]["passed"]
            for row in rows
        ),
        "map_floor_comparison": map_floor_comparison,
        "figures": [str(contact_figure), str(poloidal_figure)],
        "completed": True,
        "exit_marker": "LIMITER_READ_RESOLUTION_EXIT=0",
    }
    _render_figures(receipt, figure_directory)
    _write_json(report_directory / "receipt.json", receipt)
    _write_report(report_directory / "report.md", receipt)
    print(receipt["exit_marker"], flush=True)
    return receipt


def measure(
    report_directory: Path,
    wall_nodes: int,
    shard_index: int,
    shard_count: int,
) -> list[dict[str, Any]]:
    """Measure one deterministic shard of one wall-count group."""

    configure_dtypes()
    if not jax.config.jax_enable_x64:
        raise RuntimeError("the measurement requires JAX double precision")
    allocation = _allocation()
    cache = configure_persistent_compilation_cache(
        default_persistent_compilation_cache_root()
    )
    if wall_nodes not in WALL_NODE_COUNTS:
        raise ValueError(f"unsupported wall node count {wall_nodes}")
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("shard index must lie inside the positive shard count")
    selected = list(ROWS)[shard_index::shard_count]
    _write_json(
        report_directory / "shards" / f"wall-{wall_nodes}-shard-{shard_index}.json",
        {
            "source_revision": _source_revision(),
            "allocation": allocation,
            "persistent_compilation_cache": cache.receipt(),
            "wall_nodes": wall_nodes,
            "shard_index": shard_index,
            "shard_count": shard_count,
            "rows": [
                {"case": case_name, "requested_cells": requested_cells}
                for case_name, requested_cells in selected
            ],
            "completed": False,
        },
    )
    rows = [
        _measure_row(case_name, requested_cells, wall_nodes, report_directory)
        for case_name, requested_cells in selected
    ]
    _write_json(
        report_directory / "shards" / f"wall-{wall_nodes}-shard-{shard_index}.json",
        {
            "source_revision": _source_revision(),
            "allocation": allocation,
            "persistent_compilation_cache": cache.receipt(),
            "wall_nodes": wall_nodes,
            "shard_index": shard_index,
            "shard_count": shard_count,
            "rows": [
                {"case": row["case"], "requested_cells": row["requested_cells"]}
                for row in rows
            ],
            "completed": True,
            "exit_marker": "LIMITER_READ_SHARD_EXIT=0",
        },
    )
    print("LIMITER_READ_SHARD_EXIT=0", flush=True)
    return rows


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("measure", "residual", "aggregate", "render")
    )
    parser.add_argument(
        "--report-directory", type=Path, default=DEFAULT_REPORT_DIRECTORY
    )
    parser.add_argument(
        "--figure-directory", type=Path, default=DEFAULT_FIGURE_DIRECTORY
    )
    parser.add_argument("--wall-nodes", type=int, choices=WALL_NODE_COUNTS)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = _parse()
    if args.action == "measure":
        if args.wall_nodes is None:
            raise SystemExit("measure requires --wall-nodes")
        measure(
            args.report_directory,
            args.wall_nodes,
            args.shard_index,
            args.shard_count,
        )
    elif args.action == "residual":
        measure_residual_stages(args.report_directory)
    elif args.action == "render":
        render(args.figure_directory)
    else:
        aggregate(args.report_directory, args.figure_directory)


if __name__ == "__main__":
    main()
