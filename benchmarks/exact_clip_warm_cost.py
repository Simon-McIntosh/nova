"""Measure exact-clip warm solve cost and the device workspace that carries it.

The executable entry point runs one weak Solovev row per process so allocator
peaks belong to one cell-count rung.  The parent command launches every child
inside its existing scheduler allocation, persists each part immediately, and
then aggregates the scaling receipt and figure.

The current production implementation already scans a fixed-capacity bank of
cut cells.  The two reference rungs therefore compare that production scan to
the former compact integrator, rather than presenting the scan as hypothetical.
The terminal current moments must be bit-identical before either memory or wall
measurement is admitted.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import tempfile
from time import perf_counter
from typing import Any, Callable

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmarks import solovev_certificate as certificate
from nova.equilibrium import clip_quadrature
from nova.equilibrium import forward_operator
from nova.equilibrium.forward_operator import set_support_clip_mode
from nova.equilibrium.topology import TopologyClass
from nova.jax.config import configure_dtypes, configure_persistent_compilation_cache


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_ROOT = Path(forward_operator.__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = ROOT / "docs/figures/cut-cell-current-attribution/exact-clip-cost"
DEFAULT_REPORT_ROOT = Path(
    "/home/ITER/mcintos/.config/reckon/crew/reports/nova/s19-handoff/exact-clip-cost"
)
REQUESTED_CELLS = (110, 300, 1000)
REFERENCE_CELLS = (110, 300)
REPEATS = 3
CASE = "weak-rotation-reactor-static"
SCHEMA = "nova.exact-clip-warm-cost"
_ALLOCATION = re.compile(r"^allocation (?P<index>\d+): size (?P<size>\d+)(?P<tail>.*)$")
_VALUE = re.compile(
    r"^ value: <(?P<identity>[^>]+)> \(size=(?P<size>\d+),offset=(?P<offset>\d+)\): "
    r"(?P<shape>.+)$"
)


def _strict(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _strict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        json.dump(_strict(payload), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def _revision(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _lane() -> dict[str, Any]:
    device = jax.devices()[0]
    job = os.environ.get("SLURM_JOB_ID")
    if job is None:
        raise RuntimeError("the cost measurement requires one scheduler allocation")
    if device.platform != "gpu" or "H200" not in device.device_kind:
        raise RuntimeError(f"the cost measurement requires one H200, got {device}")
    if os.environ.get("SLURM_JOB_PARTITION") != "betelgeuse":
        raise RuntimeError("the cost measurement requires the betelgeuse partition")
    if os.environ.get("SLURM_JOB_RESERVATION") != "gpu_0003_grpA":
        raise RuntimeError("the cost measurement requires the declared reservation")
    if os.environ.get("SLURM_CPUS_PER_TASK") != "8":
        raise RuntimeError("the cost measurement requires eight requested CPUs")
    if os.environ.get("TMPDIR") != "/tmp":
        raise RuntimeError("TMPDIR must be /tmp inside the allocation")
    return {
        "job_id": int(job),
        "partition": os.environ["SLURM_JOB_PARTITION"],
        "reservation": os.environ["SLURM_JOB_RESERVATION"],
        "node": os.environ.get("SLURMD_NODENAME", socket.gethostname()),
        "allocated_cpus": int(os.environ["SLURM_CPUS_PER_TASK"]),
        "memory_mb": int(os.environ.get("SLURM_MEM_PER_NODE", "0")),
        "device": device.device_kind,
        "platform": device.platform,
        "jax_platforms": os.environ.get("JAX_PLATFORMS"),
        "tmpdir": os.environ["TMPDIR"],
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
    }


def _require_promoted_source() -> dict[str, Any]:
    implicit = hasattr(forward_operator, "_implicit_level_root_jvp")
    scanned = hasattr(forward_operator, "_cell_banked_current_moments")
    if not implicit or not scanned:
        raise RuntimeError(
            "promoted exact-clip source is absent: the implicit-root derivative "
            "and cell-banked scan are both required"
        )
    production_revision = _revision(PRODUCTION_ROOT)
    expected_revision = os.environ.get("EXACT_CLIP_PRODUCTION_REVISION")
    if expected_revision is not None and production_revision != expected_revision:
        raise RuntimeError(
            "production source revision moved after submission: "
            f"expected {expected_revision}, observed {production_revision}"
        )
    return {
        "production_root": str(PRODUCTION_ROOT),
        "production_revision": production_revision,
        "benchmark_root": str(ROOT),
        "benchmark_revision": _revision(ROOT),
        "implicit_root_derivative_present": implicit,
        "cell_banked_scan_present": scanned,
    }


def _memory_fields(analysis: Any) -> dict[str, int]:
    return {
        name: int(getattr(analysis, name, 0) or 0)
        for name in (
            "generated_code_size_in_bytes",
            "argument_size_in_bytes",
            "output_size_in_bytes",
            "alias_size_in_bytes",
            "temp_size_in_bytes",
            "host_generated_code_size_in_bytes",
            "host_argument_size_in_bytes",
            "host_output_size_in_bytes",
            "host_alias_size_in_bytes",
            "host_temp_size_in_bytes",
        )
    }


def _device_statistics() -> dict[str, int | str]:
    raw = jax.devices()[0].memory_stats() or {}
    statistics = {
        str(name): int(value) if isinstance(value, int | float) else str(value)
        for name, value in raw.items()
    }
    byte_counters = {
        name: value
        for name, value in statistics.items()
        if "bytes" in name and isinstance(value, int)
    }
    peak = statistics.get("peak_bytes_in_use")
    largest = statistics.get("largest_alloc_size")
    if not byte_counters or not isinstance(peak, int) or peak <= 0:
        raise RuntimeError("accelerator counters did not observe a positive peak")
    if not isinstance(largest, int) or largest <= 0:
        raise RuntimeError("accelerator counters did not name a largest allocation")
    return statistics


def _parse_buffer_assignment(dump_root: Path) -> dict[str, Any]:
    candidates = sorted(dump_root.glob("*solve_program*buffer-assignment.txt"))
    if not candidates:
        raise RuntimeError("optimized solve buffer assignment was not dumped")
    path = candidates[-1]
    allocations: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    with path.open(encoding="utf-8", errors="replace") as stream:
        for raw in stream:
            line = raw.rstrip("\n")
            allocation = _ALLOCATION.match(line)
            if allocation:
                current = {
                    "index": int(allocation.group("index")),
                    "size_bytes": int(allocation.group("size")),
                    "description": allocation.group("tail").strip(),
                    "values": [],
                }
                allocations.append(current)
                continue
            value = _VALUE.match(line)
            if value and current is not None:
                current["values"].append(
                    {
                        "identity": value.group("identity"),
                        "size_bytes": int(value.group("size")),
                        "offset": int(value.group("offset")),
                        "shape": value.group("shape"),
                    }
                )
    non_parameters = [
        item for item in allocations if "parameter" not in item["description"]
    ]
    if not non_parameters:
        raise RuntimeError("buffer assignment named no non-parameter allocation")
    largest = max(non_parameters, key=lambda item: item["size_bytes"])
    if not largest["values"]:
        raise RuntimeError("largest allocation carries no named optimized HLO value")
    largest_value = max(largest["values"], key=lambda item: item["size_bytes"])
    memory_reports = sorted(dump_root.glob("*solve_program*memory-usage-report.txt"))
    if not memory_reports:
        raise RuntimeError("optimized solve memory report was not dumped")
    retained = []
    for source in (path, memory_reports[-1]):
        target = source.with_suffix(source.suffix + ".gz")
        with source.open("rb") as incoming, gzip.open(target, "wb") as outgoing:
            while block := incoming.read(1024 * 1024):
                outgoing.write(block)
        retained.append(
            {
                "path": str(target),
                "sha256": _sha256(target),
                "uncompressed_bytes": source.stat().st_size,
            }
        )
    return {
        "largest_nonparameter_allocation": {
            key: value for key, value in largest.items() if key != "values"
        },
        "largest_value_in_allocation": largest_value,
        "retained_optimized_hlo_evidence": retained,
        "instrument_check": {
            "allocation_count": len(allocations),
            "named_values_in_largest_allocation": len(largest["values"]),
            "positive_largest_value": largest_value["size_bytes"] > 0,
        },
    }


def _solve_program(profile: Any, request: Any) -> Callable[[jax.Array], Any]:
    mapped = profile.flux_map(
        request.current,
        target_current=request.target_current,
        prescribed_current=request.prescribed_current,
    )
    shadowed = profile.operator.flux_map_with_shadow(
        request.current,
        target_current=request.target_current,
        prescribed_current=request.prescribed_current,
    )

    def shadow(state):
        return profile.operator.residual_shadow_mask(state)

    def promoted(state, previous):
        return profile.operator.residual_shadow_mask(state, previous_shadow=previous)

    def solve_program(initial):
        result = certificate.recovery.fixed_point.newton_krylov(
            mapped,
            initial,
            shadow_mask_fn=shadow,
            promoted_shadow_mask_fn=promoted,
            shadowed_map_fn=shadowed,
            **request.policy.kernel_options(),
        )
        return (
            result.state,
            result.residual,
            result.converged,
            result.active_set_iterations,
            result.active_set_residuals,
            result.active_set_mask_differences,
        )

    return solve_program


def _certificate_problem(cells: int) -> tuple[Any, np.ndarray, Any, dict[str, Any]]:
    """Build one certificate row across the quadrature-node ownership move."""
    if not hasattr(certificate.observation, "_UNIT_NODE"):
        certificate.observation._UNIT_NODE = clip_quadrature._UNIT_NODE
    return certificate._certificate_compile_problem(CASE, -abs(cells))


def _compile_and_time(
    function: Callable[[Any], Any], argument: Any, repeats: int = REPEATS
) -> tuple[Any, dict[str, Any], Any]:
    compile_started = perf_counter()
    compiled = jax.jit(function).lower(argument).compile()
    compile_wall = perf_counter() - compile_started
    walls = []
    result = None
    for _ in range(repeats):
        started = perf_counter()
        result = jax.block_until_ready(compiled(argument))
        walls.append(perf_counter() - started)
    values = np.asarray(walls, dtype=np.float64)
    return (
        compiled,
        {
            "compile_wall_seconds": compile_wall,
            "warm_wall_seconds": values,
            "warm_min_seconds": float(values.min()),
            "warm_median_seconds": float(np.median(values)),
            "warm_max_seconds": float(values.max()),
            "memory_analysis": _memory_fields(compiled.memory_analysis()),
        },
        result,
    )


def _stage_times(profile: Any, seed: jax.Array) -> dict[str, Any]:
    operator = profile.operator
    requested = int(TopologyClass.LIMITED)

    def topology(candidate):
        physical = candidate[: operator.physical_node_number]
        masks, read, connected, admitted = operator._fixed_design_read(
            physical, requested
        )
        return masks, read, connected, admitted

    def boundary(candidate):
        return operator._frozen_topology_partition(candidate, requested)

    _compiled, topology_timing, _result = _compile_and_time(topology, seed)
    _compiled, boundary_timing, frozen = _compile_and_time(boundary, seed)
    partition = operator._partition_for_state(seed, frozen)
    partition = jax.block_until_ready(partition)

    def moments(carried):
        return operator._partitioned_current_moments(carried)

    _compiled, moment_timing, moment_result = _compile_and_time(moments, partition)
    cut_count = int(jnp.sum(jnp.asarray(frozen.profile_support.boundary)))
    current = np.asarray(moment_result.cell_current)
    if cut_count <= 0 or not np.any(current != 0.0):
        raise RuntimeError("stage instrument saw neither cut cells nor nonzero current")
    topology_wall = topology_timing["warm_median_seconds"]
    boundary_wall = boundary_timing["warm_median_seconds"]
    moment_wall = moment_timing["warm_median_seconds"]
    return {
        "topology_read": topology_timing,
        "compiled_boundary": boundary_timing,
        "clipped_moments_and_quadrature": moment_timing,
        "cut_cell_count": cut_count,
        "instrument_check": {
            "known_cut_cells_present": True,
            "nonzero_current_cells": int(np.count_nonzero(current)),
        },
        "clip_wall_seconds": max(boundary_wall - topology_wall, 0.0) + moment_wall,
    }


def _terminal_summary(result: Any) -> dict[str, Any]:
    state, residual, converged, trips, trip_residuals, mask_differences = result
    state_array = np.ascontiguousarray(np.asarray(state), dtype="<f8")
    trip_count = int(trips)
    if trip_count <= 0:
        raise RuntimeError("production solve reported no executed active-set trip")
    return {
        "state_sha256_binary64": hashlib.sha256(state_array.tobytes()).hexdigest(),
        "state": state_array,
        "residual": float(residual),
        "converged": bool(converged),
        "active_set_iterations": trip_count,
        "active_set_residuals": np.asarray(trip_residuals),
        "active_set_mask_differences": np.asarray(mask_differences),
    }


def _production_worker(
    cells: int, output: Path, cache_root: Path, dump_root: Path
) -> dict[str, Any]:
    configure_dtypes()
    if not jax.config.jax_enable_x64:
        raise RuntimeError("the exact-clip cost requires binary64")
    source = _require_promoted_source()
    cache = configure_persistent_compilation_cache(cache_root)
    set_support_clip_mode("exact")
    profile, seed, request, dimensions = _certificate_problem(cells)
    seed_array = jnp.asarray(seed, dtype=jnp.float64)
    compiled, solve_timing, result = _compile_and_time(
        _solve_program(profile, request), seed_array
    )
    terminal = _terminal_summary(result)
    stages = _stage_times(profile, seed_array)
    per_trip = solve_timing["warm_median_seconds"] / terminal["active_set_iterations"]
    stages["fraction_of_warm_trip_attributed_to_clip"] = min(
        stages["clip_wall_seconds"] / per_trip, 1.0
    )
    allocator = _device_statistics()
    buffer_assignment = _parse_buffer_assignment(dump_root)
    largest_alloc = int(allocator["largest_alloc_size"])
    assigned = int(buffer_assignment["largest_nonparameter_allocation"]["size_bytes"])
    if abs(largest_alloc - assigned) > max(4096, int(1.0e-5 * largest_alloc)):
        raise RuntimeError(
            "allocator largest allocation and optimized buffer assignment disagree"
        )
    receipt = {
        "schema": SCHEMA,
        "route": "production_cell_banked_scan",
        "case": CASE,
        "requested_cells": cells,
        "realised_cells": dimensions["realised_cells"],
        "dimensions": dimensions,
        "source": source,
        "lane": _lane(),
        "persistent_compilation_cache": cache.receipt(),
        "solve": solve_timing
        | {
            "warm_wall_per_trip_seconds": per_trip,
            "repeats_from_identical_seed": REPEATS,
        },
        "stages": stages,
        "terminal": terminal,
        "allocator": {
            "statistics": allocator,
            "peak_bytes": allocator["peak_bytes_in_use"],
            "largest_single_allocation_bytes": largest_alloc,
        },
        "optimized_hlo_buffer_assignment": buffer_assignment,
        "completed": True,
    }
    _atomic_json(output, receipt)
    print(
        "EXACT_CLIP_COST "
        f"requested={cells} realised={dimensions['realised_cells']} "
        f"compile_s={solve_timing['compile_wall_seconds']:.6f} "
        f"warm_trip_s={per_trip:.6f} "
        f"peak_gib={int(allocator['peak_bytes_in_use']) / 2**30:.6f}",
        flush=True,
    )
    return receipt


@contextmanager
def _former_compact_integrator():
    original = forward_operator._cell_banked_current_moments
    forward_operator._cell_banked_current_moments = (
        clip_quadrature.clipped_support_current_moments
    )
    try:
        yield
    finally:
        forward_operator._cell_banked_current_moments = original


def _moment_worker(cells: int, production_part: Path, output: Path) -> dict[str, Any]:
    configure_dtypes()
    source = _require_promoted_source()
    set_support_clip_mode("exact")
    production = json.loads(production_part.read_text(encoding="utf-8"))
    terminal = jnp.asarray(production["terminal"]["state"], dtype=jnp.float64)
    profile, _seed, _request, dimensions = _certificate_problem(cells)
    partition = jax.block_until_ready(
        profile.operator._support_partition(terminal, int(TopologyClass.LIMITED))
    )

    def measured(carried):
        return profile.operator._partitioned_current_moments(carried)

    scan_compiled, scan_timing, scan = _compile_and_time(measured, partition)
    with _former_compact_integrator():
        compact_compiled, compact_timing, compact = _compile_and_time(
            measured, partition
        )
    comparisons = {}
    for name in scan._fields:
        left = np.asarray(getattr(scan, name))
        right = np.asarray(getattr(compact, name))
        comparisons[name] = {
            "bit_identical": bool(np.array_equal(left, right)),
            "maximum_absolute_difference": float(np.max(np.abs(left - right))),
        }
    if not all(item["bit_identical"] for item in comparisons.values()):
        raise RuntimeError("cell-banked scan changed terminal current moments")
    scan_memory = _memory_fields(scan_compiled.memory_analysis())
    compact_memory = _memory_fields(compact_compiled.memory_analysis())
    receipt = {
        "schema": f"{SCHEMA}.moment-route-comparison",
        "case": CASE,
        "requested_cells": cells,
        "realised_cells": dimensions["realised_cells"],
        "source": source,
        "lane": _lane(),
        "terminal_state_sha256_binary64": production["terminal"][
            "state_sha256_binary64"
        ],
        "production_cell_banked_scan": scan_timing | {"memory_analysis": scan_memory},
        "former_compact_integrator": compact_timing
        | {"memory_analysis": compact_memory},
        "terminal_moment_comparison": comparisons,
        "memory_floor_bytes": scan_memory["temp_size_in_bytes"],
        "completed": True,
    }
    _atomic_json(output, receipt)
    print(
        "EXACT_CLIP_MOMENT_ROUTES "
        f"requested={cells} scan_temp={scan_memory['temp_size_in_bytes']} "
        f"compact_temp={compact_memory['temp_size_in_bytes']} bit_identical=true",
        flush=True,
    )
    return receipt


def _scaling_exponent(
    rows: list[dict[str, Any]], field: Callable[[dict], float]
) -> float:
    cells = np.asarray([row["realised_cells"] for row in rows], dtype=np.float64)
    values = np.asarray([field(row) for row in rows], dtype=np.float64)
    if np.any(values <= 0.0):
        raise RuntimeError("scaling fit requires positive measurements")
    return float(np.polyfit(np.log(cells), np.log(values), 1)[0])


def _historical_whole_cell_reference() -> dict[str, Any]:
    trip = PRODUCTION_ROOT / "docs/figures/solver-trip-orchestration/trip-quantum.json"
    width = (
        PRODUCTION_ROOT / "docs/figures/millisecond-converged-solve/trip-quantum/"
        "width-one-compiled.json"
    )
    trip_payload = json.loads(trip.read_text(encoding="utf-8"))
    width_payload = json.loads(width.read_text(encoding="utf-8"))
    direct = (
        float(trip_payload["banked_quantum"]["measured_ms_per_member_per_trip"]) / 1.0e3
    )
    boundaries = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "boundary_s" and isinstance(item, int | float):
                    boundaries.append(float(item))
                else:
                    visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(width_payload)
    one_trip_boundaries = [value for value in boundaries if 0.12 < value < 0.15]
    if not one_trip_boundaries:
        raise RuntimeError("whole-cell one-trip reference named no boundary wall")
    return {
        "compiled_trip_seconds": float(direct),
        "one_trip_solve_seconds": float(np.median(one_trip_boundaries)),
        "scope": "banked H200 whole-cell reference, not remeasured by this job",
        "sources": [
            {"path": str(trip.relative_to(PRODUCTION_ROOT)), "sha256": _sha256(trip)},
            {"path": str(width.relative_to(PRODUCTION_ROOT)), "sha256": _sha256(width)},
        ],
    }


def _draw(
    path: Path,
    production: list[dict[str, Any]],
    moments: list[dict[str, Any]],
    whole: dict[str, Any],
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), constrained_layout=True)
    cells = np.asarray([row["realised_cells"] for row in production])
    wall = np.asarray(
        [row["solve"]["warm_wall_per_trip_seconds"] for row in production]
    )
    peak = np.asarray([row["allocator"]["peak_bytes"] for row in production]) / 2**30
    axes[0].loglog(
        cells, wall, "o-", color="#7a3e9d", label="exact clip, production scan"
    )
    axes[0].axhline(
        whole["compiled_trip_seconds"],
        color="#3366cc",
        linestyle="--",
        label="whole-cell compiled trip reference",
    )
    axes[0].axhline(
        whole["one_trip_solve_seconds"],
        color="#3366cc",
        linestyle=":",
        label="whole-cell one-trip solve reference",
    )
    axes[0].set_xlabel("realised cells")
    axes[0].set_ylabel("warm wall per trip [s]")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].loglog(
        cells, peak, "o-", color="#7a3e9d", label="full exact solve allocator peak"
    )
    moment_cells = np.asarray([row["realised_cells"] for row in moments])
    scan = (
        np.asarray(
            [
                row["production_cell_banked_scan"]["memory_analysis"][
                    "temp_size_in_bytes"
                ]
                for row in moments
            ]
        )
        / 2**30
    )
    compact = (
        np.asarray(
            [
                row["former_compact_integrator"]["memory_analysis"][
                    "temp_size_in_bytes"
                ]
                for row in moments
            ]
        )
        / 2**30
    )
    axes[1].loglog(
        moment_cells, scan, "s--", color="#2a8c6f", label="moment scan temporary"
    )
    axes[1].loglog(
        moment_cells, compact, "x:", color="#cc7722", label="former compact temporary"
    )
    axes[1].set_xlabel("realised cells")
    axes[1].set_ylabel("peak or compiled temporary [GiB]")
    axes[1].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.grid(False)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, format="svg")
    plt.close(figure)


def _aggregate(output_root: Path, report_root: Path) -> dict[str, Any]:
    production = [
        json.loads((output_root / "parts" / f"production-{cells}.json").read_text())
        for cells in REQUESTED_CELLS
    ]
    moments = [
        json.loads((output_root / "parts" / f"moments-{cells}.json").read_text())
        for cells in REFERENCE_CELLS
    ]
    whole = _historical_whole_cell_reference()
    figure = output_root / "exact-clip-warm-cost-and-memory.svg"
    _draw(figure, production, moments, whole)
    receipt = {
        "schema": SCHEMA,
        "source": production[-1]["source"],
        "lane": production[-1]["lane"],
        "production_rows": production,
        "moment_route_rows": moments,
        "whole_cell_reference": whole,
        "scaling": {
            "warm_wall_per_trip_exponent_against_realised_cells": _scaling_exponent(
                production, lambda row: row["solve"]["warm_wall_per_trip_seconds"]
            ),
            "allocator_peak_exponent_against_realised_cells": _scaling_exponent(
                production, lambda row: row["allocator"]["peak_bytes"]
            ),
        },
        "figure": {
            "filesystem_path": str(figure),
            "project_absolute_src": (
                "/nova/figures/cut-cell-current-attribution/exact-clip-cost/"
                "exact-clip-warm-cost-and-memory.svg"
            ),
            "sha256": _sha256(figure),
        },
        "completed": True,
    }
    _atomic_json(output_root / "receipt.json", receipt)
    production_revision = receipt["source"]["production_revision"]
    job_id = receipt["lane"]["job_id"]
    wall_exponent = receipt["scaling"][
        "warm_wall_per_trip_exponent_against_realised_cells"
    ]
    peak_exponent = receipt["scaling"]["allocator_peak_exponent_against_realised_cells"]
    lines = [
        "# Exact-clip warm solve cost and workspace",
        "",
        f"Production revision: `{production_revision}`. H200 job: `{job_id}`.",
        "",
        "| requested | realised | cold compile s | warm fixed point s | trips | "
        "warm s/trip | clip fraction | peak GiB | largest allocation GiB |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in production:
        lines.append(
            (
                "| {requested_cells} | {realised_cells} | {compile:.3f} | "
                "{warm:.6f} | {trips} | {per_trip:.6f} | {fraction:.3f} | "
                "{peak:.3f} | {largest:.3f} |"
            ).format(
                requested_cells=row["requested_cells"],
                realised_cells=row["realised_cells"],
                compile=row["solve"]["compile_wall_seconds"],
                warm=row["solve"]["warm_median_seconds"],
                trips=row["terminal"]["active_set_iterations"],
                per_trip=row["solve"]["warm_wall_per_trip_seconds"],
                fraction=row["stages"]["fraction_of_warm_trip_attributed_to_clip"],
                peak=row["allocator"]["peak_bytes"] / 2**30,
                largest=row["allocator"]["largest_single_allocation_bytes"] / 2**30,
            )
        )
    lines += [
        "",
        f"Warm wall scaling exponent against realised cells: `{wall_exponent:.4f}`. "
        f"Allocator-peak exponent: `{peak_exponent:.4f}`.",
        "",
        "The production source already carries the fixed-capacity cut-cell scan. "
        "The paired 110- and 300-cell comparison therefore treats the former "
        "compact integrator as the reference and requires every terminal current "
        "moment to be bit-identical.",
        "",
        "| requested | scan wall s | compact wall s | scan temporary GiB | "
        "compact temporary GiB | bit-identical |",
        "|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in moments:
        identical = all(
            item["bit_identical"] for item in row["terminal_moment_comparison"].values()
        )
        lines.append(
            (
                "| {requested_cells} | {scan_wall:.6f} | {compact_wall:.6f} | "
                "{scan_memory:.6f} | {compact_memory:.6f} | {identical} |"
            ).format(
                requested_cells=row["requested_cells"],
                scan_wall=row["production_cell_banked_scan"]["warm_median_seconds"],
                compact_wall=row["former_compact_integrator"]["warm_median_seconds"],
                scan_memory=row["production_cell_banked_scan"]["memory_analysis"][
                    "temp_size_in_bytes"
                ]
                / 2**30,
                compact_memory=row["former_compact_integrator"]["memory_analysis"][
                    "temp_size_in_bytes"
                ]
                / 2**30,
                identical="yes" if identical else "no",
            )
        )
    lines += [
        "",
        "The optimized buffer-assignment evidence in each production part names "
        "the largest temporary allocation and its largest live HLO value. The "
        "device allocator peak is independently required to be positive and its "
        "largest allocation must agree with that assignment.",
        "",
        "Whole-cell context from the banked H200 receipts is "
        f"`{whole['compiled_trip_seconds']:.6f}` s for the compiled trip and "
        f"`{whole['one_trip_solve_seconds']:.6f}` s for the one-trip solve.",
        "",
        "![Warm exact-clip cost and memory]"
        f"({receipt['figure']['project_absolute_src']})",
        "",
    ]
    report_root.mkdir(parents=True, exist_ok=True)
    report = report_root / "report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return receipt


def _child_environment(dump_root: Path) -> dict[str, str]:
    environment = dict(os.environ)
    existing = environment.get("XLA_FLAGS", "").strip()
    flags = (
        f"--xla_dump_to={dump_root} --xla_dump_hlo_as_text "
        "--xla_dump_hlo_module_re=solve_program"
    )
    environment["XLA_FLAGS"] = f"{existing} {flags}".strip()
    environment["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(PRODUCTION_ROOT), str(ROOT), environment.get("PYTHONPATH", "")]
    )
    return environment


def _orchestrate(
    output_root: Path, report_root: Path, cache_root: Path
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)
    parts = output_root / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    for cells in REQUESTED_CELLS:
        part = parts / f"production-{cells}.json"
        dump = report_root / "xla" / f"production-{cells}"
        dump.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "production",
                "--cells",
                str(cells),
                "--output",
                str(part),
                "--cache-root",
                str(cache_root),
                "--dump-root",
                str(dump),
            ],
            check=True,
            env=_child_environment(dump),
        )
    for cells in REFERENCE_CELLS:
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "moments",
                "--cells",
                str(cells),
                "--production-part",
                str(parts / f"production-{cells}.json"),
                "--output",
                str(parts / f"moments-{cells}.json"),
            ],
            check=True,
            env=dict(os.environ)
            | {
                "PYTHONPATH": os.pathsep.join(
                    [str(PRODUCTION_ROOT), str(ROOT), os.environ.get("PYTHONPATH", "")]
                )
            },
        )
    return _aggregate(output_root, report_root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=("production", "moments"))
    parser.add_argument("--cells", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--production-part", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--dump-root", type=Path)
    arguments = parser.parse_args()
    if arguments.worker == "production":
        if None in (
            arguments.cells,
            arguments.output,
            arguments.cache_root,
            arguments.dump_root,
        ):
            parser.error(
                "production worker requires cells, output, cache-root and dump-root"
            )
        _production_worker(
            arguments.cells,
            arguments.output.resolve(),
            arguments.cache_root.resolve(),
            arguments.dump_root.resolve(),
        )
        return
    if arguments.worker == "moments":
        if None in (arguments.cells, arguments.output, arguments.production_part):
            parser.error("moment worker requires cells, output and production-part")
        _moment_worker(
            arguments.cells,
            arguments.production_part.resolve(),
            arguments.output.resolve(),
        )
        return
    cache_root = arguments.cache_root or (arguments.report_root / "compilation-cache")
    _orchestrate(
        arguments.output_root.resolve(),
        arguments.report_root.resolve(),
        cache_root.resolve(),
    )


if __name__ == "__main__":
    main()
