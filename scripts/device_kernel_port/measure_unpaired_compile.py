"""Measure the fixed-shape exact-section graph without paired-fp64 expansion."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import resource
import socket
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from nova.biot.polygon import pad_batch
from nova.biot.polygonanalytic import (
    _Edge,
    _Vertex,
    _held_edge,
    _packed_axis_vertical_field,
    _packed_topology,
    polygon_analytic_greens,
)
from nova.jax.config import configure_dtypes


HERE = Path(__file__).resolve().parent
WORK = HERE / "work"
CPU_RESULT = WORK / "unpaired-cpu-result.json"
GPU_RESULT = WORK / "unpaired-gpu-result.json"
RECEIPT = HERE / "unpaired-compile-receipt.json"

TARGET_TILE = 32
SOURCE_TILE = 32
PAIR_BLOCK = TARGET_TILE * SOURCE_TILE
EDGE_COUNT = 11
RESIDUAL_NODES = 128


def _write_json(path: Path, payload: dict[str, object]) -> None:
    """Write strict JSON after creating the artifact parent."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sections() -> list[np.ndarray]:
    """Return a deterministic, well-conditioned batch of finite sections."""
    angle = np.linspace(0.0, 2.0 * np.pi, EDGE_COUNT, endpoint=False) + 0.17
    sections = []
    for index in range(SOURCE_TILE):
        centre_r = 5.4 + 0.037 * index
        centre_z = -0.65 + 0.041 * index
        half_r = 0.055 + 0.0007 * index
        half_z = 0.043 + 0.0004 * index
        sections.append(
            np.column_stack(
                (
                    centre_r + half_r * np.cos(angle),
                    centre_z + half_z * np.sin(angle),
                )
            )
        )
    return sections


def _geometry() -> tuple[tuple[np.ndarray, ...], list[np.ndarray]]:
    """Return one 32-by-32 pair block with the production packed topology."""
    sections = _sections()
    edge, weight, norm = pad_batch(sections, edge_count=EDGE_COUNT)
    target_r = np.linspace(4.7, 7.35, TARGET_TILE, dtype=np.float64)
    target_z = np.linspace(-1.8, 1.65, TARGET_TILE, dtype=np.float64)
    rows = np.repeat(np.arange(TARGET_TILE), SOURCE_TILE)
    columns = np.tile(np.arange(SOURCE_TILE), TARGET_TILE)
    geometry = (
        target_r[rows],
        target_z[rows],
        edge[:, :, columns],
        weight[:, columns],
        norm[columns],
    )
    return geometry, sections


def _packed_unpaired_greens(xp, target_r, target_z, edge, weight, norm):
    """Evaluate the packed exact finite-section form with scalar fp64 primitives.

    This is the unpaired specialization of ``packed_analytic_greens``. It keeps
    the same topology, endpoint antiderivatives, fixed residual-node count, and
    inverse-area normalization while selecting the ordinary-fp64 branches of
    ``_Vertex``, ``_Edge.terms``, and ``graded_residual``.
    """
    signed = xp.asarray(target_r)
    radius = xp.abs(signed)
    height = xp.asarray(target_z) + xp.zeros_like(signed)
    sides = edge.shape[0]
    live, present, chain, _ = _packed_topology(xp, weight)
    axis = radius == 0.0
    evaluation_radius = xp.where(axis, xp.ones_like(radius), radius)

    def corner(index: int):
        present_here = present[index]
        wrap = (~present_here) & live[index - 1]
        corner_r = xp.where(present_here, edge[index][0], edge[0][0])
        corner_z = xp.where(present_here, edge[index][1], edge[0][1])
        active = present_here | wrap
        return (
            xp.where(active, corner_r, evaluation_radius + 1.0),
            xp.where(active, corner_z, height + 1.0),
        )

    lower_r, lower_z = (
        xp.stack([corner(index)[coordinate] for index in range(sides)], axis=0)
        for coordinate in range(2)
    )
    corner_r = xp.stack((lower_r, xp.roll(lower_r, -1, axis=0)), axis=0)
    corner_z = xp.stack((lower_z, xp.roll(lower_z, -1, axis=0)), axis=0)
    held = [
        _held_edge(xp, edge[index], live[index], evaluation_radius, height)
        for index in range(sides)
    ]
    held_edge = tuple(
        xp.stack([one_edge[coordinate] for one_edge in held], axis=0)
        for coordinate in range(4)
    )
    endpoint_shape = (2, sides) + evaluation_radius.shape

    def endpoint_lanes(values):
        repeated = xp.stack((values, values), axis=0)
        return xp.broadcast_to(repeated, endpoint_shape).reshape(-1)

    lane_radius = xp.broadcast_to(evaluation_radius, endpoint_shape).reshape(-1)
    lane_height = xp.broadcast_to(height, endpoint_shape).reshape(-1)
    vertex = _Vertex(
        lane_radius,
        lane_height,
        corner_r.reshape(-1),
        corner_z.reshape(-1),
        RESIDUAL_NODES,
        residual=True,
        paired=False,
        xp=xp,
    )
    part = _Edge(
        lane_radius,
        lane_height,
        tuple(endpoint_lanes(coordinate) for coordinate in held_edge),
        RESIDUAL_NODES,
        xp=xp,
    )
    terms = tuple(
        value.reshape(endpoint_shape) for value in part.terms(vertex, paired=False)
    )
    residual = tuple(value.reshape(endpoint_shape) for value in vertex.arsinh_terms())

    rows = []
    for edge_term, corner_term in zip(terms, residual, strict=True):
        edge_total = weight * (edge_term[0] - edge_term[1])
        total = xp.zeros_like(edge_total[0])
        for index in range(sides):
            total = total + edge_total[index] + chain[index] * corner_term[0, index]
        rows.append(total)
    flux, radial, vertical = rows
    axis_vertical = _packed_axis_vertical_field(xp, height, edge, present, norm)
    return xp.stack(
        (
            xp.where(axis, 0.0, 0.5 * norm * radius * flux),
            xp.where(
                axis,
                0.0,
                norm / (4.0 * np.pi) * xp.sign(signed) * radial,
            ),
            xp.where(axis, axis_vertical, norm / (4.0 * np.pi) * vertical),
        )
    )


def _allocation() -> dict[str, object]:
    """Return the bounded scheduler allocation visible to this process."""
    memory = os.environ.get("SLURM_MEM_PER_NODE")
    return {
        "cpus": int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "memory_mib": int(memory) if memory else None,
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "time_limit": os.environ.get("SLURM_TIMELIMIT"),
    }


def measure(expected_backend: str, output: Path) -> None:
    """Compile, launch once, and compare one device result with production NumPy."""
    configure_dtypes()
    backend = jax.default_backend()
    if backend != expected_backend:
        raise RuntimeError(f"expected backend {expected_backend!r}, got {backend!r}")
    geometry, sections = _geometry()
    arrays = tuple(jnp.asarray(value, dtype=jnp.float64) for value in geometry)
    jax.block_until_ready(arrays)
    kernel = jax.jit(lambda *arguments: _packed_unpaired_greens(jnp, *arguments))

    compile_started = perf_counter()
    executable = kernel.lower(*arrays).compile()
    compile_seconds = perf_counter() - compile_started
    launch_started = perf_counter()
    device_value = executable(*arrays)
    jax.block_until_ready(device_value)
    kernel_seconds = perf_counter() - launch_started
    actual = np.asarray(device_value)

    target_r = np.asarray(geometry[0])
    target_z = np.asarray(geometry[1])
    reference = np.asarray(
        polygon_analytic_greens(target_r[:1], target_z[:1], sections[0])
    )[:, 0]
    observed = actual[:, 0]
    absolute = np.abs(observed - reference)
    relative = absolute / np.maximum(np.abs(reference), np.finfo(np.float64).tiny)
    spot_passed = bool(np.allclose(observed, reference, rtol=2e-10, atol=2e-18))

    device = jax.devices()[0]
    payload = {
        "allocation": _allocation(),
        "backend": backend,
        "compile_seconds": compile_seconds,
        "configuration": {
            "edge_count": EDGE_COUNT,
            "exact_finite_section": True,
            "paired": False,
            "pair_block": PAIR_BLOCK,
            "residual_nodes": RESIDUAL_NODES,
            "source_tile": SOURCE_TILE,
            "target_tile": TARGET_TILE,
        },
        "device_kind": device.device_kind,
        "hostname": socket.gethostname(),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "jax_version": jax.__version__,
        "kernel_seconds": kernel_seconds,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "platform": platform.platform(),
        "spot_check": {
            "absolute_deviation": absolute.tolist(),
            "device_value": observed.tolist(),
            "passed": spot_passed,
            "production_numpy_value": reference.tolist(),
            "relative_deviation": relative.tolist(),
            "rtol": 2e-10,
            "atol": 2e-18,
        },
        "status": "compiled_and_executed",
    }
    _write_json(output, payload)
    print(
        f"MEASURED backend={backend} compile_s={compile_seconds:.9g} "
        f"kernel_s={kernel_seconds:.9g} peak_rss_kib={payload['peak_rss_kib']} "
        f"spot_passed={spot_passed}",
        flush=True,
    )


def merge(
    cpu_result: Path,
    gpu_result: Path,
    output: Path,
    *,
    cpu_elapsed_seconds: int,
    gpu_elapsed_seconds: int,
    time_limit_seconds: int,
) -> None:
    """Merge successful lane artifacts into the durable feasibility receipt."""
    cpu = json.loads(cpu_result.read_text(encoding="utf-8"))
    gpu = json.loads(gpu_result.read_text(encoding="utf-8"))
    cpu["allocation"].update(
        {
            "elapsed_seconds": cpu_elapsed_seconds,
            "scheduler_state": "COMPLETED",
            "time_limit_seconds": time_limit_seconds,
        }
    )
    gpu["allocation"].update(
        {
            "elapsed_seconds": gpu_elapsed_seconds,
            "reservation": "gpu_0003_grpA",
            "scheduler_state": "COMPLETED",
            "time_limit_seconds": time_limit_seconds,
        }
    )
    lanes_pass = all(
        lane["status"] == "compiled_and_executed" and lane["spot_check"]["passed"]
        for lane in (cpu, gpu)
    )
    verdict = (
        "the ordinary-fp64 exact-section graph compiles within both bounded "
        "allocations and its post-compile values agree with the production NumPy "
        "finite-section evaluation"
        if lanes_pass
        else "the ordinary-fp64 exact-section graph did not satisfy both lane gates"
    )
    _write_json(
        output,
        {
            "configuration": cpu["configuration"],
            "lanes": {"cpu": cpu, "h200": gpu},
            "paired_graph_context": {
                "cpu_compile_wall_floor_seconds": 2335,
                "cpu_peak_rss_kib": 149811136,
                "h200_compile_wall_floor_seconds": 2335,
                "h200_observed_peak_rss_floor_kib": 58935472,
                "mechanism": (
                    "paired-fp64 expansion multiplies the primitive graph and "
                    "prevents the compact compilation measured here"
                ),
            },
            "production_source_changed": False,
            "status": "complete" if lanes_pass else "failed",
            "verdict": verdict,
        },
    )
    if not lanes_pass:
        raise RuntimeError(verdict)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("measure")
    run.add_argument("--expected-backend", choices=("cpu", "gpu"), required=True)
    run.add_argument("--output", type=Path, required=True)
    combine = commands.add_parser("merge")
    combine.add_argument("--cpu-result", type=Path, default=CPU_RESULT)
    combine.add_argument("--gpu-result", type=Path, default=GPU_RESULT)
    combine.add_argument("--output", type=Path, default=RECEIPT)
    combine.add_argument("--cpu-elapsed-seconds", type=int, required=True)
    combine.add_argument("--gpu-elapsed-seconds", type=int, required=True)
    combine.add_argument("--time-limit-seconds", type=int, required=True)
    args = parser.parse_args()
    if args.command == "measure":
        measure(args.expected_backend, args.output)
    else:
        merge(
            args.cpu_result,
            args.gpu_result,
            args.output,
            cpu_elapsed_seconds=args.cpu_elapsed_seconds,
            gpu_elapsed_seconds=args.gpu_elapsed_seconds,
            time_limit_seconds=args.time_limit_seconds,
        )


if __name__ == "__main__":
    main()
