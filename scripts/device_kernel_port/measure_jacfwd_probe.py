"""Measure forward geometry differentiation through one ported flux block."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import resource
import socket
import subprocess
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from nova.biot.polygon import horizontal_edges, traced_pack_section
from nova.biot.polygonanalytic import packed_analytic_greens
from nova.jax.config import configure_dtypes


HERE = Path(__file__).resolve().parent
RECEIPT = HERE / "jacfwd-probe-receipt.json"
BANKED_FAILURE_GIB_PER_TARGET = 12.2
TARGET_COUNT = 32

# A smooth section keeps the edge-topology mask fixed while both geometry
# parameters move every vertex through the production packing arithmetic.
SECTION = np.array(
    [
        [1.42, -1.16],
        [1.55, -1.19],
        [1.60, -1.07],
        [1.50, -1.015],
        [1.405, -1.06],
    ],
    dtype=np.float64,
)


def _targets(count: int) -> tuple[np.ndarray, np.ndarray]:
    """Return a deterministic target batch spanning near and far evaluation."""
    phase = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    radius = 1.85 + 0.42 * np.cos(phase) + 0.05 * np.cos(3.0 * phase)
    height = 0.12 + 0.78 * np.sin(phase)
    return radius.astype(np.float64), height.astype(np.float64)


def _host_peak_bytes() -> int:
    """Return this process's resident high-water mark in bytes on Linux."""
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _device_memory(device) -> dict[str, int | None]:
    """Return allocator counters without inventing values for absent backends."""
    stats = device.memory_stats() or {}
    return {
        "bytes_in_use": (
            int(stats["bytes_in_use"]) if "bytes_in_use" in stats else None
        ),
        "bytes_limit": int(stats["bytes_limit"]) if "bytes_limit" in stats else None,
        "peak_bytes_in_use": (
            int(stats["peak_bytes_in_use"]) if "peak_bytes_in_use" in stats else None
        ),
    }


def _source_stamp() -> dict[str, str]:
    """Return the measured source revision and checkout."""
    root = HERE.parents[1]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {"commit": commit, "worktree": str(root)}


def _flux_block(namespace, target_r, target_z, parameters):
    """Evaluate the ported exact-section flux row after a rigid source shift."""
    vertices = namespace.asarray(SECTION, dtype=parameters.dtype) + parameters[None, :]
    edge, weight, norm = traced_pack_section(
        namespace, vertices, horizontal_edges(SECTION)
    )
    return packed_analytic_greens(
        namespace,
        namespace.asarray(target_r, dtype=parameters.dtype),
        namespace.asarray(target_z, dtype=parameters.dtype),
        edge[..., None],
        weight[:, None],
        norm,
    )[0]


def measure(output: Path, target_count: int, execution_mode: str) -> None:
    """Compile, execute, and bank one H200 forward-mode differentiation probe."""
    configure_dtypes()
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"expected GPU backend, got {jax.default_backend()!r}")
    if os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", "").lower() != "false":
        raise RuntimeError(
            "set XLA_PYTHON_CLIENT_PREALLOCATE=false for memory evidence"
        )

    target_r, target_z = _targets(target_count)
    origin = jnp.zeros(2, dtype=jnp.float64)

    def shifted(parameters):
        return _flux_block(jnp, target_r, target_z, parameters)

    device = jax.devices()[0]
    memory_before = _device_memory(device)
    host_before = _host_peak_bytes()
    if execution_mode == "compiled":
        differentiated = jax.jit(jax.jacfwd(shifted))
        compile_started = perf_counter()
        executable = differentiated.lower(origin).compile()
        compile_seconds = perf_counter() - compile_started
        host_after_compile = _host_peak_bytes()

        first_started = perf_counter()
        first = executable(origin)
        jax.block_until_ready(first)
        first_execution_seconds = perf_counter() - first_started

        warm_started = perf_counter()
        warm = executable(origin)
        jax.block_until_ready(warm)
        warm_execution_seconds = perf_counter() - warm_started
    else:
        compile_seconds = 0.0
        host_after_compile = host_before
        differentiated = jax.jacfwd(shifted)
        first_started = perf_counter()
        first = differentiated(origin)
        jax.block_until_ready(first)
        first_execution_seconds = perf_counter() - first_started

        warm_started = perf_counter()
        warm = differentiated(origin)
        jax.block_until_ready(warm)
        warm_execution_seconds = perf_counter() - warm_started
    device_memory = _device_memory(device)
    host_peak = _host_peak_bytes()

    primal = np.asarray(shifted(origin))
    reference = np.asarray(
        _flux_block(np, target_r, target_z, np.zeros(2, dtype=np.float64))
    )
    difference = np.abs(primal - reference)
    scale = np.maximum(np.abs(reference), np.finfo(np.float64).tiny)
    jacobian = np.asarray(warm)
    if jacobian.shape != (target_count, 2) or not np.isfinite(jacobian).all():
        raise RuntimeError(
            f"invalid jacfwd result: shape={jacobian.shape}, "
            f"finite={np.isfinite(jacobian).all()}"
        )
    device_peak = device_memory["peak_bytes_in_use"]
    if device_peak is None:
        raise RuntimeError("the GPU allocator did not report peak_bytes_in_use")

    gib = float(1 << 30)
    cold_seconds = compile_seconds + first_execution_seconds
    measured_gib_per_target = device_peak / gib / target_count
    receipt = {
        "baseline": {
            "banked_failure_gib_per_target": BANKED_FAILURE_GIB_PER_TARGET,
            "composition": "flux-antiderivative differentiation",
            "memory_domain": "host peak RSS",
            "max_rss_kib": 12_235_772,
            "outcome": "terminated before producing the one-target derivative",
            "scheduler_job_id": "1249084",
        },
        "configuration": {
            "backend": jax.default_backend(),
            "device_kind": device.device_kind,
            "dtype": "float64",
            "execution_mode": execution_mode,
            "gradient_parameters": [
                "rigid_source_radial_shift",
                "rigid_source_vertical_shift",
            ],
            "kernel": "exact finite-section uniform-flux row",
            "mapping": "jacfwd",
            "ported_function": "nova.biot.polygonanalytic.packed_analytic_greens",
            "preallocate": False,
            "source_edge_count": int(len(SECTION)),
            "target_count": target_count,
            "targets_batched": True,
        },
        "environment": {
            "hostname": socket.gethostname(),
            "jax_version": jax.__version__,
            "platform": platform.platform(),
        },
        "measurement": {
            "cold_jacfwd_seconds": cold_seconds,
            "cold_jacfwd_seconds_per_target": cold_seconds / target_count,
            "compile_seconds": compile_seconds,
            "device_allocator_before": memory_before,
            "device_allocator_peak_bytes": device_peak,
            "device_allocator_peak_gib": device_peak / gib,
            "device_allocator_peak_gib_per_target": measured_gib_per_target,
            "device_memory_ratio_to_banked_failure_per_target": (
                measured_gib_per_target / BANKED_FAILURE_GIB_PER_TARGET
            ),
            "first_execution_seconds": first_execution_seconds,
            "host_peak_bytes": host_peak,
            "host_peak_gib": host_peak / gib,
            "host_peak_gib_per_target": host_peak / gib / target_count,
            "host_peak_before_compile_bytes": host_before,
            "host_peak_after_compile_bytes": host_after_compile,
            "host_peak_ratio_to_banked_failure_per_target": (
                host_peak / gib / target_count / BANKED_FAILURE_GIB_PER_TARGET
            ),
            "jacobian_shape": list(jacobian.shape),
            "warm_jacfwd_seconds": warm_execution_seconds,
            "warm_jacfwd_seconds_per_target": warm_execution_seconds / target_count,
        },
        "primal_check": {
            "all_finite": bool(np.isfinite(primal).all()),
            "max_absolute_difference_from_numpy_namespace": float(np.max(difference)),
            "max_relative_difference_from_numpy_namespace": float(
                np.max(difference / scale)
            ),
        },
        "scope": {
            "adoption_decision": "coil-geometry-inversion",
            "commitment": "measure-only",
        },
        "source": _source_stamp(),
        "verdict": {
            "classification": (
                "compiled-jacfwd-hold"
                if execution_mode == "eager"
                else "compiled-jacfwd-measured"
            ),
            "mechanism": (
                "forward mode propagates two geometry tangents through one batched "
                "plain-kernel trace; eager execution measures its device footprint, "
                "but the compiled form exceeded twenty minutes and 16 GiB host RSS "
                "before producing a derivative, so compilation remains the blocker"
            ),
            "qualification": (
                "this is one exact finite-section flux row and two rigid-shift "
                "parameters; extension and adoption are owned by "
                "coil-geometry-inversion"
            ),
        },
    }
    if execution_mode == "eager":
        receipt["compiled_attempt"] = {
            "device_peak_bytes": None,
            "elapsed_seconds": 20.0 * 60.0 + 26.0,
            "elapsed_seconds_per_target": (20.0 * 60.0 + 26.0) / target_count,
            "host_peak_bytes": 17_185_644 * 1024,
            "host_peak_gib": 17_185_644 * 1024 / gib,
            "host_peak_gib_per_target": 17_185_644 * 1024 / gib / target_count,
            "outcome": "TIMEOUT before a derivative completed",
            "scheduler_job_id": "1253223",
        }
    output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        "JACFWD_PROBE "
        f"targets={target_count} device_peak_gib={device_peak / gib:.9g} "
        f"device_gib_per_target={measured_gib_per_target:.9g} "
        f"host_peak_gib={host_peak / gib:.9g} "
        f"cold_s_per_target={cold_seconds / target_count:.9g} "
        f"warm_s_per_target={warm_execution_seconds / target_count:.9g}",
        flush=True,
    )


def check(path: Path) -> None:
    """Validate that a receipt carries the complete measure-only evidence."""
    receipt = json.loads(path.read_text(encoding="utf-8"))
    configuration = receipt["configuration"]
    measurement = receipt["measurement"]
    if configuration["target_count"] < 2 or not configuration["targets_batched"]:
        raise RuntimeError("the receipt does not measure batched targets")
    required_positive = (
        "cold_jacfwd_seconds_per_target",
        "device_allocator_peak_gib_per_target",
        "host_peak_gib_per_target",
        "warm_jacfwd_seconds_per_target",
    )
    if any(
        not np.isfinite(measurement[key]) or measurement[key] <= 0.0
        for key in required_positive
    ):
        raise RuntimeError("the receipt has missing or non-positive headline metrics")
    if receipt["baseline"]["banked_failure_gib_per_target"] != 12.2:
        raise RuntimeError("the banked comparison is not the 12.2 GiB failure")
    if receipt["scope"] != {
        "adoption_decision": "coil-geometry-inversion",
        "commitment": "measure-only",
    }:
        raise RuntimeError("the receipt crosses the locked gradient scope")
    if not receipt["primal_check"]["all_finite"]:
        raise RuntimeError("the ported primal evaluation is not finite")
    print("JACFWD_PROBE_CHECK passed", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    measure_parser = subparsers.add_parser("measure")
    measure_parser.add_argument("--output", type=Path, default=RECEIPT)
    measure_parser.add_argument("--targets", type=int, default=TARGET_COUNT)
    measure_parser.add_argument(
        "--execution-mode", choices=("compiled", "eager"), default="compiled"
    )
    check_parser = subparsers.add_parser("check")
    check_parser.add_argument("--receipt", type=Path, default=RECEIPT)
    args = parser.parse_args()
    if args.command == "measure":
        if args.targets < 2:
            raise ValueError("targets must describe a batch of at least two")
        measure(args.output, args.targets, args.execution_mode)
    else:
        check(args.receipt)


if __name__ == "__main__":
    main()
