"""Measure the normalized hex-null precision contract on one JAX device."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import socket
import subprocess
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from nova.biot.null import Null2D
from nova.geometry.hexstencil import hex_stencil
from nova.jax.config import Precision, configure_dtypes, resolve_precision


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs/figures/jax-dissolution/select_precision_contract.json"
REPEATS = 7


def _automatic_fit_dtype() -> np.dtype:
    """Return the dtype selected by the current null-fit automatic policy."""
    precision = resolve_precision(Precision.AUTOMATIC, Precision.DOUBLE)
    return np.dtype(np.float32 if precision is Precision.SINGLE else np.float64)


def _print_field_null_precision(*working_dtypes: Any) -> None:
    """Print the JAX x64 capability and every dtype used by a null fit."""
    configure_dtypes()
    for dtype in dict.fromkeys(np.dtype(value).name for value in working_dtypes):
        print(
            "FIELD_NULL_PRECISION "
            f"x64_enabled={bool(jax.config.x64_enabled)} working_dtype={dtype}",
            flush=True,
        )


def _strict(value: Any) -> Any:
    """Convert values into strict JSON data."""
    if isinstance(value, dict):
        return {str(key): _strict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict(item) for item in value]
    if isinstance(value, np.ndarray):
        return _strict(value.tolist())
    if isinstance(value, np.generic):
        return _strict(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write(path: Path, payload: dict[str, Any]) -> None:
    """Write an indented strict JSON record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_strict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _revision() -> str:
    """Return the measured source revision or its frozen-snapshot override."""
    if supplied := os.environ.get("NOVA_SELECT_PRECISION_REVISION"):
        return supplied
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _source_hashes() -> dict[str, str]:
    """Hash the benchmark and production paths used by the measurement."""
    paths = (
        "benchmarks/select_precision_contract.py",
        "nova/biot/null.py",
        "nova/geometry/select.py",
        "nova/jax/config.py",
    )
    return {
        path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in paths
    }


def _geometry() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return an ITER-scale grid and two independent analytic null fields."""
    radial = np.linspace(6.14, 6.26, 11, dtype=np.float64)
    vertical = np.linspace(-3.745, -3.655, 11, dtype=np.float64)
    rr, zz = np.meshgrid(radial, vertical, indexing="ij")
    coordinate = np.column_stack((rr.ravel(), zz.ravel()))
    stencil = hex_stencil(rr.shape)
    truth = np.array([6.2031, -3.6982], dtype=np.float64)
    delta_r = rr - truth[0]
    delta_z = zz - truth[1]
    extremum = 1.2 * delta_r**2 + 0.8 * delta_z**2 + 0.15 * delta_r * delta_z
    saddle = 1.1 * delta_r**2 - 0.9 * delta_z**2 + 0.2 * delta_r * delta_z
    fields = np.stack((extremum.ravel(), saddle.ravel()))
    return coordinate, stencil, fields, truth


def _synchronize(value):
    """Block until every array leaf is ready."""
    for leaf in jax.tree.leaves(value):
        block = getattr(leaf, "block_until_ready", None)
        if block is not None:
            block()
    return value


def _selected(result, category, truth):
    """Return the finite category row nearest the independent truth."""
    rows = np.asarray(result[category])
    finite = np.isfinite(rows[:, 0])
    rows = rows[finite]
    if not len(rows):
        raise AssertionError("analytic null was not detected")
    return rows[np.argmin(np.linalg.norm(rows[:, :2] - truth, axis=1))]


def _measure_precision(coordinate, stencil, fields, truth, precision):
    """Measure both analytic classes at one per-instance precision."""
    locator = Null2D.from_coordinates(
        coordinate,
        stencil,
        maxsize=4,
        precision=precision,
    )
    rows = []
    compile_ms = 0.0
    steady_ms = 0.0
    for field_index, (category, expected_kind) in enumerate(((0, -1.0), (1, 0.0))):
        argument = jnp.asarray(fields[field_index], dtype=locator.fit_dtype)
        start = time.perf_counter()
        executable = jax.jit(locator.__call__).lower(argument).compile()
        compile_ms += 1e3 * (time.perf_counter() - start)
        _synchronize(executable(argument))
        samples = []
        for _ in range(REPEATS):
            start = time.perf_counter()
            result = _synchronize(executable(argument))
            samples.append(1e3 * (time.perf_counter() - start))
        steady_ms += min(samples)
        selected = _selected(result, category, truth)
        spacing = np.array([0.012, 0.009])
        error = selected[:2] - truth
        rows.append(
            {
                "field": "extremum" if category == 0 else "saddle",
                "expected_kind": expected_kind,
                "observed_kind": float(selected[3]),
                "coordinate": selected[:2],
                "truth": truth,
                "coordinate_error_metres": error,
                "coordinate_error_cells": error / spacing,
                "location_error_metres": float(np.linalg.norm(error)),
                "location_error_cells_max_axis": float(np.max(np.abs(error) / spacing)),
            }
        )
    return {
        "precision": precision.value,
        "fit_dtype": str(locator.local_coordinate_stencil.dtype),
        "metadata_dtype": str(locator.physical_origin.dtype),
        "compile_ms_two_fields": compile_ms,
        "steady_ms_two_fields": steady_ms,
        "rows": rows,
        "classification_errors": sum(
            row["observed_kind"] != row["expected_kind"] for row in rows
        ),
        "worst_location_error_cells": max(
            row["location_error_cells_max_axis"] for row in rows
        ),
    }


def measure(platform_name: str) -> dict[str, Any]:
    """Capture automatic fp32 and explicit fp64 variants on one device."""
    configure_dtypes()
    devices = jax.devices(platform_name)
    if not devices:
        raise RuntimeError(f"no {platform_name} device is available")
    coordinate, stencil, fields, truth = _geometry()
    with jax.default_device(devices[0]):
        precision = [
            _measure_precision(coordinate, stencil, fields, truth, Precision.AUTOMATIC),
            _measure_precision(coordinate, stencil, fields, truth, Precision.DOUBLE),
        ]
    return _strict(
        {
            "schema": "nova.select-precision-contract",
            "captured_at": datetime.now(UTC).isoformat(),
            "revision": _revision(),
            "source_hashes": _source_hashes(),
            "platform": platform_name,
            "environment": {
                "host": socket.gethostname(),
                "system": platform.platform(),
                "python": platform.python_version(),
                "jax": jax.__version__,
                "device": str(devices[0]),
                "jax_enable_x64": bool(jax.config.x64_enabled),
                "jax_explicit_x64_dtypes": str(jax.config.jax_explicit_x64_dtypes),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
            },
            "precision": precision,
        }
    )


def assemble(cpu_path: Path, gpu_path: Path) -> dict[str, Any]:
    """Combine immutable CPU and GPU captures and evaluate the contract gates."""
    cpu = json.loads(cpu_path.read_text(encoding="utf-8"))
    gpu = json.loads(gpu_path.read_text(encoding="utf-8"))
    if cpu["revision"] != gpu["revision"]:
        raise RuntimeError("CPU and GPU captures used different revisions")
    if cpu["source_hashes"] != gpu["source_hashes"]:
        raise RuntimeError("CPU and GPU captures used different source bytes")
    rows = cpu["precision"] + gpu["precision"]
    automatic = [row for row in rows if row["precision"] == "auto"]
    parity = []
    for field_index in range(2):
        cpu_row = cpu["precision"][0]["rows"][field_index]
        gpu_row = gpu["precision"][0]["rows"][field_index]
        parity.append(
            {
                "field": cpu_row["field"],
                "coordinate_max_abs_metres": float(
                    np.max(
                        np.abs(
                            np.asarray(cpu_row["coordinate"])
                            - np.asarray(gpu_row["coordinate"])
                        )
                    )
                ),
                "kind_equal": cpu_row["observed_kind"] == gpu_row["observed_kind"],
            }
        )
    gate = {
        "automatic_classification_errors": sum(
            row["classification_errors"] for row in automatic
        ),
        "automatic_worst_location_error_cells": max(
            row["worst_location_error_cells"] for row in automatic
        ),
        "normalized_exact_limit_cells": 0.02,
        "cpu_gpu_coordinate_limit_metres": 1e-8,
        "pass": (
            sum(row["classification_errors"] for row in automatic) == 0
            and max(row["worst_location_error_cells"] for row in automatic) <= 0.02
            and all(row["kind_equal"] for row in parity)
            and max(row["coordinate_max_abs_metres"] for row in parity) < 1e-8
        ),
    }
    return {
        "schema": "nova.select-precision-contract",
        "assembled_at": datetime.now(UTC).isoformat(),
        "revision": cpu["revision"],
        "source_hashes": cpu["source_hashes"],
        "contract": {
            "automatic": "normalized local fp32 fit with fp64 physical metadata",
            "explicit_double": "normalized local fp64 fit with fp64 physical metadata",
            "absolute_fp32_geometry": "rejected before normalization",
        },
        "gate": gate,
        "cpu_gpu_parity": parity,
        "captures": {"cpu": cpu, "gpu": gpu},
    }


def main() -> None:
    """Measure one platform or assemble two captured records."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    measure_parser = subparsers.add_parser("measure")
    measure_parser.add_argument("--platform", choices=("cpu", "gpu"), required=True)
    measure_parser.add_argument("--output", type=Path, required=True)
    assemble_parser = subparsers.add_parser("assemble")
    assemble_parser.add_argument("--cpu", type=Path, required=True)
    assemble_parser.add_argument("--gpu", type=Path, required=True)
    assemble_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    subparsers.add_parser("precision")
    args = parser.parse_args()
    if args.command == "measure":
        payload = measure(args.platform)
        _print_field_null_precision(*(row["fit_dtype"] for row in payload["precision"]))
    elif args.command == "assemble":
        payload = assemble(args.cpu, args.gpu)
    else:
        _print_field_null_precision(_automatic_fit_dtype(), np.float64)
        return
    _write(args.output, payload)


if __name__ == "__main__":
    main()
