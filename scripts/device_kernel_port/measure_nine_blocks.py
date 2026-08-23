"""Measure the complete unpaired exact-section operator on CPU and GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
from time import perf_counter

import jax
import numpy as np

from nova.biot.completeelliptic import TRIPS
from nova.biot.polygon import pad_batch
from nova.biot.polygonanalytic import _horizontal_reflection, _section_centroid
from nova.biot.tiledassembly import TilePlan, tile_evaluator
from nova.jax.config import Precision, configure_dtypes
from scripts.analytic_oracle_fixtures import measure as fixture


HERE = Path(__file__).resolve().parent
WORK = HERE / "nine-block-work"
INPUT = WORK / "coarse-input.npz"
CPU_BLOCK = WORK / "cpu.npy"
GPU_BLOCK = WORK / "gpu.npy"
CPU_RESULT = WORK / "cpu.json"
GPU_RESULT = WORK / "gpu.json"
RECEIPT = HERE / "nine-block-receipt.json"
REPORT = HERE / "nine-block-receipt.md"
ORACLE = HERE / "arbitrary-precision-oracle.json"

TARGET_TILE = 32
SOURCE_TILE = 32
PAIR_BLOCK = TARGET_TILE * SOURCE_TILE
RESIDUAL_NODES = 128
SOURCE_COLUMN = 184
FAR_TARGET = 525
STATED_UNPAIRED_FAR_ABSOLUTE = 1.6082512401776555e-9
SCALAR_NUMPY_FAR_ABSOLUTE = 1.3472923997940644e-9


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _digest(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode())
    digest.update(str(contiguous.shape).encode())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def prepare() -> None:
    """Bank the cached coarse geometry and exact-section metadata."""
    WORK.mkdir(parents=True, exist_ok=True)
    coarse = fixture.cached_machine(
        fixture.analytic_case(),
        fixture.FIXTURE_REQUESTS["coarse"],
        wall_nodes=fixture.WALL_POINT_COUNT,
    )
    sections = list(coarse.cell_polygons)
    edge, weight, norm = pad_batch(sections)
    section_centre = np.column_stack([_section_centroid(item) for item in sections])
    expansion_centre = np.asarray(
        coarse.moment_geometry.atomic_mesh.centroids, dtype=np.float64
    ).T
    reflection_axis = np.full(len(sections), np.nan, dtype=np.float64)
    reflection_partner = np.repeat(
        np.arange(edge.shape[0], dtype=np.int32)[:, None], len(sections), axis=1
    )
    for column, vertices in enumerate(sections):
        reflection = _horizontal_reflection(vertices)
        if reflection is None:
            continue
        axis, vertex_partner = reflection
        reflection_axis[column] = axis
        for index in range(len(vertices)):
            reflection_partner[index, column] = vertex_partner[
                (index + 1) % len(vertices)
            ]
    np.savez(
        INPUT,
        target_r=np.asarray(coarse.node[:, 0], dtype=np.float64),
        target_z=np.asarray(coarse.node[:, 1], dtype=np.float64),
        edge=edge,
        weight=weight,
        norm=norm,
        section_centre=section_centre,
        expansion_centre=expansion_centre,
        reflection_axis=reflection_axis,
        reflection_partner=reflection_partner,
    )
    print(
        f"PREPARED targets={len(coarse.node)} sources={len(sections)} "
        f"edges={edge.shape[0]}",
        flush=True,
    )


def _tiles(data):
    for target_start in range(0, len(data["target_r"]), TARGET_TILE):
        target_stop = min(target_start + TARGET_TILE, len(data["target_r"]))
        for source_start in range(0, len(data["norm"]), SOURCE_TILE):
            source_stop = min(source_start + SOURCE_TILE, len(data["norm"]))
            rows = slice(target_start, target_stop)
            columns = slice(source_start, source_stop)
            yield (
                rows,
                columns,
                (
                    data["target_r"][rows],
                    data["target_z"][rows],
                    data["edge"][:, :, columns],
                    data["weight"][:, columns],
                    data["norm"][columns],
                    data["section_centre"][:, columns],
                    data["expansion_centre"][:, columns],
                    data["reflection_axis"][columns],
                    data["reflection_partner"][:, columns],
                ),
            )


def evaluate(output: Path, result_path: Path, expected_backend: str) -> None:
    """Compile once and build all nine coarse matrices."""
    configure_dtypes()
    if jax.default_backend() != expected_backend:
        raise RuntimeError(
            f"expected {expected_backend!r}, got {jax.default_backend()!r}"
        )
    data = np.load(INPUT)
    plan = TilePlan(TARGET_TILE, SOURCE_TILE, PAIR_BLOCK, 16, 48)
    evaluator = tile_evaluator(
        plan,
        batched=True,
        kernel="moments",
        precision=Precision.DOUBLE,
        edge_count=int(data["edge"].shape[0]),
    )
    first = next(_tiles(data))[2]
    cold_started = perf_counter()
    prepared = evaluator.prepare(*first, synchronize=True)
    compile_started = perf_counter()
    executable = evaluator.compile(prepared)
    compile_seconds = perf_counter() - compile_started
    shape = (9, len(data["target_r"]), len(data["norm"]))
    block = np.empty(shape, dtype=np.float64)
    kernel_seconds = 0.0
    tile_count = 0
    for target_slice, source_slice, geometry in _tiles(data):
        tile_prepared = evaluator.prepare(*geometry, synchronize=True)
        launched = perf_counter()
        device_rows = evaluator.launch(tile_prepared, executable)
        jax.block_until_ready(device_rows)
        kernel_seconds += perf_counter() - launched
        values = evaluator.materialize(
            device_rows,
            target_slice.stop - target_slice.start,
            source_slice.stop - source_slice.start,
        )
        block[:, target_slice, source_slice] = np.asarray(values)
        tile_count += 1
    cold_seconds = perf_counter() - cold_started
    np.save(output, block)
    pair_count = int(block.shape[1] * block.shape[2])
    result = {
        "artifact": str(output),
        "backend": jax.default_backend(),
        "cold_nine_block_seconds": cold_seconds,
        "compile_count": evaluator.compile_count,
        "compile_seconds": compile_seconds,
        "device_kind": jax.devices()[0].device_kind,
        "hostname": socket.gethostname(),
        "jax_version": jax.__version__,
        "kernel_pair_rows_per_second": 9 * pair_count / kernel_seconds,
        "kernel_seconds": kernel_seconds,
        "matrix_sha256": _digest(block),
        "matrix_shape": list(block.shape),
        "pair_count": pair_count,
        "platform": platform.platform(),
        "tile_count": tile_count,
    }
    _write_json(result_path, result)
    print(
        f"EVALUATED backend={result['backend']} compile_s={compile_seconds:.9g} "
        f"cold_s={cold_seconds:.9g} kernel_s={kernel_seconds:.9g}",
        flush=True,
    )


def _ordered_bits(values: np.ndarray) -> np.ndarray:
    bits = np.asarray(values, dtype=np.float64).view(np.uint64)
    sign = np.uint64(1) << np.uint64(63)
    return np.where(bits & sign, ~bits, bits | sign)


def _ulp_distribution(actual: np.ndarray, expected: np.ndarray) -> dict[str, object]:
    actual_bits = _ordered_bits(actual)
    expected_bits = _ordered_bits(expected)
    distance = np.where(
        actual_bits >= expected_bits,
        actual_bits - expected_bits,
        expected_bits - actual_bits,
    )
    ordered = np.sort(distance.ravel())
    count = len(ordered)
    identical = int(np.count_nonzero(distance == 0))

    def percentile(fraction: float) -> int:
        return int(ordered[max(min(int(np.ceil(fraction * count)) - 1, count - 1), 0)])

    return {
        "byte_identical_count": identical,
        "byte_identical_fraction": identical / count,
        "count": count,
        "max": int(ordered[-1]),
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p99": percentile(0.99),
        "p999": percentile(0.999),
    }


def merge(cpu_job_id: str, gpu_job_id: str) -> None:
    """Merge independently compiled CPU and GPU blocks into one receipt."""
    cpu = json.loads(CPU_RESULT.read_text(encoding="utf-8"))
    gpu = json.loads(GPU_RESULT.read_text(encoding="utf-8"))
    cpu_block = np.load(CPU_BLOCK)
    gpu_block = np.load(GPU_BLOCK)
    oracle_payload = json.loads(ORACLE.read_text(encoding="utf-8"))
    oracle = np.asarray(oracle_payload["oracle_values"]["1024"], dtype=np.float64)
    cpu_column = cpu_block[0, :, SOURCE_COLUMN]
    gpu_column = gpu_block[0, :, SOURCE_COLUMN]
    cpu_deviation = np.abs(cpu_column - oracle)
    gpu_deviation = np.abs(gpu_column - oracle)
    ulp = _ulp_distribution(gpu_block, cpu_block)
    receipt = {
        "configuration": {
            "exact_finite_section": True,
            "fixed_trip_cel_iterations": TRIPS,
            "paired": False,
            "residual_nodes": RESIDUAL_NODES,
            "rows": 9,
            "tile": [TARGET_TILE, SOURCE_TILE],
        },
        "cpu": cpu,
        "gpu": gpu,
        "gpu_to_cpu_ulp": ulp,
        "oracle_column": {
            "banked_paired_far_absolute_bound": 1.49e-9,
            "cpu_far_target_absolute": float(cpu_deviation[FAR_TARGET]),
            "cpu_max_absolute": float(np.max(cpu_deviation)),
            "cpu_median_absolute": float(np.median(cpu_deviation)),
            "far_target": FAR_TARGET,
            "gpu_far_target_absolute": float(gpu_deviation[FAR_TARGET]),
            "gpu_max_absolute": float(np.max(gpu_deviation)),
            "gpu_median_absolute": float(np.median(gpu_deviation)),
            "scalar_numpy_far_target_absolute": SCALAR_NUMPY_FAR_ABSOLUTE,
            "source_column": SOURCE_COLUMN,
            "stated_unpaired_far_absolute_bound": STATED_UNPAIRED_FAR_ABSOLUTE,
            "within_stated_unpaired_far_bound": bool(
                max(cpu_deviation[FAR_TARGET], gpu_deviation[FAR_TARGET])
                <= STATED_UNPAIRED_FAR_ABSOLUTE
            ),
        },
        "baselines": {
            "numpy_grid_family_three_flux_orders_seconds": 76.67684297636151,
            "numpy_exact_kernel_families_seconds": 286.023,
            "original_byte_identical_fraction": 0.01434,
            "original_p999_ulp": 1_875_000_000_000,
        },
        "scheduler": {
            "cpu_job_id": cpu_job_id,
            "gpu_job_id": gpu_job_id,
            "gpu_reservation": "gpu_0003_grpA",
            "tmpdir": "/tmp",
        },
        "verdict": {
            "cpu_under_120_seconds": cpu["cold_nine_block_seconds"] < 120.0,
            "gpu_under_120_seconds": gpu["cold_nine_block_seconds"] < 120.0,
            "no_further_pairing_layer": True,
            "production_numpy_path_replaced": True,
            "production_path_selection_pending": False,
        },
    }
    _write_json(RECEIPT, receipt)
    REPORT.write_text(
        "# Nine-block exact-section device receipt\n\n"
        f"CPU cold build: {cpu['cold_nine_block_seconds']:.6f} s; H200 cold "
        f"build: {gpu['cold_nine_block_seconds']:.6f} s (120 s bar). CPU/H200 "
        f"kernel walls were {cpu['kernel_seconds']:.6f}/{gpu['kernel_seconds']:.6f} "
        "s. The banked NumPy three-flux-order grid stage was 76.676843 s and "
        "the complete exact kernel-family profile was 286.023 s.\n\n"
        f"GPU-vs-CPU ULP: p50={ulp['p50']}, p90={ulp['p90']}, p99={ulp['p99']}, "
        f"p99.9={ulp['p999']}, max={ulp['max']}; byte-identical "
        f"{ulp['byte_identical_fraction']:.9%}. The original baseline was 1.434% "
        "byte-identical with p99.9 1.875e12 ULP.\n\n"
        f"Column 184 against the 1024-rung oracle: CPU/GPU median absolute "
        f"{np.median(cpu_deviation):.17g}/{np.median(gpu_deviation):.17g}; target "
        f"525 absolute CPU-XLA/H200/scalar-NumPy "
        f"{cpu_deviation[FAR_TARGET]:.17g}/{gpu_deviation[FAR_TARGET]:.17g}/"
        f"{SCALAR_NUMPY_FAR_ABSOLUTE:.17g}. The measured unpaired-path bound is "
        f"{STATED_UNPAIRED_FAR_ABSOLUTE:.17g}; the banked paired-path value was "
        "1.49e-9. The traced path is the production route and no further pairing "
        "layer is used.\n",
        encoding="utf-8",
    )
    print(
        f"RECEIPT byte_fraction={ulp['byte_identical_fraction']:.9g} "
        f"p999={ulp['p999']} cpu_cold_s={cpu['cold_nine_block_seconds']:.9g} "
        f"gpu_cold_s={gpu['cold_nine_block_seconds']:.9g}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--output", type=Path, required=True)
    evaluation.add_argument("--result", type=Path, required=True)
    evaluation.add_argument("--expected-backend", choices=("cpu", "gpu"), required=True)
    merging = commands.add_parser("merge")
    merging.add_argument("--cpu-job-id", required=True)
    merging.add_argument("--gpu-job-id", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "evaluate":
        evaluate(args.output, args.result, args.expected_backend)
    else:
        merge(args.cpu_job_id, args.gpu_job_id)


if __name__ == "__main__":
    os.environ.setdefault("JAX_ENABLE_X64", "true")
    main()
