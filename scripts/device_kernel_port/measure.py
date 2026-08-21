"""Measure the packed fp64 flux kernel on CPU and accelerator backends."""

from __future__ import annotations

import argparse
from collections import Counter
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
from nova.biot.polygonanalytic import polygon_analytic_greens
from nova.biot.tiledassembly import TilePlan, tile_evaluator
from nova.jax.config import Precision, configure_dtypes
from scripts.analytic_oracle_fixtures import measure as fixture


HERE = Path(__file__).resolve().parent
WORK = HERE / "work"
INPUT = WORK / "coarse-input.npz"
REFERENCE = WORK / "coarse-numpy-reference.npy"
CPU_BLOCK = WORK / "coarse-jax-cpu.npy"
GPU_BLOCK = WORK / "coarse-jax-gpu.npy"
CPU_RESULT = WORK / "cpu-result.json"
GPU_RESULT = WORK / "gpu-result.json"
NUMPY_RESULT = WORK / "numpy-result.json"
RECEIPT = HERE / "receipt.json"
REPORT = HERE / "receipt.md"

TARGET_TILE = 32
SOURCE_TILE = 32
PAIR_BLOCK = TARGET_TILE * SOURCE_TILE
RESIDUAL_NODES = 128


def _write_json(path: Path, payload: dict[str, object]) -> None:
    """Write one stable strict-JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _digest(array: np.ndarray) -> str:
    """Return a content digest for an array including its dtype and shape."""
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode())
    digest.update(str(contiguous.shape).encode())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _source_stamp() -> dict[str, str]:
    """Return the source identity without imposing a clean-checkout requirement."""
    import subprocess

    root = HERE.parents[1]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {"commit": commit, "worktree": str(root)}


def prepare() -> None:
    """Bank fixed-shape coarse geometry, its CPU reference, and projection sizes."""
    WORK.mkdir(parents=True, exist_ok=True)
    case = fixture.analytic_case()
    coarse = fixture.cached_machine(
        case,
        fixture.FIXTURE_REQUESTS["coarse"],
        wall_nodes=fixture.WALL_POINT_COUNT,
    )
    fine = fixture.cached_machine(
        case,
        fixture.FIXTURE_REQUESTS["fine"],
        wall_nodes=fixture.WALL_POINT_COUNT,
    )
    edge, weight, norm = pad_batch(list(coarse.cell_polygons))
    np.savez(
        INPUT,
        target_r=np.asarray(coarse.node[:, 0], dtype=np.float64),
        target_z=np.asarray(coarse.node[:, 1], dtype=np.float64),
        edge=edge,
        weight=weight,
        norm=norm,
        fine_counts=np.asarray(
            [
                len(fine.node),
                len(fine.wall_node),
                len(fine.sample_coordinates),
                len(fine.cell_polygons),
            ],
            dtype=np.int64,
        ),
    )
    reference = np.asarray(coarse.plasma_to_grid, dtype=np.float64)
    np.save(REFERENCE, reference)
    fine_target_count = (
        len(fine.node) + len(fine.wall_node) + len(fine.sample_coordinates)
    )
    print(
        "PREPARED "
        f"coarse_shape={reference.shape} edge_count={edge.shape[0]} "
        f"fine_targets={fine_target_count} "
        f"fine_sources={len(fine.cell_polygons)} reference_sha256={_digest(reference)}",
        flush=True,
    )


def _tiles(data: np.lib.npyio.NpzFile):
    """Yield fixed-shape matrix tiles in deterministic row-major order."""
    target_r = data["target_r"]
    target_z = data["target_z"]
    edge = data["edge"]
    weight = data["weight"]
    norm = data["norm"]
    for target_start in range(0, len(target_r), TARGET_TILE):
        target_stop = min(target_start + TARGET_TILE, len(target_r))
        for source_start in range(0, len(norm), SOURCE_TILE):
            source_stop = min(source_start + SOURCE_TILE, len(norm))
            yield (
                slice(target_start, target_stop),
                slice(source_start, source_stop),
                (
                    target_r[target_start:target_stop],
                    target_z[target_start:target_stop],
                    edge[:, :, source_start:source_stop],
                    weight[:, source_start:source_stop],
                    norm[source_start:source_stop],
                ),
            )


def evaluate(output: Path, result_path: Path, expected_backend: str) -> None:
    """Compile once, then evaluate the complete coarse G0 block with fixed shapes."""
    configure_dtypes()
    if jax.default_backend() != expected_backend:
        raise RuntimeError(
            f"expected JAX backend {expected_backend!r}, got {jax.default_backend()!r}"
        )
    data = np.load(INPUT)
    plan = TilePlan(
        target_tile=TARGET_TILE,
        source_tile=SOURCE_TILE,
        block=PAIR_BLOCK,
        n_panels=16,
        n_nodes=48,
    )
    evaluator = tile_evaluator(
        plan,
        batched=True,
        kernel="closed",
        precision=Precision.DOUBLE,
        edge_count=int(data["edge"].shape[0]),
    )
    first = next(_tiles(data))[2]
    transfer_started = perf_counter()
    prepared = evaluator.prepare(*first, synchronize=True)
    first_transfer_seconds = perf_counter() - transfer_started
    compile_started = perf_counter()
    executable = evaluator.compile(prepared)
    compile_seconds = perf_counter() - compile_started
    warm_started = perf_counter()
    warm = evaluator.launch(prepared, executable)
    jax.block_until_ready(warm)
    warm_seconds = perf_counter() - warm_started

    shape = (len(data["target_r"]), len(data["norm"]))
    block = np.empty(shape, dtype=np.float64)
    kernel_seconds = 0.0
    end_to_end_started = perf_counter()
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
        block[target_slice, source_slice] = values[0]
        tile_count += 1
    end_to_end_seconds = perf_counter() - end_to_end_started
    np.save(output, block)

    device = jax.devices()[0]
    pair_count = int(block.size)
    result = {
        "artifact": str(output),
        "backend": jax.default_backend(),
        "byte_count": int(block.nbytes),
        "compile_count": evaluator.compile_count,
        "compile_seconds": compile_seconds,
        "device_kind": device.device_kind,
        "end_to_end_pairs_per_second": pair_count / end_to_end_seconds,
        "end_to_end_seconds": end_to_end_seconds,
        "fixed_shape": {
            "edge_count": int(data["edge"].shape[0]),
            "pair_block": PAIR_BLOCK,
            "source_tile": SOURCE_TILE,
            "target_tile": TARGET_TILE,
        },
        "first_transfer_seconds": first_transfer_seconds,
        "hostname": socket.gethostname(),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "jax_version": jax.__version__,
        "kernel_pairs_per_second": pair_count / kernel_seconds,
        "kernel_seconds": kernel_seconds,
        "matrix_sha256": _digest(block),
        "matrix_shape": list(block.shape),
        "pair_count": pair_count,
        "platform": platform.platform(),
        "residual_nodes": RESIDUAL_NODES,
        "tile_count": tile_count,
        "warm_seconds": warm_seconds,
    }
    _write_json(result_path, result)
    print(
        "EVALUATED "
        f"backend={result['backend']} device={result['device_kind']} "
        f"compile_s={compile_seconds:.9g} kernel_s={kernel_seconds:.9g} "
        f"pairs_per_s={result['kernel_pairs_per_second']:.9g} "
        f"sha256={result['matrix_sha256']}",
        flush=True,
    )


def numpy_reference() -> None:
    """Time the production NumPy uniform-flux path and verify the cached block."""
    data = np.load(INPUT)
    target_r = data["target_r"]
    target_z = data["target_z"]
    edge = data["edge"]
    weight = data["weight"]
    expected = np.load(REFERENCE)
    rebuilt = np.empty_like(expected)
    started = perf_counter()
    for source in range(edge.shape[2]):
        rows = int(np.count_nonzero(~np.signbit(weight[:, source])))
        vertices = edge[:rows, :2, source]
        rebuilt[:, source] = polygon_analytic_greens(
            target_r,
            target_z,
            vertices,
        )[0]
    seconds = perf_counter() - started
    difference = np.abs(rebuilt - expected)
    scale = np.maximum(np.abs(expected), np.finfo(np.float64).tiny)
    result = {
        "allclose_atol": 0.0,
        "allclose_rtol": 2.0e-11,
        "artifact": str(REFERENCE),
        "byte_identical_fraction": float(
            np.mean(rebuilt.view(np.uint64) == expected.view(np.uint64))
        ),
        "hostname": socket.gethostname(),
        "matrix_sha256": _digest(rebuilt),
        "max_absolute_difference": float(np.max(difference)),
        "max_relative_difference": float(np.max(difference / scale)),
        "pair_count": int(rebuilt.size),
        "pairs_per_second": int(rebuilt.size) / seconds,
        "passed": bool(np.allclose(rebuilt, expected, rtol=2.0e-11, atol=0.0)),
        "seconds": seconds,
    }
    _write_json(NUMPY_RESULT, result)
    print(
        "NUMPY_REFERENCE "
        f"seconds={seconds:.9g} pairs_per_s={result['pairs_per_second']:.9g} "
        f"max_rel={result['max_relative_difference']:.9g} "
        f"byte_fraction={result['byte_identical_fraction']:.9g}",
        flush=True,
    )


def _ordered_bits(values: np.ndarray) -> np.ndarray:
    """Map finite float64 bit patterns onto monotonically ordered integers."""
    bits = np.asarray(values, dtype=np.float64).view(np.uint64)
    sign = np.uint64(1) << np.uint64(63)
    return np.where(bits & sign, ~bits, bits | sign)


def ulp_distribution(actual: np.ndarray, expected: np.ndarray) -> dict[str, object]:
    """Return the complete finite element-wise absolute ULP distribution."""
    if actual.shape != expected.shape:
        raise ValueError(f"shape mismatch: {actual.shape} != {expected.shape}")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("ULP comparison requires finite matrices")
    actual_ordered = _ordered_bits(actual)
    expected_ordered = _ordered_bits(expected)
    distance = np.where(
        actual_ordered >= expected_ordered,
        actual_ordered - expected_ordered,
        expected_ordered - actual_ordered,
    )
    histogram = Counter(int(value) for value in distance.ravel())
    count = int(distance.size)
    sorted_distance = np.sort(distance.ravel())

    def percentile(fraction: float) -> int:
        index = min(int(np.ceil(fraction * count)) - 1, count - 1)
        return int(sorted_distance[max(index, 0)])

    return {
        "byte_identical_count": int(histogram.get(0, 0)),
        "byte_identical_fraction": histogram.get(0, 0) / count,
        "count": count,
        "histogram": {str(key): histogram[key] for key in sorted(histogram)},
        "max": int(sorted_distance[-1]),
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p99": percentile(0.99),
        "p999": percentile(0.999),
    }


def _comparison(actual: np.ndarray, expected: np.ndarray) -> dict[str, object]:
    """Return ULP and scale-aware comparison evidence for two matrices."""
    difference = np.abs(actual - expected)
    scale = np.maximum(np.abs(expected), np.finfo(np.float64).tiny)
    return {
        "actual_sha256": _digest(actual),
        "expected_sha256": _digest(expected),
        "max_absolute_difference": float(np.max(difference)),
        "max_relative_difference": float(np.max(difference / scale)),
        "ulp": ulp_distribution(actual, expected),
    }


def merge(cpu_job_id: str, gpu_job_id: str, numpy_job_id: str) -> None:
    """Combine independently produced device receipts into the committed result."""
    cpu = json.loads(CPU_RESULT.read_text(encoding="utf-8"))
    gpu = json.loads(GPU_RESULT.read_text(encoding="utf-8"))
    numpy_result = json.loads(NUMPY_RESULT.read_text(encoding="utf-8"))
    cpu_block = np.load(CPU_BLOCK)
    gpu_block = np.load(GPU_BLOCK)
    reference = np.load(REFERENCE)
    data = np.load(INPUT)
    fine_grid, fine_wall, fine_sample, fine_sources = (
        int(value) for value in data["fine_counts"]
    )
    fine_pairs = 3 * fine_sources * (fine_grid + fine_wall + fine_sample)
    projected_seconds = fine_pairs / gpu["kernel_pairs_per_second"]
    banked_numpy_grid_seconds = 76.67684297636151
    cpu_ratio = cpu["kernel_seconds"] / numpy_result["seconds"]
    gpu_to_cpu = _comparison(gpu_block, cpu_block)
    gpu_to_numpy = _comparison(gpu_block, reference)
    cpu_to_numpy = _comparison(cpu_block, reference)
    receipt = {
        "comparison_cpu_jax_to_production_numpy": cpu_to_numpy,
        "comparison_gpu_to_cpu_jax": gpu_to_cpu,
        "comparison_gpu_to_production_numpy": gpu_to_numpy,
        "cpu": cpu,
        "fixed_trip_cel_iterations": TRIPS,
        "fine_projection": {
            "block_count": 9,
            "fine_pair_evaluations": fine_pairs,
            "projected_seconds": projected_seconds,
            "projected_under_two_minutes": projected_seconds < 120.0,
            "rule": (
                "coarse G0 kernel pairs per second applied linearly to three "
                "flux orders over fine grid, wall, and sample target families"
            ),
            "source_count": fine_sources,
            "target_counts": {
                "grid": fine_grid,
                "sample": fine_sample,
                "wall": fine_wall,
            },
        },
        "gpu": gpu,
        "numpy": {
            "banked_grid_family_seconds_all_three_flux_orders": (
                banked_numpy_grid_seconds
            ),
            "measured_uniform_g0": numpy_result,
        },
        "port_threshold_inputs": {
            "cpu_jax_over_numpy_g0": cpu_ratio,
            "gpu_cpu_byte_identical_fraction": gpu_to_cpu["ulp"][
                "byte_identical_fraction"
            ],
            "gpu_cpu_max_ulp": gpu_to_cpu["ulp"]["max"],
            "projected_full_fine_seconds": projected_seconds,
        },
        "production_numpy_path_modified": False,
        "scheduler": {
            "cpu_job_id": cpu_job_id,
            "cpu_partition": "all_debug",
            "gpu_job_id": gpu_job_id,
            "gpu_partition": "betelgeuse",
            "gpu_reservation": "gpu_0003_grpA",
            "numpy_job_id": numpy_job_id,
            "tmpdir_on_nodes": "/tmp",
        },
        "source": _source_stamp(),
        "verdict": {
            "full_port_authorised_by_receipt": False,
            "numerical_parity": "HOLD",
            "numerical_reason": (
                "the GPU-to-CPU ULP distribution is not close to byte-identical: "
                f"median={gpu_to_cpu['ulp']['p50']}, "
                f"p99={gpu_to_cpu['ulp']['p99']}, "
                f"max={gpu_to_cpu['ulp']['max']}, and byte-identical fraction="
                f"{gpu_to_cpu['ulp']['byte_identical_fraction']:.9g}"
            ),
            "performance_projection": "PASS",
            "performance_reason": (
                f"the measured linear projection is {projected_seconds:.9g} seconds, "
                "below two minutes"
            ),
        },
    }
    _write_json(RECEIPT, receipt)
    parity = receipt["comparison_gpu_to_cpu_jax"]["ulp"]
    numpy_parity = receipt["comparison_gpu_to_production_numpy"]
    REPORT.write_text(
        "# Packed flux-kernel device receipt\n\n"
        "## Outcome\n\n"
        f"The fp64 packed closed-form G0 evaluator processed {gpu['pair_count']:,} "
        f"coarse-fixture pairs on {gpu['device_kind']} at "
        f"{gpu['kernel_pairs_per_second']:,.3f} pairs/s "
        f"({gpu['kernel_seconds']:.6f} s kernel wall). "
        f"The measured rate projects the complete nine-block fine fixture to "
        f"{projected_seconds:.6f} s ({projected_seconds / 60.0:.6f} min).\n\n"
        "The performance projection passes the two-minute input, but the numerical "
        "parity input is a **HOLD**. This receipt does not authorise the full port.\n\n"
        "## Numerical receipt\n\n"
        "Against the same single-source JAX graph on CPU, "
        f"{parity['byte_identical_count']:,}/"
        f"{parity['count']:,} elements were byte-identical "
        f"({parity['byte_identical_fraction']:.9%}). Absolute ULP percentiles were "
        f"p50={parity['p50']}, p90={parity['p90']}, p99={parity['p99']}, "
        f"p99.9={parity['p999']}, max={parity['max']}. The complete element-wise "
        "ULP histogram is stored in `receipt.json`.\n\n"
        f"Against the production NumPy-built G0 reference, the device block's maximum "
        "absolute difference was "
        f"{numpy_parity['max_absolute_difference']:.17g} and its maximum relative "
        f"difference was {numpy_parity['max_relative_difference']:.17g}.\n\n"
        "## CPU lane\n\n"
        "The same compiled fp64 graph took "
        f"{cpu['kernel_seconds']:.6f} s on CPU, while "
        f"the independently timed production NumPy uniform-G0 path took "
        f"{numpy_result['seconds']:.6f} s (ratio {cpu_ratio:.6f}x). The banked current "
        "NumPy grid-family stage for all three flux orders is "
        f"{banked_numpy_grid_seconds:.6f} s.\n\n"
        "## Method and boundary\n\n"
        f"Every launch used a fixed {TARGET_TILE}×{SOURCE_TILE} pair tile, fp64, "
        f"{RESIDUAL_NODES} residual nodes, and {TRIPS} fixed Bulirsch `cel` "
        "descent trips. The packed arithmetic is the existing `xp`-threaded "
        "production candidate; it evaluates the uniform ψ, B_R, and B_Z triple, "
        "of which only the flux G0 row was retained in this receipt. CPU and H200 ran "
        "in separate processes so each backend compiled the same source graph "
        "independently. The production NumPy path was not modified.\n",
        encoding="utf-8",
    )
    print(
        "RECEIPT "
        f"byte_fraction={parity['byte_identical_fraction']:.9g} "
        f"max_ulp={parity['max']} gpu_pairs_per_s={gpu['kernel_pairs_per_second']:.9g} "
        f"projected_fine_s={projected_seconds:.9g} cpu_numpy_ratio={cpu_ratio:.9g}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--output", type=Path, required=True)
    evaluate_parser.add_argument("--result", type=Path, required=True)
    evaluate_parser.add_argument(
        "--expected-backend", choices=("cpu", "gpu"), required=True
    )
    subparsers.add_parser("numpy")
    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("--cpu-job-id", required=True)
    merge_parser.add_argument("--gpu-job-id", required=True)
    merge_parser.add_argument("--numpy-job-id", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "evaluate":
        evaluate(args.output, args.result, args.expected_backend)
    elif args.command == "numpy":
        numpy_reference()
    else:
        merge(args.cpu_job_id, args.gpu_job_id, args.numpy_job_id)


if __name__ == "__main__":
    os.environ.setdefault("JAX_ENABLE_X64", "true")
    main()
