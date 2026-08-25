"""Measure single-branch and two-branch converged solve cost on one H200.

The benchmark uses the coexisting limited and diverted roots from the shipped
portfolio fixture.  Every batch member is a distinct near-root state.  The
single arm solves the limited seed, while the portfolio arm solves that same
limited seed together with its paired diverted seed.  Input-file read,
host-to-device transfer, compilation, warm-up, resident execution, and the
read-inclusive total are timed and reported separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile
import time
from typing import Any, Callable

import jax
import numpy as np

from benchmarks.portfolio_warm_start import _limited_root, _problem
from nova.equilibrium.topology import TopologyClass
from nova.jax.config import configure_dtypes


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs/figures/dual-branch-selection/two-branch-batch-cost.json"
BATCH_WIDTHS = (4, 16)
NEWTON_PROMOTIONS = 2
GMRES_ITERATIONS = 30
CONVERGENCE_TOLERANCE = 1.0e-10
TIMING_REPEATS = 5
CENSUS_SLICE_COUNT = 1_341_435
MINIMUM_SEED_DISTANCE = 2.5e-4
MAXIMUM_SEED_DISTANCE = 1.0e-3


def _strict(value: Any) -> Any:
    """Return a strict JSON-compatible tree."""
    if isinstance(value, dict):
        return {str(key): _strict(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_strict(item) for item in value]
    if isinstance(value, np.ndarray):
        return _strict(value.tolist())
    if isinstance(value, np.generic):
        return _strict(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write the receipt atomically as strict JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_strict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _digest(values: np.ndarray) -> str:
    """Return the binary digest of one float64 input state."""
    array = np.ascontiguousarray(values, dtype=np.float64)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _file_digest(path: Path) -> str:
    """Return the digest of one serialized input bank."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_revision() -> str:
    """Return the exact source revision executed by the benchmark."""
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _summary(samples: list[float]) -> dict[str, Any]:
    """Return all samples and robust headline statistics."""
    return {
        "samples_seconds": samples,
        "minimum_seconds": float(np.min(samples)),
        "median_seconds": float(np.median(samples)),
        "maximum_seconds": float(np.max(samples)),
        "repeat_count": len(samples),
    }


def _distinct_inputs(
    profile: Any,
    cold_flux: np.ndarray,
    roots: np.ndarray,
    count: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Construct distinct limited/diverted near-root slice seeds."""
    spans = []
    directions = []
    for branch_index, branch_class in enumerate(
        (TopologyClass.LIMITED, TopologyClass.DIVERTED)
    ):
        _masks, topology = profile.operator.read(roots[branch_index], branch_class)
        span = abs(float(np.asarray(topology.flux_span)))
        direction = np.asarray(cold_flux[branch_index]) - roots[branch_index]
        scale = float(np.max(np.abs(direction)))
        if not np.isfinite(span) or span <= 0.0 or scale <= 0.0:
            raise RuntimeError("a branch span or seed direction is degenerate")
        spans.append(span)
        directions.append(direction / scale)

    distances = np.linspace(
        MINIMUM_SEED_DISTANCE,
        MAXIMUM_SEED_DISTANCE,
        count,
        dtype=np.float64,
    )
    portfolio = roots[None, :, :] + (
        distances[:, None, None]
        * np.asarray(spans)[None, :, None]
        * np.asarray(directions)[None, :, :]
    )
    single = np.array(portfolio[:, int(TopologyClass.LIMITED), :], copy=True)
    single_digests = [_digest(row) for row in single]
    portfolio_digests = [_digest(row) for row in portfolio]
    if len(set(single_digests)) != count or len(set(portfolio_digests)) != count:
        raise RuntimeError("the input constructor produced duplicate slice states")
    if not np.array_equal(single, portfolio[:, int(TopologyClass.LIMITED), :]):
        raise RuntimeError("the single arm does not share the portfolio limited seed")
    return (
        single,
        portfolio,
        {
            "construction": (
                "distinct deterministic distances along each production cold-seed "
                "direction from its banked converged root"
            ),
            "distance_relative_to_branch_flux_span": distances,
            "minimum_distance": MINIMUM_SEED_DISTANCE,
            "maximum_distance": MAXIMUM_SEED_DISTANCE,
            "single_slice_sha256": single_digests,
            "portfolio_slice_pair_sha256": portfolio_digests,
            "single_unique_slice_count": len(set(single_digests)),
            "portfolio_unique_slice_pair_count": len(set(portfolio_digests)),
            "single_seed_is_portfolio_limited_seed": True,
            "broadcast_or_tile_used": False,
        },
    )


def _read_input(path: Path, key: str) -> tuple[np.ndarray, float]:
    """Read and materialize one arm's input from its local serialized bank."""
    started = time.perf_counter()
    with np.load(path, allow_pickle=False) as bank:
        values = np.array(bank[key], dtype=np.float64, copy=True)
    return values, time.perf_counter() - started


def _transfer(values: np.ndarray) -> tuple[jax.Array, float]:
    """Place one input batch on the H200 and synchronize the transfer."""
    started = time.perf_counter()
    device_values = jax.device_put(values)
    device_values.block_until_ready()
    return device_values, time.perf_counter() - started


def _execute(
    compiled: Callable[[jax.Array], Any], values: jax.Array
) -> tuple[Any, float]:
    """Run one complete solve and synchronize every result leaf."""
    started = time.perf_counter()
    result = compiled(values)
    jax.block_until_ready(result)
    return result, time.perf_counter() - started


def _terminal_summary(
    arm: str,
    result: Any,
    roots: np.ndarray,
    batch_width: int,
) -> dict[str, Any]:
    """Retain convergence, failure, residual, topology, and root-error counts."""
    if arm == "single_pinned_limited":
        converged = np.asarray(result.converged, dtype=bool)
        consistent = np.asarray(result.topology_consistent, dtype=bool)
        residual = np.asarray(result.residual, dtype=np.float64)
        flux = np.asarray(result.equilibrium.flux, dtype=np.float64)
        scale = max(float(np.max(np.abs(roots[0]))), np.finfo(float).tiny)
        root_error = np.max(np.abs(flux - roots[0]), axis=1) / scale
        slice_converged = converged
        branches_attempted = batch_width
        branches_converged = int(np.sum(converged))
        per_branch = {"limited": branches_converged}
    else:
        converged = np.asarray(result.branches.converged, dtype=bool)
        consistent = np.asarray(result.branches.topology_consistent, dtype=bool)
        residual = np.asarray(result.branches.residual, dtype=np.float64)
        flux = np.asarray(result.branches.equilibrium.flux, dtype=np.float64)
        scale = np.maximum(np.max(np.abs(roots), axis=1), np.finfo(float).tiny)
        root_error = np.max(np.abs(flux - roots[None, :, :]), axis=2) / scale
        slice_converged = np.all(converged, axis=1)
        branches_attempted = 2 * batch_width
        branches_converged = int(np.sum(converged))
        per_branch = {
            "limited": int(np.sum(converged[:, int(TopologyClass.LIMITED)])),
            "diverted": int(np.sum(converged[:, int(TopologyClass.DIVERTED)])),
        }
    return {
        "input_slices_solved": batch_width,
        "input_slices_converged": int(np.sum(slice_converged)),
        "input_slice_failure_or_nonconvergence_count": int(
            batch_width - np.sum(slice_converged)
        ),
        "branches_attempted": branches_attempted,
        "branches_converged": branches_converged,
        "branch_failure_or_nonconvergence_count": branches_attempted
        - branches_converged,
        "converged_by_branch": per_branch,
        "topology_consistent_count": int(np.sum(consistent)),
        "maximum_relative_residual": float(np.max(residual)),
        "maximum_root_relative_error": float(np.max(root_error)),
        "all_finite_residuals": bool(np.all(np.isfinite(residual))),
    }


def _measure_arm(
    arm: str,
    solve: Callable[[jax.Array], Any],
    bank_path: Path,
    bank_key: str,
    roots: np.ndarray,
    batch_width: int,
) -> dict[str, Any]:
    """Measure compile, warm-up, execute-only, and read-inclusive solve cost."""
    compile_host, compile_read_seconds = _read_input(bank_path, bank_key)
    compile_input, compile_transfer_seconds = _transfer(compile_host)
    compile_started = time.perf_counter()
    compiled = jax.jit(solve).lower(compile_input).compile()
    compile_seconds = time.perf_counter() - compile_started

    warm_host, warm_read_seconds = _read_input(bank_path, bank_key)
    warm_input, warm_transfer_seconds = _transfer(warm_host)
    warm_result, warm_execute_seconds = _execute(compiled, warm_input)
    warm_terminal = _terminal_summary(arm, warm_result, roots, batch_width)

    execute_samples = []
    steady_terminal_rows = []
    for _ in range(TIMING_REPEATS):
        result, elapsed = _execute(compiled, warm_input)
        execute_samples.append(elapsed)
        steady_terminal_rows.append(_terminal_summary(arm, result, roots, batch_width))

    total_samples = []
    read_samples = []
    transfer_samples = []
    execute_with_input_samples = []
    for _ in range(TIMING_REPEATS):
        total_started = time.perf_counter()
        host_values, read_seconds = _read_input(bank_path, bank_key)
        device_values, transfer_seconds = _transfer(host_values)
        result, execute_seconds = _execute(compiled, device_values)
        total_samples.append(time.perf_counter() - total_started)
        read_samples.append(read_seconds)
        transfer_samples.append(transfer_seconds)
        execute_with_input_samples.append(execute_seconds)
        steady_terminal_rows.append(_terminal_summary(arm, result, roots, batch_width))

    execute = _summary(execute_samples)
    total = _summary(total_samples)
    median_execute = execute["median_seconds"]
    median_total = total["median_seconds"]
    return {
        "arm": arm,
        "batch_width": batch_width,
        "input_bank_key": bank_key,
        "input_shape": list(compile_host.shape),
        "input_bytes": int(compile_host.nbytes),
        "compile": {
            "input_read_seconds": compile_read_seconds,
            "host_to_device_seconds": compile_transfer_seconds,
            "lower_and_compile_seconds": compile_seconds,
        },
        "warm_up": {
            "input_read_seconds": warm_read_seconds,
            "host_to_device_seconds": warm_transfer_seconds,
            "execute_seconds": warm_execute_seconds,
            "total_seconds": (
                warm_read_seconds + warm_transfer_seconds + warm_execute_seconds
            ),
            "terminal": warm_terminal,
        },
        "steady_state_execute_only": {
            **execute,
            "seconds_per_input_slice": median_execute / batch_width,
            "input_slices_per_second": batch_width / median_execute,
            "input_residency": "device resident before the timed execute call",
        },
        "steady_state_total_including_input_read_and_transfer": {
            **total,
            "seconds_per_input_slice": median_total / batch_width,
            "input_slices_per_second": batch_width / median_total,
            "input_read_samples_seconds": read_samples,
            "host_to_device_samples_seconds": transfer_samples,
            "execute_samples_seconds": execute_with_input_samples,
            "input_source": "node-local uncompressed NPZ",
            "page_cache_policy": (
                "not flushed; samples include ordinary node-local reads after "
                "the bank was staged"
            ),
        },
        "steady_state_accounting": {
            "solve_invocations": len(steady_terminal_rows),
            "input_slice_solves": batch_width * len(steady_terminal_rows),
            "input_slice_convergences": sum(
                row["input_slices_converged"] for row in steady_terminal_rows
            ),
            "input_slice_failure_or_nonconvergence_count": sum(
                row["input_slice_failure_or_nonconvergence_count"]
                for row in steady_terminal_rows
            ),
            "branch_solves": sum(
                row["branches_attempted"] for row in steady_terminal_rows
            ),
            "branch_convergences": sum(
                row["branches_converged"] for row in steady_terminal_rows
            ),
            "branch_failure_or_nonconvergence_count": sum(
                row["branch_failure_or_nonconvergence_count"]
                for row in steady_terminal_rows
            ),
            "per_invocation": steady_terminal_rows,
        },
    }


def _run(output: Path) -> dict[str, Any]:
    """Run both arms at every declared width and write the banked receipt."""
    configure_dtypes()
    device = jax.devices()[0]
    if device.platform != "gpu" or "H200" not in device.device_kind:
        raise RuntimeError(f"reserved H200 GPU required, got {device}")
    if not platform.node().startswith("98dci4-gpu-0003"):
        raise RuntimeError(f"reserved H200 host required, got {platform.node()}")
    if os.environ.get("SLURM_JOB_RESERVATION") != "gpu_0003_grpA":
        raise RuntimeError("the gpu_0003_grpA reservation is required")
    if not jax.config.x64_enabled:
        raise RuntimeError("float64 support is disabled")

    setup_started = time.perf_counter()
    profile, cold, diverted_root = _problem()
    limited_root, limited_preparation = _limited_root(
        profile, cold.branches.flux[int(TopologyClass.LIMITED)]
    )
    roots = np.stack((limited_root, diverted_root))
    single_all, portfolio_all, input_policy = _distinct_inputs(
        profile,
        np.asarray(cold.branches.flux, dtype=np.float64),
        roots,
        max(BATCH_WIDTHS),
    )
    setup_seconds = time.perf_counter() - setup_started

    def solve_single(states: jax.Array) -> Any:
        return jax.vmap(
            lambda state: profile.solve_branch(
                state,
                TopologyClass.LIMITED,
                route="newton_krylov",
                tolerance=CONVERGENCE_TOLERANCE,
                newton_steps=NEWTON_PROMOTIONS,
                gmres_iterations=GMRES_ITERATIONS,
                warmup=0,
            )
        )(states)

    def solve_portfolio(states: jax.Array) -> Any:
        return jax.vmap(
            lambda branch_states: profile.solve_portfolio(
                branch_states,
                route="newton_krylov",
                tolerance=CONVERGENCE_TOLERANCE,
                newton_steps=NEWTON_PROMOTIONS,
                gmres_iterations=GMRES_ITERATIONS,
                warmup=0,
            )
        )(states)

    rows = []
    scratch_root = Path(os.environ.get("SLURM_TMPDIR", "/tmp"))
    with tempfile.TemporaryDirectory(
        prefix="nova-two-branch-cost-", dir=scratch_root
    ) as temporary_directory:
        temporary = Path(temporary_directory)
        for width in BATCH_WIDTHS:
            bank_path = temporary / f"inputs-{width}.npz"
            np.savez(
                bank_path,
                single=single_all[:width],
                portfolio=portfolio_all[:width],
            )
            single = _measure_arm(
                "single_pinned_limited",
                solve_single,
                bank_path,
                "single",
                roots,
                width,
            )
            portfolio = _measure_arm(
                "two_branch_portfolio",
                solve_portfolio,
                bank_path,
                "portfolio",
                roots,
                width,
            )
            execute_ratio = (
                portfolio["steady_state_execute_only"]["seconds_per_input_slice"]
                / single["steady_state_execute_only"]["seconds_per_input_slice"]
            )
            total_ratio = (
                portfolio["steady_state_total_including_input_read_and_transfer"][
                    "seconds_per_input_slice"
                ]
                / single["steady_state_total_including_input_read_and_transfer"][
                    "seconds_per_input_slice"
                ]
            )
            rows.append(
                {
                    "batch_width": width,
                    "input_bank": {
                        "format": "uncompressed NPZ staged on node-local scratch",
                        "file_bytes": bank_path.stat().st_size,
                        "sha256": _file_digest(bank_path),
                        "retained_after_run": False,
                    },
                    "single_pinned_branch": single,
                    "two_branch_portfolio": portfolio,
                    "cost_ratio_portfolio_over_single": {
                        "execute_only": execute_ratio,
                        "total_including_input_read_and_transfer": total_ratio,
                    },
                    "campaign_extrapolation": {
                        "label": "extrapolation_not_measured_campaign_throughput",
                        "slice_count": CENSUS_SLICE_COUNT,
                        "assumptions": (
                            "one H200 repeats this batch-width cost with identical "
                            "convergence and node-local input staging across every "
                            "catalog slice; compile, warm-up, and fixture setup are "
                            "excluded from the multiplied term"
                        ),
                        "single_total_seconds": (
                            CENSUS_SLICE_COUNT
                            * single[
                                "steady_state_total_including_input_read_and_transfer"
                            ]["seconds_per_input_slice"]
                        ),
                        "portfolio_total_seconds": (
                            CENSUS_SLICE_COUNT
                            * portfolio[
                                "steady_state_total_including_input_read_and_transfer"
                            ]["seconds_per_input_slice"]
                        ),
                    },
                }
            )
            del single, portfolio
            jax.clear_caches()

    receipt = {
        "schema": "nova.two-branch-batch-cost",
        "source_revision": _source_revision(),
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": {
            "hostname": platform.node(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
            "slurm_reservation": os.environ.get("SLURM_JOB_RESERVATION"),
            "device_kind": device.device_kind,
            "device_count": len(jax.devices()),
            "jax_version": jax.__version__,
            "production_dtype": "float64",
            "x64_enabled": bool(jax.config.x64_enabled),
        },
        "measurement_contract": {
            "batch_widths": BATCH_WIDTHS,
            "solver_entry_points": {
                "single": "ForwardProfile.solve_branch pinned LIMITED",
                "portfolio": "ForwardProfile.solve_portfolio LIMITED plus DIVERTED",
            },
            "newton_promotions": NEWTON_PROMOTIONS,
            "gmres_iterations_per_promotion": GMRES_ITERATIONS,
            "convergence_tolerance": CONVERGENCE_TOLERANCE,
            "timing_repeats": TIMING_REPEATS,
            "complete_converged_solves_timed": all(
                row[arm]["steady_state_accounting"][
                    "input_slice_failure_or_nonconvergence_count"
                ]
                == 0
                for row in rows
                for arm in ("single_pinned_branch", "two_branch_portfolio")
            ),
            "one_map_application_benchmark": False,
            "input_read_in_total_timed_region": True,
            "compile_and_warm_up_excluded_from_steady_state": True,
            "failure_policy": (
                "every input slice and branch is counted; failed or non-converged "
                "results remain in terminal counts and are never dropped"
            ),
        },
        "fixture": {
            "definition": (
                "analytic production-profile fixture with banked coexisting "
                "limited and diverted roots"
            ),
            "limited_root_preparation": limited_preparation,
            "diverted_root_sha256": _digest(diverted_root),
            "root_shape": list(roots.shape),
            "setup_seconds_excluded_from_measurement": setup_seconds,
        },
        "input_policy": input_policy,
        "measurements": rows,
        "decision_scope": {
            "provides": (
                "measured accelerator cost multiple for a two-branch portfolio "
                "relative to one pinned branch"
            ),
            "does_not_provide": (
                "catalog margin distribution or a policy decision by itself"
            ),
        },
    }
    _write_json(output, receipt)
    return receipt


def main() -> None:
    """Run the H200 measurement from a reserved SLURM allocation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    receipt = _run(arguments.output)
    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "source_revision": receipt["source_revision"],
                "rows": [
                    {
                        "batch_width": row["batch_width"],
                        "execute_ratio": row["cost_ratio_portfolio_over_single"][
                            "execute_only"
                        ],
                        "total_ratio": row["cost_ratio_portfolio_over_single"][
                            "total_including_input_read_and_transfer"
                        ],
                    }
                    for row in receipt["measurements"]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
