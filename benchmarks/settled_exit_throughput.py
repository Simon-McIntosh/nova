"""Measure settled-member masking and replay its committed-bank comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import subprocess
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks.diiid_batched_throughput import build_workload
from nova.jax.config import (
    configure_dtypes,
    configure_persistent_compilation_cache,
    default_persistent_compilation_cache_root,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT / "docs/figures/solver-trip-orchestration/settled-exit-throughput.json"
)
SETTLEMENT_RECEIPT = (
    ROOT / "docs/figures/solver-trip-orchestration/settlement-histogram.json"
)
MAST_BANK = (
    ROOT / "docs/figures/primary-xpoint-evidence/efit-topology-corroboration.json"
)
DIIID_BANK = (
    ROOT / "docs/figures/diiid-forward-onboarding/forward-gs/forward_gs_receipt.json"
)
WIDTH = 1024
TRIP_LIMIT = 16
COMPARISON_FLOOR_MS = 1.75


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _strict(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _strict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict(item) for item in value]
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    return value


def _float_bits(value: float) -> str:
    return np.asarray(value, dtype=np.float64).tobytes().hex()


def _paired_bank_comparison(census: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for machine, payload in census["machines"].items():
        for record in payload["records"]:
            settlement_trip = record["settlement_trip_count"]
            residuals = record["active_set_residuals"]
            differences = record["mask_differences"]
            if settlement_trip is None:
                rows.append(
                    {
                        "machine": machine,
                        "identity": record["identity"],
                        "classification": "non_settling_full_trip_fallback",
                        "executed_trips_after_policy": TRIP_LIMIT,
                        "recorded_trips": record["recorded_trips"],
                    }
                )
                continue
            index = settlement_trip - 1
            suffix = differences[index:]
            exit_residual = float(residuals[index])
            terminal_residual = float(residuals[-1])
            suffix_residual_bits = [
                _float_bits(float(value)) for value in residuals[index:]
            ]
            suffix_is_mask_settled = bool(suffix) and all(
                value == 0 for value in suffix
            )
            suffix_is_result_noop = len(set(suffix_residual_bits)) == 1
            terminal_bit_identical = _float_bits(exit_residual) == _float_bits(
                terminal_residual
            )
            rows.append(
                {
                    "machine": machine,
                    "identity": record["identity"],
                    "classification": (
                        "qualified_recorded_noop_suffix"
                        if suffix_is_result_noop
                        else "qualification_unproven_residual_progress"
                    ),
                    "recorded_trips": record["recorded_trips"],
                    "executed_trips_after_policy": (
                        settlement_trip
                        if suffix_is_result_noop
                        else record["recorded_trips"]
                    ),
                    "mask_suffix_all_zero": suffix_is_mask_settled,
                    "post_settlement_recorded_residual_is_noop": suffix_is_result_noop,
                    "exit_relative_residual": exit_residual,
                    "full_relative_residual": terminal_residual,
                    "absolute_residual_difference": abs(
                        exit_residual - terminal_residual
                    ),
                    "terminal_relative_residual_bit_identical": terminal_bit_identical,
                    "acceptance_qualification": (
                        "retained result is a recorded no-op"
                        if suffix_is_result_noop
                        else "not established because the recorded residual improves"
                    ),
                    "identity_required": suffix_is_result_noop,
                    "identity_gate_passed": (
                        not suffix_is_result_noop or terminal_bit_identical
                    ),
                }
            )
    mask_settling = [
        row
        for row in rows
        if row["classification"] != "non_settling_full_trip_fallback"
    ]
    identity_required = [row for row in mask_settling if row["identity_required"]]
    differing = [
        row
        for row in mask_settling
        if not row["terminal_relative_residual_bit_identical"]
    ]
    if not all(row["mask_suffix_all_zero"] for row in mask_settling):
        raise RuntimeError("the committed settlement census contains a nonzero suffix")
    if not all(row["identity_gate_passed"] for row in identity_required):
        raise RuntimeError("a recorded no-op suffix changed its terminal residual bits")
    return {
        "method": (
            "paired replay of each committed bank's recorded full terminal residual "
            "against the residual at the censused final all-zero mask suffix; only "
            "bit-identical recorded no-op suffixes establish the acceptance "
            "qualification because promotion telemetry was not banked; no "
            "equilibrium bank is regenerated"
        ),
        "rows": rows,
        "summary": {
            "rows": len(rows),
            "mask_settling_rows": len(mask_settling),
            "non_settling_rows": len(rows) - len(mask_settling),
            "qualified_recorded_noop_suffix_rows": len(identity_required),
            "qualified_recorded_noop_suffix_bit_identical_rows": sum(
                row["terminal_relative_residual_bit_identical"]
                for row in identity_required
            ),
            "qualification_unproven_rows": len(differing),
            "maximum_absolute_residual_difference": max(
                (row["absolute_residual_difference"] for row in differing),
                default=0.0,
            ),
            "identity_gate_passed": all(
                row["identity_gate_passed"] for row in identity_required
            ),
        },
    }


def _scheduler() -> dict[str, Any]:
    job_id = os.environ.get("SLURM_JOB_ID")
    accepted = None
    if job_id:
        completed = subprocess.run(
            ["scontrol", "show", "job", "-o", job_id],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode == 0:
            fields = {
                token.split("=", 1)[0]: token.split("=", 1)[1]
                for token in completed.stdout.split()
                if "=" in token
            }
            accepted = fields.get("TimeLimit")
    return {
        "job_id": job_id,
        "job_name": os.environ.get("SLURM_JOB_NAME"),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "reservation": os.environ.get("SLURM_JOB_RESERVATION"),
        "job_gpus": os.environ.get("SLURM_JOB_GPUS"),
        "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "memory_per_node_mb": os.environ.get("SLURM_MEM_PER_NODE"),
        "accepted_time_limit": accepted,
        "temporary_directory": os.environ.get("TMPDIR"),
    }


def _distribution(samples: list[float]) -> dict[str, Any]:
    values = np.asarray(samples, dtype=float)
    return {
        "samples_batch_seconds": samples,
        "sample_count": len(samples),
        "median_batch_seconds": float(np.median(values)),
        "minimum_batch_seconds": float(values.min()),
        "maximum_batch_seconds": float(values.max()),
        "median_ms_per_member": 1.0e3 * float(np.median(values)) / WIDTH,
    }


def _measure(compiled, initial, current, settlement, repeats, name):
    first_started = time.perf_counter()
    first = compiled(initial, current, jnp.asarray(settlement))
    jax.block_until_ready(first.flux)
    first_seconds = time.perf_counter() - first_started
    samples = []
    result = first
    for repeat in range(repeats):
        started = time.perf_counter()
        result = compiled(initial, current, jnp.asarray(settlement))
        jax.block_until_ready(result.flux)
        seconds = time.perf_counter() - started
        samples.append(seconds)
        print(
            f"SAMPLE_DONE name={name} repeat={repeat + 1}/{repeats} "
            f"seconds={seconds:.9f}",
            flush=True,
        )
    iterations = np.asarray(result.fixed_point.active_set_iterations, dtype=int)
    return result, {
        "first_execute_batch_seconds": first_seconds,
        "steady": _distribution(samples),
        "active_set_iterations": {
            "minimum": int(iterations.min()),
            "median": float(np.median(iterations)),
            "mean": float(np.mean(iterations)),
            "maximum": int(iterations.max()),
        },
    }


def run(output: Path, repeats: int) -> dict[str, Any]:
    configure_dtypes()
    if jax.devices()[0].platform != "gpu":
        raise RuntimeError("measurement requires a JAX GPU")
    if os.environ.get("SLURM_JOB_PARTITION") != "betelgeuse":
        raise RuntimeError("measurement requires partition betelgeuse")
    if os.environ.get("SLURM_JOB_RESERVATION") != "gpu_0003_grpA":
        raise RuntimeError("measurement requires reservation gpu_0003_grpA")
    if os.environ.get("TMPDIR") != "/tmp":
        raise RuntimeError("measurement requires TMPDIR=/tmp")
    cache = configure_persistent_compilation_cache(
        default_persistent_compilation_cache_root()
    )
    census = json.loads(SETTLEMENT_RECEIPT.read_text(encoding="utf-8"))
    comparison = _paired_bank_comparison(census)
    profile, seed = build_workload()
    initial = jnp.repeat(seed[None, :], WIDTH, axis=0)
    base_current = jnp.asarray(profile.operator.external_current)
    current = jnp.repeat(base_current[None, :], WIDTH, axis=0)

    def solve(state, conductor, settlement):
        return profile.solve_batch(
            state,
            route="newton_krylov",
            current=conductor,
            newton_steps=1,
            gmres_iterations=4,
            warmup=1,
            active_set_steps=TRIP_LIMIT,
            stop_on_active_set_settlement=settlement,
        )

    print("COMPILE_START", flush=True)
    started = time.perf_counter()
    compiled = jax.jit(solve).lower(initial, current, jnp.asarray(False)).compile()
    compile_seconds = time.perf_counter() - started
    print(f"COMPILE_DONE seconds={compile_seconds:.6f}", flush=True)
    baseline_result, baseline = _measure(
        compiled, initial, current, False, repeats, "full_trip_control"
    )
    settled_result, settled = _measure(
        compiled, initial, current, True, repeats, "settled_exit"
    )
    baseline_seconds = baseline["steady"]["median_batch_seconds"]
    settled_seconds = settled["steady"]["median_batch_seconds"]
    speedup = baseline_seconds / settled_seconds
    normalized_achieved = COMPARISON_FLOOR_MS / speedup
    trip_projection = (
        COMPARISON_FLOOR_MS * settled["active_set_iterations"]["mean"] / TRIP_LIMIT
    )
    mast_projection = census["machines"]["MAST"]["summary"]["settled_only_projection"][
        "projected_ms_per_slice"
    ]
    diiid_projection = census["machines"]["DIII-D"]["summary"][
        "settled_only_projection"
    ]["projected_ms_per_slice"]
    projection_band = [
        min(mast_projection, diiid_projection),
        max(mast_projection, diiid_projection),
    ]
    baseline_flux = np.asarray(baseline_result.flux)
    settled_flux = np.asarray(settled_result.flux)
    receipt = {
        "schema": "nova.settled_exit_throughput",
        "measurement_state": "complete",
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "driver": {
            "path": str(Path(__file__).relative_to(ROOT)),
            "sha256": _sha256(Path(__file__)),
        },
        "scheduler": _scheduler(),
        "device": {
            "host": platform.node(),
            "platform": jax.devices()[0].platform,
            "kind": jax.devices()[0].device_kind,
            "jax_version": jax.__version__,
            "jax_enable_x64": bool(jax.config.jax_enable_x64),
        },
        "configuration": {
            "width": WIDTH,
            "trip_limit": TRIP_LIMIT,
            "newton_steps_per_trip": 1,
            "gmres_iterations": 4,
            "warmup_sweeps": 1,
            "steady_samples_per_arm": repeats,
        },
        "persistent_compilation_cache": cache.receipt(),
        "evidence_inputs": {
            "settlement_histogram": {
                "path": str(SETTLEMENT_RECEIPT.relative_to(ROOT)),
                "sha256": _sha256(SETTLEMENT_RECEIPT),
            },
            "mast_bank": {
                "path": str(MAST_BANK.relative_to(ROOT)),
                "sha256": _sha256(MAST_BANK),
            },
            "diiid_bank": {
                "path": str(DIIID_BANK.relative_to(ROOT)),
                "sha256": _sha256(DIIID_BANK),
            },
        },
        "paired_committed_bank_comparison": comparison,
        "h200_width_1024": {
            "compile_seconds": compile_seconds,
            "full_trip_control": baseline,
            "settled_exit": settled,
            "measured_speedup_x": speedup,
            "direct_settled_ms_per_member": settled["steady"]["median_ms_per_member"],
            "comparison_normalized_achieved_ms_per_slice": normalized_achieved,
            "trip_count_projected_ms_per_slice": trip_projection,
            "projected_bank_band_ms_per_slice": projection_band,
            "comparison_normalized_achieved_within_projected_band": (
                projection_band[0] <= normalized_achieved <= projection_band[1]
            ),
            "flux_bit_identical_member_count": int(
                np.sum(
                    np.all(
                        baseline_flux.view(np.uint64) == settled_flux.view(np.uint64),
                        axis=1,
                    )
                )
            ),
            "flux_member_count": WIDTH,
            "maximum_absolute_flux_difference": float(
                np.max(np.abs(baseline_flux - settled_flux))
            ),
        },
        "verdict": {
            "bank_identity_gate_passed": comparison["summary"]["identity_gate_passed"],
            "projection_statement": (
                "the achieved comparison-normalized H200 time is reported beside "
                "the committed 0.22-0.47 ms/slice bank projection; direct wall "
                "time remains separately labelled per member"
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(_strict(receipt), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"RECEIPT_WRITTEN={output}", flush=True)
    return receipt


def preflight() -> None:
    configure_dtypes()
    census = json.loads(SETTLEMENT_RECEIPT.read_text(encoding="utf-8"))
    comparison = _paired_bank_comparison(census)
    print(
        json.dumps(
            {
                "status": "preflight_complete",
                "width": WIDTH,
                "jax_enable_x64": bool(jax.config.jax_enable_x64),
                "paired_bank_summary": comparison["summary"],
                "bank_sha256": {
                    "settlement": _sha256(SETTLEMENT_RECEIPT),
                    "mast": _sha256(MAST_BANK),
                    "diiid": _sha256(DIIID_BANK),
                },
            },
            indent=2,
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--preflight", action="store_true")
    arguments = parser.parse_args()
    if arguments.preflight:
        preflight()
    else:
        run(arguments.output, arguments.repeats)


if __name__ == "__main__":
    main()
