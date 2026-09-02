"""Measure one traced MAST coil-current sweep through the public solve seam."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import threading
import time
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from benchmarks import efit_forward_parity_slice as parity
from benchmarks import mast_response_carrier_warm as response_carrier
from nova.equilibrium.fixed_point import FixedPointTerminationReason
from nova.equilibrium.topology import TopologyClass
from nova.imas.mast_solve_inputs import SHOT_STORE
from nova.jax.config import (
    configure_dtypes,
    configure_persistent_compilation_cache,
    default_persistent_compilation_cache_root,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs/figures/forward-solve-api/coil-edit-latency.json"
DEFAULT_FIGURE = ROOT / "docs/figures/forward-solve-api/coil-edit-latency.png"
SHOT = 21989
SLICE_INDEX = 55
EDIT_FRACTIONS = 0.05 * np.asarray(
    (0, 1, 2, 3, 4, 3, 2, 1, 0, -1, -2, -3, -4, -3, -2, -1, 0, 1, 2, 3, 4),
    dtype=np.float64,
)
EDIT_COUNT = len(EDIT_FRACTIONS)
COIL_FAMILY = "p6_upper"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _archive_scalar(archive: Any, name: str) -> str:
    values = np.asarray(archive[name])
    if values.shape != ():
        raise ValueError(f"persisted {name} must be scalar")
    return str(values.item())


def _response_cache(carrier_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the persisted response and its complete input ledger."""
    response, metadata = response_carrier.load_carrier(carrier_path)
    with np.load(carrier_path, allow_pickle=False) as archive:
        input_digests = json.loads(_archive_scalar(archive, "input_digests_json"))
        audit = json.loads(_archive_scalar(archive, "audit_json"))
    audit["stored_circuit_count"] = metadata["stored_circuit_count"]
    return {
        "response": response,
        "input_digests": input_digests,
        "audit": audit,
    }, metadata


def _scheduler() -> dict[str, Any]:
    job_id = os.environ.get("SLURM_JOB_ID")
    accepted_time = None
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
            accepted_time = fields.get("TimeLimit")
    return {
        "job_id": job_id,
        "job_name": os.environ.get("SLURM_JOB_NAME"),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "reservation": os.environ.get("SLURM_JOB_RESERVATION"),
        "node": socket.gethostname(),
        "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "accepted_time_limit": accepted_time,
    }


def _require_measurement_host() -> None:
    device = jax.devices()[0]
    if device.platform != "gpu" or "H200" not in device.device_kind:
        raise RuntimeError(f"one H200 is required, got {device}")
    if os.environ.get("SLURM_JOB_PARTITION") != "betelgeuse":
        raise RuntimeError("the betelgeuse partition is required")
    if os.environ.get("SLURM_JOB_RESERVATION") != "gpu_0003_grpA":
        raise RuntimeError("the gpu_0003_grpA reservation is required")
    if os.environ.get("SLURM_CPUS_PER_TASK") != "1":
        raise RuntimeError("the measurement requires exactly one requested CPU")
    if os.environ.get("JAX_PLATFORMS") != "cuda,cpu":
        raise RuntimeError("JAX_PLATFORMS=cuda,cpu must be set in the job body")
    if os.environ.get("TMPDIR") != "/tmp":
        raise RuntimeError("TMPDIR=/tmp must be set in the job body")


def _heartbeat(stop: threading.Event, started: float) -> None:
    """Emit liveness while carrier assembly or compilation is quiet."""
    while not stop.wait(30.0):
        print(
            f"HEARTBEAT elapsed_seconds={time.perf_counter() - started:.1f}",
            flush=True,
        )


def _tree_digest(value: Any) -> str:
    digest = hashlib.sha256()
    for leaf in jax.tree.leaves(value):
        array = np.ascontiguousarray(np.asarray(leaf))
        digest.update(array.dtype.str.encode())
        digest.update(b"\0")
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _tree_bit_identical(left: Any, right: Any) -> bool:
    left_leaves = jax.tree.leaves(left)
    right_leaves = jax.tree.leaves(right)
    return len(left_leaves) == len(right_leaves) and all(
        np.array_equal(np.asarray(a), np.asarray(b))
        for a, b in zip(left_leaves, right_leaves, strict=True)
    )


def _termination_name(value: Any) -> str:
    return FixedPointTerminationReason(int(np.asarray(value))).name.lower()


def _render(rows: list[dict[str, Any]], figure_path: Path) -> None:
    indices = np.asarray([row["edit_index"] for row in rows])
    milliseconds = np.asarray([row["wall_milliseconds"] for row in rows])
    displacement = 1.0e3 * np.asarray([row["boundary_displacement_m"] for row in rows])
    colours = [
        "#d97706" if row["compilation_cache"] == "miss" else "#2563eb" for row in rows
    ]
    figure, axes = plt.subplots(2, 1, figsize=(8.4, 7.2), constrained_layout=True)
    axes[0].plot(indices, milliseconds, color="0.72", lw=1.0, zorder=1)
    axes[0].scatter(indices, milliseconds, c=colours, s=34, zorder=2)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Solve wall time [ms]")
    axes[0].set_title("MAST 21989/55 warm traced P6-upper edits")
    axes[0].grid(True, which="both", alpha=0.25)
    axes[0].scatter([], [], color="#d97706", label="compile miss")
    axes[0].scatter([], [], color="#2563eb", label="cache hit")
    axes[0].legend()
    axes[1].plot(indices, displacement, color="#059669", marker="o", ms=4)
    axes[1].axhline(0.0, color="0.5", lw=0.8)
    axes[1].set_xlabel("Successive edit index")
    axes[1].set_ylabel("Boundary displacement [mm]")
    axes[1].grid(True, alpha=0.25)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)


def _prepare_case(carrier_path: Path) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    response_cache, carrier = _response_cache(carrier_path)
    selected = {"shot": SHOT, "slice_index": SLICE_INDEX}
    case, context = parity._mast_case_from_selection(
        SHOT_STORE,
        selected,
        qualification=None,
    )
    passive_case, profile, policy = parity._passive_inclusive_case(
        case,
        context,
        response_cache,
    )
    if not policy["response_matrix_reused"]:
        raise RuntimeError("the persisted response carrier was not reused")
    if policy["stored_circuit_count"] != 101:
        raise RuntimeError("the passive-inclusive current vector is not complete")
    matching = [row for row in policy["active_mapping"] if row["family"] == COIL_FAMILY]
    if len(matching) != 1:
        raise RuntimeError("the P6 upper circuit mapping is not unique")
    circuit_index = int(matching[0]["stored_circuit"])
    prescribed = profile.operator.prescribed_current_field
    if prescribed is None or prescribed.current.shape != (101,):
        raise RuntimeError("the operator does not hold the 101-circuit vector")
    prepared = {
        "initial": jnp.asarray(passive_case["state"]),
        "prescribed_current": jnp.asarray(prescribed.current),
        "target_current": abs(float(case["reference"]["plasma_current_a"])),
        "circuit_index": circuit_index,
        "coil_mapping": matching[0],
        "reference": case["reference"],
        "policy": policy,
    }
    return profile, prepared, carrier


def run(
    output: Path,
    figure: Path,
    carrier_path: Path,
) -> dict[str, Any]:
    """Compile once and measure successive warm prescribed-current edits."""
    total_started = time.perf_counter()
    configure_dtypes()
    _require_measurement_host()
    cache = configure_persistent_compilation_cache(
        default_persistent_compilation_cache_root()
    )
    stop = threading.Event()
    reporter = threading.Thread(
        target=_heartbeat,
        args=(stop, total_started),
        daemon=True,
    )
    reporter.start()
    try:
        profile, prepared, carrier = _prepare_case(carrier_path)
        initial = prepared["initial"]
        base_current = prepared["prescribed_current"]
        circuit_index = prepared["circuit_index"]
        target_current = prepared["target_current"]
        edit_vectors = []
        for fraction in EDIT_FRACTIONS:
            values = np.asarray(base_current, dtype=np.float64).copy()
            values[circuit_index] *= 1.0 + fraction
            edit_vectors.append(jnp.asarray(values))

        def solve(state: jax.Array, prescribed_current: jax.Array) -> Any:
            return profile.solve_branch(
                state,
                TopologyClass.DIVERTED,
                route="newton_krylov",
                prescribed_current=prescribed_current,
                target_current=target_current,
                tolerance=parity.FIXED_POINT_CRITERION,
                newton_steps=parity.NEWTON_STEPS,
                gmres_iterations=parity.GMRES_ITERATIONS,
                warmup=parity.WARMUP_SWEEPS,
                relaxation=parity.RELAXATION,
                step_cap=parity.STEP_CAP,
            )

        jitted = jax.jit(solve)
        stablehlo = jitted.lower(initial, edit_vectors[0]).as_text(dialect="stablehlo")
        stablehlo_identity = _sha256_bytes(stablehlo.encode())
        rows: list[dict[str, Any]] = []
        executable_identity = None
        state = initial
        reference_lcfs = None
        reference_boundary = None
        for index, (fraction, current) in enumerate(
            zip(EDIT_FRACTIONS, edit_vectors, strict=True)
        ):
            cache_before = int(jitted._cache_size())
            started = time.perf_counter()
            branch = jitted(state, current)
            jax.block_until_ready(branch)
            wall_milliseconds = 1.0e3 * (time.perf_counter() - started)
            cache_after = int(jitted._cache_size())
            compiled = jitted.lower(state, current).compile()
            fingerprint = compiled.runtime_executable().fingerprint.decode()
            if executable_identity is None:
                executable_identity = fingerprint
            elif fingerprint != executable_identity:
                raise RuntimeError("a coil edit changed the executable identity")
            fixed_point = branch.equilibrium.fixed_point
            labelled = branch.equilibrium.labelled_flux
            lcfs_count = int(np.asarray(labelled.lcfs_vertex_count))
            lcfs = np.asarray(labelled.lcfs)[:lcfs_count]
            boundary = np.asarray(branch.equilibrium.topology.boundary)
            if reference_boundary is None:
                reference_boundary = boundary
                reference_lcfs = lcfs
            if lcfs_count and len(reference_lcfs):
                distances = np.linalg.norm(
                    lcfs[:, None, :] - reference_lcfs[None, :, :], axis=2
                )
                boundary_displacement = float(
                    max(
                        np.max(np.min(distances, axis=0)),
                        np.max(np.min(distances, axis=1)),
                    )
                )
                displacement_source = "lcfs_symmetric_sup"
            else:
                boundary_displacement = float(
                    np.linalg.norm(boundary - reference_boundary)
                )
                displacement_source = "binding_point"
            row = {
                "edit_index": index,
                "edit_fraction": float(fraction),
                "coil_current_a": float(np.asarray(current[circuit_index])),
                "wall_milliseconds": wall_milliseconds,
                "compilation_cache": ("miss" if cache_after > cache_before else "hit"),
                "compile_count": cache_after,
                "jit_cache_size_before": cache_before,
                "jit_cache_size_after": cache_after,
                "executable_identity": fingerprint,
                "stablehlo_sha256": stablehlo_identity,
                "converged": bool(np.asarray(branch.converged)),
                "residual": float(np.asarray(branch.residual)),
                "trip_count": int(np.asarray(fixed_point.active_set_iterations)),
                "fixed_iteration_count": int(np.asarray(branch.iterations)),
                "termination": _termination_name(fixed_point.termination_reason),
                "lcfs_vertex_count": lcfs_count,
                "boundary_displacement_m": boundary_displacement,
                "boundary_displacement_source": displacement_source,
            }
            rows.append(row)
            state = branch.equilibrium.flux
            print(
                "EDIT_DONE "
                f"index={index + 1}/{EDIT_COUNT} "
                f"fraction={fraction:+.6f} "
                f"milliseconds={wall_milliseconds:.6f} "
                f"cache={row['compilation_cache']} "
                f"converged={row['converged']} trips={row['trip_count']} "
                f"boundary_mm={1.0e3 * boundary_displacement:.6f}",
                flush=True,
            )

        explicit_stored = jitted(initial, base_current)
        jax.block_until_ready(explicit_stored)

        def solve_omitted(state: jax.Array) -> Any:
            return profile.solve_branch(
                state,
                TopologyClass.DIVERTED,
                route="newton_krylov",
                target_current=target_current,
                tolerance=parity.FIXED_POINT_CRITERION,
                newton_steps=parity.NEWTON_STEPS,
                gmres_iterations=parity.GMRES_ITERATIONS,
                warmup=parity.WARMUP_SWEEPS,
                relaxation=parity.RELAXATION,
                step_cap=parity.STEP_CAP,
            )

        omitted = jax.jit(solve_omitted)(initial)
        jax.block_until_ready(omitted)
        omitted_identity = _tree_digest(omitted)
        explicit_identity = _tree_digest(explicit_stored)
        omitted_bit_identical = _tree_bit_identical(omitted, explicit_stored)
        warm_ms = np.asarray(
            [row["wall_milliseconds"] for row in rows[1:]], dtype=np.float64
        )
        all_cache_hits_after_first = all(
            row["compilation_cache"] == "hit" for row in rows[1:]
        )
        one_executable = len({row["executable_identity"] for row in rows}) == 1
        all_converged = all(row["converged"] for row in rows)
        median_warm_ms = float(np.median(warm_ms))
        latency_regime = "millisecond" if median_warm_ms < 1.0e3 else "second"
        boundary_displacements = np.asarray(
            [row["boundary_displacement_m"] for row in rows], dtype=np.float64
        )
        gates = {
            "at_least_twenty_edits_recorded": len(rows) >= 20,
            "successive_edits_are_five_percent": bool(
                np.allclose(np.abs(np.diff(EDIT_FRACTIONS)), 0.05)
            ),
            "first_edit_is_compile_miss": rows[0]["compilation_cache"] == "miss",
            "all_later_edits_are_cache_hits": all_cache_hits_after_first,
            "compile_count_is_one": int(jitted._cache_size()) == 1,
            "executable_identity_unchanged": one_executable,
            "all_edits_converged": all_converged,
            "boundary_displacement_is_finite": bool(
                np.all(np.isfinite(boundary_displacements))
            ),
            "omitted_path_bit_identical_to_stored_vector": omitted_bit_identical,
        }
        passed = all(gates.values())
        exit_marker = 0 if passed else 2
        _render(rows, figure)
        receipt = {
            "schema": "nova.coil-edit-latency",
            "measurement_state": "complete",
            "verdict": "PASS" if passed else "FAIL",
            "gates": gates,
            "source_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "forward_module": {
                "path": "nova/equilibrium/forward.py",
                "sha256": _sha256(ROOT / "nova/equilibrium/forward.py"),
            },
            "driver": {
                "path": str(Path(__file__).relative_to(ROOT)),
                "sha256": _sha256(Path(__file__)),
            },
            "scheduler": _scheduler(),
            "runtime": {
                "host": platform.node(),
                "python": platform.python_version(),
                "jax": jax.__version__,
                "device": jax.devices()[0].device_kind,
                "platform": jax.devices()[0].platform,
                "jax_platforms": os.environ.get("JAX_PLATFORMS"),
                "tmpdir": os.environ.get("TMPDIR"),
                "elapsed_seconds": time.perf_counter() - total_started,
                "exit_marker": exit_marker,
            },
            "persistent_compilation_cache": cache.receipt(),
            "carrier": carrier,
            "case": {
                "machine": "MAST",
                "shot": SHOT,
                "slice_index": SLICE_INDEX,
                "time_s": float(prepared["reference"]["time_s"]),
                "seed_policy": (
                    "cold frozen-six state for the first solve; each later solve "
                    "starts from the preceding terminal flux"
                ),
                "route": "newton_krylov",
                "target_current_a": target_current,
                "current_pin": True,
                "stored_circuit_count": 101,
                "coil_family": COIL_FAMILY,
                "coil_circuit_index": circuit_index,
                "shot_coil_current_a": float(np.asarray(base_current[circuit_index])),
                "edit_fraction_bounds": [
                    float(np.min(EDIT_FRACTIONS)),
                    float(np.max(EDIT_FRACTIONS)),
                ],
                "successive_edit_step_fraction": 0.05,
            },
            "compile": {
                "count": int(jitted._cache_size()),
                "executable_identity": executable_identity,
                "stablehlo_sha256": stablehlo_identity,
            },
            "summary": {
                "edit_count": len(rows),
                "median_warm_wall_milliseconds": median_warm_ms,
                "minimum_warm_wall_milliseconds": float(warm_ms.min()),
                "maximum_warm_wall_milliseconds": float(warm_ms.max()),
                "latency_regime": latency_regime,
                "latency_statement": (
                    f"The median warm solve is {median_warm_ms:.3f} ms, in the "
                    f"{latency_regime} regime."
                ),
                "converged_edit_count": sum(row["converged"] for row in rows),
                "trip_count_minimum": min(row["trip_count"] for row in rows),
                "trip_count_median": float(
                    np.median([row["trip_count"] for row in rows])
                ),
                "trip_count_maximum": max(row["trip_count"] for row in rows),
                "maximum_boundary_displacement_m": float(boundary_displacements.max()),
            },
            "omitted_argument_identity": {
                "omitted_receipt_sha256": omitted_identity,
                "explicit_stored_vector_receipt_sha256": explicit_identity,
                "bit_identical": omitted_bit_identical,
            },
            "edits": rows,
            "figure": str(figure),
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(f"EXIT_MARKER={exit_marker}", flush=True)
        if not passed:
            raise SystemExit(exit_marker)
        return receipt
    finally:
        stop.set()
        reporter.join(timeout=2.0)


def _sbatch_script(arguments: argparse.Namespace) -> str:
    log_directory = arguments.log_directory.resolve()
    worktree = ROOT.resolve()
    environment = Path("/home/ITER/mcintos/Code/nova/.venv")
    command = (
        f"UV_PROJECT_ENVIRONMENT={environment} PYTHONPATH={worktree} "
        "uv run --no-sync python benchmarks/coil_edit_latency.py run "
        f"--carrier {arguments.carrier.resolve()} "
        f"--output {arguments.output.resolve()} "
        f"--figure {arguments.figure.resolve()}"
    )
    return f"""#!/usr/bin/env bash
#SBATCH --job-name=coil-edit-latency
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output={log_directory}/coil-edit-latency-%j.log
set -uo pipefail
export JAX_PLATFORMS=cuda,cpu
export TMPDIR=/tmp
cd {worktree}
{command}
result=$?
echo EXIT_MARKER=$result
exit $result
"""


def _submit(arguments: argparse.Namespace) -> None:
    arguments.log_directory.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        ["sbatch", "--parsable"],
        input=_sbatch_script(arguments),
        check=True,
        capture_output=True,
        text=True,
    )
    print(completed.stdout.strip())


def _harvest(output: Path) -> None:
    receipt = json.loads(output.read_text(encoding="utf-8"))
    scheduler = receipt["scheduler"]
    print(
        json.dumps(
            {
                "verdict": receipt["verdict"],
                "job_id": scheduler["job_id"],
                "node": scheduler["node"],
                "elapsed_seconds": receipt["runtime"]["elapsed_seconds"],
                "exit_marker": receipt["runtime"]["exit_marker"],
                "edit_count": receipt["summary"]["edit_count"],
                "median_warm_wall_milliseconds": receipt["summary"][
                    "median_warm_wall_milliseconds"
                ],
                "converged_edit_count": receipt["summary"]["converged_edit_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    run_parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    run_parser.add_argument(
        "--carrier", type=Path, default=response_carrier.DEFAULT_CARRIER
    )
    for name in ("sbatch", "submit"):
        job_parser = subparsers.add_parser(name)
        job_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
        job_parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
        job_parser.add_argument(
            "--carrier", type=Path, default=response_carrier.DEFAULT_CARRIER
        )
        job_parser.add_argument(
            "--log-directory",
            type=Path,
            default=Path(
                "/home/ITER/mcintos/.config/reckon/crew/runs/"
                "r-20260902T151655477529-fsa-coil-edit-latency/logs"
            ),
        )

    harvest_parser = subparsers.add_parser("harvest")
    harvest_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    if arguments.command == "run":
        run(
            arguments.output,
            arguments.figure,
            arguments.carrier,
        )
    elif arguments.command == "sbatch":
        print(_sbatch_script(arguments), end="")
    elif arguments.command == "submit":
        _submit(arguments)
    else:
        _harvest(arguments.output)


if __name__ == "__main__":
    main()
