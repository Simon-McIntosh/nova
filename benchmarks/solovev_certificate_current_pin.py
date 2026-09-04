"""Compare current-pinned and unpinned Solovev certificate solves.

This diagnostic imports the certificate's cases, machine construction, production
seed, exact-field evaluation, and topology scoring.  It changes only the public
``target_current`` argument passed to the flux map; the solver and source remain
unchanged.  Map-evaluation callbacks retain the current-normalisation amplitude
without participating in the numerical result.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
from time import perf_counter
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks import solovev_certificate as certificate
from nova.equilibrium import ForwardProfile, fixed_point
from nova.equilibrium.solve_request import (
    ExplicitSolveSeed,
    ForwardSolveRequest,
    ResolvedForwardSolveDefaults,
)
from nova.equilibrium.stencil_mesh import StencilMesh
from nova.jax.config import (
    configure_dtypes,
    configure_persistent_compilation_cache,
    default_persistent_compilation_cache_root,
)
from scripts.analytic_oracle_fixtures import measure as oracle_fixture
from scripts.oracle_rebaseline import measure as recovery


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs/figures/gs-absolute-accuracy/certificate-current-pin.json"
STATIC_CASES = certificate.CASE_NAMES[:3]
REQUESTED_CELLS = (-110, -300)
ARM_NAMES = ("unpinned", "current_pinned")
EXIT_MARKER = "CERTIFICATE_CURRENT_PIN_EXIT=0"


def _strict(value: Any) -> Any:
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_strict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _lane() -> dict[str, Any]:
    return {
        "execution": "slurm" if os.environ.get("SLURM_JOB_ID") else "local",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "node": os.environ.get("SLURM_JOB_NODELIST"),
        "hostname": socket.gethostname(),
        "allocated_cpus": int(os.environ.get("SLURM_CPUS_ON_NODE", "0")) or None,
        "jax_platforms": os.environ.get("JAX_PLATFORMS"),
        "jax_default_backend": jax.default_backend(),
        "precision": "float64",
    }


def _prepare(
    case_name: str, requested_cells: int
) -> tuple[Any, ForwardProfile, Any, np.ndarray, np.ndarray, float, dict[str, Any]]:
    carrier_case, source_case, exact = certificate._case(case_name)
    machine = oracle_fixture.cached_machine(
        carrier_case,
        requested_cells,
        wall_nodes=oracle_fixture.WALL_POINT_COUNT,
    )
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    oracle_state = certificate._exact_state(case_name, exact, coordinates)
    empty_operator = oracle_fixture.forward_operator(source_case, machine)
    exact_physical = oracle_fixture.exact_current_moments(
        source_case, empty_operator, oracle_state
    )
    exact_coefficients = empty_operator.coupling_current_moments(exact_physical)
    exact_internal = oracle_fixture._internal_flux_image(
        empty_operator, exact_coefficients
    )
    operator = oracle_fixture.forward_operator(
        source_case, machine, oracle_state - exact_internal
    )
    profile = ForwardProfile(
        operator,
        StencilMesh(machine.node, machine.stencil, machine.area),
        newton_steps=recovery.NEWTON_STEPS,
    )
    seed, _moment_image, seed_receipt = recovery._moment_seed(
        source_case, machine, operator
    )
    declared_current, _centroid, aggregate = recovery._aggregate_current_moment(
        source_case
    )
    if not np.isfinite(declared_current) or declared_current == 0.0:
        raise RuntimeError("the closed-form case has no finite nonzero plasma current")
    target_receipt = {
        "value_a": declared_current,
        "source": "RotatingEquilibrium.plasma_current closed-form declaration",
        "density_quadrature_a": aggregate["quadrature_current_a"],
        "density_quadrature_relative_closure": aggregate["quadrature_relative_closure"],
        "sign_preserved": True,
    }
    return (
        machine,
        profile,
        operator,
        seed,
        oracle_state,
        declared_current,
        {
            "seed": seed_receipt,
            "target_current": target_receipt,
        },
    )


def _current_pin_solve_request(
    profile: ForwardProfile,
    seed: object,
    target_current: float,
    *,
    pinned: bool,
    carrier_identity: str,
) -> ForwardSolveRequest:
    """Declare whether the comparison arm applies the public current pin."""

    return ForwardSolveRequest.from_defaults(
        carrier_identity=carrier_identity,
        source_profile=profile.source,
        seed_policy=ExplicitSolveSeed(seed),
        policy_overrides={"current_pin": pinned},
        target_current=target_current if pinned else None,
    )


def _run_fixed_point_request(operator: Any, request: ForwardSolveRequest):
    """Apply the typed budget to the certificate's direct fixed-point route."""

    policy = request.policy
    mapped = operator.flux_map(target_current=request.target_current)
    history = fixed_point.newton_krylov(
        mapped,
        jnp.asarray(request.seed_policy.resolve(operator)),
        newton_steps=policy.newton_steps,
        gmres_iterations=policy.gmres_iterations,
        warmup=policy.warmup,
    )
    jax.block_until_ready(history.state)
    return history


def _amplitude(operator: Any, state: jax.Array, target_current: float) -> jax.Array:
    _moments, amplitude = operator.normalised_current_moments(state, target_current)
    return amplitude


def _solve_arm(
    profile: ForwardProfile,
    operator: Any,
    seed: np.ndarray,
    target_current: float,
    *,
    pinned: bool,
    carrier_identity: str,
) -> dict[str, Any]:
    request = _current_pin_solve_request(
        profile,
        seed,
        target_current,
        pinned=pinned,
        carrier_identity=carrier_identity,
    )
    seed_amplitude = float(_amplitude(operator, jnp.asarray(seed), target_current))

    compile_started = perf_counter()
    first = _run_fixed_point_request(operator, request)
    first_wall = perf_counter() - compile_started
    warm_started = perf_counter()
    terminal = _run_fixed_point_request(operator, request)
    warm_wall = perf_counter() - warm_started
    jax.block_until_ready(terminal.state)
    state = np.asarray(terminal.state, dtype=np.float64)
    trace = np.asarray(terminal.trace, dtype=np.float64)
    finite_trace = trace[np.isfinite(trace)]
    termination_code = int(np.asarray(terminal.termination_reason))
    termination = fixed_point.FixedPointTerminationReason(termination_code).name.lower()
    terminal_amplitude = float(_amplitude(operator, terminal.state, target_current))
    observed_amplitudes = [seed_amplitude, terminal_amplitude]
    terminal_moments = operator.cell_current_moments(terminal.state)
    unscaled_current = float(jnp.sum(terminal_moments.cell_current))
    achieved_current = (
        terminal_amplitude * unscaled_current if pinned else unscaled_current
    )
    applied_history = (
        list(observed_amplitudes) if pinned else [1.0 for _ in observed_amplitudes]
    )
    return {
        "state": state,
        "terminal_relative_residual": float(terminal.residual),
        "finite_residual_history": finite_trace.tolist(),
        "recorded_residual_evaluations": int(finite_trace.size),
        "attempted_newton_trip_count": int(
            np.asarray(terminal.attempted_newton_promotions)
        ),
        "accepted_newton_trip_count": int(
            np.asarray(terminal.accepted_newton_promotions)
        ),
        "termination": termination,
        "converged": bool(np.asarray(terminal.converged)),
        "carrier_identity": request.carrier_identity,
        "resolved_defaults": ResolvedForwardSolveDefaults.from_policy(
            request.policy
        ).to_dict(),
        "compile_and_first_solve_wall_seconds": first_wall,
        "compile_warm_solve_wall_seconds": warm_wall,
        "first_run_terminal_relative_residual": float(first.residual),
        "lambda": {
            "policy": (
                "declared scalar current applied at every map evaluation"
                if pinned
                else "absolute source; target-current lambda observed but not applied"
            ),
            "terminal_amplitude": terminal_amplitude,
            "map_evaluation_amplitude_history": observed_amplitudes,
            "applied_amplitude_history": applied_history,
            "history_entry_count": len(observed_amplitudes),
            "history_semantics": (
                "seed and terminal state amplitudes; the fixed-point result does "
                "not expose intermediate states"
            ),
        },
        "current": {
            "target_a": target_current,
            "unscaled_terminal_a": unscaled_current,
            "achieved_terminal_a": achieved_current,
            "terminal_relative_error": abs(achieved_current - target_current)
            / abs(target_current),
        },
    }


def _score_arm(
    arm: dict[str, Any],
    machine: Any,
    operator: Any,
    oracle_state: np.ndarray,
    exact: Any,
) -> dict[str, Any]:
    state = arm.pop("state")
    cell_count = len(machine.node)
    error = state[:cell_count] - oracle_state[:cell_count]
    topology = certificate._topology(operator, state)
    axis_reference = np.asarray(exact.magnetic_axis, dtype=np.float64)
    axis = topology["axis_rz_m"]
    arm["whole_domain_psi_error"] = {
        "sup_wb": float(np.max(np.abs(error))),
        "rms_wb": float(np.sqrt(np.mean(error**2))),
    }
    arm["magnetic_axis_position_error_m"] = (
        float(np.linalg.norm(np.asarray(axis) - axis_reference))
        if axis is not None
        else None
    )
    arm["terminal_topology"] = topology
    return arm


def _banked_unpinned(case_name: str, requested_cells: int) -> dict[str, Any]:
    """Return the immutable certificate row used as the unpinned control."""

    receipt = json.loads(
        (ROOT / "docs/figures/gs-absolute-accuracy/solovev-certificate.json").read_text(
            encoding="utf-8"
        )
    )
    row = next(
        item
        for item in receipt["cases"][case_name]["rows"]
        if item["requested_cells"] == requested_cells
    )
    residual = row["solver"]["terminal_fixed_point_residual"]
    return {
        "provenance": {
            "kind": "banked immutable certificate control",
            "source_revision": receipt["preregistration"]["source_revision"],
            "source_path": (
                "docs/figures/gs-absolute-accuracy/solovev-certificate.json"
            ),
            "same_case_machine_seed_and_solver_budget": True,
            "fresh_rerun": False,
        },
        "terminal_relative_residual": residual,
        "finite_residual_history": [residual],
        "recorded_residual_evaluations": None,
        "attempted_newton_trip_count": row["solver"]["newton_steps"],
        "accepted_newton_trip_count": None,
        "termination": (
            "fixed_point_residual_within_qualification_bound"
            if row["solver"]["qualification"] == "qualified"
            else "fixed_point_residual_above_qualification_bound_after_iteration_budget"
        ),
        "converged": row["solver"]["qualification"] == "qualified",
        "compile_and_first_solve_wall_seconds": row["solver"][
            "compile_and_solve_wall_seconds"
        ],
        "compile_warm_solve_wall_seconds": row["solver"]["compile_warm_wall_seconds"],
        "first_run_terminal_relative_residual": row["solver"][
            "first_run_terminal_residual"
        ],
        "lambda": {
            "policy": "absolute source; no current normalisation applied",
            "terminal_amplitude": 1.0,
            "map_evaluation_amplitude_history": [1.0],
            "applied_amplitude_history": [1.0],
            "history_entry_count": 1,
            "history_semantics": (
                "the unpinned map applies unit amplitude at every evaluation; "
                "the banked receipt did not retain per-evaluation states"
            ),
        },
        "current": {
            "target_a": None,
            "unscaled_terminal_a": None,
            "achieved_terminal_a": None,
            "terminal_relative_error": None,
        },
        "whole_domain_psi_error": {
            "sup_wb": row["norms"]["psi"]["whole_domain"]["sup"],
            "rms_wb": row["norms"]["psi"]["whole_domain"]["rms"],
        },
        "magnetic_axis_position_error_m": row["geometry"][
            "magnetic_axis_position_error_m"
        ],
        "terminal_topology": row["geometry"]["root_topology"],
    }


def _measure(case_name: str, requested_cells: int) -> dict[str, Any]:
    configure_dtypes()
    cache = configure_persistent_compilation_cache(
        default_persistent_compilation_cache_root()
    )
    started = perf_counter()
    (
        machine,
        profile,
        operator,
        seed,
        oracle_state,
        target_current,
        inputs,
    ) = _prepare(case_name, requested_cells)
    _carrier, _source, exact = certificate._case(case_name)
    arms = {
        "unpinned": _banked_unpinned(case_name, requested_cells),
        "current_pinned": _score_arm(
            _solve_arm(
                profile,
                operator,
                seed,
                target_current,
                pinned=True,
                carrier_identity=f"solovev-current-pin:{case_name}:{requested_cells}",
            ),
            machine,
            operator,
            oracle_state,
            exact,
        ),
    }
    unpinned = arms["unpinned"]
    pinned = arms["current_pinned"]
    residual_ratio = pinned["terminal_relative_residual"] / max(
        unpinned["terminal_relative_residual"], np.finfo(float).tiny
    )
    rms_ratio = pinned["whole_domain_psi_error"]["rms_wb"] / max(
        unpinned["whole_domain_psi_error"]["rms_wb"], np.finfo(float).tiny
    )
    return {
        "case": case_name,
        "requested_cells": requested_cells,
        "realised_cells": len(machine.node),
        "machine_cache": machine.cache,
        "persistent_compilation_cache": cache.receipt(),
        "production_inputs": inputs,
        "arms": arms,
        "comparison": {
            "pinned_over_unpinned_terminal_residual": residual_ratio,
            "residual_improvement_orders": -math.log10(residual_ratio)
            if residual_ratio > 0.0
            else None,
            "pinned_over_unpinned_psi_rms": rms_ratio,
            "psi_rms_collapse_factor": 1.0 / rms_ratio if rms_ratio > 0.0 else None,
        },
        "lane": _lane(),
        "wall_seconds": perf_counter() - started,
    }


def _validate(receipt: dict[str, Any]) -> None:
    if receipt["schema"]["id"] != "nova.solovev-certificate-current-pin":
        raise RuntimeError("unexpected receipt schema")
    rows = receipt["comparisons"]
    expected = {(case, rung) for case in STATIC_CASES for rung in REQUESTED_CELLS}
    actual = {(row["case"], row["requested_cells"]) for row in rows}
    if actual != expected or len(rows) != len(expected):
        raise RuntimeError("the current-pin comparison matrix is incomplete")
    for row in rows:
        if set(row["arms"]) != set(ARM_NAMES):
            raise RuntimeError("a comparison arm is missing")
        for arm in row["arms"].values():
            required = (
                "terminal_relative_residual",
                "attempted_newton_trip_count",
                "termination",
                "lambda",
                "whole_domain_psi_error",
                "magnetic_axis_position_error_m",
            )
            if any(name not in arm for name in required):
                raise RuntimeError("an arm is missing required evidence")
            if not arm["lambda"]["map_evaluation_amplitude_history"]:
                raise RuntimeError("an arm has no lambda amplitude history")
    execution = receipt["execution"]
    if not execution["slurm_job_id"] or not execution["node"]:
        raise RuntimeError("the receipt does not identify its SLURM execution")
    if execution["exit_marker"] != EXIT_MARKER:
        raise RuntimeError("the successful exit marker is missing")


def _verdict(rows: list[dict[str, Any]]) -> dict[str, Any]:
    improved_by_orders = [
        row["comparison"]["residual_improvement_orders"] >= 2.0 for row in rows
    ]
    psi_collapsed = [
        row["comparison"]["pinned_over_unpinned_psi_rms"] <= 0.1 for row in rows
    ]
    recovered = all(improved_by_orders) and all(psi_collapsed)
    sentence = (
        "Pinning the exact net plasma current recovers convergence toward the "
        "closed-form root: every row lowers the residual by at least two orders "
        "of magnitude and collapses whole-domain psi RMS by at least tenfold."
        if recovered
        else "Pinning the exact net plasma current does not by itself recover "
        "convergence toward the closed-form root: not every row lowers the residual "
        "by two orders of magnitude and collapses whole-domain psi RMS tenfold."
    )
    return {
        "sentence": sentence,
        "recovers_convergence_toward_root": recovered,
        "rows_with_residual_improvement_of_at_least_two_orders": sum(
            improved_by_orders
        ),
        "rows_with_psi_rms_collapse_of_at_least_tenfold": sum(psi_collapsed),
        "row_count": len(rows),
    }


def _prewarm_machines() -> list[dict[str, Any]]:
    records = []
    for case_name in STATIC_CASES:
        carrier, _source, _exact = certificate._case(case_name)
        for requested_cells in REQUESTED_CELLS:
            machine = oracle_fixture.cached_machine(
                carrier,
                requested_cells,
                wall_nodes=oracle_fixture.WALL_POINT_COUNT,
            )
            records.append(
                {
                    "case": case_name,
                    "requested_cells": requested_cells,
                    "cache": machine.cache,
                }
            )
    return records


def _run_all(output: Path, work_dir: Path, workers: int) -> dict[str, Any]:
    if os.environ.get("JAX_PLATFORMS") != "cpu":
        raise RuntimeError("the comparison requires JAX_PLATFORMS=cpu")
    started = perf_counter()
    prewarm = _prewarm_machines()
    work_dir.mkdir(parents=True, exist_ok=True)
    jobs: list[tuple[str, int, Path]] = []
    for case_name in STATIC_CASES:
        for requested_cells in REQUESTED_CELLS:
            path = work_dir / f"{case_name}-{abs(requested_cells)}.json"
            jobs.append((case_name, requested_cells, path))

    def launch(item: tuple[str, int, Path]) -> Path:
        case_name, requested_cells, path = item
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--case",
            case_name,
            "--requested-cells",
            str(requested_cells),
            "--part-output",
            str(path),
        ]
        subprocess.run(command, cwd=ROOT, check=True)
        return path

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(launch, item) for item in jobs]
        part_paths = [future.result() for future in as_completed(futures)]
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in part_paths]
    rows.sort(key=lambda row: (row["case"], row["requested_cells"]))
    source_files = {
        "benchmarks/solovev_certificate.py": _sha256(
            ROOT / "benchmarks/solovev_certificate.py"
        ),
        "docs/figures/gs-absolute-accuracy/solovev-certificate.json": _sha256(
            ROOT / "docs/figures/gs-absolute-accuracy/solovev-certificate.json"
        ),
    }
    lane = _lane()
    receipt = {
        "schema": {
            "id": "nova.solovev-certificate-current-pin",
            "version": 1,
            "case_count": len(STATIC_CASES),
            "rung_count_per_case": len(REQUESTED_CELLS),
            "arms": list(ARM_NAMES),
        },
        "preregistration": {
            "cases": list(STATIC_CASES),
            "requested_cells": list(REQUESTED_CELLS),
            "shared_seed": "production current-centroid uniform-disc moment seed",
            "controlled_difference": (
                "only target_current supplied to ForwardFluxOperator.flux_map"
            ),
            "control_provenance": (
                "unpinned metrics are the immutable same-seed certificate rows; "
                "only the current-pinned intervention is rerun"
            ),
            "gauge": "shared exact exterior supply; no post-solve re-zeroing",
            "source_revision": _source_revision(),
            "solver_source_modified": False,
            "certificate_driver_modified": False,
            "original_certificate_modified": False,
            "source_file_sha256": source_files,
            "verdict_rule": (
                "recovery requires every row to improve terminal residual by at "
                "least two orders and reduce whole-domain psi RMS by at least tenfold"
            ),
        },
        "execution": {
            **lane,
            "worker_processes": workers,
            "launch_then_harvest": True,
            "machine_cache_prewarm": prewarm,
            "wall_seconds": perf_counter() - started,
            "exit_marker": EXIT_MARKER,
            "scheduler_harvest": None,
        },
        "comparisons": rows,
        "verdict": _verdict(rows),
    }
    _validate(receipt)
    _write_json(output, receipt)
    return receipt


def _harvest(output: Path, job_id: str) -> dict[str, Any]:
    receipt = json.loads(output.read_text(encoding="utf-8"))
    command = [
        "sacct",
        "-j",
        job_id,
        "-X",
        "-n",
        "-P",
        "-o",
        "JobIDRaw,State,ExitCode,Elapsed,ElapsedRaw,NodeList",
    ]
    lines = [
        line
        for line in subprocess.check_output(command, text=True).splitlines()
        if line
    ]
    fields = lines[0].split("|")
    if fields[0] != job_id:
        raise RuntimeError(f"scheduler harvest did not return job {job_id}")
    receipt["execution"]["scheduler_harvest"] = {
        "job_id": fields[0],
        "state": fields[1],
        "exit_code": fields[2],
        "elapsed": fields[3],
        "elapsed_seconds": int(fields[4]),
        "node": fields[5],
    }
    _validate(receipt)
    _write_json(output, receipt)
    return receipt


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=STATIC_CASES)
    parser.add_argument("--requested-cells", type=int, choices=REQUESTED_CELLS)
    parser.add_argument("--part-output", type=Path)
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--harvest-job")
    return parser.parse_args()


def main() -> None:
    arguments = _parse()
    if arguments.validate:
        receipt = json.loads(arguments.output.read_text(encoding="utf-8"))
        _validate(receipt)
        print(json.dumps(receipt["verdict"], sort_keys=True), flush=True)
        return
    if arguments.harvest_job:
        receipt = _harvest(arguments.output, arguments.harvest_job)
        print(json.dumps(receipt["execution"]["scheduler_harvest"], sort_keys=True))
        return
    if arguments.run_all:
        work_dir = arguments.work_dir or Path(
            tempfile.mkdtemp(prefix="solovev-current-pin-")
        )
        receipt = _run_all(arguments.output, work_dir, arguments.workers)
        print(receipt["verdict"]["sentence"], flush=True)
        print(EXIT_MARKER, flush=True)
        return
    if (
        arguments.case is None
        or arguments.requested_cells is None
        or arguments.part_output is None
    ):
        raise SystemExit("a case, requested-cell rung, and part output are required")
    row = _measure(arguments.case, arguments.requested_cells)
    _write_json(arguments.part_output, row)
    print(
        json.dumps(
            {
                "case": row["case"],
                "requested_cells": row["requested_cells"],
                "part": str(arguments.part_output),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
