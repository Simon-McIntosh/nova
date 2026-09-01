"""Attribute a banked active-set stagnation and probe its seed basin."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks import mast_response_carrier_warm as response_carrier
from benchmarks.diverted_basin_probe import PERTURBATION_AMPLITUDES
from benchmarks.efit_forward_parity_slice import (
    DECOMPOSITION_BANK,
    FIXED_POINT_CRITERION,
    GMRES_ITERATIONS,
    NEWTON_STEPS,
    RELAXATION,
    STEP_CAP,
    WARMUP_SWEEPS,
    _mast_case_from_selection,
    _passive_inclusive_case,
    select_slices_by_shot,
)
from benchmarks.label_seed_residual_field import _persisted_response_cache
from nova.equilibrium.fixed_point import FixedPointTerminationReason
from nova.equilibrium.forward import PerturbedSeedPolicy
from nova.equilibrium.topology import TopologyClass
from nova.imas.mast_solve_inputs import SHOT_STORE
from nova.jax.config import configure_dtypes


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BANK = (
    ROOT / "docs/figures/primary-xpoint-evidence/efit-topology-corroboration.json"
)
DEFAULT_OUTPUT = ROOT / "docs/figures/solver-convergence-regression/head-mechanism.json"
TARGETS = ((22086, 43), (21978, 35))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _termination_name(value: Any) -> str:
    code = int(np.asarray(value))
    try:
        return FixedPointTerminationReason(code).name.lower()
    except ValueError:
        return f"unknown_{code}"


def _executed(values: Any, iterations: int, dtype: Any) -> list[Any]:
    array = np.asarray(values, dtype=dtype).reshape(-1)[:iterations]
    if np.issubdtype(array.dtype, np.floating):
        return [float(value) if np.isfinite(value) else None for value in array]
    return array.tolist()


def _relative_residual(mapped: Any, state: Any) -> float:
    mapped_array = np.asarray(mapped, dtype=np.float64)
    state_array = np.asarray(state, dtype=np.float64)
    denominator = max(float(np.max(np.abs(mapped_array))), 1.0e-30)
    return float(np.max(np.abs(mapped_array - state_array)) / denominator)


def _strict_float(value: Any) -> float | None:
    number = float(np.asarray(value))
    return number if np.isfinite(number) else None


def _first_settled_trip(mask_differences: list[int]) -> int | None:
    for index in range(len(mask_differences)):
        if mask_differences[index] == 0 and not any(mask_differences[index:]):
            return index + 1
    return None


def bank_attribution(bank: Path = DEFAULT_BANK) -> dict[str, Any]:
    """Name the terminal convergence refusal from retained trip telemetry."""
    payload = json.loads(bank.read_text())
    rows = {
        (int(row["shot"]), int(row["slice_index"]), str(row["arm"])): row
        for row in payload["rows"]
    }
    target = rows[(22086, 43, "pure")]
    masks = [int(value) for value in target["active_set_mask_differences"]]
    residuals = [float(value) for value in target["active_set_residuals"]]
    tolerance = float(target["tolerance"])
    first_settled = _first_settled_trip(masks)
    if first_settled is None:
        raise RuntimeError("22086/43 pure has no settled-mask suffix")
    first_residual = residuals[first_settled - 1]
    terminal_residual = float(target["terminal_residual"])
    residual_plateau = bool(len(residuals) >= 2 and residuals[-1] == residuals[-2])
    if not (
        masks[first_settled - 1 :] == [0] * (len(masks) - first_settled + 1)
        and first_residual > tolerance
        and terminal_residual > tolerance
        and residual_plateau
        and target["termination_reason"] == "active_set_stagnated"
        and not target["converged"]
        and not target["qualified_terminal"]
    ):
        raise RuntimeError("the banked stagnation no longer has the declared shape")

    anomalous = []
    for shot, slice_index in TARGETS:
        row = rows[(shot, slice_index, "pure")]
        anomalous.append(
            {
                "identity": row["identity"],
                "arm": "pure",
                "active_set_iterations": int(row["active_set_iterations"]),
                "active_set_mask_differences": row["active_set_mask_differences"],
                "active_set_residuals": row["active_set_residuals"],
                "terminal_residual": row["terminal_residual"],
                "tolerance": row["tolerance"],
                "termination_reason": row["termination_reason"],
                "converged": row["converged"],
                "qualified_terminal": row["qualified_terminal"],
            }
        )

    return {
        "source": {
            "path": str(bank.relative_to(ROOT)),
            "sha256": _sha256(bank),
            "regeneration_revision": payload["regeneration_receipt"][
                "measurement_revision"
            ],
        },
        "refusing_criterion": "fixed_point_relative_sup_residual_threshold",
        "criterion_definition": "max(abs(F(x)-x)) / max(abs(F(x))) <= tolerance",
        "criterion_attribution": {
            "identity": "22086/43",
            "arm": "pure",
            "first_settled_trip_one_based": first_settled,
            "first_settled_residual": first_residual,
            "terminal_residual": terminal_residual,
            "registered_tolerance": tolerance,
            "first_settled_residual_over_tolerance": first_residual / tolerance,
            "terminal_residual_over_tolerance": terminal_residual / tolerance,
            "terminal_plateau_exactly_repeated": residual_plateau,
            "mask_settled_through_terminal": True,
            "termination_reason": target["termination_reason"],
            "solver_converged": bool(target["converged"]),
            "qualified_terminal": bool(target["qualified_terminal"]),
        },
        "excluded_alternatives": {
            "own_mask_acceptance": (
                "candidate admission is upstream of the terminal conjunction; "
                "the retained terminal has an unchanged mask but a residual above "
                "the registered tolerance"
            ),
            "qualification_seam": (
                "qualified_terminal is a downstream conjunction requiring the "
                "solver to report convergence, the same residual threshold, and "
                "the converged termination reason; it does not independently "
                "refuse this terminal"
            ),
        },
        "banked_anomalous_arms": anomalous,
    }


def _allocation() -> dict[str, Any]:
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        raise RuntimeError("measurement requires a reserved scheduler allocation")
    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "0"))
    reservation = os.environ.get("SLURM_JOB_RESERVATION", "")
    platforms = os.environ.get("JAX_PLATFORMS", "")
    if cpus != 4:
        raise RuntimeError(f"measurement requires exactly four CPUs, received {cpus}")
    if reservation != "gpu_0003_grpA":
        raise RuntimeError(f"unexpected reservation {reservation!r}")
    if platforms != "cuda,cpu":
        raise RuntimeError(f"JAX_PLATFORMS must be cuda,cpu, received {platforms!r}")
    gpu = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    if "H200" not in gpu:
        raise RuntimeError(f"measurement requires an H200, received {gpu!r}")
    return {
        "job_id": int(job_id),
        "job_name": os.environ.get("SLURM_JOB_NAME"),
        "node": os.environ.get("SLURMD_NODENAME", socket.gethostname()),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "reservation": reservation,
        "allocated_cpus": cpus,
        "allocated_gpus": int(os.environ.get("SLURM_GPUS_ON_NODE", "1")),
        "gpu": gpu,
        "tmpdir": os.environ.get("TMPDIR"),
        "jax_platforms": platforms.split(","),
        "jax_cuda_devices": [str(device) for device in jax.devices("gpu")],
        "jax_cpu_devices": [str(device) for device in jax.devices("cpu")],
    }


def _prepare_reference(selected_row, qualification, response_cache):
    case, context = _mast_case_from_selection(SHOT_STORE, selected_row, qualification)
    passive_case, profile, policy = _passive_inclusive_case(
        case, context, response_cache
    )
    if int(policy["section_kernel_evaluations_this_shot"]) != 0:
        raise RuntimeError("mechanism probe entered a direct response builder")
    reference = passive_case["reference"]
    return (
        profile,
        jnp.asarray(passive_case["state"]),
        abs(float(reference["plasma_current_a"])),
        {
            "shot": int(reference["shot"]),
            "slice_index": int(reference["slice_index"]),
            "time_s": float(reference["time_s"]),
            "seed_sha256": _array_sha256(passive_case["state"]),
        },
    )


def _pure_arm(profile, seed, target_current: float, *, telemetry: bool):
    started = time.perf_counter()
    portfolio = profile.solve_portfolio(
        jnp.stack((seed, seed)),
        route="newton_krylov",
        target_current=target_current,
        tolerance=FIXED_POINT_CRITERION,
        newton_steps=NEWTON_STEPS,
        gmres_iterations=GMRES_ITERATIONS,
        warmup=WARMUP_SWEEPS,
        relaxation=RELAXATION,
        step_cap=STEP_CAP,
        stream_active_set=telemetry,
    )
    portfolio.branches.equilibrium.flux.block_until_ready()
    jax.effects_barrier()
    elapsed = time.perf_counter() - started
    branch = jax.tree.map(
        lambda value: value[int(TopologyClass.DIVERTED)], portfolio.branches
    )
    fixed = branch.equilibrium.fixed_point
    iterations = int(np.asarray(fixed.active_set_iterations))
    return {
        "telemetry_compiled_in": telemetry,
        "wall_seconds_including_compilation": elapsed,
        "solver_converged": bool(np.asarray(fixed.converged)),
        "branch_qualified_converged": bool(np.asarray(branch.converged)),
        "finite_state": bool(np.asarray(branch.equilibrium.finite.passed)),
        "topology_consistent": bool(np.asarray(branch.topology_consistent)),
        "terminal_residual": float(np.asarray(fixed.residual)),
        "termination_reason": _termination_name(fixed.termination_reason),
        "active_set_iterations": iterations,
        "active_set_residuals": _executed(
            fixed.active_set_residuals, iterations, np.float64
        ),
        "active_set_mask_differences": _executed(
            fixed.active_set_mask_differences, iterations, np.int64
        ),
        "active_set_cycle_damping_activations": _executed(
            fixed.active_set_cycle_damping_activations, iterations, np.int64
        ),
        "terminal_state_sha256": _array_sha256(branch.equilibrium.flux),
    }


def _telemetry_comparison(profile, seed, target_current: float) -> dict[str, Any]:
    jax.clear_caches()
    gc.collect()
    compiled_out = _pure_arm(profile, seed, target_current, telemetry=False)
    jax.clear_caches()
    gc.collect()
    compiled_in = _pure_arm(profile, seed, target_current, telemetry=True)
    numeric_keys = (
        "solver_converged",
        "branch_qualified_converged",
        "finite_state",
        "topology_consistent",
        "terminal_residual",
        "termination_reason",
        "active_set_iterations",
        "active_set_residuals",
        "active_set_mask_differences",
        "active_set_cycle_damping_activations",
        "terminal_state_sha256",
    )
    differences = {
        key: {"compiled_out": compiled_out[key], "compiled_in": compiled_in[key]}
        for key in numeric_keys
        if compiled_out[key] != compiled_in[key]
    }
    restores = bool(
        compiled_out["solver_converged"] and not compiled_in["solver_converged"]
    )
    return {
        "compiled_out": compiled_out,
        "compiled_in": compiled_in,
        "numeric_receipts_equal": not differences,
        "numeric_differences": differences,
        "telemetry_out_restores_convergence": restores,
        "telemetry_effect": (
            "restores_convergence"
            if restores
            else "no_numeric_effect"
            if not differences
            else "changes_numeric_receipt_without_restoring_convergence"
        ),
    }


def _perturbed_seed_probe(profile, seed, target_current: float) -> dict[str, Any]:
    mapped = profile.flux_map(
        requested_class=TopologyClass.DIVERTED,
        target_current=target_current,
    )
    mapped_seed = jax.block_until_ready(mapped(seed))
    direction = mapped_seed - seed
    policy = PerturbedSeedPolicy(
        relative_amplitudes=PERTURBATION_AMPLITUDES,
        newton_steps=NEWTON_STEPS,
        gmres_iterations=GMRES_ITERATIONS,
        tolerance=FIXED_POINT_CRITERION,
    )
    jax.clear_caches()
    gc.collect()
    started = time.perf_counter()
    solved = profile.solve_diverted_perturbations(
        seed,
        direction,
        policy,
        target_current=target_current,
    )
    jax.block_until_ready(solved)
    elapsed = time.perf_counter() - started
    mapped_seeds = jax.block_until_ready(jax.vmap(mapped)(solved.seed_flux))
    initial_residuals = [
        _relative_residual(mapped_state, state)
        for mapped_state, state in zip(mapped_seeds, solved.seed_flux, strict=True)
    ]
    rows = []
    for index, amplitude in enumerate(np.asarray(solved.relative_amplitude)):
        rung = jax.tree.map(lambda value: value[index], solved.rungs)
        fixed = rung.equilibrium.fixed_point
        iterations = int(np.asarray(fixed.active_set_iterations))
        terminal_residual = _strict_float(fixed.residual)
        rows.append(
            {
                "relative_amplitude": float(amplitude),
                "initial_residual": _strict_float(initial_residuals[index]),
                "terminal_residual": terminal_residual,
                "terminal_residual_nonfinite": terminal_residual is None,
                "solver_converged": bool(np.asarray(fixed.converged)),
                "branch_qualified_converged": bool(np.asarray(rung.converged)),
                "topology_consistent": bool(np.asarray(rung.topology_consistent)),
                "achieved_branch": (
                    "diverted" if int(np.asarray(rung.achieved_class)) else "limited"
                ),
                "termination_reason": _termination_name(fixed.termination_reason),
                "active_set_iterations": iterations,
                "active_set_residuals": _executed(
                    fixed.active_set_residuals, iterations, np.float64
                ),
                "active_set_mask_differences": _executed(
                    fixed.active_set_mask_differences, iterations, np.int64
                ),
                "terminal_relative_distance_from_label": _strict_float(
                    solved.root_relative_error[index]
                ),
                "terminal_state_sha256": _array_sha256(rung.equilibrium.flux),
            }
        )
    terminal_reasons = sorted({row["termination_reason"] for row in rows})
    terminal_digests = {row["terminal_state_sha256"] for row in rows}
    terminal_residuals = [
        row["terminal_residual"] for row in rows if row["terminal_residual"] is not None
    ]
    convergence_outcomes = {row["solver_converged"] for row in rows}
    topology_outcomes = {row["achieved_branch"] for row in rows}
    return {
        "direction": "production map residual F(seed)-seed",
        "relative_amplitudes": list(PERTURBATION_AMPLITUDES),
        "wall_seconds_including_compilation": elapsed,
        "rows": rows,
        "summary": {
            "seed_count": len(rows),
            "solver_converged_count": sum(row["solver_converged"] for row in rows),
            "qualified_converged_count": sum(
                row["branch_qualified_converged"] for row in rows
            ),
            "diverted_terminal_count": sum(
                row["achieved_branch"] == "diverted" for row in rows
            ),
            "distinct_terminal_reason_count": len(terminal_reasons),
            "terminal_reasons": terminal_reasons,
            "distinct_terminal_state_count": len(terminal_digests),
            "finite_terminal_residual_count": len(terminal_residuals),
            "nonfinite_terminal_residual_count": len(rows) - len(terminal_residuals),
            "terminal_residual_min": (
                min(terminal_residuals) if terminal_residuals else None
            ),
            "terminal_residual_max": (
                max(terminal_residuals) if terminal_residuals else None
            ),
            "convergence_outcomes": sorted(convergence_outcomes),
            "achieved_branches": sorted(topology_outcomes),
            "categorically_basin_sensitive": bool(
                len(terminal_reasons) > 1
                or len(convergence_outcomes) > 1
                or len(topology_outcomes) > 1
            ),
            "basin_sensitive": bool(
                len(terminal_reasons) > 1 or len(terminal_digests) > 1
            ),
        },
    }


def measure(bank: Path = DEFAULT_BANK) -> dict[str, Any]:
    """Run the telemetry and perturbed-seed discriminators on one H200."""
    allocation = _allocation()
    configure_dtypes()
    response_cache, carrier_evidence = _persisted_response_cache(
        response_carrier.DEFAULT_CARRIER, response_carrier.DEFAULT_RECEIPT
    )
    selected = {
        (int(row["shot"]), int(row["slice_index"])): (row, qualification)
        for row, qualification in select_slices_by_shot(DECOMPOSITION_BANK)
    }
    missing = set(TARGETS).difference(selected)
    if missing:
        raise RuntimeError(f"frozen selection lacks targets {sorted(missing)}")

    profile, seed, current, telemetry_reference = _prepare_reference(
        *selected[(22086, 43)], response_cache
    )
    telemetry = _telemetry_comparison(profile, seed, current)
    print("perturbed-seed probe 22086/43 pure", flush=True)
    perturbed = [
        {
            "reference": telemetry_reference,
            "arm": "pure",
            "probe": _perturbed_seed_probe(profile, seed, current),
        }
    ]
    del profile, seed
    jax.clear_caches()
    gc.collect()

    for target in TARGETS[1:]:
        profile, seed, current, reference = _prepare_reference(
            *selected[target], response_cache
        )
        print(f"perturbed-seed probe {target[0]}/{target[1]} pure", flush=True)
        perturbed.append(
            {
                "reference": reference,
                "arm": "pure",
                "probe": _perturbed_seed_probe(profile, seed, current),
            }
        )
        jax.clear_caches()
        gc.collect()

    receipt = {
        "receipt": "MAST pure-arm stagnation mechanism probe",
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "bank_attribution": bank_attribution(bank),
        "allocation": allocation,
        "execution_contract": {
            "telemetry_toggle": (
                "stream_active_set is a static Python branch, with compilation "
                "caches cleared between the compiled-out and compiled-in solves"
            ),
            "same_arm": "22086/43 pure production diverted portfolio branch",
            "solver": {
                "newton_steps": NEWTON_STEPS,
                "gmres_iterations": GMRES_ITERATIONS,
                "warmup_sweeps": WARMUP_SWEEPS,
                "relaxation": RELAXATION,
                "step_cap": STEP_CAP,
                "fixed_point_tolerance": FIXED_POINT_CRITERION,
            },
            "perturbation_relative_amplitudes": list(PERTURBATION_AMPLITUDES),
            "persisted_response_carrier": carrier_evidence,
            "direct_green_operator_builder_entries": 0,
            "solver_source_modified": False,
        },
        "telemetry_discriminator": {
            "reference": telemetry_reference,
            "arm": "pure",
            **telemetry,
        },
        "perturbed_seed_outcomes": perturbed,
    }
    check(receipt)
    return receipt


def check(receipt: dict[str, Any]) -> None:
    """Fail closed unless all three mechanism measurements are retained."""
    attribution = receipt["bank_attribution"]
    if attribution["refusing_criterion"] != (
        "fixed_point_relative_sup_residual_threshold"
    ):
        raise RuntimeError("the refusing criterion is not explicitly attributed")
    numbers = attribution["criterion_attribution"]
    if not numbers["mask_settled_through_terminal"]:
        raise RuntimeError("the banked mask did not settle")
    if numbers["terminal_residual_over_tolerance"] <= 1.0:
        raise RuntimeError("the terminal residual no longer refuses convergence")
    telemetry = receipt["telemetry_discriminator"]
    if not isinstance(telemetry["telemetry_out_restores_convergence"], bool):
        raise RuntimeError("telemetry restoration verdict is absent")
    if set(telemetry) < {
        "compiled_out",
        "compiled_in",
        "numeric_receipts_equal",
    }:
        raise RuntimeError("the telemetry comparison is incomplete")
    probes = receipt["perturbed_seed_outcomes"]
    identities = {
        (row["reference"]["shot"], row["reference"]["slice_index"]) for row in probes
    }
    if identities != set(TARGETS):
        raise RuntimeError("the perturbed-seed cohort differs from the two anomalies")
    for row in probes:
        probe = row["probe"]
        if probe["summary"]["seed_count"] != len(PERTURBATION_AMPLITUDES):
            raise RuntimeError("a perturbed-seed ladder is incomplete")
        if len(probe["rows"]) != len(PERTURBATION_AMPLITUDES):
            raise RuntimeError("a perturbed-seed outcome row is missing")
    if receipt["execution_contract"]["solver_source_modified"]:
        raise RuntimeError("the measurement cannot modify solver source")


def _write_atomic(path: Path, receipt: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("analyze", "measure", "check"))
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    if arguments.command == "analyze":
        print(json.dumps(bank_attribution(arguments.bank), indent=2))
        return
    if arguments.command == "measure":
        receipt = measure(arguments.bank)
        _write_atomic(arguments.output, receipt)
    else:
        receipt = json.loads(arguments.output.read_text())
        check(receipt)
    telemetry = receipt["telemetry_discriminator"]
    basins = {
        f"{row['reference']['shot']}/{row['reference']['slice_index']}": row["probe"][
            "summary"
        ]
        for row in receipt["perturbed_seed_outcomes"]
    }
    print(
        "STAGNATION_MECHANISM "
        f"criterion={receipt['bank_attribution']['refusing_criterion']} "
        f"telemetry_out_restores={telemetry['telemetry_out_restores_convergence']} "
        f"telemetry_equal={telemetry['numeric_receipts_equal']} "
        f"basins={basins} PASS"
    )


if __name__ == "__main__":
    main()
