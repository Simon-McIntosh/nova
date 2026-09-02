"""Reproduce selected MAST pure arms at the bank-producing revision."""

from __future__ import annotations

import argparse
import gc
import hashlib
from importlib.util import module_from_spec, spec_from_file_location
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TRUE_REVISION = "d47f7cd1ce1c72138b6691c559bbf1be0bcd465e"
WINDOW_END = "a4bec44f5cbf80ad5e210c01c984ac8d02a89de9"
TARGETS = ((22086, 43), (21978, 35))
CARRIER_FILE_SHA256 = "1da2b7bdb4a79d6b81513fa4aba909d318bd157c9b5453340b122bfd595428c9"
CARRIER_SEMANTIC_IDENTITY = (
    "1d2c4a2b2f448ab8f1ae981031bbaf85fe4ee87f8ed9606fe6847d0fc9f1e994"
)
DEFAULT_SHADOW_ROOT = ROOT.parent / "scr-true-pin-reproduction-shadow"
DEFAULT_COMPARISON = (
    ROOT / "docs/figures/solver-convergence-regression/backend-divergence.json"
)
DEFAULT_TRUE_PIN = (
    ROOT / "docs/figures/solver-convergence-regression/true-pin-reproduction.json"
)
TRAJECTORY_REVISION = "4ee90ece25ad47cc655dc4531249da24d13763e1"
MODEL_TRUST_REVISION = "565a8c8eec80a0e0d5fb8c0122c0180de4f2c3ed"
OWN_MASK_REVISION = "a2e65fda4c860edd261fee6a0c7af8f7083e4a7b"
NULL_POLISH_MERGE = "aecea6a7d8913b105b5ab15317280f167d818cf2"
PRE_POLISH_REVISION = "69fc5e3ea38cf7e550efee13a5c55dc9f14250d3"
COMMITTED_SADDLES = {
    "22086/43 pure": (0.5904799259797128, 1.2250633080352826),
    "21978/35 pure": (0.5628646606371921, 1.2634878586583973),
}
DEFAULT_ROOT_CAUSE = ROOT / "docs/figures/solver-convergence-regression/root-cause.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(root: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *arguments], text=True
    ).strip()


def _validate_shadow(
    shadow_root: Path, expected_revision: str = TRUE_REVISION
) -> dict[str, Any]:
    head = _git(shadow_root, "rev-parse", "HEAD")
    if head != expected_revision:
        raise RuntimeError(
            f"shadow revision {head} does not equal expected revision "
            f"{expected_revision}"
        )
    solver_changed = subprocess.run(
        ["git", "-C", str(shadow_root), "diff", "--quiet", head, "--", "nova"],
        check=False,
    ).returncode
    if solver_changed:
        raise RuntimeError("bank-revision shadow contains solver changes")
    return {
        "revision": head,
        "shadow_root": str(shadow_root),
        "solver_source_modified": False,
    }


def _configure_shadow_import(shadow_root: Path) -> None:
    resolved = str(shadow_root.resolve())
    sys.path = [entry for entry in sys.path if Path(entry or ".").resolve() != ROOT]
    if resolved in sys.path:
        sys.path.remove(resolved)
    sys.path.insert(0, resolved)


def _corroboration_module(shadow_root: Path):
    source = (
        shadow_root
        / "docs/figures/primary-xpoint-evidence/efit_topology_corroboration.py"
    )
    spec = spec_from_file_location("bank_corroboration", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load bank corroboration driver {source}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _allocation(expected_cpus: int = 4) -> dict[str, Any]:
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id is None:
        raise RuntimeError("capture requires a scheduler allocation")
    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "0"))
    platforms = os.environ.get("JAX_PLATFORMS", "")
    reservation = os.environ.get("SLURM_JOB_RESERVATION", "")
    if cpus != expected_cpus:
        raise RuntimeError(f"capture requires {expected_cpus} CPUs, received {cpus}")
    if platforms != "cuda,cpu":
        raise RuntimeError(
            f"capture requires JAX_PLATFORMS=cuda,cpu, received {platforms!r}"
        )
    if reservation != "gpu_0003_grpA":
        raise RuntimeError(f"unexpected H200 reservation {reservation!r}")
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,uuid", "--format=csv,noheader"],
        text=True,
    ).strip()
    if "H200" not in gpu:
        raise RuntimeError(f"capture requires an H200, received {gpu!r}")
    return {
        "job_id": int(job_id),
        "job_name": os.environ.get("SLURM_JOB_NAME"),
        "node": os.environ.get("SLURMD_NODENAME", socket.gethostname()),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "reservation": reservation,
        "allocated_cpus": cpus,
        "allocated_gpus": int(os.environ.get("SLURM_GPUS_ON_NODE", "0")),
        "tmpdir": os.environ.get("TMPDIR"),
        "jax_platforms": platforms.split(","),
        "gpu": gpu,
        "python": sys.executable,
    }


def _strict_floats(values: Any, count: int) -> list[float] | None:
    if values is None or count <= 0:
        return None
    array = np.asarray(values, dtype=np.float64).reshape(-1)[:count]
    finite = array[np.isfinite(array)]
    return finite.tolist() if finite.size else None


def _stationary_point_polish_receipt(operator, state) -> dict[str, Any] | None:
    """Return the final qualified O/X polish attempt when the revision exposes it."""
    from nova.equilibrium.topology import TopologyClass, require_qualified_axis

    physical = state[: operator.physical_node_number]
    requested_class = int(TopologyClass.DIVERTED)
    topology = operator._fixed_design_topology
    initial = topology.read_qualification(
        physical,
        operator.polarity,
        operator.inside_material,
        requested_class,
    )
    _seed, material = operator.connectivity_axis_seed(initial.state.axis)
    result = topology.read_qualification(
        physical,
        operator.polarity,
        material,
        requested_class,
    )
    require_qualified_axis(initial.axis_admitted & result.axis_admitted)
    if not hasattr(result, "polish_receipt"):
        return None

    psi_grid, psi_wall = topology.split_flux_map(physical)
    candidates_o, candidates_x = topology.grid(psi_grid)
    wall = topology.wall(psi_wall, operator.polarity)
    qualified_o = topology.qualified_o_candidates(
        candidates_o,
        candidates_x,
        wall,
        operator.polarity,
        psi_grid,
        material,
    )
    axis = topology.o_point_qualification(
        candidates_o, operator.polarity, qualified_o
    ).data
    saddle = topology.x_point_data(candidates_x, operator.polarity, axis[2])
    seed_values = np.asarray((axis[2], saddle[2]), dtype=np.float64)
    receipt = result.polish_receipt

    slots = {}
    for index, name in enumerate(("o", "x")):
        seed_value = float(seed_values[index])
        polished_value = float(np.asarray(receipt["value"])[index])
        selected_value = float(np.asarray(receipt["selected_value"])[index])
        slots[name] = {
            "accepted": bool(np.asarray(receipt["converged"])[index]),
            "seed_position_rz_m": np.asarray(
                receipt["census_position_rz"], dtype=np.float64
            )[index].tolist(),
            "polished_position_rz_m": np.asarray(
                receipt["position_rz"], dtype=np.float64
            )[index].tolist(),
            "selected_position_rz_m": np.asarray(
                receipt["selected_position_rz"], dtype=np.float64
            )[index].tolist(),
            "seed_value_wb": seed_value,
            "polished_value_wb": polished_value,
            "selected_value_wb": selected_value,
            "polished_minus_seed_value_wb": polished_value - seed_value,
            "normalized_gradient": float(
                np.asarray(receipt["normalized_gradient"])[index]
            ),
            "roundoff_floor": float(np.asarray(receipt["roundoff_floor"])[index]),
            "representation_floor": float(
                np.asarray(receipt["representation_floor"])[index]
            ),
        }
    return {
        "slot_order": ["o", "x"],
        "slots": slots,
    }


def _solve_pure_arm(
    profile,
    seed,
    target_current: float,
    corroboration,
) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp

    from benchmarks.efit_forward_parity_slice import (
        FIXED_POINT_CRITERION,
        GMRES_ITERATIONS,
        NEWTON_STEPS,
        RELAXATION,
        STEP_CAP,
        WARMUP_SWEEPS,
    )
    from nova.equilibrium.fixed_point import FixedPointTerminationReason
    from nova.equilibrium.topology import TopologyClass

    initial = jnp.stack((seed, seed))
    started = time.perf_counter()
    portfolio = profile.solve_portfolio(
        initial,
        route="newton_krylov",
        target_current=target_current,
        tolerance=FIXED_POINT_CRITERION,
        newton_steps=NEWTON_STEPS,
        gmres_iterations=GMRES_ITERATIONS,
        warmup=WARMUP_SWEEPS,
        relaxation=RELAXATION,
        step_cap=STEP_CAP,
    )
    portfolio.branches.equilibrium.flux.block_until_ready()
    elapsed = time.perf_counter() - started
    branch = jax.tree.map(
        lambda value: value[int(TopologyClass.DIVERTED)], portfolio.branches
    )
    fixed = branch.equilibrium.fixed_point
    reason_value = int(np.asarray(fixed.termination_reason))
    try:
        reason = FixedPointTerminationReason(reason_value).name.lower()
    except ValueError:
        reason = f"unknown_{reason_value}"
    state = branch.equilibrium.flux
    _masks, topology = profile.operator.read(state)
    geometry = corroboration._post_cutover_geometry(profile, state, topology)
    polish_receipt = _stationary_point_polish_receipt(profile.operator, state)
    trip_count = int(np.asarray(getattr(fixed, "active_set_iterations", 0)))
    return {
        "converged": bool(np.asarray(branch.converged)),
        "terminal_residual": float(np.asarray(branch.residual)),
        "termination_reason": reason,
        "trip_count": trip_count,
        "active_set_residuals": _strict_floats(
            getattr(fixed, "active_set_residuals", None), trip_count
        ),
        "active_set_mask_differences": (
            np.asarray(
                getattr(fixed, "active_set_mask_differences", []), dtype=np.int64
            )
            .reshape(-1)[:trip_count]
            .tolist()
            if trip_count
            else None
        ),
        "selected_saddle_m": np.asarray(geometry["selected_saddle"], dtype=np.float64)[
            :2
        ].tolist(),
        "topology_qualification_polish_receipt": polish_receipt,
        "solve_wall_seconds_including_compilation": elapsed,
        "solver": {
            "fixed_point_tolerance": FIXED_POINT_CRITERION,
            "newton_steps": NEWTON_STEPS,
            "gmres_iterations": GMRES_ITERATIONS,
            "warmup_sweeps": WARMUP_SWEEPS,
            "relaxation": RELAXATION,
            "step_cap": STEP_CAP,
        },
    }


def capture(
    shadow_root: Path,
    *,
    expected_revision: str = TRUE_REVISION,
    targets: tuple[tuple[int, int], ...] = TARGETS,
    expected_cpus: int = 4,
) -> dict[str, Any]:
    source = _validate_shadow(shadow_root, expected_revision)
    _configure_shadow_import(shadow_root)
    os.chdir(shadow_root)

    import jax
    import jax.numpy as jnp

    from benchmarks import mast_response_carrier_warm as response_carrier
    from benchmarks.efit_forward_parity_slice import (
        DECOMPOSITION_BANK,
        _mast_case_from_selection,
        _passive_inclusive_case,
        select_slices_by_shot,
    )
    from benchmarks.label_seed_residual_field import _persisted_response_cache
    from nova.imas.mast_vacuum_cohort import SHOT_STORE
    from nova.jax.config import configure_dtypes

    configure_dtypes()
    if jax.default_backend() != "gpu":
        raise RuntimeError(
            f"capture requires the GPU backend, selected {jax.default_backend()!r}"
        )
    allocation = _allocation(expected_cpus)
    response_cache, carrier_evidence = _persisted_response_cache(
        response_carrier.DEFAULT_CARRIER, response_carrier.DEFAULT_RECEIPT
    )
    carrier = carrier_evidence.get("carrier", carrier_evidence)
    if carrier["file_sha256"] != CARRIER_FILE_SHA256:
        raise RuntimeError("persisted carrier file digest does not match the bank")
    if carrier["semantic_response_identity"] != CARRIER_SEMANTIC_IDENTITY:
        raise RuntimeError(
            "persisted carrier semantic identity does not match the bank"
        )
    selected = {
        (int(row["shot"]), int(row["slice_index"])): (row, qualification)
        for row, qualification in select_slices_by_shot(DECOMPOSITION_BANK)
    }
    bank = json.loads(
        (
            shadow_root
            / "docs/figures/primary-xpoint-evidence/efit-topology-corroboration.json"
        ).read_text()
    )
    committed = {
        f"{row['shot']}/{row['slice_index']} pure": {
            "converged": row["converged"],
            "terminal_residual": row["terminal_residual"],
            "termination_reason": row["termination_reason"],
            "selected_saddle_m": row["nova_selected_saddle_m"],
        }
        for row in bank["rows"]
        if row.get("arm") == "pure"
        and (int(row["shot"]), int(row["slice_index"])) in targets
    }
    corroboration = _corroboration_module(shadow_root)
    arms = {}
    for target in targets:
        row, qualification = selected[target]
        case, context = _mast_case_from_selection(SHOT_STORE, row, qualification)
        passive_case, profile, policy = _passive_inclusive_case(
            case, context, response_cache
        )
        if int(policy["section_kernel_evaluations_this_shot"]) != 0:
            raise RuntimeError("historical reproduction entered a direct builder")
        reference = passive_case["reference"]
        key = f"{target[0]}/{target[1]} pure"
        arms[key] = {
            "reference": {
                "shot": int(reference["shot"]),
                "slice_index": int(reference["slice_index"]),
                "time_s": float(reference["time_s"]),
            },
            "solve": _solve_pure_arm(
                profile,
                jnp.asarray(passive_case["state"]),
                abs(float(reference["plasma_current_a"])),
                corroboration,
            ),
        }
        jax.clear_caches()
        gc.collect()
    return {
        "receipt": "bank-revision MAST pure-arm capture",
        "source": source,
        "allocation": {
            **allocation,
            "jax_backend": jax.default_backend(),
            "jax_devices": [str(device) for device in jax.devices()],
        },
        "carrier": carrier,
        "committed_rows": committed,
        "arms": arms,
        "execution_contract": {
            "same_bank_generation_route": (
                "passive-inclusive case plus the production solve_portfolio pure arm"
            ),
            "solver_source_modified": False,
            "targets": [list(target) for target in targets],
        },
    }


def _target(value: str) -> tuple[int, int]:
    try:
        shot, slice_index = value.split("/", 1)
        return int(shot), int(slice_index)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            f"target must use SHOT/SLICE integers, received {value!r}"
        ) from error


def _candidate_window() -> dict[str, Any]:
    command = [
        "git",
        "log",
        "--format=%H%x09%s",
        f"{TRUE_REVISION}..{WINDOW_END}",
        "--",
        "nova/equilibrium",
    ]
    verbatim = subprocess.check_output(command, cwd=ROOT, text=True).rstrip("\n")
    lines = verbatim.splitlines()
    if len(lines) != 10:
        raise RuntimeError(f"candidate window contains {len(lines)} commits, not ten")
    return {
        "command": " ".join(command),
        "verbatim": verbatim,
        "count": len(lines),
        "commits": [
            {"sha": line.split("\t", 1)[0], "subject": line.split("\t", 1)[1]}
            for line in lines
        ],
    }


def _verified_hunk(commit: str, function: str, quote: str) -> dict[str, str]:
    command = [
        "git",
        "show",
        "--format=",
        commit,
        "--",
        "nova/equilibrium/fixed_point.py",
    ]
    diff = subprocess.check_output(command, cwd=ROOT, text=True)
    source_lines = "\n".join(
        line[1:] for line in diff.splitlines() if line.startswith(("+", " "))
    )
    if quote not in source_lines:
        raise RuntimeError(f"quoted hunk is absent from {commit} in {function}")
    return {
        "commit": commit,
        "function": function,
        "command": " ".join(command),
        "quote": quote,
    }


def _root_cause_reading() -> list[dict[str, Any]]:
    classifications = {
        "6f08fb19fa4790449d9ff2b652ea488f6ccf9d53": (
            "telemetry only; disabled by default and outside solver decisions",
            ["_active_set_newton_krylov", "newton_krylov"],
        ),
        "ce686206627a20b2797b70145859a6a42350c90c": (
            "no fixed_point.py hunk; wall-height eligibility only",
            [],
        ),
        "ebb658bd0d6679bba604cef1e554c08d6a3a38ab": (
            "no fixed_point.py hunk; wall-height reporting only",
            [],
        ),
        "420feb23dc040ae1d0a2f874cff4e93c8f023030": (
            "terminal stop only; detects the repeated settled residual but cannot "
            "cause the earlier trip-2 state difference",
            ["_active_set_newton_krylov", "newton_krylov"],
        ),
        "7fc24f01064d37633ed3424f9850aa763ad38f1b": (
            "inner diagnostics plus an explicit GMRES tolerance equal to the prior "
            "library default; excluded by the later trajectory-boundary capture",
            ["_newton_krylov_inner", "_qualified_krylov_step"],
        ),
        "736e2553ffe18f0e146809ca984a1cf3e4647714": (
            "retains an incoming state only when selected_difference is zero, so it "
            "cannot itself reject the banked trip-4 one-cell reopening",
            ["_active_set_newton_krylov"],
        ),
        TRAJECTORY_REVISION: (
            "resumes the terminal Newton trajectory after an unchanged mask instead "
            "of restarting from the reported best fallback",
            ["_active_set_newton_krylov", "_newton_krylov_inner"],
        ),
        "9c3c518fdddbe9f00d1c0d350cd2b475b60ff8b8": (
            "continues merit and recovery state only after the trajectory continuation "
            "is already eligible; cannot affect trip 2",
            ["_active_set_newton_krylov", "_newton_krylov_inner"],
        ),
        MODEL_TRUST_REVISION: (
            "requires realized merit decrease to substantiate the frozen local model "
            "before selecting a promotion family",
            ["_backtracked_promotion", "_newton_krylov_inner"],
        ),
        OWN_MASK_REVISION: (
            "scores candidates and incumbents on their induced masks and requires a "
            "strict own-mask residual improvement",
            ["_backtracked_promotion", "_newton_krylov_inner"],
        ),
    }
    records = []
    for commit in reversed(_candidate_window()["commits"]):
        sha = commit["sha"]
        changed = _git(ROOT, "show", "--format=", "--name-only", sha).splitlines()
        classification, functions = classifications[sha]
        records.append(
            {
                **commit,
                "fixed_point_touched": "nova/equilibrium/fixed_point.py" in changed,
                "functions": functions,
                "reading": classification,
            }
        )
    return records


def _captured_arm(raw: dict[str, Any]) -> dict[str, Any]:
    arm = raw["arms"]["22086/43 pure"]
    return {
        "revision": raw["source"]["revision"],
        "shadow_root": raw["source"]["shadow_root"],
        "solve": arm["solve"],
        "allocation": raw["allocation"],
    }


def _plot_root_cause(receipt: dict[str, Any], figure: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(11.5, 6.8), constrained_layout=True)
    styles = (
        ("banked_old", "d47f7cd1 banked convergence", "o", "-", 2.4),
        ("after_trajectory", "4ee90ece trajectory continuation", "s", "--", 1.8),
        ("after_model_trust", "565a8c8e model-trust selection", "D", "--", 1.8),
        ("banked_new", "a4bec44f own-mask policy", "^", "-", 2.4),
    )
    for key, label, marker, linestyle, linewidth in styles:
        history = receipt["histories"][key]["active_set_residuals"]
        axis.semilogy(
            range(1, len(history) + 1),
            history,
            marker=marker,
            linestyle=linestyle,
            linewidth=linewidth,
            label=label,
        )
    axis.axhline(
        receipt["histories"]["banked_old"]["terminal_residual"],
        color="black",
        linewidth=1.1,
        linestyle=":",
        label="committed converged residual",
    )
    axis.axvline(2, color="0.75", linewidth=1.0, linestyle=":")
    axis.axvline(4, color="0.75", linewidth=1.0, linestyle=":")
    axis.text(2.05, 1.7e-3, "promotion paths separate", fontsize=8)
    axis.text(4.05, 1.1e-5, "banked mask re-opens", fontsize=8)
    axis.set_xlabel("active-set trip")
    axis.set_ylabel("relative sup residual")
    axis.set_title("22086/43 pure: policy changes across the localized commit window")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(fontsize=8)
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure, dpi=180)
    plt.close(fig)


def compile_root_cause(
    trajectory_raw_path: Path,
    model_trust_raw_path: Path,
    true_pin_path: Path,
    later_path: Path,
    output: Path,
    figure: Path,
    job_elapsed: str,
    exit_marker: int,
) -> dict[str, Any]:
    trajectory_raw = json.loads(trajectory_raw_path.read_text())
    model_trust_raw = json.loads(model_trust_raw_path.read_text())
    true_pin = json.loads(true_pin_path.read_text())
    later = json.loads(later_path.read_text())
    trajectory = _captured_arm(trajectory_raw)
    model_trust = _captured_arm(model_trust_raw)
    banked_old = true_pin["arms"]["22086/43 pure"]["solve"]
    banked_new = later["backends"]["gpu"]

    old_trip_two = float(banked_old["active_set_residuals"][1])
    new_trip_two = float(banked_new["active_set_residuals"][1])
    trajectory_trip_two = float(trajectory["solve"]["active_set_residuals"][1])
    trust_trip_two = float(model_trust["solve"]["active_set_residuals"][1])
    trajectory_preserves_early_path = math.isclose(
        trajectory_trip_two, old_trip_two, rel_tol=1.0e-12, abs_tol=1.0e-14
    )
    trust_matches_new_path = math.isclose(
        trust_trip_two, new_trip_two, rel_tol=1.0e-12, abs_tol=1.0e-14
    )
    trust_matches_old_path = math.isclose(
        trust_trip_two, old_trip_two, rel_tol=1.0e-12, abs_tol=1.0e-14
    )
    if trust_matches_new_path:
        promotion_commit = MODEL_TRUST_REVISION
        promotion_function = "_backtracked_promotion"
        promotion_mechanism = (
            "model-trust selection refuses a candidate family when realized merit "
            "decrease is more than tenfold weaker than its frozen local prediction"
        )
        promotion_quote = (
            "ladder_accepted = jnp.any(sufficient) & (\n"
            "        ~jnp.asarray(model_trust_selection) | ladder_trusted\n"
            "    )"
        )
    elif trust_matches_old_path:
        promotion_commit = OWN_MASK_REVISION
        promotion_function = "_backtracked_promotion"
        promotion_mechanism = (
            "own-mask acceptance refuses the banked promotion because its residual "
            "does not strictly improve when candidate and incumbent are each scored "
            "on the mask they induce"
        )
        promotion_quote = (
            "sufficient &= ~jnp.asarray(own_mask_acceptance) | "
            "(residuals < incumbent_residual)"
        )
    else:
        raise RuntimeError("model-trust capture matches neither banked trip-2 path")

    trajectory_masks = trajectory["solve"]["active_set_mask_differences"]
    trust_masks = model_trust["solve"]["active_set_mask_differences"]
    trajectory_preserves_reopening = trajectory_masks[3:5] == [1, 22]
    trust_preserves_reopening = trust_masks[3:5] == [1, 22]
    own_mask_blocks_reopening = banked_new["active_set_mask_differences"][3:5] == [
        0,
        0,
    ]
    if not (
        trajectory_preserves_reopening
        and trust_preserves_reopening
        and own_mask_blocks_reopening
    ):
        raise RuntimeError("intermediate captures do not isolate mask reopening")
    acceptance_map_quote = (
        "def acceptance_map(candidate):\n"
        "            if acceptance_shadow_mask_fn is None or not "
        "own_mask_acceptance:\n"
        "                return frozen_map(candidate)\n"
        "            candidate_shadow = jnp.ravel(\n"
        "                acceptance_shadow_mask_fn(candidate, carry.shadow_mask)\n"
        "            )\n"
        "            return acceptance_shadowed_map_fn(candidate, candidate_shadow)"
    )
    job_id = trajectory["allocation"]["job_id"]
    if model_trust["allocation"]["job_id"] != job_id:
        raise RuntimeError("intermediate captures did not share one H200 allocation")

    scheduler = {
        "job_id": job_id,
        "node": trajectory["allocation"]["node"],
        "elapsed": job_elapsed,
        "exit_marker": exit_marker,
        "partition": trajectory["allocation"]["partition"],
        "reservation": trajectory["allocation"]["reservation"],
    }
    histories = {
        "banked_old": {
            "revision": TRUE_REVISION,
            "active_set_residuals": banked_old["active_set_residuals"],
            "active_set_mask_differences": banked_old["active_set_mask_differences"],
            "terminal_residual": banked_old["terminal_residual"],
            "termination_reason": banked_old["termination_reason"],
            "trip_count": banked_old["trip_count"],
        },
        "after_trajectory": {
            "revision": TRAJECTORY_REVISION,
            **trajectory["solve"],
            "scheduler": scheduler,
            "raw_capture": str(trajectory_raw_path),
            "raw_capture_sha256": _sha256(trajectory_raw_path),
        },
        "after_model_trust": {
            "revision": MODEL_TRUST_REVISION,
            **model_trust["solve"],
            "scheduler": scheduler,
            "raw_capture": str(model_trust_raw_path),
            "raw_capture_sha256": _sha256(model_trust_raw_path),
        },
        "banked_new": {
            "revision": WINDOW_END,
            "active_set_residuals": banked_new["active_set_residuals"],
            "active_set_mask_differences": banked_new["active_set_mask_differences"],
            "terminal_residual": banked_new["terminal_residual"],
            "termination_reason": banked_new["termination_reason"],
            "trip_count": banked_new["active_set_iterations"],
        },
    }
    receipt = {
        "receipt": "fixed-point root-cause reading",
        "arm": "22086/43 pure",
        "sources": {
            "true_pin_reproduction": str(true_pin_path),
            "true_pin_reproduction_sha256": _sha256(true_pin_path),
            "later_backend_receipt": str(later_path),
            "later_backend_receipt_sha256": _sha256(later_path),
        },
        "candidate_window": _candidate_window(),
        "targeted_reading": _root_cause_reading(),
        "histories": histories,
        "behavioral_differences": {
            "trip_2_newton_promotion": {
                "banked_old_residual": old_trip_two,
                "banked_new_residual": new_trip_two,
                "absolute_difference": abs(new_trip_two - old_trip_two),
                "trajectory_revision_residual": trajectory_trip_two,
                "model_trust_revision_residual": trust_trip_two,
                "trajectory_revision_preserves_banked_path": (
                    trajectory_preserves_early_path
                ),
                "model_trust_revision_matches_banked_new_path": (
                    trust_matches_new_path
                ),
                "model_trust_revision_matches_banked_old_path": (
                    trust_matches_old_path
                ),
                "mechanism": promotion_mechanism,
                "responsible_hunks": [
                    _verified_hunk(
                        promotion_commit, promotion_function, promotion_quote
                    )
                ],
            },
            "trip_4_to_5_mask_reopening": {
                "banked_old_mask_differences_trip_4_to_5": banked_old[
                    "active_set_mask_differences"
                ][3:5],
                "trajectory_revision_mask_differences_trip_4_to_5": (
                    trajectory_masks[3:5]
                ),
                "model_trust_revision_mask_differences_trip_4_to_5": (trust_masks[3:5]),
                "banked_new_mask_differences_trip_4_to_5": banked_new[
                    "active_set_mask_differences"
                ][3:5],
                "trajectory_revision_preserves_banked_reopening": (
                    trajectory_preserves_reopening
                ),
                "model_trust_revision_preserves_banked_reopening": (
                    trust_preserves_reopening
                ),
                "own_mask_revision_blocks_reopening": (own_mask_blocks_reopening),
                "mechanism": (
                    "the induced-mask acceptance map and strict residual-improvement "
                    "guard reject the banked trip-2 promotion; the resulting state "
                    "reaches trip 3 on a path whose candidate-induced mask stays "
                    "fixed, instead of later reopening one cell and then twenty-two"
                ),
                "responsible_hunks": [
                    _verified_hunk(
                        OWN_MASK_REVISION,
                        "_newton_krylov_inner",
                        acceptance_map_quote,
                    ),
                    _verified_hunk(
                        OWN_MASK_REVISION,
                        "_backtracked_promotion",
                        (
                            "sufficient &= ~jnp.asarray(own_mask_acceptance) | "
                            "(residuals < incumbent_residual)"
                        ),
                    ),
                ],
            },
        },
        "defect_or_policy": {
            "classification": "deliberate_policy_consequence",
            "sentence": (
                "The new behavior is a deliberate policy consequence, not a solver "
                "defect: it refuses a promotion the banked code accepted without the "
                "current induced-mask evidence, and that policy-selected path keeps "
                "the mask settled until honest stagnation."
            ),
        },
        "remedy_ranking": [
            {
                "rank": 1,
                "rung": "adaptive Krylov depth",
                "reason": (
                    "the accepted policy exposes a near-neutral direction that a "
                    "twelve-action inner solve still fails to contract, so allocate "
                    "additional actions only after measured contraction stalls"
                ),
            },
            {
                "rank": 2,
                "rung": "deflation or recycling",
                "reason": (
                    "the repeated settled-mask residual supplies a stable mode "
                    "estimate "
                    "that can augment or deflate the next Krylov solve"
                ),
            },
            {
                "rank": 3,
                "rung": "Anderson acceleration",
                "reason": (
                    "the post-policy map has the fixed-mask, near-unit-eigenvalue "
                    "shape "
                    "for which a short outer history is directly useful"
                ),
            },
            {
                "rank": 4,
                "rung": "mode-aware step handling",
                "reason": (
                    "it becomes justified only if the inner trace shows that the "
                    "near-neutral direction is repeatedly clipped or refused"
                ),
            },
            {
                "rank": 5,
                "rung": "vertical-mode regularization",
                "reason": (
                    "it is physics-shaped and the mixed arm supports it, but it "
                    "changes "
                    "the objective more than the numerical remedies above"
                ),
            },
        ],
        "execution_contract": {
            "at_most_two_re_solves": 2,
            "actual_re_solves": 2,
            "one_reserved_h200": True,
            "allocated_cpus": trajectory["allocation"]["allocated_cpus"],
            "jax_platforms": trajectory["allocation"]["jax_platforms"],
            "solver_source_modified": False,
            "assigned_worktree_nova_diff_stat": subprocess.check_output(
                ["git", "diff", "--stat", "--", "nova"], cwd=ROOT, text=True
            ).strip(),
        },
        "figure": str(figure),
    }
    _check_root_cause(receipt, require_figure=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    _plot_root_cause(receipt, figure)
    _check_root_cause(receipt, require_figure=True)
    return receipt


def _check_root_cause(receipt: dict[str, Any], *, require_figure: bool = True) -> None:
    if receipt["candidate_window"]["count"] != 10:
        raise RuntimeError("root-cause receipt omits the ten-commit window")
    if len(receipt["targeted_reading"]) != 10:
        raise RuntimeError("root-cause receipt omits targeted commit readings")
    if receipt["execution_contract"]["actual_re_solves"] > 2:
        raise RuntimeError("root-cause receipt exceeds the re-solve limit")
    if receipt["execution_contract"]["assigned_worktree_nova_diff_stat"]:
        raise RuntimeError("assigned worktree contains a change under nova")
    if receipt["execution_contract"]["solver_source_modified"]:
        raise RuntimeError("root-cause receipt reports modified solver source")
    for key in ("after_trajectory", "after_model_trust"):
        history = receipt["histories"][key]
        for field in (
            "active_set_residuals",
            "active_set_mask_differences",
            "terminal_residual",
            "termination_reason",
            "trip_count",
            "selected_saddle_m",
            "scheduler",
        ):
            if history.get(field) is None:
                raise RuntimeError(f"{key} omits {field}")
        if history["scheduler"]["exit_marker"] != 0:
            raise RuntimeError(f"{key} did not exit cleanly")
    for difference in receipt["behavioral_differences"].values():
        if not difference["responsible_hunks"]:
            raise RuntimeError("behavioral difference omits a responsible hunk")
    if not receipt["defect_or_policy"]["sentence"]:
        raise RuntimeError("root-cause receipt omits the policy verdict")
    if [row["rank"] for row in receipt["remedy_ranking"]] != [1, 2, 3, 4, 5]:
        raise RuntimeError("root-cause remedy ranking is incomplete")
    if require_figure:
        figure = Path(receipt["figure"])
        if not figure.exists() or figure.stat().st_size == 0:
            raise RuntimeError("root-cause figure is absent")


def _intervening_nova_commits(start: str, end: str) -> dict[str, Any]:
    command = [
        "git",
        "log",
        "--reverse",
        "--format=%H%x09%P%x09%s",
        f"{start}..{end}",
        "--",
        "nova/",
    ]
    verbatim = subprocess.check_output(command, cwd=ROOT, text=True).rstrip("\n")
    rows = []
    for line in verbatim.splitlines():
        sha, parents, subject = line.split("\t", 2)
        rows.append({"sha": sha, "parents": parents.split(), "subject": subject})
    return {
        "command": " ".join(command),
        "verbatim": verbatim,
        "count": len(rows),
        "commits": rows,
    }


def _attribution_arm(
    raw: dict[str, Any], key: str, *, elapsed: str, exit_marker: int
) -> dict[str, Any]:
    captured = raw["arms"][key]
    solve = captured["solve"]
    committed_saddle = np.asarray(COMMITTED_SADDLES[key], dtype=np.float64)
    measured_saddle = np.asarray(solve["selected_saddle_m"], dtype=np.float64)
    return {
        **captured,
        "committed_row": raw["committed_rows"][key],
        "terminal_residual_below_1e-8": float(solve["terminal_residual"]) < 1.0e-8,
        "selected_saddle_distance_from_committed_m": float(
            np.linalg.norm(measured_saddle - committed_saddle)
        ),
        "scheduler": {
            "job_id": raw["allocation"]["job_id"],
            "node": raw["allocation"]["node"],
            "cpu_count": raw["allocation"]["allocated_cpus"],
            "elapsed": elapsed,
            "exit_marker": exit_marker,
        },
    }


def _plot_null_polish_attribution(receipt: dict[str, Any], figure: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(12, 7), constrained_layout=True)
    styles = {
        ("before_null_polish", "22086/43 pure"): ("o", "-", "C0"),
        ("before_null_polish", "21978/35 pure"): ("s", "-", "C1"),
        ("main_head", "22086/43 pure"): ("o", "--", "C2"),
        ("main_head", "21978/35 pure"): ("s", "--", "C3"),
    }
    labels = {
        "before_null_polish": "69fc5e3e before null polish",
        "main_head": f"{receipt['revisions']['main_head']['revision'][:8]} main HEAD",
    }
    for (revision_key, arm_key), (marker, linestyle, color) in styles.items():
        history = receipt["revisions"][revision_key]["arms"][arm_key]["solve"][
            "active_set_residuals"
        ]
        axis.semilogy(
            range(1, len(history) + 1),
            history,
            marker=marker,
            linestyle=linestyle,
            linewidth=2.0,
            color=color,
            label=f"{labels[revision_key]} · {arm_key}",
        )
    for key, label, marker, color in (
        ("banked_old", "d47f7cd1 banked 22086/43", "D", "0.25"),
        ("own_mask", "a4bec44f own-mask 22086/43", "^", "0.55"),
    ):
        history = receipt["reference_histories"][key]["active_set_residuals"]
        axis.semilogy(
            range(1, len(history) + 1),
            history,
            marker=marker,
            linestyle=":" if key == "banked_old" else "-.",
            linewidth=1.7,
            color=color,
            label=label,
        )
    axis.axhline(1.0e-8, color="black", linestyle=":", linewidth=1.2, label="1e-8")
    axis.set_xlabel("active-set trip")
    axis.set_ylabel("relative sup residual")
    axis.set_title("Pure-arm convergence before and after census null polishing")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(fontsize=8, ncol=2)
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure, dpi=180)
    plt.close(fig)


def compile_null_polish_attribution(
    before_raw_path: Path,
    head_raw_path: Path,
    root_cause_path: Path,
    output: Path,
    figure: Path,
    job_elapsed: str,
    exit_marker: int,
) -> dict[str, Any]:
    before_raw = json.loads(before_raw_path.read_text())
    head_raw = json.loads(head_raw_path.read_text())
    reference = json.loads(root_cause_path.read_text())
    if before_raw["allocation"]["job_id"] != head_raw["allocation"]["job_id"]:
        raise RuntimeError("revision captures did not share one H200 allocation")

    revisions = {}
    for revision_key, raw, raw_path in (
        ("before_null_polish", before_raw, before_raw_path),
        ("main_head", head_raw, head_raw_path),
    ):
        revisions[revision_key] = {
            "revision": raw["source"]["revision"],
            "shadow_root": raw["source"]["shadow_root"],
            "allocation": raw["allocation"],
            "raw_capture": str(raw_path),
            "raw_capture_sha256": _sha256(raw_path),
            "arms": {
                key: _attribution_arm(
                    raw, key, elapsed=job_elapsed, exit_marker=exit_marker
                )
                for key in ("22086/43 pure", "21978/35 pure")
            },
        }

    head_x = revisions["main_head"]["arms"]["22086/43 pure"]["solve"][
        "topology_qualification_polish_receipt"
    ]["slots"]["x"]
    x_changes_selected_flux = bool(head_x["accepted"]) and (
        float(head_x["polished_minus_seed_value_wb"]) != 0.0
    )
    if x_changes_selected_flux:
        attributed_commit = NULL_POLISH_MERGE
        mechanism = "census_selected_null_polish"
        verdict = (
            "The convergence flip is attributed to the census-selected null polish: "
            "the main-HEAD 22086/43 X slot accepted a polish with a nonzero flux-value "
            "change, so the boundary flux entering the exact bank route moved."
        )
    else:
        attributed_commit = "990cf0caa5071e54e119eb05ec6b72bc4b00c291"
        mechanism = "distorted_hex_ring_order"
        verdict = (
            "The convergence flip is not attributable to the null polish because the "
            "main-HEAD 22086/43 X slot did not accept a nonzero flux-value change; "
            "the remaining candidate is the distorted-hex ring-order correction."
        )

    banked_old = reference["histories"]["banked_old"]
    own_mask = reference["histories"]["banked_new"]
    receipt = {
        "receipt": "census null-polish convergence attribution",
        "revisions": revisions,
        "current_main_22086_43_converges_below_1e-8": revisions["main_head"]["arms"][
            "22086/43 pure"
        ]["terminal_residual_below_1e-8"],
        "attribution": {
            "mechanism": mechanism,
            "commit": attributed_commit,
            "head_22086_x_slot_accepted": bool(head_x["accepted"]),
            "head_22086_x_polished_minus_seed_value_wb": float(
                head_x["polished_minus_seed_value_wb"]
            ),
            "x_slot_accepted_with_nonzero_value_change": x_changes_selected_flux,
            "verdict": verdict,
        },
        "intervening_nova_commits": _intervening_nova_commits(
            PRE_POLISH_REVISION, head_raw["source"]["revision"]
        ),
        "reference_histories": {
            "source": str(root_cause_path),
            "source_sha256": _sha256(root_cause_path),
            "banked_old": banked_old,
            "own_mask": own_mask,
        },
        "scheduler": {
            "job_id": head_raw["allocation"]["job_id"],
            "node": head_raw["allocation"]["node"],
            "cpu_count": head_raw["allocation"]["allocated_cpus"],
            "elapsed": job_elapsed,
            "exit_marker": exit_marker,
        },
        "carrier": head_raw["carrier"],
        "execution_contract": {
            "one_reserved_h200": True,
            "same_bank_generation_route": head_raw["execution_contract"][
                "same_bank_generation_route"
            ],
            "solver_source_modified": False,
            "assigned_worktree_nova_diff_stat": subprocess.check_output(
                ["git", "diff", "--stat", "--", "nova"], cwd=ROOT, text=True
            ).strip(),
        },
        "figure": str(figure),
    }
    _check_null_polish_attribution(receipt, require_figure=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    _plot_null_polish_attribution(receipt, figure)
    _check_null_polish_attribution(receipt, require_figure=True)
    return receipt


def _check_null_polish_attribution(
    receipt: dict[str, Any], *, require_figure: bool = True
) -> None:
    if receipt["revisions"]["before_null_polish"]["revision"] != PRE_POLISH_REVISION:
        raise RuntimeError("pre-polish capture uses the wrong revision")
    if receipt["execution_contract"]["solver_source_modified"]:
        raise RuntimeError("receipt reports modified solver source")
    if receipt["execution_contract"]["assigned_worktree_nova_diff_stat"]:
        raise RuntimeError("assigned worktree contains a change under nova")
    if receipt["scheduler"]["cpu_count"] != 1:
        raise RuntimeError("attribution capture did not use one CPU")
    if receipt["scheduler"]["exit_marker"] != 0:
        raise RuntimeError("attribution capture did not exit cleanly")
    if receipt["intervening_nova_commits"]["count"] < 2:
        raise RuntimeError("intervening nova commit list is incomplete")
    for revision_key in ("before_null_polish", "main_head"):
        revision = receipt["revisions"][revision_key]
        if revision["allocation"]["allocated_cpus"] != 1:
            raise RuntimeError(f"{revision_key} did not use one CPU")
        for arm_key in ("22086/43 pure", "21978/35 pure"):
            arm = revision["arms"][arm_key]
            solve = arm["solve"]
            for field in (
                "active_set_residuals",
                "active_set_mask_differences",
                "termination_reason",
                "trip_count",
                "terminal_residual",
                "selected_saddle_m",
            ):
                if solve.get(field) is None:
                    raise RuntimeError(f"{revision_key} {arm_key} omits {field}")
            if arm["selected_saddle_distance_from_committed_m"] < 0.0:
                raise RuntimeError(f"{revision_key} {arm_key} has invalid distance")
            if revision_key == "main_head":
                polish = solve.get("topology_qualification_polish_receipt")
                if polish is None:
                    raise RuntimeError(f"main HEAD {arm_key} omits polish receipt")
                for slot_name in ("x", "o"):
                    slot = polish["slots"][slot_name]
                    for field in (
                        "accepted",
                        "seed_position_rz_m",
                        "polished_position_rz_m",
                        "seed_value_wb",
                        "polished_value_wb",
                        "normalized_gradient",
                        "roundoff_floor",
                        "representation_floor",
                    ):
                        if field not in slot:
                            raise RuntimeError(
                                f"main HEAD {arm_key} {slot_name} omits {field}"
                            )
    if not isinstance(receipt["current_main_22086_43_converges_below_1e-8"], bool):
        raise RuntimeError("main HEAD convergence verdict is not explicit")
    if not receipt["attribution"]["verdict"]:
        raise RuntimeError("attribution verdict is absent")
    if require_figure:
        figure = Path(receipt["figure"])
        if not figure.exists() or figure.stat().st_size == 0:
            raise RuntimeError("null-polish attribution figure is absent")


def _comparison(solve: dict[str, Any], committed: dict[str, Any]) -> dict[str, Any]:
    residual_difference = abs(
        float(solve["terminal_residual"]) - float(committed["terminal_residual"])
    )
    saddle_distance = float(
        np.linalg.norm(
            np.asarray(solve["selected_saddle_m"], dtype=np.float64)
            - np.asarray(committed["selected_saddle_m"], dtype=np.float64)
        )
    )
    residual_matches = math.isclose(
        float(solve["terminal_residual"]),
        float(committed["terminal_residual"]),
        rel_tol=1.0e-9,
        abs_tol=1.0e-14,
    )
    reason_matches = solve["termination_reason"] == committed["termination_reason"]
    convergence_matches = bool(solve["converged"]) == bool(committed["converged"])
    saddle_matches = saddle_distance <= 1.0e-9
    return {
        "terminal_residual_absolute_difference": residual_difference,
        "termination_reason_matches": reason_matches,
        "convergence_matches": convergence_matches,
        "selected_saddle_absolute_distance_m": saddle_distance,
        "residual_matches": residual_matches,
        "selected_saddle_matches": saddle_matches,
        "criteria": {
            "residual": "math.isclose(rel_tol=1e-9, abs_tol=1e-14)",
            "selected_saddle_distance_m_max": 1.0e-9,
            "termination_reason_exact": True,
            "convergence_flag_exact": True,
        },
        "reproduced": bool(
            residual_matches
            and reason_matches
            and convergence_matches
            and saddle_matches
        ),
    }


def _plot(receipt: dict[str, Any], comparison: dict[str, Any], figure: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(11, 6.5), constrained_layout=True)
    for key, arm in receipt["arms"].items():
        history = arm["solve"]["active_set_residuals"]
        if history:
            axis.semilogy(
                range(1, len(history) + 1), history, marker="o", label=f"d47f7cd1 {key}"
            )
        else:
            axis.semilogy(
                [arm["solve"]["trip_count"] or 1],
                [arm["solve"]["terminal_residual"]],
                marker="o",
                label=f"d47f7cd1 {key} terminal only",
            )
    for backend, marker in (("cpu", "s"), ("gpu", "^")):
        history = comparison["backends"][backend]["active_set_residuals"]
        axis.semilogy(
            range(1, len(history) + 1),
            history,
            marker=marker,
            linestyle="--",
            label=f"a4bec44f 22086/43 pure {backend}",
        )
    committed = receipt["arms"]["22086/43 pure"]["committed"]
    axis.axhline(
        committed["terminal_residual"],
        color="black",
        linewidth=1.2,
        linestyle=":",
        label="committed 22086/43 converged residual",
    )
    axis.set_xlabel("active-set trip")
    axis.set_ylabel("relative sup residual")
    axis.set_title("True bank-revision reproduction against the later pinned histories")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(fontsize=8)
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure, dpi=180)
    plt.close(fig)


def compile_receipt(
    raw_path: Path,
    comparison_path: Path,
    output: Path,
    figure: Path,
    job_elapsed: str,
    exit_marker: int,
) -> dict[str, Any]:
    raw = json.loads(raw_path.read_text())
    later = json.loads(comparison_path.read_text())
    arms = {}
    for key, captured in raw["arms"].items():
        committed = raw["committed_rows"][key]
        solve = captured["solve"]
        arms[key] = {
            **captured,
            "committed": committed,
            "comparison": _comparison(solve, committed),
            "scheduler": {
                "job_id": raw["allocation"]["job_id"],
                "node": raw["allocation"]["node"],
                "elapsed": job_elapsed,
                "exit_marker": exit_marker,
            },
        }
    receipt = {
        "receipt": "true bank-revision MAST reproduction",
        "source": raw["source"],
        "carrier": raw["carrier"],
        "allocation": raw["allocation"],
        "arms": arms,
        "candidate_window": _candidate_window(),
        "later_revision_histories": {
            "source": str(comparison_path),
            "source_sha256": _sha256(comparison_path),
            "revision": WINDOW_END,
            "cpu": later["backends"]["cpu"]["active_set_residuals"],
            "gpu": later["backends"]["gpu"]["active_set_residuals"],
        },
        "figure": str(figure),
        "assigned_worktree_nova_diff_stat": subprocess.check_output(
            ["git", "diff", "--stat", "--", "nova"], cwd=ROOT, text=True
        ).strip(),
        "execution_contract": {
            "solver_source_modified": False,
            "raw_capture": str(raw_path),
            "raw_capture_sha256": _sha256(raw_path),
        },
    }
    _check(receipt, require_figure=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    _plot(receipt, later, figure)
    _check(receipt, require_figure=True)
    return receipt


def _check(receipt: dict[str, Any], *, require_figure: bool = True) -> None:
    if receipt["source"]["revision"] != TRUE_REVISION:
        raise RuntimeError("receipt does not use the bank-producing revision")
    if receipt["candidate_window"]["count"] != 10:
        raise RuntimeError("receipt does not retain the ten-commit window")
    if receipt["execution_contract"]["solver_source_modified"]:
        raise RuntimeError("solver source was modified")
    if receipt["assigned_worktree_nova_diff_stat"]:
        raise RuntimeError("assigned worktree has a change under nova")
    for key in ("22086/43 pure", "21978/35 pure"):
        arm = receipt["arms"][key]
        for field in ("terminal_residual", "termination_reason", "trip_count"):
            if arm["solve"].get(field) is None:
                raise RuntimeError(f"{key} omits {field}")
        if len(arm["solve"]["selected_saddle_m"]) != 2:
            raise RuntimeError(f"{key} omits its selected saddle")
        if arm["scheduler"]["exit_marker"] != 0:
            raise RuntimeError(f"{key} capture did not exit cleanly")
    if require_figure:
        figure = Path(receipt["figure"])
        if not figure.exists() or figure.stat().st_size == 0:
            raise RuntimeError("reproduction figure is absent")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture_parser = subparsers.add_parser("capture")
    capture_parser.add_argument("--shadow-root", type=Path, default=DEFAULT_SHADOW_ROOT)
    capture_parser.add_argument("--expected-revision", default=TRUE_REVISION)
    capture_parser.add_argument("--expected-cpus", type=int, default=4)
    capture_parser.add_argument("--target", type=_target, action="append")
    capture_parser.add_argument("--output", type=Path, required=True)
    compile_parser = subparsers.add_parser("compile")
    compile_parser.add_argument("--raw", type=Path, required=True)
    compile_parser.add_argument("--comparison", type=Path, default=DEFAULT_COMPARISON)
    compile_parser.add_argument("--output", type=Path, required=True)
    compile_parser.add_argument("--figure", type=Path, required=True)
    compile_parser.add_argument("--job-elapsed", required=True)
    compile_parser.add_argument("--exit-marker", type=int, required=True)
    check_parser = subparsers.add_parser("check")
    check_parser.add_argument("--receipt", type=Path, required=True)
    root_parser = subparsers.add_parser("root-cause-compile")
    root_parser.add_argument("--trajectory-raw", type=Path, required=True)
    root_parser.add_argument("--model-trust-raw", type=Path, required=True)
    root_parser.add_argument("--true-pin", type=Path, default=DEFAULT_TRUE_PIN)
    root_parser.add_argument("--later", type=Path, default=DEFAULT_COMPARISON)
    root_parser.add_argument("--output", type=Path, required=True)
    root_parser.add_argument("--figure", type=Path, required=True)
    root_parser.add_argument("--job-elapsed", required=True)
    root_parser.add_argument("--exit-marker", type=int, required=True)
    root_check_parser = subparsers.add_parser("root-cause-check")
    root_check_parser.add_argument("--receipt", type=Path, required=True)
    attribution_parser = subparsers.add_parser("null-polish-compile")
    attribution_parser.add_argument("--before-raw", type=Path, required=True)
    attribution_parser.add_argument("--head-raw", type=Path, required=True)
    attribution_parser.add_argument(
        "--root-cause", type=Path, default=DEFAULT_ROOT_CAUSE
    )
    attribution_parser.add_argument("--output", type=Path, required=True)
    attribution_parser.add_argument("--figure", type=Path, required=True)
    attribution_parser.add_argument("--job-elapsed", required=True)
    attribution_parser.add_argument("--exit-marker", type=int, required=True)
    attribution_check_parser = subparsers.add_parser("null-polish-check")
    attribution_check_parser.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "capture":
        payload = capture(
            args.shadow_root,
            expected_revision=args.expected_revision,
            targets=tuple(args.target) if args.target else TARGETS,
            expected_cpus=args.expected_cpus,
        )
        _write_json(args.output, payload)
        print(
            "BANK_REVISION_CAPTURE PASS "
            + " ".join(
                f"{key.replace(' ', '_')}={arm['solve']['termination_reason']}:"
                f"{arm['solve']['terminal_residual']:.17g}"
                for key, arm in payload["arms"].items()
            ),
            flush=True,
        )
    elif args.command == "compile":
        receipt = compile_receipt(
            args.raw,
            args.comparison,
            args.output,
            args.figure,
            args.job_elapsed,
            args.exit_marker,
        )
        print(
            "BANK_REVISION_REPRODUCTION PASS "
            + " ".join(
                f"{key.replace(' ', '_')}={arm['comparison']['reproduced']}"
                for key, arm in receipt["arms"].items()
            ),
            flush=True,
        )
    elif args.command == "check":
        receipt = json.loads(args.receipt.read_text())
        _check(receipt)
        print("BANK_REVISION_REPRODUCTION_CHECK PASS", flush=True)
    elif args.command == "root-cause-compile":
        receipt = compile_root_cause(
            args.trajectory_raw,
            args.model_trust_raw,
            args.true_pin,
            args.later,
            args.output,
            args.figure,
            args.job_elapsed,
            args.exit_marker,
        )
        promotion = receipt["behavioral_differences"]["trip_2_newton_promotion"]
        reopening = receipt["behavioral_differences"]["trip_4_to_5_mask_reopening"]
        print(
            "ROOT_CAUSE_COMPILE PASS "
            f"promotion_commit={promotion['responsible_hunks'][0]['commit'][:8]} "
            "own_mask_blocks_reopening="
            f"{reopening['own_mask_revision_blocks_reopening']}",
            flush=True,
        )
    elif args.command == "root-cause-check":
        receipt = json.loads(args.receipt.read_text())
        _check_root_cause(receipt)
        print("ROOT_CAUSE_CHECK PASS", flush=True)
    elif args.command == "null-polish-compile":
        receipt = compile_null_polish_attribution(
            args.before_raw,
            args.head_raw,
            args.root_cause,
            args.output,
            args.figure,
            args.job_elapsed,
            args.exit_marker,
        )
        print(
            "NULL_POLISH_ATTRIBUTION PASS "
            f"main_22086_below_1e-8="
            f"{receipt['current_main_22086_43_converges_below_1e-8']} "
            f"mechanism={receipt['attribution']['mechanism']}",
            flush=True,
        )
    else:
        receipt = json.loads(args.receipt.read_text())
        _check_null_polish_attribution(receipt)
        print("NULL_POLISH_ATTRIBUTION_CHECK PASS", flush=True)


if __name__ == "__main__":
    main()
