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


def _validate_shadow(shadow_root: Path) -> dict[str, Any]:
    head = _git(shadow_root, "rev-parse", "HEAD")
    if head != TRUE_REVISION:
        raise RuntimeError(
            f"shadow revision {head} does not equal bank revision {TRUE_REVISION}"
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


def _allocation() -> dict[str, Any]:
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id is None:
        raise RuntimeError("capture requires a scheduler allocation")
    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "0"))
    platforms = os.environ.get("JAX_PLATFORMS", "")
    reservation = os.environ.get("SLURM_JOB_RESERVATION", "")
    if cpus != 4:
        raise RuntimeError(f"capture requires four CPUs, received {cpus}")
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


def capture(shadow_root: Path) -> dict[str, Any]:
    source = _validate_shadow(shadow_root)
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
    allocation = _allocation()
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
        and (int(row["shot"]), int(row["slice_index"])) in TARGETS
    }
    corroboration = _corroboration_module(shadow_root)
    arms = {}
    for target in TARGETS:
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
        },
    }


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
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "capture":
        payload = capture(args.shadow_root)
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
    else:
        receipt = json.loads(args.receipt.read_text())
        _check(receipt)
        print("BANK_REVISION_REPRODUCTION_CHECK PASS", flush=True)


if __name__ == "__main__":
    main()
