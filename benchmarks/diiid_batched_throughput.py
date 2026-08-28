"""Measure fixed-shape batched forward-solve throughput on an accelerator.

The workload is Nova's production ``ForwardProfile.solve_batch`` route on a
bootstrapped free-boundary equilibrium.  Each timed slice executes a fixed
Newton--Krylov policy, takes the preceding slice as its warm start, verifies
that every ensemble member remains on the same plasma branch, and emits the
vacuum decay-index conditioning at its magnetic-axis radius.

The benchmark measures compute throughput.  It does not claim that the
bootstrapped source is a fitted DIII-D discharge or that a fixed evaluation
budget establishes convergence.  The output records the residual and branch
qualification beside the timing so downstream sizing cannot erase that
distinction.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.constants import mu_0

from nova.biot.greens import hybrid_greens
from nova.equilibrium.conservation import FluxLattice
from nova.equilibrium.diagnostics import vertical_conditioning_receipt
from nova.equilibrium.forward import ForwardProfile
from nova.equilibrium.source import DomainProfile, ForwardSource
from nova.jax.config import configure_dtypes

DEFAULT_OUTPUT = Path("batched_throughput_budget.json")
DEFAULT_DETAIL_OUTPUT = Path("batched_throughput_member_details.json.gz")

P_PRIME = -3.0e5
FF_PRIME = -0.25
AXIS_RADIUS = 1.0
SEED_SPAN = 0.35
DRIVE = 1.4
BOUNDARY_FIELD_FUNCTION = 5.0
CONDUCTOR_COUNT = 16
GRID_POINTS = 25
ROOT = Path(__file__).resolve().parents[1]
COMMITTED_RECEIPT_LIMIT_BYTES = 10_000_000


@dataclass(frozen=True)
class BranchReceipt:
    """Identity and location of one solved plasma branch."""

    identity: str
    diverted: bool
    plasma_current_a: float
    axis_r_m: float
    axis_z_m: float


def validate_warm_start_continuity(
    shot: str,
    previous: list[BranchReceipt],
    current: list[BranchReceipt],
    *,
    maximum_axis_step_m: float,
) -> None:
    """Refuse a warm-start batch that changes branch along one shot."""

    if len(previous) != len(current):
        raise RuntimeError(f"{shot}: warm-start ensemble size changed")
    for member, (before, after) in enumerate(zip(previous, current, strict=True)):
        displacement = float(
            np.hypot(after.axis_r_m - before.axis_r_m, after.axis_z_m - before.axis_z_m)
        )
        if not np.isfinite(displacement):
            raise RuntimeError(
                f"{shot}: ensemble member {member} has a non-finite branch axis"
            )
        if before.identity != after.identity:
            raise RuntimeError(
                f"{shot}: ensemble member {member} changed branch "
                f"from {before.identity} to {after.identity}"
            )
        if displacement > maximum_axis_step_m:
            raise RuntimeError(
                f"{shot}: ensemble member {member} axis moved {displacement:.6g} m "
                f"across a {maximum_axis_step_m:.6g} m continuity bound"
            )


def _terms() -> tuple[float, float, float]:
    alpha = np.pi**2 * mu_0 * P_PRIME / 2.0
    return alpha, -2.0 * alpha * AXIS_RADIUS**2, 2.0 * np.pi**2 * FF_PRIME


def _solovev(radius, height):
    alpha, offset, beta = _terms()
    return alpha * radius**4 + offset * radius**2 + beta * height**2


def _wall_loop(points: int = 61) -> tuple[np.ndarray, float]:
    alpha, offset, beta = _terms()
    wall_flux = _solovev(AXIS_RADIUS, 0.0) - SEED_SPAN
    inner, outer = np.sqrt(np.sort(np.roots([alpha, offset, -wall_flux])))
    centre, half = 0.5 * (inner + outer), 0.5 * (outer - inner)
    angle = 2.0 * np.pi * np.arange(points) / points
    radius = centre + half * np.cos(angle)
    argument = np.clip((wall_flux - _solovev(radius, 0.0)) / beta, 0.0, None)
    wall = np.c_[radius, np.sign(np.sin(angle)) * np.sqrt(argument)]
    return wall, float(wall_flux)


def _green_block(target: np.ndarray, source: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            hybrid_greens(target[:, 0], target[:, 1], r, z, 0.05, 0.05)[0]
            for r, z in source
        ],
        axis=1,
    )


def _gradient(amplitude: float):
    def tapered(psi_norm):
        return amplitude * (1.0 - jnp.clip(jnp.asarray(psi_norm), 0.0, 1.0))

    return tapered


def build_workload() -> tuple[ForwardProfile, jax.Array]:
    """Build the receipt-bearing free-boundary workload used for every row."""

    lattice = FluxLattice(
        np.linspace(0.6, 1.42, GRID_POINTS),
        np.linspace(-0.42, 0.42, GRID_POINTS),
    )
    coordinate = lattice.coordinate
    wall, wall_flux = _wall_loop()
    seed_grid = _solovev(coordinate[:, 0], coordinate[:, 1])
    seed_wall = _solovev(wall[:, 0], wall[:, 1])
    seed = jnp.asarray(np.r_[seed_grid, seed_wall])
    inside = seed_grid >= wall_flux
    angle = 2.0 * np.pi * np.arange(CONDUCTOR_COUNT) / CONDUCTOR_COUNT
    conductor = np.c_[
        AXIS_RADIUS + 0.62 * np.cos(angle),
        0.62 * np.sin(angle),
    ]
    coupling = {
        "plasma_to_grid": _green_block(coordinate, coordinate),
        "plasma_to_wall": _green_block(wall, coordinate),
        "source_to_grid": _green_block(coordinate, conductor),
        "source_to_wall": _green_block(wall, conductor),
    }

    flat_source = ForwardSource(
        core=DomainProfile(
            p_prime=lambda psi: jnp.full_like(jnp.asarray(psi), P_PRIME),
            ff_prime=lambda psi: jnp.full_like(jnp.asarray(psi), FF_PRIME),
        ),
        boundary_field_function=BOUNDARY_FIELD_FUNCTION,
    )
    flat = ForwardProfile.from_lattice(
        lattice,
        flat_source,
        external_current=np.zeros(CONDUCTOR_COUNT),
        wall_coordinate=wall,
        polarity=1,
        inside_material=inside,
        **coupling,
    )
    cell_current = np.asarray(flat.operator.cell_current(seed))
    target = np.r_[
        seed_grid - coupling["plasma_to_grid"] @ cell_current,
        seed_wall - coupling["plasma_to_wall"] @ cell_current,
    ]
    weight = np.r_[inside.astype(float), np.ones(len(wall))]
    matrix = np.r_[coupling["source_to_grid"], coupling["source_to_wall"]]
    conductor_current = np.linalg.lstsq(
        matrix * weight[:, None], target * weight, rcond=None
    )[0]
    source = ForwardSource(
        core=DomainProfile(
            p_prime=_gradient(2.0 * DRIVE * P_PRIME),
            ff_prime=_gradient(2.0 * DRIVE * FF_PRIME),
        ),
        boundary_field_function=BOUNDARY_FIELD_FUNCTION,
    )
    profile = ForwardProfile.from_lattice(
        lattice,
        source,
        external_current=conductor_current,
        wall_coordinate=wall,
        polarity=1,
        inside_material=inside,
        **coupling,
    )
    return profile, seed


def _branch_receipts(equilibrium) -> list[BranchReceipt]:
    receipts = []
    for member in range(equilibrium.flux.shape[0]):
        plasma_current = float(equilibrium.ledger.total[member])
        diverted = bool(equilibrium.topology.diverted[member])
        identity = (
            "vacuum"
            if not np.isfinite(plasma_current) or abs(plasma_current) <= 1.0e-6
            else ("diverted_plasma" if diverted else "limited_plasma")
        )
        axis = np.asarray(equilibrium.topology.axis[member], dtype=float)
        receipts.append(
            BranchReceipt(
                identity=identity,
                diverted=diverted,
                plasma_current_a=plasma_current,
                axis_r_m=float(axis[0]),
                axis_z_m=float(axis[1]),
            )
        )
    return receipts


def _conditioning(profile, current: np.ndarray, branch: BranchReceipt) -> dict:
    source_flux = np.asarray(profile.operator.grid.source_target) @ current
    source_flux = source_flux.reshape(profile.lattice.shape)
    radius = np.asarray(profile.lattice.radius, dtype=float)
    height = np.asarray(profile.lattice.height, dtype=float)
    vertical_index = int(np.argmin(np.abs(height - branch.axis_z_m)))
    vertical_field = np.gradient(source_flux[:, vertical_index], radius) / (
        2.0 * np.pi * radius
    )
    return asdict(
        vertical_conditioning_receipt(radius, vertical_field, branch.axis_r_m)
    )


def _strict(value):
    """Map non-finite numeric leaves to null for strict JSON receipts."""

    if isinstance(value, dict):
        return {key: _strict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict(item) for item in value]
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _source_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _driver_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _device_record() -> dict[str, str | int | bool | None]:
    device = jax.devices()[0]
    stats = device.memory_stats() or {}
    return {
        "platform": device.platform,
        "kind": device.device_kind,
        "id": int(device.id),
        "host": platform.node(),
        "jax_version": jax.__version__,
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "memory_bytes_limit": int(stats.get("bytes_limit", 0)) or None,
        "bytes_in_use": int(stats.get("bytes_in_use", 0)),
        "peak_device_memory_bytes": int(
            stats.get("peak_bytes_in_use", stats.get("bytes_in_use", 0))
        ),
    }


def _scheduler_record() -> dict[str, str | None]:
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
            fields = dict(
                token.split("=", 1)
                for token in completed.stdout.split()
                if "=" in token
            )
            accepted = fields.get("TimeLimit")
    return {
        "job_id": job_id,
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "reservation": os.environ.get("SLURM_JOB_RESERVATION"),
        "job_gpus": os.environ.get("SLURM_JOB_GPUS"),
        "accepted_time_limit": accepted,
    }


def _settled_active_set_record() -> dict[str, str | bool]:
    introducing_commit = subprocess.check_output(
        [
            "git",
            "log",
            "-Sactive_set_iterations",
            "-1",
            "--format=%H",
            "--",
            "nova/equilibrium/fixed_point.py",
        ],
        cwd=ROOT,
        text=True,
    ).strip()
    contains = (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", introducing_commit, "HEAD"],
            cwd=ROOT,
            check=False,
        ).returncode
        == 0
    )
    return {
        "contains_settled_active_set_outer_loop": contains,
        "introducing_commit": introducing_commit,
        "measured_route": "ForwardProfile.solve_batch(newton_krylov)",
    }


def _fixed_point_receipt(fixed_point, member: int) -> dict:
    termination_reason = int(np.asarray(fixed_point.termination_reason)[member])
    return {
        "relative_residual": float(np.asarray(fixed_point.residual)[member]),
        "converged": bool(np.asarray(fixed_point.converged)[member]),
        "termination_reason_code": termination_reason,
        "attempted_newton_promotions": int(
            np.asarray(fixed_point.attempted_newton_promotions)[member]
        ),
        "accepted_newton_promotions": int(
            np.asarray(fixed_point.accepted_newton_promotions)[member]
        ),
        "active_set_iterations": int(
            np.asarray(fixed_point.active_set_iterations)[member]
        ),
        "residual_trace": np.asarray(fixed_point.trace)[member].tolist(),
        "active_set_residuals": np.asarray(fixed_point.active_set_residuals)[
            member
        ].tolist(),
        "active_set_mask_differences": np.asarray(
            fixed_point.active_set_mask_differences
        )[member].tolist(),
        "active_set_cycle_damping_activations": np.asarray(
            fixed_point.active_set_cycle_damping_activations
        )[member].tolist(),
    }


class DetailWriter:
    """Stream one strict compressed JSON array without retaining member arrays."""

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = gzip.open(self.path, "wt", encoding="utf-8")
        self._stream.write('{"member_receipts":[')
        self._first = True
        self.count = 0

    def write(self, receipt: dict) -> None:
        if not self._first:
            self._stream.write(",")
        json.dump(
            _strict(receipt), self._stream, allow_nan=False, separators=(",", ":")
        )
        self._first = False
        self.count += 1

    def close(self) -> dict[str, str | int]:
        self._stream.write("]}\n")
        self._stream.close()
        digest = hashlib.sha256(self.path.read_bytes()).hexdigest()
        return {
            "absolute_path": str(self.path),
            "byte_size": self.path.stat().st_size,
            "sha256": digest,
            "member_receipt_count": self.count,
            "format": "gzip-compressed strict JSON",
        }


def measure(
    initial_batch_sizes: tuple[int, ...],
    *,
    output: Path,
    detail_output: Path,
    frames: int,
    repeats: int,
    newton_steps: int,
    gmres_iterations: int,
    warmup: int,
    relaxation: float,
    step_cap: float,
    maximum_axis_step_m: float,
    plateau_relative_gain: float,
    plateau_consecutive_widths: int,
    minimum_plateau_batch_size: int,
) -> dict:
    """Run widths through the mandatory floor, then stop at a measured limit."""

    configure_dtypes()
    login_preflight = os.environ.get("SLURM_JOB_ID") is None
    if jax.devices()[0].platform != "gpu":
        raise RuntimeError("the saturation measurement requires a JAX GPU device")
    if not jax.config.jax_enable_x64:
        raise RuntimeError("JAX float64 is not active")
    if not login_preflight and os.environ.get("SLURM_JOB_PARTITION") != "betelgeuse":
        raise RuntimeError("the saturation measurement requires partition betelgeuse")
    if (
        not login_preflight
        and os.environ.get("SLURM_JOB_RESERVATION") != "gpu_0003_grpA"
    ):
        raise RuntimeError("the saturation measurement requires gpu_0003_grpA")

    profile, seed = build_workload()
    frame_scale = np.linspace(0.99, 1.01, frames)
    convergence_policy = (
        "settled active-set outer loop; each frozen-mask inner solve uses "
        f"{warmup} relaxed warmup steps plus at most {newton_steps} Newton step "
        f"with {gmres_iterations} GMRES iterations; terminal qualification "
        "requires residual tolerance and zero active-mask difference"
    )
    rows: list[dict] = []
    detail = DetailWriter(detail_output)
    plateau_run = 0
    batch_sizes = list(initial_batch_sizes)
    next_batch_size = batch_sizes[-1] * 2
    stop = None
    failed_width = None

    def checkpoint() -> None:
        payload = _base_receipt(
            rows=rows,
            initial_batch_sizes=initial_batch_sizes,
            frames=frames,
            repeats=repeats,
            newton_steps=newton_steps,
            gmres_iterations=gmres_iterations,
            warmup=warmup,
            relaxation=relaxation,
            step_cap=step_cap,
            plateau_relative_gain=plateau_relative_gain,
            plateau_consecutive_widths=plateau_consecutive_widths,
            minimum_plateau_batch_size=minimum_plateau_batch_size,
            convergence_policy=convergence_policy,
        )
        payload["measurement_state"] = "running"
        payload["detail_receipt"] = {
            "absolute_path": str(detail.path),
            "member_receipt_count_so_far": detail.count,
            "complete": False,
        }
        _write_compact(payload, output, enforce_limit=False)

    while True:
        if not batch_sizes:
            batch_sizes.append(next_batch_size)
            next_batch_size *= 2
        batch_size = batch_sizes.pop(0)
        solve = jax.jit(
            lambda state, current: profile.solve_batch(
                state,
                route="newton_krylov",
                current=current,
                newton_steps=newton_steps,
                gmres_iterations=gmres_iterations,
                warmup=warmup,
                relaxation=relaxation,
                step_cap=step_cap,
            )
        )
        base_current = np.asarray(profile.operator.external_current, dtype=float)
        initial = jnp.repeat(seed[None, :], batch_size, axis=0)
        compile_current = jnp.repeat(base_current[None, :], batch_size, axis=0)
        compile_started = time.perf_counter()
        try:
            compiled = solve(initial, compile_current)
            jax.block_until_ready(compiled.flux)
        except Exception as error:
            message = str(error).lower()
            if any(
                marker in message
                for marker in (
                    "resource exhausted",
                    "resource_exhausted",
                    "out of memory",
                    "cuda_error_out_of_memory",
                )
            ):
                failed_width = {
                    "batch_size": batch_size,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
                stop = {
                    "kind": "device_memory_exhaustion",
                    "deciding_number": batch_size,
                    "criterion": (
                        "the first attempted width raised accelerator out-of-memory"
                    ),
                }
                break
            raise
        compile_seconds = time.perf_counter() - compile_started

        max_residual = 0.0
        stable_receipts = 0
        total_receipts = 0
        converged_receipts = 0
        branch_identities: set[str] = set()
        raw_repetitions = []
        for repeat in range(repeats):
            state = initial
            jax.block_until_ready(state)
            previous: list[BranchReceipt] | None = None
            repetition_latency = 0.0
            repetition_max_residual = 0.0
            repetition_stable = 0
            repetition_converged = 0
            repetition_branches: set[str] = set()
            for frame, scale in enumerate(frame_scale):
                current = np.repeat((scale * base_current)[None, :], batch_size, axis=0)
                device_current = jnp.asarray(current)
                jax.block_until_ready(device_current)
                frame_started = time.perf_counter_ns()
                equilibrium = solve(state, device_current)
                jax.block_until_ready(equilibrium.flux)
                repetition_latency += (time.perf_counter_ns() - frame_started) / 1.0e9
                branches = _branch_receipts(equilibrium)
                vacuum_members = [
                    member
                    for member, branch in enumerate(branches)
                    if branch.identity == "vacuum"
                ]
                if vacuum_members:
                    raise RuntimeError(
                        "bootstrapped-shot: fixed-budget solve left the plasma "
                        f"branch for ensemble members {vacuum_members}"
                    )
                if previous is not None:
                    validate_warm_start_continuity(
                        "bootstrapped-shot",
                        previous,
                        branches,
                        maximum_axis_step_m=maximum_axis_step_m,
                    )
                previous = branches
                residuals = np.asarray(equilibrium.fixed_point.residual, dtype=float)
                max_residual = max(max_residual, float(np.max(residuals)))
                repetition_max_residual = max(
                    repetition_max_residual, float(np.max(residuals))
                )
                for member, branch in enumerate(branches):
                    conditioning = _conditioning(profile, current[member], branch)
                    stable_receipts += int(conditioning["stable"])
                    repetition_stable += int(conditioning["stable"])
                    total_receipts += 1
                    branch_identities.add(branch.identity)
                    repetition_branches.add(branch.identity)
                    fixed_point = _fixed_point_receipt(equilibrium.fixed_point, member)
                    converged_receipts += int(fixed_point["converged"])
                    repetition_converged += int(fixed_point["converged"])
                    detail.write(
                        {
                            "batch_size": batch_size,
                            "repeat": repeat,
                            "frame": frame,
                            "ensemble_member": member,
                            "branch": asdict(branch),
                            "branch_qualified": True,
                            "branch_qualification": (
                                "non-vacuum and warm-start continuous"
                            ),
                            "conditioning": conditioning,
                            "fixed_point": fixed_point,
                        }
                    )
                state = equilibrium.flux
            raw_repetitions.append(
                {
                    "repeat": repeat,
                    "latency_seconds": repetition_latency,
                    "timed_slices": batch_size * frames,
                    "slices_per_second": batch_size * frames / repetition_latency,
                    "maximum_terminal_relative_residual": repetition_max_residual,
                    "branch_identities": sorted(repetition_branches),
                    "convergence_policy": convergence_policy,
                    "terminal_receipts": batch_size * frames,
                    "converged_terminal_receipts": repetition_converged,
                    "nonconverged_terminal_receipts": (
                        batch_size * frames - repetition_converged
                    ),
                    "stable_conditioning_receipts": repetition_stable,
                }
            )
        raw_latencies = [row["latency_seconds"] for row in raw_repetitions]
        elapsed = float(np.sum(raw_latencies))
        slices = batch_size * frames * repeats
        throughput = slices / elapsed
        row = {
            "device": jax.devices()[0].device_kind,
            "batch_size": batch_size,
            "precision": str(seed.dtype),
            "convergence_policy": convergence_policy,
            "frames": frames,
            "repeats": repeats,
            "timed_slices": slices,
            "compile_and_first_execute_seconds": compile_seconds,
            "elapsed_seconds": elapsed,
            "slices_per_second": throughput,
            "raw_repetition_latency_seconds": raw_latencies,
            "raw_repetitions": raw_repetitions,
            "maximum_terminal_relative_residual": max_residual,
            "branch_identities": sorted(branch_identities),
            "terminal_receipts": total_receipts,
            "converged_terminal_receipts": converged_receipts,
            "nonconverged_terminal_receipts": total_receipts - converged_receipts,
            "stable_conditioning_receipts": stable_receipts,
            "peak_device_memory_bytes": _device_record()["peak_device_memory_bytes"],
        }
        if rows:
            gain = throughput / rows[-1]["slices_per_second"] - 1.0
            row["relative_throughput_gain"] = gain
            plateau_run = plateau_run + 1 if gain <= plateau_relative_gain else 0
        else:
            row["relative_throughput_gain"] = None
        rows.append(row)
        checkpoint()

        if login_preflight and not batch_sizes:
            stop = {
                "kind": "login_command_preflight_complete",
                "deciding_number": batch_size,
                "criterion": (
                    "execute each explicitly supplied login width exactly once"
                ),
            }
            break
        if (
            batch_size >= minimum_plateau_batch_size
            and plateau_run >= plateau_consecutive_widths
        ):
            stop = {
                "kind": "measured_throughput_plateau",
                "deciding_number": plateau_run,
                "criterion": (
                    f"{plateau_consecutive_widths} consecutive wider widths each "
                    f"improve throughput by at most {plateau_relative_gain:.1%}, "
                    f"after measuring batch {minimum_plateau_batch_size}"
                ),
            }
            break
        if not batch_sizes:
            batch_sizes.append(next_batch_size)
            next_batch_size *= 2

    detail_record = detail.close()
    result = _base_receipt(
        rows=rows,
        initial_batch_sizes=initial_batch_sizes,
        frames=frames,
        repeats=repeats,
        newton_steps=newton_steps,
        gmres_iterations=gmres_iterations,
        warmup=warmup,
        relaxation=relaxation,
        step_cap=step_cap,
        plateau_relative_gain=plateau_relative_gain,
        plateau_consecutive_widths=plateau_consecutive_widths,
        minimum_plateau_batch_size=minimum_plateau_batch_size,
        convergence_policy=convergence_policy,
    )
    result.update(
        {
            "measurement_state": "preflight" if login_preflight else "complete",
            "sweep_stop": stop,
            "first_failed_width": failed_width,
            "detail_receipt": {**detail_record, "complete": True},
        }
    )
    _write_compact(result, output, enforce_limit=True)
    return result


def _base_receipt(
    *,
    rows: list[dict],
    initial_batch_sizes: tuple[int, ...],
    frames: int,
    repeats: int,
    newton_steps: int,
    gmres_iterations: int,
    warmup: int,
    relaxation: float,
    step_cap: float,
    plateau_relative_gain: float,
    plateau_consecutive_widths: int,
    minimum_plateau_batch_size: int,
    convergence_policy: str,
) -> dict:
    return {
        "schema": "nova.diiid_batched_throughput",
        "measurement": "batched forward-solve accelerator saturation",
        "source_revision": _source_revision(),
        "driver_sha256": _driver_sha256(),
        "solver_semantics": _settled_active_set_record(),
        "scheduler": _scheduler_record(),
        "device": _device_record(),
        "configuration": {
            "initial_batch_sizes": list(initial_batch_sizes),
            "measured_batch_sizes": [row["batch_size"] for row in rows],
            "dynamic_continuation": "double until measured plateau or device OOM",
            "frames": frames,
            "repeats": repeats,
            "newton_steps": newton_steps,
            "gmres_iterations": gmres_iterations,
            "warmup": warmup,
            "relaxation": relaxation,
            "step_cap": step_cap,
            "plateau_relative_gain": plateau_relative_gain,
            "plateau_consecutive_widths": plateau_consecutive_widths,
            "minimum_plateau_batch_size": minimum_plateau_batch_size,
        },
        "convergence_policy": convergence_policy,
        "warm_start": {
            "shot": "bootstrapped-shot",
            "policy": "previous solved slice seeds the next slice per member",
            "branch_switches_accepted": 0,
        },
        "qualification": (
            "This is fixed-budget throughput evidence, not a convergence claim. "
            "The pilot operand and this sweep use a limited_plasma branch and one "
            "Newton step; the pilot maximum relative residual was "
            "0.0011089898293892464 with zero stable conditioning receipts."
        ),
        "budget_rows": rows,
    }


def _write_compact(result: dict, output: Path, *, enforce_limit: bool) -> None:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(
        json.dumps(_strict(result), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    size = temporary.stat().st_size
    if enforce_limit and size >= COMMITTED_RECEIPT_LIMIT_BYTES:
        raise RuntimeError(
            f"compact receipt is {size} bytes, above the "
            f"{COMMITTED_RECEIPT_LIMIT_BYTES}-byte limit"
        )
    temporary.replace(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--detail-output", type=Path, default=DEFAULT_DETAIL_OUTPUT)
    parser.add_argument("--batch-sizes", default="1,4,8,16,32,64,128,256,512")
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--newton-steps", type=int, default=1)
    parser.add_argument("--gmres-iterations", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--relaxation", type=float, default=0.5)
    parser.add_argument("--step-cap", type=float, default=0.25)
    parser.add_argument("--maximum-axis-step-m", type=float, default=0.08)
    parser.add_argument("--plateau-relative-gain", type=float, default=0.05)
    parser.add_argument("--plateau-consecutive-widths", type=int, default=2)
    parser.add_argument("--minimum-plateau-batch-size", type=int, default=512)
    arguments = parser.parse_args()
    batch_sizes = tuple(int(value) for value in arguments.batch_sizes.split(","))
    if os.environ.get("SLURM_JOB_ID") is not None and (
        batch_sizes[0] != 1 or batch_sizes[-1] < arguments.minimum_plateau_batch_size
    ):
        raise ValueError("the initial batch ladder must start at 1 and reach 512")
    measure(
        batch_sizes,
        output=arguments.output,
        detail_output=arguments.detail_output,
        frames=arguments.frames,
        repeats=arguments.repeats,
        newton_steps=arguments.newton_steps,
        gmres_iterations=arguments.gmres_iterations,
        warmup=arguments.warmup,
        relaxation=arguments.relaxation,
        step_cap=arguments.step_cap,
        maximum_axis_step_m=arguments.maximum_axis_step_m,
        plateau_relative_gain=arguments.plateau_relative_gain,
        plateau_consecutive_widths=arguments.plateau_consecutive_widths,
        minimum_plateau_batch_size=arguments.minimum_plateau_batch_size,
    )
    print(arguments.output.resolve())
    print(arguments.detail_output.resolve())


if __name__ == "__main__":
    main()
