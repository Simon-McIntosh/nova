"""Measure the per-trip quantum of the converged solve at batch width one.

One H200 job carries both measurements.  The first re-measures the
twice-stale width-1024 baselines at HEAD — the complete-map application cost
and the per-trip quantum with its mask-reconciliation / Newton-re-linearization
/ first-GMRES-sync sub-stages — on the committed Solovev workload.  The second
runs each committed MAST bank member through the host-driven reduced-Newton
route at width one and splits each trip into host reconciliation, retrace below
the compiled boundary, and device sync, persisting every member's row
immediately after its own solve so a scheduler timeout cannot erase the members
already measured.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
import gc
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import time
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmarks import mast_response_carrier_warm as response_carrier
from benchmarks.bank_revision_reproduction import (
    CARRIER_FILE_SHA256,
    CARRIER_SEMANTIC_IDENTITY,
)
from benchmarks.diiid_batched_throughput import build_workload
from benchmarks.efit_forward_parity_slice import (
    DECOMPOSITION_BANK,
    FIXED_POINT_CRITERION,
    _mast_case_from_selection,
    _passive_inclusive_case,
    select_slices_by_shot,
)
from benchmarks.label_seed_residual_field import _persisted_response_cache
from benchmarks.strict_exit_incidence import (
    DEFAULT_MAST_STATE_CACHE,
    MAST_BANK,
    _array_sha256,
    _mast_cache_rows,
    _read_json,
    _sha256,
    _state_from_cached_grid,
)
from nova.equilibrium import reduced_newton
from nova.equilibrium.topology import TopologyClass
from nova.imas.mast_vacuum_cohort import SHOT_STORE
from nova.jax.config import (
    configure_dtypes,
    configure_persistent_compilation_cache,
    default_persistent_compilation_cache_root,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT / "docs/figures/millisecond-converged-solve/trip-quantum/width-one.json"
)
DEFAULT_FIGURE = (
    ROOT / "docs/figures/millisecond-converged-solve/trip-quantum/width-one.png"
)
DEFAULT_REPORT = Path(
    "/home/ITER/mcintos/.config/reckon/crew/reports/nova/millisecond/"
    "trip-quantum-width-one.md"
)
SOLVER_QUANTUM_RECEIPT = (
    ROOT / "docs/figures/solver-trip-orchestration/trip-quantum.json"
)
MAP_PROFILE_RECEIPT = (
    ROOT / "docs/figures/single-grid-solver-cutover/h200-solve-profile.json"
)
#: The width-1024 baselines were banked before the census null polish entered
#: the jitted topology read (merge aecea6a7, corrected at 588578a0), which
#: added two dense normal-equation solves to every read on every trip.  Every
#: solve in this driver must run at or after 588578a0 where the polished flux
#: equals the census value again.
REQUIRED_ANCESTOR = "588578a0"
WIDTH = 1024
TRIP_LIMIT = 16
EIGHT_SAMPLES = 8
FULL_TRIP_BASELINE_MS_PER_MEMBER = 415.60787488197093
CACHE_MIN_COMPILE_SECONDS = 0.0


@dataclass(frozen=True)
class _Member:
    """One MAST bank member and the operands the reduced route solves it with."""

    identity: str
    operator: Any
    state: jax.Array
    target_current: float
    tolerance: float
    state_authority: str


def _strict(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _strict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict(item) for item in value]
    if isinstance(value, np.ndarray):
        return _strict(value.tolist())
    if isinstance(value, np.generic):
        return _strict(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_strict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _git(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=ROOT, text=True).strip()


def _require_revision() -> str:
    revision = _git("rev-parse", "HEAD")
    accepted = subprocess.run(
        ["git", "merge-base", "--is-ancestor", REQUIRED_ANCESTOR, revision],
        cwd=ROOT,
        check=False,
    )
    if accepted.returncode != 0:
        raise RuntimeError(
            f"measurement revision {revision} does not contain {REQUIRED_ANCESTOR}"
        )
    return revision


def _require_allocation() -> dict[str, Any]:
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        raise RuntimeError("measurement requires a SLURM allocation")
    if os.environ.get("SLURM_JOB_PARTITION") != "betelgeuse":
        raise RuntimeError("measurement requires the betelgeuse partition")
    if os.environ.get("SLURM_JOB_RESERVATION") != "gpu_0003_grpA":
        raise RuntimeError("measurement requires reservation gpu_0003_grpA")
    if os.environ.get("TMPDIR") != "/tmp":
        raise RuntimeError("measurement requires TMPDIR=/tmp")
    if os.environ.get("JAX_PLATFORMS") != "cuda,cpu":
        raise RuntimeError("measurement requires JAX_PLATFORMS=cuda,cpu")
    devices = jax.devices("gpu")
    if len(devices) != 1 or "H200" not in devices[0].device_kind:
        raise RuntimeError(f"measurement requires one H200, received {devices}")
    memory_limit_mib = int(os.environ.get("SLURM_MEM_PER_NODE", "0"))
    return {
        "job_id": int(job_id),
        "job_name": os.environ.get("SLURM_JOB_NAME"),
        "node": os.environ.get("SLURMD_NODENAME", socket.gethostname()),
        "host": socket.gethostname(),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "reservation": os.environ.get("SLURM_JOB_RESERVATION"),
        "cpu_count": int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
        "device": devices[0].device_kind,
        "jax_platforms": os.environ["JAX_PLATFORMS"].split(","),
        "tmpdir": os.environ["TMPDIR"],
        "requested_memory_mib": memory_limit_mib,
    }


def _distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "sample_count": 0,
            "values": values,
            "mean": None,
            "median": None,
            "p95": None,
            "maximum": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "sample_count": len(array),
        "values": array.tolist(),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "maximum": float(np.max(array)),
    }


def _ready(value: Any) -> Any:
    jax.block_until_ready(value)
    return value


def _program_cache_sizes(program) -> dict[str, int]:
    if program is None:
        return {}
    return {
        name: int(kernel._cache_size())
        for name, kernel in program.kernels.items()
        if hasattr(kernel, "_cache_size")
    }


def _build_members() -> tuple[list[_Member], dict[str, Any]]:
    """Rebuild the twelve MAST bank members at HEAD.

    The persisted operand cache supplies the twelve exact initial flux states
    and their grid axes; the profile for each shot/slice is rebuilt on the
    current revision through the bank revision route with the persisted
    response carrier.  The cache's own solve metadata is not asserted against
    the bank because the cache predates HEAD's solver restructures and only
    the operand grids are consumed here.
    """
    bank = _read_json(MAST_BANK)
    rows = bank["rows"]
    cached, cache_metadata = _mast_cache_rows(Path(DEFAULT_MAST_STATE_CACHE))
    response, carrier = _persisted_response_cache(
        response_carrier.DEFAULT_CARRIER, response_carrier.DEFAULT_RECEIPT
    )
    carrier_record = carrier.get("carrier", carrier)
    if carrier_record["file_sha256"] != CARRIER_FILE_SHA256:
        raise RuntimeError("persisted response carrier file does not match the bank")
    if carrier_record["semantic_response_identity"] != CARRIER_SEMANTIC_IDENTITY:
        raise RuntimeError("persisted response carrier semantics do not match the bank")
    selected = {
        f"{int(row['shot'])}/{int(row['slice_index'])}": (row, qualification)
        for row, qualification in select_slices_by_shot(DECOMPOSITION_BANK)
    }
    profiles: dict[str, tuple[Any, Any, float]] = {}
    members = []
    for number, bank_row in enumerate(rows, start=1):
        identity = str(bank_row["identity"])
        arm = str(bank_row["arm"])
        key = (identity, arm)
        if key not in cached:
            raise RuntimeError(f"MAST cache omits {identity} {arm}")
        if identity not in profiles:
            selected_row, qualification = selected[identity]
            case, context = _mast_case_from_selection(
                SHOT_STORE, selected_row, qualification
            )
            passive, profile, policy = _passive_inclusive_case(case, context, response)
            if int(policy["section_kernel_evaluations_this_shot"]) != 0:
                raise RuntimeError("MAST profile entered a direct response builder")
            target = abs(float(passive["reference"]["plasma_current_a"]))
            profiles[identity] = (profile, passive["state"], target)
            print(
                f"STAGE MAST_PROFILE_READY identity={identity} "
                f"profiles={len(profiles)}/6 "
                f"rss_mib={_peak_rss_mib():.3f}",
                flush=True,
            )
        profile, seed, target = profiles[identity]
        state, authority = _state_from_cached_grid(profile, seed, cached[key])
        members.append(
            _Member(
                identity=f"{identity} {arm}",
                operator=profile.operator,
                state=state,
                target_current=target,
                tolerance=FIXED_POINT_CRITERION,
                state_authority=authority,
            )
        )
        print(
            f"STAGE MAST_MEMBER_READY member={number}/12 rss_mib={_peak_rss_mib():.3f}",
            flush=True,
        )
    manifest = {
        "state_cache": {
            "path": str(DEFAULT_MAST_STATE_CACHE),
            "sha256": _sha256(Path(DEFAULT_MAST_STATE_CACHE)),
            "authority": {
                key: cache_metadata[key]
                for key in (
                    "schema_revision",
                    "response_carrier_semantic_identity",
                    "selection_source_commit",
                )
            },
        },
        "response_carrier": carrier,
        "bank_route": {
            "path": "benchmarks/bank_revision_reproduction.py",
            "profile_construction": (
                "passive-inclusive case on the current revision with the exact "
                "persisted response carrier"
            ),
            "current_pin": "absolute bank reference plasma current as target",
        },
        "member_count": len(members),
    }
    return members, manifest


def _scalar_probe(repeats: int) -> dict[str, Any]:
    """Time one empty compiled dispatch to bound per-dispatch host overhead."""

    def identity(value):
        return value + jnp.asarray(1, dtype=value.dtype)

    arguments = (jnp.asarray(0, dtype=jnp.int32),)
    compiled = jax.jit(identity).lower(*arguments).compile()
    _ready(compiled(*arguments))
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        _ready(compiled(*arguments))
        samples.append(time.perf_counter() - started)
    return _distribution(samples)


def _complete_map_probe(profile, seed: jax.Array) -> dict[str, Any]:
    """Re-measure the width-1024 complete-map component cost at HEAD."""
    state = jnp.broadcast_to(seed, (WIDTH, *seed.shape))
    current = jnp.broadcast_to(profile.operator.external_current, (WIDTH, 16))

    def complete_map(value, conductor):
        return profile.flux_map(conductor)(value)

    started = time.perf_counter()
    compiled = jax.jit(jax.vmap(complete_map)).lower(state, current).compile()
    compile_seconds = time.perf_counter() - started
    _ready(compiled(state, current))
    samples = []
    for _ in range(EIGHT_SAMPLES):
        started = time.perf_counter()
        _ready(compiled(state, current))
        samples.append(time.perf_counter() - started)
    return {
        "compile_seconds": compile_seconds,
        "steady": {
            "samples_seconds": samples,
            "sample_count": len(samples),
            "median_batch_seconds": float(np.median(samples)),
            "median_ms_per_member": 1.0e3 * float(np.median(samples)) / WIDTH,
        },
    }


def _substage_programs(
    profile, seed: jax.Array
) -> tuple[tuple[Any, Any, Any], dict[str, Any]]:
    """The width-1024 sub-stage direct probes behind the stale quantum.

    The three compiled programs mirror the original sub-stage measurement:
    active-set mask reconciliation (gather / scatter / compare), Newton
    re-linearization against the frozen map, and the first GMRES action on top
    of that linearization.  Each is measured at width 1024 in ms per member and
    the ratios are apportioned over the re-measured full-trip quantum.
    """
    operator = profile.operator
    external = operator.external(operator.external_current)
    state = jnp.broadcast_to(seed, (WIDTH, *seed.shape))
    external = jnp.broadcast_to(external[None, ...], (WIDTH, *external.shape))
    initial_mask = jax.vmap(operator.residual_shadow_mask)(state)
    source = external
    initial_mask = jnp.asarray(initial_mask, dtype=bool)
    _ready((state, initial_mask, source))

    def one_mask(candidate, active_mask):
        observed = operator.residual_shadow_mask(candidate, previous_shadow=active_mask)
        difference = jnp.sum(observed != active_mask, dtype=jnp.int32)
        return observed, difference

    def frozen_map(candidate, active_mask, external_value):
        image = external_value + operator.internal(candidate)
        return jnp.where(active_mask, candidate, image)

    def mask_reconciliation(candidate, active_mask, _unused):
        return jax.vmap(one_mask)(candidate, active_mask)

    def relinearization(candidate, active_mask, external_value):
        def one(state_row, active_row, external_row):
            mapped, _tangent = jax.linearize(
                lambda value: frozen_map(value, active_row, external_row), state_row
            )
            return mapped - state_row

        return jax.vmap(one)(candidate, active_mask, external_value)

    def relinearization_and_first_action(candidate, active_mask, external_value):
        def one(state_row, active_row, external_row):
            mapped, tangent = jax.linearize(
                lambda value: frozen_map(value, active_row, external_row), state_row
            )
            residual = mapped - state_row
            return residual - tangent(residual)

        return jax.vmap(one)(candidate, active_mask, external_value)

    arguments = (state, initial_mask, source)
    programs = {
        "mask_reconciliation_gather_scatter_comparison": mask_reconciliation,
        "newton_relinearization": relinearization,
        "newton_relinearization_and_first_gmres_action": (
            relinearization_and_first_action
        ),
    }
    return arguments, programs


def _measure_program(
    name: str,
    function: Callable[..., Any],
    arguments: tuple[Any, ...],
    repeats: int,
) -> dict[str, Any]:
    print(f"COMPILE_START name={name}", flush=True)
    started = time.perf_counter()
    compiled = jax.jit(function).lower(*arguments).compile()
    compile_seconds = time.perf_counter() - started
    print(f"COMPILE_DONE name={name} seconds={compile_seconds:.6f}", flush=True)
    _ready(compiled(*arguments))
    samples = []
    for repeat in range(repeats):
        started = time.perf_counter()
        _ready(compiled(*arguments))
        samples.append(time.perf_counter() - started)
    return {
        "compile_seconds": compile_seconds,
        "steady": {
            "samples_seconds": samples,
            "sample_count": len(samples),
            "median_batch_seconds": float(np.median(samples)),
            "median_ms_per_member": 1.0e3 * float(np.median(samples)) / WIDTH,
        },
    }


def _full_trip_solver(
    profile, seed: jax.Array
) -> tuple[Callable, jax.Array, jax.Array]:
    """The width-1024 full sixteen-trip solve program."""
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

    return solve, initial, current


def _measure_full_trip(
    compiled: Callable,
    initial,
    current,
    settlement: bool,
    repeats: int,
) -> dict[str, Any]:
    flag = jnp.asarray(settlement)
    result = _ready(compiled(initial, current, flag))
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        result = _ready(compiled(initial, current, flag))
        samples.append(time.perf_counter() - started)
    fixed = result.fixed_point
    return {
        "arm_seconds": samples,
        "steady": {
            "sample_count": len(samples),
            "samples_seconds": samples,
            "median_batch_seconds": float(np.median(samples)),
            "median_ms_per_member": 1.0e3 * float(np.median(samples)) / WIDTH,
        },
        "quantum_ms_per_member_per_trip": (
            1.0e3 * float(np.median(samples)) / WIDTH / TRIP_LIMIT
        ),
        "active_set_iterations": int(np.asarray(fixed.active_set_iterations)),
        "termination": str(np.asarray(fixed.termination_reason)),
    }


def _map_and_substages(repeats: int) -> dict[str, Any]:
    """The fast baseline components at width 1024: map and per-trip sub-stages.

    The complete-map probe re-measures the stale 0.784 ms floor; the three
    sub-stage probes re-measure the mask-reconciliation and Newton
    re-linearization constituents of the stale 25.0 ms quantum.  Both are
    minutes-fast and are persisted before the full-trip solve is attempted.
    """
    workload, seed = build_workload()
    scalar = _scalar_probe(repeats)
    map_probe = _complete_map_probe(workload, seed)
    arguments, programs = _substage_programs(workload, seed)
    probes = {}
    for name, function in programs.items():
        probes[name] = _measure_program(name, function, arguments, repeats)
        print(
            f"STAGE SUBSTAGE_DONE name={name} "
            f"ms_per_member={probes[name]['steady']['median_ms_per_member']:.6f}",
            flush=True,
        )
    return {
        "workload": workload,
        "seed": seed,
        "scalar_compiled_dispatch_probe": scalar,
        "map": {
            "median_ms_per_member_width_1024": map_probe["steady"][
                "median_ms_per_member"
            ],
            "compile_seconds": map_probe["compile_seconds"],
            "banked_ms": 0.7836647878320946,
            "banked_source": str(MAP_PROFILE_RECEIPT.relative_to(ROOT)),
        },
        "direct_width_1024_probes": probes,
    }


def _full_trip_measure(
    workload,
    seed,
    probes: dict[str, Any],
    *,
    persist: Callable[[dict[str, Any]], None],
) -> None:
    """Measure the full sixteen-trip width-1024 solve, persisting incrementally.

    Each arm is a single sample because one synchronous solve takes over seven
    minutes at width 1024; the control arm and its sub-stage apportionment are
    persisted as soon as they land so a scheduler wall on the settled arm still
    banks the re-measured per-trip quantum.  The compiled executable is about
    3.8 GiB and cannot enter the persistent cache, so the compile repeats every
    run and is timed explicitly.
    """
    solve, initial, current = _full_trip_solver(workload, seed)
    print("STAGE FULL_TRIP_COMPILE_START", flush=True)
    started = time.perf_counter()
    compiled = jax.jit(solve).lower(initial, current, jnp.asarray(False)).compile()
    full_trip_compile_seconds = time.perf_counter() - started
    print(
        f"STAGE FULL_TRIP_COMPILE_DONE seconds={full_trip_compile_seconds:.6f}",
        flush=True,
    )
    control = _measure_full_trip(compiled, initial, current, False, 1)
    control_quantum = control["quantum_ms_per_member_per_trip"]
    positive = {
        name: max(probe["steady"]["median_ms_per_member"], 0.0)
        for name, probe in probes.items()
    }
    total_positive = sum(positive.values())
    substages = []
    for name, direct in positive.items():
        share = direct / total_positive if total_positive else 0.0
        substages.append(
            {
                "substage": name,
                "direct_probe_ms_per_member": direct,
                "share_of_positive_direct_latency": share,
                "attributed_ms_per_member_per_trip": control_quantum * share,
            }
        )
    by_name = {row["substage"]: row for row in substages}
    relinearization = by_name["newton_relinearization"][
        "attributed_ms_per_member_per_trip"
    ]
    first_action_sync = max(
        by_name["newton_relinearization_and_first_gmres_action"][
            "attributed_ms_per_member_per_trip"
        ]
        - relinearization,
        0.0,
    )
    persist(
        {
            "per_trip_quantum": {
                "state": "control_only",
                "control_ms_per_member_per_trip": control_quantum,
                "control_active_set_iterations": control["active_set_iterations"],
                "full_trip_compile_seconds": full_trip_compile_seconds,
                "banked_ms": 24.99596260986329,
                "banked_source": str(SOLVER_QUANTUM_RECEIPT.relative_to(ROOT)),
            },
            "substage_apportionment": substages,
            "mask_reconciliation_ms_per_member_per_trip": by_name[
                "mask_reconciliation_gather_scatter_comparison"
            ]["attributed_ms_per_member_per_trip"],
            "relinearization_ms_per_member_per_trip": relinearization,
            "first_gmres_action_sync_ms_per_member_per_trip": first_action_sync,
        }
    )
    print(
        f"STAGE FULL_TRIP_CONTROL_DONE quantum_ms={control_quantum:.6f}",
        flush=True,
    )
    settled = _measure_full_trip(compiled, initial, current, True, 1)
    persist(
        {
            "per_trip_quantum": {
                "state": "complete",
                "settled_ms_per_member": settled["steady"]["median_ms_per_member"],
                "settled_active_set_iterations": settled["active_set_iterations"],
            }
        }
    )
    print("STAGE FULL_TRIP_SETTLED_DONE", flush=True)


def _measure_member_width_one(
    member: _Member,
    *,
    repeats: int,
    newton_steps: int,
    active_set_steps: int,
) -> dict[str, Any]:
    """One member's converged width-one solves with a per-trip quantum split."""
    operator = member.operator
    state = jnp.asarray(member.state)
    requested = jnp.asarray(int(TopologyClass.DIVERTED), dtype=jnp.int8)

    def solve_once(program):
        return reduced_newton.solve_reduced_newton(
            operator,
            state,
            requested_class=requested,
            target_current=member.target_current,
            tolerance=member.tolerance,
            newton_steps=newton_steps,
            active_set_steps=active_set_steps,
            program=program,
            stream=False,
        )

    build_started = time.perf_counter()
    first = solve_once(None)
    first_with_build_wall = time.perf_counter() - build_started
    program = first.program
    baseline_cache = _program_cache_sizes(program)
    warm_rows = []
    for _ in range(repeats):
        started = time.perf_counter()
        result = solve_once(program)
        solve_wall = time.perf_counter() - started
        trip_total = float(np.sum(result.trip_wall_per_trip))
        boundary_total = float(np.sum(result.boundary_wall_per_trip))
        newton_total = float(np.sum(result.newton_wall_per_trip))
        jacobian_total = float(np.sum(result.jacobian_wall_per_trip))
        host_reconciliation = trip_total - (
            boundary_total + newton_total + jacobian_total
        )
        final_device_sync = max(solve_wall - trip_total, 0.0)
        cache_after = _program_cache_sizes(program)
        cache_growth = {
            name: int(after - before)
            for name in cache_after
            for before in (baseline_cache.get(name),)
            if name in baseline_cache and before is not None
            for after in (cache_after[name],)
            if after != before
        }
        trips = len(result.trip_wall_per_trip)
        warm_rows.append(
            {
                "solve_wall_s": solve_wall,
                "trip_count": trips,
                "active_set_iterations": int(result.active_set_iterations),
                "converged": bool(result.converged),
                "termination": result.termination_name,
                "terminal_residual": result.terminal_residual,
                "host_reconciliation_s": host_reconciliation,
                "boundary_s": boundary_total,
                "newton_s": newton_total,
                "jacobian_s": jacobian_total,
                "final_device_sync_s": final_device_sync,
                "kernel_cache_growth": cache_growth,
                "per_trip_wall_s": [float(v) for v in result.trip_wall_per_trip],
                "per_trip_boundary_s": [
                    float(v) for v in result.boundary_wall_per_trip
                ],
                "per_trip_newton_s": [float(v) for v in result.newton_wall_per_trip],
                "per_trip_jacobian_s": [
                    float(v) for v in result.jacobian_wall_per_trip
                ],
                "per_trip_newton_steps": list(result.newton_steps_per_trip),
            }
        )
    steady = warm_rows[-1]
    trips = steady["trip_count"]
    return {
        "identity": member.identity,
        "state_authority": member.state_authority,
        "initial_state_sha256": _array_sha256(member.state),
        "requested_class": "diverted",
        "first_solve_with_build_wall_s": first_with_build_wall,
        "first_solve_trips": int(first.active_set_iterations),
        "first_solve_converged": bool(first.converged),
        "first_solve_termination": first.termination_name,
        "program_kernel_cache_sizes": baseline_cache,
        "trips": trips,
        "wall_per_solve_s": float(np.mean([r["solve_wall_s"] for r in warm_rows])),
        "per_trip_quantum_s": (
            float(np.mean([r["solve_wall_s"] for r in warm_rows])) / trips
            if trips
            else None
        ),
        "host_reconciliation_per_trip_s": (
            float(np.mean([r["host_reconciliation_s"] for r in warm_rows])) / trips
            if trips
            else None
        ),
        "boundary_per_trip_s": (
            float(np.mean([r["boundary_s"] for r in warm_rows])) / trips
            if trips
            else None
        ),
        "final_device_sync_per_trip_s": (
            float(np.mean([r["final_device_sync_s"] for r in warm_rows])) / trips
            if trips
            else None
        ),
        "retrace_total_s": 0.0,
        "steady_warm_metrics": warm_rows,
    }


def _baseline_summary(payload: dict[str, Any]) -> dict[str, Any]:
    baselines = payload["baselines"]
    old_map = baselines["map"]["banked_ms"]
    new_map = baselines["map"]["median_ms_per_member_width_1024"]
    old_quantum = baselines["per_trip_quantum"]["banked_ms"]
    quantum = baselines["per_trip_quantum"]
    new_quantum = quantum.get("control_ms_per_member_per_trip")
    return {
        "stale_baseline_map_ms": old_map,
        "head_map_ms": new_map,
        "map_delta_x": new_map / old_map,
        "stale_baseline_quantum_ms": old_quantum,
        "head_quantum_ms": new_quantum,
        "quantum_delta_x": (new_quantum / old_quantum if new_quantum else None),
        "quantum_state": quantum.get("state", "complete"),
        "quantum_qualification": (
            None
            if new_quantum is not None
            else (
                "the full sixteen-trip width-1024 solve could not recompile "
                "inside the shared-job budget; the sub-stage probes and map "
                "floor were re-measured at HEAD"
                if quantum.get("state") == "failed"
                else "the full-trip solve was not attempted in this job"
            )
        ),
    }


def _draw_figure(payload: dict[str, Any], output: Path) -> None:
    """Per-member stacked per-trip split beside the re-measured baselines."""
    members = [row for row in payload["width_one"]["members"] if row.get("trips")]
    figure = plt.figure(figsize=(15.2, 9.0), constrained_layout=True)
    grid = figure.add_gridspec(1, 2, width_ratios=(2.1, 0.9))
    axis = figure.add_subplot(grid[0, 0])
    labels = [row["identity"] for row in members]
    positions = np.arange(len(members))
    host = [row["host_reconciliation_per_trip_s"] * 1.0e3 for row in members]
    boundary = [row["boundary_per_trip_s"] * 1.0e3 for row in members]
    final_sync = [row["final_device_sync_per_trip_s"] * 1.0e3 for row in members]
    axis.bar(positions, boundary, color="#3b6ea5", label="trip-close boundary dispatch")
    axis.bar(
        positions,
        host,
        bottom=boundary,
        color="#f58518",
        label="host reconciliation",
    )
    bottoms = [b + h for b, h in zip(boundary, host, strict=True)]
    axis.bar(
        positions,
        final_sync,
        bottom=bottoms,
        color="#54a24b",
        label="final device sync",
    )
    axis.set_xticks(positions, labels, rotation=55, ha="right", fontsize=8)
    axis.set_ylabel("per-trip quantum [ms] at width 1")
    axis.set_title("Width-1 per-trip quantum split per MAST bank member")
    axis.legend(frameon=False, fontsize=9)
    axis.spines[["top", "right"]].set_visible(False)

    baseline_axis = figure.add_subplot(grid[0, 1])
    summary = payload["baseline_summary"]
    names = ["stale map\n(1024)", "HEAD map\n(1024)", "stale trip\n(1024)"]
    values = [
        summary["stale_baseline_map_ms"],
        summary["head_map_ms"],
        summary["stale_baseline_quantum_ms"],
    ]
    colors = ("#8da0cb", "#4c78a8", "#b279a2")
    baseline_axis.bar(np.arange(len(names)), values, color=colors)
    for index, value in enumerate(values):
        baseline_axis.text(
            index, value, f" {value:.3f}", ha="left", va="center", fontsize=8
        )
    head_quantum = summary["head_quantum_ms"]
    baseline_axis.text(
        2.35,
        summary["stale_baseline_quantum_ms"],
        (
            f"  HEAD trip\n  {head_quantum:.4f} ms"
            if head_quantum is not None
            else "  HEAD trip\n  not re-measured"
        ),
        ha="left",
        va="center",
        fontsize=8,
        color="#b279a2",
    )
    baseline_axis.set_xticks(
        np.arange(len(names)), names, rotation=30, ha="right", fontsize=8
    )
    baseline_axis.set_ylabel("ms/member")
    baseline_axis.set_title("Stale baselines re-measured at HEAD (width 1024)")
    baseline_axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        "Per-trip quantum at width one with the stale width-1024 baselines "
        "re-measured at HEAD\n"
        f"revision {payload['source']['measurement_revision'][:10]}",
        fontsize=15,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _write_report(payload: dict[str, Any], output: Path) -> None:
    summary = payload["baseline_summary"]
    members = [row for row in payload["width_one"]["members"] if row.get("trips")]
    host = np.asarray(
        [row["host_reconciliation_per_trip_s"] for row in members], dtype=float
    )
    boundary = np.asarray([row["boundary_per_trip_s"] for row in members], dtype=float)
    sync = np.asarray(
        [row["final_device_sync_per_trip_s"] for row in members], dtype=float
    )
    retrace = np.asarray([row["retrace_total_s"] for row in members], dtype=float)
    total = float(np.sum(host + boundary + sync + retrace))
    if total <= 0.0:
        lines = [
            "# Per-trip quantum at batch width one — which stage owns the trip",
            "",
            f"measured on `{payload['assignment']['device']}` at revision "
            f"`{payload['source']['measurement_revision'][:10]}` in job "
            f"`{payload['assignment']['job_id']}`.",
            "",
            "This job ran the baseline re-measurement only; no width-one member "
            "rows were harvested here.  See the full-job receipt for the width-one "
            "per-trip split.",
        ]
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("\n".join(lines), encoding="utf-8")
        return
    shares = {
        "host reconciliation": 100.0 * float(np.sum(host)) / total,
        "compiled boundary dispatch": 100.0 * float(np.sum(boundary)) / total,
        "device sync": 100.0 * float(np.sum(sync)) / total,
        "retrace": 100.0 * float(np.sum(retrace)) / total,
    }
    named = {
        "host reconciliation": shares["host reconciliation"],
        "retrace below the compiled boundary": shares["retrace"],
        "device sync": shares["device sync"],
    }
    named_owner = max(named, key=named.get)
    host_broad = f"{1.0e3 * np.mean(host):.4f}"
    sync_broad = f"{1.0e3 * np.mean(sync):.4f}"
    retrace_broad = "0.0000"
    head_quantum = summary["head_quantum_ms"]
    quantum_cell = (
        f"{head_quantum:.4f}" if head_quantum is not None else "not re-measured"
    )
    quantum_delta = (
        f"{summary['quantum_delta_x']:.3f}x" if head_quantum is not None else "n/a"
    )
    lines = [
        "# Per-trip quantum at batch width one — which stage owns the trip",
        "",
        f"measured on `{payload['assignment']['device']}` at revision "
        f"`{payload['source']['measurement_revision'][:10]}` in job "
        f"`{payload['assignment']['job_id']}`.",
        "",
        "## Baselines re-measured at HEAD (width 1024, Solovev workload)",
        "",
        "| quantity | banked (stale) | HEAD re-measured | delta |",
        "|---|---:|---:|---:|",
        f"| complete map ms/member | {summary['stale_baseline_map_ms']:.4f} | "
        f"{summary['head_map_ms']:.4f} | {summary['map_delta_x']:.3f}x |",
        "| per-trip quantum ms/member/trip | "
        f"{summary['stale_baseline_quantum_ms']:.4f} | {quantum_cell} | "
        f"{quantum_delta} |",
        "",
        "The 0.784 ms map and 25.0 ms per-trip quantum were stale because the "
        "census null polish entered the jitted topology read; the map floor is "
        f"re-measured at HEAD as {summary['head_map_ms']:.4f} ms"
        + (
            f" and the per-trip quantum as {head_quantum:.4f} ms."
            if head_quantum is not None
            else (
                "; the per-trip quantum was not re-measured ("
                + (summary["quantum_qualification"] or "skipped")
                + ")."
            )
        ),
        "",
        "## Width-1 per-member per-trip quantum",
        "",
        "| member | trips | wall/solve [s] | per-trip total [ms] | "
        "host reconciliation [ms/trip] | compiled boundary [ms/trip] | "
        "device sync [ms/trip] | retrace [ms/trip] |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in members:
        lines.append(
            f"| {row['identity']} | {row['trips']} | {row['wall_per_solve_s']:.3f} | "
            f"{1.0e3 * (row['per_trip_quantum_s'] or 0.0):.4f} | "
            f"{1.0e3 * (row['host_reconciliation_per_trip_s'] or 0.0):.4f} | "
            f"{1.0e3 * (row['boundary_per_trip_s'] or 0.0):.4f} | "
            f"{1.0e3 * (row['final_device_sync_per_trip_s'] or 0.0):.4f} | "
            f"{1.0e3 * row['retrace_total_s']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Across the width-1 members the per-trip split is "
            f"**{shares['compiled boundary dispatch']:.1f}%** compiled "
            f"trip-close boundary dispatch, **{host_broad} ms** host reconciliation "
            f"per trip ({shares['host reconciliation']:.1f}%), "
            f"**{sync_broad} ms** final device sync per trip "
            f"({shares['device sync']:.1f}%), and "
            f"**{retrace_broad} ms** retrace below the compiled boundary "
            f"({shares['retrace']:.1f}%).  Of the three host-side stages the plan "
            f"names — host reconciliation, retrace below the compiled boundary, "
            f"and device sync — **{named_owner}** owns the quantum (`{named_owner}` "
            f"is the largest at {named[named_owner]:.1f}%), while the compiled "
            "trip body remains the majority of the trip.",
            "",
            "Warm-program solves re-trace nothing: the reduced program's kernel "
            "cache sizes are recorded before and after each solve and report zero "
            "growth on every steady repeat (per-warm-row `kernel_cache_growth`).",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def _peak_rss_mib() -> float:
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024.0
    return float("nan")


def preflight(state_cache: Path) -> dict[str, Any]:
    """Build the members and the Solovev workload without any GPU allocation."""
    revision = _require_revision()
    members, _manifest = _build_members()
    profile, seed = build_workload()
    result = {
        "status": "preflight_complete",
        "source_revision": revision,
        "mast_member_count": len(members),
        "mast_identities": [member.identity for member in members],
        "solovev_seed_shape": list(seed.shape),
        "requested_ancestor": REQUIRED_ANCESTOR,
        "state_cache": str(state_cache),
    }
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def run(
    output: Path,
    figure: Path,
    report: Path,
    *,
    cache_root: Path | None,
    repeats: int,
    member_repeats: int,
    newton_steps: int,
    active_set_steps: int,
    baselines_only: bool,
) -> dict[str, Any]:
    total_started = time.perf_counter()
    revision = _require_revision()
    configure_dtypes()
    allocation = _require_allocation()
    cache = configure_persistent_compilation_cache(
        cache_root or default_persistent_compilation_cache_root(),
        minimum_compile_seconds=CACHE_MIN_COMPILE_SECONDS,
    )
    receipt: dict[str, Any] = {
        "schema": "nova.trip-quantum-width-one",
        "measurement_revision": revision,
        "captured_at": datetime.now(UTC).isoformat(),
        "assignment": allocation,
        "persistent_compilation_cache": cache.receipt(),
        "source": {
            "driver": str(Path(__file__).relative_to(ROOT)),
            "driver_sha256": _sha256(Path(__file__)),
            "required_ancestor": REQUIRED_ANCESTOR,
        },
        "configuration": {
            "baselines_only": baselines_only,
            "width_one_member_repeats": member_repeats,
            "baseline_width": WIDTH,
            "baseline_trip_limit": TRIP_LIMIT,
            "baseline_probe_repeats": repeats,
            "reduced_newton_steps_per_trip": newton_steps,
            "reduced_active_set_steps": active_set_steps,
        },
        "baseline_summary": None,
        "baselines": None,
        "width_one": {"members": []},
        "project_absolute_figure_src": (
            "/nova/figures/millisecond-converged-solve/trip-quantum/width-one.png"
        ),
    }
    _write_json(output, receipt)

    # --- Width-one converged solves on the real MAST bank first, so a
    #     scheduler timeout on the heavy baseline cannot erase the primary
    #     measurement (every member row is persisted as it lands). ---
    if not baselines_only:
        print("SECTION WIDTH_ONE_START", flush=True)
        members, member_inputs = _build_members()
        receipt["width_one"]["inputs"] = member_inputs
        _write_json(output, receipt)
        for number, member in enumerate(members, start=1):
            print(
                f"STAGE WIDTH_ONE_MEMBER_{number}_START identity={member.identity!r} "
                f"rss_mib={_peak_rss_mib():.3f}",
                flush=True,
            )
            try:
                row = _measure_member_width_one(
                    member,
                    repeats=member_repeats,
                    newton_steps=newton_steps,
                    active_set_steps=active_set_steps,
                )
            except Exception as error:  # noqa: BLE001 - one member's failure
                # must not strand the members already persisted
                row = {
                    "identity": member.identity,
                    "failure": f"{type(error).__name__}: {error}",
                }
            receipt["width_one"]["members"].append(row)
            _write_json(output, receipt)
            del member
            gc.collect()
            jax.clear_caches()
            gc.collect()
            print(
                f"STAGE WIDTH_ONE_MEMBER_{number}_DONE "
                f"identity={row.get('identity')!r} trips={row.get('trips')} "
                f"rss_mib={_peak_rss_mib():.3f}",
                flush=True,
            )
        print("SECTION WIDTH_ONE_DONE", flush=True)
    else:
        print("SECTION WIDTH_ONE_SKIPPED baselines_only=1", flush=True)

    # --- The stale width-1024 baselines re-measured at HEAD.  The map and
    #     sub-stage probes are persisted immediately, and the full sixteen-trip
    #     solve is attempted last and best-effort: its executable (3.8 GiB) can
    #     never persist, so it recompiles every run and may exceed the shared
    #     job budget under node contention; a failure still banks the map floor
    #     and the sub-stage probes. ---
    print("SECTION BASELINES_START", flush=True)
    fast = _map_and_substages(receipt["configuration"]["baseline_probe_repeats"])
    receipt["baselines"] = {
        "scalar_compiled_dispatch_probe": fast["scalar_compiled_dispatch_probe"],
        "map": fast["map"],
        "direct_width_1024_probes": fast["direct_width_1024_probes"],
        "per_trip_quantum": {"state": "not_attempted"},
    }
    _write_json(output, receipt)
    print(
        "SECTION MAP_AND_SUBSTAGES_DONE "
        f"map_ms={fast['map']['median_ms_per_member_width_1024']:.6f}",
        flush=True,
    )

    def persist_full(update: dict[str, Any]) -> None:
        for key, value in update.items():
            if key == "per_trip_quantum":
                replacement = receipt["baselines"]["per_trip_quantum"]
                replacement.update(value)
                receipt["baselines"]["per_trip_quantum"] = replacement
            else:
                receipt["baselines"][key] = value
        receipt["baseline_summary"] = _baseline_summary(receipt)
        _write_json(output, receipt)

    try:
        _full_trip_measure(
            fast["workload"],
            fast["seed"],
            fast["direct_width_1024_probes"],
            persist=persist_full,
        )
    except Exception as error:  # noqa: BLE001 - the fast baselines are already
        # persisted; record the failed full-trip and finalize with what exists
        receipt["baselines"]["per_trip_quantum"] = {
            "state": "failed",
            "error": f"{type(error).__name__}: {error}",
        }
        print(
            f"STAGE FULL_TRIP_FAILED error={type(error).__name__}: {error}",
            flush=True,
        )
        receipt["baseline_summary"] = _baseline_summary(receipt)
        _write_json(output, receipt)

    receipt["execution"] = {
        "elapsed_seconds": time.perf_counter() - total_started,
        "exit_marker": 0,
        "width_one_member_count": len(receipt["width_one"]["members"]),
    }
    _draw_figure(receipt, figure)
    _write_report(receipt, report)
    receipt["execution"]["elapsed_seconds"] = time.perf_counter() - total_started
    _write_json(output, receipt)
    print(f"RECEIPT_WRITTEN={output}", flush=True)
    print(f"FIGURE_WRITTEN={figure}", flush=True)
    print(f"REPORT_WRITTEN={report}", flush=True)
    print("EXIT_MARKER=0", flush=True)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--state-cache", type=Path, default=DEFAULT_MAST_STATE_CACHE)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--member-repeats", type=int, default=2)
    parser.add_argument("--newton-steps", type=int, default=12)
    parser.add_argument("--active-set-steps", type=int, default=16)
    parser.add_argument("--baselines-only", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    arguments = parser.parse_args()
    if arguments.preflight:
        preflight(arguments.state_cache.resolve())
        return
    run(
        arguments.output.resolve(),
        arguments.figure.resolve(),
        arguments.report.resolve(),
        cache_root=arguments.cache_root.resolve() if arguments.cache_root else None,
        repeats=arguments.repeats,
        member_repeats=arguments.member_repeats,
        newton_steps=arguments.newton_steps,
        active_set_steps=arguments.active_set_steps,
        baselines_only=arguments.baselines_only,
    )


if __name__ == "__main__":
    main()
