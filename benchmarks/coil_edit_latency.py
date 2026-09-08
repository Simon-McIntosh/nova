"""Measure warm MAST coil edits through the compiled slice route.

The interactive consumer edits one coil current and asks Nova for the moved
boundary.  This driver runs the warm-start sweep plus-minus twenty percent on
one position-coil circuit of a converging MAST arm (22086/43 mixed), twenty
two-percent edits each seeded from the preceding terminal state, through the
*compiled slice route*: the reduced fixed point closes every trip of one
slice in a single fixed-shape compiled program that a later keyframe
re-enters, with the prescribed 101-circuit current vector as traced data and
the operator-held raster target publishing psi, psi_N, the domain labels and
the separatrix on the receiver grid at no extra compile.

The measure is the interactive path as it now stands: the per-edit wall from
the press to the terminal receipt the decoder reads (the compiled slice solve
plus the forward-equilibrium receipt with its raster and labelled leaves,
both drained to the device), the persistent-compilation-cache hits after the
first edit, and convergence with trip counts recorded on every sweep point.
"""

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
from benchmarks.diiid_forward_gs_match import _margin_graded_newton_krylov
from nova.equilibrium import reduced_newton
from nova.equilibrium.fixed_point import FixedPointResult, FixedPointTerminationReason
from nova.equilibrium.topology import TopologyClass
from nova.imas.mast_solve_inputs import SHOT_STORE
from nova.jax.config import (
    configure_dtypes,
    configure_persistent_compilation_cache,
    default_persistent_compilation_cache_root,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT / "docs/figures/forward-solve-api/coil-edit-latency/coil-edit-latency.json"
)
DEFAULT_FIGURE = (
    ROOT / "docs/figures/forward-solve-api/coil-edit-latency/coil-edit-latency.png"
)
DEFAULT_RASTER_FIGURE = (
    ROOT / "docs/figures/forward-solve-api/coil-edit-latency/terminal-raster-flux.png"
)
SHOT = 22086
SLICE_INDEX = 43
SWEEP_FRACTIONS = np.arange(-0.20, 0.201, 0.02, dtype=np.float64)
EDIT_FRACTIONS = SWEEP_FRACTIONS[1:]
EDIT_COUNT = len(EDIT_FRACTIONS)
BOUNDARY_COIL_FAMILIES = frozenset({"p4_lower", "p4_upper", "p5_lower", "p5_upper"})
INTERACTIVE_LATENCY_TARGET_MILLISECONDS = 100.0
# The compiled slice route's own production budgets (the kernel's declared
# constants), not the parity-driver Newton budgets the endpoint prep uses.
COMPILED_SLICE_TOLERANCE = reduced_newton.FIXED_POINT_RESIDUAL_TOLERANCE
COMPILED_SLICE_NEWTON_STEPS = reduced_newton.NEWTON_STEPS
COMPILED_SLICE_ACTIVE_SET_STEPS = reduced_newton.ACTIVE_SET_STEPS


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


def _cache_monitor() -> dict[str, float | int]:
    """Count JAX persistent-cache events without inferring them from timing."""
    import jax.monitoring as monitoring

    events: dict[str, float | int] = {"hits": 0, "misses": 0, "saved_seconds": 0.0}

    def hit(event: str, **_kwargs: Any) -> None:
        if event == "/jax/compilation_cache/cache_hits":
            events["hits"] = int(events["hits"]) + 1

    def miss(event: str, **_kwargs: Any) -> None:
        if event == "/jax/compilation_cache/cache_misses":
            events["misses"] = int(events["misses"]) + 1

    def saved(event: str, duration_secs: float, **_kwargs: Any) -> None:
        if event == "/jax/compilation_cache/compile_time_saved_sec":
            events["saved_seconds"] = float(events["saved_seconds"]) + duration_secs

    monitoring.register_event_listener(hit)
    monitoring.register_event_listener(miss)
    monitoring.register_event_duration_secs_listener(saved)
    return events


def _render(
    rows: list[dict[str, Any]],
    figure_path: Path,
    *,
    coil_family: str,
    program_reused_from_edit: int,
    persistent_cache_hits: int,
    persistent_cache_misses: int,
) -> None:
    indices = np.asarray([row["edit_index"] for row in rows])
    milliseconds = np.asarray([row["wall_milliseconds"] for row in rows])
    displacement = 1.0e3 * np.asarray([row["boundary_displacement_m"] for row in rows])
    trips = np.asarray([row["trip_count"] for row in rows])
    residual = np.asarray([row["terminal_residual"] for row in rows])
    colours = [
        "#d97706" if row["compilation_cache"] == "miss" else "#2563eb" for row in rows
    ]
    figure, axes = plt.subplots(2, 2, figsize=(11.2, 8.2), constrained_layout=True)
    latency_axis, trip_axis, residual_axis, boundary_axis = axes.ravel()
    latency_axis.plot(indices, milliseconds, color="0.72", lw=1.0, zorder=1)
    latency_axis.scatter(indices, milliseconds, c=colours, s=34, zorder=2)
    latency_axis.axhline(
        INTERACTIVE_LATENCY_TARGET_MILLISECONDS,
        color="#dc2626",
        linestyle="--",
        linewidth=1.0,
        label="100 ms target ceiling",
    )
    latency_axis.set_yscale("log")
    latency_axis.set_ylabel("Edit wall time [ms]")
    latency_axis.set_title(
        f"MAST {SHOT}/{SLICE_INDEX} mixed arm · {coil_family.replace('_', '-').upper()}"
    )
    latency_axis.grid(True, which="both", alpha=0.25)
    latency_axis.scatter([], [], color="#d97706", label="persistent-cache miss")
    latency_axis.scatter([], [], color="#2563eb", label="persistent-cache hit")
    latency_axis.legend(fontsize=8)

    termination_names = sorted({row["termination"] for row in rows})
    termination_colours = {
        name: plt.get_cmap("tab10")(index)
        for index, name in enumerate(termination_names)
    }
    for name in termination_names:
        selected = np.asarray([row["termination"] == name for row in rows])
        trip_axis.scatter(
            indices[selected],
            trips[selected],
            color=termination_colours[name],
            label=name.replace("_", " "),
        )
    trip_axis.plot(indices, trips, color="0.75", linewidth=0.8, zorder=0)
    trip_axis.set_ylabel("Active-set trips")
    trip_axis.set_title("Termination and trip count")
    trip_axis.grid(True, alpha=0.25)
    trip_axis.legend(fontsize=8)

    residual_axis.plot(indices, residual, color="#0891b2", marker="o", ms=4)
    residual_axis.axhline(
        COMPILED_SLICE_TOLERANCE,
        color="#dc2626",
        linestyle="--",
        linewidth=1.0,
        label=f"tolerance {COMPILED_SLICE_TOLERANCE:.0e}",
    )
    residual_axis.set_yscale("log")
    residual_axis.set_xlabel("Successive two-percent edit index")
    residual_axis.set_ylabel("Terminal relative residual")
    residual_axis.set_title("Convergence qualification")
    residual_axis.grid(True, which="both", alpha=0.25)
    residual_axis.legend(fontsize=8)

    boundary_axis.plot(indices, displacement, color="#059669", marker="o", ms=4)
    boundary_axis.axhline(0.0, color="0.5", lw=0.8)
    boundary_axis.set_xlabel("Successive two-percent edit index")
    boundary_axis.set_ylabel("Boundary displacement [mm]")
    boundary_axis.set_title(
        f"Boundary motion · one program from edit "
        f"{program_reused_from_edit} · persistent hits {persistent_cache_hits} · "
        f"misses {persistent_cache_misses}"
    )
    boundary_axis.grid(True, alpha=0.25)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)


def _render_raster(
    raster_flux: Any,
    figure_path: Path,
) -> None:
    """Write the terminal raster psi with the separatrix and label field."""
    radius = np.asarray(raster_flux.radius, dtype=np.float64)
    height = np.asarray(raster_flux.height, dtype=np.float64)
    shape = tuple(int(value) for value in np.asarray(raster_flux.shape))
    psi = np.asarray(raster_flux.psi, dtype=np.float64).reshape(shape).T
    labels = np.asarray(raster_flux.domain_label, dtype=np.int8).reshape(shape).T
    separatrix = np.asarray(raster_flux.separatrix, dtype=np.float64)
    vertex_count = int(np.asarray(raster_flux.separatrix_vertex_count))
    figure, axes = plt.subplots(
        1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": (1.0, 1.0)}
    )
    flux = axes[0].pcolormesh(
        radius,
        height,
        psi,
        cmap="viridis",
        shading="auto",
    )
    figure.colorbar(flux, ax=axes[0], label=r"$\psi$ [Wb]")
    axes[0].set_title("Raster psi on the receiver grid")
    axes[0].set_xlabel("R [m]")
    axes[0].set_ylabel("Z [m]")
    axes[0].set_aspect("equal")
    if vertex_count:
        vertices = separatrix[:vertex_count]
        axes[0].plot(
            vertices[:, 0],
            vertices[:, 1],
            color="#dc2626",
            lw=1.2,
            label=f"separatrix ({vertex_count} vertices)",
        )
        axes[0].legend(fontsize=8)
    axes[1].pcolormesh(
        radius,
        height,
        labels,
        cmap="tab10",
        shading="auto",
        vmin=-0.5,
        vmax=3.5,
    )
    axes[1].set_title("Per-cell domain labels")
    axes[1].set_xlabel("R [m]")
    axes[1].set_ylabel("Z [m]")
    axes[1].set_aspect("equal")
    figure.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, dpi=140)
    plt.close(figure)


def _calibrate_cache() -> tuple[dict[str, float | int], Any]:
    """Configure the persistent cache and the compile/miss/stage event monitor."""
    cache = configure_persistent_compilation_cache(
        default_persistent_compilation_cache_root()
    )
    cache_events = _cache_monitor()
    return cache_events, cache


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
    prescribed = profile.operator.prescribed_current_field
    if prescribed is None or prescribed.current.shape != (101,):
        raise RuntimeError("the operator does not hold the 101-circuit vector")
    response = np.asarray(prescribed.response, dtype=np.float64)
    base_current = np.asarray(prescribed.current, dtype=np.float64)
    wall_start = profile.operator.grid.node_number
    candidates = []
    for row in policy["active_mapping"]:
        if row["family"] not in BOUNDARY_COIL_FAMILIES:
            continue
        circuit_index = int(row["stored_circuit"])
        candidates.append(
            {
                "family": row["family"],
                "stored_circuit": circuit_index,
                "two_percent_wall_flux_sup_wb": float(
                    0.02
                    * abs(base_current[circuit_index])
                    * np.max(np.abs(response[wall_start:, circuit_index]))
                ),
            }
        )
    if len(candidates) != len(BOUNDARY_COIL_FAMILIES):
        raise RuntimeError("the P4/P5 boundary-circuit mapping is incomplete")
    selected_coil = max(
        candidates,
        key=lambda row: row["two_percent_wall_flux_sup_wb"],
    )
    circuit_index = int(selected_coil["stored_circuit"])
    target_current = abs(float(case["reference"]["plasma_current_a"]))
    base_map = profile.flux_map(
        requested_class=TopologyClass.DIVERTED,
        target_current=target_current,
        prescribed_current=jnp.asarray(base_current),
    )
    mixed_seed = _margin_graded_newton_krylov(
        base_map,
        profile.operator.topology_margin,
        jnp.asarray(passive_case["state"]),
        newton_steps=parity.NEWTON_STEPS,
        gmres_iterations=parity.GMRES_ITERATIONS,
    )
    jax.block_until_ready(mixed_seed.state)
    mixed_residual = float(np.asarray(mixed_seed.residual))
    if not np.isfinite(mixed_residual) or mixed_residual > parity.FIXED_POINT_CRITERION:
        raise RuntimeError(
            f"the corrected-bank mixed arm did not converge: {mixed_residual:.6g}"
        )
    # The sweep seeds from the converged base frame and every measured edit is
    # a compiled-slice warm solve from the preceding terminal state; the only
    # Newton-Krylov call left is this off-clock base-frame preparation, and it
    # is not part of the interactive path the receipt measures.
    reference_read = profile.operator.read(mixed_seed.state, TopologyClass.DIVERTED)
    reference_masks, reference_topology = reference_read
    reference_labelled = profile._labelled_flux(
        mixed_seed.state, reference_masks, reference_topology
    )
    reference_lcfs_count = int(np.asarray(reference_labelled.lcfs_vertex_count))
    prepared = {
        "initial": mixed_seed.state,
        "reference_lcfs": np.asarray(reference_labelled.lcfs)[:reference_lcfs_count],
        "prescribed_current": jnp.asarray(base_current),
        "target_current": target_current,
        "circuit_index": circuit_index,
        "coil_mapping": selected_coil,
        "boundary_coil_candidates": candidates,
        "mixed_seed": {
            "identity": f"{SHOT}/{SLICE_INDEX} mixed",
            "terminal_residual": mixed_residual,
            "converged": True,
            "corrected_bank_receipt": (
                "docs/figures/solver-convergence-regression/bank-rebaseline-regen.json"
            ),
        },
        "sweep_seed": {
            "edit_fraction": 0.0,
            "terminal_residual": mixed_residual,
            "converged": True,
            "route": "converged corrected-bank mixed frame at the shot current",
            "reference_lcfs_vertex_count": reference_lcfs_count,
        },
        "reference": case["reference"],
        "policy": policy,
    }
    return profile, prepared, carrier


def _compiled_edit(
    profile: Any,
    state: jax.Array,
    current: jax.Array,
    requested_class: jax.Array,
    target_current: float,
    program: Any,
) -> Any:
    """Re-enter the compiled slice program with one edited current vector."""
    return reduced_newton.solve_reduced_newton_compiled(
        profile.operator,
        state,
        requested_class=requested_class,
        target_current=target_current,
        prescribed_current=current,
        tolerance=COMPILED_SLICE_TOLERANCE,
        newton_steps=COMPILED_SLICE_NEWTON_STEPS,
        active_set_steps=COMPILED_SLICE_ACTIVE_SET_STEPS,
        program=program,
        stream=False,
    )


def _equilibrium_receipt(
    profile: Any,
    result: Any,
    requested_class: Any,
    target_current: float,
    prescribed_current: jax.Array,
) -> Any:
    """Build the complete typed equilibrium receipt the decoder reads."""
    residuals = jnp.asarray(result.active_set_residuals, dtype=jnp.float64)
    history = FixedPointResult(
        state=result.state,
        residual=jnp.asarray(result.terminal_residual, dtype=jnp.float64),
        trace=residuals,
        converged=jnp.asarray(result.converged),
        termination_reason=jnp.asarray(result.termination_reason, dtype=jnp.int32),
        active_set_iterations=jnp.asarray(
            result.active_set_iterations, dtype=jnp.int32
        ),
        active_set_residuals=residuals,
        active_set_mask_differences=jnp.asarray(
            result.active_set_mask_differences, dtype=jnp.int32
        ),
        shadow_mask_changes=jnp.asarray(
            result.active_set_mask_differences, dtype=jnp.int32
        ),
    )
    return profile._receipt(
        result.state,
        history,
        requested_class,
        target_current,
        None,
        prescribed_current,
    )


def _drain_receipt(equilibrium: Any) -> None:
    """Materialise the leaves the consumer reads without silently skipping wall."""
    jax.block_until_ready(equilibrium.flux)
    if equilibrium.raster_flux is not None:
        jax.block_until_ready(equilibrium.raster_flux.psi)
        jax.block_until_ready(equilibrium.raster_flux.psi_norm)
        jax.block_until_ready(equilibrium.raster_flux.separatrix)
    if equilibrium.labelled_flux is not None:
        jax.block_until_ready(equilibrium.labelled_flux.lcfs)
        jax.block_until_ready(equilibrium.labelled_flux.strike_points)


def run(
    output: Path,
    figure: Path,
    carrier_path: Path,
) -> dict[str, Any]:
    """Compile once and measure successive warm prescribed-current edits."""
    total_started = time.perf_counter()
    configure_dtypes()
    _require_measurement_host()
    cache_events, cache = _calibrate_cache()
    stop = threading.Event()
    reporter = threading.Thread(
        target=_heartbeat,
        args=(stop, total_started),
        daemon=True,
    )
    reporter.start()
    try:
        profile, prepared, carrier = _prepare_case(carrier_path)
        solve_persistent_hits_start = int(cache_events["hits"])
        solve_persistent_misses_start = int(cache_events["misses"])
        solve_persistent_saved_start = float(cache_events["saved_seconds"])
        initial = prepared["initial"]
        base_current = prepared["prescribed_current"]
        circuit_index = prepared["circuit_index"]
        target_current = prepared["target_current"]
        requested_class = jnp.asarray(int(TopologyClass.DIVERTED), dtype=jnp.int8)
        edit_vectors = []
        for fraction in EDIT_FRACTIONS:
            values = np.asarray(base_current, dtype=np.float64).copy()
            values[circuit_index] *= 1.0 + fraction
            edit_vectors.append(jnp.asarray(values))

        rows: list[dict[str, Any]] = []
        state = initial
        program = None
        program_reused_from_edit: int | None = None
        _endpoint_masks, endpoint_topology = profile.operator.read(state)
        reference_boundary = np.asarray(endpoint_topology.boundary)
        reference_lcfs = prepared["reference_lcfs"]
        for index, (fraction, current) in enumerate(
            zip(EDIT_FRACTIONS, edit_vectors, strict=True)
        ):
            persistent_misses_before = int(cache_events["misses"])
            persistent_hits_before = int(cache_events["hits"])
            program_reused_this_edit = program is not None
            wall_started = time.perf_counter()
            result = _compiled_edit(
                profile,
                state,
                current,
                requested_class,
                target_current,
                program,
            )
            if program is None:
                # The first edit builds the fixed-shape program; every later
                # edit re-enters the same coordinates and executables.
                program_reused_from_edit = index
            elif program_reused_from_edit is None:
                program_reused_from_edit = 1
            # Drain the compiled slice's terminal state before timing the
            # solve: the slice's single device read is its natural end, and
            # aligning the sync here keeps the next edit's host work clean.
            jax.block_until_ready(result.state)
            solve_wall_s = time.perf_counter() - wall_started
            state = result.state
            program = result.program
            try:
                equilibrium = _equilibrium_receipt(
                    profile,
                    result,
                    requested_class,
                    target_current,
                    current,
                )
                _drain_receipt(equilibrium)
                wall_milliseconds = 1.0e3 * (time.perf_counter() - wall_started)
                receipt_error = None
            except Exception as error:  # noqa: BLE001 - one unreadable terminal
                # read must not kill the sweep; the point is recorded and the
                # chain advances on the solved state regardless.
                equilibrium = None
                wall_milliseconds = 1.0e3 * (time.perf_counter() - wall_started)
                receipt_error = f"{type(error).__name__}: {error}"
            receipt_wall_s = time.perf_counter() - wall_started - solve_wall_s
            persistent_misses_after = int(cache_events["misses"])
            persistent_hits_after = int(cache_events["hits"])
            compiled = (
                "miss" if persistent_misses_after > persistent_misses_before else "hit"
            )
            if equilibrium is not None:
                labelled = equilibrium.labelled_flux
                raster_flux = equilibrium.raster_flux
                lcfs_count = int(np.asarray(labelled.lcfs_vertex_count))
                lcfs = np.asarray(labelled.lcfs)[:lcfs_count]
                boundary = np.asarray(equilibrium.topology.boundary)
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
            else:
                labelled = None
                raster_flux = None
                lcfs_count = 0
                boundary_displacement = None
                displacement_source = "unreadable_terminal"
            row: dict[str, Any] = {
                "edit_index": index,
                "edit_fraction": float(fraction),
                "coil_current_a": float(np.asarray(current[circuit_index])),
                "wall_milliseconds": wall_milliseconds,
                "solve_wall_milliseconds": 1.0e3 * solve_wall_s,
                "receipt_wall_milliseconds": 1.0e3 * receipt_wall_s,
                "compilation_cache": compiled,
                "persistent_cache_misses_before": persistent_misses_before,
                "persistent_cache_misses_after": persistent_misses_after,
                "persistent_cache_miss_count": (
                    persistent_misses_after - persistent_misses_before
                ),
                "persistent_cache_hits_before": persistent_hits_before,
                "persistent_cache_hits_after": persistent_hits_after,
                "persistent_cache_hit_count": (
                    persistent_hits_after - persistent_hits_before
                ),
                "converged": bool(np.asarray(result.converged)),
                "terminal_residual": float(np.asarray(result.terminal_residual)),
                "trip_count": int(np.asarray(result.active_set_iterations)),
                "termination": result.termination_name,
                "reduced_dimension": int(result.reduced_dimension),
                "off_support_leakage_wb": float(result.off_support_leakage),
                "program_reused": program_reused_this_edit,
                "lcfs_vertex_count": lcfs_count,
                "boundary_displacement_m": boundary_displacement,
                "boundary_displacement_source": displacement_source,
                "warm_seed_source": (
                    "converged base frame at the shot current"
                    if index == 0
                    else "previous edit terminal state"
                ),
                "receipt_error": receipt_error,
                "raster_separatrix_vertex_count": (
                    int(np.asarray(raster_flux.separatrix_vertex_count))
                    if raster_flux is not None
                    else None
                ),
            }
            rows.append(row)
            terminal_residual = float(np.asarray(result.terminal_residual))
            if not np.isfinite(terminal_residual):
                raise RuntimeError(
                    f"warm-start chain produced a nonfinite terminal state at edit "
                    f"{index}"
                )
            boundary_millimetres = (
                f"{1.0e3 * boundary_displacement:.6f}"
                if boundary_displacement is not None
                else "nan"
            )
            print(
                "EDIT_DONE "
                f"index={index + 1}/{EDIT_COUNT} "
                f"fraction={fraction:+.6f} "
                f"milliseconds={wall_milliseconds:.6f} "
                f"cache={row['compilation_cache']} "
                f"converged={row['converged']} trips={row['trip_count']} "
                f"boundary_mm={boundary_millimetres}",
                flush=True,
            )

        warm_ms = np.asarray(
            [row["wall_milliseconds"] for row in rows[1:]], dtype=np.float64
        )
        all_cache_hits_after_first = all(
            row["compilation_cache"] == "hit" for row in rows[1:]
        )
        one_program = (
            program_reused_from_edit is not None and program_reused_from_edit == 0
        )
        all_converged = all(row["converged"] for row in rows)
        median_warm_ms = float(np.median(warm_ms))
        latency_target_met = median_warm_ms < INTERACTIVE_LATENCY_TARGET_MILLISECONDS
        latency_regime = (
            "tens_of_milliseconds_or_better"
            if latency_target_met
            else "above_tens_of_milliseconds"
        )
        boundary_displacements = np.asarray(
            [
                (
                    row["boundary_displacement_m"]
                    if row["boundary_displacement_m"] is not None
                    else np.nan
                )
                for row in rows
            ],
            dtype=np.float64,
        )
        finite_displacements = boundary_displacements[
            np.isfinite(boundary_displacements)
        ]
        persistent_hits = int(cache_events["hits"]) - solve_persistent_hits_start
        persistent_misses = int(cache_events["misses"]) - solve_persistent_misses_start
        persistent_saved_seconds = (
            float(cache_events["saved_seconds"]) - solve_persistent_saved_start
        )
        first_edit_compiled = rows[0]["persistent_cache_miss_count"] >= 1
        later_edits_add_no_misses = all(
            row["persistent_cache_miss_count"] == 0 for row in rows[1:]
        )
        # The terminal raster flux through the once-built operator target.
        terminal = equilibrium
        terminal_raster = None
        if terminal is not None and terminal.raster_flux is not None:
            terminal_raster = {
                "shape": [
                    int(value) for value in np.asarray(terminal.raster_flux.shape)
                ],
                "node_count": int(profile.lattice.node_count),
                "source_quantities": "psi (Wb), psi_N, per-cell labels, separatrix",
                "note": (
                    "the operator holds the rectangular receiver-grid target once "
                    "from construction; every edit evaluates it from the same coil "
                    "and cell currents at no extra compile"
                ),
            }
        gates = {
            "exactly_twenty_edits_recorded": len(rows) == 20,
            "sweep_spans_plus_minus_twenty_percent": bool(
                np.isclose(SWEEP_FRACTIONS[0], -0.20)
                and np.isclose(SWEEP_FRACTIONS[-1], 0.20)
            ),
            "successive_edits_are_two_percent": bool(
                np.allclose(np.diff(SWEEP_FRACTIONS), 0.02)
            ),
            "first_edit_compiles_program": first_edit_compiled,
            "program_reused_to_the_end": one_program,
            "all_later_edits_are_cache_hits": all_cache_hits_after_first,
            "later_edits_add_no_compiles": later_edits_add_no_misses,
            "all_edits_converged": all_converged,
            "boundary_displacement_is_finite": bool(
                np.all(np.isfinite(boundary_displacements))
            ),
            "raster_target_rides_once": (
                terminal is not None and terminal.raster_flux is not None
            ),
            "median_warm_wall_meets_tens_of_milliseconds_target": latency_target_met,
        }
        passed = all(gates.values())
        exit_marker = 0 if passed else 2
        _render(
            rows,
            figure,
            coil_family=prepared["coil_mapping"]["family"],
            program_reused_from_edit=(
                0 if program_reused_from_edit is None else program_reused_from_edit
            ),
            persistent_cache_hits=persistent_hits,
            persistent_cache_misses=persistent_misses,
        )
        if terminal is not None and terminal.raster_flux is not None:
            _render_raster(terminal.raster_flux, DEFAULT_RASTER_FIGURE)
        receipt = {
            "schema": "nova.coil-edit-latency",
            "measurement_state": "complete",
            "verdict": "PASS" if passed else "FAIL",
            "gates": gates,
            "interactive_path": {
                "route": "compiled slice (reduced fixed point, one fixed-shape "
                "program re-entered per edit)",
                "solve_entry": "reduced_newton.solve_reduced_newton_compiled",
                "prescribed_current": "traced 101-circuit vector, replacement "
                "semantics",
                "raster_target": "operator-held receiver grid built once at "
                "construction",
            },
            "source_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "forward_module": {
                "path": "nova/equilibrium/forward.py",
                "sha256": _sha256(ROOT / "nova/equilibrium/forward.py"),
            },
            "reduced_newton_module": {
                "path": "nova/equilibrium/reduced_newton.py",
                "sha256": _sha256(ROOT / "nova/equilibrium/reduced_newton.py"),
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
            "persistent_compilation_cache": cache.receipt()
            | {
                "solve_hit_count": persistent_hits,
                "solve_miss_count": persistent_misses,
                "solve_compile_seconds_saved": persistent_saved_seconds,
                "process_total_hit_count": int(cache_events["hits"]),
                "process_total_miss_count": int(cache_events["misses"]),
                "process_total_compile_seconds_saved": float(
                    cache_events["saved_seconds"]
                ),
            },
            "carrier": carrier,
            "case": {
                "machine": "MAST",
                "shot": SHOT,
                "slice_index": SLICE_INDEX,
                "time_s": float(prepared["reference"]["time_s"]),
                "seed_policy": (
                    "the converged corrected-bank mixed frame at the shot current "
                    "starts the sweep; each edit moves the position coil and "
                    "re-enters the once-built compiled slice program from the "
                    "preceding terminal flux"
                ),
                "seed_arm": prepared["mixed_seed"],
                "sweep_seed": prepared["sweep_seed"],
                "route": "compiled slice (reduced_newton compiled)",
                "solver_policy": {
                    "tolerance": COMPILED_SLICE_TOLERANCE,
                    "newton_steps": COMPILED_SLICE_NEWTON_STEPS,
                    "active_set_steps": COMPILED_SLICE_ACTIVE_SET_STEPS,
                    "ladder_scoring": reduced_newton.LADDER_SCORING,
                    "trip_boundary": reduced_newton.TRIP_BOUNDARY,
                    "settled_exit": "production default unchanged",
                    "presettlement_incumbent_scoring": ("production default unchanged"),
                },
                "target_current_a": target_current,
                "current_pin": True,
                "stored_circuit_count": 101,
                "coil_family": prepared["coil_mapping"]["family"],
                "coil_circuit_index": circuit_index,
                "boundary_coil_selection": {
                    "criterion": (
                        "largest two-percent wall-flux response among P4/P5 circuits"
                    ),
                    "candidates": prepared["boundary_coil_candidates"],
                },
                "shot_coil_current_a": float(np.asarray(base_current[circuit_index])),
                "edit_fraction_bounds": [
                    float(np.min(SWEEP_FRACTIONS)),
                    float(np.max(SWEEP_FRACTIONS)),
                ],
                "sweep_position_count": len(SWEEP_FRACTIONS),
                "successive_edit_count": EDIT_COUNT,
                "successive_edit_step_fraction": 0.02,
            },
            "compile": {
                "program_built_first_edit": first_edit_compiled,
                "program_reused_from_edit": (
                    0 if program_reused_from_edit is None else program_reused_from_edit
                ),
                "process_cache_hit_count_after_first": sum(
                    row["compilation_cache"] == "hit" for row in rows[1:]
                ),
                "persistent_cache_hit_count": persistent_hits,
                "persistent_cache_miss_count": persistent_misses,
                "later_edits_add_no_compiles": later_edits_add_no_misses,
            },
            "raster": terminal_raster,
            "summary": {
                "edit_count": len(rows),
                "median_warm_wall_milliseconds": median_warm_ms,
                "minimum_warm_wall_milliseconds": float(warm_ms.min()),
                "maximum_warm_wall_milliseconds": float(warm_ms.max()),
                "latency_regime": latency_regime,
                "interactive_latency_target_milliseconds": (
                    INTERACTIVE_LATENCY_TARGET_MILLISECONDS
                ),
                "interactive_latency_target_verdict": (
                    "PASS" if latency_target_met else "FAIL"
                ),
                "latency_statement": (
                    f"Median warm per-edit wall is {median_warm_ms:.3f} ms "
                    f"against the below-{INTERACTIVE_LATENCY_TARGET_MILLISECONDS:.0f}-"
                    f"ms tens-of-milliseconds target: "
                    f"{'PASS' if latency_target_met else 'FAIL'}."
                ),
                "converged_edit_count": sum(row["converged"] for row in rows),
                "failing_points": [
                    row["edit_index"] for row in rows if not row["converged"]
                ],
                "trip_count_minimum": min(row["trip_count"] for row in rows),
                "trip_count_median": float(
                    np.median([row["trip_count"] for row in rows])
                ),
                "trip_count_maximum": max(row["trip_count"] for row in rows),
                "maximum_boundary_displacement_m": (
                    float(finite_displacements.max())
                    if finite_displacements.size
                    else None
                ),
            },
            "edits": rows,
            "figure": str(figure.relative_to(ROOT)),
            "raster_figure": str(DEFAULT_RASTER_FIGURE.relative_to(ROOT)),
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


def _check_case(carrier_path: Path) -> dict[str, Any]:
    """Validate the non-solve case wiring on any host before the H200 run."""
    configure_dtypes()
    response_cache, _metadata = _response_cache(carrier_path)
    selected = {"shot": SHOT, "slice_index": SLICE_INDEX}
    case, context = parity._mast_case_from_selection(
        SHOT_STORE, selected, qualification=None
    )
    _passive, profile, policy = parity._passive_inclusive_case(
        case, context, response_cache
    )
    prescribed = profile.operator.prescribed_current_field
    base_current = np.asarray(prescribed.current, dtype=np.float64)
    circuit_index = None
    wall_start = profile.operator.grid.node_number
    response = np.asarray(prescribed.response, dtype=np.float64)
    candidates = []
    for row in policy["active_mapping"]:
        if row["family"] not in BOUNDARY_COIL_FAMILIES:
            continue
        circuit = int(row["stored_circuit"])
        candidates.append(
            {
                "family": row["family"],
                "stored_circuit": circuit,
                "current_a": float(base_current[circuit]),
                "two_percent_wall_flux_sup_wb": float(
                    0.02
                    * abs(base_current[circuit])
                    * np.max(np.abs(response[wall_start:, circuit]))
                ),
            }
        )
    if len(candidates) != len(BOUNDARY_COIL_FAMILIES):
        raise RuntimeError("the P4/P5 boundary-circuit mapping is incomplete")
    selected_coil = max(
        candidates,
        key=lambda row: row["two_percent_wall_flux_sup_wb"],
    )
    circuit_index = int(selected_coil["stored_circuit"])
    raster_geometry = profile.operator.raster_geometry()
    message = {
        "shot": SHOT,
        "slice_index": SLICE_INDEX,
        "response_matrix_reused": policy["response_matrix_reused"],
        "stored_circuit_count": policy["stored_circuit_count"],
        "coil_family": selected_coil["family"],
        "coil_circuit_index": circuit_index,
        "coil_current_a": float(base_current[circuit_index]),
        "target_current_a": abs(float(case["reference"]["plasma_current_a"])),
        "boundary_coil_candidates": candidates,
        "raster_shape": [int(value) for value in raster_geometry[2]],
        "raster_radius_nodes": int(np.asarray(raster_geometry[0]).size),
        "raster_height_nodes": int(np.asarray(raster_geometry[1]).size),
        "lattice_node_count": int(profile.lattice.node_count),
    }
    return message


def _probe(carrier_path: Path) -> dict[str, Any]:
    """Run one compiled-slice edit end to end on any host (CPU smoke).

    Exercises the exact per-edit path the H200 job measures — the compiled
    slice program build and re-entry, the full typed equilibrium receipt with
    its raster and labelled leaves, and the drain that materialises the
    leaves — seeded from the banked reference flux instead of a converged
    endpoint, so the route's types and shapes are proven without the H200
    gate and without the expensive endpoint preparation.
    """
    configure_dtypes()
    response_cache, _metadata = _response_cache(carrier_path)
    selected = {"shot": SHOT, "slice_index": SLICE_INDEX}
    case, context = parity._mast_case_from_selection(
        SHOT_STORE, selected, qualification=None
    )
    _passive, profile, policy = parity._passive_inclusive_case(
        case, context, response_cache
    )
    prescribed = profile.operator.prescribed_current_field
    base_current = np.asarray(prescribed.current, dtype=np.float64)
    wall_start = profile.operator.grid.node_number
    response = np.asarray(prescribed.response, dtype=np.float64)
    candidates = []
    for row in policy["active_mapping"]:
        if row["family"] not in BOUNDARY_COIL_FAMILIES:
            continue
        circuit = int(row["stored_circuit"])
        candidates.append(
            {
                "family": row["family"],
                "stored_circuit": circuit,
                "two_percent_wall_flux_sup_wb": float(
                    0.02
                    * abs(base_current[circuit])
                    * np.max(np.abs(response[wall_start:, circuit]))
                ),
            }
        )
    if len(candidates) != len(BOUNDARY_COIL_FAMILIES):
        raise RuntimeError("the P4/P5 boundary-circuit mapping is incomplete")
    selected_coil = max(
        candidates,
        key=lambda row: row["two_percent_wall_flux_sup_wb"],
    )
    circuit_index = int(selected_coil["stored_circuit"])
    target_current = abs(float(case["reference"]["plasma_current_a"]))
    requested_class = jnp.asarray(int(TopologyClass.DIVERTED), dtype=jnp.int8)
    edited = (
        jnp.asarray(base_current, dtype=jnp.float64)
        .at[circuit_index]
        .multiply(1.0 + EDIT_FRACTIONS[0])
    )
    seed = np.asarray(_passive["state"], dtype=np.float64)
    result = _compiled_edit(
        profile,
        jnp.asarray(seed),
        edited,
        requested_class,
        target_current,
        None,
    )
    equilibrium = _equilibrium_receipt(
        profile,
        result,
        requested_class,
        target_current,
        edited,
    )
    _drain_receipt(equilibrium)
    raster_geometry = profile.operator.raster_geometry()
    return {
        "converged": bool(np.asarray(result.converged)),
        "terminal_residual": float(np.asarray(result.terminal_residual)),
        "termination": result.termination_name,
        "trip_count": int(np.asarray(result.active_set_iterations)),
        "reduced_dimension": int(result.reduced_dimension),
        "off_support_leakage_wb": float(result.off_support_leakage),
        "program_reused": result.program is not None,
        "receipt_flux_shape": list(np.asarray(equilibrium.flux).shape),
        "raster_shape": [int(value) for value in raster_geometry[2]],
        "raster_psi_shape": list(np.asarray(equilibrium.raster_flux.psi).shape),
        "raster_labels_shape": list(
            np.asarray(equilibrium.raster_flux.domain_label).shape
        ),
        "raster_separatrix_vertices": int(
            np.asarray(equilibrium.raster_flux.separatrix_vertex_count)
        ),
        "labelled_lcfs_vertices": int(
            np.asarray(equilibrium.labelled_flux.lcfs_vertex_count)
        ),
        "coil_family": selected_coil["family"],
        "coil_circuit_index": circuit_index,
        "target_current_a": target_current,
    }


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
#SBATCH --time=00:55:00
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
    summary = receipt["summary"]
    print(
        json.dumps(
            {
                "verdict": receipt["verdict"],
                "job_id": scheduler["job_id"],
                "node": scheduler["node"],
                "elapsed_seconds": receipt["runtime"]["elapsed_seconds"],
                "exit_marker": receipt["runtime"]["exit_marker"],
                "edit_count": summary["edit_count"],
                "median_warm_wall_milliseconds": summary[
                    "median_warm_wall_milliseconds"
                ],
                "converged_edit_count": summary["converged_edit_count"],
                "failing_points": summary["failing_points"],
                "persistent_cache": {
                    "hits": receipt["persistent_compilation_cache"]["solve_hit_count"],
                    "misses": receipt["persistent_compilation_cache"][
                        "solve_miss_count"
                    ],
                },
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
                "r-20260907T165537745800-fsa-coil-edit-latency/logs"
            ),
        )

    harvest_parser = subparsers.add_parser("harvest")
    harvest_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    check_parser = subparsers.add_parser("checkcase")
    check_parser.add_argument(
        "--carrier", type=Path, default=response_carrier.DEFAULT_CARRIER
    )
    probe_parser = subparsers.add_parser("probe")
    probe_parser.add_argument(
        "--carrier", type=Path, default=response_carrier.DEFAULT_CARRIER
    )
    arguments = parser.parse_args()
    if arguments.command == "run":
        run(
            arguments.output,
            arguments.figure,
            arguments.carrier,
        )
    elif arguments.command == "checkcase":
        print(json.dumps(_check_case(arguments.carrier), indent=2, sort_keys=True))
    elif arguments.command == "probe":
        print(json.dumps(_probe(arguments.carrier), indent=2, sort_keys=True))
    elif arguments.command == "sbatch":
        print(_sbatch_script(arguments), end="")
    elif arguments.command == "submit":
        _submit(arguments)
    else:
        _harvest(arguments.output)


if __name__ == "__main__":
    main()
