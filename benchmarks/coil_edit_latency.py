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
from benchmarks.diiid_forward_gs_match import _margin_graded_newton_krylov
from nova.equilibrium.fixed_point import FixedPointTerminationReason
from nova.equilibrium.forward import SaddleSeedGeometry
from nova.equilibrium.topology import TopologyClass
from nova.imas.mast_solve_inputs import SHOT_STORE
from nova.jax.config import (
    configure_dtypes,
    configure_persistent_compilation_cache,
    default_persistent_compilation_cache_root,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT / "docs/figures/forward-solve-api/coil-edit-latency-converging.json"
)
DEFAULT_FIGURE = (
    ROOT / "docs/figures/forward-solve-api/coil-edit-latency-converging.png"
)
SHOT = 22086
SLICE_INDEX = 43
SWEEP_FRACTIONS = np.arange(-0.20, 0.201, 0.02, dtype=np.float64)
EDIT_FRACTIONS = SWEEP_FRACTIONS[1:]
EDIT_COUNT = len(EDIT_FRACTIONS)
COLD_CONTROL_EDIT_INDICES = frozenset((0, 6, 13, 19))
BOUNDARY_COIL_FAMILIES = frozenset({"p4_lower", "p4_upper", "p5_lower", "p5_upper"})
INTERACTIVE_LATENCY_TARGET_MILLISECONDS = 100.0


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

    events: dict[str, float | int] = {"hits": 0, "saved_seconds": 0.0}

    def hit(event: str, **_kwargs: Any) -> None:
        if event == "/jax/compilation_cache/cache_hits":
            events["hits"] = int(events["hits"]) + 1

    def saved(event: str, duration_secs: float, **_kwargs: Any) -> None:
        if event == "/jax/compilation_cache/compile_time_saved_sec":
            events["saved_seconds"] = float(events["saved_seconds"]) + duration_secs

    monitoring.register_event_listener(hit)
    monitoring.register_event_duration_secs_listener(saved)
    return events


def _render(
    rows: list[dict[str, Any]],
    figure_path: Path,
    *,
    coil_family: str,
    compile_count: int,
    persistent_cache_hits: int,
) -> None:
    indices = np.asarray([row["edit_index"] for row in rows])
    milliseconds = np.asarray([row["wall_milliseconds"] for row in rows])
    displacement = 1.0e3 * np.asarray([row["boundary_displacement_m"] for row in rows])
    trips = np.asarray([row["trip_count"] for row in rows])
    residual = np.asarray([row["terminal_residual"] for row in rows])
    colours = [
        "#d97706" if row["compilation_cache"] == "miss" else "#2563eb" for row in rows
    ]
    cold_indices = np.asarray(
        [row["edit_index"] for row in rows if row["cold_control"] is not None]
    )
    cold_milliseconds = np.asarray(
        [
            row["cold_control"]["wall_milliseconds"]
            for row in rows
            if row["cold_control"] is not None
        ]
    )
    figure, axes = plt.subplots(2, 2, figsize=(11.2, 8.2), constrained_layout=True)
    latency_axis, trip_axis, residual_axis, boundary_axis = axes.ravel()
    latency_axis.plot(indices, milliseconds, color="0.72", lw=1.0, zorder=1)
    latency_axis.scatter(indices, milliseconds, c=colours, s=34, zorder=2)
    latency_axis.scatter(
        cold_indices,
        cold_milliseconds,
        facecolors="none",
        edgecolors="#7c3aed",
        marker="s",
        s=55,
        label="cold-portfolio control",
    )
    latency_axis.axhline(
        INTERACTIVE_LATENCY_TARGET_MILLISECONDS,
        color="#dc2626",
        linestyle="--",
        linewidth=1.0,
        label="100 ms target ceiling",
    )
    latency_axis.set_yscale("log")
    latency_axis.set_ylabel("Solve wall time [ms]")
    latency_axis.set_title(
        f"MAST {SHOT}/{SLICE_INDEX} mixed arm · {coil_family.replace('_', '-').upper()}"
    )
    latency_axis.grid(True, which="both", alpha=0.25)
    latency_axis.scatter([], [], color="#d97706", label="compile miss")
    latency_axis.scatter([], [], color="#2563eb", label="process-cache hit")
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
        parity.FIXED_POINT_CRITERION,
        color="#dc2626",
        linestyle="--",
        linewidth=1.0,
        label=f"tolerance {parity.FIXED_POINT_CRITERION:.0e}",
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
        f"Boundary motion · compiles {compile_count} · persistent hits "
        f"{persistent_cache_hits}"
    )
    boundary_axis.grid(True, alpha=0.25)
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
    endpoint_current = base_current.copy()
    endpoint_current[circuit_index] *= 1.0 + SWEEP_FRACTIONS[0]
    endpoint_map = profile.flux_map(
        requested_class=TopologyClass.DIVERTED,
        target_current=target_current,
        prescribed_current=jnp.asarray(endpoint_current),
    )
    endpoint_seed = _margin_graded_newton_krylov(
        endpoint_map,
        profile.operator.topology_margin,
        mixed_seed.state,
        newton_steps=parity.NEWTON_STEPS,
        gmres_iterations=parity.GMRES_ITERATIONS,
    )
    jax.block_until_ready(endpoint_seed.state)
    endpoint_residual = float(np.asarray(endpoint_seed.residual))
    if (
        not np.isfinite(endpoint_residual)
        or endpoint_residual > parity.FIXED_POINT_CRITERION
    ):
        raise RuntimeError(
            f"the minus-twenty-percent endpoint did not converge: "
            f"{endpoint_residual:.6g}"
        )
    axis = np.asarray(case["axis"], dtype=np.float64)
    x_points = np.asarray(case["x_points"], dtype=np.float64)
    x_points = x_points[np.all(np.isfinite(x_points), axis=1)]
    if not len(x_points):
        raise RuntimeError("the frozen-six reference supplies no finite saddle")
    cold = profile.cold_seed_portfolio(
        target_current,
        axis,
        diverted_geometry=SaddleSeedGeometry(tuple(axis), tuple(x_points[0])),
    )
    prepared = {
        "initial": endpoint_seed.state,
        "cold_diverted_seed": cold.branches.flux[int(TopologyClass.DIVERTED)],
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
        "endpoint_seed": {
            "edit_fraction": float(SWEEP_FRACTIONS[0]),
            "terminal_residual": endpoint_residual,
            "converged": True,
            "route": "margin-graded fixed-ladder Newton-Krylov preparation",
        },
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
    cache_events = _cache_monitor()
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
        solve_persistent_saved_start = float(cache_events["saved_seconds"])
        initial = prepared["initial"]
        cold_diverted_seed = prepared["cold_diverted_seed"]
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
        _endpoint_masks, endpoint_topology = profile.operator.read(state)
        reference_boundary = np.asarray(endpoint_topology.boundary)
        reference_lcfs = np.empty((0, 2), dtype=np.float64)
        for index, (fraction, current) in enumerate(
            zip(EDIT_FRACTIONS, edit_vectors, strict=True)
        ):
            cache_before = int(jitted._cache_size())
            persistent_hits_before = int(cache_events["hits"])
            started = time.perf_counter()
            branch = jitted(state, current)
            jax.block_until_ready(branch)
            wall_milliseconds = 1.0e3 * (time.perf_counter() - started)
            cache_after = int(jitted._cache_size())
            persistent_hits_after = int(cache_events["hits"])
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
            cold_control = None
            if index in COLD_CONTROL_EDIT_INDICES:
                cold_started = time.perf_counter()
                cold_branch = jitted(cold_diverted_seed, current)
                jax.block_until_ready(cold_branch)
                cold_wall_milliseconds = 1.0e3 * (time.perf_counter() - cold_started)
                cold_fixed_point = cold_branch.equilibrium.fixed_point
                cold_control = {
                    "seed_source": "ForwardProfile.cold_seed_portfolio diverted branch",
                    "wall_milliseconds": cold_wall_milliseconds,
                    "converged": bool(np.asarray(cold_branch.converged)),
                    "terminal_residual": float(np.asarray(cold_branch.residual)),
                    "trip_count": int(
                        np.asarray(cold_fixed_point.active_set_iterations)
                    ),
                    "termination": _termination_name(
                        cold_fixed_point.termination_reason
                    ),
                }
            row = {
                "edit_index": index,
                "edit_fraction": float(fraction),
                "coil_current_a": float(np.asarray(current[circuit_index])),
                "wall_milliseconds": wall_milliseconds,
                "compilation_cache": ("miss" if cache_after > cache_before else "hit"),
                "compile_count": cache_after,
                "persistent_cache_hits_before": persistent_hits_before,
                "persistent_cache_hits_after": persistent_hits_after,
                "persistent_cache_hit_count": (
                    persistent_hits_after - persistent_hits_before
                ),
                "jit_cache_size_before": cache_before,
                "jit_cache_size_after": cache_after,
                "executable_identity": fingerprint,
                "stablehlo_sha256": stablehlo_identity,
                "converged": bool(np.asarray(branch.converged)),
                "terminal_residual": float(np.asarray(branch.residual)),
                "trip_count": int(np.asarray(fixed_point.active_set_iterations)),
                "fixed_iteration_count": int(np.asarray(branch.iterations)),
                "termination": _termination_name(fixed_point.termination_reason),
                "lcfs_vertex_count": lcfs_count,
                "boundary_displacement_m": boundary_displacement,
                "boundary_displacement_source": displacement_source,
                "cold_control": cold_control,
            }
            rows.append(row)
            if not row["converged"]:
                raise RuntimeError(
                    f"warm-start chain lost convergence at edit {index}: "
                    f"residual={row['terminal_residual']:.6g}"
                )
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
        warm_ms = np.asarray(
            [row["wall_milliseconds"] for row in rows[1:]], dtype=np.float64
        )
        all_cache_hits_after_first = all(
            row["compilation_cache"] == "hit" for row in rows[1:]
        )
        one_executable = len({row["executable_identity"] for row in rows}) == 1
        all_converged = all(row["converged"] for row in rows)
        median_warm_ms = float(np.median(warm_ms))
        latency_target_met = median_warm_ms < INTERACTIVE_LATENCY_TARGET_MILLISECONDS
        latency_regime = (
            "tens_of_milliseconds_or_better"
            if latency_target_met
            else "above_tens_of_milliseconds"
        )
        boundary_displacements = np.asarray(
            [row["boundary_displacement_m"] for row in rows], dtype=np.float64
        )
        controls = [
            row["cold_control"] for row in rows if row["cold_control"] is not None
        ]
        cold_ms = np.asarray(
            [control["wall_milliseconds"] for control in controls],
            dtype=np.float64,
        )
        persistent_hits = int(cache_events["hits"]) - solve_persistent_hits_start
        persistent_saved_seconds = (
            float(cache_events["saved_seconds"]) - solve_persistent_saved_start
        )
        gates = {
            "exactly_twenty_edits_recorded": len(rows) == 20,
            "sweep_spans_plus_minus_twenty_percent": bool(
                np.isclose(SWEEP_FRACTIONS[0], -0.20)
                and np.isclose(SWEEP_FRACTIONS[-1], 0.20)
            ),
            "successive_edits_are_two_percent": bool(
                np.allclose(np.diff(SWEEP_FRACTIONS), 0.02)
            ),
            "first_edit_is_compile_miss": rows[0]["compilation_cache"] == "miss",
            "all_later_edits_are_cache_hits": all_cache_hits_after_first,
            "compile_count_is_one": int(jitted._cache_size()) == 1,
            "executable_identity_unchanged": one_executable,
            "all_edits_converged": all_converged,
            "at_least_four_cold_portfolio_controls": len(controls) >= 4,
            "boundary_displacement_is_finite": bool(
                np.all(np.isfinite(boundary_displacements))
            ),
            "median_warm_wall_meets_tens_of_milliseconds_target": latency_target_met,
        }
        passed = all(gates.values())
        exit_marker = 0 if passed else 2
        _render(
            rows,
            figure,
            coil_family=prepared["coil_mapping"]["family"],
            compile_count=int(jitted._cache_size()),
            persistent_cache_hits=persistent_hits,
        )
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
            "persistent_compilation_cache": cache.receipt()
            | {
                "solve_hit_count": persistent_hits,
                "solve_compile_seconds_saved": persistent_saved_seconds,
                "process_total_hit_count": int(cache_events["hits"]),
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
                    "the converged corrected-bank mixed arm prepares the minus-twenty-"
                    "percent endpoint; each of twenty two-percent edits starts from "
                    "the preceding edit's convergence-qualified terminal flux"
                ),
                "seed_arm": prepared["mixed_seed"],
                "sweep_start_endpoint": prepared["endpoint_seed"],
                "route": "newton_krylov",
                "solver_policy": {
                    "tolerance": parity.FIXED_POINT_CRITERION,
                    "newton_steps": parity.NEWTON_STEPS,
                    "gmres_iterations": parity.GMRES_ITERATIONS,
                    "warmup_sweeps": parity.WARMUP_SWEEPS,
                    "relaxation": parity.RELAXATION,
                    "step_cap": parity.STEP_CAP,
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
                "count": int(jitted._cache_size()),
                "process_cache_hit_count_after_first": sum(
                    row["compilation_cache"] == "hit" for row in rows[1:]
                ),
                "persistent_cache_hit_count": persistent_hits,
                "executable_identity": executable_identity,
                "stablehlo_sha256": stablehlo_identity,
            },
            "summary": {
                "edit_count": len(rows),
                "cold_portfolio_control_count": len(controls),
                "median_warm_wall_milliseconds": median_warm_ms,
                "minimum_warm_wall_milliseconds": float(warm_ms.min()),
                "maximum_warm_wall_milliseconds": float(warm_ms.max()),
                "median_cold_portfolio_wall_milliseconds": float(np.median(cold_ms)),
                "warm_to_cold_median_ratio": float(median_warm_ms / np.median(cold_ms)),
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
                "cold_control_converged_count": sum(
                    control["converged"] for control in controls
                ),
                "trip_count_minimum": min(row["trip_count"] for row in rows),
                "trip_count_median": float(
                    np.median([row["trip_count"] for row in rows])
                ),
                "trip_count_maximum": max(row["trip_count"] for row in rows),
                "maximum_boundary_displacement_m": float(boundary_displacements.max()),
            },
            "edits": rows,
            "figure": str(figure.relative_to(ROOT)),
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
#SBATCH --job-name=coil-edit-converging
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:55:00
#SBATCH --output={log_directory}/coil-edit-converging-%j.log
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
                "r-20260904T054255696439-fsa-warm-start-converging-codex/logs"
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
