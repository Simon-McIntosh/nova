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
from nova.equilibrium.constraint import (
    CircuitCurrentUnknown,
    ConstraintBinding,
    ConstraintPair,
    CurrentCentroidConstraint,
)
from nova.equilibrium.fixed_point import FixedPointResult, FixedPointTerminationReason
from nova.equilibrium.observation import MomentIntegralSupport
from nova.equilibrium.flux_surface_connectivity import (
    fit_tensor_spline,
    traced_spline_contour,
)
from nova.equilibrium.separatrix_branches import (
    assemble_separatrix_branches,
    boundary_flux_at_admitted_saddle,
)
from nova.equilibrium.topology import NoQualifiedAxisError, TopologyClass
from nova.equilibrium.wall_mask import WallUnit
from nova.imas.mast_solve_inputs import SHOT_STORE
from nova.jax.config import (
    configure_dtypes,
    configure_persistent_compilation_cache,
    default_persistent_compilation_cache_root,
)
from nova.media import poloidal
from nova.media.ink import DEFAULT_INK, poloidal_axes
from nova.media.sources.frame import inside_wall_units


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
DIAGNOSTIC_ROOT = ROOT / "docs/figures/forward-solve-api/coil-edit-nonconvergence"
DEFAULT_DIAGNOSTICS = DIAGNOSTIC_ROOT / "coil-edit-nonconvergence.json"
DEFAULT_PANEL_DATA = DIAGNOSTIC_ROOT / "panel-states.npz"
DEFAULT_PANEL_FIGURE = DIAGNOSTIC_ROOT / "converged-vs-nonconverged.png"
SHOT = 22086
SLICE_INDEX = 43
SWEEP_FRACTIONS = np.arange(-0.20, 0.201, 0.02, dtype=np.float64)
EDIT_FRACTIONS = SWEEP_FRACTIONS[1:]
EDIT_COUNT = len(EDIT_FRACTIONS)
BOUNDARY_COIL_FAMILIES = frozenset({"p4_lower", "p4_upper", "p5_lower", "p5_upper"})
# Set to the reason a run is not on the measurement host.  A CPU re-run
# regenerates the persisted sweep states only; its per-edit walls are not the
# interactive measurement, and the receipt echoes the marker so the lowered
# provenance travels with the artefact.
CPU_PROVENANCE_MARKER = "NOVA_COIL_EDIT_CPU_PROVENANCE"
#: Set to the reason a run is on a GPU that is not the H200 measurement host --
#: the titan rung of the compute hierarchy. Its per-edit walls are that device's
#: walls and are not the interactive measurement; the receipt echoes both the
#: marker and the device kind so the lowered provenance travels with the
#: artefact rather than being inferred from a field the refusal removed.
OFF_HOST_PROVENANCE_MARKER = "NOVA_COIL_EDIT_OFF_HOST_PROVENANCE"
INTERACTIVE_LATENCY_TARGET_MILLISECONDS = 100.0
# The compiled slice route's own production budgets (the kernel's declared
# constants), not the parity-driver Newton budgets the endpoint prep uses.
COMPILED_SLICE_TOLERANCE = reduced_newton.FIXED_POINT_RESIDUAL_TOLERANCE
COMPILED_SLICE_NEWTON_STEPS = reduced_newton.NEWTON_STEPS
COMPILED_SLICE_ACTIVE_SET_STEPS = reduced_newton.ACTIVE_SET_STEPS
# The vertical-current-centre row's declared tolerance and its declared scale,
# both in metres: the tolerance is the physical residual the augmented solve
# must reach on the centroid, and the scale is the length the row is normalised
# by, so a scaled residual of one means a whole panel height of displacement.
VERTICAL_CENTROID_TOLERANCE = 1.0e-6


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
    """Load the persisted response and its complete input ledger.

    The contract the carrier is held to is the one its own grid declares, found
    by identity among the pinned grids, so a carrier built on the full stored
    axes is verified as strictly as the coarse one rather than refused against
    the other grid's pin.  A carrier whose identity is not pinned anywhere is
    still refused.
    """
    grid = response_carrier.grid_for_carrier(carrier_path)
    response, metadata = response_carrier.load_carrier(
        carrier_path,
        semantic_identity=grid.semantic_identity,
        resolved_target_digest=grid.resolved_target_digest,
        response_shape=grid.response_shape,
    )
    with np.load(carrier_path, allow_pickle=False) as archive:
        input_digests = json.loads(_archive_scalar(archive, "input_digests_json"))
        audit = json.loads(_archive_scalar(archive, "audit_json"))
    audit["stored_circuit_count"] = metadata["stored_circuit_count"]
    return {
        "response": response,
        "input_digests": input_digests,
        "audit": audit,
    }, metadata


_SCHEDULER_CACHE: dict[str, Any] | None = None


def _scheduler() -> dict[str, Any]:
    global _SCHEDULER_CACHE
    if _SCHEDULER_CACHE is None:
        _SCHEDULER_CACHE = _read_scheduler()
    return _SCHEDULER_CACHE


def _read_scheduler() -> dict[str, Any]:
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
    """Require the H200 measurement host, or an explicitly marked CPU re-run.

    The persisted sweep states can be regenerated while the shared reservation
    is held, so a run that names its own reason on a CPU platform is accepted;
    the receipt records the host, the partition, the reservation and the JAX
    platform of whatever ran, so the lowered provenance is read from the
    artefact rather than inferred from a field the refusal removed.  An
    unmarked run off the reservation is still refused.
    """
    if os.environ.get("TMPDIR") != "/tmp":
        raise RuntimeError("TMPDIR=/tmp must be set in the job body")
    device = jax.devices()[0]
    provenance = os.environ.get(CPU_PROVENANCE_MARKER, "").strip()
    if provenance:
        if device.platform != "cpu":
            raise RuntimeError(
                "the CPU provenance marker requires a CPU platform, got "
                f"{device.platform} on {device.device_kind}"
            )
        return
    off_host = os.environ.get(OFF_HOST_PROVENANCE_MARKER, "").strip()
    if off_host:
        # The titan rung: a GPU that is not the measurement host. It is
        # admitted only with its reason named, and the H200 checks below are
        # the ones that do not apply to it -- an unmarked run off the
        # reservation is refused exactly as before.
        if device.platform != "gpu":
            raise RuntimeError(
                "the off-host provenance marker requires a GPU platform, got "
                f"{device.platform} on {device.device_kind}"
            )
        return
    if device.platform != "gpu" or "H200" not in device.device_kind:
        raise RuntimeError(f"one H200 is required, got {device}")
    if os.environ.get("SLURM_JOB_PARTITION") != "betelgeuse":
        raise RuntimeError("the betelgeuse partition is required")
    if os.environ.get("SLURM_JOB_RESERVATION") != "gpu_0003_grpA":
        raise RuntimeError("the gpu_0003_grpA reservation is required")
    if os.environ.get("SLURM_CPUS_PER_TASK") != "8":
        raise RuntimeError("the measurement requires exactly eight requested CPUs")
    requested_mib = os.environ.get("SLURM_MEM_PER_NODE")
    if requested_mib not in ("128G", "131072"):
        raise RuntimeError("the measurement requires a 128 GiB memory allocation")
    if os.environ.get("JAX_PLATFORMS") != "cuda,cpu":
        raise RuntimeError("JAX_PLATFORMS=cuda,cpu must be set in the job body")


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


def _circuit_families(policy: dict[str, Any]) -> dict[int, str]:
    """Return the zero-based circuit index of every named active family."""
    return {
        int(row["stored_circuit"]) - 1: str(row["family"])
        for row in policy["active_mapping"]
    }


def _vertical_centroid_pair(
    profile: Any,
    policy: dict[str, Any],
    flux: jax.Array,
    *,
    requested_class: jax.Array,
    target_current: float,
    target: float,
) -> tuple[ConstraintPair, dict[str, Any]]:
    """Return the vertical current-centre row and the actuator that drives it.

    The row reads the plasma's own current centroid on the whole authored
    domain and is eliminated by a circuit-current direction the machine's own
    response matrix picks out over its active mapping: no direction is written
    here, the matrix supplies it, and the ampere scale is the current that
    moves the row by one declared scale.  Pinning this row is what holds a
    solved state on the vertical position its reference belongs to while a
    boundary coil is edited, since an unconstrained slice leaves the vertical
    current centre free and converges on a different equilibrium instead.
    """
    prescribed = profile.operator.prescribed_current_field
    if prescribed is None:
        raise RuntimeError("the vertical-centroid row needs the prescribed field")
    circuit_count = int(prescribed.circuit_count)
    families = _circuit_families(policy)
    if not families:
        raise RuntimeError("the vertical-centroid row needs a drivable circuit set")
    position_scale = float(np.ptp(np.asarray(profile.lattice.height)))
    seed = ConstraintPair(
        functional=CurrentCentroidConstraint(
            components=("centroid_z",),
            support=MomentIntegralSupport.ALL_DOMAIN,
        ),
        unknown=CircuitCurrentUnknown(
            direction=jnp.zeros(circuit_count, dtype=jnp.float64),
            ampere_scale=jnp.asarray([1.0], dtype=jnp.float64),
        ),
        binding=ConstraintBinding(
            target=jnp.atleast_1d(jnp.asarray(target, dtype=jnp.float64)),
            tolerance=jnp.asarray([VERTICAL_CENTROID_TOLERANCE], dtype=jnp.float64),
            scale=jnp.asarray([position_scale], dtype=jnp.float64),
            initial_unknown=jnp.asarray([0.0], dtype=jnp.float64),
            payload=None,
            policy="imposed",
        ),
    )
    derived, selection = profile.derived_constraint_pairs(
        (seed,),
        flux,
        requested_class=requested_class,
        target_current=target_current,
        circuits=sorted(families),
    )
    (pair,) = derived
    authority = float(
        np.asarray(selection.direction_authority, dtype=float).reshape(-1)[0]
    )
    if not np.isfinite(authority) or authority <= 0.0:
        raise RuntimeError("the vertical-centroid row has no circuit authority")
    direction = np.asarray(pair.unknown.direction, dtype=float).reshape(-1)
    pair = ConstraintPair(
        functional=pair.functional,
        unknown=CircuitCurrentUnknown(
            direction=pair.unknown.direction,
            ampere_scale=jnp.asarray([1.0 / authority], dtype=jnp.float64),
            singular_values=pair.unknown.singular_values,
            authority=pair.unknown.authority,
            rule=pair.unknown.rule,
        ),
        binding=pair.binding,
    )
    actuator = {
        "definition": "vertical current-centre row, matrix-led over the active mapping",
        "components": ["centroid_z"],
        "support": MomentIntegralSupport.ALL_DOMAIN.value,
        "target_m": float(target),
        "tolerance_m": VERTICAL_CENTROID_TOLERANCE,
        "position_scale_m": position_scale,
        "selection_rule": selection.rule.name.lower(),
        "ampere_scale_a": float(1.0 / authority),
        "direction_authority_m_per_a": authority,
        "drivable_circuits": [
            {
                "circuit": index,
                "family": families[index],
                "direction_weight": float(direction[index]),
            }
            for index in sorted(families)
            if abs(direction[index]) > 0.0
        ],
    }
    return pair, actuator


def _prepare_case(
    carrier_path: Path,
    grid_points: int | None = None,
    flux_function_factory: Any = None,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Prepare the sweep's base frame, optionally on a named axis count.

    ``grid_points`` selects the uniform per-axis node count the case is built
    on and must agree with the grid the supplied carrier was built for; the
    default keeps the stored-axis stride this driver has always used.
    """
    response_cache, carrier = _response_cache(carrier_path)
    selected = {"shot": SHOT, "slice_index": SLICE_INDEX}
    case, context = parity._mast_case_from_selection(
        SHOT_STORE,
        selected,
        qualification=None,
        grid_points=grid_points,
        flux_function_factory=flux_function_factory,
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
    # The receiver-grid separatrix of the base frame (fraction 0) is the
    # reference every edited raster is displaced against; it is built through
    # the same integral-state and raster path the per-edit receipt uses, with
    # the operator's own external-conductor current in the conductor slot.
    reference_moments, _reference_support, reference_masks, reference_topology, _ = (
        profile._integral_state(
            mixed_seed.state, TopologyClass.DIVERTED, target_current
        )
    )
    reference_raster = profile._raster_flux(
        reference_moments,
        reference_masks,
        reference_topology,
        prescribed_current=jnp.asarray(base_current),
    )
    reference_raster_vertex_count = int(
        np.asarray(reference_raster.separatrix_vertex_count)
    )
    reference_raster_separatrix = np.asarray(reference_raster.separatrix)[
        :reference_raster_vertex_count
    ]
    # The base frame's own raster field and landmarks: the panel that compares
    # a converged edit against a non-converged one draws both states on one
    # physical level array, so the reference is persisted beside the edits.
    reference_panel = {
        "radius": np.asarray(reference_raster.radius, dtype=float),
        "height": np.asarray(reference_raster.height, dtype=float),
        "shape": np.asarray(reference_raster.shape, dtype=int),
        "psi": np.asarray(reference_raster.psi, dtype=float),
        "separatrix": reference_raster_separatrix,
        "nulls": _null_points(profile, mixed_seed.state),
        "branches": _branches_of(profile, mixed_seed.state, reference_raster),
    }
    # The vertical current centre the reference frame already carries is the
    # target every edit is held to: the row is measured on the base frame's own
    # terminal state, through the same observation the constraint reads, and
    # the compensating direction is derived there once so it is a constant of
    # the compiled slice rather than a per-edit re-derivation.
    reference_centroid = profile.current_moment_observation(
        mixed_seed.state,
        support=MomentIntegralSupport.ALL_DOMAIN,
        requested_class=TopologyClass.DIVERTED,
        target_current=target_current,
    )
    reference_centroid_z = float(np.asarray(reference_centroid.centroid_z))
    if not np.isfinite(reference_centroid_z):
        raise RuntimeError("the reference frame carries no vertical current centre")
    vertical_pair, vertical_actuator = _vertical_centroid_pair(
        profile,
        policy,
        mixed_seed.state,
        requested_class=jnp.asarray(int(TopologyClass.DIVERTED), dtype=jnp.int8),
        target_current=target_current,
        target=reference_centroid_z,
    )
    reference_panel["vertical_centroid"] = {
        "reference_m": reference_centroid_z,
        "target_m": reference_centroid_z,
        "tolerance_m": VERTICAL_CENTROID_TOLERANCE,
    }
    prepared = {
        "initial": mixed_seed.state,
        "vertical_centroid_pair": vertical_pair,
        "vertical_centroid_actuator": vertical_actuator,
        "reference_centroid_z": reference_centroid_z,
        "reference_lcfs": np.asarray(reference_labelled.lcfs)[:reference_lcfs_count],
        "reference_raster_separatrix": reference_raster_separatrix,
        "reference_panel": reference_panel,
        "reference_raster_separatrix_vertex_count": reference_raster_vertex_count,
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
            "reference_raster_separatrix_vertex_count": (reference_raster_vertex_count),
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
    *,
    newton_steps: int = COMPILED_SLICE_NEWTON_STEPS,
    active_set_steps: int = COMPILED_SLICE_ACTIVE_SET_STEPS,
    constraint_pairs: tuple[ConstraintPair, ...] = (),
) -> Any:
    """Re-enter the compiled slice program with one edited current vector.

    With rows registered the slice solves the augmented fixed point, folding
    each row's compensating circuit current into the prescribed vector it is
    handed; with none it is the same unconstrained entry the sweep always took.
    """
    return reduced_newton.solve_constrained_reduced_newton_compiled(
        profile,
        state,
        constraint_pairs=constraint_pairs,
        requested_class=requested_class,
        target_current=target_current,
        prescribed_current=current,
        tolerance=COMPILED_SLICE_TOLERANCE,
        newton_steps=newton_steps,
        active_set_steps=active_set_steps,
        program=program,
        stream=False,
    )


def _vertical_centroid_row(result: Any) -> dict[str, Any] | None:
    """Return the terminal state of the vertical current-centre row, or nothing.

    The row is the difference between the solved state's own current centroid
    and the vertical position its reference carries, together with the circuit
    current the augmented solve spent to hold it, so every sweep point states
    both the constraint it was solved under and what that constraint cost.
    """
    records = tuple(getattr(result, "constraints", ()) or ())
    if not records:
        return None
    record = records[0]
    compensating = getattr(result, "compensating_current", None)
    return {
        "observed_m": float(np.asarray(record.observed).reshape(-1)[0]),
        "target_m": float(np.asarray(record.target).reshape(-1)[0]),
        "residual_m": float(np.asarray(record.physical_residual).reshape(-1)[0]),
        "scaled_residual": float(np.asarray(record.scaled_residual).reshape(-1)[0]),
        "tolerance_m": float(np.asarray(record.tolerance).reshape(-1)[0]),
        "qualified": bool(np.asarray(record.qualified)),
        "compensating_current_a": float(
            np.asarray(record.physical_unknown).reshape(-1)[0]
        ),
        "compensating_current_norm_a": (
            None
            if compensating is None
            else float(np.linalg.norm(np.asarray(compensating, dtype=float)))
        ),
    }


def _achieved_class(profile: Any, state: Any) -> dict[str, Any]:
    """Read the saddle-aware class the read derives, or its refusal."""
    try:
        _masks, achieved = profile.operator.read(state)
    except NoQualifiedAxisError as error:
        return {
            "read_status": "no_qualified_axis",
            "exception_text": str(error),
        }
    margin = np.asarray(achieved.class_margin, dtype=float).reshape(-1)[0]
    determinate = bool(np.asarray(achieved.class_determinate))
    diverted = bool(np.asarray(achieved.diverted))
    return {
        "read_status": "qualified",
        "class": (
            "indeterminate"
            if not determinate
            else ("diverted" if diverted else "limited")
        ),
        "class_margin": float(margin) if np.isfinite(margin) else None,
    }


def _null_points(profile: Any, state: Any) -> dict[str, Any]:
    """Return the read landmarks a panel draws, or absent points as NaN.

    The admitted saddle is named rather than assumed to be the first row: the
    admitted one is the candidate sitting on the boundary flux the read
    selected, and the remaining qualified nulls are carried too so a panel can
    draw them hollow instead of silently dropping them.
    """
    try:
        _masks, achieved = profile.operator.read(state)
    except NoQualifiedAxisError:
        return {
            "axis": np.full(2, np.nan),
            "x_points": np.full((2, 2), np.nan),
            "x_point_flux": np.full(2, np.nan),
            "saddle_index": 0,
        }
    x_points = np.asarray(achieved.x_point, dtype=float).reshape(-1, 2)
    flux = np.asarray(achieved.x_point_flux, dtype=float).reshape(-1)
    boundary_flux = float(np.asarray(achieved.boundary_flux))
    if flux.size != x_points.shape[0]:
        padded = np.full(x_points.shape[0], np.nan)
        padded[: min(flux.size, x_points.shape[0])] = flux[: x_points.shape[0]]
        flux = padded
    finite = np.isfinite(flux)
    saddle_index = int(np.argmin(np.abs(flux - boundary_flux))) if finite.any() else 0
    return {
        "axis": np.asarray(achieved.axis, dtype=float).reshape(-1)[:2],
        "x_points": x_points,
        "x_point_flux": flux[: x_points.shape[0]],
        "saddle_index": saddle_index,
    }


def _seed_probe(
    profile: Any,
    state: Any,
    current: jax.Array,
    requested_class: jax.Array,
    target_current: float,
) -> dict[str, Any]:
    """Measure the seed's own residual on the edited operator, before any trip.

    One trip closed with zero Newton steps evaluates the trip boundary at the
    seed state itself, so the reported residual is the seed's residual on the
    edited operator rather than the post-correction residual a full edit
    reports.  The program is not threaded from the sweep: this probe's policy
    key differs from the sweep's, so reusing the sweep program would only add
    the probe's solver to a chain whose reuse the latency gates measure.
    """
    result = _compiled_edit(
        profile,
        state,
        current,
        requested_class,
        target_current,
        None,
        newton_steps=0,
        active_set_steps=1,
    )
    jax.block_until_ready(result.state)
    residuals = [float(value) for value in result.active_set_residuals]
    return {
        "residual": float(np.asarray(result.terminal_residual)),
        "trip_count": int(np.asarray(result.active_set_iterations)),
        "termination": result.termination_name,
        "converged": bool(np.asarray(result.converged)),
        "trip_residual_trace": residuals,
        "trip_mask_difference_trace": [
            int(value) for value in result.active_set_mask_differences
        ],
        "achieved_class": _achieved_class(profile, state),
        "achieved_class_after_trip": _achieved_class(profile, result.state),
    }


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


def _terminal_raster_document(terminal: Any, profile: Any) -> dict[str, Any] | None:
    """Describe the operator-held receiver grid target for the receipt."""
    if terminal is None or terminal.raster_flux is None:
        return None
    return {
        "shape": [int(value) for value in np.asarray(terminal.raster_flux.shape)],
        "node_count": int(profile.lattice.node_count),
        "source_quantities": "psi (Wb), psi_N, per-cell labels, separatrix",
        "note": (
            "the operator holds the rectangular receiver-grid target once "
            "from construction; every edit evaluates it from the same coil "
            "and cell currents at no extra compile"
        ),
    }


def _wall_units(operator: Any) -> tuple[Any, ...]:
    """Return the operator's wall as its own typed units.

    The wall is stored flat with unit offsets and per-unit closure and kind,
    so a panel draws every unit on its own terms rather than one invented
    ring: an open material unit is dashed and is never joined to a neighbour.
    """
    coordinate = np.asarray(operator.wall.coordinate, dtype=float).reshape(-1, 2)
    offsets = np.asarray(operator.wall_unit_offsets, dtype=int)
    closed = np.asarray(operator.wall_unit_closed, dtype=bool)
    kinds = tuple(operator.wall_unit_kinds)
    return tuple(
        WallUnit(
            coordinate[start:stop, 0],
            coordinate[start:stop, 1],
            kind=kinds[index],
            closed=bool(closed[index]),
        )
        for index, (start, stop) in enumerate(
            zip(offsets[:-1], offsets[1:], strict=True)
        )
    )


def _preflight_panel_wall(profile: Any) -> None:
    """Resolve the vessel units and one interior mask before the sweep runs.

    The panel writer builds its interior mask from the operator's wall units,
    so an operator whose wall is not a unit collection fails the write only
    after every edit has been solved.  Making the same call first turns that
    into a failure of seconds rather than of the whole allocation.
    """
    units = _wall_units(profile.operator)
    if not units:
        raise RuntimeError("the operator carries no wall units to draw")
    mask = _wall_interior(
        np.asarray(profile.lattice.radius, dtype=float),
        np.asarray(profile.lattice.height, dtype=float),
        units,
    )
    print(
        "PREFLIGHT_WALL_OK units=%d interior_samples=%d"
        % (len(units), int(np.count_nonzero(mask))),
        flush=True,
    )


def _grid_field(psi: Any, shape: Any) -> np.ndarray:
    """Return one raster psi as the (height, radius) contour array."""
    return (
        np.asarray(psi, dtype=float)
        .reshape(tuple(int(value) for value in np.asarray(shape)))
        .T
    )


def _mechanism_of(row: dict[str, Any]) -> str:
    """Name the mechanism one non-converged edit's own trace supports.

    Two facts decide it and nothing else: whether any Newton step was accepted
    at all, and whether a closing active-set read ever moved the mask.  The
    termination word cannot decide it, because all three mechanisms share one
    termination word -- the compiled trip loop folds the host route's line
    search refusal into the same settled word a stall carries.  No accepted
    step is a refused line search; with steps accepted, a mask that moved is a
    walk cut short while still descending, and a mask that never moved is a
    trip that spent its whole Newton budget at a settled active set.
    """
    accepted = any(int(value) for value in row["newton_steps_per_trip"])
    moved = max([int(value) for value in row["trip_mask_difference_trace"]] or [0]) > 0
    if not accepted:
        return "refused_first_step"
    if moved:
        return "partial_walk"
    return "stall"


def _branches_of(profile: Any, state: Any, raster_flux: Any) -> dict[str, Any] | None:
    """Assemble the closed lobe and its legs from the solved lattice field.

    The field assembled is the solution's own lattice spline, whose cells the
    contour arcs are traced on, and not the receiver raster: the raster is a
    resampled image on a coarser fixed grid, and a level bracket it resolves
    too coarsely yields no axis-enclosing cycle at all, so a raster would
    report an empty boundary where the solved field has one.

    The raw level set of either grid is unsplit and unbounded: it starts
    inside the centre column and leaves the grid at the top and bottom, so
    drawing it as a boundary is what sprays coil-adjacent flux through the
    column. Assembling the solved field at the boundary flux splits it at the
    polished saddle and keeps the axis-enclosing lobe as one cycle.
    """
    if raster_flux is None:
        return None
    try:
        _masks, topology = profile.operator.read(state)
    except NoQualifiedAxisError:
        return None
    lattice = profile.lattice
    values = np.asarray(state[: lattice.node_count], dtype=float).reshape(
        tuple(int(value) for value in np.asarray(lattice.shape))
    )
    branches = jax.device_get(
        assemble_separatrix_branches(
            jnp.asarray(values.T),
            jnp.asarray(np.asarray(lattice.radius, dtype=float)),
            jnp.asarray(np.asarray(lattice.height, dtype=float)),
            jnp.asarray(topology.boundary_flux),
            jnp.asarray(np.asarray(topology.axis, dtype=float)),
        )
    )
    return {
        "closed_controls_rz": np.asarray(branches["closed_controls_rz"], dtype=float),
        "closed_valid": np.asarray(branches["closed_valid"], dtype=bool),
        "open_controls_rz": np.asarray(branches["open_controls_rz"], dtype=float),
        "open_valid": np.asarray(branches["open_valid"], dtype=bool),
        "open_branch_valid": np.asarray(branches["open_branch_valid"], dtype=bool),
        "boundary_flux": float(np.asarray(topology.boundary_flux)),
        "axis_flux": float(np.asarray(topology.axis_flux)),
        "well_formed": bool(np.asarray(branches["well_formed"])),
        "closed_candidate_count": int(
            np.asarray(branches["closed_candidate_count"]).item()
        ),
        "cycle_component_count": int(
            np.asarray(branches["cycle_component_count"]).item()
        ),
        "axis_enclosing_component_count": int(
            np.asarray(branches["axis_enclosing_component_count"]).item()
        ),
        "closed_segment_count": int(
            np.asarray(branches["closed_segment_count"]).item()
        ),
        "open_branch_count": int(np.asarray(branches["open_branch_count"]).item()),
        "overflow": bool(np.asarray(branches["overflow"])),
    }


def _group_nonconverged(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Group the non-converged edits by the reason their solve stopped.

    The three mechanism groups carry their member edit ids so the prose is
    checkable as data; the termination word is kept beside them because all
    three mechanisms share it.
    """
    mechanisms: dict[str, list[int]] = {
        "stall": [],
        "refused_first_step": [],
        "partial_walk": [],
    }
    terminations: dict[str, list[int]] = {}
    for row in rows:
        if row["converged"]:
            continue
        terminations.setdefault(row["termination"], []).append(row["edit_index"])
        mechanisms[_mechanism_of(row)].append(row["edit_index"])
    return {
        "failure_count": sum(len(index) for index in terminations.values()),
        "groups": mechanisms,
        "termination_groups": terminations,
        "convergence": {
            "converged_points": [row["edit_index"] for row in rows if row["converged"]],
            "trip_count_by_point": {
                str(row["edit_index"]): row["trip_count"] for row in rows
            },
        },
    }


def _write_diagnostics(
    path: Path,
    *,
    prepared: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    """Persist the per-edit convergence forensics beside the latency receipt."""
    document = {
        "schema": "nova.coil-edit-nonconvergence",
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "case": {
            "machine": "MAST",
            "shot": SHOT,
            "slice_index": SLICE_INDEX,
            "coil_family": prepared["coil_mapping"]["family"],
            "coil_circuit_index": int(prepared["circuit_index"]),
            "solver_policy": {
                "tolerance": COMPILED_SLICE_TOLERANCE,
                "newton_steps": COMPILED_SLICE_NEWTON_STEPS,
                "active_set_steps": COMPILED_SLICE_ACTIVE_SET_STEPS,
                "trip_boundary": reduced_newton.TRIP_BOUNDARY,
                "ladder_scoring": reduced_newton.LADDER_SCORING,
            },
            "constraint": prepared["vertical_centroid_actuator"],
        },
        "seed_probe_definition": (
            "one trip closed with zero Newton steps on this edit's current "
            "vector, seeded from the state the edit starts from; the reported "
            "residual is the seed's own residual on the edited operator, "
            "measured without the vertical current-centre row so it stays the "
            "same quantity every earlier sweep recorded"
        ),
        "summary": _group_nonconverged(rows),
        "edits": rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _persist_branches(
    payload: dict[str, Any], prefix: str, branches: dict[str, Any] | None
) -> None:
    """Write one assembled branch set under a key prefix, or write nothing."""
    if branches is None:
        return
    for key in (
        "closed_controls_rz",
        "closed_valid",
        "open_controls_rz",
        "open_valid",
        "open_branch_valid",
    ):
        payload[f"{prefix}{key}"] = np.asarray(branches[key])
    payload[f"{prefix}boundary_flux"] = np.asarray(branches["boundary_flux"])
    payload[f"{prefix}axis_flux"] = np.asarray(branches["axis_flux"])
    # The assembler returns zero geometry for any violation, so the verdict it
    # took is persisted beside the geometry: without it an archive of zeros
    # cannot say whether the level carried no axis-enclosing cycle, the graph
    # carried a junction, or a slot overflowed.
    payload[f"{prefix}well_formed"] = np.asarray(branches["well_formed"])
    payload[f"{prefix}closed_candidate_count"] = np.asarray(
        branches["closed_candidate_count"]
    )
    for name in (
        "cycle_component_count",
        "axis_enclosing_component_count",
        "closed_segment_count",
        "open_branch_count",
    ):
        payload[f"{prefix}{name}"] = np.asarray(branches[name])
    payload[f"{prefix}overflow"] = np.asarray(branches["overflow"])


SADDLE_FLUX_TOLERANCE = 1.0e-9


def _field_admitted_saddle(
    radius: Any,
    height: Any,
    psi: Any,
    axis: Any,
    xpoint_rz: Any,
) -> tuple[np.ndarray, float, Any]:
    """Read a persisted field's own admitted saddle coordinate and level.

    A persisted field's X-point coordinate and its boundary flux have to be
    read off the same field the panel contours, inside one tensor-spline fit.
    A coordinate carried over from the solve lattice names a point in a
    different field, and the flux read beside it -- or a level carried from the
    lattice -- misses the raster's own stationary cell by far more than the
    pairing tolerance, so the traced lobe leaves through a divertor leg instead
    of closing on its separatrix.

    ``xpoint_rz`` only has to land in the saddle's cell; the returned
    coordinate and value are that cell's own polished stationary pair. Where
    the field carries no stationary cell the locator and its flux are returned
    unchanged.
    """
    surface = fit_tensor_spline(
        jnp.asarray(radius), jnp.asarray(height), jnp.asarray(psi)
    )
    locator = surface(jnp.asarray(xpoint_rz[0]), jnp.asarray(xpoint_rz[1]))
    contour = traced_spline_contour(
        jnp.asarray(psi),
        jnp.asarray(radius),
        jnp.asarray(height),
        locator,
        40,
        8,
        surface=surface,
        axis_rz=jnp.asarray(axis).reshape(2),
    )
    stationary = np.asarray(contour["saddle_stationary"]).reshape(-1)
    saddle_value = np.asarray(contour["saddle_value"]).reshape(-1)
    saddle_rz = np.asarray(contour["saddle_rz"]).reshape(-1, 2)
    reference = np.asarray(xpoint_rz, dtype=float).reshape(2)
    distance = np.where(
        stationary, np.linalg.norm(saddle_rz - reference[None, :], axis=-1), np.inf
    )
    nearest = int(np.argmin(distance))
    if not np.isfinite(distance[nearest]):
        return reference, float(np.asarray(locator)), surface
    return (
        np.asarray(saddle_rz[nearest], dtype=float),
        float(np.asarray(saddle_value[nearest])),
        surface,
    )


def _persisted_positions(data: Any) -> list[int]:
    """Return the state positions an archive carries a persisted field for."""
    keys = getattr(data, "files", None)
    if keys is None:
        keys = list(data.keys())
    return sorted(int(key.split("_")[1]) for key in keys if key.startswith("psi_"))


def _persisted_saddle_residuals(data: Any) -> dict[int, float]:
    """Read every persisted X-point pair back against its own stored field.

    The archive is what a panel draws from, so the pair is measured where it
    is stored: each stored field is refit independently and its value
    evaluated at that state's own stored X-point coordinate, and the
    difference against the stored flux is the disagreement the archive
    actually carries. Nothing is assumed zero here -- this is the same
    measurement the writer refuses an archive for, and against a field stored
    beside a field it is the audit of whether the pair is one terminal state.

    A stored index outside either stored array is a pair that cannot be read
    back at all, and is refused rather than skipped.
    """
    radius = np.asarray(data["radius"], dtype=float)
    height = np.asarray(data["height"], dtype=float)
    residuals: dict[int, float] = {}
    for position in _persisted_positions(data):
        surface = fit_tensor_spline(
            jnp.asarray(radius),
            jnp.asarray(height),
            jnp.asarray(np.asarray(data[f"psi_{position}"], dtype=float)),
        )
        x_points = np.asarray(data[f"xpoints_{position}"], dtype=float).reshape(-1, 2)
        flux = np.asarray(data[f"xpoint_flux_{position}"], dtype=float).reshape(-1)
        index = int(np.asarray(data[f"saddle_index_{position}"]))
        if not (0 <= index < x_points.shape[0] and 0 <= index < flux.shape[0]):
            raise ValueError(
                "panel state %d persists a saddle index %d outside its stored "
                "X-point pair" % (position, index)
            )
        coordinate = x_points[index]
        residuals[position] = abs(
            float(np.asarray(surface(coordinate[0], coordinate[1])))
            - float(flux[index])
        )
    return residuals


def _refuse_inconsistent_persisted_pairs(data: Any) -> dict[int, float]:
    """Raise unless every persisted pair reads back as its own stored field."""
    residuals = _persisted_saddle_residuals(data)
    for position, residual in residuals.items():
        if residual > SADDLE_FLUX_TOLERANCE:
            raise ValueError(
                "panel state %d persists an xpoint_flux that is not its own "
                "field at its own X-point coordinate: residual %.3e Wb exceeds "
                "%.1e" % (position, residual, SADDLE_FLUX_TOLERANCE)
            )
    return residuals


def _write_panel_data(
    path: Path,
    *,
    profile: Any,
    panel_states: list[dict[str, Any]],
    reference_panel: dict[str, Any],
) -> None:
    """Persist the terminal raster fields the poloidal panel contours."""
    payload: dict[str, Any] = {
        "radius": np.asarray(reference_panel["radius"], dtype=float),
        "height": np.asarray(reference_panel["height"], dtype=float),
    }
    payload["reference_psi"] = _grid_field(
        reference_panel["psi"], reference_panel["shape"]
    )
    payload["reference_separatrix"] = np.asarray(
        reference_panel["separatrix"], dtype=float
    )
    payload["reference_axis"] = np.asarray(
        reference_panel["nulls"]["axis"], dtype=float
    )
    payload["reference_xpoints"] = np.asarray(
        reference_panel["nulls"]["x_points"], dtype=float
    )
    _persist_branches(payload, "reference_", reference_panel.get("branches"))
    payload["reference_xpoint_flux"] = np.asarray(
        reference_panel["nulls"]["x_point_flux"], dtype=float
    )
    payload["reference_saddle_index"] = np.asarray(
        reference_panel["nulls"]["saddle_index"], dtype=int
    )
    wall = profile.operator
    payload["wall_coordinate"] = np.asarray(wall.wall.coordinate, dtype=float).reshape(
        -1, 2
    )
    payload["wall_offsets"] = np.asarray(wall.wall_unit_offsets, dtype=int)
    payload["wall_closed"] = np.asarray(wall.wall_unit_closed, dtype=bool)
    payload["wall_kinds"] = np.asarray(tuple(wall.wall_unit_kinds), dtype=str)
    payload["edit_index"] = np.asarray(
        [state["edit_index"] for state in panel_states], dtype=int
    )
    centroid = reference_panel.get("vertical_centroid")
    if centroid is not None:
        payload["vertical_centroid_target_m"] = np.asarray(float(centroid["target_m"]))
        payload["vertical_centroid_reference_m"] = np.asarray(
            float(centroid["reference_m"])
        )
        payload["vertical_centroid_tolerance_m"] = np.asarray(
            float(centroid["tolerance_m"])
        )
    payload["edit_fraction"] = np.asarray(
        [state["edit_fraction"] for state in panel_states], dtype=float
    )
    payload["terminal_residual"] = np.asarray(
        [state["terminal_residual"] for state in panel_states], dtype=float
    )
    payload["converged"] = np.asarray(
        [state["converged"] for state in panel_states], dtype=bool
    )
    payload["trip_count"] = np.asarray(
        [state["trip_count"] for state in panel_states], dtype=int
    )
    payload["termination"] = np.asarray(
        [state["termination"] for state in panel_states], dtype=str
    )
    payload["achieved_class"] = np.asarray(
        [state["class"] for state in panel_states], dtype=str
    )
    wall = _wall_units(profile.operator)
    inside = _wall_interior(payload["radius"], payload["height"], wall)
    for position, state in enumerate(panel_states):
        if state["psi"] is None:
            continue
        field = _grid_field(state["psi"], state["shape"])
        payload[f"psi_{position}"] = field
        payload[f"separatrix_{position}"] = np.asarray(state["separatrix"], dtype=float)
        axis = np.asarray(state["nulls"]["axis"], dtype=float).reshape(2)
        payload[f"axis_{position}"] = axis
        x_points = np.asarray(state["nulls"]["x_points"], dtype=float).reshape(-1, 2)
        saddle_index = int(np.asarray(state["nulls"]["saddle_index"]))
        locator = (
            x_points[saddle_index] if 0 <= saddle_index < x_points.shape[0] else axis
        )
        # Every persisted quantity of this edit is read off the field persisted
        # for this edit, so the coordinate the panel marks and the level the
        # census traces belong to one terminal state.
        coordinate, level, surface = _field_admitted_saddle(
            payload["radius"], payload["height"], field, axis, locator
        )
        flux = (
            np.asarray(state["nulls"]["x_point_flux"], dtype=float).reshape(-1).copy()
        )
        x_points = x_points.copy()
        # One range guard for the coordinate and the flux together: a state
        # whose flux row is shorter than its coordinate row would otherwise
        # take the raster coordinate beside an unoverwritten archived flux.
        in_range = (
            0 <= saddle_index < x_points.shape[0] and 0 <= saddle_index < flux.shape[0]
        )
        if in_range:
            archived = x_points[saddle_index]
            # The archived pair's own disagreement with the field persisted
            # beside it. It is the same quantity the read-back guard refuses a
            # stored pair for, recorded before the substitution so the audit
            # trail carries what the raster pair replaced rather than a zero.
            payload[f"saddle_flux_archived_delta_{position}"] = np.asarray(
                float(flux[saddle_index])
                - float(np.asarray(surface(archived[0], archived[1]))),
                dtype=float,
            )
            x_points[saddle_index] = coordinate
            flux[saddle_index] = level
        payload[f"xpoints_{position}"] = x_points
        payload[f"xpoint_flux_{position}"] = flux
        payload[f"saddle_index_{position}"] = np.asarray(saddle_index, dtype=int)
        _persist_branches(
            payload,
            f"branches_{position}_",
            _assemble_raster_branches(
                payload["radius"],
                payload["height"],
                field,
                axis,
                x_points,
                saddle_index,
                inside,
            ),
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    # The guard measures the archive on disk rather than the arrays that were
    # handed to it: the stored field is refit and read at the stored
    # coordinate, so what passes is the pair a panel will actually draw.
    with np.load(path, allow_pickle=False) as archive:
        stored = {key: np.asarray(archive[key]) for key in archive.files}
    for position, residual in _refuse_inconsistent_persisted_pairs(stored).items():
        payload[f"saddle_flux_residual_{position}"] = np.asarray(residual, dtype=float)
    np.savez_compressed(path, **payload)


def _panel_edits(data: Any) -> tuple[int, int]:
    """Return the converged and non-converged positions the panel draws.

    One of each is chosen by the largest terminal residual, so the panel
    shows the widest separation the sweep produced, not an arbitrary row.
    """
    converged = np.asarray(data["converged"], dtype=bool)
    residual = np.asarray(data["terminal_residual"], dtype=float)
    finite = np.isfinite(residual)
    passed = np.flatnonzero(converged & finite)
    failed = np.flatnonzero(~converged & finite)
    if passed.size == 0 or failed.size == 0:
        raise RuntimeError("the panel needs one converged and one non-converged edit")
    return int(passed[np.argmax(residual[passed])]), int(
        failed[np.argmax(residual[failed])]
    )


def _panel_wall(data: Any) -> tuple[Any, ...]:
    """Rebuild the typed wall units from the persisted flat coordinates."""
    coordinate = np.asarray(data["wall_coordinate"], dtype=float).reshape(-1, 2)
    offsets = np.asarray(data["wall_offsets"], dtype=int)
    closed = np.asarray(data["wall_closed"], dtype=bool)
    kinds = tuple(str(value) for value in np.asarray(data["wall_kinds"]))
    return tuple(
        WallUnit(
            coordinate[start:stop, 0],
            coordinate[start:stop, 1],
            kind=kinds[index],
            closed=bool(closed[index]),
        )
        for index, (start, stop) in enumerate(
            zip(offsets[:-1], offsets[1:], strict=True)
        )
    )


def _optional_scalar(data: Any, name: str) -> float | None:
    """Return one persisted scalar, or None where the archive predates it."""
    if name not in data.files:
        return None
    value = float(np.asarray(data[name]))
    return value if np.isfinite(value) else None


def _optional_flag(data: Any, name: str) -> bool | None:
    """Return one persisted boolean, or None where the archive predates it."""
    if name not in data.files:
        return None
    return bool(np.asarray(data[name]))


def _optional_int(data: Any, name: str) -> int | None:
    """Return one persisted integer, or None where the archive predates it."""
    if name not in data.files:
        return None
    return int(np.asarray(data[name]))


def _load_branches(data: Any, prefix: str) -> dict[str, Any] | None:
    """Return one persisted branch set under a prefix, or None if absent."""
    key = f"{prefix}closed_controls_rz"
    if key not in data.files:
        return None
    loaded = {
        name: np.asarray(data[f"{prefix}{name}"])
        for name in (
            "closed_controls_rz",
            "closed_valid",
            "open_controls_rz",
            "open_valid",
            "open_branch_valid",
        )
    }
    loaded["boundary_flux"] = _optional_scalar(data, f"{prefix}boundary_flux")
    loaded["axis_flux"] = _optional_scalar(data, f"{prefix}axis_flux")
    loaded["well_formed"] = _optional_flag(data, f"{prefix}well_formed")
    loaded["closed_candidate_count"] = _optional_int(
        data, f"{prefix}closed_candidate_count"
    )
    loaded["cycle_component_count"] = _optional_int(
        data, f"{prefix}cycle_component_count"
    )
    loaded["axis_enclosing_component_count"] = _optional_int(
        data, f"{prefix}axis_enclosing_component_count"
    )
    loaded["closed_segment_count"] = _optional_int(
        data, f"{prefix}closed_segment_count"
    )
    loaded["open_branch_count"] = _optional_int(data, f"{prefix}open_branch_count")
    loaded["source"] = "archive"
    return loaded


def _raster_boundary_flux(
    radius: Any,
    height: Any,
    psi: Any,
    axis: Any,
    xpoints: Any,
    saddle_index: int,
) -> float:
    """Return a persisted raster field's own boundary flux at its admitted saddle.

    The level a field is traced at has to be that field's own stationary value.
    A level carried in from the solve lattice the raster resamples misses the
    raster's stationary value by far more than the decision tolerance, and the
    four-crossing cell holding the saddle then pairs by corner sign and runs
    the axis-enclosing lobe out along a divertor leg. The reference point only
    has to land in the saddle's cell; the value returned is that cell's own
    polished stationary value.
    """
    candidates = np.atleast_2d(np.asarray(xpoints, dtype=float))
    if 0 <= saddle_index < candidates.shape[0]:
        reference = candidates[saddle_index]
    else:
        reference = np.asarray(axis, dtype=float).reshape(2)
    level = boundary_flux_at_admitted_saddle(
        jnp.asarray(np.asarray(psi, dtype=float)),
        jnp.asarray(np.asarray(radius, dtype=float)),
        jnp.asarray(np.asarray(height, dtype=float)),
        jnp.asarray(np.asarray(axis, dtype=float).reshape(2)),
        jnp.asarray(reference, dtype=jnp.float64),
    )
    return float(np.asarray(level))


def _raster_axis_flux(psi: Any, inside: Any) -> float:
    """Return a raster field's own axis flux as its interior extremum.

    The axis is the field's extremum inside the vessel, and masking to the wall
    interior is what keeps coil-adjacent flux from stretching the drawn band
    past the plasma.
    """
    values = np.where(
        np.asarray(inside, dtype=bool), np.asarray(psi, dtype=float), -np.inf
    )
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("the raster field carries no value inside the wall")
    return float(np.max(finite))


def _assemble_raster_branches(
    radius: Any,
    height: Any,
    psi: Any,
    axis: Any,
    xpoints: Any,
    saddle_index: int,
    inside: Any,
) -> dict[str, Any]:
    """Assemble a persisted raster field's own lobe and legs for the panel.

    A sweep persists the lattice-field assembly; an archive written before that
    landed carries only the raster, and the raster is the field this panel
    contours. Tracing the raster at the raster's own boundary flux keeps the
    drawn branch and the drawn contours the same field, which an assembly
    imported from the lattice is not.
    """
    configure_dtypes()
    level = _raster_boundary_flux(radius, height, psi, axis, xpoints, saddle_index)
    assembled = jax.device_get(
        assemble_separatrix_branches(
            jnp.asarray(np.asarray(psi, dtype=float)),
            jnp.asarray(np.asarray(radius, dtype=float)),
            jnp.asarray(np.asarray(height, dtype=float)),
            jnp.asarray(level),
            jnp.asarray(np.asarray(axis, dtype=float).reshape(2)),
        )
    )
    return {
        "closed_controls_rz": np.asarray(assembled["closed_controls_rz"], dtype=float),
        "closed_valid": np.asarray(assembled["closed_valid"], dtype=bool),
        "open_controls_rz": np.asarray(assembled["open_controls_rz"], dtype=float),
        "open_valid": np.asarray(assembled["open_valid"], dtype=bool),
        "open_branch_valid": np.asarray(assembled["open_branch_valid"], dtype=bool),
        "boundary_flux": level,
        "axis_flux": _raster_axis_flux(psi, inside),
        "well_formed": bool(np.asarray(assembled["well_formed"])),
        "closed_candidate_count": int(
            np.asarray(assembled["closed_candidate_count"]).item()
        ),
        "cycle_component_count": int(
            np.asarray(assembled["cycle_component_count"]).item()
        ),
        "axis_enclosing_component_count": int(
            np.asarray(assembled["axis_enclosing_component_count"]).item()
        ),
        "closed_segment_count": int(
            np.asarray(assembled["closed_segment_count"]).item()
        ),
        "open_branch_count": int(np.asarray(assembled["open_branch_count"]).item()),
        "overflow": bool(np.asarray(assembled["overflow"])),
        "source": "raster-saddle",
    }


def _panel_load(data_path: Path) -> dict[str, Any]:
    """Load the persisted panel fields, wall units and shared levels."""
    with np.load(data_path, allow_pickle=False) as data:
        radius = np.asarray(data["radius"], dtype=float)
        height = np.asarray(data["height"], dtype=float)
        reference = np.asarray(data["reference_psi"], dtype=float)
        wall = _panel_wall(data)
        inside = _wall_interior(radius, height, wall)
        reference_axis = np.asarray(data["reference_axis"], dtype=float)
        reference_xpoints = np.asarray(data["reference_xpoints"], dtype=float)
        reference_saddle_index = (
            int(np.asarray(data["reference_saddle_index"]))
            if "reference_saddle_index" in data.files
            else 0
        )
        reference_branches = _load_branches(data, "reference_")
        if reference_branches is None:
            reference_branches = _assemble_raster_branches(
                radius,
                height,
                reference,
                reference_axis,
                reference_xpoints,
                reference_saddle_index,
                inside,
            )
        boundary_flux = reference_branches["boundary_flux"]
        if boundary_flux is None:
            boundary_flux = _raster_boundary_flux(
                radius,
                height,
                reference,
                reference_axis,
                reference_xpoints,
                reference_saddle_index,
            )
        axis_flux = reference_branches["axis_flux"]
        if axis_flux is None:
            axis_flux = _raster_axis_flux(reference, inside)
        # The reference set's own plasma range, so the shared levels lie on the
        # flux the plasma occupies instead of spanning the whole raster: the
        # raw span reaches coil-adjacent flux and contours it into the column.
        levels = poloidal.contour_levels(
            reference, count=14, boundary=boundary_flux, axis=axis_flux
        )
        loaded: dict[str, Any] = {
            "radius": radius,
            "height": height,
            "reference": reference,
            "levels": levels,
            "reference_axis": reference_axis,
            "reference_separatrix": np.asarray(
                data["reference_separatrix"], dtype=float
            ),
            "reference_xpoints": reference_xpoints,
            "reference_branches": reference_branches,
            "reference_boundary_flux": float(boundary_flux),
            "reference_axis_flux": float(axis_flux),
            "reference_saddle_index": reference_saddle_index,
            "wall": wall,
            "inside": inside,
            "vertical_centroid": (
                {
                    "reference_m": float(
                        np.asarray(data["vertical_centroid_reference_m"])
                    ),
                    "target_m": float(np.asarray(data["vertical_centroid_target_m"])),
                    "tolerance_m": float(
                        np.asarray(data["vertical_centroid_tolerance_m"])
                    ),
                }
                if "vertical_centroid_target_m" in data.files
                else None
            ),
        }
        passed, failed = _panel_edits(data)
        selected = (("converged", passed), ("failed", failed))
        for label, position in selected:
            loaded[label] = {}

            panel = loaded[label]
            panel["edit_index"] = int(np.asarray(data["edit_index"])[position])
            panel["fraction"] = float(np.asarray(data["edit_fraction"])[position])
            panel["residual"] = float(np.asarray(data["terminal_residual"])[position])

            panel["trips"] = int(np.asarray(data["trip_count"])[position])
            panel["converged"] = bool(np.asarray(data["converged"])[position])
            panel["termination"] = str(np.asarray(data["termination"])[position])
            panel["class_name"] = str(np.asarray(data["achieved_class"])[position])

            saddle_name = "saddle_index_%d" % position
            panel["saddle_index"] = (
                int(np.asarray(data[saddle_name])) if saddle_name in data.files else 0
            )
            for key in ("psi", "separatrix", "axis", "xpoints"):
                name = "%s_%d" % (key, position)
                panel[key] = np.asarray(data[name], dtype=float)
            panel["branches"] = _load_branches(data, "branches_%d_" % position)
            if panel["branches"] is None:
                panel["branches"] = _assemble_raster_branches(
                    radius,
                    height,
                    panel["psi"],
                    panel["axis"],
                    panel["xpoints"],
                    panel["saddle_index"],
                    inside,
                )
    return loaded


def _wall_interior(radius: Any, height: Any, wall: Any) -> np.ndarray:
    """Return the boolean raster mask of the wall's interior."""
    grid_radius, grid_height = np.meshgrid(
        np.asarray(radius, dtype=float), np.asarray(height, dtype=float)
    )
    points = np.column_stack((grid_radius.reshape(-1), grid_height.reshape(-1)))
    keep = np.asarray(inside_wall_units(points, wall), dtype=bool)
    return keep.reshape(len(height), len(radius))


def _draw_branch_set(
    axis: Any, branches: dict[str, Any] | None, color: str, fallback: Any = None
) -> dict[str, int]:
    """Draw an assembled branch set, or the raw level set where none assembled.

    The assembler returns zero geometry for any violation of its terms, so a
    present set is not the same as a drawn boundary: a rejected set falls back
    to its crossings drawn as unconnected marks, and reports the fallback
    beside its zero counts rather than leaving the panel without the boundary
    the reader is looking for.  A set whose terms did not travel with it -- an
    archive written before the terms accompanied the geometry -- is drawn as
    stored.
    """
    tally = {"closed_drawn": 0, "open_drawn": 0, "unassembled": 0}
    verdict = None if branches is None else branches.get("well_formed")
    if branches is not None and (verdict is None or bool(verdict)):
        tally.update(
            poloidal.draw_separatrix_branches(
                axis,
                branches,
                style=DEFAULT_INK.variant(separatrix_color=color),
                closed_color=color,
                open_color=color,
            )
        )
        return tally
    array = (
        np.asarray(fallback, dtype=float) if fallback is not None else np.empty((0, 2))
    )
    if array.size:
        # The fallback is a scan-order crossing list, not an ordered path: two
        # consecutive entries are neighbours in the raster walk, not endpoints
        # of one segment.  Joining them draws a fan of chords across the vessel,
        # so the crossings are drawn as separate unconnected marks and the
        # caption says the boundary is not assembled here.
        axis.plot(
            array[:, 0],
            array[:, 1],
            color=color,
            linestyle="none",
            marker="+",
            markersize=3.0,
            markeredgewidth=0.7,
        )
        tally["unassembled"] = 1
    return tally


def _draw_null_set(
    axis: Any,
    magnetic_axis: Any,
    x_points: Any,
    saddle_index: int,
    color: str,
    wall: Any = None,
) -> dict[str, int]:
    """Draw the admitted saddle filled and the other qualified nulls hollow.

    The admitted saddle is the one the read selected onto the boundary flux and
    is the only member tested against the wall; the remaining qualified nulls
    are drawn hollow and unfiltered, so a set that carries a second saddle
    outside the vessel reads as two stationary points rather than one.
    """
    array = np.atleast_2d(np.asarray(x_points, dtype=float))
    if not 0 <= saddle_index < array.shape[0]:
        admitted, other = array, array[:0]
    else:
        admitted = array[[saddle_index]]
        other = np.delete(array, saddle_index, axis=0)
    return poloidal.draw_nulls(
        axis,
        magnetic_axis=magnetic_axis,
        x_points=admitted,
        other_x_points=other,
        style=DEFAULT_INK.variant(
            axis_color=color,
            xpoint_color=color,
            axis_marker="^",
            xpoint_marker="X",
        ),
        contain=wall,
    )


def _paint_panel(
    axis: Any, loaded: dict[str, Any], panel: dict[str, Any]
) -> dict[str, dict[str, int]]:
    """Draw one edit's terminal flux over the reference field.

    Both rasters are masked to the wall interior before contouring, so a level
    that reaches coil-adjacent flux is not contoured into the centre column.
    Both null sets are drawn in their own styles, each with its admitted saddle
    filled and its remaining qualified nulls hollow, so a solved null is never
    mistaken for the reference one.

    Returns what each painter actually drew, keyed by set. The admitted saddle
    is dropped when it falls outside the wall and the hollow cup is empty for a
    set with no second qualified null, so the two sets can be drawn with
    different glyph vocabularies; carrying the counts out of the painter is what
    lets the caption state the drawing rather than assert it.
    """
    radius = loaded["radius"]
    height = loaded["height"]
    levels = loaded["levels"]
    inside = loaded["inside"]
    poloidal.draw_flux_contours(
        axis,
        radius,
        height,
        np.where(inside, loaded["reference"], np.nan),
        levels,
        color="#9aa4b2",
    )
    poloidal.draw_flux_contours(
        axis,
        radius,
        height,
        np.where(inside, panel["psi"], np.nan),
        levels,
        color="#cc7722",
    )
    poloidal.draw_wall(axis, units=loaded["wall"])

    loaded["reference_branches_drawn"] = _draw_branch_set(
        axis, loaded["reference_branches"], "#3366cc", loaded["reference_separatrix"]
    )
    panel["branches_drawn"] = _draw_branch_set(
        axis, panel["branches"], "#cc7722", panel["separatrix"]
    )

    loaded["reference_nulls_drawn"] = _draw_null_set(
        axis,
        loaded["reference_axis"],
        loaded["reference_xpoints"],
        loaded["reference_saddle_index"],
        "#3366cc",
        loaded["wall"],
    )
    panel["nulls_drawn"] = _draw_null_set(
        axis,
        panel["axis"],
        panel["xpoints"],
        panel["saddle_index"],
        "#cc7722",
        loaded["wall"],
    )
    poloidal_axes(axis)
    return {
        "reference": dict(loaded["reference_nulls_drawn"]),
        "solved": dict(panel["nulls_drawn"]),
    }


def _branch_terms(branches: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return an assembled set's verdict terms, or None where none was drawn.

    The assembler returns zero geometry for any violation, so the terms travel
    beside the geometry in the receipt: an empty drawn set alone cannot say
    whether the level carried no axis-enclosing cycle, the graph carried a
    junction, or a slot overflowed.
    """
    if branches is None:
        return None
    return {
        "source": branches.get("source"),
        "well_formed": branches.get("well_formed"),
        "closed_candidate_count": branches.get("closed_candidate_count"),
        "cycle_component_count": branches.get("cycle_component_count"),
        "axis_enclosing_component_count": branches.get(
            "axis_enclosing_component_count"
        ),
        "closed_segment_count": branches.get("closed_segment_count"),
        "open_branch_count": branches.get("open_branch_count"),
        "boundary_flux": branches.get("boundary_flux"),
    }


def _panel_summary(panel: dict[str, Any]) -> dict[str, Any]:
    """Return the json-safe scalars the panel caption reports."""
    return {
        "edit_index": panel["edit_index"],
        "fraction": panel["fraction"],
        "residual": panel["residual"],
        "trips": panel["trips"],
        "converged": panel["converged"],
        "termination": panel["termination"],
        "achieved_class": panel["class_name"],
        "branches_drawn": dict(panel.get("branches_drawn", {})),
        "nulls_drawn": dict(panel.get("nulls_drawn", {})),
        "branches": _branch_terms(panel.get("branches")),
    }


def _saddle_note(x_points: Any, saddle_index: int) -> str:
    """Return the admitted saddle of one null set as a caption fragment."""
    array = np.atleast_2d(np.asarray(x_points, dtype=float))
    if not 0 <= saddle_index < array.shape[0]:
        return "saddle absent"
    point = array[saddle_index]
    return "saddle (%.3f, %+.3f) m" % (point[0], point[1])


def _branch_style_note(drawn: dict[str, int]) -> str:
    """Return the caption fragment describing how one set was drawn.

    A set the assembler rejected carries zero geometry, so it is drawn as its
    own scan-order crossings, unconnected; the caption has to say which of the
    two drawings the reader is looking at, because the two are indistinguishable
    on the panel and one of them is not an assembled boundary.
    """
    if drawn.get("unassembled"):
        return "unassembled, crossings drawn unconnected"
    return "lobe solid, legs dashed"


def _null_style_note(tally: dict[str, int]) -> str:
    """Return the caption fragment for what a null set was drawn as.

    The caption reads the glyphs off the painter's own tally rather than
    restating the intent: a set whose only qualified null is the admitted
    saddle has no hollow cup to draw, and an admitted saddle outside the wall
    is dropped instead of drawn. A caption claiming hollow markers for a panel
    that carries none sends the reader looking for a glyph that is not there.
    """
    drawn = int(tally.get("x_points_drawn", 0))
    dropped = int(tally.get("x_points_dropped_outside_wall", 0))
    other = int(tally.get("other_x_points_drawn", 0))
    phrase = "as drawn %d admitted filled"
    arguments = [drawn]
    if dropped:
        phrase += ", %d admitted dropped outside the wall"
        arguments.append(dropped)
    if other:
        phrase += ", %d other qualified hollow"
        arguments.append(other)
    else:
        phrase += ", no other qualified nulls"
    return phrase % tuple(arguments)


def _null_ordering(x_points: Any, saddle_index: int, axis: Any) -> str:
    """Compare the admitted saddle's height with the magnetic axis's.

    Both the reference and every solved state carry a qualified null set, and
    which one is admitted as the boundary saddle differs between them: the
    reference's sits above its axis, the solved states' below.  The caption
    states the comparison rather than assuming it, so a regenerated reference
    whose ordering changed does not silently contradict its own caption.
    """
    array = np.atleast_2d(np.asarray(x_points, dtype=float))
    if not 0 <= saddle_index < array.shape[0]:
        return "saddle absent"
    point = np.asarray(axis, dtype=float).reshape(2)
    return "above its axis" if array[saddle_index][1] > point[1] else "below its axis"


def _constraint_note(centroid: dict[str, Any] | None) -> str:
    """Name the vertical current-centre row the solved states were held to."""
    if centroid is None:
        return "(none recorded in this archive)"
    return "at %.6f m (the reference's own centre, tolerance %.0e m)" % (
        centroid["target_m"],
        centroid["tolerance_m"],
    )


def _render_panel(data_path: Path, figure_path: Path) -> dict[str, Any]:
    """Write the converged-versus-non-converged poloidal panel."""
    loaded = _panel_load(data_path)
    figure, axes = plt.subplots(1, 2, figsize=(10.6, 4.6), constrained_layout=True)

    tallies: dict[str, dict[str, dict[str, int]]] = {}
    for axis, label in zip(axes, ("converged", "failed"), strict=True):
        panel = loaded[label]
        tallies[label] = _paint_panel(axis, loaded, panel)
        axis.set_title(
            "edit %d  %+d%%  residual %.3e  converged %s  trips %d"
            % (
                panel["edit_index"],
                round(100.0 * panel["fraction"]),
                panel["residual"],
                "yes" if panel["converged"] else "no",
                panel["trips"],
            )
        )

    reference_note = _saddle_note(
        loaded["reference_xpoints"], loaded["reference_saddle_index"]
    )
    reference_tally = tallies["converged"]["reference"]
    solved_note = _saddle_note(
        loaded["failed"]["xpoints"], loaded["failed"]["saddle_index"]
    )
    constraint_note = _constraint_note(loaded.get("vertical_centroid"))
    caption = (
        "terminal poloidal flux on shared levels between the axis and boundary "
        "flux (%.4f to %.4f Wb)  |  every solved state imposes the "
        "vertical current-centre row %s  |  reference is the unedited "
        "equilibrium, an "
        "upper-null state whose admitted saddle sits above its axis, while every "
        "solved state admits a lower null (blue, admitted saddle %s)  |  "
        "reference set blue: %s, admitted %s, %s  |  solved set orange: %s, "
        "admitted %s, %s  |  wall drawn"
        % (
            loaded["reference_axis_flux"],
            loaded["reference_boundary_flux"],
            constraint_note,
            _null_ordering(
                loaded["reference_xpoints"],
                loaded["reference_saddle_index"],
                loaded["reference_axis"],
            ),
            _branch_style_note(loaded.get("reference_branches_drawn", {})),
            reference_note,
            _null_style_note(reference_tally),
            _branch_style_note(loaded["failed"].get("branches_drawn", {})),
            solved_note,
            _null_style_note(tallies["failed"]["solved"]),
        )
    )
    figure.suptitle(caption, fontsize=9)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, dpi=140)
    vector_path = figure_path.with_suffix(".svg")
    figure.savefig(vector_path)
    plt.close(figure)

    receipt_path = figure_path.with_suffix(".json")
    document = {
        "figure": str(figure_path),
        "figure_vector": str(vector_path),
        "levels": [float(value) for value in loaded["levels"]],
        "level_band": {
            "axis_flux": loaded["reference_axis_flux"],
            "boundary_flux": loaded["reference_boundary_flux"],
        },
        "reference_branches": _branch_terms(loaded["reference_branches"]),
        "reference_branches_drawn": dict(loaded.get("reference_branches_drawn", {})),
        "reference_nulls_drawn": reference_tally,
        "vertical_centroid": loaded.get("vertical_centroid"),
        "converged": _panel_summary(loaded["converged"]),
        "non_converged": _panel_summary(loaded["failed"]),
    }
    receipt = dict(document)
    receipt["caption"] = " ".join(caption.split())
    receipt["receipt"] = str(receipt_path)
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return document
    return document


def _declared_path(path: Path) -> str:
    """Return a repository-relative path, or the absolute one outside the root.

    A run that keeps its figures in a scratch directory writes outside the
    repository, and a receipt field must name where the file actually is rather
    than fail to describe it.
    """
    resolved = Path(path)
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _receipt_document(
    *,
    prepared: dict[str, Any],
    carrier: dict[str, Any],
    rows: list[dict[str, Any]],
    program_reused_from_edit: int | None,
    cache_events: dict[str, float | int],
    cache: Any,
    solve_persistent_hits_start: int,
    solve_persistent_misses_start: int,
    solve_persistent_saved_start: float,
    terminal_raster: dict[str, Any] | None,
    measurement_state: str,
    elapsed_seconds: float,
    exit_marker: int | None,
    figure: Path,
    raster_figure: Path | None = None,
) -> dict[str, Any]:
    """Assemble the wire receipt over the edits recorded so far.

    Writing this after each edit lands makes an expiry lose one row at the
    most; the final call marks the measurement complete.
    """
    warm_ms = np.asarray(
        [row["wall_milliseconds"] for row in rows[1:]], dtype=np.float64
    )
    if warm_ms.size:
        median_warm_ms = float(np.median(warm_ms))
        warm_min_ms = float(warm_ms.min())
        warm_max_ms = float(warm_ms.max())
    else:
        median_warm_ms = None
        warm_min_ms = None
        warm_max_ms = None
    all_cache_hits_after_first = all(
        row["compilation_cache"] == "hit" for row in rows[1:]
    )
    one_program = program_reused_from_edit is not None and program_reused_from_edit == 0
    all_converged = all(row["converged"] for row in rows)
    latency_target_met = (
        median_warm_ms is not None
        and median_warm_ms < INTERACTIVE_LATENCY_TARGET_MILLISECONDS
    )
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
    finite_displacements = boundary_displacements[np.isfinite(boundary_displacements)]
    persistent_hits = int(cache_events["hits"]) - solve_persistent_hits_start
    persistent_misses = int(cache_events["misses"]) - solve_persistent_misses_start
    persistent_saved_seconds = (
        float(cache_events["saved_seconds"]) - solve_persistent_saved_start
    )
    first_edit_compiled = bool(rows) and rows[0]["persistent_cache_miss_count"] >= 1
    later_edits_add_no_misses = all(
        row["persistent_cache_miss_count"] == 0 for row in rows[1:]
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
        "first_edit_compiles_program": first_edit_compiled,
        "program_reused_to_the_end": one_program,
        "all_later_edits_are_cache_hits": all_cache_hits_after_first,
        "later_edits_add_no_compiles": later_edits_add_no_misses,
        "all_edits_converged": all_converged,
        "boundary_displacement_is_finite": bool(
            np.all(np.isfinite(boundary_displacements))
        ),
        "raster_target_rides_once": terminal_raster is not None,
        "median_warm_wall_meets_tens_of_milliseconds_target": latency_target_met,
    }
    passed = all(gates.values())
    final_exit_marker = 0 if passed else 2
    if measurement_state == "complete":
        verdict = "PASS" if passed else "FAIL"
        marker = final_exit_marker if exit_marker is None else exit_marker
    else:
        verdict = "PENDING"
        marker = 0
    circuit_index = int(prepared["circuit_index"])
    return {
        "schema": "nova.coil-edit-latency",
        "measurement_state": measurement_state,
        "verdict": verdict,
        "gates": gates,
        "interactive_path": {
            "route": "compiled slice (reduced fixed point, one fixed-shape "
            "program re-entered per edit)",
            "solve_entry": ("reduced_newton.solve_constrained_reduced_newton_compiled"),
            "prescribed_current": "traced 101-circuit vector, replacement semantics",
            "raster_target": "operator-held receiver grid built once at construction",
            "constraint": prepared["vertical_centroid_actuator"],
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
            "path": _declared_path(Path(__file__)),
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
            "measurement_host_marker": os.environ.get(CPU_PROVENANCE_MARKER),
        "off_host_marker": os.environ.get(OFF_HOST_PROVENANCE_MARKER),
            "elapsed_seconds": elapsed_seconds,
            "exit_marker": marker,
        },
        "persistent_compilation_cache": cache.receipt()
        | {
            "solve_hit_count": persistent_hits,
            "solve_miss_count": persistent_misses,
            "solve_compile_seconds_saved": persistent_saved_seconds,
            "process_total_hit_count": int(cache_events["hits"]),
            "process_total_miss_count": int(cache_events["misses"]),
            "process_total_compile_seconds_saved": float(cache_events["saved_seconds"]),
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
            "target_current_a": float(prepared["target_current"]),
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
            "shot_coil_current_a": float(
                np.asarray(prepared["prescribed_current"])[circuit_index]
            ),
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
            "minimum_warm_wall_milliseconds": warm_min_ms,
            "maximum_warm_wall_milliseconds": warm_max_ms,
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
                if median_warm_ms is not None
                else "no warm per-edit wall yet"
            ),
            "converged_edit_count": sum(row["converged"] for row in rows),
            "failing_points": [
                row["edit_index"] for row in rows if not row["converged"]
            ],
            "trip_count_minimum": (
                min(row["trip_count"] for row in rows) if rows else None
            ),
            "trip_count_median": (
                float(np.median([row["trip_count"] for row in rows])) if rows else None
            ),
            "trip_count_maximum": (
                max(row["trip_count"] for row in rows) if rows else None
            ),
            "maximum_boundary_displacement_m": (
                float(finite_displacements.max()) if finite_displacements.size else None
            ),
        },
        "edits": rows,
        "figure": _declared_path(figure),
        "raster_figure": _declared_path(
            DEFAULT_RASTER_FIGURE if raster_figure is None else raster_figure
        ),
    }


def run(
    output: Path,
    figure: Path,
    carrier_path: Path,
    diagnostics: Path,
    panel_data: Path,
    panel_figure: Path,
    raster_figure: Path = DEFAULT_RASTER_FIGURE,
    grid_points: int | None = None,
) -> dict[str, Any]:
    """Compile once and measure successive warm prescribed-current edits.

    ``grid_points`` selects the uniform per-axis node count and must agree
    with the grid the supplied carrier was built for; the default keeps the
    stored-axis stride every banked measurement on this driver was taken on.
    """
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
        profile, prepared, carrier = _prepare_case(carrier_path, grid_points)
        _preflight_panel_wall(profile)
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
        panel_states: list[dict[str, Any]] = []
        constraint_pairs = (prepared["vertical_centroid_pair"],)
        state = initial
        program = None
        program_reused_from_edit: int | None = None
        reference_raster_separatrix = prepared["reference_raster_separatrix"]
        for index, (fraction, current) in enumerate(
            zip(EDIT_FRACTIONS, edit_vectors, strict=True)
        ):
            # The seed's own residual on THIS edit's operator, measured before
            # the first trip: one trip closed with zero Newton steps evaluates
            # the trip boundary at the seed state, so the reported residual is
            # the seed's, not the post-correction residual a full edit
            # reports.  It runs off the clock and before the miss counter is
            # sampled, so the probe's one-off program build lands in neither
            # an edit's wall or its cache accounting.
            seed_probe = _seed_probe(
                profile, state, current, requested_class, target_current
            )
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
                constraint_pairs=constraint_pairs,
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
                # The boundary column is the receiver-grid separatrix
                # displacement: the symmetric-sup difference between the
                # edited raster's separatrix and the base-frame reference
                # separatrix read from the same fixed receiver grid.
                if raster_flux is not None:
                    separatrix_count = int(
                        np.asarray(raster_flux.separatrix_vertex_count)
                    )
                    separatrix = np.asarray(raster_flux.separatrix)[:separatrix_count]
                    if separatrix_count and len(reference_raster_separatrix):
                        distances = np.linalg.norm(
                            separatrix[:, None, :]
                            - reference_raster_separatrix[None, :, :],
                            axis=2,
                        )
                        boundary_displacement = float(
                            max(
                                np.max(np.min(distances, axis=0)),
                                np.max(np.min(distances, axis=1)),
                            )
                        )
                        displacement_source = "receiver_grid_separatrix_symmetric_sup"
                    else:
                        boundary_displacement = None
                        displacement_source = "receiver_grid_separatrix_unreadable"
                else:
                    boundary_displacement = None
                    displacement_source = "receiver_grid_separatrix_unreadable"
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
                "trip_residual_trace": [
                    float(value) for value in result.active_set_residuals
                ],
                "trip_mask_difference_trace": [
                    int(value) for value in result.active_set_mask_differences
                ],
                "newton_steps_per_trip": [
                    int(value) for value in result.newton_steps_per_trip
                ],
                "rejected_steps_per_trip": [
                    int(value) for value in result.rejected_steps_per_trip
                ],
                "achieved_class": _achieved_class(profile, result.state),
                "seed_probe": seed_probe,
                "vertical_centroid": _vertical_centroid_row(result),
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
            panel_states.append(
                {
                    "edit_index": index,
                    "edit_fraction": float(fraction),
                    "terminal_residual": float(np.asarray(result.terminal_residual)),
                    "converged": bool(np.asarray(result.converged)),
                    "trip_count": int(np.asarray(result.active_set_iterations)),
                    "termination": result.termination_name,
                    "class": _achieved_class(profile, result.state)["class"],
                    "radius": (
                        None
                        if raster_flux is None
                        else np.asarray(raster_flux.radius, dtype=float)
                    ),
                    "height": (
                        None
                        if raster_flux is None
                        else np.asarray(raster_flux.height, dtype=float)
                    ),
                    "psi": (
                        None
                        if raster_flux is None
                        else np.asarray(raster_flux.psi, dtype=float)
                    ),
                    "shape": (
                        None
                        if raster_flux is None
                        else np.asarray(raster_flux.shape, dtype=int)
                    ),
                    "separatrix": (
                        None
                        if raster_flux is None
                        else np.asarray(raster_flux.separatrix, dtype=float)[
                            : int(np.asarray(raster_flux.separatrix_vertex_count))
                        ]
                    ),
                    "nulls": _null_points(profile, result.state),
                    "branches": _branches_of(profile, result.state, raster_flux),
                }
            )
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
            # The receipt persists per edit as it lands: an expiry or node loss
            # loses at most one row, never the run.
            landing = _receipt_document(
                prepared=prepared,
                carrier=carrier,
                rows=rows,
                program_reused_from_edit=program_reused_from_edit,
                cache_events=cache_events,
                cache=cache,
                solve_persistent_hits_start=solve_persistent_hits_start,
                solve_persistent_misses_start=solve_persistent_misses_start,
                solve_persistent_saved_start=solve_persistent_saved_start,
                terminal_raster=_terminal_raster_document(equilibrium, profile),
                measurement_state="in_progress",
                elapsed_seconds=time.perf_counter() - total_started,
                exit_marker=None,
                figure=figure,
                raster_figure=raster_figure,
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(landing, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            _write_diagnostics(diagnostics, prepared=prepared, rows=rows)
            _write_panel_data(
                panel_data,
                profile=profile,
                panel_states=panel_states,
                reference_panel=prepared["reference_panel"],
            )

        terminal_raster = _terminal_raster_document(equilibrium, profile)
        persistent_hits = int(cache_events["hits"]) - solve_persistent_hits_start
        persistent_misses = int(cache_events["misses"]) - solve_persistent_misses_start
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
        if equilibrium is not None and equilibrium.raster_flux is not None:
            _render_raster(equilibrium.raster_flux, raster_figure)
        try:
            panel = _render_panel(panel_data, panel_figure)
            print("PANEL=" + json.dumps(panel, sort_keys=True), flush=True)
        except (RuntimeError, KeyError) as error:
            print(f"PANEL_SKIPPED reason={error}", flush=True)
        receipt = _receipt_document(
            prepared=prepared,
            carrier=carrier,
            rows=rows,
            program_reused_from_edit=program_reused_from_edit,
            cache_events=cache_events,
            cache=cache,
            solve_persistent_hits_start=solve_persistent_hits_start,
            solve_persistent_misses_start=solve_persistent_misses_start,
            solve_persistent_saved_start=solve_persistent_saved_start,
            terminal_raster=terminal_raster,
            measurement_state="complete",
            elapsed_seconds=time.perf_counter() - total_started,
            exit_marker=None,
            figure=figure,
            raster_figure=raster_figure,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        exit_marker = int(receipt["runtime"]["exit_marker"])
        print(f"EXIT_MARKER={exit_marker}", flush=True)
        if exit_marker:
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


def _probe(
    carrier_path: Path, *, impose_vertical_centroid: bool = False
) -> dict[str, Any]:
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
    constraint_pairs: tuple[ConstraintPair, ...] = ()
    if impose_vertical_centroid:
        reference_centroid_z = float(
            np.asarray(
                profile.current_moment_observation(
                    jnp.asarray(seed),
                    support=MomentIntegralSupport.ALL_DOMAIN,
                    requested_class=requested_class,
                    target_current=target_current,
                ).centroid_z
            )
        )
        pair, actuator = _vertical_centroid_pair(
            profile,
            policy,
            jnp.asarray(seed),
            requested_class=requested_class,
            target_current=target_current,
            target=reference_centroid_z,
        )
        constraint_pairs = (pair,)
    result = _compiled_edit(
        profile,
        jnp.asarray(seed),
        edited,
        requested_class,
        target_current,
        None,
        constraint_pairs=constraint_pairs,
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
        "vertical_centroid": _vertical_centroid_row(result),
        "vertical_centroid_actuator": (actuator if impose_vertical_centroid else None),
    }


def _sbatch_script(arguments: argparse.Namespace) -> str:
    log_directory = arguments.log_directory.resolve()
    worktree = ROOT.resolve()
    interpreter = Path("/home/ITER/mcintos/Code/nova/.venv/bin/python")
    command = (
        f"PYTHONPATH={worktree} {interpreter} "
        "benchmarks/coil_edit_latency.py run "
        f"--carrier {arguments.carrier.resolve()} "
        f"--output {arguments.output.resolve()} "
        f"--figure {arguments.figure.resolve()} "
        f"--diagnostics {arguments.diagnostics.resolve()} "
        f"--panel-data {arguments.panel_data.resolve()} "
        f"--panel-figure {arguments.panel_figure.resolve()} "
        f"--raster-figure {arguments.raster_figure.resolve()}"
    )
    return f"""#!/usr/bin/env bash
#SBATCH --job-name=coil-edit-latency
#SBATCH --partition=betelgeuse
#SBATCH --reservation=gpu_0003_grpA
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
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
    # Export TMPDIR in the submit environment as well as the payload:
    # slurmstepd inherits the login-node value before the payload's own
    # export runs, so point it at /tmp from the start.
    submit_env = dict(os.environ)
    submit_env["TMPDIR"] = "/tmp"
    completed = subprocess.run(
        ["sbatch", "--parsable"],
        input=_sbatch_script(arguments),
        check=True,
        capture_output=True,
        text=True,
        env=submit_env,
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
    run_parser.add_argument("--diagnostics", type=Path, default=DEFAULT_DIAGNOSTICS)
    run_parser.add_argument("--panel-data", type=Path, default=DEFAULT_PANEL_DATA)
    run_parser.add_argument("--panel-figure", type=Path, default=DEFAULT_PANEL_FIGURE)
    run_parser.add_argument("--raster-figure", type=Path, default=DEFAULT_RASTER_FIGURE)
    run_parser.add_argument(
        "--carrier", type=Path, default=response_carrier.DEFAULT_CARRIER
    )
    run_parser.add_argument("--grid-points", type=int, default=None)
    for name in ("sbatch", "submit"):
        job_parser = subparsers.add_parser(name)
        job_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
        job_parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
        job_parser.add_argument("--diagnostics", type=Path, default=DEFAULT_DIAGNOSTICS)
        job_parser.add_argument("--panel-data", type=Path, default=DEFAULT_PANEL_DATA)
        job_parser.add_argument(
            "--panel-figure", type=Path, default=DEFAULT_PANEL_FIGURE
        )
        job_parser.add_argument(
            "--raster-figure", type=Path, default=DEFAULT_RASTER_FIGURE
        )
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
    panel_parser = subparsers.add_parser("panel")
    panel_parser.add_argument("--panel-data", type=Path, default=DEFAULT_PANEL_DATA)
    panel_parser.add_argument("--panel-figure", type=Path, default=DEFAULT_PANEL_FIGURE)
    check_parser = subparsers.add_parser("checkcase")
    check_parser.add_argument(
        "--carrier", type=Path, default=response_carrier.DEFAULT_CARRIER
    )
    probe_parser = subparsers.add_parser("probe")
    probe_parser.add_argument(
        "--carrier", type=Path, default=response_carrier.DEFAULT_CARRIER
    )
    probe_parser.add_argument(
        "--impose-vertical-centroid",
        action="store_true",
        help="carry the vertical current-centre row on the probed edit",
    )
    arguments = parser.parse_args()
    if arguments.command == "run":
        run(
            arguments.output,
            arguments.figure,
            arguments.carrier,
            arguments.diagnostics,
            arguments.panel_data,
            arguments.panel_figure,
            arguments.raster_figure,
            arguments.grid_points,
        )
    elif arguments.command == "panel":
        print(
            json.dumps(
                _render_panel(arguments.panel_data, arguments.panel_figure),
                indent=2,
                sort_keys=True,
            )
        )
    elif arguments.command == "checkcase":
        print(json.dumps(_check_case(arguments.carrier), indent=2, sort_keys=True))
    elif arguments.command == "probe":
        print(
            json.dumps(
                _probe(
                    arguments.carrier,
                    impose_vertical_centroid=arguments.impose_vertical_centroid,
                ),
                indent=2,
                sort_keys=True,
            )
        )
    elif arguments.command == "sbatch":
        print(_sbatch_script(arguments), end="")
    elif arguments.command == "submit":
        _submit(arguments)
    else:
        _harvest(arguments.output)


if __name__ == "__main__":
    main()
