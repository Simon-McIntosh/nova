#!/usr/bin/env python3
"""Refill fixed device slots while host workers assemble steering frames."""

from __future__ import annotations

from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass, field, replace
import multiprocessing
from pathlib import Path
import subprocess
import time
from typing import Any, Callable, Protocol, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import zarr

from benchmarks import mast_response_carrier_warm as response_carrier
from benchmarks.efit_forward_parity_slice import (
    FIXED_POINT_CRITERION,
    TOTAL_FLUX_FACTOR,
)
from benchmarks.forward_labeller_throughput import (
    NEWTON_STEPS,
    SHOT_STORE,
    _centroid_pair,
    _circuit_names,
    _requested_class,
    _slices_seed,
)
from nova.equilibrium import reduced_newton
from nova.equilibrium.steering_frames import (
    SteeringAction,
    SteeringFrame,
    assemble_frame,
)
from nova.equilibrium.topology import TopologyClass
from scripts.labeller_batch.shard import (
    BRANCH_GUARD_TOLERANCE_M,
    PreparedLabeller,
    _centroid_coordinates,
    _forward_receipt,
    _internal_geometry,
    _masked_frame,
    _slice_inputs,
    _solve_policy,
    _write_companion as _write_diagnostics,
    _write_json,
    _write_session_file,
    policy_digest,
)

STATE_SIZE = 1_126
CURRENT_SIZE = 101
ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class SourceIdentity:
    """Immutable source identity captured once by the driver process."""

    nova_revision: str
    nova_equilibrium_tree: str
    labeller_batch_tree: str

    @classmethod
    def capture(cls) -> SourceIdentity:
        def revision(specification: str) -> str:
            return subprocess.run(
                ["git", "-C", str(ROOT), "rev-parse", specification],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()

        return cls(
            nova_revision=revision("HEAD"),
            nova_equilibrium_tree=revision("HEAD:nova/equilibrium"),
            labeller_batch_tree=revision("HEAD:scripts/labeller_batch"),
        )


@dataclass(frozen=True)
class SliceInput:
    """One admitted reconstruction slice and its frame ride-along values."""

    shot: int
    row: int
    time: float
    initial_state: np.ndarray
    prescribed_current: np.ndarray
    target_current: float
    requested_class: int
    centroid_target_z: float
    p_prime_psi_norm: np.ndarray
    p_prime: np.ndarray
    ff_prime_psi_norm: np.ndarray
    ff_prime: np.ndarray
    reset_warm_state: bool = False


@dataclass(frozen=True)
class ShotInput:
    """Admitted work and excluded records for one ranked shot."""

    shot: int
    slices: tuple[SliceInput, ...]
    excluded_records: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class EngineBatch:
    """Dense arrays passed to one engine step."""

    active: np.ndarray
    shot: np.ndarray
    row: np.ndarray
    time: np.ndarray
    initial_state: np.ndarray
    prescribed_current: np.ndarray
    target_current: np.ndarray
    requested_class: np.ndarray
    centroid_target_z: np.ndarray

    def validate(self) -> int:
        """Validate the fixed boundary used by the vectorised engine."""
        batch = int(self.active.shape[0])
        actual = {name: getattr(self, name).shape for name in self.__dataclass_fields__}
        expected = {
            "active": (batch,),
            "shot": (batch,),
            "row": (batch,),
            "time": (batch,),
            "initial_state": (batch, STATE_SIZE),
            "prescribed_current": (batch, CURRENT_SIZE),
            "target_current": (batch,),
            "requested_class": (batch,),
            "centroid_target_z": (batch,),
        }
        if actual != expected:
            raise ValueError(f"engine input contract mismatch: {actual!r}")
        return batch


@dataclass(frozen=True)
class SolvedSlice:
    """Non-array payload accompanying one active engine row."""

    result: reduced_newton.ReducedNewtonResult | None
    applied_current: np.ndarray
    record: dict[str, Any]


@dataclass(frozen=True)
class EngineResult:
    """One engine result with a stable dense-array contract."""

    state: np.ndarray
    converged: np.ndarray
    termination: np.ndarray
    trips: np.ndarray
    terminal_residual: np.ndarray
    centroid: np.ndarray
    conditioned: np.ndarray
    labelled_fields: dict[str, np.ndarray] = field(default_factory=dict)
    solved: tuple[SolvedSlice | None, ...] = ()

    def validate(self, batch: int) -> None:
        actual = {
            "state": self.state.shape,
            "converged": self.converged.shape,
            "termination": self.termination.shape,
            "trips": self.trips.shape,
            "terminal_residual": self.terminal_residual.shape,
            "centroid": self.centroid.shape,
            "conditioned": self.conditioned.shape,
        }
        expected = {
            "state": (batch, STATE_SIZE),
            "converged": (batch,),
            "termination": (batch,),
            "trips": (batch,),
            "terminal_residual": (batch,),
            "centroid": (batch, 2),
            "conditioned": (batch,),
        }
        if actual != expected:
            raise ValueError(f"engine output contract mismatch: {actual!r}")
        if self.solved and len(self.solved) != batch:
            raise ValueError("engine side payload does not match the batch")
        for name, values in self.labelled_fields.items():
            if values.shape[:1] != (batch,):
                raise ValueError(f"labelled field {name!r} lacks a batch axis")


class BatchEngine(Protocol):
    """Array boundary shared with the future vectorised implementation."""

    def step(self, batch: EngineBatch) -> EngineResult: ...


@dataclass
class SequentialCompiledEngine:
    """Run one compiled production slice program per active device slot."""

    prepared: PreparedLabeller
    device_count: int
    condition_on_guard_failure: bool
    free_programs: list[reduced_newton.ReducedProgram | None] = field(init=False)
    conditioned_programs: list[reduced_newton.ReducedProgram | None] = field(init=False)

    def __post_init__(self) -> None:
        if self.device_count < 1:
            raise ValueError("device_count must be positive")
        if len(jax.devices()) < self.device_count:
            available = len(jax.devices())
            raise ValueError(
                f"requested {self.device_count} devices, JAX exposes {available}"
            )
        self.free_programs = []
        self.conditioned_programs = []

    def _ensure_slots(self, size: int) -> None:
        missing = size - len(self.free_programs)
        if missing > 0:
            self.free_programs.extend([None] * missing)
            self.conditioned_programs.extend([None] * missing)

    def _solve_slot(self, batch: EngineBatch, index: int) -> SolvedSlice:
        device = jax.devices()[index % self.device_count]
        started = time.perf_counter()
        free_result = conditioned_result = None
        free_exception = conditioning_exception = None
        free_centroid_r = free_centroid_z = free_centroid_error = free_guard = None
        conditioned_centroid_r = conditioned_centroid_z = None
        conditioned_centroid_error = conditioned_guard = None
        conditioned = False
        free_wall_seconds = conditioned_wall_seconds = 0.0
        initial = np.asarray(batch.initial_state[index])
        requested_value = int(batch.requested_class[index])
        target_current = float(batch.target_current[index])
        current = np.asarray(batch.prescribed_current[index])

        free_started = time.perf_counter()
        try:
            with jax.default_device(device):
                free_result = reduced_newton.solve_reduced_newton_compiled(
                    self.prepared.profile.operator,
                    jax.device_put(initial, device),
                    requested_class=jax.device_put(
                        jnp.asarray(requested_value, dtype=jnp.int8), device
                    ),
                    target_current=jax.device_put(jnp.asarray(target_current), device),
                    prescribed_current=jax.device_put(jnp.asarray(current), device),
                    tolerance=FIXED_POINT_CRITERION,
                    newton_steps=NEWTON_STEPS,
                    program=self.free_programs[index],
                    stream=False,
                )
            self.free_programs[index] = free_result.program
            jax.block_until_ready(free_result.state)
            free_wall_seconds = time.perf_counter() - free_started
            free_centroid_r, free_centroid_z = _centroid_coordinates(
                self.prepared, free_result.state, target_current
            )
            free_centroid_error = free_centroid_z - float(
                batch.centroid_target_z[index]
            )
            free_guard = bool(
                np.isfinite(free_centroid_error)
                and abs(free_centroid_error) <= BRANCH_GUARD_TOLERANCE_M
            )
        except Exception as error:
            free_wall_seconds = time.perf_counter() - free_started
            free_exception = f"{type(error).__name__}: {error}"

        should_condition = self.condition_on_guard_failure and (
            free_result is None
            or not bool(free_result.converged)
            or free_guard is not True
        )
        if should_condition:
            conditioned = True
            conditioned_started = time.perf_counter()
            try:
                requested = jnp.asarray(requested_value, dtype=jnp.int8)
                pair, _selection = _centroid_pair(
                    self.prepared.profile,
                    jnp.asarray(initial),
                    target=float(batch.centroid_target_z[index]),
                    unknown=None,
                    target_current=target_current,
                    requested=requested,
                    names=_circuit_names(self.prepared.policy_evidence),
                )
                with jax.default_device(device):
                    conditioned_result = (
                        reduced_newton.solve_constrained_reduced_newton_compiled(
                            self.prepared.profile,
                            jax.device_put(initial, device),
                            constraint_pairs=(pair,),
                            requested_class=jax.device_put(requested, device),
                            target_current=jax.device_put(
                                jnp.asarray(target_current), device
                            ),
                            prescribed_current=jax.device_put(
                                jnp.asarray(current), device
                            ),
                            tolerance=FIXED_POINT_CRITERION,
                            newton_steps=NEWTON_STEPS,
                            program=self.conditioned_programs[index],
                            stream=False,
                        )
                    )
                self.conditioned_programs[index] = conditioned_result.program
                jax.block_until_ready(conditioned_result.state)
                conditioned_wall_seconds = time.perf_counter() - conditioned_started
                conditioned_centroid_r, conditioned_centroid_z = _centroid_coordinates(
                    self.prepared, conditioned_result.state, target_current
                )
                conditioned_centroid_error = conditioned_centroid_z - float(
                    batch.centroid_target_z[index]
                )
                conditioned_guard = bool(
                    np.isfinite(conditioned_centroid_error)
                    and abs(conditioned_centroid_error) <= BRANCH_GUARD_TOLERANCE_M
                )
            except Exception as error:
                conditioned_wall_seconds = time.perf_counter() - conditioned_started
                conditioning_exception = f"{type(error).__name__}: {error}"

        selected = conditioned_result if conditioned else free_result
        solve_wall_seconds = time.perf_counter() - started
        applied_current = current
        if (
            selected is not None
            and getattr(selected, "prescribed_current", None) is not None
        ):
            applied_current = np.asarray(selected.prescribed_current)
        final_centroid_r = conditioned_centroid_r if conditioned else free_centroid_r
        final_centroid_z = conditioned_centroid_z if conditioned else free_centroid_z
        final_centroid_error = (
            conditioned_centroid_error if conditioned else free_centroid_error
        )
        selected_exception = conditioning_exception if conditioned else free_exception
        record = {
            "row": int(batch.row[index]),
            "time": float(batch.time[index]),
            "written": True,
            "excluded": False,
            "geometry_masked": True,
            "converged": False,
            "qualified": False,
            "terminal_residual": float(selected.terminal_residual)
            if selected
            else None,
            "trips": int(selected.active_set_iterations) if selected else 0,
            "newton_steps": int(sum(selected.newton_steps_per_trip)) if selected else 0,
            "free_trips": int(free_result.active_set_iterations) if free_result else 0,
            "conditioned_trips": (
                int(conditioned_result.active_set_iterations)
                if conditioned_result
                else 0
            ),
            "wall_seconds": solve_wall_seconds,
            "free_wall_seconds": free_wall_seconds,
            "conditioned_wall_seconds": conditioned_wall_seconds,
            "termination": selected.termination_name if selected else "slice_exception",
            "conditioned": conditioned,
            "conditioning_flag": conditioned,
            "conditioning_target_source": "efm/current_centrd_z"
            if conditioned
            else None,
            "free_converged": bool(free_result.converged) if free_result else False,
            "conditioned_converged": (
                bool(conditioned_result.converged) if conditioned_result else None
            ),
            "free_branch_guard_ok": free_guard,
            "conditioned_branch_guard_ok": conditioned_guard,
            "free_centroid_error_m": free_centroid_error,
            "conditioned_centroid_error_m": conditioned_centroid_error,
            "achieved_current_centroid_r": final_centroid_r,
            "achieved_current_centroid_z": final_centroid_z,
            "target_current_centroid_z": float(batch.centroid_target_z[index]),
            "centroid_error_m": final_centroid_error,
            "target_source": "efm/current_centrd_z",
            "branch_guard_ok": False,
            "requested_class": requested_value,
        }
        if selected_exception is not None:
            record["exception"] = selected_exception
        if free_exception is not None:
            record["free_solve_exception"] = free_exception
        if conditioning_exception is not None:
            record["conditioning_exception"] = conditioning_exception
        if selected is not None:
            selected = replace(selected, program=None)
        return SolvedSlice(selected, np.asarray(applied_current), record)

    def step(self, batch: EngineBatch) -> EngineResult:
        size = batch.validate()
        self._ensure_slots(size)
        solved: list[SolvedSlice | None] = [None] * size
        state = np.asarray(batch.initial_state).copy()
        converged = np.zeros(size, dtype=bool)
        termination = np.full(size, -1, dtype=np.int32)
        trips = np.zeros(size, dtype=np.int32)
        residual = np.full(size, np.nan)
        centroid = np.full((size, 2), np.nan)
        conditioned = np.zeros(size, dtype=bool)
        for raw_index in np.flatnonzero(batch.active):
            index = int(raw_index)
            payload = self._solve_slot(batch, index)
            solved[index] = payload
            result = payload.result
            conditioned[index] = bool(payload.record["conditioned"])
            if result is None:
                continue
            state[index] = np.asarray(result.state)
            converged[index] = bool(result.converged)
            termination[index] = int(result.termination_reason)
            trips[index] = int(result.active_set_iterations)
            residual[index] = float(result.terminal_residual)
            centroid[index] = (
                payload.record["achieved_current_centroid_r"],
                payload.record["achieved_current_centroid_z"],
            )
        return EngineResult(
            state=state,
            converged=converged,
            termination=termination,
            trips=trips,
            terminal_residual=residual,
            centroid=centroid,
            conditioned=conditioned,
            labelled_fields={
                "requested_class": np.asarray(batch.requested_class, dtype=np.int8)
            },
            solved=tuple(solved),
        )


_ASSEMBLY_PREPARED: PreparedLabeller | None = None
_CONDITION_ON_GUARD_FAILURE = False


@dataclass(frozen=True)
class AssemblyRequest:
    """Solve and ride-along payload sent to one host process."""

    item: SliceInput
    solved: SolvedSlice


@dataclass(frozen=True)
class AssembledSlice:
    """Host-produced frame and completed sequential manifest row."""

    shot: int
    row: int
    time: float
    frame: SteeringFrame | None
    record: dict[str, Any]
    context: dict[str, Any]
    final_ok: bool
    state: np.ndarray | None
    assembly_wall_seconds: float


def assemble_frame_on_host(request: AssemblyRequest) -> AssembledSlice:
    """Build the production geometry and steering frame in a host process."""
    if _ASSEMBLY_PREPARED is None:
        raise RuntimeError("host assembly was not configured before forking")
    started = time.perf_counter()
    item, solved = request.item, request.solved
    result = solved.result
    record = dict(solved.record)
    frame = solve_receipt = None
    processing_exception = None
    if result is not None:
        try:
            solve_receipt = _forward_receipt(
                _ASSEMBLY_PREPARED,
                result,
                requested_class=jnp.asarray(item.requested_class, dtype=jnp.int8),
                target_current=item.target_current,
                prescribed_current=solved.applied_current,
                solve_wall_seconds=float(record["wall_seconds"]),
            )
            geometry = _internal_geometry(
                _ASSEMBLY_PREPARED,
                solve_receipt.terminal_state,
                diverted=item.requested_class == int(TopologyClass.DIVERTED),
            )
            frame = assemble_frame(
                solve_receipt,
                action=SteeringAction(
                    name="label",
                    delta=0.0,
                    commanded_control_points=np.empty((0, 2), dtype=float),
                ),
                carrier_identity=response_carrier.DEFAULT_CARRIER.stem,
                applied_current=solved.applied_current,
                p_prime_psi_norm=item.p_prime_psi_norm,
                p_prime=item.p_prime,
                ff_prime_psi_norm=item.ff_prime_psi_norm,
                ff_prime=item.ff_prime,
                p_prime_source="efm",
                reference_centroid_z=item.centroid_target_z,
                compensating_current=(
                    None
                    if bool(record["conditioned"])
                    else np.zeros(int(_CONDITION_ON_GUARD_FAILURE))
                ),
                internal_geometry=geometry,
                wall=_ASSEMBLY_PREPARED.wall,
            )
        except Exception as error:
            processing_exception = f"{type(error).__name__}: {error}"
    exceptions = [
        value for value in (record.get("exception"), processing_exception) if value
    ]
    final_ok = result is not None and frame is not None and not exceptions
    record.update(
        geometry_masked=not final_ok,
        converged=bool(result.converged) if final_ok else False,
        qualified=bool(solve_receipt.qualified) if final_ok else False,
        branch_guard_ok=bool(frame.branch_guard_ok) if final_ok else False,
    )
    if exceptions:
        record["exception"] = "; ".join(exceptions)
    if processing_exception is not None:
        record["frame_exception"] = processing_exception
    context = {
        "current": solved.applied_current,
        "wall_seconds": float(record["wall_seconds"]),
        "trips": int(record["trips"]),
        "p_prime_psi_norm": item.p_prime_psi_norm,
        "p_prime": item.p_prime,
        "ff_prime_psi_norm": item.ff_prime_psi_norm,
        "ff_prime": item.ff_prime,
        "reference_centroid_z": item.centroid_target_z,
        "compensating_slots": int(_CONDITION_ON_GUARD_FAILURE),
    }
    return AssembledSlice(
        item.shot,
        item.row,
        item.time,
        frame,
        record,
        context,
        final_ok,
        np.asarray(result.state) if final_ok else None,
        time.perf_counter() - started,
    )


def load_shot(shot: int, *, max_slices: int | None) -> ShotInput:
    """Read the same admitted rows and reconstruction seeds as the shard."""
    group = zarr.open_group(str(SHOT_STORE / f"{shot}.zarr"), mode="r")["efm"]
    full_r = np.asarray(group["gridr"], dtype=np.float64)
    full_z = np.asarray(group["gridz"], dtype=np.float64)
    psi_norm = np.asarray(group["psi_norm"], dtype=np.float64)
    slices, excluded = [], []
    reset_warm = False
    admitted = 0
    for row in range(int(group["time"].shape[0])):
        inputs = _slice_inputs(group, row)
        if inputs is None:
            reset_warm = True
            excluded.append(
                {
                    "row": row,
                    "time": float(group["time"][row]),
                    "written": False,
                    "excluded": True,
                    "converged": False,
                    "qualified": False,
                    "exclusion": "no reconstruction",
                    "target_source": "efm/current_centrd_z",
                    "branch_guard_ok": False,
                    "conditioned": False,
                    "conditioning_target_source": None,
                    "free_branch_guard_ok": None,
                    "conditioned_branch_guard_ok": None,
                    "requested_class": int(_requested_class(group, row)),
                }
            )
            continue
        if max_slices is not None and admitted >= max_slices:
            break
        admitted += 1
        slices.append(
            SliceInput(
                shot=shot,
                row=row,
                time=float(inputs["time"]),
                initial_state=np.asarray(_slices_seed(group, row, full_r, full_z)),
                prescribed_current=np.asarray(inputs["current"], dtype=np.float64),
                target_current=abs(float(inputs["reference_plasma_current"])),
                requested_class=int(_requested_class(group, row)),
                centroid_target_z=float(inputs["target_centroid_z"]),
                p_prime_psi_norm=psi_norm,
                p_prime=-np.asarray(group["pprime"][row], dtype=np.float64)
                / TOTAL_FLUX_FACTOR,
                ff_prime_psi_norm=psi_norm,
                ff_prime=-np.asarray(group["ffprime"][row], dtype=np.float64)
                / TOTAL_FLUX_FACTOR,
                reset_warm_state=reset_warm,
            )
        )
        reset_warm = False
    if not slices:
        raise RuntimeError(f"shot {shot} has no admitted EFM slices")
    return ShotInput(shot, tuple(slices), tuple(excluded))


@dataclass
class _Slot:
    work: ShotInput
    index: int = 0
    warm_state: np.ndarray | None = None
    awaiting: Future[AssembledSlice] | None = None

    @property
    def done(self) -> bool:
        return self.index >= len(self.work.slices)

    def current(self) -> SliceInput:
        item = self.work.slices[self.index]
        if self.warm_state is None or item.reset_warm_state:
            return item
        return replace(item, initial_state=self.warm_state)


def _shot_manifest(
    prepared: PreparedLabeller,
    output_root: Path,
    work: ShotInput,
    assembled: Sequence[AssembledSlice],
    *,
    include_raster: bool,
    condition_on_guard_failure: bool,
    setup_wall_seconds: float,
    shot_wall_seconds: float,
    source_identity: SourceIdentity,
) -> dict[str, Any]:
    ordered = sorted(assembled, key=lambda item: item.row)
    template = next((item.frame for item in ordered if item.frame is not None), None)
    frames = [
        item.frame
        if item.frame is not None
        else _masked_frame(prepared, template=template, **item.context)
        for item in ordered
    ]
    rows = sorted(
        [dict(item.record) for item in ordered]
        + [dict(item) for item in work.excluded_records],
        key=lambda item: int(item["row"]),
    )
    companion_rows = [
        {
            "row": item.record["row"],
            "time": item.record["time"],
            "conditioned": item.record["conditioned"],
            "conditioning_target_source": item.record["conditioning_target_source"],
            "free_branch_guard_ok": item.record["free_branch_guard_ok"],
            "conditioned_branch_guard_ok": item.record["conditioned_branch_guard_ok"],
            "free_centroid_error_m": item.record["free_centroid_error_m"],
            "conditioned_centroid_error_m": item.record["conditioned_centroid_error_m"],
        }
        for item in ordered
    ]
    session = output_root / f"{work.shot}.nc"
    companion = output_root / f"{work.shot}.npz"
    manifest_path = output_root / f"{work.shot}.manifest.json"
    _write_session_file(
        frames,
        session.resolve(),
        time_values=[item.time for item in ordered],
        include_raster=include_raster,
    )
    _write_diagnostics(companion_rows, companion)
    converged = sum(
        bool(item.get("converged")) for item in rows if not item["excluded"]
    )
    policy = _solve_policy()
    manifest = {
        "schema": "nova-forward-labeller-shot",
        "shot": work.shot,
        "status": "complete",
        "session": str(session.resolve()),
        "companion": str(companion.resolve()),
        "nova_revision": source_identity.nova_revision,
        "nova_equilibrium_tree": source_identity.nova_equilibrium_tree,
        "labeller_batch_tree": source_identity.labeller_batch_tree,
        "declared_additions": [
            "manifest.nova_equilibrium_tree",
            "manifest.labeller_batch_tree",
            "manifest.declared_additions",
            "manifest.slices.requested_class",
        ],
        "carrier_identity": response_carrier.DEFAULT_CARRIER.stem,
        "carrier": prepared.carrier_evidence,
        "policy_digest": policy_digest(policy),
        "policy": policy.to_dict(),
        "constraint": {
            "mode": "diagnostic_branch_guard",
            "target_source": "efm/current_centrd_z",
            "tolerance_m": BRANCH_GUARD_TOLERANCE_M,
            "condition_on_guard_failure": condition_on_guard_failure,
            "conditioning_trigger": (
                "free solve raised, did not converge, or missed the branch guard"
            ),
        },
        "companion_fields_without_frame_home": [
            "conditioned",
            "conditioning_target_source",
            "free_branch_guard_ok",
            "conditioned_branch_guard_ok",
            "free_centroid_error_m",
            "conditioned_centroid_error_m",
        ],
        "include_raster": include_raster,
        "setup_wall_seconds": setup_wall_seconds,
        "shot_wall_seconds": shot_wall_seconds,
        "slice_count": len(rows),
        "admitted_slice_count": len(ordered),
        "written_slice_count": len(ordered),
        "converged_slice_count": converged,
        "unconverged_slice_count": len(ordered) - converged,
        "excluded_slice_count": len(work.excluded_records),
        "slices": rows,
        "companion_slice_count": len(companion_rows),
        "flux_function_grid_points": int(ordered[0].context["p_prime_psi_norm"].size),
    }
    _write_json(manifest, manifest_path)
    return manifest


class CorpusScheduler:
    """Keep ready slots on devices while completed rows assemble on hosts."""

    def __init__(
        self,
        *,
        engine: BatchEngine,
        device_count: int,
        batch_per_device: int,
        host_workers: int,
        assembler: Callable[[AssemblyRequest], AssembledSlice] = assemble_frame_on_host,
        include_raster: bool = False,
        condition_on_guard_failure: bool = False,
    ) -> None:
        if min(device_count, batch_per_device, host_workers) < 1:
            raise ValueError("device, batch and host worker counts must be positive")
        self.engine = engine
        self.device_count = device_count
        self.batch_per_device = batch_per_device
        self.host_workers = host_workers
        self.assembler = assembler
        self.include_raster = include_raster
        self.condition_on_guard_failure = condition_on_guard_failure

    @property
    def capacity(self) -> int:
        return self.device_count * self.batch_per_device

    def _pack(self, slots: Sequence[_Slot | None]) -> EngineBatch:
        ready = [slot is not None and slot.awaiting is None for slot in slots]
        items = [
            slot.current() if flag else None
            for slot, flag in zip(slots, ready, strict=True)
        ]
        return EngineBatch(
            active=np.asarray(ready, dtype=bool),
            shot=np.asarray(
                [item.shot if item else -1 for item in items], dtype=np.int64
            ),
            row=np.asarray(
                [item.row if item else -1 for item in items], dtype=np.int32
            ),
            time=np.asarray([item.time if item else np.nan for item in items]),
            initial_state=np.stack(
                [item.initial_state if item else np.zeros(STATE_SIZE) for item in items]
            ),
            prescribed_current=np.stack(
                [
                    item.prescribed_current if item else np.zeros(CURRENT_SIZE)
                    for item in items
                ]
            ),
            target_current=np.asarray(
                [item.target_current if item else 0.0 for item in items]
            ),
            requested_class=np.asarray(
                [item.requested_class if item else 0 for item in items], dtype=np.int8
            ),
            centroid_target_z=np.asarray(
                [item.centroid_target_z if item else 0.0 for item in items]
            ),
        )

    def run(
        self,
        ranked_shots: Sequence[ShotInput],
        output_root: Path,
        *,
        prepared: PreparedLabeller,
        source_identity: SourceIdentity,
    ) -> dict[str, Any]:
        output_root.mkdir(parents=True, exist_ok=True)
        pending = deque(
            work
            for work in ranked_shots
            if not (
                (output_root / f"{work.shot}.nc").is_file()
                and (output_root / f"{work.shot}.manifest.json").is_file()
            )
        )
        skipped = len(ranked_shots) - len(pending)
        slots: list[_Slot | None] = [None] * self.capacity
        completed: dict[int, list[AssembledSlice]] = {}
        shot_started: dict[int, float] = {}
        futures: dict[Future[AssembledSlice], int] = {}
        engine_wall = assembly_wall = 0.0
        engine_slices = written = written_shots = 0
        started = time.perf_counter()
        setup_unassigned = prepared.setup_wall_seconds

        def refill(index: int) -> None:
            if pending:
                work = pending.popleft()
                slots[index] = _Slot(work)
                completed[work.shot] = []
                shot_started[work.shot] = time.perf_counter()
            else:
                slots[index] = None

        def finish_future(future: Future[AssembledSlice]) -> None:
            nonlocal assembly_wall, written, written_shots, setup_unassigned
            index = futures.pop(future)
            assembled = future.result()
            slot = slots[index]
            if slot is None or slot.awaiting is not future:
                raise RuntimeError("completed assembly no longer owns its slot")
            completed[assembled.shot].append(assembled)
            assembly_wall += assembled.assembly_wall_seconds
            written += 1
            slot.warm_state = assembled.state if assembled.final_ok else None
            slot.index += 1
            slot.awaiting = None
            if slot.done:
                _shot_manifest(
                    prepared,
                    output_root,
                    slot.work,
                    completed.pop(slot.work.shot),
                    include_raster=self.include_raster,
                    condition_on_guard_failure=self.condition_on_guard_failure,
                    setup_wall_seconds=setup_unassigned,
                    shot_wall_seconds=time.perf_counter()
                    - shot_started.pop(slot.work.shot),
                    source_identity=source_identity,
                )
                setup_unassigned = 0.0
                written_shots += 1
                refill(index)

        for index in range(self.capacity):
            refill(index)
        global _ASSEMBLY_PREPARED, _CONDITION_ON_GUARD_FAILURE
        _ASSEMBLY_PREPARED = prepared
        _CONDITION_ON_GUARD_FAILURE = self.condition_on_guard_failure
        context = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(
            max_workers=self.host_workers, mp_context=context
        ) as pool:
            while any(slot is not None for slot in slots) or futures:
                batch = self._pack(slots)
                if np.any(batch.active):
                    step_started = time.perf_counter()
                    result = self.engine.step(batch)
                    engine_wall += time.perf_counter() - step_started
                    result.validate(self.capacity)
                    for raw_index in np.flatnonzero(batch.active):
                        index = int(raw_index)
                        slot = slots[index]
                        solved = result.solved[index]
                        if slot is None or solved is None:
                            raise RuntimeError(
                                "active engine row returned no solve payload"
                            )
                        future = pool.submit(
                            self.assembler, AssemblyRequest(slot.current(), solved)
                        )
                        slot.awaiting = future
                        futures[future] = index
                        engine_slices += 1
                ready = [future for future in futures if future.done()]
                if (
                    not ready
                    and futures
                    and not any(
                        slot is not None and slot.awaiting is None for slot in slots
                    )
                ):
                    ready = list(wait(futures, return_when=FIRST_COMPLETED).done)
                for future in ready:
                    finish_future(future)

        wall = time.perf_counter() - started
        engine_rate = engine_slices / engine_wall if engine_wall else 0.0
        pool_rate = written / assembly_wall if assembly_wall else 0.0
        return {
            "schema": "nova-forward-labeller-parallel-receipt",
            "engine": "sequential-compiled",
            "device_count": self.device_count,
            "batch_per_device": self.batch_per_device,
            "slot_count": self.capacity,
            "host_workers": self.host_workers,
            "shot_count": written_shots,
            "skipped_shot_count": skipped,
            "slice_count": engine_slices,
            "engine_wall_seconds": engine_wall,
            "engine_slices_per_second": engine_rate,
            "host_assembly_wall_seconds": assembly_wall,
            "host_assembly_wall_seconds_per_slice": assembly_wall / written,
            "pool_slices_per_second": pool_rate,
            "frames_per_second_written": written / wall,
            "wall_seconds": wall,
            "pool_below_engine": pool_rate < engine_rate,
        }
