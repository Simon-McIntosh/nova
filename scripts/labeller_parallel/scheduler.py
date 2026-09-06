#!/usr/bin/env python3
"""Refill a fixed device batch with time-ordered slices from ranked shots.

The scheduler owns only orchestration.  Its engine boundary is deliberately an
array contract, so the deterministic stub and the production batched engine
are interchangeable.  A slot stays with one shot until that shot ends; its
terminal state becomes the next slice's warm state.  Completed engine rows are
sent immediately to a host process pool, while the next device step proceeds.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import time
from typing import Any, Callable, Protocol, Sequence

import numpy as np

from nova.equilibrium.steering_frames import (
    N_DIVERTOR_LEG_POINTS,
    N_DIVERTOR_LEGS,
    N_RHO,
    N_SURFACE,
    N_THETA,
    TORAX_PROFILE_FIELDS,
    SteeringAction,
    SteeringFrame,
)
from scripts.labeller_batch.shard import (
    _write_companion as _write_diagnostics,
    _write_session_file,
)


STATE_SIZE = 1_126


@dataclass(frozen=True)
class SliceInput:
    """One admitted reconstruction slice in a shot."""

    shot: int
    row: int
    time: float
    initial_state: np.ndarray
    prescribed_current: np.ndarray
    target_current: float
    requested_class: int
    centroid_target_z: float


@dataclass(frozen=True)
class EngineBatch:
    """Dense arrays passed to one batched engine step."""

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
        """Validate the leading batch dimension and fixed state contract."""
        batch = int(self.active.shape[0])
        shapes = {
            "active": self.active.shape,
            "shot": self.shot.shape,
            "row": self.row.shape,
            "time": self.time.shape,
            "initial_state": self.initial_state.shape,
            "prescribed_current": self.prescribed_current.shape,
            "target_current": self.target_current.shape,
            "requested_class": self.requested_class.shape,
            "centroid_target_z": self.centroid_target_z.shape,
        }
        expected = {
            "active": (batch,),
            "shot": (batch,),
            "row": (batch,),
            "time": (batch,),
            "initial_state": (batch, STATE_SIZE),
            "prescribed_current": (batch, 101),
            "target_current": (batch,),
            "requested_class": (batch,),
            "centroid_target_z": (batch,),
        }
        if shapes != expected:
            raise ValueError(f"engine input contract mismatch: {shapes!r}")
        return batch


@dataclass(frozen=True)
class EngineResult:
    """One device-to-host engine result for every slot."""

    state: np.ndarray
    converged: np.ndarray
    termination: np.ndarray
    trips: np.ndarray
    terminal_residual: np.ndarray
    centroid: np.ndarray
    conditioned: np.ndarray
    labelled_fields: dict[str, np.ndarray] = field(default_factory=dict)

    def validate(self, batch: int) -> None:
        """Validate the result shape promised by the batched engine node."""
        shapes = {
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
        if shapes != expected:
            raise ValueError(f"engine output contract mismatch: {shapes!r}")
        for name, values in self.labelled_fields.items():
            if values.shape[:1] != (batch,):
                raise ValueError(
                    f"labelled field {name!r} has no batch leading axis: "
                    f"{values.shape!r}"
                )


class BatchEngine(Protocol):
    """Protocol implemented by the stub and production device engine."""

    def step(self, batch: EngineBatch) -> EngineResult: ...


@dataclass
class ArrayBatchEngine:
    """Deterministic array-contract stub used until the device engine lands."""

    device_count: int

    def step(self, batch: EngineBatch) -> EngineResult:
        """Advance active states deterministically without changing topology."""
        size = batch.validate()
        state = np.array(batch.initial_state, copy=True)
        active = np.asarray(batch.active, dtype=bool)
        # The update depends only on the slice inputs and is stable across batch
        # packing, device count, scheduling order and process-pool completion.
        current_signal = np.sum(batch.prescribed_current, axis=1) * 1e-12
        row_signal = batch.row.astype(np.float64) * 1e-9
        state[active, 0] += current_signal[active] + row_signal[active]
        state[active, 1] = batch.target_current[active] * 1e-9
        centroid = np.column_stack(
            (
                0.8 + state[:, 0] * 1e-3,
                batch.centroid_target_z + state[:, 1] * 1e-4,
            )
        )
        return EngineResult(
            state=state,
            converged=active,
            termination=np.where(active, 0, -1).astype(np.int32),
            trips=np.where(active, 1, 0).astype(np.int32),
            terminal_residual=np.where(active, 0.0, np.nan),
            centroid=centroid,
            conditioned=np.zeros(size, dtype=bool),
            labelled_fields={
                "state_head": state[:, :4].copy(),
                "requested_class": batch.requested_class.copy(),
            },
        )


@dataclass(frozen=True)
class AssemblyRequest:
    """Pickle-safe payload sent from the scheduler to a host process."""

    shot: int
    row: int
    time: float
    state: np.ndarray
    converged: bool
    termination: int
    trips: int
    terminal_residual: float
    centroid: np.ndarray
    conditioned: bool
    labelled_fields: dict[str, np.ndarray]


@dataclass(frozen=True)
class AssembledSlice:
    """Host-produced frame channels and the sequential manifest row."""

    shot: int
    row: int
    time: float
    frame: SteeringFrame
    record: dict[str, Any]
    assembly_wall_seconds: float


def assemble_stub_frame(request: AssemblyRequest) -> AssembledSlice:
    """Assemble the deterministic smoke frame in a host worker process."""
    started = time.perf_counter()
    radius = np.linspace(0.6, 1.5, 4, dtype=np.float64)
    height = np.linspace(-0.4, 0.4, 3, dtype=np.float64)
    psi = np.asarray(request.state[:12], dtype=np.float64).reshape(4, 3)
    psi_span = float(np.ptp(psi))
    psi_norm = np.zeros_like(psi) if psi_span == 0.0 else (psi - psi.min()) / psi_span
    angle = np.linspace(0.0, 2.0 * np.pi, N_THETA, endpoint=False)
    levels = np.linspace(0.0, 1.0, N_SURFACE)
    surface_r = 0.9 + 0.3 * levels[:, None] * np.cos(angle)[None, :]
    surface_z = request.centroid[1] + 0.35 * levels[:, None] * np.sin(angle)[None, :]
    boundary = np.column_stack((surface_r[-1], surface_z[-1]))
    faces = np.linspace(0.0, 1.0, N_RHO + 1)
    profiles = {
        name: 0.1 + faces
        for name in TORAX_PROFILE_FIELDS
        if name not in {"rho_face_norm", "psi_norm_face"}
    }
    profiles["psi_norm_face"] = faces
    frame = SteeringFrame(
        radius=radius,
        height=height,
        shape=np.asarray([radius.size, height.size], dtype=np.int32),
        psi=psi,
        psi_norm=psi_norm,
        domain_label=np.zeros_like(psi, dtype=np.int8),
        separatrix=boundary,
        separatrix_vertex_count=np.int32(boundary.shape[0]),
        magnetic_axis_r=float(request.centroid[0]),
        magnetic_axis_z=float(request.centroid[1]),
        x_point_r=np.full(2, np.nan),
        x_point_z=np.full(2, np.nan),
        strike_points_r=np.full(2, np.nan),
        strike_points_z=np.full(2, np.nan),
        lcfs_r=boundary[:, 0],
        lcfs_z=boundary[:, 1],
        n_boundary_coords=np.int32(boundary.shape[0]),
        finite_mask=np.asarray([True, False, False, False, False, True]),
        coil_current=np.zeros(101, dtype=np.float64),
        compensating_current=np.zeros(1, dtype=np.float64),
        action=SteeringAction(
            name="label",
            delta=0.0,
            commanded_control_points=np.empty((0, 2), dtype=np.float64),
        ),
        wall_seconds=0.0,
        trip_count=request.trips,
        carrier_identity="array-contract-stub",
        nova_version="stub",
        policy_digest="0" * 64,
        p_prime_source="efm",
        flux_surface_psi_norm=levels,
        flux_surface_psi=levels,
        flux_surface_r=surface_r,
        flux_surface_z=surface_z,
        flux_surface_angle=angle,
        rho_face_norm=faces,
        p_prime_face=np.zeros_like(faces),
        ff_prime_face=np.zeros_like(faces),
        current_centroid_r=float(request.centroid[0]),
        current_centroid_z=float(request.centroid[1]),
        reference_centroid_z=float(request.centroid[1]),
        branch_guard_ok=True,
        R_major=0.9,
        a_minor=0.3,
        B_0=1.0,
        boundary_toroidal_flux=1.0,
        magnetic_axis_z_scalar=float(request.centroid[1]),
        diverted=False,
        divertor_leg_r=np.full((N_DIVERTOR_LEGS, N_DIVERTOR_LEG_POINTS), np.nan),
        divertor_leg_z=np.full((N_DIVERTOR_LEGS, N_DIVERTOR_LEG_POINTS), np.nan),
        divertor_leg_finite=np.zeros(N_DIVERTOR_LEGS, dtype=bool),
        **profiles,
    )
    record = {
        "row": request.row,
        "time": request.time,
        "written": True,
        "excluded": False,
        "converged": request.converged,
        "qualified": request.converged and request.terminal_residual <= 1e-10,
        "terminal_residual": request.terminal_residual,
        "trips": request.trips,
        "termination": request.termination,
        "conditioned": request.conditioned,
        "conditioning_flag": request.conditioned,
        "conditioning_target_source": (
            "efm/current_centrd_z" if request.conditioned else None
        ),
        "free_branch_guard_ok": True,
        "conditioned_branch_guard_ok": True if request.conditioned else None,
        "free_centroid_error_m": 0.0,
        "conditioned_centroid_error_m": 0.0 if request.conditioned else None,
        "achieved_current_centroid_r": float(request.centroid[0]),
        "achieved_current_centroid_z": float(request.centroid[1]),
    }
    return AssembledSlice(
        shot=request.shot,
        row=request.row,
        time=request.time,
        frame=frame,
        record=record,
        assembly_wall_seconds=time.perf_counter() - started,
    )


@dataclass
class _Slot:
    shot: int
    slices: Sequence[SliceInput]
    index: int = 0
    warm_state: np.ndarray | None = None

    @property
    def done(self) -> bool:
        return self.index >= len(self.slices)

    def current(self) -> SliceInput:
        item = self.slices[self.index]
        if self.warm_state is None:
            return item
        return SliceInput(
            shot=item.shot,
            row=item.row,
            time=item.time,
            initial_state=self.warm_state,
            prescribed_current=item.prescribed_current,
            target_current=item.target_current,
            requested_class=item.requested_class,
            centroid_target_z=item.centroid_target_z,
        )


def _json_default(value: Any) -> int | float | bool:
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"cannot encode {type(value).__name__}")


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_shot(
    output_root: Path,
    shot: int,
    slices: Sequence[AssembledSlice],
    *,
    run_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Write one shot through the same session and diagnostic helpers as shards."""
    output_root.mkdir(parents=True, exist_ok=True)
    ordered = sorted(slices, key=lambda item: item.row)
    session_path = output_root / f"{shot}.nc"
    diagnostics_path = output_root / f"{shot}.npz"
    manifest_path = output_root / f"{shot}.manifest.json"
    if session_path.is_file() and manifest_path.is_file():
        return {"shot": shot, "status": "skipped", "resumed": True}
    if not ordered:
        raise ValueError(f"shot {shot} has no admitted slices")
    _write_session_file(
        [item.frame for item in ordered],
        session_path.resolve(),
        time_values=[item.time for item in ordered],
        include_raster=False,
    )
    rows = [item.record for item in ordered]
    _write_diagnostics(rows, diagnostics_path)
    manifest = {
        "schema": "nova-forward-labeller-shot",
        "shot": shot,
        "status": "complete",
        "session": str(session_path.resolve()),
        "companion": str(diagnostics_path.resolve()),
        "slice_count": len(rows),
        "admitted_slice_count": len(rows),
        "written_slice_count": len(rows),
        "converged_slice_count": sum(bool(row["converged"]) for row in rows),
        "unconverged_slice_count": sum(not bool(row["converged"]) for row in rows),
        "excluded_slice_count": 0,
        "slices": rows,
        **run_metadata,
    }
    _write_json(manifest, manifest_path)
    return manifest


class CorpusScheduler:
    """Keep all device slots full while assembling completed slices on hosts."""

    def __init__(
        self,
        *,
        engine: BatchEngine,
        device_count: int,
        batch_per_device: int,
        host_workers: int,
        assembler: Callable[[AssemblyRequest], AssembledSlice] = assemble_stub_frame,
    ) -> None:
        if device_count < 1 or batch_per_device < 1 or host_workers < 1:
            raise ValueError(
                "device_count, batch_per_device and host_workers must be positive"
            )
        self.engine = engine
        self.device_count = device_count
        self.batch_per_device = batch_per_device
        self.host_workers = host_workers
        self.assembler = assembler

    @property
    def capacity(self) -> int:
        return self.device_count * self.batch_per_device

    def _pack(self, slots: Sequence[_Slot | None]) -> EngineBatch:
        active = np.asarray([slot is not None for slot in slots], dtype=bool)
        current_size = next(
            (
                slot.current().prescribed_current.size
                for slot in slots
                if slot is not None
            ),
            101,
        )
        if current_size != 101:
            raise ValueError(
                f"prescribed-current width is {current_size}, expected 101"
            )
        items = [slot.current() if slot is not None else None for slot in slots]
        return EngineBatch(
            active=active,
            shot=np.asarray(
                [item.shot if item else -1 for item in items], dtype=np.int64
            ),
            row=np.asarray(
                [item.row if item else -1 for item in items], dtype=np.int32
            ),
            time=np.asarray([item.time if item else np.nan for item in items]),
            initial_state=np.stack(
                [
                    np.asarray(item.initial_state, dtype=np.float64)
                    if item
                    else np.zeros(STATE_SIZE, dtype=np.float64)
                    for item in items
                ]
            ),
            prescribed_current=np.stack(
                [
                    np.asarray(item.prescribed_current, dtype=np.float64)
                    if item
                    else np.zeros(101, dtype=np.float64)
                    for item in items
                ]
            ),
            target_current=np.asarray(
                [item.target_current if item else 0.0 for item in items],
                dtype=np.float64,
            ),
            requested_class=np.asarray(
                [item.requested_class if item else 0 for item in items], dtype=np.int8
            ),
            centroid_target_z=np.asarray(
                [item.centroid_target_z if item else 0.0 for item in items],
                dtype=np.float64,
            ),
        )

    def run(
        self,
        ranked_shots: Sequence[tuple[int, Sequence[SliceInput]]],
        output_root: Path,
        *,
        run_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run the ranked corpus and return measured engine and host rates."""
        output_root.mkdir(parents=True, exist_ok=True)
        metadata = dict(run_metadata or {})
        pending = deque(
            (shot, slices)
            for shot, slices in ranked_shots
            if not (
                (output_root / f"{shot}.nc").is_file()
                and (output_root / f"{shot}.manifest.json").is_file()
            )
        )
        skipped = len(ranked_shots) - len(pending)
        slots: list[_Slot | None] = [None] * self.capacity
        completed: dict[int, list[AssembledSlice]] = {}
        expected: dict[int, int] = {}
        futures: dict[Future[AssembledSlice], int] = {}
        engine_wall = 0.0
        engine_slices = 0
        started = time.perf_counter()

        def refill(index: int) -> None:
            if pending:
                shot, slices = pending.popleft()
                if not slices:
                    raise ValueError(f"shot {shot} has no admitted slices")
                slots[index] = _Slot(shot=shot, slices=slices)
                expected[shot] = len(slices)
                completed[shot] = []
            else:
                slots[index] = None

        for index in range(self.capacity):
            refill(index)

        context = __import__("multiprocessing").get_context("fork")
        with ProcessPoolExecutor(
            max_workers=self.host_workers,
            mp_context=context,
        ) as pool:
            while any(slot is not None for slot in slots):
                batch = self._pack(slots)
                step_started = time.perf_counter()
                result = self.engine.step(batch)
                engine_wall += time.perf_counter() - step_started
                result.validate(self.capacity)
                for index, slot in enumerate(slots):
                    if slot is None:
                        continue
                    item = slot.current()
                    fields = {
                        name: np.asarray(value[index])
                        for name, value in result.labelled_fields.items()
                    }
                    future = pool.submit(
                        self.assembler,
                        AssemblyRequest(
                            shot=item.shot,
                            row=item.row,
                            time=item.time,
                            state=np.asarray(result.state[index]),
                            converged=bool(result.converged[index]),
                            termination=int(result.termination[index]),
                            trips=int(result.trips[index]),
                            terminal_residual=float(result.terminal_residual[index]),
                            centroid=np.asarray(result.centroid[index]),
                            conditioned=bool(result.conditioned[index]),
                            labelled_fields=fields,
                        ),
                    )
                    futures[future] = item.shot
                    engine_slices += 1
                    slot.warm_state = np.asarray(result.state[index])
                    slot.index += 1
                    if slot.done:
                        refill(index)

            for future in as_completed(futures):
                assembled = future.result()
                completed[assembled.shot].append(assembled)

        written = 0
        assembly_wall = 0.0
        for shot, slices in completed.items():
            if len(slices) != expected[shot]:
                raise RuntimeError(
                    f"shot {shot} assembled {len(slices)} of {expected[shot]} slices"
                )
            assembly_wall += sum(item.assembly_wall_seconds for item in slices)
            write_shot(output_root, shot, slices, run_metadata=metadata)
            written += len(slices)
        wall = time.perf_counter() - started
        engine_rate = engine_slices / engine_wall if engine_wall else 0.0
        pool_rate = written / assembly_wall if assembly_wall else 0.0
        frames_rate = written / wall if wall else 0.0
        return {
            "schema": "nova-forward-labeller-parallel-receipt",
            "device_count": self.device_count,
            "batch_per_device": self.batch_per_device,
            "slot_count": self.capacity,
            "host_workers": self.host_workers,
            "shot_count": len(completed),
            "skipped_shot_count": skipped,
            "slice_count": engine_slices,
            "engine_wall_seconds": engine_wall,
            "engine_slices_per_second": engine_rate,
            "host_assembly_wall_seconds": assembly_wall,
            "host_assembly_wall_seconds_per_slice": assembly_wall / written,
            "pool_slices_per_second": pool_rate,
            "frames_per_second_written": frames_rate,
            "wall_seconds": wall,
            "pool_below_engine": pool_rate < engine_rate,
        }
