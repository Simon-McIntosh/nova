#!/usr/bin/env python3
"""Run one resumable scheduler job over the ranked decoder corpus."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
import time
import traceback
from typing import ClassVar, Iterator, Sequence

import jax
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.labeller_batch.shard import (  # noqa: E402
    DEFAULT_COHORT_REPORT,
    DEFAULT_MANIFEST,
    PreparedLabeller,
    ShotWork,
    _write_json,
    decoder_corpus,
    prepare_labeller,
)
from nova.equilibrium import reduced_newton  # noqa: E402
from nova.equilibrium.batched_labeller import BatchedLabeller  # noqa: E402
from scripts.labeller_parallel.scheduler import (  # noqa: E402
    CorpusScheduler,
    EngineBatch,
    EngineResult,
    FIXED_POINT_CRITERION,
    HostRouteEngine,
    NEWTON_STEPS,
    SequentialCompiledEngine,
    ShotInput,
    SolvedSlice,
    SourceIdentity,
    load_shot,
)


@dataclass
class BatchedRouteEngine:
    """Adapt the device-batched solver to the production array boundary."""

    name: ClassVar[str] = "batched"
    prepared: PreparedLabeller
    device_count: int
    condition_on_guard_failure: bool

    def __post_init__(self) -> None:
        if self.device_count < 1:
            raise ValueError("device_count must be positive")
        available = len(jax.devices())
        if available != self.device_count:
            raise ValueError(
                f"batched engine requested {self.device_count} devices, "
                f"JAX exposes {available}"
            )
        self.labeller = BatchedLabeller(
            self.prepared.profile,
            tolerance=FIXED_POINT_CRITERION,
            newton_steps=NEWTON_STEPS,
            condition_on_guard_failure=self.condition_on_guard_failure,
        )

    def step(self, batch: EngineBatch) -> EngineResult:
        """Solve all resident rows once and retain scheduler receipt semantics."""
        size = batch.validate()
        reference = np.stack(
            (
                np.full(size, np.nan),
                np.asarray(batch.centroid_target_z),
            ),
            axis=1,
        )
        started = time.perf_counter()
        result = self.labeller.solve(
            batch.initial_state,
            prescribed_current=batch.prescribed_current,
            target_current=batch.target_current,
            requested_class=batch.requested_class,
            reference_centroid=reference,
            centroid_target=np.asarray(batch.centroid_target_z)[:, None],
            active=batch.active,
        )
        wall_per_active = (time.perf_counter() - started) / max(
            1, int(np.count_nonzero(batch.active))
        )
        solved: list[SolvedSlice | None] = []
        for index, active in enumerate(np.asarray(batch.active, dtype=bool)):
            if not active:
                solved.append(None)
                continue
            conditioned = bool(result.conditioned[index])
            trips = int(result.trips[index])
            free_trips = int(result.free_trips[index])
            per_trip_steps = [
                int(value)
                for value in np.asarray(result.newton_steps_per_trip[index])[:trips]
            ]
            applied_current = np.asarray(result.applied_current[index])
            result_type = (
                reduced_newton.ConstrainedReducedNewtonResult
                if conditioned
                else reduced_newton.ReducedNewtonResult
            )
            result_arguments = {
                "state": np.asarray(result.state[index]),
                "terminal_residual": float(result.terminal_residual[index]),
                "active_set_iterations": trips,
                "converged": bool(result.converged[index]),
                "termination_reason": int(result.termination[index]),
                "newton_steps_per_trip": per_trip_steps,
            }
            if conditioned:
                result_arguments["prescribed_current"] = applied_current
                result_arguments["row_count"] = 1
            selected = result_type(**result_arguments)
            free_centroid = np.asarray(result.free_centroid[index])
            selected_centroid = np.asarray(result.achieved_centroid[index])
            target_centroid_z = float(batch.centroid_target_z[index])
            free_error = float(free_centroid[1] - target_centroid_z)
            conditioned_error = (
                float(selected_centroid[1] - target_centroid_z) if conditioned else None
            )
            conditioned_guard = (
                bool(
                    np.isfinite(conditioned_error)
                    and abs(conditioned_error) <= self.labeller.guard_tolerance
                )
                if conditioned
                else None
            )
            record = {
                "row": int(batch.row[index]),
                "time": float(batch.time[index]),
                "written": True,
                "excluded": False,
                "geometry_masked": True,
                "converged": False,
                "qualified": False,
                "terminal_residual": float(result.terminal_residual[index]),
                "trips": trips,
                "newton_steps": sum(per_trip_steps),
                "free_trips": free_trips,
                "conditioned_trips": trips if conditioned else 0,
                "wall_seconds": wall_per_active,
                "free_wall_seconds": wall_per_active if not conditioned else 0.0,
                "conditioned_wall_seconds": wall_per_active if conditioned else 0.0,
                "termination": selected.termination_name,
                "conditioned": conditioned,
                "conditioning_flag": conditioned,
                "conditioning_target_source": (
                    "efm/current_centrd_z" if conditioned else None
                ),
                "free_converged": bool(result.free_converged[index]),
                "conditioned_converged": (
                    bool(result.converged[index]) if conditioned else None
                ),
                "free_branch_guard_ok": bool(result.guard[index]),
                "conditioned_branch_guard_ok": conditioned_guard,
                "free_centroid_error_m": free_error,
                "conditioned_centroid_error_m": conditioned_error,
                "achieved_current_centroid_r": float(selected_centroid[0]),
                "achieved_current_centroid_z": float(selected_centroid[1]),
                "target_current_centroid_z": target_centroid_z,
                "centroid_error_m": (conditioned_error if conditioned else free_error),
                "target_source": "efm/current_centrd_z",
                "branch_guard_ok": False,
                "requested_class": int(batch.requested_class[index]),
            }
            solved.append(SolvedSlice(selected, applied_current, record))
        labelled = {
            name: np.asarray(getattr(result.labelled_flux, name))
            for name in result.labelled_flux._fields
        }
        return EngineResult(
            state=np.asarray(result.state),
            converged=np.asarray(result.converged),
            termination=np.asarray(result.termination),
            trips=np.asarray(result.trips),
            terminal_residual=np.asarray(result.terminal_residual),
            centroid=np.asarray(result.achieved_centroid),
            conditioned=np.asarray(result.conditioned),
            labelled_fields=labelled,
            solved=tuple(solved),
        )


def _is_written(output_root: Path, shot: int) -> bool:
    """Return whether the per-shot session and manifest already exist."""
    return (output_root / f"{shot}.nc").is_file() and (
        output_root / f"{shot}.manifest.json"
    ).is_file()


def _default_host_workers() -> int:
    """Return the default host-assembly worker pool size.

    The pool defaults to the allocated cores minus one: SLURM_CPUS_PER_TASK
    minus one when the scheduler sets it, the logical CPU count minus one
    otherwise, floored at one worker.
    """
    if "SLURM_CPUS_PER_TASK" in os.environ:
        return max(1, int(os.environ["SLURM_CPUS_PER_TASK"]) - 1)
    return max(1, int(os.cpu_count() or 1) - 1)


def resolve_host_workers(requested: int | None) -> int:
    """Return the host-assembly worker pool size for a run.

    A positive explicit request wins; otherwise the pool defaults to the
    machine's allocated cores minus one.
    """
    if requested is not None:
        return requested
    return _default_host_workers()


def _load_ranked_shots(
    work: Sequence[ShotWork],
    output_root: Path,
    source_identity: SourceIdentity,
    failures: list[int],
) -> Iterator[ShotInput]:
    """Load one ranked shot at refill time so the corpus is never resident."""
    for item in work:
        try:
            yield load_shot(item.shot, max_slices=None)
        except Exception as error:
            failure = {
                "schema": "nova-forward-labeller-shot",
                "shot": item.shot,
                "status": "failed",
                "nova_revision": source_identity.nova_revision,
                "nova_equilibrium_tree": source_identity.nova_equilibrium_tree,
                "labeller_batch_tree": source_identity.labeller_batch_tree,
                "declared_additions": [
                    "manifest.nova_equilibrium_tree",
                    "manifest.labeller_batch_tree",
                    "manifest.declared_additions",
                    "manifest.slices.requested_class",
                ],
                "failure": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
                "slices": [],
            }
            _write_json(failure, output_root / f"{item.shot}.manifest.json")
            failures.append(item.shot)
            print(json.dumps(failure, sort_keys=True), flush=True)


def run_corpus(arguments: argparse.Namespace) -> dict[str, object]:
    """Prepare once, walk the ranked shots lazily, and persist a final receipt."""
    output_root = arguments.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    source_identity = SourceIdentity.capture()
    corpus = decoder_corpus(arguments.manifest, arguments.cohort_report)
    if arguments.max_shots is not None:
        corpus = corpus[: arguments.max_shots]
    pending = [item for item in corpus if not _is_written(output_root, item.shot)]
    previously_written = len(corpus) - len(pending)
    prepared = prepare_labeller()
    engine_type = {
        "host": HostRouteEngine,
        "compiled": SequentialCompiledEngine,
        "batched": BatchedRouteEngine,
    }[arguments.engine]
    engine = engine_type(
        prepared,
        device_count=arguments.devices,
        condition_on_guard_failure=arguments.condition_on_guard_failure,
    )
    scheduler = CorpusScheduler(
        engine=engine,
        device_count=arguments.devices,
        batch_per_device=arguments.batch_per_device,
        host_workers=arguments.host_workers,
        include_raster=arguments.include_raster,
        condition_on_guard_failure=arguments.condition_on_guard_failure,
    )
    failures: list[int] = []
    receipt = scheduler.run(
        _load_ranked_shots(pending, output_root, source_identity, failures),
        output_root,
        prepared=prepared,
        source_identity=source_identity,
        previously_written_shots=previously_written,
    )
    receipt.update(
        corpus_shot_count=len(corpus),
        failed_shot_count=len(failures),
        failed_shots=failures,
        corpus_source=str(arguments.manifest),
        cohort_source=str(arguments.cohort_report),
        source_identity={
            "nova_revision": source_identity.nova_revision,
            "nova_equilibrium_tree": source_identity.nova_equilibrium_tree,
            "labeller_batch_tree": source_identity.labeller_batch_tree,
        },
        condition_on_guard_failure=arguments.condition_on_guard_failure,
        include_raster=arguments.include_raster,
    )
    _write_json(receipt, output_root / "parallel-receipt.json")
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cohort-report", type=Path, default=DEFAULT_COHORT_REPORT)
    parser.add_argument(
        "--engine", choices=("host", "compiled", "batched"), default="host"
    )
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--batch-per-device", type=int, default=1)
    parser.add_argument("--host-workers", type=int)
    parser.add_argument("--max-shots", type=int)
    parser.add_argument("--include-raster", action="store_true")
    parser.add_argument("--condition-on-guard-failure", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    host_workers = resolve_host_workers(arguments.host_workers)
    arguments.host_workers = host_workers
    if min(arguments.devices, arguments.batch_per_device, host_workers) < 1:
        raise ValueError("device, batch and host worker counts must be positive")
    if arguments.max_shots is not None and arguments.max_shots < 1:
        raise ValueError("--max-shots must be positive")
    print(f"host_workers={host_workers}", flush=True)
    receipt = run_corpus(arguments)
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return 1 if receipt["failed_shot_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
