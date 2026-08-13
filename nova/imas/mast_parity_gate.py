"""Bank per-slice parity verdicts for the frozen MAST benchmark cohort.

The reconstruction and EFIT reads remain separated by
``run_refereed_parity_chain``.  This module only consumes its completed result,
expands the shot-level scorecard into the eleven registered metrics for every
aligned slice, and persists both machine-readable evidence and compact figures.
An unavailable reference row is a recorded skip; a non-finite metric on an
otherwise aligned row is a scored failure rather than a silent omission.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass, field as dataclass_field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from nova.equilibrium.measurement import SliceMeasurement
from nova.equilibrium.moment import UnsupportedSlice
from nova.imas.mast_chain_factory import build_mast_parity_chain
from nova.imas.mast_efit_referee import (
    FROZEN_SHOTS,
    RefereedParityResult,
    compare_reference_geometry,
    read_efit_referee,
    score_with_efit_referee,
)
from nova.imas.mast_parity_chain import (
    AcceleratorSettings,
    GeometryScores,
    ParityChainResult,
    PhysicsScores,
    SliceScorecard,
    SolveHealthScores,
    TemporalScores,
    _accelerated_profile_solve,
    _lcfs_distance,
    _pack_source_currents,
    _raw_magnetics_residuals,
    _registered_scorecard_metrics,
    _sensor_scales,
)
from nova.imas.mast_solve_inputs import (
    SHOT_STORE,
    CorrectedSolveInputs,
    read_corrected_solve_inputs,
)
from nova.imas.parity_tolerances import (
    SCORECARD_FIELDS,
    MagneticsBudgetClass,
    ScorecardField,
    registered_tolerances,
    scorecard_verdicts,
)

DEFAULT_ARTIFACT = Path("docs/figures/spine-efit-parity/frozen-six-scorecard.json")
DEFAULT_FIGURE_DIR = DEFAULT_ARTIFACT.parent


@dataclass(frozen=True)
class ScoredSlice:
    """One reference-aligned slice carrying every registered metric verdict."""

    shot: int
    slice_index: int
    time_s: float
    reference_time_s: float
    metrics: Mapping[str, float]
    verdicts: Mapping[str, bool]


@dataclass(frozen=True)
class SkippedSlice:
    """One reconstruction slice excluded for an explicit reference cause."""

    shot: int
    slice_index: int
    time_s: float
    cause: str
    details: Mapping[str, Any] = dataclass_field(default_factory=dict)


@dataclass(frozen=True)
class ProductionShotScore:
    """Supported reconstruction rows plus corrected-input-only seed skips."""

    shot: int
    available_slices: int
    source_slice_indices: tuple[int, ...]
    result: RefereedParityResult | None
    skipped_slices: tuple[SkippedSlice, ...]
    magnetics_budget: MagneticsBudgetClass
    min_cells: int


@dataclass(frozen=True)
class ShotSummary:
    """Coverage and per-metric pass fractions for one requested shot."""

    shot: int
    available_slices: int
    scored_slices: int
    skipped_slices: int
    skip_causes: Mapping[str, int]
    pass_fraction_by_metric: Mapping[str, float]


@dataclass(frozen=True)
class PartitionCoverage:
    """One disjoint source-slice interval contributing to a complete report."""

    slice_start: int
    slice_stop: int
    scored_slices: int
    skipped_slices: int
    artifact: str


@dataclass(frozen=True)
class FrozenGateReport:
    """Banked result for the complete requested shot cohort."""

    generated_at: str
    requested_shots: tuple[int, ...]
    completed_shots: tuple[int, ...]
    incomplete_shots: tuple[int, ...]
    not_attempted_shots: tuple[int, ...]
    magnetics_budget: str
    status: str
    scored_slices: tuple[ScoredSlice, ...]
    skipped_slices: tuple[SkippedSlice, ...]
    shot_summaries: tuple[ShotSummary, ...]
    pass_fraction_by_metric: Mapping[str, float]
    run_errors: Mapping[int, str]
    figures: tuple[str, ...]
    partitions: tuple[PartitionCoverage, ...] = ()
    radial_points: int | None = None
    vertical_points: int | None = None
    min_cells: int | None = None


@dataclass(frozen=True)
class SlicePartitionReport:
    """Durable score rows for one disjoint interval of a production shot."""

    generated_at: str
    shot: int
    available_slices: int
    slice_start: int
    slice_stop: int
    radial_points: int
    vertical_points: int
    min_cells: int
    magnetics_budget: str
    scored_slices: tuple[ScoredSlice, ...]
    skipped_slices: tuple[SkippedSlice, ...]


def _supported_moment_seeds(
    shot: int,
    inputs: CorrectedSolveInputs,
    moment_solver: Any,
    profile_solver: Any,
    *,
    source_slice_offset: int = 0,
) -> tuple[tuple[int, ...], tuple[Any, ...], tuple[SkippedSlice, ...], np.ndarray]:
    """Attempt every corrected slice and retain only supported moment seeds."""

    source_current = _pack_source_currents(profile_solver, inputs)
    scale = _sensor_scales(inputs.sensor_signals, None)
    vacuum_sensor = source_current @ np.asarray(profile_solver.source_to_sensor).T
    vacuum_flux = source_current @ np.asarray(profile_solver.source_to_grid).T
    mask = np.isfinite(inputs.sensor_signals)
    indices: list[int] = []
    seeds: list[Any] = []
    skipped: list[SkippedSlice] = []
    for input_index, time_s in enumerate(np.asarray(inputs.time_s, dtype=float)):
        source_index = source_slice_offset + input_index
        measurement = SliceMeasurement(
            measured=inputs.sensor_signals[input_index],
            vacuum=vacuum_sensor[input_index],
            mask=mask[input_index],
            scale=scale,
            plasma_current=float(inputs.plasma_current_a[input_index]),
            vacuum_flux=vacuum_flux[input_index],
        )
        try:
            seed = moment_solver.solve(measurement)
        except UnsupportedSlice as error:
            skipped.append(
                SkippedSlice(
                    shot=int(shot),
                    slice_index=source_index,
                    time_s=float(time_s),
                    cause=error.condition,
                    details=error.details,
                )
            )
            continue
        indices.append(source_index)
        seeds.append(seed)
    return tuple(indices), tuple(seeds), tuple(skipped), scale


def _select_inputs(
    inputs: CorrectedSolveInputs, indices: tuple[int, ...]
) -> CorrectedSolveInputs:
    """Select corrected solve rows while preserving their channel metadata."""

    rows = np.asarray(indices, dtype=int)
    return replace(
        inputs,
        time_s=np.asarray(inputs.time_s)[rows],
        coil_currents_a=np.asarray(inputs.coil_currents_a)[rows],
        sensor_signals=np.asarray(inputs.sensor_signals)[rows],
        plasma_current_a=np.asarray(inputs.plasma_current_a)[rows],
    )


def _scorecard_convergence_fraction(trace: np.ndarray, final: np.ndarray) -> np.ndarray:
    """Reduce residual traces while retaining non-convergence as a failed score."""

    trace_array = np.asarray(trace, dtype=float)
    final_array = np.asarray(final, dtype=float)
    if trace_array.ndim != 2 or trace_array.shape[0] != final_array.shape[0]:
        raise ValueError("residual trace rows must match the final residual row count")
    finite = np.isfinite(trace_array)
    has_finite = np.any(finite, axis=1)
    first_index = np.argmax(finite, axis=1)
    first = np.full(final_array.shape, np.nan, dtype=float)
    rows = np.flatnonzero(has_finite)
    first[rows] = trace_array[rows, first_index[rows]]
    ratio = np.divide(
        final_array,
        first,
        out=np.ones_like(final_array, dtype=float),
        where=has_finite & np.isfinite(final_array) & (first > 0.0),
    )
    return np.clip(1.0 - ratio, 0.0, 1.0)


def _run_supported_chain(
    inputs: CorrectedSolveInputs,
    seeds: tuple[Any, ...],
    *,
    profile_solver: Any,
    topology_labeler: Any,
    temporal_scorer: Any,
    sensor_scale: np.ndarray,
    accelerator: AcceleratorSettings = AcceleratorSettings(),
    magnetics_budget: MagneticsBudgetClass = MagneticsBudgetClass.SOURCE_CUTOVER,
) -> ParityChainResult:
    """Run the unchanged batch solve for rows with valid moment seeds."""

    source_current = _pack_source_currents(profile_solver, inputs)
    mask = np.isfinite(inputs.sensor_signals)
    initial_flux = np.stack([np.asarray(seed.flux).reshape(-1) for seed in seeds])
    expected = (inputs.slice_count, np.asarray(profile_solver.source_to_grid).shape[0])
    if initial_flux.shape != expected:
        raise ValueError(
            f"moment seed flux shape {initial_flux.shape} must be {expected}"
        )
    solve = _accelerated_profile_solve(
        profile_solver,
        source_current,
        inputs,
        sensor_scale,
        mask,
        initial_flux,
        accelerator,
    )
    topology = topology_labeler(np.asarray(solve.flux))
    topology.validate(inputs.slice_count)
    raw_residual = _raw_magnetics_residuals(
        profile_solver,
        source_current,
        inputs,
        sensor_scale,
        mask,
        solve.flux,
    )
    temporal = np.asarray(temporal_scorer(inputs, solve.flux), dtype=float)
    if temporal.shape != (inputs.slice_count,):
        raise ValueError(
            f"temporal scorer shape {temporal.shape} must be ({inputs.slice_count},)"
        )
    core = np.asarray(topology.core_mask, dtype=bool)
    confinement = core.reshape(inputs.slice_count, -1).mean(axis=1)
    throughput = inputs.slice_count / max(solve.elapsed_s, np.finfo(float).tiny)
    scorecard = SliceScorecard(
        shot=int(inputs.shot),
        time_s=np.asarray(inputs.time_s),
        geometry=GeometryScores(
            magnetic_axis_m=np.asarray(topology.magnetic_axis_m),
            lcfs_m=np.asarray(topology.lcfs_m),
            x_point_m=np.asarray(topology.x_point_m),
            diverted=np.asarray(topology.diverted, dtype=bool),
            seed_to_solved_lcfs_distance_m=np.asarray(
                [
                    _lcfs_distance(seed, topology.lcfs_m[index])
                    for index, seed in enumerate(seeds)
                ]
            ),
        ),
        physics=PhysicsScores(
            profile_residual=np.asarray(solve.residual),
            whitened_raw_magnetics_residual=raw_residual,
        ),
        solve_health=SolveHealthScores(
            convergence_fraction=_scorecard_convergence_fraction(
                solve.trace, solve.residual
            ),
            confinement_fraction=confinement,
            iteration_count=np.full(inputs.slice_count, accelerator.evaluation_count),
            throughput_slices_per_second=np.full(inputs.slice_count, throughput),
        ),
        temporal=TemporalScores(current_diffusion_flux_ledger_consistency=temporal),
        registered_metrics=_registered_scorecard_metrics(
            solve, raw_residual, topology, temporal, accelerator
        ),
        magnetics_budget=magnetics_budget,
    )
    return ParityChainResult(inputs, seeds, solve, topology, scorecard)


def _score_completed_chain(
    chain: ParityChainResult, referee: Any
) -> RefereedParityResult:
    """Score reference rows or retain an explicit empty comparison for skipping."""

    geometry = compare_reference_geometry(
        chain.scorecard.time_s, chain.topology, referee
    )
    if geometry.usable_slice_count:
        return score_with_efit_referee(chain, referee)
    return RefereedParityResult(chain, referee, geometry)


def score_production_shot(
    shot: int,
    *,
    artifact_cache: Path | str,
    artifact_digest: str,
    store: Path | str = SHOT_STORE,
    radial_points: int = 33,
    vertical_points: int = 49,
    slice_start: int = 0,
    slice_stop: int | None = None,
) -> ProductionShotScore:
    """Score one corrected-input interval using only production components."""

    components = build_mast_parity_chain(
        int(shot),
        artifact_cache=artifact_cache,
        artifact_digest=artifact_digest,
        store=store,
        radial_points=radial_points,
        vertical_points=vertical_points,
    )
    inputs = read_corrected_solve_inputs(int(shot), store=store)
    stop = inputs.slice_count if slice_stop is None else int(slice_stop)
    start = int(slice_start)
    if start < 0 or stop <= start or stop > inputs.slice_count:
        raise ValueError(
            f"slice interval [{start}, {stop}) must lie within "
            f"[0, {inputs.slice_count})"
        )
    selected_inputs = _select_inputs(inputs, tuple(range(start, stop)))
    indices, seeds, skipped, scale = _supported_moment_seeds(
        int(shot),
        selected_inputs,
        components.moment_solver,
        components.profile_solver,
        source_slice_offset=start,
    )
    result = None
    if indices:
        supported_inputs = _select_inputs(
            selected_inputs, tuple(index - start for index in indices)
        )
        chain = _run_supported_chain(
            supported_inputs,
            seeds,
            profile_solver=components.profile_solver,
            topology_labeler=components.topology_labeler,
            temporal_scorer=components.temporal_scorer,
            sensor_scale=scale,
        )
        referee = read_efit_referee(int(shot), store=store)
        result = _score_completed_chain(chain, referee)
    return ProductionShotScore(
        shot=int(shot),
        available_slices=inputs.slice_count,
        source_slice_indices=indices,
        result=result,
        skipped_slices=skipped,
        magnetics_budget=MagneticsBudgetClass.SOURCE_CUTOVER,
        min_cells=int(components.moment_solver.config.min_cells),
    )


def _slice_metrics(result: RefereedParityResult, index: int) -> dict[str, float]:
    """Expand one aligned result row into the exact registered field schema."""

    scorecard = result.scorecard
    geometry = result.geometry_scores
    chain = result.chain
    residual = float(np.asarray(chain.solve.residual)[index])
    core = np.asarray(chain.topology.core_mask, dtype=bool)[index]
    metrics = {
        ScorecardField.MAGNETIC_AXIS_DISTANCE_M.value: float(
            geometry.magnetic_axis_distance_m[index]
        ),
        ScorecardField.LCFS_DISTANCE_M.value: float(geometry.lcfs_distance_m[index]),
        ScorecardField.X_POINT_DISTANCE_M.value: float(
            geometry.x_point_distance_m[index]
        ),
        ScorecardField.TOPOLOGY_CLASS_AGREEMENT_FRACTION.value: float(
            geometry.topology_class_agreement[index]
        ),
        ScorecardField.PROFILE_RESIDUAL_RMS.value: abs(residual),
        ScorecardField.MAGNETICS_RESIDUAL_WHITENED_RMS.value: float(
            scorecard.physics.whitened_raw_magnetics_residual[index]
        ),
        ScorecardField.CONVERGED_FRACTION.value: float(
            math.isfinite(residual) and residual < 1.0e-8
        ),
        ScorecardField.CONFINED_FRACTION.value: float(np.any(core)),
        ScorecardField.ITERATION_COUNT.value: float(
            scorecard.solve_health.iteration_count[index]
        ),
        ScorecardField.THROUGHPUT_SLICES_PER_CORE_S.value: float(
            scorecard.solve_health.throughput_slices_per_second[index]
        ),
        ScorecardField.CURRENT_DIFFUSION_FLUX_LEDGER_RMS_FRACTION.value: abs(
            float(scorecard.temporal.current_diffusion_flux_ledger_consistency[index])
        ),
    }
    if set(metrics) != SCORECARD_FIELDS:
        raise RuntimeError("per-slice parity metrics do not match the registry")
    return metrics


def _score_result(
    result: RefereedParityResult | None,
    *,
    shot: int | None = None,
    source_slice_indices: tuple[int, ...] | None = None,
    initial_skips: tuple[SkippedSlice, ...] = (),
    available_slices: int | None = None,
) -> tuple[list[ScoredSlice], list[SkippedSlice], ShotSummary]:
    """Score all aligned slices and retain every unaligned row with its cause."""

    if result is None:
        if shot is None or available_slices is None:
            raise ValueError("an empty shot result requires shot and coverage counts")
        pass_fractions = {field: float("nan") for field in sorted(SCORECARD_FIELDS)}
        causes = Counter(row.cause for row in initial_skips)
        summary = ShotSummary(
            shot=int(shot),
            available_slices=int(available_slices),
            scored_slices=0,
            skipped_slices=len(initial_skips),
            skip_causes=dict(sorted(causes.items())),
            pass_fraction_by_metric=pass_fractions,
        )
        return [], list(initial_skips), summary

    scorecard = result.scorecard
    geometry = result.geometry_scores
    scored: list[ScoredSlice] = []
    skipped = list(initial_skips)
    if source_slice_indices is None:
        source_slice_indices = tuple(range(scorecard.slice_count))
    if len(source_slice_indices) != scorecard.slice_count:
        raise ValueError("source slice indices do not match the scorecard rows")
    for index, (slice_index, time_s) in enumerate(
        zip(
            source_slice_indices,
            np.asarray(scorecard.time_s, dtype=float),
            strict=True,
        )
    ):
        if not bool(geometry.usable_reference[index]):
            reference_index = int(geometry.reference_index[index])
            cause = (
                "no-reference-within-time-tolerance"
                if reference_index < 0
                else "reference-geometry-unusable"
            )
            skipped.append(
                SkippedSlice(
                    shot=int(scorecard.shot),
                    slice_index=int(slice_index),
                    time_s=float(time_s),
                    cause=cause,
                )
            )
            continue

        metrics = _slice_metrics(result, index)
        verdicts = dict(scorecard_verdicts(metrics, scorecard.magnetics_budget))
        scored.append(
            ScoredSlice(
                shot=int(scorecard.shot),
                slice_index=int(slice_index),
                time_s=float(time_s),
                reference_time_s=float(geometry.reference_time_s[index]),
                metrics=metrics,
                verdicts=verdicts,
            )
        )

    pass_fractions = {
        field: (
            float(np.mean([row.verdicts[field] for row in scored]))
            if scored
            else float("nan")
        )
        for field in sorted(SCORECARD_FIELDS)
    }
    causes = Counter(row.cause for row in skipped)
    summary = ShotSummary(
        shot=int(scorecard.shot),
        available_slices=(
            int(scorecard.slice_count)
            if available_slices is None
            else int(available_slices)
        ),
        scored_slices=len(scored),
        skipped_slices=len(skipped),
        skip_causes=dict(sorted(causes.items())),
        pass_fraction_by_metric=pass_fractions,
    )
    return scored, skipped, summary


def _json_value(value: Any) -> Any:
    """Convert dataclass and NumPy values to strict JSON-compatible values."""

    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _bank_report(report: Any, path: Path) -> None:
    """Write the scorecard atomically as strict, deterministic JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_value(asdict(report)), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def bank_production_partition(
    shot: int,
    *,
    slice_start: int,
    slice_stop: int,
    artifact_path: Path | str,
    artifact_cache: Path | str,
    artifact_digest: str,
    store: Path | str = SHOT_STORE,
    radial_points: int = 33,
    vertical_points: int = 49,
) -> SlicePartitionReport:
    """Score and durably bank every row in one source-slice interval."""

    outcome = score_production_shot(
        int(shot),
        artifact_cache=artifact_cache,
        artifact_digest=artifact_digest,
        store=store,
        radial_points=radial_points,
        vertical_points=vertical_points,
        slice_start=slice_start,
        slice_stop=slice_stop,
    )
    scored, skipped, _summary = _score_result(
        outcome.result,
        shot=outcome.shot,
        source_slice_indices=outcome.source_slice_indices,
        initial_skips=outcome.skipped_slices,
        available_slices=outcome.available_slices,
    )
    expected_indices = set(range(int(slice_start), int(slice_stop)))
    observed_indices = {row.slice_index for row in scored}
    observed_indices.update(row.slice_index for row in skipped)
    if observed_indices != expected_indices or len(scored) + len(skipped) != len(
        expected_indices
    ):
        raise RuntimeError(
            "partition rows do not cover their source interval exactly once"
        )
    report = SlicePartitionReport(
        generated_at=datetime.now(UTC).isoformat(),
        shot=int(shot),
        available_slices=outcome.available_slices,
        slice_start=int(slice_start),
        slice_stop=int(slice_stop),
        radial_points=int(radial_points),
        vertical_points=int(vertical_points),
        min_cells=outcome.min_cells,
        magnetics_budget=str(outcome.magnetics_budget),
        scored_slices=tuple(scored),
        skipped_slices=tuple(skipped),
    )
    _bank_report(report, Path(artifact_path))
    return report


def _read_partition(path: Path) -> SlicePartitionReport:
    """Read a strict partition artifact into its typed representation."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    return SlicePartitionReport(
        generated_at=str(payload["generated_at"]),
        shot=int(payload["shot"]),
        available_slices=int(payload["available_slices"]),
        slice_start=int(payload["slice_start"]),
        slice_stop=int(payload["slice_stop"]),
        radial_points=int(payload["radial_points"]),
        vertical_points=int(payload["vertical_points"]),
        min_cells=int(payload["min_cells"]),
        magnetics_budget=str(payload["magnetics_budget"]),
        scored_slices=tuple(ScoredSlice(**row) for row in payload["scored_slices"]),
        skipped_slices=tuple(SkippedSlice(**row) for row in payload["skipped_slices"]),
    )


def aggregate_scorecard_partitions(
    partition_paths: tuple[Path | str, ...],
    *,
    artifact_path: Path | str,
    shot: int = 21978,
    radial_points: int = 33,
    vertical_points: int = 49,
    min_cells: int = 5,
) -> FrozenGateReport:
    """Atomically bank a shot only when partitions cover every source row once."""

    paths = tuple(Path(path) for path in partition_paths)
    if not paths:
        raise ValueError("at least one partition artifact is required")
    reports = sorted(
        ((path, _read_partition(path)) for path in paths),
        key=lambda item: item[1].slice_start,
    )
    available_values = {report.available_slices for _path, report in reports}
    if len(available_values) != 1:
        raise ValueError("partition artifacts disagree on available slice count")
    available = available_values.pop()
    expected_start = 0
    budgets: set[str] = set()
    scored: list[ScoredSlice] = []
    skipped: list[SkippedSlice] = []
    coverage: list[PartitionCoverage] = []
    for path, report in reports:
        observed_metadata = (
            report.shot,
            report.radial_points,
            report.vertical_points,
            report.min_cells,
        )
        expected_metadata = (int(shot), radial_points, vertical_points, min_cells)
        if observed_metadata != expected_metadata:
            raise ValueError(
                f"partition {path} metadata {observed_metadata} must be "
                f"{expected_metadata}"
            )
        if report.slice_start != expected_start:
            raise ValueError(
                f"partition coverage expected slice {expected_start}, "
                f"found {report.slice_start}"
            )
        if report.slice_stop <= report.slice_start:
            raise ValueError(f"partition {path} has an empty or reversed interval")
        rows = (*report.scored_slices, *report.skipped_slices)
        observed_indices = [row.slice_index for row in rows]
        expected_indices = list(range(report.slice_start, report.slice_stop))
        if sorted(observed_indices) != expected_indices or len(observed_indices) != len(
            set(observed_indices)
        ):
            raise ValueError(
                f"partition {path} does not cover its interval exactly once"
            )
        for row in report.scored_slices:
            if (
                set(row.metrics) != SCORECARD_FIELDS
                or set(row.verdicts) != SCORECARD_FIELDS
            ):
                raise ValueError(f"partition {path} carries an invalid scorecard row")
        budgets.add(report.magnetics_budget)
        scored.extend(report.scored_slices)
        skipped.extend(report.skipped_slices)
        coverage.append(
            PartitionCoverage(
                slice_start=report.slice_start,
                slice_stop=report.slice_stop,
                scored_slices=len(report.scored_slices),
                skipped_slices=len(report.skipped_slices),
                artifact=str(path.resolve()),
            )
        )
        expected_start = report.slice_stop
    if expected_start != available:
        raise ValueError(
            f"partition coverage stops at {expected_start}, expected {available}"
        )
    if len(budgets) != 1:
        raise ValueError(f"partitions carry mixed magnetics budgets: {budgets}")
    if len(scored) + len(skipped) != available:
        raise RuntimeError("aggregated rows do not equal the available slice count")
    pass_fractions = {
        field: (
            float(np.mean([row.verdicts[field] for row in scored]))
            if scored
            else float("nan")
        )
        for field in sorted(SCORECARD_FIELDS)
    }
    causes = Counter(row.cause for row in skipped)
    summary = ShotSummary(
        shot=int(shot),
        available_slices=available,
        scored_slices=len(scored),
        skipped_slices=len(skipped),
        skip_causes=dict(sorted(causes.items())),
        pass_fraction_by_metric=pass_fractions,
    )
    status = (
        "pass"
        if scored and all(all(row.verdicts.values()) for row in scored)
        else "fail"
    )
    report = FrozenGateReport(
        generated_at=datetime.now(UTC).isoformat(),
        requested_shots=(int(shot),),
        completed_shots=(int(shot),),
        incomplete_shots=(),
        not_attempted_shots=(),
        magnetics_budget=budgets.pop(),
        status=status,
        scored_slices=tuple(sorted(scored, key=lambda row: row.slice_index)),
        skipped_slices=tuple(sorted(skipped, key=lambda row: row.slice_index)),
        shot_summaries=(summary,),
        pass_fraction_by_metric=pass_fractions,
        run_errors={},
        figures=(),
        partitions=tuple(coverage),
        radial_points=int(radial_points),
        vertical_points=int(vertical_points),
        min_cells=int(min_cells),
    )
    _bank_report(report, Path(artifact_path))
    return report


def _normalised_metric(
    rows: list[ScoredSlice], field: str, budget: MagneticsBudgetClass
) -> np.ndarray:
    """Return metric values divided by their registered threshold."""

    tolerance = registered_tolerances(budget)[field]
    return np.asarray([row.metrics[field] / tolerance.bound for row in rows])


def _write_figures(
    results: Mapping[int, RefereedParityResult],
    rows: list[ScoredSlice],
    figure_dir: Path,
) -> tuple[str, ...]:
    """Plot spatial, distributional and residual relationships from the run."""

    if not rows:
        return ()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir.mkdir(parents=True, exist_ok=True)
    sources: list[str] = []

    completed = sorted(results)
    flux_figure, axes = plt.subplots(
        2, 3, figsize=(9.0, 5.0), constrained_layout=True, squeeze=False
    )
    for axis, shot in zip(axes.ravel(), completed, strict=False):
        flux = np.asarray(results[shot].chain.solve.flux)[0]
        if flux.ndim == 1:
            width = math.isqrt(flux.size)
            flux = (
                flux.reshape(width, width) if width * width == flux.size else flux[None]
            )
        image = axis.imshow(flux, origin="lower", aspect="auto", cmap="viridis")
        axis.set_title(str(shot))
        axis.set_xlabel("poloidal-grid index")
        axis.set_ylabel("poloidal-grid index")
        flux_figure.colorbar(image, ax=axis, shrink=0.72)
    for axis in axes.ravel()[len(completed) :]:
        axis.set_visible(False)
    flux_path = figure_dir / "flux-map-overlays.svg"
    flux_figure.suptitle("First scored reconstruction slice per shot")
    flux_figure.savefig(flux_path)
    plt.close(flux_figure)
    sources.append("/nova/figures/spine-efit-parity/flux-map-overlays.svg")

    boundary_fields = (
        ScorecardField.MAGNETIC_AXIS_DISTANCE_M.value,
        ScorecardField.LCFS_DISTANCE_M.value,
        ScorecardField.X_POINT_DISTANCE_M.value,
    )
    boundary_figure, boundary_axis = plt.subplots(figsize=(7.0, 3.8))
    boundary_axis.boxplot(
        [[row.metrics[field] * 1.0e3 for row in rows] for field in boundary_fields],
        tick_labels=("axis", "LCFS", "x-point"),
        showfliers=True,
    )
    boundary_axis.set_ylabel("distance (mm)")
    boundary_axis.set_title("Reference-boundary distance distributions")
    boundary_path = figure_dir / "boundary-distance-distributions.svg"
    boundary_figure.savefig(boundary_path, bbox_inches="tight")
    plt.close(boundary_figure)
    sources.append(
        "/nova/figures/spine-efit-parity/boundary-distance-distributions.svg"
    )

    residual_fields = (
        ScorecardField.PROFILE_RESIDUAL_RMS.value,
        ScorecardField.MAGNETICS_RESIDUAL_WHITENED_RMS.value,
        ScorecardField.CURRENT_DIFFUSION_FLUX_LEDGER_RMS_FRACTION.value,
    )
    residual_figure, residual_axis = plt.subplots(figsize=(8.0, 4.0))
    positions = np.arange(len(completed), dtype=float)
    width = 0.24
    budget = results[completed[0]].scorecard.magnetics_budget
    for offset, (field, label) in enumerate(
        zip(residual_fields, ("profile", "magnetics", "transport"), strict=True)
    ):
        values = _normalised_metric(rows, field, budget)
        by_shot = [
            float(np.nanmedian(values[[row.shot == shot for row in rows]]))
            for shot in completed
        ]
        residual_axis.bar(positions + (offset - 1) * width, by_shot, width, label=label)
    residual_axis.axhline(1.0, color="black", linewidth=1.0, linestyle="--")
    residual_axis.set_xticks(positions, [str(shot) for shot in completed])
    residual_axis.set_ylabel("median / registered tolerance")
    residual_axis.set_title("Residual decomposition by registered channel group")
    residual_axis.legend(frameon=False)
    residual_path = figure_dir / "residual-decomposition.svg"
    residual_figure.savefig(residual_path, bbox_inches="tight")
    plt.close(residual_figure)
    sources.append("/nova/figures/spine-efit-parity/residual-decomposition.svg")
    return tuple(sources)


def bank_frozen_scorecard(
    scorer: Callable[[int], RefereedParityResult | ProductionShotScore] | None = None,
    *,
    shots: tuple[int, ...] = FROZEN_SHOTS,
    artifact_path: Path | str = DEFAULT_ARTIFACT,
    figure_dir: Path | str = DEFAULT_FIGURE_DIR,
    artifact_cache: Path | str | None = None,
    artifact_digest: str | None = None,
    store: Path | str = SHOT_STORE,
    radial_points: int = 33,
    vertical_points: int = 49,
    max_shots: int | None = None,
) -> FrozenGateReport:
    """Score every requested shot, bank coverage, and name every failed run."""

    if scorer is None:
        if artifact_cache is None or artifact_digest is None:
            raise ValueError(
                "production scoring requires artifact_cache and artifact_digest"
            )

        def scorer(shot: int) -> RefereedParityResult:
            return score_production_shot(
                shot,
                artifact_cache=artifact_cache,
                artifact_digest=artifact_digest,
                store=store,
                radial_points=radial_points,
                vertical_points=vertical_points,
            )

    if max_shots is not None and max_shots < 0:
        raise ValueError("max_shots must be non-negative")

    scored: list[ScoredSlice] = []
    skipped: list[SkippedSlice] = []
    summaries: list[ShotSummary] = []
    results: dict[int, RefereedParityResult] = {}
    completed_set: set[int] = set()
    errors: dict[int, str] = {}
    budgets: set[str] = set()

    attempted: list[int] = []
    for shot in shots:
        if max_shots is not None and len(attempted) >= max_shots:
            break
        attempted.append(int(shot))
        try:
            outcome = scorer(int(shot))
            if isinstance(outcome, ProductionShotScore):
                if int(outcome.shot) != int(shot):
                    raise ValueError(
                        f"scorer returned shot {outcome.shot} for request {shot}"
                    )
                shot_scored, shot_skipped, summary = _score_result(
                    outcome.result,
                    shot=outcome.shot,
                    source_slice_indices=outcome.source_slice_indices,
                    initial_skips=outcome.skipped_slices,
                    available_slices=outcome.available_slices,
                )
                budget = outcome.magnetics_budget
                figure_result = outcome.result
            else:
                result = outcome
                if int(result.scorecard.shot) != int(shot):
                    raise ValueError(
                        "scorer returned shot "
                        f"{result.scorecard.shot} for request {shot}"
                    )
                shot_scored, shot_skipped, summary = _score_result(result)
                budget = result.scorecard.magnetics_budget
                figure_result = result
            if int(summary.shot) != int(shot):
                raise ValueError(
                    f"score summary returned shot {summary.shot} for request {shot}"
                )
        except Exception as error:  # continue so the bank names every uncovered shot
            errors[int(shot)] = f"{type(error).__name__}: {error}"
            continue
        completed_set.add(int(shot))
        if figure_result is not None:
            results[int(shot)] = figure_result
        budgets.add(str(budget))
        scored.extend(shot_scored)
        skipped.extend(shot_skipped)
        summaries.append(summary)

    if len(budgets) > 1:
        raise ValueError(f"shots were scored with mixed magnetics budgets: {budgets}")
    completed = tuple(shot for shot in shots if shot in completed_set)
    incomplete = tuple(shot for shot in shots if shot not in completed_set)
    not_attempted = tuple(shot for shot in shots if shot not in attempted)
    figures = _write_figures(results, scored, Path(figure_dir))
    if incomplete:
        status = "incomplete"
    elif all(all(row.verdicts.values()) for row in scored) and scored:
        status = "pass"
    else:
        status = "fail"
    pass_fractions = {
        field: (
            float(np.mean([row.verdicts[field] for row in scored]))
            if scored
            else float("nan")
        )
        for field in sorted(SCORECARD_FIELDS)
    }
    report = FrozenGateReport(
        generated_at=datetime.now(UTC).isoformat(),
        requested_shots=tuple(int(shot) for shot in shots),
        completed_shots=completed,
        incomplete_shots=incomplete,
        not_attempted_shots=not_attempted,
        magnetics_budget=next(iter(budgets), "not-scored"),
        status=status,
        scored_slices=tuple(scored),
        skipped_slices=tuple(skipped),
        shot_summaries=tuple(summaries),
        pass_fraction_by_metric=pass_fractions,
        run_errors=dict(sorted(errors.items())),
        figures=figures,
    )
    _bank_report(report, Path(artifact_path))
    return report


def print_frozen_gate_report(report: FrozenGateReport) -> None:
    """Print cohort coverage and every registered metric pass fraction."""

    print(f"status: {report.status}")
    print(f"completed_shots: {list(report.completed_shots)}")
    print(f"not_attempted_shots: {list(report.not_attempted_shots)}")
    for summary in report.shot_summaries:
        print(
            f"shot {summary.shot}: available={summary.available_slices} "
            f"scored={summary.scored_slices} skipped={summary.skipped_slices} "
            f"skip_causes={dict(summary.skip_causes)}"
        )
        for field, fraction in summary.pass_fraction_by_metric.items():
            print(f"  {field}: {fraction:.12g}")
    for shot, error in report.run_errors.items():
        print(f"shot {shot}: run_error={error}")
    print("overall_pass_fraction_by_metric:")
    for field, fraction in report.pass_fraction_by_metric.items():
        print(f"  {field}: {fraction:.12g}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-cache", type=Path)
    parser.add_argument("--artifact-digest")
    parser.add_argument("--store", type=Path, default=SHOT_STORE)
    parser.add_argument("--artifact-path", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--radial-points", type=int, default=33)
    parser.add_argument("--vertical-points", type=int, default=49)
    parser.add_argument("--max-shots", type=int)
    parser.add_argument("--shot", type=int)
    parser.add_argument("--slice-start", type=int)
    parser.add_argument("--slice-stop", type=int)
    parser.add_argument("--aggregate-partitions", nargs="+", type=Path)
    return parser


def main() -> None:
    """Run and bank the production frozen-cohort scorecard."""

    args = _parser().parse_args()
    if args.aggregate_partitions:
        report = aggregate_scorecard_partitions(
            tuple(args.aggregate_partitions),
            artifact_path=args.artifact_path,
            shot=21978 if args.shot is None else args.shot,
            radial_points=args.radial_points,
            vertical_points=args.vertical_points,
        )
        print_frozen_gate_report(report)
        return
    partition_requested = args.slice_start is not None or args.slice_stop is not None
    if partition_requested:
        if args.slice_start is None or args.slice_stop is None or args.shot is None:
            raise ValueError(
                "partition scoring requires --shot, --slice-start and --slice-stop"
            )
        if args.artifact_cache is None or args.artifact_digest is None:
            raise ValueError(
                "partition scoring requires --artifact-cache and --artifact-digest"
            )
        partition = bank_production_partition(
            args.shot,
            slice_start=args.slice_start,
            slice_stop=args.slice_stop,
            artifact_path=args.artifact_path,
            artifact_cache=args.artifact_cache,
            artifact_digest=args.artifact_digest,
            store=args.store,
            radial_points=args.radial_points,
            vertical_points=args.vertical_points,
        )
        print(
            f"shot {partition.shot} [{partition.slice_start}, "
            f"{partition.slice_stop}): scored={len(partition.scored_slices)} "
            f"skipped={len(partition.skipped_slices)} "
            "skip_causes="
            f"{dict(Counter(row.cause for row in partition.skipped_slices))}"
        )
        return
    if args.artifact_cache is None or args.artifact_digest is None:
        raise ValueError("production scoring requires artifact cache and digest")
    report = bank_frozen_scorecard(
        artifact_path=args.artifact_path,
        figure_dir=args.figure_dir,
        artifact_cache=args.artifact_cache,
        artifact_digest=args.artifact_digest,
        store=args.store,
        radial_points=args.radial_points,
        vertical_points=args.vertical_points,
        max_shots=args.max_shots,
    )
    print_frozen_gate_report(report)


__all__ = [
    "DEFAULT_ARTIFACT",
    "DEFAULT_FIGURE_DIR",
    "FrozenGateReport",
    "PartitionCoverage",
    "ProductionShotScore",
    "ScoredSlice",
    "SlicePartitionReport",
    "ShotSummary",
    "SkippedSlice",
    "aggregate_scorecard_partitions",
    "bank_production_partition",
    "bank_frozen_scorecard",
    "main",
    "print_frozen_gate_report",
    "score_production_shot",
]


if __name__ == "__main__":
    main()
