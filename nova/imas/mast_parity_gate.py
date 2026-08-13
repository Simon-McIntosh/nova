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
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from nova.imas.mast_chain_factory import build_mast_parity_chain
from nova.imas.mast_efit_referee import (
    FROZEN_SHOTS,
    RefereedParityResult,
    run_refereed_parity_chain,
)
from nova.imas.mast_solve_inputs import SHOT_STORE
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


def score_production_shot(
    shot: int,
    *,
    artifact_cache: Path | str,
    artifact_digest: str,
    store: Path | str = SHOT_STORE,
    radial_points: int = 33,
    vertical_points: int = 49,
) -> RefereedParityResult:
    """Score one shot using only components returned by the production factory."""

    components = build_mast_parity_chain(
        int(shot),
        artifact_cache=artifact_cache,
        artifact_digest=artifact_digest,
        store=store,
        radial_points=radial_points,
        vertical_points=vertical_points,
    )
    return run_refereed_parity_chain(
        int(shot),
        moment_solver=components.moment_solver,
        profile_solver=components.profile_solver,
        topology_labeler=components.topology_labeler,
        temporal_scorer=components.temporal_scorer,
        magnetics_budget=MagneticsBudgetClass.SOURCE_CUTOVER,
        store=store,
        referee_store=store,
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
    result: RefereedParityResult,
) -> tuple[list[ScoredSlice], list[SkippedSlice], ShotSummary]:
    """Score all aligned slices and retain every unaligned row with its cause."""

    scorecard = result.scorecard
    geometry = result.geometry_scores
    scored: list[ScoredSlice] = []
    skipped: list[SkippedSlice] = []
    for index, time_s in enumerate(np.asarray(scorecard.time_s, dtype=float)):
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
                    slice_index=index,
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
                slice_index=index,
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
        available_slices=int(scorecard.slice_count),
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


def _bank_report(report: FrozenGateReport, path: Path) -> None:
    """Write the scorecard atomically as strict, deterministic JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_value(asdict(report)), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


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
    scorer: Callable[[int], RefereedParityResult] | None = None,
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
    errors: dict[int, str] = {}
    budgets: set[str] = set()

    attempted: list[int] = []
    for shot in shots:
        if max_shots is not None and len(attempted) >= max_shots:
            break
        attempted.append(int(shot))
        try:
            result = scorer(int(shot))
            if int(result.scorecard.shot) != int(shot):
                raise ValueError(
                    f"scorer returned shot {result.scorecard.shot} for request {shot}"
                )
            shot_scored, shot_skipped, summary = _score_result(result)
        except Exception as error:  # continue so the bank names every uncovered shot
            errors[int(shot)] = f"{type(error).__name__}: {error}"
            continue
        results[int(shot)] = result
        budgets.add(str(result.scorecard.magnetics_budget))
        scored.extend(shot_scored)
        skipped.extend(shot_skipped)
        summaries.append(summary)

    if len(budgets) > 1:
        raise ValueError(f"shots were scored with mixed magnetics budgets: {budgets}")
    completed = tuple(shot for shot in shots if shot in results)
    incomplete = tuple(shot for shot in shots if shot not in results)
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
    parser.add_argument("--artifact-cache", type=Path, required=True)
    parser.add_argument("--artifact-digest", required=True)
    parser.add_argument("--store", type=Path, default=SHOT_STORE)
    parser.add_argument("--artifact-path", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--radial-points", type=int, default=33)
    parser.add_argument("--vertical-points", type=int, default=49)
    parser.add_argument("--max-shots", type=int)
    return parser


def main() -> None:
    """Run and bank the production frozen-cohort scorecard."""

    args = _parser().parse_args()
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
    "ScoredSlice",
    "ShotSummary",
    "SkippedSlice",
    "bank_frozen_scorecard",
    "main",
    "print_frozen_gate_report",
    "score_production_shot",
]


if __name__ == "__main__":
    main()
