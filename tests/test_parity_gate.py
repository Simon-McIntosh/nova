"""Frozen-cohort banking preserves per-slice verdicts and coverage causes."""

import json
from types import SimpleNamespace

import numpy as np

from nova.imas.mast_efit_referee import FROZEN_SHOTS
from nova.imas.mast_parity_gate import bank_frozen_scorecard
from nova.imas.parity_tolerances import (
    SCORECARD_FIELDS,
    MagneticsBudgetClass,
    ScorecardField,
)


def _scored_result(shot: int, *, missing_reference: bool = True):
    count = 3
    time = np.array([0.10, 0.11, 0.12])
    usable = np.array([True, not missing_reference, True])
    reference_index = np.array([0, -1 if missing_reference else 1, 2])
    geometry = SimpleNamespace(
        usable_reference=usable,
        reference_index=reference_index,
        reference_time_s=np.where(usable, time, np.nan),
        magnetic_axis_distance_m=np.array([0.0001, 0.0002, 0.0003]),
        lcfs_distance_m=np.array([0.0001, 0.0002, 0.0003]),
        x_point_distance_m=np.array([0.001, 0.002, np.nan]),
        topology_class_agreement=np.ones(count),
    )
    scorecard = SimpleNamespace(
        shot=shot,
        time_s=time,
        slice_count=count,
        magnetics_budget=MagneticsBudgetClass.SOURCE_CUTOVER,
        physics=SimpleNamespace(whitened_raw_magnetics_residual=np.full(count, 0.1)),
        solve_health=SimpleNamespace(
            iteration_count=np.full(count, 8),
            throughput_slices_per_second=np.ones(count),
        ),
        temporal=SimpleNamespace(
            current_diffusion_flux_ledger_consistency=np.full(count, 0.001)
        ),
    )
    chain = SimpleNamespace(
        scorecard=scorecard,
        solve=SimpleNamespace(
            residual=np.full(count, 1.0e-10),
            flux=np.arange(count * 4, dtype=float).reshape(count, 4),
        ),
        topology=SimpleNamespace(core_mask=np.ones((count, 2), dtype=bool)),
    )
    return SimpleNamespace(scorecard=scorecard, geometry_scores=geometry, chain=chain)


def test_bank_covers_all_six_shots_and_all_registered_fields(tmp_path):
    artifact = tmp_path / "scorecard.json"
    figures = tmp_path / "figures"

    report = bank_frozen_scorecard(
        _scored_result,
        artifact_path=artifact,
        figure_dir=figures,
    )

    assert report.completed_shots == FROZEN_SHOTS
    assert report.incomplete_shots == ()
    assert report.status == "fail"
    assert len(report.scored_slices) == 12
    assert len(report.skipped_slices) == 6
    for row in report.scored_slices:
        assert set(row.metrics) == SCORECARD_FIELDS
        assert set(row.verdicts) == SCORECARD_FIELDS
    for summary in report.shot_summaries:
        assert summary.available_slices == 3
        assert summary.scored_slices == 2
        assert summary.skipped_slices == 1
        assert summary.skip_causes == {"no-reference-within-time-tolerance": 1}
        assert set(summary.pass_fraction_by_metric) == SCORECARD_FIELDS
        assert (
            summary.pass_fraction_by_metric[ScorecardField.X_POINT_DISTANCE_M.value]
            == 0.5
        )

    banked = json.loads(artifact.read_text())
    assert banked["requested_shots"] == list(FROZEN_SHOTS)
    assert banked["incomplete_shots"] == []
    assert len(banked["scored_slices"]) == 12
    assert banked["scored_slices"][-1]["metrics"]["x_point_distance_m"] is None
    assert {path.name for path in figures.iterdir()} == {
        "boundary-distance-distributions.svg",
        "flux-map-overlays.svg",
        "residual-decomposition.svg",
    }
    assert report.figures == (
        "/nova/figures/spine-efit-parity/flux-map-overlays.svg",
        "/nova/figures/spine-efit-parity/boundary-distance-distributions.svg",
        "/nova/figures/spine-efit-parity/residual-decomposition.svg",
    )


def test_nonfinite_metric_is_a_scored_failure_not_a_skip(tmp_path):
    report = bank_frozen_scorecard(
        lambda shot: _scored_result(shot, missing_reference=False),
        shots=(21978,),
        artifact_path=tmp_path / "scorecard.json",
        figure_dir=tmp_path / "figures",
    )

    assert report.shot_summaries[0].scored_slices == 3
    assert report.shot_summaries[0].skipped_slices == 0
    failed = report.scored_slices[-1]
    assert np.isnan(failed.metrics[ScorecardField.X_POINT_DISTANCE_M.value])
    assert not failed.verdicts[ScorecardField.X_POINT_DISTANCE_M.value]


def test_failed_shot_is_named_while_remaining_shots_continue(tmp_path):
    def scorer(shot):
        if shot == 21986:
            raise FileNotFoundError("referee catalogue unavailable")
        return _scored_result(shot)

    report = bank_frozen_scorecard(
        scorer,
        artifact_path=tmp_path / "scorecard.json",
        figure_dir=tmp_path / "figures",
    )

    assert report.status == "incomplete"
    assert report.incomplete_shots == (21986,)
    assert set(report.completed_shots) == set(FROZEN_SHOTS) - {21986}
    assert report.run_errors == {
        21986: "FileNotFoundError: referee catalogue unavailable"
    }
