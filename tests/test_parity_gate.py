"""Frozen-cohort banking preserves per-slice verdicts and coverage causes."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

import nova.imas.mast_parity_gate as gate_module
from nova.equilibrium.moment import UnsupportedSlice
from nova.imas.mast_efit_referee import FROZEN_SHOTS
from nova.imas.mast_parity_gate import (
    ScoredSlice,
    SlicePartitionReport,
    ProductionShotScore,
    SkippedSlice,
    aggregate_scorecard_partitions,
    bank_frozen_scorecard,
    print_frozen_gate_report,
    score_production_shot,
)
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
    assert report.not_attempted_shots == ()
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
    assert set(banked["pass_fraction_by_metric"]) == SCORECARD_FIELDS
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


def test_unsupported_first_slice_is_skipped_and_later_slices_are_scored(tmp_path):
    def scorer(shot):
        result = _scored_result(shot, missing_reference=False)
        return ProductionShotScore(
            shot=shot,
            available_slices=4,
            source_slice_indices=(1, 2, 3),
            result=result,
            skipped_slices=(
                SkippedSlice(
                    shot=shot,
                    slice_index=0,
                    time_s=0.09,
                    cause="seed-disc-insufficient-supported-cells",
                    details={
                        "supported_cell_count": 4,
                        "minimum_cell_count": 5,
                    },
                ),
            ),
            magnetics_budget=MagneticsBudgetClass.SOURCE_CUTOVER,
            min_cells=5,
        )

    report = bank_frozen_scorecard(
        scorer,
        shots=(21978,),
        artifact_path=tmp_path / "scorecard.json",
        figure_dir=tmp_path / "figures",
    )

    assert report.completed_shots == (21978,)
    assert [row.slice_index for row in report.scored_slices] == [1, 2, 3]
    assert report.skipped_slices[0].slice_index == 0
    assert report.shot_summaries[0].available_slices == 4
    assert report.shot_summaries[0].scored_slices == 3
    assert report.shot_summaries[0].skip_causes == {
        "seed-disc-insufficient-supported-cells": 1
    }
    assert report.skipped_slices[0].details == {
        "supported_cell_count": 4,
        "minimum_cell_count": 5,
    }


def test_seed_preparation_catches_only_typed_unsupported_slices():
    inputs = SimpleNamespace(
        slice_count=2,
        time_s=np.array([0.1, 0.2]),
        coil_channels=("coil",),
        coil_currents_a=np.ones((2, 1)),
        sensor_signals=np.ones((2, 1)),
        plasma_current_a=np.ones(2),
    )
    profile = SimpleNamespace(
        source_names=("coil",),
        source_to_sensor=np.ones((1, 1)),
        source_to_grid=np.ones((4, 1)),
    )

    class MomentSolver:
        calls = 0

        def solve(self, _measurement):
            self.calls += 1
            if self.calls == 1:
                raise UnsupportedSlice(
                    "seed-disc-insufficient-supported-cells",
                    supported_cell_count=4,
                    minimum_cell_count=5,
                )
            return SimpleNamespace(flux=np.zeros(4))

    indices, seeds, skipped, _scale = gate_module._supported_moment_seeds(
        21978, inputs, MomentSolver(), profile
    )

    assert indices == (1,)
    assert len(seeds) == 1
    assert skipped[0].cause == "seed-disc-insufficient-supported-cells"
    assert skipped[0].details["supported_cell_count"] == 4

    class BrokenMomentSolver:
        def solve(self, _measurement):
            raise ValueError("linear algebra defect")

    with pytest.raises(ValueError, match="linear algebra defect"):
        gate_module._supported_moment_seeds(
            21978, inputs, BrokenMomentSolver(), profile
        )


def _partition_report(start, stop, *, scored_indices=()):
    scored_set = set(scored_indices)
    metrics = {field: 0.0 for field in SCORECARD_FIELDS}
    verdicts = {field: False for field in SCORECARD_FIELDS}
    return SlicePartitionReport(
        generated_at="2026-08-13T00:00:00+00:00",
        shot=21978,
        available_slices=4,
        slice_start=start,
        slice_stop=stop,
        radial_points=33,
        vertical_points=49,
        min_cells=5,
        magnetics_budget=str(MagneticsBudgetClass.SOURCE_CUTOVER),
        scored_slices=tuple(
            ScoredSlice(
                shot=21978,
                slice_index=index,
                time_s=0.01 * index,
                reference_time_s=0.01 * index,
                metrics=metrics,
                verdicts=verdicts,
            )
            for index in range(start, stop)
            if index in scored_set
        ),
        skipped_slices=tuple(
            SkippedSlice(
                shot=21978,
                slice_index=index,
                time_s=0.01 * index,
                cause="seed-disc-insufficient-supported-cells",
            )
            for index in range(start, stop)
            if index not in scored_set
        ),
    )


def test_partition_aggregation_requires_exact_coverage_and_banks_boundaries(tmp_path):
    first = tmp_path / "partition-0.json"
    second = tmp_path / "partition-1.json"
    artifact = tmp_path / "scorecard.json"
    gate_module._bank_report(_partition_report(0, 2, scored_indices=(1,)), first)
    gate_module._bank_report(_partition_report(2, 4, scored_indices=(2, 3)), second)

    report = aggregate_scorecard_partitions((first, second), artifact_path=artifact)

    assert report.completed_shots == (21978,)
    assert report.shot_summaries[0].available_slices == 4
    assert report.shot_summaries[0].scored_slices == 3
    assert report.shot_summaries[0].skipped_slices == 1
    assert [row.slice_index for row in report.scored_slices] == [1, 2, 3]
    assert [(row.slice_start, row.slice_stop) for row in report.partitions] == [
        (0, 2),
        (2, 4),
    ]
    assert (report.radial_points, report.vertical_points, report.min_cells) == (
        33,
        49,
        5,
    )
    banked = json.loads(artifact.read_text())
    assert banked["partitions"][0]["artifact"] == str(first.resolve())
    assert banked["radial_points"] == 33
    assert banked["vertical_points"] == 49
    assert banked["min_cells"] == 5
    assert not artifact.with_suffix(".json.tmp").exists()


def test_partition_aggregation_rejects_a_coverage_gap(tmp_path):
    first = tmp_path / "partition-0.json"
    second = tmp_path / "partition-1.json"
    gate_module._bank_report(_partition_report(0, 1), first)
    gate_module._bank_report(_partition_report(2, 4), second)

    with pytest.raises(ValueError, match="expected slice 1, found 2"):
        aggregate_scorecard_partitions(
            (first, second), artifact_path=tmp_path / "scorecard.json"
        )


def test_nonfinite_trace_is_retained_as_zero_convergence():
    trace = np.array([[np.nan, np.nan], [4.0, 2.0]])
    final = np.array([np.nan, 1.0])

    fraction = gate_module._scorecard_convergence_fraction(trace, final)

    np.testing.assert_allclose(fraction, [0.0, 0.75])


def test_completed_chain_with_no_reference_rows_remains_available_for_skipping(
    monkeypatch,
):
    geometry = SimpleNamespace(usable_slice_count=0)
    chain = SimpleNamespace(
        scorecard=SimpleNamespace(time_s=np.array([0.1])), topology=object()
    )
    referee = object()
    monkeypatch.setattr(gate_module, "compare_reference_geometry", lambda *_a: geometry)
    monkeypatch.setattr(
        gate_module,
        "score_with_efit_referee",
        lambda *_a: pytest.fail("empty reference comparison must not be reduced"),
    )

    result = gate_module._score_completed_chain(chain, referee)

    assert result.chain is chain
    assert result.referee is referee
    assert result.geometry_scores is geometry


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


def test_shot_budget_names_unattempted_remainder_and_prints_overall_fractions(
    tmp_path, capsys
):
    report = bank_frozen_scorecard(
        _scored_result,
        artifact_path=tmp_path / "scorecard.json",
        figure_dir=tmp_path / "figures",
        max_shots=2,
    )

    print_frozen_gate_report(report)
    output = capsys.readouterr().out

    assert report.completed_shots == FROZEN_SHOTS[:2]
    assert report.not_attempted_shots == FROZEN_SHOTS[2:]
    assert report.incomplete_shots == FROZEN_SHOTS[2:]
    assert "not_attempted_shots: [21985, 21986, 21989, 22086]" in output
    assert "overall_pass_fraction_by_metric:" in output
    for field in SCORECARD_FIELDS:
        assert f"  {field}:" in output


def test_production_scorer_passes_only_factory_components_to_refereed_chain(
    monkeypatch,
):
    components = SimpleNamespace(
        moment_solver=SimpleNamespace(config=SimpleNamespace(min_cells=5)),
        profile_solver=object(),
        topology_labeler=object(),
        temporal_scorer=object(),
    )
    calls = []

    def build(shot, **arguments):
        calls.append(("build", shot, arguments))
        return components

    inputs = SimpleNamespace(slice_count=3)
    supported_inputs = object()
    seeds = (object(), object())
    scale = np.ones(2)
    chain = object()
    referee = object()
    expected = object()

    def prepare(
        shot,
        received_inputs,
        moment_solver,
        profile_solver,
        *,
        source_slice_offset,
    ):
        calls.append(
            (
                "prepare",
                shot,
                received_inputs,
                moment_solver,
                profile_solver,
                source_slice_offset,
            )
        )
        return (1, 2), seeds, (), scale

    def run(received_inputs, received_seeds, **arguments):
        calls.append(("run", received_inputs, received_seeds, arguments))
        return chain

    monkeypatch.setattr(gate_module, "build_mast_parity_chain", build)
    monkeypatch.setattr(
        gate_module, "read_corrected_solve_inputs", lambda *_a, **_k: inputs
    )
    monkeypatch.setattr(gate_module, "_supported_moment_seeds", prepare)
    monkeypatch.setattr(gate_module, "_select_inputs", lambda *_a: supported_inputs)
    monkeypatch.setattr(gate_module, "_run_supported_chain", run)
    monkeypatch.setattr(gate_module, "read_efit_referee", lambda *_a, **_k: referee)
    monkeypatch.setattr(
        gate_module,
        "_score_completed_chain",
        lambda received_chain, received_referee: (
            expected if (received_chain, received_referee) == (chain, referee) else None
        ),
    )

    result = score_production_shot(
        21978,
        artifact_cache="/machine",
        artifact_digest="sha256:" + "a" * 64,
        store="/shots",
        radial_points=17,
        vertical_points=25,
    )

    assert result.result is expected
    assert result.source_slice_indices == (1, 2)
    assert result.available_slices == 3
    assert calls[0] == (
        "build",
        21978,
        {
            "artifact_cache": "/machine",
            "artifact_digest": "sha256:" + "a" * 64,
            "store": "/shots",
            "radial_points": 17,
            "vertical_points": 25,
        },
    )
    assert calls[1] == (
        "prepare",
        21978,
        supported_inputs,
        components.moment_solver,
        components.profile_solver,
        0,
    )
    assert calls[2] == (
        "run",
        supported_inputs,
        seeds,
        {
            "profile_solver": components.profile_solver,
            "topology_labeler": components.topology_labeler,
            "temporal_scorer": components.temporal_scorer,
            "sensor_scale": scale,
        },
    )
