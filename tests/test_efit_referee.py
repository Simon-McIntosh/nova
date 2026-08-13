"""EFIT geometry is available to scoring and absent from reconstruction inputs."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import nova.imas.mast_efit_referee as referee_module
from nova.imas.mast_efit_referee import (
    FROZEN_SHOTS,
    EfitReferee,
    read_efit_referee,
    run_refereed_parity_chain,
    score_with_efit_referee,
)
from nova.imas.mast_parity_chain import (
    AcceleratedProfileSolve,
    GeometryScores,
    ParityChainResult,
    PhysicsScores,
    SliceScorecard,
    SolveHealthScores,
    TemporalScores,
    TopologyLabels,
)
from nova.imas.parity_tolerances import SCORECARD_FIELDS, ScorecardField

SHOT_STORE = Path("/work/projects/imas_gpu/mast/level1/shots")
USABLE_REFERENCE_COUNTS = {
    21978: 70,
    21983: 69,
    21985: 67,
    21986: 61,
    21989: 73,
    22086: 57,
}


def _topology() -> TopologyLabels:
    count = 3
    shifts = np.array([0.001, 0.002, 0.003])
    reference_lcfs = np.array([[0.8, -0.2], [1.2, -0.2], [1.2, 0.2], [0.8, 0.2]])
    lcfs = np.stack([reference_lcfs + np.array([shift, 0.0]) for shift in shifts])
    core = np.ones((count, 1, 1), dtype=bool)
    return TopologyLabels(
        magnetic_axis_m=np.column_stack([1.0 + shifts, np.zeros(count)]),
        x_point_m=np.array([[1.004, -0.2], [np.nan, np.nan], [1.006, -0.2]]),
        lcfs_m=lcfs,
        diverted=np.array([True, False, True]),
        core_mask=core,
        common_scrape_off_mask=np.zeros_like(core),
        private_flux_mask=np.zeros_like(core),
        excluded_material_mask=np.zeros_like(core),
    )


def _referee() -> EfitReferee:
    count = 3
    lcfs = np.broadcast_to(
        np.array([[0.8, -0.2], [1.2, -0.2], [1.2, 0.2], [0.8, 0.2]]),
        (count, 4, 2),
    ).copy()
    return EfitReferee(
        shot=21978,
        time_s=np.array([0.0, 0.005, 0.010]),
        magnetic_axis_m=np.array([[1.0, 0.0]] * count),
        lcfs_m=lcfs,
        x_points_m=np.array(
            [
                [[1.0, -0.2], [np.nan, np.nan]],
                [[np.nan, np.nan], [np.nan, np.nan]],
                [[1.0, -0.2], [1.0, 0.2]],
            ]
        ),
        diverted=np.array([True, False, True]),
        usable=np.ones(count, dtype=bool),
    )


def _chain_result() -> ParityChainResult:
    count = 3
    time_s = np.array([0.0, 0.005, 0.010])
    topology = _topology()
    metrics = {field: 0.0 for field in SCORECARD_FIELDS}
    for field in (
        ScorecardField.MAGNETIC_AXIS_DISTANCE_M,
        ScorecardField.LCFS_DISTANCE_M,
        ScorecardField.X_POINT_DISTANCE_M,
        ScorecardField.TOPOLOGY_CLASS_AGREEMENT_FRACTION,
    ):
        metrics[field.value] = float("nan")
    scorecard = SliceScorecard(
        shot=21978,
        time_s=time_s,
        geometry=GeometryScores(
            magnetic_axis_m=topology.magnetic_axis_m,
            lcfs_m=topology.lcfs_m,
            x_point_m=topology.x_point_m,
            diverted=topology.diverted,
            seed_to_solved_lcfs_distance_m=np.zeros(count),
        ),
        physics=PhysicsScores(
            profile_residual=np.zeros(count),
            whitened_raw_magnetics_residual=np.zeros(count),
        ),
        solve_health=SolveHealthScores(
            convergence_fraction=np.ones(count),
            confinement_fraction=np.ones(count),
            iteration_count=np.ones(count),
            throughput_slices_per_second=np.ones(count),
        ),
        temporal=TemporalScores(
            current_diffusion_flux_ledger_consistency=np.zeros(count)
        ),
        registered_metrics=metrics,
    )
    return ParityChainResult(
        inputs=SimpleNamespace(time_s=time_s, slice_count=count),
        moment_seeds=(),
        solve=AcceleratedProfileSolve(
            flux=np.zeros((count, 1)),
            residual=np.zeros(count),
            trace=np.zeros((count, 1)),
            elapsed_s=1.0,
        ),
        topology=topology,
        scorecard=scorecard,
    )


def test_reference_geometry_replaces_every_nan_scorecard_field():
    """The pilot comparison emits finite registered geometry metrics."""

    scored = score_with_efit_referee(_chain_result(), _referee())
    metrics = scored.scorecard.registered_metrics

    assert scored.usable_reference_slices == 3
    assert metrics[ScorecardField.MAGNETIC_AXIS_DISTANCE_M] == pytest.approx(0.002)
    assert metrics[ScorecardField.LCFS_DISTANCE_M] == pytest.approx(0.002)
    assert metrics[ScorecardField.X_POINT_DISTANCE_M] == pytest.approx(0.004)
    assert metrics[ScorecardField.TOPOLOGY_CLASS_AGREEMENT_FRACTION] == pytest.approx(
        1.0
    )
    assert all(np.isfinite(metrics[field.value]) for field in ScorecardField)


def test_referee_is_opened_only_after_the_solve_and_never_reaches_solve_inputs(
    monkeypatch,
):
    """Any attempt to expose referee state to reconstruction fails this test."""

    reference = _referee()
    events = []
    moment_solver = object()
    profile_solver = object()
    topology_labeler = object()

    def run_chain(shot, **arguments):
        assert events == []
        assert all(value is not reference for value in arguments.values())
        assert all(
            value is not reference.magnetic_axis_m for value in arguments.values()
        )
        assert arguments == {
            "moment_solver": moment_solver,
            "profile_solver": profile_solver,
            "topology_labeler": topology_labeler,
            "store": "/measurements",
        }
        events.append("solve-complete")
        return _chain_result()

    def read_reference(shot, *, store):
        assert events == ["solve-complete"]
        assert store == "/references"
        events.append("referee-opened")
        return reference

    monkeypatch.setattr(referee_module, "run_parity_chain", run_chain)
    monkeypatch.setattr(referee_module, "read_efit_referee", read_reference)
    scored = run_refereed_parity_chain(
        21978,
        moment_solver=moment_solver,
        profile_solver=profile_solver,
        topology_labeler=topology_labeler,
        store="/measurements",
        referee_store="/references",
    )

    assert events == ["solve-complete", "referee-opened"]
    assert all(
        np.isfinite(scored.scorecard.registered_metrics[field.value])
        for field in (
            ScorecardField.MAGNETIC_AXIS_DISTANCE_M,
            ScorecardField.LCFS_DISTANCE_M,
            ScorecardField.X_POINT_DISTANCE_M,
            ScorecardField.TOPOLOGY_CLASS_AGREEMENT_FRACTION,
        )
    )


@pytest.mark.parametrize("shot", FROZEN_SHOTS)
def test_frozen_shot_catalogues_supply_read_only_reference_geometry(shot):
    """Every frozen shot reports its measured count of usable EFIT rows."""

    if not (SHOT_STORE / f"{shot}.zarr").exists():
        pytest.skip("MAST level-one catalogue is not mounted")
    reference = read_efit_referee(shot, store=SHOT_STORE)
    usable = reference.usable

    assert reference.usable_slice_count == USABLE_REFERENCE_COUNTS[shot]
    assert reference.magnetic_axis_m.shape == (reference.slice_count, 2)
    assert reference.lcfs_m.shape[0] == reference.slice_count
    assert reference.lcfs_m.shape[1] >= 3
    assert reference.lcfs_m.shape[2] == 2
    assert reference.x_points_m.shape == (reference.slice_count, 2, 2)
    assert reference.diverted.shape == (reference.slice_count,)
    assert np.all(np.isfinite(reference.magnetic_axis_m[usable]))
    assert np.all(np.sum(np.isfinite(reference.lcfs_m[usable, :, 0]), axis=1) >= 3)
    assert np.all(
        np.any(np.all(np.isfinite(reference.x_points_m), axis=2), axis=1)[
            usable & reference.diverted
        ]
    )
    for values in (
        reference.time_s,
        reference.magnetic_axis_m,
        reference.lcfs_m,
        reference.x_points_m,
        reference.diverted,
        reference.usable,
    ):
        assert not values.flags.writeable
