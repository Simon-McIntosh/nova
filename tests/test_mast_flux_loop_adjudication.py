"""What decides a loop's position, and what must not be allowed to decide it.

The loop adjudication has one failure mode worth more than all the others: calling
a disagreement settled because it is *settleable*.  A seven-millimetre
displacement changes a loop's linked flux by fifty times its own noise, which says
a fit can answer the question and says nothing at all about which answer is right.
So the tests here insist that separability and decision stay apart, and that a
decision needs a winner outside the shots that chose it.

The other failure mode is the join.  Six centre-column channels sit at one table
position within a nanometre, so a positional join hands them all the same loop and
every per-loop statement made through it is about an arbitrary member of a
degenerate set.  The tests pin the resolved join against the real registry, where
that degeneracy is present and has to not matter.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from nova.catalog.mast_geometry import MachineGeometryRegistry
from nova.imas.mast_flux_loop_adjudication import (
    DECISION_MARGIN,
    MINIMUM_HELD_OUT_SHOTS,
    MIRROR_TOLERANCE,
    SEPARATION_MARGIN,
    CandidateFit,
    LoopComparison,
    LoopDisposition,
    LoopShotResidual,
    build_ledger,
    described_loop_positions,
    dispose,
    fit_candidate_positions,
    loop_flux_response,
    mirror_pairs,
)
from nova.imas.mast_solve_inputs import reconstruction_loop_rows
from nova.imas.mast_vacuum_response import ResponseError

REPRESENTATIVE_SHOT = 11766
"""Registry selection the loop tables are read from."""

SCATTER = 1.0e-4
"""Webers of quiescent scatter the synthetic channels are given."""


@pytest.fixture(scope="module")
def geometry():
    """Return the registry configuration the loops are described in."""

    return (
        MachineGeometryRegistry.default()
        .select(REPRESENTATIVE_SHOT)
        .configuration.geometry
    )


def comparison(
    channel: str = "fl_p2u_1",
    *,
    displacement: float = 9.0e-3,
    flux_separation: float = 7.0e-3,
    scatter: float = SCATTER,
    described_index: int | None = 12,
) -> LoopComparison:
    """Build one loop's two candidates with its separability stated outright."""

    return LoopComparison(
        channel=channel,
        described_r=0.4450,
        described_z=1.5780,
        reconstruction_r=0.4450,
        reconstruction_z=1.5780 - displacement,
        displacement=displacement,
        mirror_displacement=0.0,
        flux_separation=flux_separation,
        scatter=scatter,
        described_index=described_index,
    )


def candidate_fit(
    channel: str = "fl_p2u_1",
    *,
    described_train: float,
    reconstruction_train: float,
    described_held: float,
    reconstruction_held: float,
    held_out: int = 55,
) -> CandidateFit:
    """Build a candidate comparison with both residual pairs stated outright."""

    return CandidateFit(
        channel=channel,
        shot_count=208,
        held_out_count=held_out,
        described_residual=described_train,
        reconstruction_residual=reconstruction_train,
        described_held_out=described_held,
        reconstruction_held_out=reconstruction_held,
        described_scale=1.0,
        reconstruction_scale=1.0,
    )


# --- separability is not a verdict --------------------------------------


def test_a_separable_loop_with_no_fit_stays_dual_valued():
    """Being answerable is not an answer, and this is the whole point.

    A displacement whose flux difference is seventy times the channel's noise
    still needs a fit to say which position the data prefers; promoting on the
    separation alone would write whichever table was read first into a sensor
    pose the artifact's identity carries.
    """

    row = comparison(flux_separation=SCATTER * 70.0)
    assert row.separable
    assert row.separation_ratio == pytest.approx(70.0)
    assert dispose(row, None) is LoopDisposition.DUAL_VALUED


def test_a_fit_that_wins_in_and_out_of_sample_promotes():
    """A candidate the data prefers on shots it never saw is the promoted one."""

    row = comparison()
    fit = candidate_fit(
        described_train=4.23e-3,
        reconstruction_train=4.80e-3,
        described_held=4.52e-3,
        reconstruction_held=5.16e-3,
    )
    assert fit.margin > DECISION_MARGIN
    assert fit.prefers_described
    assert fit.agrees_in_sample
    assert fit.decided
    assert dispose(row, fit) is LoopDisposition.PROMOTED


def test_an_advantage_below_the_margin_decides_nothing():
    """Two positions millimetres apart differ by percent even when both are wrong."""

    row = comparison()
    fit = candidate_fit(
        described_train=2.5925e-2,
        reconstruction_train=2.5971e-2,
        described_held=2.8882e-2,
        reconstruction_held=2.8930e-2,
    )
    assert 0.0 < fit.margin < DECISION_MARGIN
    assert not fit.decided
    assert dispose(row, fit) is LoopDisposition.DUAL_VALUED


def test_a_winner_that_changes_out_of_sample_is_not_a_winner():
    """Training and held-out disagreeing means the fit chose noise."""

    fit = candidate_fit(
        described_train=4.0e-3,
        reconstruction_train=5.0e-3,
        described_held=5.0e-3,
        reconstruction_held=4.0e-3,
    )
    assert not fit.agrees_in_sample
    assert not fit.decided
    assert dispose(comparison(), fit) is LoopDisposition.DUAL_VALUED


def test_too_little_held_out_coverage_decides_nothing():
    """A handful of unseen shots makes the challenge a coin toss."""

    fit = candidate_fit(
        described_train=4.0e-3,
        reconstruction_train=5.0e-3,
        described_held=4.0e-3,
        reconstruction_held=5.0e-3,
        held_out=MINIMUM_HELD_OUT_SHOTS - 1,
    )
    assert fit.margin > DECISION_MARGIN
    assert fit.agrees_in_sample
    assert not fit.decided


def test_the_reconstruction_can_win_too():
    """The criterion is symmetric; nothing privileges the description."""

    fit = candidate_fit(
        described_train=5.0e-3,
        reconstruction_train=4.0e-3,
        described_held=5.0e-3,
        reconstruction_held=4.0e-3,
    )
    assert fit.decided
    assert not fit.prefers_described
    assert fit.margin < 0.0


def test_an_unseparable_displacement_is_reported_as_bounded():
    """Below the separation margin the loop is not merely undecided but bounded."""

    row = comparison(flux_separation=SCATTER * 1.05)
    assert not row.separable
    assert row.separation_ratio == pytest.approx(1.05)
    assert row.separation_ratio < SEPARATION_MARGIN


def test_a_channel_with_no_measured_scatter_is_never_called_separable():
    """A separability claim needs a measured floor, not a missing one."""

    row = comparison(scatter=0.0)
    assert not row.separable
    assert math.isinf(row.separation_ratio)


def test_a_loop_the_description_does_not_carry_is_its_own_disposition():
    """A reconstruction loop with no counterpart is a gap, not a disagreement."""

    row = comparison(described_index=None, displacement=0.24)
    assert dispose(row, None) is LoopDisposition.NO_DESCRIBED_COUNTERPART


def test_loops_the_sources_agree_on_need_no_decision():
    """Agreement is decided by position, before any fit is consulted."""

    row = comparison(displacement=MIRROR_TOLERANCE / 2.0)
    assert row.agrees
    assert dispose(row, None) is LoopDisposition.AGREED


# --- the join and the described table -----------------------------------


def test_the_resolved_join_is_a_bijection_over_the_channel_blocks():
    """Every channel names one reconstruction row and no row is named twice."""

    rows = reconstruction_loop_rows()
    assert len(set(rows.values())) == len(rows)
    assert min(rows.values()) == 0
    assert max(rows.values()) == len(rows) - 1


def test_the_described_table_has_a_degeneracy_the_join_must_not_use(geometry):
    """Several described loops share one position, so proximity cannot key a loop.

    This is the case the resolved join exists for: matching by nearest position
    would hand every channel at this radius the same loop.
    """

    positions = described_loop_positions(geometry)
    unique = {(round(r, 6), round(z, 6)) for r, z in positions}
    assert len(unique) < len(positions)


def test_reflection_is_a_property_some_families_have_and_others_do_not(geometry):
    """Mirroring is evidence where it holds, so it is detected and not assumed.

    Pairing runs one way at a time.  A loop whose reflection is unique is paired
    even when that reflection has two candidates of its own and is therefore left
    unpaired itself, so the property to hold is that no two loops ever disagree
    about each other -- not that every pairing comes back.
    """

    positions = described_loop_positions(geometry)
    pairs = mirror_pairs(positions)
    assert 0 < len(pairs) < len(positions)
    for index, other in pairs.items():
        assert positions[index, 0] == pytest.approx(
            positions[other, 0], abs=MIRROR_TOLERANCE
        )
        assert positions[index, 1] == pytest.approx(
            -positions[other, 1], abs=MIRROR_TOLERANCE
        )
        assert pairs.get(other, index) == index


def test_a_pair_thirty_microns_from_reflecting_is_two_measurements(geometry):
    """The tolerance is tight so a surveyed near-symmetry is not read as mirrored."""

    positions = described_loop_positions(geometry)
    pairs = mirror_pairs(positions)
    outboard = [
        index
        for index, (radius, _) in enumerate(positions)
        if abs(radius - 1.7493) < 1.0e-6
    ]
    assert outboard, "the outboard loop family must be present to test this"
    assert not any(index in pairs for index in outboard)


# --- the flux the candidates predict ------------------------------------


def test_moving_a_loop_changes_the_flux_it_links(geometry):
    """The separation is computed, not estimated, so it must respond to position."""

    near = np.asarray([[0.4450, 1.5780]], dtype=float)
    far = np.asarray([[0.4450, 1.5690]], dtype=float)
    first = loop_flux_response(geometry, near)
    second = loop_flux_response(geometry, far)
    assert first.shape == second.shape
    assert np.all(np.isfinite(first))
    assert not np.allclose(first, second)


def test_a_component_without_a_cross_section_cannot_be_coupled(geometry):
    """A degenerate outline is an error, never a silently skipped column.

    The refusal comes from the shared kernel rather than from this module, which
    is the point of routing both sensor kinds through one coupling: a coil that
    cannot be coupled to a probe cannot be coupled to a loop either.
    """

    import shapely

    empty = dict(geometry)
    empty["active_components"] = dict(geometry["active_components"])
    empty["active_components"]["sol"] = shapely.Polygon(
        [(1.0, 0.0), (1.0, 0.0), (1.0, 0.0)]
    ).wkb.hex()
    with pytest.raises(ResponseError, match="no cross-section"):
        loop_flux_response(empty, np.asarray([[0.4450, 1.5780]], dtype=float))


# --- pooling and the ledger ---------------------------------------------


def test_pooling_keeps_the_two_candidates_apart():
    """Each candidate is pooled on its own, so neither borrows the other's fit."""

    training = [
        LoopShotResidual(
            shot=index,
            channel="fl_p2u_1",
            candidate=candidate,
            scale=1.0,
            residual=residual,
            signal=1.0e-2,
        )
        for index in range(6)
        for candidate, residual in (("described", 4.0e-3), ("reconstruction", 5.0e-3))
    ]
    held = [
        LoopShotResidual(
            shot=100 + index,
            channel="fl_p2u_1",
            candidate=candidate,
            scale=1.0,
            residual=residual,
            signal=1.0e-2,
        )
        for index in range(MINIMUM_HELD_OUT_SHOTS)
        for candidate, residual in (("described", 4.1e-3), ("reconstruction", 5.2e-3))
    ]
    fits = fit_candidate_positions(training, held)
    assert len(fits) == 1
    fit = fits[0]
    assert fit.described_residual == pytest.approx(4.0e-3)
    assert fit.reconstruction_residual == pytest.approx(5.0e-3)
    assert fit.shot_count == 6
    assert fit.held_out_count == MINIMUM_HELD_OUT_SHOTS
    assert fit.decided and fit.prefers_described


def test_a_candidate_scored_on_no_unseen_shot_is_not_decided():
    """With nothing held out the challenge cannot run, so the loop stays open."""

    training = [
        LoopShotResidual(
            shot=index,
            channel="fl_p2u_1",
            candidate=candidate,
            scale=1.0,
            residual=residual,
            signal=1.0e-2,
        )
        for index in range(6)
        for candidate, residual in (("described", 4.0e-3), ("reconstruction", 5.0e-3))
    ]
    fit = fit_candidate_positions(training, [])[0]
    assert math.isinf(fit.described_held_out)
    assert not fit.decided


def test_the_ledger_names_the_described_loops_no_channel_reaches(geometry):
    """A described loop nothing resolves onto is reported, not silently dropped."""

    rows = [comparison(described_index=0), comparison("fl_p2u_2", described_index=1)]
    ledger = build_ledger(geometry, rows)
    described = described_loop_positions(geometry)
    assert len(ledger.unreached_indices) == described.shape[0] - 2
    assert 0 not in ledger.unreached_indices
    assert len(ledger.unreached_positions) == len(ledger.unreached_indices)


def test_the_ledger_counts_every_disposition(geometry):
    """A disposition absent from a run appears as a zero, so a reader can tell."""

    ledger = build_ledger(geometry, [comparison(described_index=0)])
    counts = ledger.counts
    assert set(counts) == {str(state) for state in LoopDisposition}
    assert counts[str(LoopDisposition.DUAL_VALUED)] == 1


def test_the_ledger_reports_which_source_each_promotion_chose(geometry):
    """A promotion is only useful if the record says which table won."""

    rows = [comparison(described_index=0)]
    fits = [
        candidate_fit(
            described_train=4.23e-3,
            reconstruction_train=4.80e-3,
            described_held=4.52e-3,
            reconstruction_held=5.16e-3,
        )
    ]
    ledger = build_ledger(geometry, rows, fits)
    assert ledger.promoted == (("fl_p2u_1", "described"),)
    assert ledger.disposition("fl_p2u_1") is LoopDisposition.PROMOTED
    assert ledger.disposition("fl_nowhere") is LoopDisposition.NO_CHANNEL


def test_the_ledger_record_round_trips_through_json(geometry):
    """The ledger is evidence, so it has to serialize exactly."""

    ledger = build_ledger(
        geometry,
        [comparison(described_index=0)],
        [
            candidate_fit(
                described_train=4.23e-3,
                reconstruction_train=4.80e-3,
                described_held=4.52e-3,
                reconstruction_held=5.16e-3,
            )
        ],
    )
    record = ledger.as_dict()
    assert json.loads(json.dumps(record, sort_keys=True)) == record
    assert record["decision_margin"] == DECISION_MARGIN
