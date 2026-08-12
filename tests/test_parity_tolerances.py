import math

import pytest

from nova.imas.parity_tolerances import (
    HELD_OUT_CANDIDATE_STRATA,
    HELD_OUT_DRAW_SEED,
    HELD_OUT_EXTENSION,
    PARITY_TOLERANCES,
    SCORECARD_FIELDS,
    BoundDirection,
    FieldPolarity,
    MagneticsBudgetClass,
    ScorecardField,
    draw_held_out_extension,
    registered_tolerances,
    scorecard_verdicts,
    validate_scorecard_fields,
)


def test_every_scorecard_field_has_one_documented_numeric_tolerance():
    assert set(PARITY_TOLERANCES) == SCORECARD_FIELDS
    assert SCORECARD_FIELDS == {field.value for field in ScorecardField}
    for field, tolerance in PARITY_TOLERANCES.items():
        assert field == tolerance.field
        assert isinstance(tolerance.bound, float)
        assert math.isfinite(tolerance.bound)
        assert tolerance.bound >= 0.0
        assert tolerance.basis.strip()
        assert tolerance.evidence.strip()


def test_magnetics_budget_classes_change_only_the_measured_residual_bound():
    strict = registered_tolerances(MagneticsBudgetClass.SAME_SOURCE)
    cutover = registered_tolerances(MagneticsBudgetClass.SOURCE_CUTOVER)
    residual = ScorecardField.MAGNETICS_RESIDUAL_WHITENED_RMS

    assert strict[residual].bound == pytest.approx(0.747504115002785)
    assert cutover[residual].bound == pytest.approx(0.7987932587351543)
    assert cutover[residual].bound > strict[residual].bound
    assert {
        field: tolerance for field, tolerance in strict.items() if field != residual
    } == {field: tolerance for field, tolerance in cutover.items() if field != residual}


def test_a_scorecard_cannot_be_scored_with_missing_or_unregistered_metrics():
    complete = {
        field: tolerance.bound for field, tolerance in PARITY_TOLERANCES.items()
    }
    assert set(scorecard_verdicts(complete)) == SCORECARD_FIELDS

    incomplete = dict(complete)
    incomplete.pop(ScorecardField.LCFS_DISTANCE_M)
    with pytest.raises(ValueError, match="missing registered fields"):
        scorecard_verdicts(incomplete)

    with pytest.raises(ValueError, match="unregistered fields"):
        validate_scorecard_fields((*complete, "unregistered_metric"))


def test_each_bound_direction_and_nonfinite_failure_are_enforced():
    for tolerance in PARITY_TOLERANCES.values():
        assert not tolerance.passes(float("nan"))
        assert not tolerance.passes(float("inf"))
        assert tolerance.passes(tolerance.bound)
        if tolerance.direction is BoundDirection.AT_MOST:
            assert not tolerance.passes(math.nextafter(tolerance.bound, math.inf))
        else:
            assert not tolerance.passes(math.nextafter(tolerance.bound, -math.inf))


def test_the_seeded_held_out_extension_is_pinned_across_campaigns_and_polarities():
    assert HELD_OUT_DRAW_SEED == 20260812
    assert tuple(candidate.shot_id for candidate in HELD_OUT_EXTENSION) == (
        11794,
        12417,
        27000,
        22500,
    )
    assert HELD_OUT_EXTENSION == draw_held_out_extension(HELD_OUT_DRAW_SEED)
    assert len({candidate.campaign for candidate in HELD_OUT_EXTENSION}) == 3
    assert {candidate.field_polarity for candidate in HELD_OUT_EXTENSION} == {
        FieldPolarity.FORWARD,
        FieldPolarity.REVERSED,
    }
    assert all(
        selected in stratum
        for selected, stratum in zip(HELD_OUT_EXTENSION, HELD_OUT_CANDIDATE_STRATA)
    )
    assert not {candidate.shot_id for candidate in HELD_OUT_EXTENSION} & {
        21978,
        21983,
        21985,
        21986,
        21989,
        22086,
    }
