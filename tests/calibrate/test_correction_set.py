"""Whether a correction document that cannot be applied is refused when it is read.

A correction is one multiplication or one subtraction, so every way of getting it
wrong is silent in the arithmetic and has to be caught in the document.  Four
classes of fault get their own cases here.

Two corrections covering one pulse.  A channel scoped by pulse range is the whole
point of the schema, and the fault it invites is an overlap: a read inside one
multiplies the channel twice and no array shows it.  The rule is scoped within one
status, so a superseded record may still cover the pulses of the correction that
replaced it -- refusing that would force the record to be deleted to stay valid,
which is the opposite of what a versioned set is for.

A value that is not the quantity its kind names.  An acquisition range setting moves
by discrete factors; a block whose step lands between rungs measured something real
but not a range change, and dividing it out would launder the description's own error
into the data.  Off-ladder values are refused, and so is a quantised correction that
tries to opt out by naming no ladder at all.

A status that disagrees with what the record carries.  Promoted means the read path
divides by this value, so a promoted correction without a value would report a
correction it did not make, and a withheld or refused one carrying a value hands a
number to a consumer that read past the status.

An interval that names nothing.  Bounds that run backwards, pulse and time mixed in
one interval so that neither can be ordered against the other, and pulses listed as
measured that lie outside the span the correction claims.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from nova.calibrate.correction_model import ApplicationStage, CorrectionSet
from nova.calibrate.correction_set import (
    APPLICATION_ORDER,
    KIND_STAGE,
    SCHEMA_PATH,
    SCHEMA_VERSION,
    CorrectionSetError,
    applied,
    load_correction_set,
    read_correction_set,
    stage,
    validate_correction_set,
)

LADDER = {
    "name": "acquisition_range",
    "kind": "acquisition_scale",
    "rungs": [0.5, 0.7071067811865475, 1.0, 1.4142135623730951, 2.0],
    "tolerance": 0.08,
}

PROVENANCE = {
    "method": "synthetic truth for the validator",
    "evidence_uri": "tests/calibrate/test_correction_set.py",
}


def correction(**overrides):
    """Return a promoted gain correction, overridden slot by slot."""

    row = {
        "channel": "probe01",
        "kind": "gain",
        "status": "promoted",
        "value": 0.5,
        "validity": [{"pulse_start": 100, "pulse_end": 200}],
        "provenance": dict(PROVENANCE),
    }
    row.update(overrides)
    return row


def document(*corrections, **overrides):
    """Return a document carrying the given corrections."""

    row = {
        "machine": "synthetic",
        "diagnostic_system": "magnetics",
        "schema_version": SCHEMA_VERSION,
        "set_version": "1.0.0",
        "generated_by": "tests/calibrate/test_correction_set.py",
        "ladders": [dict(LADDER)],
        "corrections": list(corrections) or [correction()],
    }
    row.update(overrides)
    return row


def validated(*corrections, **overrides):
    """Return the parsed document, having run every structural check."""

    parsed = CorrectionSet.model_validate(document(*corrections, **overrides))
    validate_correction_set(parsed)
    return parsed


def refused(*corrections, **overrides):
    """Return the message the validator refused the document with."""

    with pytest.raises(CorrectionSetError) as error:
        validated(*corrections, **overrides)
    return str(error.value)


def test_minimal_document_validates():
    parsed = validated()
    assert parsed.corrections[0].value == 0.5


def test_schema_version_matches_the_authored_schema():
    schema = yaml.safe_load(SCHEMA_PATH.read_text())
    assert schema["version"] == SCHEMA_VERSION


def test_application_order_matches_the_schema_ranks():
    schema = yaml.safe_load(SCHEMA_PATH.read_text())
    declared = schema["enums"]["ApplicationStage"]["permissible_values"]
    ranked = sorted(declared, key=lambda name: declared[name]["rank"])
    assert [ApplicationStage(name) for name in ranked] == list(APPLICATION_ORDER)
    assert [declared[name]["rank"] for name in ranked] == list(
        range(1, len(ranked) + 1)
    )


def test_every_applicable_kind_has_one_stage():
    schema = yaml.safe_load(SCHEMA_PATH.read_text())
    kinds = set(schema["enums"]["CorrectionKind"]["permissible_values"])
    staged = {str(kind.value) for kind in KIND_STAGE}
    assert kinds - staged == {"exclusion", "quality"}
    assert sorted(KIND_STAGE.values(), key=APPLICATION_ORDER.index) == list(
        APPLICATION_ORDER
    )


def test_the_read_path_applies_corrections_in_the_declared_order():
    rows = [
        correction(kind="gain", value=1.1),
        correction(kind="convention", value=6.283185307179586),
        correction(kind="offset", value=-0.002, unit="T"),
        correction(kind="acquisition_scale", value=2.0, ladder="acquisition_range"),
        correction(kind="pair_state", value=0.5, state="single_member"),
        correction(kind="drift_rate", value=1e-05, unit="T/s"),
    ]
    parsed = validated(*rows)
    assert [stage(row) for row in applied(parsed)] == list(APPLICATION_ORDER)


def test_descriptive_kinds_never_reach_the_chain():
    rows = [
        correction(
            kind="quality",
            status="recorded",
            value=None,
            quality_status="dead",
            cause="no signal",
        ),
        correction(kind="exclusion", status="recorded", value=None, cause="no signal"),
    ]
    parsed = validated(*rows)
    assert list(applied(parsed)) == []


def test_overlapping_intervals_are_refused():
    rows = [
        correction(validity=[{"pulse_start": 100, "pulse_end": 200}]),
        correction(validity=[{"pulse_start": 200, "pulse_end": 300}]),
    ]
    assert "overlapping pulse intervals" in refused(*rows)


def test_an_open_interval_overlaps_a_bounded_one():
    rows = [correction(validity=[{}]), correction()]
    assert "overlapping" in refused(*rows)


def test_adjacent_intervals_are_allowed():
    rows = [
        correction(validity=[{"pulse_start": 100, "pulse_end": 200}]),
        correction(validity=[{"pulse_start": 201, "pulse_end": 300}]),
    ]
    assert len(validated(*rows).corrections) == 2


def test_a_superseded_record_may_cover_its_replacement():
    rows = [
        correction(value=0.4, status="superseded"),
        correction(value=0.5, status="promoted"),
    ]
    assert len(validated(*rows).corrections) == 2


def test_one_channel_may_carry_several_kinds_over_one_interval():
    rows = [
        correction(kind="gain", value=1.1),
        correction(kind="acquisition_scale", value=2.0, ladder="acquisition_range"),
    ]
    assert len(validated(*rows).corrections) == 2


def test_an_off_ladder_acquisition_scale_is_refused():
    row = correction(kind="acquisition_scale", value=0.585, ladder="acquisition_range")
    message = refused(row)
    assert "misses every rung" in message
    assert "0.08" in message


def test_a_quantised_kind_may_not_opt_out_of_its_ladder():
    row = correction(kind="acquisition_scale", value=0.585)
    assert "names no ladder" in refused(row)


def test_a_value_inside_the_tolerance_lands_on_its_rung():
    row = correction(kind="acquisition_scale", value=0.53, ladder="acquisition_range")
    assert validated(row).corrections[0].value == 0.53


def test_an_undeclared_ladder_is_refused():
    row = correction(kind="acquisition_scale", value=2.0, ladder="voltage_range")
    assert "which the set does not declare" in refused(row)


def test_a_ladder_quantising_another_kind_is_refused():
    row = correction(kind="gain", value=2.0, ladder="acquisition_range")
    assert "which quantises acquisition_scale" in refused(row)


def test_a_promoted_correction_without_a_value_is_refused():
    assert "carries no value" in refused(correction(value=None))


def test_a_refused_correction_carrying_a_value_is_refused():
    row = correction(status="refused", value=0.585, measured_value=0.585)
    assert "was never established" in refused(row)


def test_a_withheld_correction_keeps_its_measurement():
    row = correction(status="withheld", value=None, measured_value=0.9248)
    assert validated(row).corrections[0].measured_value == 0.9248


def test_a_record_carrying_nothing_at_all_is_refused():
    row = correction(status="recorded", value=None)
    assert "no value, no measurement and no candidates" in refused(row)


def test_a_multiplier_of_zero_is_refused():
    assert "erases the channel" in refused(correction(value=0.0))


def test_a_promoted_correction_must_cite_evidence():
    row = correction(provenance={"method": "a fit nobody can find"})
    assert "cites no evidence" in refused(row)


def test_a_correction_naming_no_target_is_refused():
    assert "no target" in refused(correction(channel=None))


def test_a_correction_naming_two_targets_is_refused():
    row = correction(channel_group="outboard_vertical")
    assert "both a channel and a group" in refused(row)


def test_a_correction_without_a_validity_interval_is_refused():
    assert "no validity interval" in refused(correction(validity=[]))


def test_an_interval_mixing_pulse_and_time_is_refused():
    row = correction(validity=[{"pulse_start": 100, "time_end": 0.5}])
    assert "bounded in both pulse and time" in refused(row)


def test_an_interval_running_backwards_is_refused():
    row = correction(validity=[{"pulse_start": 300, "pulse_end": 200}])
    assert "runs backwards" in refused(row)


def test_measured_pulses_outside_the_interval_are_refused():
    row = correction(
        validity=[{"pulse_start": 100, "pulse_end": 200, "measured_pulses": [150, 250]}]
    )
    assert "lie outside the interval" in refused(row)


def test_measured_pulses_must_be_a_sorted_distinct_run():
    row = correction(
        validity=[{"pulse_start": 100, "pulse_end": 200, "measured_pulses": [150, 120]}]
    )
    assert "sorted distinct run" in refused(row)


def test_a_quality_state_needs_a_status():
    row = correction(kind="quality", status="recorded", value=None, cause="odd")
    assert "no quality status" in refused(row)


def test_a_pair_state_needs_a_state():
    row = correction(kind="pair_state", status="recorded", value=0.5)
    assert "no pickup state" in refused(row)


def test_an_exclusion_needs_a_cause():
    row = correction(kind="exclusion", status="recorded", value=None)
    assert "without saying why" in refused(row)


def test_an_uncertainty_running_backwards_is_refused():
    row = correction(uncertainty={"lower": 0.6, "upper": 0.5})
    assert "runs backwards" in refused(row)


def test_a_document_from_another_schema_version_is_refused():
    assert "this reader implements" in refused(schema_version="0.9.0")


def test_an_unknown_kind_is_refused(tmp_path: Path):
    path = tmp_path / "magnetics.yaml"
    path.write_text(yaml.safe_dump(document(correction(kind="range_setting"))))
    with pytest.raises(CorrectionSetError) as error:
        load_correction_set(path)
    assert "does not match the schema" in str(error.value)


def test_an_unknown_slot_is_refused(tmp_path: Path):
    path = tmp_path / "magnetics.yaml"
    path.write_text(yaml.safe_dump(document(correction(scale_factor=0.5))))
    with pytest.raises(CorrectionSetError):
        load_correction_set(path)


def test_a_missing_document_is_raised_rather_than_read_as_empty():
    with pytest.raises(CorrectionSetError) as error:
        read_correction_set("no-such-machine", "magnetics")
    assert "look identical to a consumer" in str(error.value)


def test_a_document_round_trips_through_yaml(tmp_path: Path):
    path = tmp_path / "magnetics.yaml"
    path.write_text(yaml.safe_dump(document()))
    assert load_correction_set(path).corrections[0].value == 0.5
