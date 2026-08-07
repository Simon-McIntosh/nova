"""Whether the MAST document says what the calibration record measured.

The document is a transcription, and the failure mode of a transcription is a number
that drifted from its source while still looking plausible.  So the acquisition
blocks are compared row by row against the table they were mined from -- every span,
every measured pulse, every rung -- rather than being spot-checked, and each promoted
gain is checked to be the four-decimal constant the read path divides by sitting
beside the unrounded ratio the same route measured.

The rest of the cases pin the shapes that made a schema necessary in the first place.
A channel that steps mid-life carries one record per era and not one average.  A
channel that flips faster than any era resolves carries its two observed states as
candidates and no value at all, because the mean of them describes no pulse.  A block
whose step is not a rung is refused rather than rounded.  A channel that is an
outlier no gain explains carries a quality state, which is not the same as being
excluded.  And a refusal is recorded with its reason, so a consumer can tell an
unpromoted channel from an unmeasured one.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from nova.calibrate.correction_model import ApplicationStage
from nova.calibrate.correction_set import (
    APPLICATION_ORDER,
    CORRECTION_ROOT,
    applied,
    read_correction_set,
    stage,
)

BLOCK_TABLE = Path(__file__).parent / "data" / "banked_block_scales.json"
"""The table the acquisition corrections were mined from, frozen beside the tests.

It served the read path until the document replaced it, and is kept because a
transcription can only be checked against its source.  That the two still describe
the same reads is the subject of ``test_read_path_equivalence``.
"""

PROMOTED_GAINS = {
    "obr17": 0.5011,
    "obv04": 0.8571,
    "obr05": 1.1043,
    "obv05": 0.9449,
    "ccbv35": 0.9474,
}


@pytest.fixture(scope="module")
def instance():
    """Return the validated MAST magnetics document."""

    return read_correction_set("mast", "magnetics")


def select(document, kind, channel=None):
    """Return the document's corrections of one kind, optionally one channel."""

    return [
        row
        for row in document.corrections
        if row.kind == kind and (channel is None or row.channel == channel)
    ]


def only(document, kind, channel):
    """Return the one correction of a kind on a channel."""

    rows = select(document, kind, channel)
    assert len(rows) == 1, f"{channel} carries {len(rows)} {kind} corrections"
    return rows[0]


def test_the_document_lives_where_its_scope_says():
    assert (CORRECTION_ROOT / "mast" / "magnetics.yaml").exists()


def test_the_set_is_versioned_semantically(instance):
    major, minor, patch = instance.set_version.split(".")
    assert all(part.isdigit() for part in (major, minor, patch))
    assert instance.schema_version == "1.0.0"


def test_every_correction_cites_its_evidence(instance):
    uncited = [
        row.channel or row.channel_group
        for row in instance.corrections
        if not row.provenance.evidence_uri
    ]
    assert uncited == []


def test_the_document_covers_every_measured_kind(instance):
    counts = {}
    for row in instance.corrections:
        counts[row.kind] = counts.get(row.kind, 0) + 1
    assert counts == {
        "gain": 6,
        "acquisition_scale": 113,
        "pair_state": 3,
        "quality": 3,
        "exclusion": 2,
        "convention": 1,
    }


def test_the_read_path_applies_the_gains_the_rungs_and_the_convention(instance):
    rows = list(applied(instance))
    assert len(rows) == 114
    ranks = [APPLICATION_ORDER.index(stage(row)) for row in rows]
    assert ranks == sorted(ranks)
    assert {stage(row) for row in rows} == {
        ApplicationStage.acquisition_scale,
        ApplicationStage.gain,
        ApplicationStage.convention,
    }


# --------------------------------------------------------------------------
# the acquisition block table, row by row
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def blocks():
    """Return the block table the acquisition corrections were mined from."""

    return json.loads(BLOCK_TABLE.read_text())["blocks"]


def test_every_block_is_transcribed_exactly(instance, blocks):
    rows = select(instance, "acquisition_scale")
    assert len(rows) == len(blocks)
    keyed = {(row.channel, row.validity[0].pulse_start): row for row in rows}
    assert len(keyed) == len(rows)
    for block in blocks:
        row = keyed[(block["channel"], block["first_shot"])]
        interval = row.validity[0]
        assert interval.pulse_end == block["last_shot"]
        assert list(interval.measured_pulses) == block["shots"]
        assert row.measured_value == block["scale"]
        assert row.value == block["rung"]


def test_an_off_ladder_block_is_refused_rather_than_rounded(instance, blocks):
    off_ladder = {block["channel"] for block in blocks if block["rung"] is None}
    refused = {
        row.channel
        for row in select(instance, "acquisition_scale")
        if row.status == "refused"
    }
    assert refused == off_ladder
    assert len(refused) == 5
    for row in select(instance, "acquisition_scale"):
        if row.status == "refused":
            assert row.value is None
            assert row.measured_value is not None


def test_the_ladder_is_declared_with_its_hypothesis(instance):
    ladder = {row.name: row for row in instance.ladders}["acquisition_range"]
    assert ladder.kind == "acquisition_scale"
    assert ladder.rungs == pytest.approx(
        [0.25, 0.5, 1.0 / math.sqrt(2.0), 1.0, math.sqrt(2.0), 2.0, 4.0]
    )
    assert ladder.tolerance == 0.08
    for row in select(instance, "acquisition_scale"):
        assert row.ladder == "acquisition_range"


# --------------------------------------------------------------------------
# the promoted and withheld channel scales
# --------------------------------------------------------------------------


def test_each_promoted_gain_is_the_constant_the_read_path_divides_by(instance, blocks):
    scales = {block["channel"]: block["scale"] for block in blocks}
    for channel, value in PROMOTED_GAINS.items():
        row = only(instance, "gain", channel)
        assert row.status == "promoted"
        assert row.value == value
        assert row.measured_value == scales[channel]
        assert round(row.measured_value, 4) == value


def test_a_promoted_gain_holds_for_the_machines_life(instance):
    interval = only(instance, "gain", "obr17").validity[0]
    assert interval.pulse_start is None and interval.pulse_end is None
    assert len(interval.measured_pulses) == 23


def test_the_lifetime_half_carries_its_independent_routes(instance):
    row = only(instance, "gain", "obr17")
    assert row.uncertainty.lower == 0.5011 and row.uncertainty.upper == 0.5037
    corroborations = row.provenance.corroborations
    assert len(corroborations) >= 3
    values = [entry.value for entry in corroborations if entry.value is not None]
    assert 0.5023 in values
    assert 1.1238e-07 in values
    spans = [entry.uncertainty for entry in corroborations if entry.uncertainty]
    assert (0.488, 0.514) in [(span.lower, span.upper) for span in spans]


def test_the_withheld_scale_is_recorded_with_the_reason_it_was_refused(instance):
    row = only(instance, "gain", "obv11")
    assert row.status == "withheld"
    assert row.value is None
    assert row.measured_value == pytest.approx(0.924757, abs=1e-06)
    assert (row.uncertainty.lower, row.uncertainty.upper) == (0.869, 1.065)
    assert "7.4" in row.cause


# --------------------------------------------------------------------------
# the pair-failure channel class
# --------------------------------------------------------------------------


def test_a_mid_life_step_is_two_eras_and_not_one_average(instance, blocks):
    early, late = sorted(
        select(instance, "pair_state", "obv03"),
        key=lambda row: row.validity[0].pulse_start,
    )
    assert early.state == "both_members" and early.value == 1.01
    assert late.state == "single_member" and late.value == 0.586
    assert early.validity[0].pulse_end < late.validity[0].pulse_start
    measured = {
        (block["channel"], block["first_shot"]): block["scale"] for block in blocks
    }
    step = late.value / early.value
    independent = (
        measured[("obv03", late.validity[0].pulse_start)]
        / measured[("obv03", early.validity[0].pulse_start)]
    )
    assert step == pytest.approx(independent, rel=0.01)


def test_the_bracket_between_the_eras_is_named_rather_than_a_switch_pulse(instance):
    late = max(
        select(instance, "pair_state", "obv03"),
        key=lambda row: row.validity[0].pulse_start,
    )
    assert "bracketed between pulses" in late.validity[0].notes


def test_a_channel_that_flips_carries_candidates_and_no_value(instance):
    row = only(instance, "pair_state", "obr05")
    assert row.state == "indeterminate"
    assert row.value is None
    assert sorted(row.candidate_values) == [0.5, 1.25]
    assert row.validity[0].pulse_start is None


def test_the_flipping_channels_static_gain_is_flagged_where_it_is_carried(instance):
    assert only(instance, "gain", "obr05").value == 1.1043
    assert only(instance, "quality", "obr05").quality_status == "suspect"


# --------------------------------------------------------------------------
# exclusions and quality states
# --------------------------------------------------------------------------


def test_the_dead_channel_is_excluded_and_described(instance, blocks):
    assert only(instance, "exclusion", "obv10").cause.startswith("dead")
    assert only(instance, "quality", "obv10").quality_status == "dead"
    assert "obv10" not in {block["channel"] for block in blocks}


def test_the_unstable_channel_is_excluded_beside_its_withheld_measurement(instance):
    assert only(instance, "exclusion", "obv11").status == "recorded"
    assert only(instance, "gain", "obv11").status == "withheld"


def test_the_outlier_is_suspect_rather_than_excluded(instance):
    row = only(instance, "quality", "ccbv22")
    assert row.quality_status == "suspect"
    assert row.measured_value == -2.44e-07
    assert row.unit == "T/At"
    assert select(instance, "exclusion", "ccbv22") == []


# --------------------------------------------------------------------------
# the store-to-solver convention
# --------------------------------------------------------------------------


def test_the_flux_convention_is_a_measured_two_pi_on_the_loop_group(instance):
    rows = select(instance, "convention")
    assert len(rows) == 1
    row = rows[0]
    assert row.channel is None and row.channel_group == "flux_loop"
    assert row.value == math.tau
    assert row.status == "promoted"
    assert row.provenance.corroborations[0].value == math.tau
