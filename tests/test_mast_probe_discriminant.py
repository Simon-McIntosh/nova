"""What the probe discriminant refuses, and the case each refusal was written for.

The criterion's whole value is that it was fixed before the fit, so the tests here
are about the decision rule and not about any measurement: each hypothesis is
handed statistics that exhibit it exactly, and the rule has to return the cause
that produced them.  Statistics are built by hand rather than fitted, because a
rule tested only against real data cannot be shown refusing the case it exists to
refuse.

The two orderings that carry the criterion are tested directly: excitation
selectivity outranks everything, and a stable coefficient that leaves the
residual an order of magnitude above the channel's noise floor does not earn a
probe-side verdict.
"""

from __future__ import annotations

import json

import pytest

from nova.imas.mast_probe_discriminant import (
    FLOOR_MULTIPLE,
    HYPOTHESES,
    MAXIMUM_TILT,
    MINIMUM_FAMILIES,
    SPREAD_SIGNIFICANCE,
    DiscriminantError,
    DiscriminantStatistics,
    PreRegistration,
    ProbeVerdict,
    adjudicate,
    promotable,
)

FLOOR = 5.0e-5
"""Tesla of quiescent scatter the synthetic channel is given."""


def statistics(**overrides) -> DiscriminantStatistics:
    """Build statistics that on their own say nothing, then push one term."""

    defaults = dict(
        channel="obv06",
        family_count=5,
        gain=1.0,
        gain_standard_error=0.002,
        gain_spread=0.001,
        near_coil_gain=1.0,
        distant_coil_gain=1.0,
        tilt=0.0,
        tilt_standard_error=0.001,
        tilt_variance_removed=0.0,
        partner_channel="obr06",
        partner_excess_share=0.0,
        rigid_residual=FLOOR,
        noise_floor=FLOOR,
    )
    defaults.update(overrides)
    return DiscriminantStatistics(**defaults)


def test_a_gain_that_depends_on_the_driving_coil_is_geometry() -> None:
    """Excitation-selective gain is the field-shape signature and outranks the rest."""

    selective = statistics(
        gain=1.12,
        gain_spread=0.20,
        gain_standard_error=0.01,
        near_coil_gain=1.19,
        distant_coil_gain=0.99,
    )
    assert selective.excitation_selective
    assert selective.near_field_contrast == pytest.approx(0.20)
    assert adjudicate(selective) is ProbeVerdict.FIELD_SHAPE
    assert not promotable(ProbeVerdict.FIELD_SHAPE)


def test_selectivity_outranks_a_clean_rigid_fit() -> None:
    """A residual at the floor does not rescue a gain that moved with the coil.

    A rigid fit can absorb an excitation-selective error whenever one family
    dominates the sample count, so the ordering matters: the selectivity test has
    to run first or a geometry error is promoted as a probe calibration.
    """

    disguised = statistics(
        gain_spread=0.20,
        gain_standard_error=0.01,
        gain=1.10,
        rigid_residual=FLOOR,
        tilt=0.05,
        tilt_standard_error=0.001,
        tilt_variance_removed=0.9,
    )
    assert disguised.rigid_fit_reaches_floor
    assert disguised.tilt_identified
    assert adjudicate(disguised) is ProbeVerdict.FIELD_SHAPE


def test_one_gain_across_families_is_a_probe_area_error() -> None:
    """An excitation-invariant scale that reaches the floor is the probe's own."""

    area = statistics(gain=1.08, gain_standard_error=0.004, gain_spread=0.005)
    assert not area.excitation_selective
    assert adjudicate(area) is ProbeVerdict.CALIBRATION_GAIN
    assert promotable(ProbeVerdict.CALIBRATION_GAIN)


def test_a_residual_explained_by_the_co_located_channel_is_a_tilt() -> None:
    """The angle wins when it removes the variance and the gain stays at one."""

    tilted = statistics(
        tilt=0.031,
        tilt_standard_error=0.002,
        tilt_variance_removed=0.86,
    )
    assert tilted.tilt_identified
    assert adjudicate(tilted) is ProbeVerdict.CALIBRATION_TILT
    assert promotable(ProbeVerdict.CALIBRATION_TILT)


def test_an_angle_too_large_to_be_a_mounting_error_is_refused() -> None:
    """Past the mounting tolerance the angle is the fit absorbing something else."""

    hinged = statistics(
        tilt=MAXIMUM_TILT * 1.5,
        tilt_standard_error=0.002,
        tilt_variance_removed=0.95,
    )
    assert not hinged.tilt_admissible
    assert not hinged.tilt_identified
    assert adjudicate(hinged) is ProbeVerdict.INSEPARABLE
    assert not promotable(ProbeVerdict.INSEPARABLE)


def test_a_stable_coefficient_above_the_noise_floor_explains_nothing() -> None:
    """A probe-side verdict has to reach the floor, not merely be reproducible."""

    unexplained = statistics(
        gain=1.30,
        gain_standard_error=0.003,
        gain_spread=0.004,
        rigid_residual=FLOOR * (FLOOR_MULTIPLE + 1.0),
    )
    assert not unexplained.rigid_fit_reaches_floor
    assert adjudicate(unexplained) is ProbeVerdict.INSEPARABLE


def test_a_gain_indistinguishable_from_one_promotes_nothing() -> None:
    """A probe the data finds correct stays as it is rather than gaining a record."""

    clean = statistics(gain=1.0005, gain_standard_error=0.002)
    assert adjudicate(clean) is ProbeVerdict.INSEPARABLE


def test_too_few_families_cannot_be_judged_either_way() -> None:
    """Without a spread there is no test, and the probe is reported untested."""

    thin = statistics(family_count=MINIMUM_FAMILIES - 1, gain=1.2, gain_spread=0.4)
    assert not thin.excitation_selective
    assert adjudicate(thin) is ProbeVerdict.NOT_TESTED
    assert not promotable(ProbeVerdict.NOT_TESTED)


def test_spread_significance_is_the_boundary_it_claims_to_be() -> None:
    """The selectivity test turns over exactly at the declared multiple."""

    error = 0.01
    below = statistics(
        gain_standard_error=error,
        gain_spread=SPREAD_SIGNIFICANCE * error,
    )
    above = statistics(
        gain_standard_error=error,
        gain_spread=SPREAD_SIGNIFICANCE * error * 1.001,
    )
    assert not below.excitation_selective
    assert above.excitation_selective


def test_a_channel_without_a_measured_floor_cannot_be_judged() -> None:
    """The floor is a measurement, so a missing one is an error and not a default."""

    with pytest.raises(DiscriminantError, match="noise floor"):
        adjudicate(statistics(noise_floor=0.0))


def test_statistics_reject_negative_scatter() -> None:
    """A standard error is a magnitude and a negative one is a coding fault."""

    with pytest.raises(DiscriminantError, match="gain standard error"):
        adjudicate(statistics(gain_standard_error=-1.0e-3))


def test_the_pre_registration_detects_a_moved_threshold() -> None:
    """A run that applied different thresholds cannot claim this criterion."""

    registration = PreRegistration()
    applied = registration.as_dict()
    assert registration.agrees_with(applied)
    applied["spread_significance"] = SPREAD_SIGNIFICANCE / 2.0
    assert not registration.agrees_with(applied)


def test_the_pre_registration_serializes_byte_stably() -> None:
    """The criterion is citable, so it has to round-trip through JSON unchanged."""

    registration = PreRegistration()
    text = json.dumps(registration.as_dict(), sort_keys=True)
    assert json.loads(text) == registration.as_dict()


def test_every_verdict_names_a_hypothesis_or_declines_to() -> None:
    """The three causes are documented; the two non-verdicts deliberately are not."""

    assert set(HYPOTHESES) == {"calibration_gain", "calibration_tilt", "field_shape"}
    for verdict in ProbeVerdict:
        described = str(verdict) in HYPOTHESES
        undecided = verdict in (ProbeVerdict.INSEPARABLE, ProbeVerdict.NOT_TESTED)
        assert described != undecided


def test_statistics_are_canonically_serializable() -> None:
    """Every derived flag reaches the record, so a verdict can be recomputed."""

    row = statistics(gain=1.08, gain_standard_error=0.004).as_dict()
    for key in (
        "excitation_selective",
        "near_field_contrast",
        "rigid_fit_reaches_floor",
        "tilt_identified",
    ):
        assert key in row
    assert json.loads(json.dumps(row, sort_keys=True)) == row
