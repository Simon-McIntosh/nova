"""Whether an injected offset, walk and non-closure come back out of the fits.

Every case manufactures a channel whose instrument terms were decided before the
fit ran and asks for them back.  That is the only assertion an instrument term can
be held to: a drift rate has no independent reference on real data, so recovery of
injected truth to a stated tolerance is what stands in for one.

The closure cases carry the weight.  A channel that returns to its extrapolated
baseline after the pulse says its integrator accumulated nothing it did not shed;
one that does not has a defect, and the defect is the part of a flux measurement
that no per-window offset removes.  Manufacturing the step directly is what makes
the defect checkable against a number rather than against a plot.

Two of the cases exist because they separate a defect from its look-alikes.  A walk
with real curvature is not a non-closure -- a linear extrapolation of it misses by
half the curvature times the gap squared, which grows as the pulse lengthens and is
exactly the signature a long-pulse consumer would misread as accumulated error.  And
two windows fitted about different origins are not comparable at all: the linear
part of a quadratic depends on where it was taken, so differencing them measures the
origins as much as the instrument.
"""

from __future__ import annotations

import datetime
import math

import numpy as np
import pytest

from nova.calibrate.correction_model import (
    CorrectionKind,
    CorrectionSet,
    CorrectionStatus,
    Provenance,
)
from nova.calibrate.correction_set import CorrectionSetError, validate_correction_set
from nova.calibrate.corrections import apply_corrections
from nova.calibrate.instrument import (
    InstrumentError,
    InstrumentTerms,
    closure_defect,
    fit_instrument_terms,
    instrument_corrections,
    pool_instrument_terms,
)
from nova.calibrate.windows import WindowKind, classify_pulse

SAMPLE_RATE = 5.0e3
"""Samples per second, the rate the archive digitised its magnetics at."""

DRIVE_BAR = 1.0e2
"""Amperes a conductor must carry before its field is worth classifying."""

FLOOR = 1.0e-5
"""Tesla of scatter a quiet probe channel shows about its own walk."""


def quiet_pulse(
    *,
    offset: float = 1.3e-3,
    rate: float = 2.0e-5,
    curvature: float = 0.0,
    step: float = 0.0,
    floor: float = FLOOR,
    seed: int = 3,
):
    """Return a record whose channel carries a known walk through a known pulse.

    The record is five seconds long because that is what it takes to measure a
    drift: a walk of 2e-5 T/s moves the channel by 2e-6 T across a tenth of a
    second, which is a fifth of the floor, and no fit recovers a slope from a
    window the slope does not visibly cross.  The conductor is driven between
    1.0 s and 1.5 s and the channel reads a large field over that span, so the two
    quiet windows are separated by an interval no instrument term is measurable in.

    ``step`` is added to the channel from the moment the drive ends, which is a
    non-closure by construction: the integrator came back to a different zero than
    the one it left.
    """

    time = np.arange(int(5.0 * SAMPLE_RATE), dtype=float) / SAMPLE_RATE
    drive = np.zeros(time.size)
    driving = (time >= 1.0) & (time < 1.5)
    drive[driving] = 3.0e3
    walk = offset + rate * time + 0.5 * curvature * time**2
    generator = np.random.default_rng(seed)
    signal = walk + generator.normal(0.0, floor, time.size)
    signal[driving] += 0.4
    signal[time >= 1.5] += step
    return time, drive, signal


def quiet_windows(time, drive, **kwargs):
    """Return the leading and trailing instrument-quiet windows of a record."""

    timeline = classify_pulse(
        time, drive, drive_threshold=DRIVE_BAR, minimum_samples=32, **kwargs
    )
    return timeline.leading_quiet, timeline.trailing_quiet


def test_an_injected_offset_and_drift_rate_come_back_from_a_quiet_window():
    """The rate is asked for where the window determines it: at its own centre.

    A quadratic's slope at an instant half a window-width outside the window carries
    the curvature's uncertainty over that lever arm, and the curvature is the term a
    short window pins worst.  Reporting the rate at a distant reference is still what
    the read path needs -- it removes a ramp counted from the record's origin -- but
    it is not the number a recovery test can hold the fit to.
    """

    time, drive, signal = quiet_pulse(offset=1.3e-3, rate=2.0e-5)
    lead, _ = quiet_windows(time, drive)
    terms = fit_instrument_terms(time, signal, lead, channel="p01", reference_time=0.0)
    assert terms.offset == pytest.approx(1.3e-3, abs=1.0e-5)
    assert terms.slope(terms.centre) == pytest.approx(2.0e-5, rel=0.10)
    assert terms.scatter == pytest.approx(FLOOR, rel=0.15)


def test_a_second_order_walk_is_recovered_rather_than_averaged_into_the_rate():
    time, drive, signal = quiet_pulse(rate=0.0, curvature=1.0e-4, floor=1.0e-9)
    lead, _ = quiet_windows(time, drive)
    terms = fit_instrument_terms(time, signal, lead, reference_time=0.0)
    assert terms.drift_curvature == pytest.approx(1.0e-4, rel=1.0e-3)
    assert terms.drift_rate == pytest.approx(0.0, abs=1.0e-6)


def test_the_offset_is_the_level_at_the_reference_instant_and_not_at_the_window():
    time, drive, signal = quiet_pulse(offset=1.3e-3, rate=2.0e-5, floor=1.0e-9)
    _, tail = quiet_windows(time, drive)
    at_origin = fit_instrument_terms(time, signal, tail, reference_time=0.0)
    at_window = fit_instrument_terms(time, signal, tail, reference_time=tail.start)
    assert at_origin.offset == pytest.approx(1.3e-3, abs=1.0e-7)
    assert at_window.offset == pytest.approx(1.3e-3 + 2.0e-5 * tail.start, abs=1.0e-7)
    assert at_window.drift_curvature == pytest.approx(
        at_origin.drift_curvature, rel=1.0e-12
    )


def test_a_window_under_the_sample_floor_carries_no_measurable_walk():
    time, drive, signal = quiet_pulse()
    lead, _ = quiet_windows(time, drive)
    with pytest.raises(InstrumentError, match="samples"):
        fit_instrument_terms(time, signal, lead, minimum_samples=10_000)


def test_samples_that_are_not_finite_are_skipped_rather_than_poisoning_the_fit():
    time, drive, signal = quiet_pulse(offset=1.3e-3, rate=2.0e-5, floor=1.0e-9)
    signal[10:20] = np.nan
    lead, _ = quiet_windows(time, drive)
    terms = fit_instrument_terms(time, signal, lead, reference_time=0.0)
    assert terms.offset == pytest.approx(1.3e-3, abs=1.0e-7)
    assert terms.sample_count == lead.sample_count - 10


def test_a_walk_that_continues_through_the_pulse_closes():
    time, drive, signal = quiet_pulse(rate=2.0e-5, step=0.0)
    lead, tail = quiet_windows(time, drive)
    defect = closure_defect(
        fit_instrument_terms(time, signal, lead, channel="p01", reference_time=0.0),
        fit_instrument_terms(time, signal, tail, channel="p01", reference_time=0.0),
    )
    assert defect.closes
    assert abs(defect.defect) < 5.0 * FLOOR


def test_an_integrator_returning_to_a_different_zero_does_not_close():
    injected = 5.0e-4
    time, drive, signal = quiet_pulse(rate=2.0e-5, step=injected)
    lead, tail = quiet_windows(time, drive)
    defect = closure_defect(
        fit_instrument_terms(time, signal, lead, channel="p01", reference_time=0.0),
        fit_instrument_terms(time, signal, tail, channel="p01", reference_time=0.0),
    )
    assert not defect.closes
    assert defect.defect == pytest.approx(injected, rel=0.05)
    assert defect.defect_in_scatter > 20.0


def test_a_curved_walk_is_not_a_non_closure_however_large_the_linear_defect():
    """The look-alike a long pulse turns into a false accumulated error.

    A linear extrapolation of a genuinely curved walk misses by half the curvature
    times the gap squared.  Carrying both extrapolations is what lets a consumer
    see that the miss is the walk's own shape rather than something the integrator
    failed to shed.
    """

    time, drive, signal = quiet_pulse(rate=0.0, curvature=1.0e-4, floor=1.0e-9)
    lead, tail = quiet_windows(time, drive)
    defect = closure_defect(
        fit_instrument_terms(time, signal, lead, reference_time=0.0),
        fit_instrument_terms(time, signal, tail, reference_time=0.0),
    )
    assert abs(defect.defect) > 1.0e-4
    assert abs(defect.curved_defect) < 0.01 * abs(defect.defect)


def test_two_windows_fitted_about_different_origins_cannot_be_differenced():
    time, drive, signal = quiet_pulse()
    lead, tail = quiet_windows(time, drive)
    with pytest.raises(InstrumentError, match="reference"):
        closure_defect(
            fit_instrument_terms(time, signal, lead, reference_time=0.0),
            fit_instrument_terms(time, signal, tail, reference_time=tail.start),
        )


def flat_terms(offset: float, start: float, stop: float) -> InstrumentTerms:
    """Return terms for a channel that reported one level and no noise at all."""

    return InstrumentTerms(
        channel="p01",
        offset=offset,
        drift_rate=0.0,
        drift_curvature=0.0,
        reference_time=0.0,
        scatter=0.0,
        start=start,
        stop=stop,
        sample_count=500,
    )


def test_a_channel_with_no_scatter_makes_any_defect_infinitely_significant():
    """The scale a defect is scored against can be zero, and dividing is not it.

    A fit over manufactured samples returns a scatter of order the arithmetic's own
    precision rather than exactly zero, so this case is built from terms directly.
    A channel with no floor has no excursion that its noise explains, which makes
    any defect at all significant rather than undefined.
    """

    defect = closure_defect(flat_terms(0.0, 0.0, 1.0), flat_terms(1.0e-4, 1.5, 5.0))
    assert math.isinf(defect.defect_in_scatter)
    assert not defect.closes
    assert closure_defect(flat_terms(0.0, 0.0, 1.0), flat_terms(0.0, 1.5, 5.0)).closes


def test_a_pulse_that_never_goes_quiet_after_termination_has_no_closure_to_test():
    time = np.arange(int(0.5 * SAMPLE_RATE), dtype=float) / SAMPLE_RATE
    drive = np.where(time >= 0.10, 3.0e3, 0.0)
    timeline = classify_pulse(time, drive, drive_threshold=DRIVE_BAR)
    assert timeline.trailing_quiet is None
    assert timeline.leading_quiet is not None


def test_pooling_takes_the_scatter_across_pulses_and_not_across_samples():
    offsets = [1.0e-3, 1.1e-3, 0.9e-3, 1.2e-3]
    terms = [
        InstrumentTerms(
            channel="p01",
            pulse=index,
            offset=value,
            drift_rate=2.0e-5,
            drift_curvature=0.0,
            reference_time=0.0,
            scatter=FLOOR,
            start=0.0,
            stop=0.1,
            sample_count=500,
        )
        for index, value in enumerate(offsets)
    ]
    pooled = pool_instrument_terms(terms)
    assert len(pooled) == 1
    assert pooled[0].offset == pytest.approx(float(np.mean(offsets)))
    assert pooled[0].offset_error == pytest.approx(
        float(np.std(offsets, ddof=1) / math.sqrt(len(offsets)))
    )
    assert pooled[0].pulses == (0, 1, 2, 3)
    assert pooled[0].identified


def test_one_pulse_pools_to_a_value_with_no_error_bar():
    terms = InstrumentTerms(
        channel="p01",
        pulse=7,
        offset=1.0e-3,
        drift_rate=0.0,
        drift_curvature=0.0,
        reference_time=0.0,
        scatter=FLOOR,
        start=0.0,
        stop=0.1,
        sample_count=500,
    )
    pooled = pool_instrument_terms([terms])[0]
    assert pooled.offset == 1.0e-3
    assert math.isinf(pooled.offset_error)
    assert not pooled.identified


def evidenced(**kwargs) -> Provenance:
    """Return a provenance a promoted correction can be validated against."""

    return Provenance(
        method="quadratic walk fitted over the pre-pulse instrument-quiet window",
        evidence_uri="tests/calibrate/test_instrument.py",
        fitted_at=datetime.date(2026, 8, 8),
        fitted_by="nova.calibrate.instrument",
        statement="offset and drift rate measured where the field is zero",
        **kwargs,
    )


def measured_pulses(count: int = 3) -> list[InstrumentTerms]:
    """Return one channel's terms measured on several pulses."""

    return [
        InstrumentTerms(
            channel="p01",
            pulse=100 + index,
            offset=1.3e-3,
            drift_rate=2.0e-5,
            drift_curvature=4.0e-4,
            reference_time=0.0,
            scatter=FLOOR,
            start=0.0,
            stop=0.1,
            sample_count=500,
        )
        for index in range(count)
    ]


def document(corrections) -> CorrectionSet:
    """Wrap corrections in the smallest set the reader accepts."""

    return CorrectionSet(
        machine="synthetic",
        diagnostic_system="magnetics",
        schema_version="1.0.0",
        set_version="1.0.0",
        generated_by="tests/calibrate/test_instrument.py",
        corrections=list(corrections),
    )


def test_the_terms_emit_as_offset_and_drift_rate_records_the_reader_accepts():
    pooled = pool_instrument_terms(measured_pulses())
    records = instrument_corrections(pooled, provenance=evidenced(), unit="T")
    assert [row.kind for row in records] == [
        CorrectionKind.offset,
        CorrectionKind.drift_rate,
    ]
    assert records[0].value == pytest.approx(1.3e-3)
    assert records[1].value == pytest.approx(2.0e-5)
    assert records[0].validity[0].measured_pulses == [100, 101, 102]
    assert records[0].validity[0].pulse_start == 100
    assert records[0].validity[0].pulse_end == 102
    validate_correction_set(document(records))


def test_the_curvature_is_carried_on_the_drift_record_the_schema_has_no_slot_for():
    pooled = pool_instrument_terms(measured_pulses())
    records = instrument_corrections(pooled, provenance=evidenced(), unit="T")
    assert "0.0004" in records[1].notes
    assert "T/s2" in records[1].notes


def test_terms_are_recorded_rather_than_promoted_unless_the_caller_says_otherwise():
    pooled = pool_instrument_terms(measured_pulses())
    records = instrument_corrections(pooled, provenance=evidenced(), unit="T")
    assert all(
        CorrectionStatus(row.status) is CorrectionStatus.recorded for row in records
    )


def test_a_promoted_term_citing_no_evidence_is_refused_by_the_reader():
    pooled = pool_instrument_terms(measured_pulses())
    records = instrument_corrections(
        pooled,
        provenance=Provenance(method="fitted over a quiet window"),
        unit="T",
        status=CorrectionStatus.promoted,
    )
    with pytest.raises(CorrectionSetError, match="cites no evidence"):
        validate_correction_set(document(records))


def test_the_emitted_records_remove_the_walk_they_were_measured_from():
    """The round trip the whole extraction exists for.

    Fit the terms out of a quiet window, emit them as records, and let the read
    path apply them: what comes back is a channel whose quiet intervals sit at
    zero.  Anything short of this leaves the terms as numbers in a report.
    """

    time, drive, signal = quiet_pulse(offset=1.3e-3, rate=2.0e-5, floor=1.0e-9)
    lead, tail = quiet_windows(time, drive)
    terms = fit_instrument_terms(
        time, signal, lead, channel="p01", pulse=100, reference_time=0.0
    )
    pooled = pool_instrument_terms([terms, terms, terms])
    records = instrument_corrections(
        pooled,
        provenance=evidenced(),
        unit="T",
        status=CorrectionStatus.promoted,
        pulse_start=1,
        pulse_end=1000,
    )
    corrected, chain = apply_corrections(
        document(records),
        "p01",
        signal,
        pulse=100,
        time=time,
        reference_time=0.0,
    )
    assert len(chain.steps) == 2
    for window in (lead, tail):
        assert float(np.abs(corrected[window.indices]).max()) < 1.0e-7


def test_a_timeline_hands_its_quiet_windows_to_the_fit_without_a_channel_map():
    time, drive, signal = quiet_pulse()
    timeline = classify_pulse(
        time, drive, drive_threshold=DRIVE_BAR, minimum_samples=32
    )
    fitted = [
        fit_instrument_terms(time, signal, window, channel="p01", reference_time=0.0)
        for window in timeline.of_kind(WindowKind.quiet)
    ]
    assert len(fitted) == 2
    assert all(row.channel == "p01" for row in fitted)
