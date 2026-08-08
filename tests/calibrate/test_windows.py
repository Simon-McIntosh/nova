"""Whether a manufactured pulse timeline comes back out of the classifier.

Every case here builds a record whose intervals were decided before the classifier
ran, and asks for those intervals back.  The manufactured drive steps rather than
ramps so that a window boundary lands on a sample whose time is known exactly, and
the assertions can be on times rather than on tolerances around them.

Four of the cases exist because each guards a way the classification can be right
about every sample and still wrong about the pulse.  A coil sitting at its noise
floor is not a drive, and reading it as one would put the whole record in the driven
class on a shot that energised nothing.  A window that follows a disturbance still
carries the passive currents that disturbance induced, so its first samples are not
the instrument alone however quiet the drives have gone.  A gap in the record is not
a quiet interval -- nothing was measured there, and merging across it would assert a
continuity that was never observed.  And a missing plasma channel is either a vacuum
shot or an unrecorded signal, which are opposite statements about the same absence.
"""

from __future__ import annotations

import numpy as np
import pytest

from nova.calibrate.windows import (
    PulseTimeline,
    WindowError,
    WindowKind,
    classify_pulse,
)

SAMPLE_RATE = 5.0e3
"""Samples per second, the rate the archive digitised its magnetics at."""

DRIVE_BAR = 1.0e2
"""Amperes a conductor must carry before its field is worth classifying."""

PLASMA_BAR = 1.0e4
"""Amperes of plasma current above which the pulse is no longer vacuum."""


def stepped_record(
    *,
    quiet_lead: float = 0.10,
    driven: float = 0.05,
    plasma: float = 0.20,
    quiet_tail: float = 0.15,
    drive_current: float = 3.0e3,
    plasma_current: float = 6.0e5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return one pulse's time base, drive column and plasma column.

    The shape is the archive's: nothing energised, then the conductors driven with
    no plasma, then a plasma interval, then everything back to zero.  Each span is
    given in seconds and the drive steps at the boundary, so every window edge sits
    on a sample whose time the caller already knows.
    """

    spans = (quiet_lead, driven, plasma, quiet_tail)
    counts = [int(round(span * SAMPLE_RATE)) for span in spans]
    time = np.arange(sum(counts), dtype=float) / SAMPLE_RATE
    drive = np.zeros(time.size)
    current = np.zeros(time.size)
    start = counts[0]
    stop = start + counts[1] + counts[2]
    drive[start:stop] = drive_current
    current[start + counts[1] : stop] = plasma_current
    return time, drive, current


def classify(time, drive, plasma=None, **kwargs) -> PulseTimeline:
    """Classify a manufactured record at the bars the tests share."""

    return classify_pulse(
        time,
        drive,
        plasma=plasma,
        drive_threshold=DRIVE_BAR,
        plasma_threshold=PLASMA_BAR,
        **kwargs,
    )


def kinds(timeline: PulseTimeline) -> list[WindowKind]:
    """Return the timeline's window kinds in the order they occurred."""

    return [window.kind for window in timeline.windows]


def test_a_pulse_splits_into_quiet_driven_plasma_and_quiet_again():
    timeline = classify(*stepped_record())
    assert kinds(timeline) == [
        WindowKind.quiet,
        WindowKind.driven,
        WindowKind.plasma,
        WindowKind.quiet,
    ]
    assert timeline.windows[0].start == pytest.approx(0.0)
    assert timeline.windows[1].start == pytest.approx(0.10)
    assert timeline.windows[2].start == pytest.approx(0.15)
    assert timeline.windows[3].start == pytest.approx(0.35)


def test_a_window_selects_exactly_the_samples_it_spans():
    time, drive, plasma = stepped_record()
    timeline = classify(time, drive, plasma)
    window = timeline.windows[1]
    mask = window.mask(time.size)
    assert int(mask.sum()) == window.sample_count
    assert time[mask].min() == pytest.approx(window.start)
    assert time[mask].max() == pytest.approx(window.stop)


def test_a_conductor_at_its_noise_floor_is_not_a_drive():
    """A drive bar the noise clears would classify a dead shot as driven.

    Which is the failure that matters most: the instrument-quiet windows are the
    whole product, and a floor-level current reading as a drive removes every one
    of them from a shot that energised nothing.
    """

    time, _, _ = stepped_record()
    generator = np.random.default_rng(7)
    drive = generator.normal(0.0, 0.05 * DRIVE_BAR, time.size)
    timeline = classify(time, drive)
    assert kinds(timeline) == [WindowKind.quiet]
    assert timeline.windows[0].sample_count == time.size


def test_any_energised_circuit_makes_the_sample_driven():
    time, drive, _ = stepped_record()
    columns = np.column_stack([np.zeros(time.size), drive])
    assert kinds(classify(time, columns)) == [
        WindowKind.quiet,
        WindowKind.driven,
        WindowKind.quiet,
    ]


def test_plasma_outranks_a_drive_that_is_still_on():
    time, drive, plasma = stepped_record()
    timeline = classify(time, drive, plasma)
    driven = timeline.windows[1]
    assert driven.stop < 0.15
    assert timeline.windows[2].kind is WindowKind.plasma


def test_an_unrecorded_plasma_channel_is_a_gap_and_not_a_vacuum_interval():
    """The two readings of a missing plasma signal are opposite statements.

    A designed vacuum shot has no plasma and the adapter says so by passing nothing.
    A shot whose plasma channel was not recorded says nothing about plasma, and the
    adapter says *that* by passing values that are not finite -- which classifies as
    a gap rather than silently joining the vacuum cohort.
    """

    time, drive, _ = stepped_record()
    unrecorded = np.full(time.size, np.nan)
    timeline = classify(time, drive, unrecorded)
    assert timeline.windows == ()
    assert len(timeline.rejected) == 1
    assert "not finite" in timeline.rejected[0].reason
    assert kinds(classify(time, drive)) == [
        WindowKind.quiet,
        WindowKind.driven,
        WindowKind.quiet,
    ]


def test_a_gap_in_the_record_splits_a_window_rather_than_joining_it():
    time, drive, _ = stepped_record()
    drive[200:220] = np.nan
    timeline = classify(time, drive)
    assert kinds(timeline) == [
        WindowKind.quiet,
        WindowKind.quiet,
        WindowKind.driven,
        WindowKind.quiet,
    ]
    assert timeline.windows[0].stop < timeline.windows[1].start
    assert any("not finite" in row.reason for row in timeline.rejected)


def test_the_settling_guard_delays_a_window_that_follows_a_disturbance():
    time, drive, plasma = stepped_record()
    timeline = classify(time, drive, plasma, decay_time=0.146, settling_periods=1.0)
    lead, tail = timeline.windows[0], timeline.windows[-1]
    assert lead.start == pytest.approx(0.0)
    assert not lead.guarded
    assert tail.guarded
    assert tail.start == pytest.approx(0.35 + 0.146, abs=1.0 / SAMPLE_RATE)


def test_a_quiet_predecessor_induces_nothing_so_the_guard_does_not_run():
    """The guard is against decaying passive current, not against elapsed time.

    A quiet interval drives nothing, so the window after it starts where it starts.
    Clipping there would cost the driven windows their leading samples for a
    disturbance that never happened.
    """

    time, drive, _ = stepped_record()
    timeline = classify(time, drive, decay_time=0.05, settling_periods=1.0)
    assert not timeline.windows[1].guarded
    assert timeline.windows[1].start == pytest.approx(0.10)
    assert timeline.windows[2].guarded


def test_a_guard_longer_than_the_window_removes_it_with_the_reason_said():
    time, drive, plasma = stepped_record(quiet_tail=0.05)
    timeline = classify(time, drive, plasma, decay_time=0.146, settling_periods=1.0)
    assert kinds(timeline) == [WindowKind.quiet, WindowKind.driven, WindowKind.plasma]
    assert any("passive" in row.reason for row in timeline.rejected)
    assert timeline.trailing_quiet is None


def test_a_window_under_the_sample_floor_is_dropped_and_recorded():
    time, drive, _ = stepped_record(quiet_lead=0.002)
    timeline = classify(time, drive, minimum_samples=32)
    assert kinds(timeline) == [WindowKind.driven, WindowKind.quiet]
    assert any("32" in row.reason for row in timeline.rejected)


def test_the_leading_and_trailing_quiet_windows_are_the_ones_the_pulse_sits_between():
    timeline = classify(*stepped_record())
    assert timeline.leading_quiet is timeline.windows[0]
    assert timeline.trailing_quiet is timeline.windows[-1]
    assert timeline.leading_quiet.stop < timeline.trailing_quiet.start


def test_a_record_that_never_goes_quiet_offers_no_instrument_window():
    time, _, _ = stepped_record()
    drive = np.full(time.size, 3.0e3)
    plasma = np.full(time.size, 6.0e5)
    timeline = classify(time, drive, plasma)
    assert kinds(timeline) == [WindowKind.plasma]
    assert timeline.leading_quiet is None
    assert timeline.trailing_quiet is None
    assert not timeline.quiet_windows


def test_a_quiet_window_with_nothing_after_it_is_both_leading_and_trailing():
    time, _, _ = stepped_record()
    timeline = classify(time, np.zeros(time.size))
    assert timeline.leading_quiet is timeline.trailing_quiet


def test_the_mask_of_a_kind_is_the_union_of_its_windows():
    time, drive, _ = stepped_record()
    drive[500:520] = np.nan
    timeline = classify(time, drive)
    mask = timeline.mask(WindowKind.quiet)
    assert int(mask.sum()) == sum(row.sample_count for row in timeline.quiet_windows)
    assert not mask[500:520].any()


def test_a_drive_of_a_different_length_than_the_time_base_is_refused():
    time, drive, _ = stepped_record()
    with pytest.raises(WindowError, match="samples"):
        classify(time, drive[:-1])


def test_a_threshold_that_is_not_positive_is_refused():
    time, drive, _ = stepped_record()
    with pytest.raises(WindowError, match="threshold"):
        classify_pulse(time, drive, drive_threshold=0.0, plasma_threshold=PLASMA_BAR)


def test_a_settling_guard_is_recorded_on_the_timeline_it_was_applied_with():
    time, drive, plasma = stepped_record()
    timeline = classify(time, drive, plasma, decay_time=0.146, settling_periods=2.0)
    assert timeline.settling_time == pytest.approx(0.292)
