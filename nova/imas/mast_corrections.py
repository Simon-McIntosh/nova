"""Read what the calibration ladder measured for MAST out of its own document.

One machine, one diagnostic system, one versioned document -- and this is where nova
opens it.  Every consumer that used to carry a copy of a promoted scale or a count of
stepping channels asks here instead, so the numbers a record quotes and the numbers a
read path divides by cannot drift apart: there is only one of each.

The readers below are functions rather than constants for that reason.  A module-level
dictionary is a second copy the moment the document changes, and the failure it
produces is silent -- a record stating a scale the read path no longer applies looks
exactly like a record stating one it does.
"""

from __future__ import annotations

from functools import cache

from nova.calibrate.correction_model import (
    CorrectionKind,
    CorrectionSet,
    CorrectionStatus,
)
from nova.calibrate.correction_set import read_correction_set

MACHINE = "mast"
"""Machine whose corrections this module serves."""

SYSTEM = "magnetics"
"""Diagnostic system within that machine's correction set."""


@cache
def mast_corrections() -> CorrectionSet:
    """Return the validated MAST magnetics correction document.

    Cached because it is read on the probe path and validated on every read: the
    reader checks intervals for overlap and quantised values against their ladder,
    which is worth doing once per process and not once per shot.
    """

    return read_correction_set(MACHINE, SYSTEM)


def _of_kind(kind: CorrectionKind, status: CorrectionStatus):
    """Yield the document's corrections of one kind holding one status."""

    for correction in mast_corrections().corrections:
        if (
            CorrectionKind(correction.kind) is kind
            and CorrectionStatus(correction.status) is status
        ):
            yield correction


def promoted_channel_scales() -> dict[str, float]:
    """Return the steady per-channel calibration scales the ladder promoted.

    Each is steady across every pulse it appears on, off unity by more than five
    percent, agreed to within three percent by an independent route sharing no
    estimator and no pulse selection, and confirmed on both halves of its own pulses
    taken in pulse order.  ``obr17`` is a factor of two, and it is the same channel
    whose coupling to the error-field circuit is fifty times its neighbours' -- two
    independent symptoms of one faulty signal path.

    These are recorded, not removed.  No read path divides by them today; promoting a
    static per-channel gain into the arithmetic is a separate decision from measuring
    one, and the document keeps the two apart.
    """

    return {
        str(row.channel): float(row.value)
        for row in _of_kind(CorrectionKind.gain, CorrectionStatus.promoted)
    }


def withheld_channel_scales() -> tuple[str, ...]:
    """Return the steady channels off unity that the promotion gates refused.

    ``obv11``'s two routes disagree by 7.4 percent and its own two halves read 0.869
    and 1.065, so whatever it is doing is not one scale.  Carried because a refusal
    with a reason is worth more than a silence.
    """

    return tuple(
        sorted(
            str(row.channel)
            for row in _of_kind(CorrectionKind.gain, CorrectionStatus.withheld)
        )
    )


def acquisition_stepping_channels() -> int:
    """Return how many probe channels were recorded at more than one scale.

    Of seventy-six with a measurable history.  A single calibration number cannot
    describe them: fitted across a step it returns an average of two discrete states
    weighted by pulse count, which describes no pulse and moves when the pulse
    selection moves.  Sixteen were handed a single gain by an independent route that
    did not resolve the blocks, spanning up to a factor of 4.2 -- which is why
    :func:`promoted_channel_scales` holds five channels and not thirty.

    What describes them instead is a record per block of pulses, which is what the
    document carries and what :mod:`~nova.imas.mast_block_scale` divides out where the
    channel is read.  So the quantity these channels lack is specifically a *static*
    per-channel scale, not a calibration record.
    """

    counts: dict[str, int] = {}
    for correction in mast_corrections().corrections:
        if CorrectionKind(correction.kind) is CorrectionKind.acquisition_scale:
            counts[str(correction.channel)] = counts.get(
                str(correction.channel), 0
            ) + len(correction.validity)
    return sum(1 for total in counts.values() if total > 1)


def acquisition_off_ladder_channels() -> int:
    """Return how many blocks hold a step that is not a rung of the ladder.

    A step off the ladder is not evidence of a range setting, so no factor is divided
    out there and the block is read as published and flagged.  Rounding it onto the
    nearest rung would assert a setting the ladder does not support and silently move
    the channel by the difference.
    """

    return sum(
        1 for _ in _of_kind(CorrectionKind.acquisition_scale, CorrectionStatus.refused)
    )
