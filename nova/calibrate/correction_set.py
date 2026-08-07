"""Read a correction document, and refuse one a consumer could misapply.

The generated models check that a document has the right shape: the right slots,
the right ranges, an enumerated kind that the schema declares.  What they cannot
check is the two ways a well-shaped document still lies about a channel.

The first is an overlap.  A correction is scoped to a pulse range because a channel
holds one state over a run of pulses and then steps; if two corrections the read
path would both apply cover one pulse, the channel is multiplied twice and no
consumer can tell.  The overlap rule is therefore scoped within one status: a
superseded record deliberately covers the same pulses as the correction that
replaced it, and refusing that would force the record to be deleted to stay valid.

The second is a value that is not the quantity its kind names.  An acquisition
range setting moves by discrete factors, so a block whose measured step lands
between rungs is telling us something -- but not that the range changed, and
dividing it out would launder the description's own error into the data.  The
ladder is declared in the document, and a value that misses every rung is refused
here rather than rounded onto the nearest one.

Both faults are cheap to make and silent once made, which is why validation runs on
every read rather than on authoring alone.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from pathlib import Path

import yaml
from pydantic import ValidationError

from nova.calibrate.correction_model import (
    ApplicationStage,
    ChannelCorrection,
    CorrectionKind,
    CorrectionSet,
    CorrectionStatus,
    QuantisationLadder,
    ValidityInterval,
)

SCHEMA_PATH = Path(__file__).parent / "schema" / "diagnostic_correction.yaml"
"""LinkML source the models and the JSON-Schema export are generated from."""

SCHEMA_VERSION = "1.0.0"
"""Schema version a document must declare to be read by this module.

Pinned here rather than parsed from the schema on every read, and checked against
the schema file by the test suite: a document written against a version this code
does not implement is refused instead of being half-understood.
"""

CORRECTION_ROOT = Path(__file__).parent / "corrections"
"""Directory holding one subdirectory per machine, one document per system."""

APPLICATION_ORDER: tuple[ApplicationStage, ...] = (
    ApplicationStage.offset,
    ApplicationStage.drift,
    ApplicationStage.acquisition_scale,
    ApplicationStage.pair_state,
    ApplicationStage.gain,
    ApplicationStage.convention,
)
"""The order the read path removes corrections in, first to last.

Additive terms come out before multiplicative ones.  Among the multiplicative
terms the acquisition rung comes out first because the instrument applied it last;
the pickup state comes out before the gain because a gain fitted across mixed
states carries them.  The convention factor is applied at the unit boundary, after
the per-channel chain, never inside it.

The same order is declared in the schema as the ranks of
:class:`~nova.calibrate.correction_model.ApplicationStage`, and the two are asserted
equal by the tests -- a consumer that reads either gets the same answer.
"""

KIND_STAGE: dict[CorrectionKind, ApplicationStage] = {
    CorrectionKind.offset: ApplicationStage.offset,
    CorrectionKind.drift_rate: ApplicationStage.drift,
    CorrectionKind.acquisition_scale: ApplicationStage.acquisition_scale,
    CorrectionKind.pair_state: ApplicationStage.pair_state,
    CorrectionKind.gain: ApplicationStage.gain,
    CorrectionKind.convention: ApplicationStage.convention,
}
"""Where each applicable kind enters the chain.

The kinds absent from this mapping -- ``exclusion`` and ``quality`` -- change no
sample.  One is an instruction to drop a channel and the other describes its
condition, so neither has a place in an ordering of multipliers.
"""

APPLIED_STATUS = CorrectionStatus.promoted
"""The one status the read path acts on."""

UNESTABLISHED_STATUS = frozenset({CorrectionStatus.withheld, CorrectionStatus.refused})
"""Statuses saying the quantity the kind names was never established.

A withheld correction failed its promotion gate and a refused one measured a step
that is not the discrete factor its kind moves by.  Neither may carry a value: what
they have is a measurement, which belongs in ``measured_value`` or
``candidate_values`` where no consumer will divide by it.  A recorded correction is
the opposite case -- the quantity was established and simply is not applied -- so it
keeps its value.
"""

MULTIPLICATIVE_KINDS = frozenset(
    {
        CorrectionKind.gain,
        CorrectionKind.acquisition_scale,
        CorrectionKind.pair_state,
        CorrectionKind.convention,
    }
)
"""Kinds whose value divides a signal, so a value of zero erases it."""

DESCRIPTIVE_KINDS = frozenset({CorrectionKind.exclusion, CorrectionKind.quality})
"""Kinds that carry no value: they describe or instruct, they do not scale."""


class CorrectionSetError(ValueError):
    """Raised when a correction document cannot be applied as written."""


def target(correction: ChannelCorrection) -> str:
    """Return what the correction applies to, channel or named group."""

    if correction.channel is not None:
        return correction.channel
    if correction.channel_group is not None:
        return f"group:{correction.channel_group}"
    raise CorrectionSetError("correction names neither a channel nor a channel group")


def stage(correction: ChannelCorrection) -> ApplicationStage | None:
    """Return where the correction enters the read-path chain, or None."""

    return KIND_STAGE.get(CorrectionKind(correction.kind))


def applied(document: CorrectionSet) -> Iterator[ChannelCorrection]:
    """Yield the corrections a read path acts on, in application order."""

    order = {value: rank for rank, value in enumerate(APPLICATION_ORDER)}
    rows = [
        row
        for row in document.corrections
        if CorrectionStatus(row.status) is APPLIED_STATUS and stage(row) is not None
    ]
    yield from sorted(rows, key=lambda row: (order[stage(row)], target(row)))


def load_correction_set(path: Path | str) -> CorrectionSet:
    """Return the validated document at a path."""

    path = Path(path)
    try:
        document = CorrectionSet.model_validate(yaml.safe_load(path.read_text()))
    except ValidationError as error:
        raise CorrectionSetError(
            f"{path} does not match the schema: {error}"
        ) from error
    validate_correction_set(document)
    return document


def read_correction_set(machine: str, diagnostic_system: str) -> CorrectionSet:
    """Return the validated document for one machine's diagnostic system."""

    path = CORRECTION_ROOT / machine / f"{diagnostic_system}.yaml"
    if not path.exists():
        raise CorrectionSetError(
            f"no correction document at {path}: an absent document and an "
            "uncorrected channel look identical to a consumer, so the absence is "
            "raised rather than read as an empty set"
        )
    document = load_correction_set(path)
    if document.machine != machine or document.diagnostic_system != diagnostic_system:
        raise CorrectionSetError(
            f"{path} declares {document.machine}/{document.diagnostic_system} but "
            f"sits at {machine}/{diagnostic_system}, so its path states one scope "
            "and its content another"
        )
    return document


def validate_correction_set(document: CorrectionSet) -> None:
    """Raise unless every correction in the document can be applied as written."""

    if document.schema_version != SCHEMA_VERSION:
        raise CorrectionSetError(
            f"document declares schema version {document.schema_version}, and this "
            f"reader implements {SCHEMA_VERSION}"
        )
    ladders = {row.name: row for row in document.ladders or ()}
    quantised = {CorrectionKind(row.kind) for row in ladders.values()}
    for correction in document.corrections:
        _validate_scope(correction)
        _validate_value(correction)
        _validate_ladder(correction, ladders, quantised)
        _validate_provenance(correction)
        for interval in correction.validity:
            _validate_interval(correction, interval)
    _validate_no_overlap(document)


def _label(correction: ChannelCorrection) -> str:
    """Name a correction well enough to find it in the document."""

    return f"{target(correction)} {correction.kind} ({correction.status})"


def _validate_scope(correction: ChannelCorrection) -> None:
    """Refuse a correction that names no target, or two."""

    if (correction.channel is None) == (correction.channel_group is None):
        raise CorrectionSetError(
            f"{correction.kind} correction names "
            f"{'both a channel and a group' if correction.channel else 'no target'}: "
            "exactly one of channel and channel_group carries the scope"
        )
    if not correction.validity:
        raise CorrectionSetError(
            f"{_label(correction)} carries no validity interval, so it states a "
            "correction without saying when it holds"
        )


def _validate_value(correction: ChannelCorrection) -> None:
    """Refuse a value the kind cannot carry, or a missing one it needs."""

    kind = CorrectionKind(correction.kind)
    status = CorrectionStatus(correction.status)
    label = _label(correction)
    if kind in DESCRIPTIVE_KINDS and correction.value is not None:
        raise CorrectionSetError(
            f"{label} carries a value, but this kind describes a channel rather "
            "than scaling it"
        )
    if kind is CorrectionKind.quality and correction.quality_status is None:
        raise CorrectionSetError(f"{label} states no quality status")
    if kind is not CorrectionKind.quality and correction.quality_status is not None:
        raise CorrectionSetError(f"{label} states a quality status it cannot carry")
    if kind is CorrectionKind.pair_state and correction.state is None:
        raise CorrectionSetError(f"{label} states no pickup state")
    if kind is not CorrectionKind.pair_state and correction.state is not None:
        raise CorrectionSetError(f"{label} states a pickup state it cannot carry")
    if kind is CorrectionKind.exclusion and not correction.cause:
        raise CorrectionSetError(
            f"{label} excludes a channel without saying why, which a consumer "
            "cannot weigh against its own needs"
        )
    if status is APPLIED_STATUS:
        _validate_applied_value(correction, kind, label)
    else:
        _validate_unapplied_value(correction, kind, status, label)
    _validate_uncertainty(correction, label)


def _validate_applied_value(
    correction: ChannelCorrection, kind: CorrectionKind, label: str
) -> None:
    """Refuse a promoted correction the read path could not apply."""

    if kind in DESCRIPTIVE_KINDS:
        return
    if correction.value is None:
        raise CorrectionSetError(
            f"{label} is promoted but carries no value, so the read path would "
            "apply nothing while reporting that it corrected the channel"
        )
    if not math.isfinite(correction.value):
        raise CorrectionSetError(f"{label} carries a value that is not finite")
    if kind in MULTIPLICATIVE_KINDS and correction.value == 0.0:
        raise CorrectionSetError(
            f"{label} carries a multiplier of zero, which erases the channel "
            "rather than correcting it"
        )


def _validate_unapplied_value(
    correction: ChannelCorrection,
    kind: CorrectionKind,
    status: CorrectionStatus,
    label: str,
) -> None:
    """Refuse a record that says nothing, or that names a value it refused."""

    if status in UNESTABLISHED_STATUS and correction.value is not None:
        raise CorrectionSetError(
            f"{label} carries a value although its status says the quantity the "
            "kind names was never established; the raw measurement belongs in "
            "measured_value or candidate_values, where nothing will apply it"
        )
    if kind in DESCRIPTIVE_KINDS:
        return
    if (
        correction.value is None
        and correction.measured_value is None
        and not correction.candidate_values
    ):
        raise CorrectionSetError(
            f"{label} carries no value, no measurement and no candidates, so it "
            "records that something was not promoted without recording what"
        )


def _validate_uncertainty(correction: ChannelCorrection, label: str) -> None:
    """Refuse an interval that runs backwards."""

    bound = correction.uncertainty
    if bound is None or bound.lower is None or bound.upper is None:
        return
    if bound.lower > bound.upper:
        raise CorrectionSetError(
            f"{label} carries the interval [{bound.lower}, {bound.upper}], which "
            "runs backwards"
        )


def _validate_ladder(
    correction: ChannelCorrection,
    ladders: dict[str, QuantisationLadder],
    quantised: set[CorrectionKind],
) -> None:
    """Refuse a quantised value that is not on its ladder."""

    kind = CorrectionKind(correction.kind)
    label = _label(correction)
    if correction.ladder is None:
        if kind in quantised:
            raise CorrectionSetError(
                f"{label} names no ladder, and this set declares one for its kind: "
                "a quantised correction that opts out of its ladder is exactly the "
                "free-floating value the ladder exists to refuse"
            )
        return
    ladder = ladders.get(correction.ladder)
    if ladder is None:
        raise CorrectionSetError(
            f"{label} names ladder {correction.ladder!r}, which the set does not "
            "declare"
        )
    if CorrectionKind(ladder.kind) is not kind:
        raise CorrectionSetError(
            f"{label} names ladder {ladder.name!r}, which quantises {ladder.kind}"
        )
    if correction.value is None:
        return
    distance = min(
        abs(correction.value - rung) / rung for rung in ladder.rungs if rung > 0.0
    )
    if distance > ladder.tolerance:
        raise CorrectionSetError(
            f"{label} carries {correction.value}, which misses every rung of "
            f"{ladder.name!r} by {distance:.3f} against a tolerance of "
            f"{ladder.tolerance}: a step off the ladder is not evidence of a range "
            "setting, and rounding it onto the nearest rung would assert a setting "
            "the ladder does not support"
        )


def _validate_provenance(correction: ChannelCorrection) -> None:
    """Refuse a promoted correction that cites nothing."""

    if CorrectionStatus(correction.status) is not APPLIED_STATUS:
        return
    if not correction.provenance.evidence_uri:
        raise CorrectionSetError(
            f"{_label(correction)} is applied to data and cites no evidence, so "
            "nothing distinguishes it from a guess once this session is over"
        )


def _validate_interval(
    correction: ChannelCorrection, interval: ValidityInterval
) -> None:
    """Refuse an interval that names nothing a consumer can resolve."""

    label = _label(correction)
    pulse = (interval.pulse_start, interval.pulse_end)
    moment = (interval.time_start, interval.time_end)
    if any(bound is not None for bound in pulse) and any(
        bound is not None for bound in moment
    ):
        raise CorrectionSetError(
            f"{label} carries an interval bounded in both pulse and time; the two "
            "cannot be ordered against each other, so the interval names nothing"
        )
    for lower, upper, axis in ((*pulse, "pulse"), (*moment, "time")):
        if lower is not None and upper is not None and lower > upper:
            raise CorrectionSetError(
                f"{label} carries the {axis} interval [{lower}, {upper}], which "
                "runs backwards"
            )
    pulses = interval.measured_pulses or []
    if list(pulses) != sorted(set(pulses)):
        raise CorrectionSetError(
            f"{label} lists measured pulses that are not a sorted distinct run"
        )
    outside = [
        pulse
        for pulse in pulses
        if (interval.pulse_start is not None and pulse < interval.pulse_start)
        or (interval.pulse_end is not None and pulse > interval.pulse_end)
    ]
    if outside:
        raise CorrectionSetError(
            f"{label} was measured on pulses {outside} that lie outside the "
            "interval it claims to hold over"
        )


def _spans(interval: ValidityInterval) -> dict[str, tuple[float, float]]:
    """Return the interval as a span per axis, unbounded axes included."""

    pulse = (interval.pulse_start, interval.pulse_end)
    moment = (interval.time_start, interval.time_end)
    if all(bound is None for bound in (*pulse, *moment)):
        return {
            "pulse": (-math.inf, math.inf),
            "time": (-math.inf, math.inf),
        }
    if any(bound is not None for bound in pulse):
        return {
            "pulse": (
                -math.inf if pulse[0] is None else float(pulse[0]),
                math.inf if pulse[1] is None else float(pulse[1]),
            )
        }
    return {
        "time": (
            -math.inf if moment[0] is None else float(moment[0]),
            math.inf if moment[1] is None else float(moment[1]),
        )
    }


def _validate_no_overlap(document: CorrectionSet) -> None:
    """Refuse two corrections of one kind and status covering one pulse.

    Scoped within a status because that is where the fault lives: two corrections
    the read path would both apply multiply a channel twice.  A superseded record
    covering the same pulses as its replacement is the record working as intended.
    """

    grouped: dict[tuple[str, str, str], list[ChannelCorrection]] = {}
    for correction in document.corrections:
        key = (target(correction), str(correction.kind), str(correction.status))
        grouped.setdefault(key, []).append(correction)
    for (name, kind, status), rows in grouped.items():
        spans: list[tuple[str, float, float]] = [
            (axis, lower, upper)
            for correction in rows
            for interval in correction.validity
            for axis, (lower, upper) in _spans(interval).items()
        ]
        for index, (axis, lower, upper) in enumerate(spans):
            for other_axis, other_lower, other_upper in spans[index + 1 :]:
                if axis != other_axis:
                    continue
                if lower <= other_upper and other_lower <= upper:
                    raise CorrectionSetError(
                        f"{name} {kind} ({status}) carries overlapping {axis} "
                        f"intervals [{lower}, {upper}] and "
                        f"[{other_lower}, {other_upper}]: a read inside the overlap "
                        "would apply both corrections to one channel"
                    )
