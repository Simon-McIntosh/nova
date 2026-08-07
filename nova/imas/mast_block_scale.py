"""Divide out the acquisition range setting a probe channel was recorded at.

:mod:`~nova.imas.mast_acquisition_scale` measured that nineteen probe channels were
not recorded at one setting: each sits at one scale for a run of shots, steps by a
factor from a binary ladder, holds, and steps back, while most channels hold still
on the same shots.  That leaves those channels with no static calibration record at
all -- a single number fitted across a step is an average of two discrete states
weighted by shot count, describing no shot and moving when the shot selection moves.

What such a channel needs is the setting the block a shot belongs to was recorded
at, divided out where the channel is read.  Putting it on the read path rather than
inside one fit is the point: every consumer -- a vacuum response fit, a solve-input
map, a later gate run -- then reads a channel whose amplitude means the same thing on
every shot, and none of them has to know that a range setting existed.

**What is divided out is the ladder rung, never the fitted ratio.**  A channel's
measured scale is its field over the described field, so it carries the description's
own error as well as the acquisition setting.  Dividing that ratio out would launder
model error into the data and quietly fit the description to itself.  So each block is
referred to the block reading nearest the described field -- the one recorded at the
ordinary setting -- and what the read divides by is the nearest rung of the declared
ladder to that relative factor: an exact 2, 1/2, root two or its reciprocal.  A
different anchor would move every rung of one channel by one common factor, and a
common per-channel factor is what this correction deliberately leaves alone.  The
channel's overall level, which is
where any description error sits, is left exactly as it was: promoting a static
per-channel gain is a separate question that the sensor adjudication owns and
answered separately.  A block whose relative factor does not land on a rung is
refused rather than rounded onto one.

**A block carries the shots it was measured on, not just its span.**  The cohort
that measured these settings is a few dozen plasma-free shots scattered over an
archive of seventeen thousand, so a block running from shot 14061 to 19258 rests on
thirty-four of the five thousand shots between them.  Recording the span alone would
let a read of shot 17000 come back measured when nothing measured it.

**Every read returns its warrant.**  A :class:`ScaleCorrection` carries the
disposition that justified it -- :data:`MEASURED` when the shot itself was measured,
:data:`BRACKETED` when it sits between two measurements of one block so both sides
agree and there is no switch to place, :data:`REFUSED` when the block's step is not
on the ladder, and :data:`UNMEASURED` when nothing warrants a number.  A refused or
unmeasured channel is read exactly as published and flagged.  The case that matters
is a shot between two blocks: the switch happened somewhere in there and the archive
may hold no shot that says where, so this is precisely where guessing would cost the
factor of two the ladder moves by.  :class:`ScaleBracket` reports each such gap with
its width, which is what distinguishes a switch pinned to adjacent shots from one
known only to within a campaign.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from nova.calibrate.correction_model import (
    CorrectionKind,
    CorrectionSet,
    CorrectionStatus,
)
from nova.calibrate.corrections import build_chain
from nova.imas.mast_acquisition_scale import (
    LADDER_TOLERANCE,
    ChannelScaleHistory,
    nearest_rung,
)

MEASURED = "measured"
"""The shot itself carries a scale measurement inside a block."""

BRACKETED = "bracketed"
"""The shot sits between two measurements of one block, which agree."""

REFUSED = "refused"
"""The block's step is not a ladder rung, so it is not a range setting."""

UNMEASURED = "unmeasured"
"""Nothing warrants a setting for this shot, so the channel is read as published."""

DISPOSITIONS = (MEASURED, BRACKETED, REFUSED, UNMEASURED)
"""Every warrant a read can carry, strongest first."""


class BlockScaleError(ValueError):
    """Raised when a block table is malformed or cannot serve a read."""


@dataclass(frozen=True, order=True)
class BlockScale:
    """One run of shots a channel held one setting over, and what that setting was.

    ``scale`` is the ratio the fit measured -- this channel's field over the
    described field, pooled over the block's shots.  ``rung`` is what a read divides
    by: the declared ladder's nearest factor to ``scale`` over the reference block's
    scale, exactly one on the reference block itself, and not finite where the
    relative factor missed every rung.  The two are kept apart because only the
    second is a statement about the acquisition; the first also carries whatever the
    description gets wrong about this probe.

    ``route`` names the measurement the block came from, so a reader can tell a
    setting pinned by plasma-free single-coil shots from one placed by a coarser
    sweep.  It travels per block because two blocks of one channel can rest on
    different evidence -- the archive is dense in some campaigns and holds a handful
    of shots in others.
    """

    channel: str
    scale: float
    shots: tuple[int, ...]
    rung: float = 1.0
    route: str = ""

    @property
    def first_shot(self) -> int:
        """Return the earliest shot this setting was measured on."""

        return int(self.shots[0])

    @property
    def last_shot(self) -> int:
        """Return the latest shot this setting was measured on."""

        return int(self.shots[-1])

    @property
    def shot_count(self) -> int:
        """Return how many shots the setting rests on."""

        return len(self.shots)

    @property
    def span(self) -> int:
        """Return how many shot numbers the block reaches across."""

        return self.last_shot - self.first_shot

    @property
    def on_ladder(self) -> bool:
        """Return whether this block's step is a range setting the read may remove."""

        return bool(math.isfinite(self.rung)) and self.rung > 0.0

    @property
    def unchanged(self) -> bool:
        """Return whether this block sits at the setting a read leaves alone.

        True of the block the channel is referred to and of every later block that
        returned to the same setting, which is the useful reading: it says a shot in
        this block needs no correction, not which block was picked as the anchor.
        """

        return self.on_ladder and self.rung == 1.0

    def measured(self, shot: int) -> bool:
        """Return whether this shot is one the setting was measured on."""

        return int(shot) in frozenset(self.shots)

    def inside(self, shot: int) -> bool:
        """Return whether this shot falls within the measured span."""

        return self.first_shot <= int(shot) <= self.last_shot

    def validate(self) -> None:
        """Reject a block that cannot describe a run of shots."""

        if not self.channel or self.channel.strip() != self.channel:
            raise BlockScaleError(f"block channel {self.channel!r} is not a name")
        if not self.shots:
            raise BlockScaleError(
                f"{self.channel!r} block rests on no shot, so its setting is "
                "asserted rather than measured"
            )
        if list(self.shots) != sorted(set(self.shots)):
            raise BlockScaleError(
                f"{self.channel!r} block shots are not a sorted distinct run"
            )
        if not math.isfinite(self.scale) or self.scale == 0.0:
            raise BlockScaleError(
                f"{self.channel!r} block carries scale {self.scale}, which is not a "
                "measurable ratio"
            )
        if self.rung == 0.0 or (math.isfinite(self.rung) and self.rung < 0.0):
            raise BlockScaleError(
                f"{self.channel!r} block carries a rung of {self.rung}, and a range "
                "setting neither erases a signal nor inverts one"
            )

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical JSON representation."""

        return {
            "channel": self.channel,
            "first_shot": self.first_shot,
            "last_shot": self.last_shot,
            "on_ladder": self.on_ladder,
            "route": self.route,
            "rung": None if not self.on_ladder else float(self.rung),
            "scale": float(self.scale),
            "shot_count": self.shot_count,
            "shots": [int(shot) for shot in self.shots],
        }

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> BlockScale:
        """Build a validated block from decoded JSON."""

        rung = row.get("rung")
        block = cls(
            channel=str(row["channel"]),
            scale=float(row["scale"]),
            shots=tuple(int(shot) for shot in row["shots"]),
            rung=math.nan if rung is None else float(rung),
            route=str(row.get("route", "")),
        )
        block.validate()
        return block


def channel_blocks(
    history: ChannelScaleHistory,
    shots: Iterable[int],
    *,
    route: str = "",
    tolerance: float = LADDER_TOLERANCE,
) -> tuple[BlockScale, ...]:
    """Refer one channel's blocks to a reference and snap each step to a rung.

    The reference is the block whose measured scale sits closest to unity, because
    that is the block recorded at the setting the description was written for: a
    range factor is a gross discrete multiple, so the block reading near the
    described field is the one at the ordinary setting and the others are that
    setting times a rung.  Choosing by shot count instead would let an archive that
    happened to take more shots at the doubled setting invert the whole correction
    and divide the ordinary blocks by two.  Ties -- a channel with blocks at root
    two and its reciprocal and nothing between -- go to the block resting on more
    shots and then to the earlier one, so the choice never depends on iteration
    order.

    What the reference choice cannot do is change the channel's relative structure,
    which is the only thing a read removes.  A different reference would shift every
    rung of one channel by one common factor, and a common per-channel factor is
    exactly what this correction deliberately leaves alone.

    A block whose relative factor misses every rung by more than ``tolerance`` keeps a
    rung that is not finite: the step is real -- the block finder measured it -- but it
    is not the discrete factor a range setting moves by, so removing it would be a fit
    rather than a correction.

    A block whose own measured scale is negative is refused on the same grounds and for
    a sharper reason.  The ladder is positive by declaration, so no rung inverts a
    signal; a channel reading the described field backwards on a run of shots is saying
    something about its polarity or about how little signal it had, and neither is a
    range setting.  These are rare -- a fraction of a percent of readings, on the
    quietest channels -- and refusing them is what keeps that tail out of the
    correction.
    """

    blocks = history.blocks
    if not blocks:
        return ()
    measured = sorted(int(shot) for shot in shots)
    inside = {
        block.first_shot: tuple(
            shot for shot in measured if block.first_shot <= shot <= block.last_shot
        )
        for block in blocks
    }
    reference = min(
        blocks,
        key=lambda block: (
            abs(math.log(block.scale)) if block.scale > 0.0 else math.inf,
            -len(inside[block.first_shot]),
            block.first_shot,
        ),
    )
    rows = []
    for block in blocks:
        if block.scale <= 0.0 or reference.scale <= 0.0:
            rung = math.nan
        else:
            candidate, distance = nearest_rung(block.scale / reference.scale)
            rung = candidate if distance <= tolerance else math.nan
        rows.append(
            BlockScale(
                channel=block.channel,
                scale=float(block.scale),
                shots=inside[block.first_shot],
                rung=rung,
                route=route,
            )
        )
    return tuple(row for row in rows if row.shots)


@dataclass(frozen=True, order=True)
class ScaleBracket:
    """The span between two blocks, inside which a switch happened somewhere.

    A switch is reported as a bracket rather than as a shot because that is what the
    measurement supports: the last shot of one block read one setting, the first
    shot of the next read another, and every shot between them is unmeasured.  A
    bracket two shots wide names the switch; a bracket five thousand shots wide
    names a campaign, and writing either as a single boundary shot would assert the
    same thing about both.
    """

    channel: str
    before_shot: int
    after_shot: int
    before_rung: float
    after_rung: float

    @property
    def width(self) -> int:
        """Return how many shot numbers separate the two measured sides."""

        return int(self.after_shot - self.before_shot)

    @property
    def ratio(self) -> float:
        """Return the factor the setting changed by across the bracket."""

        if self.before_rung == 0.0:
            return math.inf
        return self.after_rung / self.before_rung

    def unresolved(self, shots: Iterable[int]) -> tuple[int, ...]:
        """Return the archive shots inside this bracket that would narrow it.

        A bracket is as pinned as the archive allows once no readable shot lies
        strictly inside it.  Handing the candidates back is what lets a sweep spend
        its next read where it buys a boundary instead of a repeat.
        """

        return tuple(
            int(shot)
            for shot in sorted(shots)
            if self.before_shot < shot < self.after_shot
        )

    def pinned(self, shots: Iterable[int]) -> bool:
        """Return whether no archive shot could narrow this bracket further."""

        return not self.unresolved(shots)

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical JSON representation."""

        return {
            "after_rung": float(self.after_rung),
            "after_shot": self.after_shot,
            "before_rung": float(self.before_rung),
            "before_shot": self.before_shot,
            "channel": self.channel,
            "ratio": float(self.ratio),
            "width": self.width,
        }


@dataclass(frozen=True)
class ScaleCorrection:
    """The setting one read divided out, and what warranted it."""

    channel: str
    shot: int
    scale: float
    disposition: str
    candidates: tuple[float, ...] = ()

    @property
    def applied(self) -> bool:
        """Return whether the read divided by this setting."""

        return self.disposition in (MEASURED, BRACKETED)

    @property
    def flagged(self) -> bool:
        """Return whether the reader must know the setting was not measured here."""

        return self.disposition != MEASURED

    def normalise(self, values: Any) -> np.ndarray:
        """Return the samples with the acquisition setting removed.

        A refused or unmeasured channel comes back untouched rather than divided by
        one, so a consumer cannot tell an applied unity from a refusal by inspecting
        the array -- which is why the disposition, not the array, is what says
        whether a correction happened.
        """

        array = np.asarray(values, dtype=float)
        if not self.applied or self.scale == 1.0:
            return array
        return array / float(self.scale)

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical JSON representation."""

        return {
            "applied": self.applied,
            "candidates": [float(value) for value in self.candidates],
            "channel": self.channel,
            "disposition": self.disposition,
            "flagged": self.flagged,
            "scale": float(self.scale),
            "shot": int(self.shot),
        }


def _ordered(blocks: Iterable[BlockScale]) -> tuple[BlockScale, ...]:
    rows = sorted(blocks, key=lambda block: block.shots)
    for first, second in zip(rows, rows[1:], strict=False):
        if second.first_shot <= first.last_shot:
            raise BlockScaleError(
                f"{first.channel!r} blocks {first.first_shot}-{first.last_shot} and "
                f"{second.first_shot}-{second.last_shot} overlap, so a shot in the "
                "overlap has two settings"
            )
    return tuple(rows)


@dataclass(frozen=True)
class BlockScaleTable:
    """Every channel's measured setting blocks, and the read they warrant.

    An empty table is the raw archive: every channel comes back as published, with
    its correction reporting :data:`UNMEASURED` so a consumer can see that nothing
    was established rather than inferring it from an unchanged array.
    """

    blocks: Mapping[str, tuple[BlockScale, ...]] = field(default_factory=dict)
    route: str = ""

    @classmethod
    def create(
        cls, blocks: Iterable[BlockScale], *, route: str = ""
    ) -> BlockScaleTable:
        """Build a validated table from loose blocks."""

        grouped: dict[str, list[BlockScale]] = {}
        for block in blocks:
            block.validate()
            grouped.setdefault(block.channel, []).append(block)
        return cls(
            blocks={
                channel: _ordered(rows) for channel, rows in sorted(grouped.items())
            },
            route=route,
        )

    @classmethod
    def from_histories(
        cls,
        histories: Iterable[ChannelScaleHistory],
        series: Mapping[str, Iterable[int]],
        *,
        route: str = "",
    ) -> BlockScaleTable:
        """Adopt the blocks the acquisition-scale classifier measured.

        ``series`` supplies the shots each channel was measured on, which the
        classifier's block endpoints do not carry.  A channel whose history is too
        short to classify contributes nothing, so it reads as published and flagged
        rather than carrying a setting nobody could tell from scatter.
        """

        return cls.create(
            (
                block
                for row in histories
                if row.measured
                for block in channel_blocks(
                    row, series.get(row.channel, ()), route=route
                )
            ),
            route=route,
        )

    @property
    def channels(self) -> tuple[str, ...]:
        """Return every channel the table carries a block for."""

        return tuple(sorted(self.blocks))

    @property
    def stepping(self) -> tuple[str, ...]:
        """Return the channels measured at more than one setting."""

        return tuple(
            channel for channel in self.channels if len(self.blocks[channel]) > 1
        )

    @property
    def corrected(self) -> tuple[str, ...]:
        """Return the channels some block of which a read divides a rung out of."""

        return tuple(
            channel
            for channel in self.channels
            if any(
                block.on_ladder and block.rung != 1.0 for block in self.blocks[channel]
            )
        )

    def brackets(self) -> tuple[ScaleBracket, ...]:
        """Return the span between each pair of consecutive blocks."""

        return tuple(
            ScaleBracket(
                channel=channel,
                before_shot=first.last_shot,
                after_shot=second.first_shot,
                before_rung=first.rung,
                after_rung=second.rung,
            )
            for channel in self.channels
            for first, second in zip(
                self.blocks[channel], self.blocks[channel][1:], strict=False
            )
        )

    def correction(self, channel: str, shot: int) -> ScaleCorrection:
        """Return the setting a read of this channel on this shot may divide by."""

        shot = int(shot)
        rows = self.blocks.get(channel, ())
        if not rows:
            return ScaleCorrection(channel, shot, 1.0, UNMEASURED)
        for block in rows:
            if not (block.measured(shot) or block.inside(shot)):
                continue
            if not block.on_ladder:
                return ScaleCorrection(
                    channel, shot, 1.0, REFUSED, candidates=(block.scale,)
                )
            disposition = MEASURED if block.measured(shot) else BRACKETED
            return ScaleCorrection(channel, shot, block.rung, disposition)
        for first, second in zip(rows, rows[1:], strict=False):
            if first.last_shot < shot < second.first_shot:
                return ScaleCorrection(
                    channel,
                    shot,
                    1.0,
                    UNMEASURED,
                    candidates=(first.rung, second.rung),
                )
        nearest = rows[0] if shot < rows[0].first_shot else rows[-1]
        return ScaleCorrection(
            channel, shot, 1.0, UNMEASURED, candidates=(nearest.rung,)
        )

    def corrections(
        self, shot: int, channels: Iterable[str]
    ) -> tuple[ScaleCorrection, ...]:
        """Return one correction per channel for one shot, in channel order."""

        return tuple(
            self.correction(channel, shot) for channel in sorted(set(channels))
        )

    def normalise(
        self, shot: int, probes: Mapping[str, np.ndarray]
    ) -> tuple[dict[str, np.ndarray], tuple[ScaleCorrection, ...]]:
        """Divide one shot's probe channels by the setting each was recorded at."""

        rows = self.corrections(shot, probes)
        lookup = {row.channel: row for row in rows}
        return (
            {
                channel: lookup[channel].normalise(values)
                for channel, values in probes.items()
            },
            rows,
        )

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical JSON representation."""

        return {
            "blocks": [
                block.as_dict()
                for channel in self.channels
                for block in self.blocks[channel]
            ],
            "route": self.route,
        }

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> BlockScaleTable:
        """Build a validated table from decoded JSON."""

        return cls.create(
            (BlockScale.from_dict(block) for block in row["blocks"]),
            route=str(row.get("route", "")),
        )


@dataclass(frozen=True, order=True)
class ChannelSetting:
    """One interval of a channel's acquisition record, as a document states it.

    The document is the source of both halves of a read: what to divide by, and what
    warrants dividing.  The engine answers the first and needs no help; the second is
    a question about intervals the read falls outside of -- whether the setting was
    refused, or simply never measured near this pulse -- and no chain describes a
    correction that did not apply.  This is the index that answers it.
    """

    pulse_start: int
    pulse_end: int
    rung: float
    measured_value: float
    refused: bool

    def covers(self, shot: int) -> bool:
        """Return whether this interval holds over a shot."""

        return self.pulse_start <= int(shot) <= self.pulse_end


def _channel_settings(
    document: CorrectionSet,
) -> dict[str, tuple[ChannelSetting, ...]]:
    """Index a document's acquisition record by channel, in pulse order."""

    grouped: dict[str, list[ChannelSetting]] = {}
    for correction in document.corrections:
        if CorrectionKind(correction.kind) is not CorrectionKind.acquisition_scale:
            continue
        refused = CorrectionStatus(correction.status) is not CorrectionStatus.promoted
        for interval in correction.validity:
            if interval.pulse_start is None or interval.pulse_end is None:
                raise BlockScaleError(
                    f"{correction.channel} carries an acquisition setting over an "
                    "interval unbounded in pulse; a range setting holds over a run of "
                    "shots and steps, so an unbounded one states no block"
                )
            grouped.setdefault(str(correction.channel), []).append(
                ChannelSetting(
                    pulse_start=int(interval.pulse_start),
                    pulse_end=int(interval.pulse_end),
                    rung=math.nan
                    if correction.value is None
                    else float(correction.value),
                    measured_value=float(correction.measured_value),
                    refused=refused,
                )
            )
    return {channel: tuple(sorted(rows)) for channel, rows in sorted(grouped.items())}


@dataclass(frozen=True)
class CorrectionSetScales:
    """Serve the acquisition setting a read divides out, from a correction document.

    The same reads :class:`BlockScaleTable` serves, answered from the versioned
    correction set rather than from a table beside this module.  The document is the
    one that also carries the sensor gains, the pickup states and the exclusions, so
    the setting a channel was recorded at stops being a fact only this module knows.

    What divides a signal comes from
    :func:`~nova.calibrate.corrections.build_chain`, which orders the stages from the
    schema's own ranks -- this class states which stage it is drawing and never how
    the stages compose.  It draws the acquisition rung alone: the five promoted sensor
    gains in the same document are not removed on this path today, and quietly
    starting to remove them here would change every fit's amplitude while looking like
    a storage change.

    ``settings`` indexes the same corrections to answer what warranted a read that
    divided by nothing, which a chain cannot report because the correction it would
    describe is the one that did not apply.
    """

    document: CorrectionSet
    settings: Mapping[str, tuple[ChannelSetting, ...]] = field(default_factory=dict)

    @classmethod
    def create(cls, document: CorrectionSet) -> CorrectionSetScales:
        """Index a validated document for reading."""

        settings = _channel_settings(document)
        for channel, rows in settings.items():
            for first, second in zip(rows, rows[1:], strict=False):
                if second.pulse_start <= first.pulse_end:
                    raise BlockScaleError(
                        f"{channel!r} carries acquisition intervals "
                        f"{first.pulse_start}-{first.pulse_end} and "
                        f"{second.pulse_start}-{second.pulse_end}, which overlap, so a "
                        "shot in the overlap has two settings"
                    )
        return cls(document=document, settings=settings)

    @property
    def channels(self) -> tuple[str, ...]:
        """Return every channel the document carries a setting for."""

        return tuple(sorted(self.settings))

    @property
    def stepping(self) -> tuple[str, ...]:
        """Return the channels recorded at more than one setting."""

        return tuple(
            channel for channel in self.channels if len(self.settings[channel]) > 1
        )

    @property
    def corrected(self) -> tuple[str, ...]:
        """Return the channels some interval of which a read divides a rung out of."""

        return tuple(
            channel
            for channel in self.channels
            if any(
                not row.refused and row.rung != 1.0 for row in self.settings[channel]
            )
        )

    def correction(self, channel: str, shot: int) -> ScaleCorrection:
        """Return the setting a read of this channel on this shot may divide by."""

        shot = int(shot)
        rows = self.settings.get(channel, ())
        if not rows:
            return ScaleCorrection(channel, shot, 1.0, UNMEASURED)
        covering = [row for row in rows if row.covers(shot)]
        if covering:
            return self._covered(channel, shot, covering[0])
        before = [row for row in rows if row.pulse_end < shot]
        after = [row for row in rows if row.pulse_start > shot]
        if before and after:
            return ScaleCorrection(
                channel,
                shot,
                1.0,
                UNMEASURED,
                candidates=(before[-1].rung, after[0].rung),
            )
        nearest = before[-1] if before else after[0]
        return ScaleCorrection(
            channel, shot, 1.0, UNMEASURED, candidates=(nearest.rung,)
        )

    def _covered(
        self, channel: str, shot: int, setting: ChannelSetting
    ) -> ScaleCorrection:
        """Return the read for a shot an interval holds over."""

        if setting.refused:
            return ScaleCorrection(
                channel, shot, 1.0, REFUSED, candidates=(setting.measured_value,)
            )
        chain = build_chain(
            self.document,
            channel,
            pulse=shot,
            kinds=(CorrectionKind.acquisition_scale,),
        )
        step = chain.steps[0]
        return ScaleCorrection(
            channel, shot, step.value, MEASURED if step.measured else BRACKETED
        )

    def corrections(
        self, shot: int, channels: Iterable[str]
    ) -> tuple[ScaleCorrection, ...]:
        """Return one correction per channel for one shot, in channel order."""

        return tuple(
            self.correction(channel, shot) for channel in sorted(set(channels))
        )

    def normalise(
        self, shot: int, probes: Mapping[str, np.ndarray]
    ) -> tuple[dict[str, np.ndarray], tuple[ScaleCorrection, ...]]:
        """Divide one shot's probe channels by the setting each was recorded at."""

        rows = self.corrections(shot, probes)
        lookup = {row.channel: row for row in rows}
        return (
            {
                channel: lookup[channel].normalise(values)
                for channel, values in probes.items()
            },
            rows,
        )


def bracket_probe(
    brackets: Sequence[ScaleBracket],
    shots: Sequence[int],
    *,
    measured: Iterable[int] = (),
) -> int | None:
    """Return the archive shot whose reading would narrow the widest bracket most.

    A sweep that measures shots in archive order spends most of its reads inside
    blocks it has already established.  Bisecting instead -- reading the middle of
    the widest bracket still open -- costs a number of reads growing with the
    logarithm of a bracket's width rather than with the width, and one read serves
    every channel at once, because every channel's setting is recorded on the same
    shot.
    """

    done = {int(shot) for shot in measured}
    best: tuple[int, int] | None = None
    for bracket in brackets:
        inside = [shot for shot in bracket.unresolved(shots) if shot not in done]
        if not inside:
            continue
        centre = 0.5 * (bracket.before_shot + bracket.after_shot)
        choice = min(inside, key=lambda shot: (abs(shot - centre), shot))
        if best is None or bracket.width > best[0]:
            best = (bracket.width, choice)
    return None if best is None else best[1]


def pinning_summary(table: BlockScaleTable, shots: Sequence[int]) -> dict[str, Any]:
    """Report how tightly the archive pins every switch the table carries."""

    brackets = table.brackets()
    widths = [row.width for row in brackets]
    return {
        "brackets": [row.as_dict() for row in brackets],
        "corrected_channels": list(table.corrected),
        "median_width": float(np.median(widths)) if widths else 0.0,
        "pinned": sum(1 for row in brackets if row.pinned(shots)),
        "stepping_channels": list(table.stepping),
        "switch_count": len(brackets),
        "widest_width": max(widths) if widths else 0,
    }


PROMOTED_ROUTE = "far-field response ratio on plasma-free shots"
"""How every promoted block's setting was measured.

The ratio of a probe's recorded field to the described field, pooled over the samples
of a shot and over the shots of a block, taken only on probes standing further than
two winding-pack widths from every excited coil so that the ratio is a statement
about the channel and not about the near field of one coil.
"""

PROMOTED_PATH = Path(__file__).with_name("mast_block_scale.json")
"""Where the promoted table is carried.

Beside the module rather than in a runtime cache, because the read path applies it by
default: a table a consumer has to fetch is a table some consumer will read without,
and two runs disagreeing about whether a channel was halved is exactly the failure
this correction exists to remove.  The shot lists are what make it a file rather than
a literal -- each block names every shot its setting was measured on.
"""


@cache
def promoted_block_scales() -> BlockScaleTable:
    """Return the block table every read applies unless told otherwise.

    A missing file is an error rather than an empty table.  Silently reading the raw
    archive would make the correction vanish without a symptom, so the absence has to
    be louder than the presence.
    """

    if not PROMOTED_PATH.exists():
        raise BlockScaleError(
            f"no promoted block table at {PROMOTED_PATH}: every probe read applies "
            "it, so a missing table is a broken read path rather than an uncorrected "
            "one -- pass an empty BlockScaleTable to read the archive as published"
        )
    return BlockScaleTable.from_dict(json.loads(PROMOTED_PATH.read_text()))
