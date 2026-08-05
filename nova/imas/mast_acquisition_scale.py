"""Detect the discrete scale the acquisition applied to a probe channel per shot.

A probe channel's calibration is usually thought of as a property of the probe, so
one number describes it for the life of the machine.  These channels do not behave
that way.  Measured shot by shot against the described field, a subset of them sits
at unity for a block of shots, steps to almost exactly twice or half that, holds
there for another block, and steps back.  One channel reaches four.  The steps land
on a ladder of powers of two and their square roots, they recur at shared shot
boundaries, and they reverse -- and none of that is available to a physical
explanation.  A probe whose effective area changed does not change back, and no
sensor moves by a factor of exactly two.

What settles it is a control the data supplies for free.  At every boundary where
some channels step, most channels do not: three to six move while fifty hold at
unity on the same shots.  A wrong drive weight, a wrong coil geometry or a wrong
turn count would move every channel on that shot together, because they all read
the same currents through the same model.  Only something applied per channel,
downstream of the probe and upstream of the archive, can move a handful and leave
the rest alone.  That is an acquisition range setting, and the level-1 store has
not normalised it out.

The consequence is what makes this worth a module rather than a footnote.  A single
gain per channel, fitted over shots that straddle a step, is an average of two
discrete states weighted by however many shots each contributed -- a number that
describes no shot correctly and that moves when the shot selection moves.  So a
channel that steps must not be given a static calibration record at all: what it
needs is the scale for the block a shot belongs to, applied when the channel is
read.  A channel that does not step can carry one number, and this module's job is
to say which channels are which.

Nothing here fits geometry or promotes anything.  It takes a per-shot scale
measured elsewhere, finds where that scale changes, reports how well the changes
snap to the ladder, and reports the concurrency control beside them so a reader can
check the per-channel conclusion rather than take it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

STEP_RATIO = 1.4
"""Ratio between consecutive shots that counts as a step rather than scatter.

The ladder's smallest rung is a factor of the square root of two, about 1.41, and a
channel's shot-to-shot reproducibility on this cohort is a few percent.  Placing the
threshold just below the smallest rung separates the two without needing to know
which rung a step took.
"""

SCALE_LADDER = (0.25, 0.5, 1.0 / math.sqrt(2.0), 1.0, math.sqrt(2.0), 2.0, 4.0)
"""Rungs a range setting is expected to move between, as ratios.

Powers of two and their square roots, which is what a binary range ladder in an
acquisition chain offers.  Declared as a hypothesis rather than an assumption: every
detected step reports its distance from the nearest rung, so a set of steps that did
NOT land on this ladder would be visible as such instead of being rounded onto it.
"""

LADDER_TOLERANCE = 0.08
"""Fractional distance from a rung inside which a step is said to land on it.

Wide enough to absorb a channel's own few-percent calibration offset riding on top
of the range factor, narrow enough that the rungs -- separated by forty percent --
cannot be confused with each other.
"""

MINIMUM_BLOCK_SHOTS = 2
"""Shots a block needs before it is a block rather than one anomalous reading.

A single shot at a different scale is more likely a bad shot than a range change,
and treating it as a block would give a two-shot archive three scales.
"""

MINIMUM_HISTORY_SHOTS = 8
"""Shots a channel needs before its scale history means anything.

Below this a channel has too few blocks to distinguish stepping from scatter, and
reporting it either way would be a guess.
"""

CONCURRENCY_SHARE = 0.5
"""Share of channels that must hold still at a step for it to be per-channel.

The discriminator against every model-side cause.  A wrong drive weight or coil
geometry moves every channel that reads that shot, so a step where most channels
held is a statement about one channel's signal path.  A half is far below what the
archive shows -- fifty of fifty-five holding -- and far above what a model-side
error could leave untouched.
"""


class AcquisitionScaleError(ValueError):
    """Raised when a per-shot scale history cannot be read or classified."""


def nearest_rung(ratio: float) -> tuple[float, float]:
    """Return the ladder rung closest to a ratio, and the fractional distance."""

    if not math.isfinite(ratio) or ratio <= 0.0:
        raise AcquisitionScaleError(f"step ratio {ratio} is not a positive number")
    rung = min(SCALE_LADDER, key=lambda value: abs(math.log(ratio / value)))
    return rung, abs(ratio - rung) / rung


@dataclass(frozen=True, order=True)
class ScaleStep:
    """One change in the scale a channel was recorded at."""

    channel: str
    before_shot: int
    after_shot: int
    before_scale: float
    after_scale: float

    @property
    def ratio(self) -> float:
        """Return the factor the scale changed by."""

        if self.before_scale == 0.0:
            return math.inf
        return self.after_scale / self.before_scale

    @property
    def inverting(self) -> bool:
        """Return whether the two blocks read the described field opposite ways.

        A channel measured negative against the description on a run of shots is
        saying something about its polarity, or about having had too little signal to
        say anything -- and neither is a range setting, because the ladder is positive
        by declaration.  Such a step has no rung, which is a verdict rather than a
        failure to compute one.
        """

        return not (self.ratio > 0.0)

    @property
    def rung(self) -> float:
        """Return the ladder rung this step is closest to, if it has one."""

        return math.nan if self.inverting else nearest_rung(self.ratio)[0]

    @property
    def ladder_distance(self) -> float:
        """Return how far from that rung the step landed, as a fraction."""

        return math.inf if self.inverting else nearest_rung(self.ratio)[1]

    @property
    def on_ladder(self) -> bool:
        """Return whether the step landed on a rung of the declared ladder."""

        return self.ladder_distance <= LADDER_TOLERANCE

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical JSON representation."""

        return {
            "after_scale": self.after_scale,
            "after_shot": self.after_shot,
            "before_scale": self.before_scale,
            "before_shot": self.before_shot,
            "channel": self.channel,
            "inverting": self.inverting,
            "ladder_distance": (
                None if math.isinf(self.ladder_distance) else self.ladder_distance
            ),
            "on_ladder": self.on_ladder,
            "ratio": self.ratio,
            "rung": None if self.inverting else self.rung,
        }


@dataclass(frozen=True, order=True)
class ScaleBlock:
    """A run of shots over which one channel was recorded at one scale."""

    channel: str
    first_shot: int
    last_shot: int
    scale: float
    shot_count: int

    def covers(self, shot: int) -> bool:
        """Return whether this block is the one a shot belongs to."""

        return self.first_shot <= shot <= self.last_shot

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical JSON representation."""

        return {
            "channel": self.channel,
            "first_shot": self.first_shot,
            "last_shot": self.last_shot,
            "scale": self.scale,
            "shot_count": self.shot_count,
        }


def _per_shot(series: Mapping[int, Sequence[float]]) -> list[tuple[int, float]]:
    return sorted(
        (int(shot), float(np.median(values)))
        for shot, values in series.items()
        if len(values) > 0
    )


def scale_blocks(
    channel: str,
    series: Mapping[int, Sequence[float]],
    *,
    step_ratio: float = STEP_RATIO,
) -> tuple[ScaleBlock, ...]:
    """Split one channel's per-shot scale into runs of constant scale.

    A run ends where consecutive shots differ by more than ``step_ratio``.  A run
    shorter than :data:`MINIMUM_BLOCK_SHOTS` is folded back into its predecessor
    rather than kept, because one shot at an odd scale is a bad shot far more often
    than it is a range change that lasted one shot.

    Adjacent runs are then coalesced when their own scales agree.  One anomalous
    shot between two runs at the same scale splits the series on the way in and on
    the way out, leaving two blocks that describe the same setting -- and a channel
    with two blocks reads as stepping however alike they are.  That is the
    difference between reporting a genuine steady factor-of-two defect and refusing
    it, so the coalescing is not cosmetic.
    """

    rows = _per_shot(series)
    if not rows:
        return ()
    runs: list[list[tuple[int, float]]] = [[rows[0]]]
    for previous, current in zip(rows, rows[1:], strict=False):
        ratio = current[1] / previous[1] if previous[1] != 0.0 else math.inf
        if ratio > step_ratio or ratio < 1.0 / step_ratio:
            runs.append([current])
        else:
            runs[-1].append(current)
    merged: list[list[tuple[int, float]]] = []
    for run in runs:
        if merged and len(run) < MINIMUM_BLOCK_SHOTS:
            merged[-1].extend(run)
        else:
            merged.append(run)
    coalesced: list[list[tuple[int, float]]] = []
    for run in merged:
        if coalesced:
            previous = float(np.median([value for _, value in coalesced[-1]]))
            current = float(np.median([value for _, value in run]))
            ratio = current / previous if previous != 0.0 else math.inf
            if 1.0 / step_ratio <= ratio <= step_ratio:
                coalesced[-1].extend(run)
                continue
        coalesced.append(run)
    merged = coalesced
    return tuple(
        ScaleBlock(
            channel=channel,
            first_shot=run[0][0],
            last_shot=run[-1][0],
            scale=float(np.median([value for _, value in run])),
            shot_count=len(run),
        )
        for run in merged
    )


def scale_steps(blocks: Sequence[ScaleBlock]) -> tuple[ScaleStep, ...]:
    """Return the change between each pair of consecutive blocks."""

    return tuple(
        ScaleStep(
            channel=first.channel,
            before_shot=first.last_shot,
            after_shot=second.first_shot,
            before_scale=first.scale,
            after_scale=second.scale,
        )
        for first, second in zip(blocks, blocks[1:], strict=False)
    )


@dataclass(frozen=True, order=True)
class ChannelScaleHistory:
    """Everything one channel's per-shot scale says about how it was recorded."""

    channel: str
    blocks: tuple[ScaleBlock, ...]
    shot_count: int

    @property
    def steps(self) -> tuple[ScaleStep, ...]:
        """Return the scale changes this channel went through."""

        return scale_steps(self.blocks)

    @property
    def measured(self) -> bool:
        """Return whether enough shots back the history to classify it."""

        return self.shot_count >= MINIMUM_HISTORY_SHOTS

    @property
    def steady(self) -> bool:
        """Return whether one scale describes this channel for the whole archive."""

        return self.measured and len(self.blocks) <= 1

    @property
    def span(self) -> float:
        """Return the ratio of the largest block scale to the smallest."""

        if not self.blocks:
            return math.nan
        scales = [block.scale for block in self.blocks]
        smallest = min(scales)
        return max(scales) / smallest if smallest > 0.0 else math.inf

    @property
    def scale(self) -> float:
        """Return the one scale a steady channel carries.

        A stepping channel has no such number, and asking for it is a fault rather
        than a request for an average: the average describes no shot.
        """

        if not self.steady:
            raise AcquisitionScaleError(
                f"{self.channel!r} was recorded at {len(self.blocks)} different "
                "scales, so it has no single scale -- read the block a shot falls in"
            )
        return self.blocks[0].scale

    def scale_for(self, shot: int) -> float:
        """Return the scale the acquisition applied to this channel on one shot."""

        for block in self.blocks:
            if block.covers(shot):
                return block.scale
        raise AcquisitionScaleError(
            f"{self.channel!r} has no measured scale covering shot {shot}"
        )

    @property
    def on_ladder(self) -> bool:
        """Return whether every step this channel took landed on the ladder."""

        steps = self.steps
        return bool(steps) and all(step.on_ladder for step in steps)

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical JSON representation."""

        return {
            "blocks": [block.as_dict() for block in self.blocks],
            "channel": self.channel,
            "measured": self.measured,
            "on_ladder": self.on_ladder,
            "shot_count": self.shot_count,
            "span": None if math.isinf(self.span) else self.span,
            "steady": self.steady,
            "steps": [step.as_dict() for step in self.steps],
        }


def channel_histories(
    series: Mapping[str, Mapping[int, Sequence[float]]],
    *,
    step_ratio: float = STEP_RATIO,
) -> tuple[ChannelScaleHistory, ...]:
    """Build every channel's scale history from its per-shot scales."""

    return tuple(
        ChannelScaleHistory(
            channel=channel,
            blocks=scale_blocks(channel, rows, step_ratio=step_ratio),
            shot_count=len(rows),
        )
        for channel, rows in sorted(series.items())
    )


@dataclass(frozen=True, order=True)
class StepConcurrency:
    """How many channels moved at one boundary, and how many held still.

    The whole per-channel conclusion rests on this.  ``held`` counts the channels
    recorded on both shots whose scale did not change; a model-side error cannot
    produce a large one, because every channel reads the same currents through the
    same geometry.
    """

    before_shot: int
    after_shot: int
    moved: tuple[str, ...]
    held: tuple[str, ...]

    @property
    def shared(self) -> int:
        """Return how many channels were recorded on both shots."""

        return len(self.moved) + len(self.held)

    @property
    def held_share(self) -> float:
        """Return the share of shared channels that did not move."""

        return len(self.held) / self.shared if self.shared else math.nan

    @property
    def per_channel(self) -> bool:
        """Return whether enough channels held for the step to be per-channel."""

        return self.shared > 0 and self.held_share >= CONCURRENCY_SHARE

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical JSON representation."""

        return {
            "after_shot": self.after_shot,
            "before_shot": self.before_shot,
            "held": list(self.held),
            "held_share": self.held_share,
            "moved": list(self.moved),
            "per_channel": self.per_channel,
            "shared": self.shared,
        }


def step_concurrency(
    series: Mapping[str, Mapping[int, Sequence[float]]],
    steps: Iterable[ScaleStep],
    *,
    step_ratio: float = STEP_RATIO,
) -> tuple[StepConcurrency, ...]:
    """Measure, at every boundary a step was found on, which channels held still.

    The comparison runs over the two shots the step brackets, so it asks about the
    same acquisition and the same excitation and differs only in which channels
    changed.  Channels recorded on one shot and not the other are left out rather
    than counted as unchanged.
    """

    per_shot = {
        channel: dict(_per_shot(rows)) for channel, rows in sorted(series.items())
    }
    boundaries = sorted({(step.before_shot, step.after_shot) for step in steps})
    result = []
    for before, after in boundaries:
        moved, held = [], []
        for channel, rows in per_shot.items():
            first, second = rows.get(before), rows.get(after)
            if first is None or second is None or first == 0.0:
                continue
            ratio = second / first
            if ratio > step_ratio or ratio < 1.0 / step_ratio:
                moved.append(channel)
            else:
                held.append(channel)
        result.append(
            StepConcurrency(
                before_shot=before,
                after_shot=after,
                moved=tuple(sorted(moved)),
                held=tuple(sorted(held)),
            )
        )
    return tuple(result)


ROUTE_AGREEMENT = 0.03
"""Fractional agreement two independent routes need before a scale is promoted.

The safeguard that this promotion rule was not tuned to its own answer.  One route
pools a per-shot scale over the whole cohort and the other solves every channel at
once on a few dozen shots of a single campaign; they share no estimator and no shot
selection.  Three percent is inside the scatter either route reports and far below
the factor the range ladder moves by, so agreement at this level is a statement
about the channel and not about the method.
"""

SPLIT_HALF_TOLERANCE = 0.05
"""Fractional agreement a steady channel's two halves of shots must reach.

A channel called steady asserts one scale describes every shot it appears on, and
splitting its shots in two and comparing is that assertion's own test.  A channel
failing it is not steady, whatever the block finder said.
"""


@dataclass(frozen=True, order=True)
class SplitHalfCheck:
    """One steady channel's scale measured on each half of its shots."""

    channel: str
    early_scale: float
    late_scale: float
    early_shots: int
    late_shots: int

    @property
    def disagreement(self) -> float:
        """Return the fractional difference between the two halves."""

        centre = 0.5 * (abs(self.early_scale) + abs(self.late_scale))
        if centre <= 0.0:
            return math.inf
        return abs(self.early_scale - self.late_scale) / centre

    @property
    def holds(self) -> bool:
        """Return whether one scale really does describe both halves."""

        return (
            min(self.early_shots, self.late_shots) >= MINIMUM_BLOCK_SHOTS
            and self.disagreement <= SPLIT_HALF_TOLERANCE
        )

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical JSON representation."""

        return {
            "channel": self.channel,
            "disagreement": (
                None if math.isinf(self.disagreement) else self.disagreement
            ),
            "early_scale": self.early_scale,
            "early_shots": self.early_shots,
            "holds": self.holds,
            "late_scale": self.late_scale,
            "late_shots": self.late_shots,
        }


def split_half_check(
    channel: str,
    series: Mapping[int, Sequence[float]],
) -> SplitHalfCheck:
    """Measure a channel's scale on the earlier and later halves of its shots.

    Split by shot order rather than at random, because the thing being tested is
    whether the scale drifted or stepped -- a random split would average any
    time structure away and pass a channel that changed.
    """

    rows = _per_shot(series)
    if len(rows) < 2 * MINIMUM_BLOCK_SHOTS:
        return SplitHalfCheck(channel, math.nan, math.nan, len(rows), 0)
    middle = len(rows) // 2
    early = [value for _, value in rows[:middle]]
    late = [value for _, value in rows[middle:]]
    return SplitHalfCheck(
        channel=channel,
        early_scale=float(np.median(early)),
        late_scale=float(np.median(late)),
        early_shots=len(early),
        late_shots=len(late),
    )


@dataclass(frozen=True, order=True)
class PromotedScale:
    """A channel scale two independent routes agree on and both halves confirm."""

    channel: str
    scale: float
    independent_scale: float
    shot_count: int
    split_half: SplitHalfCheck

    @property
    def route_disagreement(self) -> float:
        """Return the fractional gap between the two routes."""

        centre = 0.5 * (abs(self.scale) + abs(self.independent_scale))
        if centre <= 0.0:
            return math.inf
        return abs(self.scale - self.independent_scale) / centre

    @property
    def corroborated(self) -> bool:
        """Return whether the second route agrees closely enough to promote."""

        return self.route_disagreement <= ROUTE_AGREEMENT

    @property
    def promoted(self) -> bool:
        """Return whether this scale enters the description."""

        return (
            self.corroborated and self.split_half.holds and abs(self.scale - 1.0) > 0.05
        )

    @property
    def interval(self) -> tuple[float, float]:
        """Return the interval the two routes and the two halves span."""

        values = (
            self.scale,
            self.independent_scale,
            self.split_half.early_scale,
            self.split_half.late_scale,
        )
        finite = [value for value in values if math.isfinite(value)]
        return (min(finite), max(finite)) if finite else (math.nan, math.nan)

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical JSON representation."""

        lower, upper = self.interval
        return {
            "channel": self.channel,
            "corroborated": self.corroborated,
            "independent_scale": self.independent_scale,
            "interval": [lower, upper],
            "promoted": self.promoted,
            "route_disagreement": self.route_disagreement,
            "scale": self.scale,
            "shot_count": self.shot_count,
            "split_half": self.split_half.as_dict(),
        }


def promote_scales(
    histories: Iterable[ChannelScaleHistory],
    series: Mapping[str, Mapping[int, Sequence[float]]],
    independent: Mapping[str, float],
) -> tuple[PromotedScale, ...]:
    """Assemble the promotion decision for every steady channel off unity.

    A stepping channel is not a candidate at all: it has no single scale, and the
    average one would report describes no shot.  Everything else must clear three
    gates -- off unity beyond the scatter, agreed by an independent route, and
    confirmed on both halves of its own shots.
    """

    result = []
    for row in histories:
        if not row.steady or row.channel not in independent:
            continue
        result.append(
            PromotedScale(
                channel=row.channel,
                scale=row.scale,
                independent_scale=float(independent[row.channel]),
                shot_count=row.shot_count,
                split_half=split_half_check(row.channel, series[row.channel]),
            )
        )
    return tuple(result)


def steady_channels(
    histories: Iterable[ChannelScaleHistory],
) -> tuple[ChannelScaleHistory, ...]:
    """Return the channels one number describes for the whole archive."""

    return tuple(row for row in histories if row.steady)


def stepping_channels(
    histories: Iterable[ChannelScaleHistory],
) -> tuple[ChannelScaleHistory, ...]:
    """Return the channels no single number describes."""

    return tuple(row for row in histories if row.measured and not row.steady)


def acquisition_record(
    histories: Sequence[ChannelScaleHistory],
    concurrency: Sequence[StepConcurrency],
) -> dict[str, Any]:
    """Assemble the run's record with its control and its ladder agreement."""

    stepping = stepping_channels(histories)
    steps = [step for row in stepping for step in row.steps]
    on_ladder = [step for step in steps if step.on_ladder]
    per_channel = [row for row in concurrency if row.per_channel]
    return {
        "concurrency": [row.as_dict() for row in concurrency],
        "concurrency_share": CONCURRENCY_SHARE,
        "histories": [row.as_dict() for row in histories],
        "ladder": list(SCALE_LADDER),
        "ladder_tolerance": LADDER_TOLERANCE,
        "step_count": len(steps),
        "step_ratio": STEP_RATIO,
        "steps_on_ladder": len(on_ladder),
        "steps_per_channel_controlled": len(per_channel),
        "steady_channels": sorted(row.channel for row in steady_channels(histories)),
        "stepping_channels": sorted(row.channel for row in stepping),
    }
