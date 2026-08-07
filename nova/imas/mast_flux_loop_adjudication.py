"""Adjudicate where each flux loop sits when the two sources place it differently.

The description and the reconstruction agree on most loops and disagree on a
handful, and the disagreements are not spread the way a survey error would be.
They cluster by family and, within a family, often by side: the P3 loops agree
above the midplane to a nanometre and differ by five and six millimetres below
it, and the P5 loops agree above and differ radially below.  Where a family's
two halves are exact reflections in one table and not in the other -- as the P2
loops are -- that is itself evidence, because a fixture pair installed
symmetrically is more likely than one built two millimetres asymmetric.

Position must be adjudicated through the *resolved* channel join, not through
proximity.  Matching a channel to a loop by nearest position collapses: six
centre-column channels share one table row within a nanometre, so a positional
join hands them all the same loop and any per-loop statement made through it is
about an arbitrary member of a degenerate set.  The reconstruction numbers its
loops in contiguous family blocks, and a channel's row is its family's block
start plus its own number less one, which is a bijection.  Everything here is
keyed on that.

Whether a disagreement *can* be settled is a computation.  A loop links the
total flux through its own contour, so moving it changes what it should read by
an amount the coil geometry fixes exactly: evaluate the flux at both candidate
positions and difference them, then compare that against the channel's own
quiescent scatter.  Whether it *is* settled is a fit, and the two must not be
confused -- a separation forty times the noise says the question is answerable,
not which answer is right.  So each candidate is scored against the measured
flux with one free scale per channel, which judges a position on the shape of
its response across coils and time rather than on an amplitude a loop's own gain
or a convention factor could supply.  A candidate promotes only when it wins on
shots the fit never saw; otherwise the loop stays explicitly dual-valued with
the measured separation recorded beside it, because "we could not tell" and "the
difference is below the noise" are different statements and only the second one
closes the question.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from nova.catalog.mast_geometry import loop_mount, position_mounts
from nova.imas.mast_solve_inputs import (
    LOOP_POSITION_TOLERANCE,
    SolveInputError,
    parse_loop_channel,
    reconstruction_loop_positions,
    reconstruction_loop_rows,
)
from nova.imas.mast_vacuum_cohort import FIELD_GROUP, SHOT_STORE
from nova.imas.mast_vacuum_response import loop_response_matrix

MIRROR_TOLERANCE = 1.0e-6
"""Metres two loops may miss being exact reflections of each other by.

Reflection is a property some of the described families have and others do not,
which is why the tolerance is this tight: the pairs that do reflect do so to the
last digit stored, so at a micron the test separates a table half built by
mirroring from one where both halves were surveyed.  Loops thirty microns from
reflecting are surveyed positions that happen to be nearly symmetric, and
reading them as a mirror would erase a real measurement.
"""

SEPARATION_MARGIN = 3.0
"""Multiples of a channel's scatter the candidates' flux difference must exceed.

Below this the two positions predict readings the channel cannot distinguish, so
a fit asked to choose between them is fitting noise and would return whichever
the residual happened to prefer.
"""

DECISION_MARGIN = 0.05
"""Fractional residual advantage a candidate needs before it is promoted.

Two positions a few millimetres apart give residuals that differ by a few
percent even when neither is right, because the part of the misfit a position
error cannot explain dominates both.  A twentieth is above that floor and far
below the advantage a genuinely wrong position shows.
"""

MINIMUM_HELD_OUT_SHOTS = 5
"""Shots outside the fit a candidate must win on to be promoted.

A single held-out shot makes the challenge a coin toss, and the loops here are
excited by whichever coils a campaign happened to run, so the challenge has to
average over a few.
"""


class LoopAdjudicationError(SolveInputError):
    """Raised when a loop position cannot be compared or dispositioned."""


class LoopDisposition(StrEnum):
    """What was decided about one loop's position."""

    AGREED = "agreed"
    PROMOTED = "promoted"
    DUAL_VALUED = "dual_valued"
    NO_DESCRIBED_COUNTERPART = "no_described_counterpart"
    NO_CHANNEL = "no_channel"


def described_loop_positions(geometry: Mapping[str, Any]) -> np.ndarray:
    """Return the described toroidally-closed loops' positions, in registry order."""

    loops = geometry["magnetics"]["flux_loops"]
    return np.asarray([[float(row[0]), float(row[1])] for row in loops], dtype=float)


def mirror_pairs(positions: np.ndarray) -> dict[int, int]:
    """Pair each loop with the one reflecting it in the midplane.

    Built from the positions themselves rather than from the names, so a family
    whose upper and lower members are numbered in opposite order still pairs
    correctly, and a loop with no reflection is simply absent from the result.

    A reflection must be UNIQUE to count.  The described table holds several loops
    at one position, so a loop can have more than one candidate reflection, and
    taking the first would make the pairing depend on table order and stop being
    an involution.  Mirroring is only used as evidence about how a table was
    built, and an ambiguous reflection is not evidence of anything.
    """

    pairs: dict[int, int] = {}
    for index, (radius, height) in enumerate(positions):
        if abs(height) <= MIRROR_TOLERANCE:
            continue
        matches = [
            other
            for other, (other_radius, other_height) in enumerate(positions)
            if other != index
            and abs(other_radius - radius) <= MIRROR_TOLERANCE
            and abs(other_height + height) <= MIRROR_TOLERANCE
        ]
        if len(matches) == 1:
            pairs[index] = matches[0]
    return pairs


def loop_flux_response(
    geometry: Mapping[str, Any],
    positions: np.ndarray,
) -> np.ndarray:
    """Return the flux each active coil links through a loop at each position.

    Row per position, column per active component in sorted order, in webers per
    ampere-turn.  The columns come from the same kernel and the same area
    weighting the probe response is built with, so the two routes cannot disagree
    about what a coil is.
    """

    return loop_response_matrix(
        geometry, positions, families=sorted(geometry["active_components"])
    )


@dataclass(frozen=True, order=True)
class ChannelJoin:
    """Whether one flux-loop channel reaches a described sensor, and why not.

    ``separation`` is how far the reconstruction places the channel from the
    nearest described loop, and ``described_on_mount`` how many loops the
    description carries on the coil the channel is mounted on.  Those two
    separate the only refusals that occur: a channel the description has no
    sensor for at all, and a channel whose two sources place it further apart
    than the catalogs agree to anywhere else.
    """

    channel: str
    mount: str
    described_index: int | None
    separation: float
    described_on_mount: int

    @property
    def served(self) -> bool:
        """Return whether this channel reaches a described loop."""

        return self.described_index is not None

    @property
    def cause(self) -> str:
        """Return why the join refused this channel, empty when it did not."""

        if self.served:
            return ""
        if self.described_on_mount == 0:
            return (
                f"the description carries no flux loop on {self.mount}, which "
                f"{self.channel} is mounted on"
            )
        return (
            f"the reconstruction places {self.channel} "
            f"{self.separation * 1e3:.0f} mm from the nearest of the "
            f"{self.described_on_mount} loops the description carries on "
            f"{self.mount}"
        )

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical JSON representation."""

        return {
            "cause": self.cause,
            "channel": self.channel,
            "described_index": self.described_index,
            "described_on_mount": self.described_on_mount,
            "mount": self.mount,
            "separation": self.separation,
            "served": self.served,
        }


def join_accounting(
    geometry: Mapping[str, Any],
    positions: np.ndarray,
    *,
    tolerance: float = LOOP_POSITION_TOLERANCE,
) -> tuple[ChannelJoin, ...]:
    """Account for every flux-loop channel against the described loop set.

    One row per reconstruction channel, served or refused, because the count of
    channels a description can receive is the measurable consequence of a
    position correction and a count with no denominator states nothing.  The
    refusals are separated by the coil each channel is mounted on rather than by
    a list, so a description that gains or loses a loop moves the accounting
    without anything here being rewritten.
    """

    described = described_loop_positions(geometry)
    outlines = geometry["active_components"]
    described_mounts = position_mounts(described, outlines)
    rows = reconstruction_loop_rows()
    targets = _claimed_targets(described, positions, rows, tolerance)

    accounting = []
    for channel, row in sorted(rows.items()):
        mount = loop_mount(channel.upper())
        if row >= positions.shape[0]:
            separation = math.inf
        else:
            separation = float(
                np.hypot(
                    described[:, 0] - positions[row, 0],
                    described[:, 1] - positions[row, 1],
                ).min()
            )
        accounting.append(
            ChannelJoin(
                channel=channel,
                mount=mount,
                described_index=targets.get(channel),
                separation=separation,
                described_on_mount=sum(one == mount for one in described_mounts),
            )
        )
    return tuple(accounting)


def _claimed_targets(
    described: np.ndarray,
    positions: np.ndarray,
    rows: Mapping[str, int],
    tolerance: float,
) -> dict[str, int]:
    """Assign each channel a described loop, nearest first and never twice.

    Two loops of a pair sit fifteen millimetres apart, so a many-to-one match
    would put two channels on one sensor and every per-loop statement made
    through it would be about an arbitrary member of the pair.
    """

    candidates = []
    for channel, row in rows.items():
        if row >= positions.shape[0]:
            continue
        distance = np.hypot(
            described[:, 0] - positions[row, 0], described[:, 1] - positions[row, 1]
        )
        candidates.append((float(distance.min()), channel, distance))
    assigned: dict[str, int] = {}
    taken: set[int] = set()
    for nearest, channel, distance in sorted(candidates, key=lambda row: row[0]):
        if nearest > tolerance:
            continue
        for index in np.argsort(distance):
            if distance[index] > tolerance:
                break
            if int(index) in taken:
                continue
            assigned[channel] = int(index)
            taken.add(int(index))
            break
    return assigned


@dataclass(frozen=True, order=True)
class LoopComparison:
    """One channel's two candidate positions and whether data can choose.

    ``flux_separation`` is the largest difference in linked flux the two
    candidate positions predict over the excitations the calibration cohort
    drove, in webers.  ``scatter`` is the channel's own quiescent scatter in the
    same unit, so the ratio of the two is what decides identifiability.
    """

    channel: str
    described_r: float
    described_z: float
    reconstruction_r: float
    reconstruction_z: float
    displacement: float
    mirror_displacement: float
    flux_separation: float
    scatter: float
    described_index: int | None

    @property
    def agrees(self) -> bool:
        """Return whether the two sources place the loop at one point."""

        return self.displacement <= MIRROR_TOLERANCE

    @property
    def separable(self) -> bool:
        """Return whether the channel can tell the two candidates apart."""

        if self.scatter <= 0.0 or not math.isfinite(self.flux_separation):
            return False
        return self.flux_separation > SEPARATION_MARGIN * self.scatter

    @property
    def separation_ratio(self) -> float:
        """Return the flux difference in units of the channel's own scatter."""

        if self.scatter <= 0.0:
            return math.inf
        return self.flux_separation / self.scatter

    @property
    def mirror_symmetric(self) -> bool:
        """Return whether the described position reflects its own counterpart.

        The reflection is checked on the DESCRIBED table, because that is the
        table whose internal consistency is evidence: a set built by mirroring a
        measured half carries the measurement in both halves, whereas a set whose
        two halves disagree by millimetres on some fixtures and not others is
        carrying a transcription in at least one of them.
        """

        return self.mirror_displacement <= MIRROR_TOLERANCE

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical JSON representation."""

        return {
            "agrees": self.agrees,
            "channel": self.channel,
            "described_index": self.described_index,
            "described_r": self.described_r,
            "described_z": self.described_z,
            "displacement": self.displacement,
            "flux_separation": self.flux_separation,
            "mirror_displacement": self.mirror_displacement,
            "mirror_symmetric": self.mirror_symmetric,
            "reconstruction_r": self.reconstruction_r,
            "reconstruction_z": self.reconstruction_z,
            "scatter": self.scatter,
            "separable": self.separable,
            "separation_ratio": (
                None if math.isinf(self.separation_ratio) else self.separation_ratio
            ),
        }


def compare_loop_positions(
    geometry: Mapping[str, Any],
    shot: int,
    scatter: Mapping[str, float],
    *,
    drive_currents: Mapping[str, float] | None = None,
    store: Path | str = SHOT_STORE,
    tolerance: float = LOOP_POSITION_TOLERANCE,
) -> tuple[LoopComparison, ...]:
    """Compare both sources' position for every channel the join resolves.

    ``drive_currents`` scales each active component's flux contribution when the
    two candidates' separation is computed, so the separation answers for the
    excitations the archive actually drove rather than for one ampere-turn.  Left
    out, every coil is given one ampere-turn and the separation is a per-unit
    number.
    """

    described = described_loop_positions(geometry)
    reconstruction = reconstruction_loop_positions(shot, store=store)
    rows = reconstruction_loop_rows()
    mirrors = mirror_pairs(described)
    families = sorted(geometry["active_components"])
    weights = np.asarray(
        [float((drive_currents or {}).get(family, 1.0)) for family in families],
        dtype=float,
    )

    comparisons = []
    for channel, index in sorted(rows.items()):
        if index >= reconstruction.shape[0]:
            continue
        source = reconstruction[index]
        distances = np.hypot(described[:, 0] - source[0], described[:, 1] - source[1])
        nearest = int(np.argmin(distances))
        if distances[nearest] > tolerance:
            comparisons.append(
                LoopComparison(
                    channel=channel,
                    described_r=math.nan,
                    described_z=math.nan,
                    reconstruction_r=float(source[0]),
                    reconstruction_z=float(source[1]),
                    displacement=float(distances[nearest]),
                    mirror_displacement=math.inf,
                    flux_separation=math.nan,
                    scatter=float(scatter.get(channel, 0.0)),
                    described_index=None,
                )
            )
            continue
        target = described[nearest]
        candidates = np.asarray([target, source], dtype=float)
        flux = loop_flux_response(geometry, candidates) @ weights
        mirror = mirrors.get(nearest)
        if mirror is None:
            mirror_gap = math.inf
        else:
            reflected = described[mirror]
            mirror_gap = float(
                math.hypot(target[0] - reflected[0], target[1] + reflected[1])
            )
        comparisons.append(
            LoopComparison(
                channel=channel,
                described_r=float(target[0]),
                described_z=float(target[1]),
                reconstruction_r=float(source[0]),
                reconstruction_z=float(source[1]),
                displacement=float(distances[nearest]),
                mirror_displacement=mirror_gap,
                flux_separation=float(abs(flux[1] - flux[0])),
                scatter=float(scatter.get(channel, 0.0)),
                described_index=nearest,
            )
        )
    return tuple(comparisons)


@dataclass(frozen=True, order=True)
class CandidateFit:
    """How well each candidate position reproduces one channel's measured flux.

    Residuals are pooled root-mean-square over shots, in webers, with one free
    scale per channel per candidate so the comparison is on the response's shape
    and not on its amplitude.  ``held_out`` residuals come from shots that took no
    part in setting the scale.
    """

    channel: str
    shot_count: int
    held_out_count: int
    described_residual: float
    reconstruction_residual: float
    described_held_out: float
    reconstruction_held_out: float
    described_scale: float
    reconstruction_scale: float

    @property
    def margin(self) -> float:
        """Return the fractional advantage of the better candidate on held-out data.

        Positive means the described position wins; the sign is what promotes,
        and the magnitude is what has to clear the declared margin.
        """

        pair = (self.described_held_out, self.reconstruction_held_out)
        best = min(pair)
        worst = max(pair)
        if not math.isfinite(worst) or worst <= 0.0:
            return 0.0
        signed = 1.0 if self.described_held_out < self.reconstruction_held_out else -1.0
        return signed * (worst - best) / worst

    @property
    def agrees_in_sample(self) -> bool:
        """Return whether training and held-out prefer the same candidate."""

        train = self.described_residual < self.reconstruction_residual
        test = self.described_held_out < self.reconstruction_held_out
        return train == test

    @property
    def prefers_described(self) -> bool:
        """Return whether the described position is the one the data chooses."""

        return self.margin > 0.0

    @property
    def decided(self) -> bool:
        """Return whether the fit chose a candidate at all.

        Three things have to hold together: enough held-out shots to be a
        prediction, the same winner in and out of sample, and an advantage larger
        than the declared margin.  Any one of them failing leaves the loop
        dual-valued rather than promoting the marginally better number.
        """

        return (
            self.held_out_count >= MINIMUM_HELD_OUT_SHOTS
            and self.agrees_in_sample
            and abs(self.margin) >= DECISION_MARGIN
        )

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical JSON representation."""

        return {
            "agrees_in_sample": self.agrees_in_sample,
            "channel": self.channel,
            "decided": self.decided,
            "described_held_out": self.described_held_out,
            "described_residual": self.described_residual,
            "described_scale": self.described_scale,
            "held_out_count": self.held_out_count,
            "margin": self.margin,
            "prefers_described": self.prefers_described,
            "reconstruction_held_out": self.reconstruction_held_out,
            "reconstruction_residual": self.reconstruction_residual,
            "reconstruction_scale": self.reconstruction_scale,
            "shot_count": self.shot_count,
        }


def dispose(
    comparison: LoopComparison,
    fit: CandidateFit | None = None,
) -> LoopDisposition:
    """Return one loop's disposition under the recorded criterion.

    A loop the sources agree on needs no decision.  A loop they disagree on is
    promoted only when a fit the loop's own measurements drove chose a candidate
    on shots it never saw -- separability alone is not a verdict, it is a
    statement that a verdict is reachable.  Everything else keeps both positions
    on the record, because choosing on an undecided difference would write a
    false precision into a sensor pose the artifact's identity then carries.
    """

    if comparison.described_index is None:
        return LoopDisposition.NO_DESCRIBED_COUNTERPART
    if comparison.agrees:
        return LoopDisposition.AGREED
    if fit is not None and fit.decided:
        return LoopDisposition.PROMOTED
    return LoopDisposition.DUAL_VALUED


@dataclass(frozen=True)
class LoopLedger:
    """Every loop's disposition, and the described loops no channel reaches."""

    comparisons: tuple[LoopComparison, ...]
    unreached_indices: tuple[int, ...]
    unreached_positions: tuple[tuple[float, float], ...]
    fits: tuple[CandidateFit, ...] = ()

    def fit(self, channel: str) -> CandidateFit | None:
        """Return one channel's candidate fit, if it was fitted."""

        for row in self.fits:
            if row.channel == channel:
                return row
        return None

    def disposition(self, channel: str) -> LoopDisposition:
        """Return one channel's disposition."""

        for row in self.comparisons:
            if row.channel == channel:
                return dispose(row, self.fit(channel))
        return LoopDisposition.NO_CHANNEL

    @property
    def counts(self) -> dict[str, int]:
        """Count loops by disposition, every disposition present as a key."""

        counts = {str(state): 0 for state in LoopDisposition}
        for row in self.comparisons:
            counts[str(dispose(row, self.fit(row.channel)))] += 1
        counts[str(LoopDisposition.NO_CHANNEL)] = len(self.unreached_indices)
        return counts

    @property
    def promoted(self) -> tuple[tuple[str, str], ...]:
        """Return each promoted channel and which source's position won."""

        result = []
        for row in self.comparisons:
            fit = self.fit(row.channel)
            if dispose(row, fit) is not LoopDisposition.PROMOTED or fit is None:
                continue
            result.append(
                (
                    row.channel,
                    "described" if fit.prefers_described else "reconstruction",
                )
            )
        return tuple(result)

    @property
    def disagreeing(self) -> tuple[LoopComparison, ...]:
        """Return the loops the two sources place differently."""

        return tuple(
            row
            for row in self.comparisons
            if row.described_index is not None and not row.agrees
        )

    @property
    def asymmetric_families(self) -> tuple[str, ...]:
        """Return the families whose disagreement sits on one side only.

        A displacement appearing on the lower members of a family whose upper
        members agree is the signature of a table transcribed rather than
        mirrored, and it is the shape of the disagreement in this archive.
        """

        sides: dict[str, set[str]] = {}
        for row in self.disagreeing:
            family, _ = parse_loop_channel(row.channel)
            sides.setdefault(family[:-1] if family != "cc" else family, set()).add(
                family[-1] if family != "cc" else "c"
            )
        return tuple(sorted(name for name, seen in sides.items() if len(seen) == 1))

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical JSON representation."""

        return {
            "asymmetric_families": list(self.asymmetric_families),
            "decision_margin": DECISION_MARGIN,
            "fits": [row.as_dict() for row in self.fits],
            "promoted": [list(row) for row in self.promoted],
            "comparisons": [
                dict(
                    row.as_dict(),
                    disposition=str(dispose(row, self.fit(row.channel))),
                )
                for row in self.comparisons
            ],
            "counts": self.counts,
            "mirror_tolerance": MIRROR_TOLERANCE,
            "separation_margin": SEPARATION_MARGIN,
            "unreached_indices": list(self.unreached_indices),
            "unreached_positions": [list(row) for row in self.unreached_positions],
        }


def build_ledger(
    geometry: Mapping[str, Any],
    comparisons: Sequence[LoopComparison],
    fits: Sequence[CandidateFit] = (),
) -> LoopLedger:
    """Assemble the ledger, naming the described loops no channel resolves onto."""

    described = described_loop_positions(geometry)
    reached = {
        row.described_index for row in comparisons if row.described_index is not None
    }
    unreached = tuple(
        index for index in range(described.shape[0]) if index not in reached
    )
    return LoopLedger(
        comparisons=tuple(sorted(comparisons)),
        unreached_indices=unreached,
        unreached_positions=tuple(
            (float(described[index, 0]), float(described[index, 1]))
            for index in unreached
        ),
        fits=tuple(sorted(fits)),
    )


@dataclass(frozen=True)
class LoopShotResidual:
    """One shot's scaled residual for one channel under one candidate position."""

    shot: int
    channel: str
    candidate: str
    scale: float
    residual: float
    signal: float


def loop_shot_residuals(
    geometry: Mapping[str, Any],
    comparisons: Sequence[LoopComparison],
    shot: int,
    weights: Mapping[str, float],
    *,
    store: Path | str = SHOT_STORE,
    stride: int = 8,
) -> tuple[LoopShotResidual, ...]:
    """Score both candidate positions against one shot's measured loop flux.

    One scale per channel per candidate is fitted and divided out, so what the
    residual compares is how the predicted flux varies with the coils and with
    time -- a loop's own gain, and any factor of two pi between the store's
    convention and the kernel's, cancel and cannot decide a position.
    """

    import zarr

    from nova.imas.mast_vacuum_cohort import read_shot_waveforms

    disagreeing = [row for row in comparisons if row.described_index is not None]
    if not disagreeing:
        return ()
    families = sorted(geometry["active_components"])
    scale = np.asarray(
        [float(weights.get(family, 0.0)) for family in families], dtype=float
    )
    described = np.asarray(
        [[row.described_r, row.described_z] for row in disagreeing], dtype=float
    )
    reconstruction = np.asarray(
        [[row.reconstruction_r, row.reconstruction_z] for row in disagreeing],
        dtype=float,
    )
    response = {
        "described": loop_flux_response(geometry, described),
        "reconstruction": loop_flux_response(geometry, reconstruction),
    }

    waveforms = read_shot_waveforms(shot, store=store)
    fields = zarr.open_group(f"{Path(store)}/{shot}.zarr", mode="r")[FIELD_GROUP]
    samples = np.flatnonzero(waveforms.sample_mask)[::stride]
    quiet = waveforms.baseline_mask
    if samples.size == 0:
        return ()
    drive = np.zeros((waveforms.time.size, len(families)), dtype=float)
    for column, family in enumerate(families):
        values = waveforms.drives.get(family)
        if values is not None:
            drive[:, column] = np.nan_to_num(values) * scale[column]

    results = []
    for index, row in enumerate(disagreeing):
        if row.channel not in fields:
            continue
        measured = np.asarray(fields[row.channel][...], dtype=float)
        if measured.shape != waveforms.time.shape:
            continue
        finite = np.isfinite(measured)
        if not (finite & quiet).any():
            continue
        centred = measured - float(np.mean(measured[finite & quiet]))
        keep = samples[finite[samples]]
        if keep.size < 100:
            continue
        observed = centred[keep]
        for candidate, table in sorted(response.items()):
            prediction = (drive[keep] * table[index, :]).sum(axis=1)
            power = float(np.dot(prediction, prediction))
            if power <= 0.0:
                continue
            fitted = float(np.dot(prediction, observed) / power)
            residual = observed - fitted * prediction
            results.append(
                LoopShotResidual(
                    shot=shot,
                    channel=row.channel,
                    candidate=candidate,
                    scale=fitted,
                    residual=float(np.sqrt(np.mean(residual**2))),
                    signal=float(np.sqrt(np.mean(observed**2))),
                )
            )
    return tuple(results)


def fit_candidate_positions(
    training: Iterable[LoopShotResidual],
    held_out: Iterable[LoopShotResidual],
) -> tuple[CandidateFit, ...]:
    """Pool per-shot residuals into one candidate comparison per channel."""

    def pooled(
        rows: Iterable[LoopShotResidual],
    ) -> dict[tuple[str, str], tuple[float, float, int]]:
        grouped: dict[tuple[str, str], list[LoopShotResidual]] = {}
        for row in rows:
            grouped.setdefault((row.channel, row.candidate), []).append(row)
        return {
            key: (
                float(np.sqrt(np.mean([r.residual**2 for r in group]))),
                float(np.mean([r.scale for r in group])),
                len(group),
            )
            for key, group in grouped.items()
        }

    train = pooled(training)
    test = pooled(held_out)
    channels = sorted({channel for channel, _ in train})
    fits = []
    for channel in channels:
        keys = [(channel, "described"), (channel, "reconstruction")]
        if not all(key in train for key in keys):
            continue
        described, reconstruction = (train[key] for key in keys)
        held = [test.get(key) for key in keys]
        fits.append(
            CandidateFit(
                channel=channel,
                shot_count=described[2],
                held_out_count=min((row[2] for row in held if row), default=0),
                described_residual=described[0],
                reconstruction_residual=reconstruction[0],
                described_held_out=held[0][0] if held[0] else math.inf,
                reconstruction_held_out=held[1][0] if held[1] else math.inf,
                described_scale=described[1],
                reconstruction_scale=reconstruction[1],
            )
        )
    return tuple(fits)


def measure_loop_scatter(
    shots: Iterable[int],
    *,
    store: Path | str = SHOT_STORE,
    minimum_samples: int = 200,
) -> dict[str, float]:
    """Measure each loop channel's scatter on shots with nothing driven.

    A loop integrator drifts, so the scatter is taken about a straight line
    through the window rather than about its mean -- the same treatment the field
    probes' floor was measured with, so the two floors are comparable numbers.
    """

    import zarr

    root = Path(store)
    pooled: dict[str, list[float]] = {}
    for shot in shots:
        try:
            fields = zarr.open_group(f"{root}/{shot}.zarr", mode="r")[FIELD_GROUP]
        except Exception:  # noqa: BLE001 - a shot the store cannot open is skipped
            continue
        if "time" not in fields:
            continue
        time = np.asarray(fields["time"][...], dtype=float)
        for channel in sorted(fields.keys()):
            try:
                parse_loop_channel(channel)
            except SolveInputError:
                continue
            values = np.asarray(fields[channel][...], dtype=float)
            if values.shape != time.shape:
                continue
            keep = np.isfinite(values) & np.isfinite(time)
            if int(keep.sum()) < minimum_samples:
                continue
            slope, offset = np.polyfit(time[keep], values[keep], 1)
            residual = values[keep] - (slope * time[keep] + offset)
            pooled.setdefault(channel, []).append(float(np.std(residual)))
    return {
        channel: float(np.median(values)) for channel, values in sorted(pooled.items())
    }
