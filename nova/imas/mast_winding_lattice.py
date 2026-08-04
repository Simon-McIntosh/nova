"""Place the turns inside a winding-pack outline and test the placement on data.

A winding-pack outline says where the copper is.  It does not say how the turns
are stacked inside it, and the two statements have different fields: spreading one
ampere-turn uniformly over the footprint puts the current at the footprint's
centroid, while stacking it on a lattice puts it wherever the lattice's own
centroid falls.  On these coils the two disagree by about a millimetre, which is
nothing to a probe across the machine and the leading term for a probe one pack
width away.

The lattice is not free.  A pack is wound in layers of equal pitch, so the turn
positions are a regular grid; the grid spans the pack, so its pitch is the pack's
extent divided by the layer and turn counts; and the outline's chamfer is a
cross-over cut-out, so a grid position whose centre falls outside the outline is a
position the winding cannot occupy.  Everything except the two counts and how far
the turns stand off the pack face therefore follows from geometry already carried
as identity.

That leaves a real prediction rather than a free parameter, and it is worth
stating because it is the reason to believe any of this: on all four of these coils
a grid of twenty-four positions has exactly one position cut by the chamfer, so
the layout returns *twenty-three* turns -- the count the archive publishes.  Three
grid shapes do it (four by six, six by four, eight by three) and the vacated
position is the outboard corner furthest from the midplane in every one, which is
where a wound pack takes its cross-over.  Nothing was fitted to obtain that; it
falls out of the chamfer.

What is fitted is the fill fraction, the share of the pack's extent the turn
centres span.  A pack has ground insulation between the outermost turn and the
pack face, so the lattice is slightly smaller than the outline, and how much
smaller sets how far the lattice's centroid moves.  One number per coil family,
bounded by the physics -- a fill above one puts copper outside the pack and a fill
far below one puts the turns in a huddle no winding machine makes.

Reading the fit
---------------
A sustained single-coil shot holds one current flat while the probes watch, so
every probe sees the same waveform and differs only in the factor in front of it.
The turn layout changes those factors by a fraction of a percent and changes them
*differently* from probe to probe, while a turn-count error, a supply calibration
or an acquisition gain changes all of them together.  So the layout is read from
the PATTERN across the array with one amplitude per shot divided out, and never
from the amplitude -- which is what makes this a field-shape measurement and not
a second attempt at a turn count.

Because the model is one waveform times one factor per probe, a shot's whole
contribution reduces to a few sums over its samples (:class:`ShotMoments`).  Every
candidate layout is then scored by algebra on those sums instead of by re-reading
the archive, which is what makes a scan over fill fractions and grid shapes cheap
enough to run as a scan rather than a search.

Residuals are whitened by each channel's own measured quiescent scatter, so a
channel that reads a hundred and thirty millitesla and a channel that reads two do
not compete on amplitude, and the score means the same thing as the misfit the
frozen gate reports.

The gain that has to be removed first
-------------------------------------
There is an obstruction in the way, and it is not small.  A probe's reading of a
sustained shot is one number, and a channel gain multiplies that number just as
the turn layout does -- so on a single coil's shots the two are the same parameter
and no amount of data separates them.  It is not a hypothetical: on these coils the
nearest probes come back wanting amplitudes of 0.50, 1.15 and 1.22, which is
between five and fifty times the effect a layout produces.  Fitting a layout
against uncalibrated channels measures the calibration.

What separates them is that a gain belongs to the channel and a layout belongs to
the coil.  A probe one pack width from P5 upper is nine pack widths from P4 lower,
and at nine widths every layout predicts the same field -- so that probe's gain can
be measured on the shots where it stands clear of everything driven, and carried
into the shots where it stands close.  :func:`calibrate_array` does exactly that:
it solves the per-channel gains and per-shot amplitudes on the FAR field, with the
uniform columns and no layout in the model at all, and hands them to the near-field
scan as fixed numbers.  The layout is then the only thing left free, and it is
being asked to explain a pattern that a gain cannot have caused.

These gains are a nuisance calibration internal to this measurement -- enough to
get the layout out from under them, and not a promoted sensor ledger, which is a
per-probe pose and area adjudication in its own right.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import shapely

from nova.biot.polygon import polygon_greens
from nova.imas.mast_vacuum_cohort import (
    ERROR_FIELD_CHANNELS,
    EXCITATION_CURRENT,
    CohortError,
    ShotWaveforms,
)
from nova.imas.mast_vacuum_response import (
    MINIMUM_STANDOFF,
    ProbeTarget,
    ResponseModel,
    baseline_window,
    coil_sections,
    energised_families,
    excited_families,
)

LATTICE_SHAPES = ((4, 6), (6, 4), (8, 3))
"""Grid shapes whose chamfered position count matches the published turn count.

Enumerated over every shape up to twelve by twelve on the four coils this applies
to, these three are the only ones that return twenty-three turns on all four, and
they are the only ones offered to the fit.  A shape is a hypothesis about how many
layers the pack has, so the data is allowed to choose between them; it is not
allowed to choose a shape that contradicts the count.
"""

FILL_BOUNDS = (0.80, 1.00)
"""Range of the pack extent the turn centres may span.

The upper bound is geometric: a fill above one places turn centres outside the
outline, which is copper where the pack says there is none.  The lower bound is
the winding -- eighty percent of a hundred and fifty millimetre pack leaves a
fifteen millimetre insulation margin on each face, already thicker than any ground
wrap these coils carry, so a fit that wants less than this is not describing
insulation.
"""

NEAR_FIELD_STANDOFF = MINIMUM_STANDOFF
"""Pack widths inside which a probe reads the layout and not just the total.

The same cut the turn fit uses, read the other way round.  A turn count is fitted
on probes standing OUTSIDE this radius because inside it a uniform-density outline
predicts the wrong field; a layout is tested on the probes INSIDE it because
outside it every layout predicts the same field.  The two uses are complementary
and share one number so they cannot drift apart.
"""

MINIMUM_TARGET_SHARE = 0.90
"""Share of predicted signal power the driven coil must carry at a probe.

These shots rarely hold one coil in isolation -- a vertical-field pair idles at a
few kiloamperes on most of them -- and a probe whose reading is a tenth somebody
else's field cannot report the driven coil's turn layout, because the other coil's
own description error enters at the same size as the effect being measured.
"""

MINIMUM_PATTERN_PROBES = 4
"""Probes a shot must keep inside the near field to report a pattern.

One amplitude is divided out of every shot, so a shot with a single admissible
probe contributes no shape information at all and a shot with two contributes one
degree of freedom.  Four is the smallest set on which a disagreement between
layouts can outvote a single misbehaving channel.
"""


class LatticeError(CohortError):
    """Raised when a turn layout cannot be built or scored."""


@dataclass(frozen=True, order=True)
class TurnLattice:
    """A regular grid of turn positions inside a winding-pack outline.

    ``columns`` and ``rows`` count the turns across and along the pack, and
    ``fill`` is the share of the pack's extent the turn centres span, measured
    about the pack's own centre so a fill below one contracts the layout
    symmetrically and leaves an insulation margin at both faces.
    """

    columns: int
    rows: int
    fill: float = 1.0

    def __post_init__(self) -> None:
        """Reject a grid or a fill that cannot describe a wound pack."""

        if self.columns < 1 or self.rows < 1:
            raise LatticeError(
                f"grid {self.columns}x{self.rows} has no positions to wind"
            )
        if not math.isfinite(self.fill) or self.fill <= 0.0:
            raise LatticeError(f"fill {self.fill} is not a positive fraction")

    @property
    def positions(self) -> int:
        """Return how many grid positions the pack is divided into."""

        return self.columns * self.rows

    def grid(self, vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the uncontracted grid centres over the outline's extent.

        These are the positions the chamfer test is applied to, and they do not
        move with ``fill``: which position the cross-over vacates is a fact about
        the pack's corner, not about how thick its insulation is.  Letting the
        contraction pull a vacated position back inside the outline would make the
        turn count a function of the fitted parameter.
        """

        r0, z0, r1, z1 = _bounds(vertices)
        step_r, step_z = (r1 - r0) / self.columns, (z1 - z0) / self.rows
        index_r, index_z = np.meshgrid(
            np.arange(self.columns), np.arange(self.rows), indexing="ij"
        )
        return (
            r0 + (index_r.ravel() + 0.5) * step_r,
            z0 + (index_z.ravel() + 0.5) * step_z,
        )

    def occupied(self, vertices: np.ndarray) -> np.ndarray:
        """Mark the grid positions the outline admits a turn at."""

        grid_r, grid_z = self.grid(vertices)
        outline = shapely.Polygon(vertices)
        return np.asarray(
            [
                outline.contains(shapely.Point(r, z))
                for r, z in zip(grid_r, grid_z, strict=True)
            ],
            dtype=bool,
        )

    def turn_count(self, vertices: np.ndarray) -> int:
        """Return how many turns this grid fits inside the outline."""

        return int(np.count_nonzero(self.occupied(vertices)))

    def centres(self, vertices: np.ndarray) -> np.ndarray:
        """Return the occupied turn centres, contracted by the fill fraction."""

        grid_r, grid_z = self.grid(vertices)
        keep = self.occupied(vertices)
        r0, z0, r1, z1 = _bounds(vertices)
        middle_r, middle_z = 0.5 * (r0 + r1), 0.5 * (z0 + z1)
        return np.column_stack(
            (
                middle_r + self.fill * (grid_r[keep] - middle_r),
                middle_z + self.fill * (grid_z[keep] - middle_z),
            )
        )

    def sections(self, vertices: np.ndarray) -> tuple[np.ndarray, ...]:
        """Return one conductor cross-section per turn.

        Each turn is given the cell it occupies rather than a zero-width
        filament, so the field stays finite for a probe that sits close to the
        pack and the layout's second moment is the pack's own and not an artefact
        of collapsing the copper to points.
        """

        r0, z0, r1, z1 = _bounds(vertices)
        half_r = 0.5 * self.fill * (r1 - r0) / self.columns
        half_z = 0.5 * self.fill * (z1 - z0) / self.rows
        return tuple(
            np.array(
                [
                    [r - half_r, z - half_z],
                    [r + half_r, z - half_z],
                    [r + half_r, z + half_z],
                    [r - half_r, z + half_z],
                ]
            )
            for r, z in self.centres(vertices)
        )

    def centroid(self, vertices: np.ndarray) -> tuple[float, float]:
        """Return the current centroid this layout puts the ampere-turns at."""

        centres = self.centres(vertices)
        return float(centres[:, 0].mean()), float(centres[:, 1].mean())

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical JSON representation."""

        return {"columns": self.columns, "fill": float(self.fill), "rows": self.rows}


def _bounds(vertices: np.ndarray) -> tuple[float, float, float, float]:
    """Return the outline's bounding extent, rejecting a degenerate outline."""

    outline = shapely.Polygon(vertices)
    r0, z0, r1, z1 = outline.bounds
    if not (r1 > r0 and z1 > z0):
        raise LatticeError("winding-pack outline has no extent to wind turns in")
    return float(r0), float(z0), float(r1), float(z1)


def admissible_shapes(
    vertices: np.ndarray,
    turns: int,
    *,
    shapes: Sequence[tuple[int, int]] = LATTICE_SHAPES,
) -> tuple[tuple[int, int], ...]:
    """Return the offered grid shapes whose chamfered count matches ``turns``."""

    return tuple(
        shape
        for shape in shapes
        if TurnLattice(*shape).turn_count(vertices) == int(turns)
    )


def search_shapes(
    vertices: np.ndarray,
    turns: int,
    *,
    limit: int = 12,
) -> tuple[tuple[int, int], ...]:
    """Return every grid up to ``limit`` per side whose count matches ``turns``.

    Kept separate from :func:`admissible_shapes` because it is the search that
    justified :data:`LATTICE_SHAPES` rather than a step in any fit, and a record
    that states which shapes were offered should be able to show which were
    rejected.
    """

    return tuple(
        (columns, rows)
        for columns in range(1, limit + 1)
        for rows in range(1, limit + 1)
        if TurnLattice(columns, rows).turn_count(vertices) == int(turns)
    )


def section_column(
    targets: Sequence[ProbeTarget],
    sections: Iterable[np.ndarray],
) -> np.ndarray:
    """Field at each probe per ampere-turn, from a set of conductor sections.

    The sections share the ampere-turns equally, which is what a series-wound
    pack does, so the column is the mean of the sections' single-turn fields and
    is directly comparable with the uniform-density column the outline gives.
    """

    target_r = np.asarray([target.r for target in targets], dtype=float)
    target_z = np.asarray([target.z for target in targets], dtype=float)
    cosine = np.asarray([target.radial_cosine for target in targets], dtype=float)
    sine = np.asarray([target.axial_sine for target in targets], dtype=float)
    total = np.zeros(target_r.shape, dtype=float)
    count = 0
    for vertices in sections:
        _, radial, axial = polygon_greens(target_r, target_z, vertices)
        total += cosine * radial + sine * axial
        count += 1
    if count == 0:
        raise LatticeError("no conductor section to couple to the probes")
    return total / count


def lattice_column(
    targets: Sequence[ProbeTarget],
    vertices: np.ndarray,
    lattice: TurnLattice,
) -> np.ndarray:
    """Field at each probe per ampere-turn, from one turn layout."""

    return section_column(targets, lattice.sections(vertices))


def translated_section(vertices: np.ndarray, offset: Sequence[float]) -> np.ndarray:
    """Return the outline rigidly displaced by ``offset``.

    The control the layout is tested against.  A layout that improves a
    prediction has moved the current centroid and changed the higher moments at
    the same time; displacing the outline moves the centroid and leaves the
    moments alone, so comparing the two says which of the layout's two effects
    the probes actually read.
    """

    shift = np.asarray(offset, dtype=float)
    if shift.shape != (2,):
        raise LatticeError(f"displacement {offset!r} is not an (r, z) pair")
    return np.asarray(vertices, dtype=float) + shift


def uniform_column(
    targets: Sequence[ProbeTarget],
    vertices: np.ndarray,
) -> np.ndarray:
    """Field at each probe per ampere-turn, current uniform over the outline."""

    return section_column(targets, (np.asarray(vertices, dtype=float),))


def error_field_quiescent(
    peaks: Mapping[str, float],
    absent: Sequence[str] = (),
    *,
    channels: Sequence[str] = ERROR_FIELD_CHANNELS,
    threshold: float = EXCITATION_CURRENT,
) -> dict[str, str]:
    """Report each error-field channel's state on one shot.

    A channel reading below the excitation floor is ``quiescent``, one above it
    is ``driven``, and one the store did not record is ``unmeasured`` -- never
    quiescent, because an absent channel is a channel nobody looked at.  The
    states are returned per channel rather than reduced to a verdict so a record
    can show what was screened and not only that a screen ran.
    """

    absent_set = set(absent)
    state: dict[str, str] = {}
    for channel in channels:
        if channel in absent_set:
            state[channel] = "unmeasured"
            continue
        peak = peaks.get(channel)
        if peak is None or not math.isfinite(float(peak)):
            state[channel] = "unmeasured"
        elif abs(float(peak)) >= threshold:
            state[channel] = "driven"
        else:
            state[channel] = "quiescent"
    return state


def passes_error_field_screen(state: Mapping[str, str]) -> bool:
    """Return whether every error-field channel was measured and quiescent."""

    return bool(state) and all(value == "quiescent" for value in state.values())


@dataclass(frozen=True)
class ShotMoments:
    """One shot's waveforms reduced to what any turn layout needs from them.

    ``drive_gram`` is the matrix of drive-channel inner products and
    ``probe_moment`` the drive channels projected onto each probe's offset-removed
    reading; ``probe_square`` is each probe's own power.  Together they reproduce
    the least-squares residual of any linear combination of the drive columns
    exactly, so a layout scan never re-reads the archive.

    The gram is carried PER CHANNEL, not once for the shot, because the channels do
    not share a sample set: a third of a probe's samples can be missing while its
    neighbour's are complete, and folding those samples in as zeroes biases the
    fitted amplitude down by exactly the missing fraction -- two thirds of the
    samples returning two thirds of the amplitude, a thirty percent error sitting
    on top of a one percent measurement.

    ``standoff`` and ``target_share`` travel with the moments because which probes
    are in the near field, and whether the driven coil dominates them, are
    properties of the shot's excitation and cannot be recovered later.

    ``rows`` is each channel's row in the response the moments were reduced
    against.  A shot records its own subset of the array in its own order, so the
    two orderings are not the same and a score that indexes a coupling matrix by
    the shot's ordering silently pairs every probe with another probe's field.

    Two standoffs are carried.  ``standoff`` is the distance to the driven coil,
    which says whether a probe can read that coil's layout; ``screen_standoff`` is
    the distance to the nearest coil the shot excited at all, which says whether
    the uniform columns describe the probe well enough to calibrate against.  They
    differ whenever a shot drove more than one coil, which is most of them.
    """

    shot: int
    family: str
    families: tuple[str, ...]
    channels: tuple[str, ...]
    rows: np.ndarray
    drive_gram: np.ndarray
    probe_moment: np.ndarray
    probe_square: np.ndarray
    standoff: np.ndarray
    screen_standoff: np.ndarray
    target_share: np.ndarray
    samples_used: np.ndarray
    sample_count: int
    scatter: np.ndarray

    def near_field(
        self,
        *,
        standoff: float = NEAR_FIELD_STANDOFF,
        share: float = MINIMUM_TARGET_SHARE,
    ) -> np.ndarray:
        """Mark the probes close enough to the driven coil, and dominated by it."""

        return (self.standoff < standoff) & (self.target_share >= share)

    def far_field(
        self,
        *,
        standoff: float = NEAR_FIELD_STANDOFF,
    ) -> np.ndarray:
        """Mark the probes standing clear of every coil the shot excited."""

        return self.screen_standoff >= standoff


def reduce_shot(
    waveforms: ShotWaveforms,
    model: ResponseModel,
    family: str,
    weights: Mapping[str, float],
    scatter: Mapping[str, float],
    *,
    stride: int = 1,
    minimum_baseline: int = 20,
) -> ShotMoments:
    """Reduce one shot to the sums a layout comparison reads.

    Every energised coil enters as its own drive column carrying its promoted
    weight, so a coil idling beside the driven one is predicted rather than
    ignored, and ``target_share`` records how much of each probe's predicted power
    the driven coil owns so a probe the neighbour dominates can be dropped.
    """

    if family not in model.families:
        raise LatticeError(f"response carries no coil {family!r}")
    quiet = baseline_window(waveforms)
    if int(quiet.sum()) < minimum_baseline:
        raise LatticeError(
            f"shot {waveforms.shot} has {int(quiet.sum())} pre-excitation samples, "
            f"below the {minimum_baseline} needed to measure probe offsets"
        )
    energised = energised_families(waveforms, model.families)
    families = tuple(dict.fromkeys((family, *energised)))
    missing = [name for name in families if name not in weights]
    if missing:
        raise LatticeError(f"no promoted drive weight for {missing}")

    samples = np.flatnonzero(waveforms.sample_mask)[::stride]
    if samples.size == 0:
        raise LatticeError(f"shot {waveforms.shot} admits no samples")
    drive = np.zeros((samples.size, len(families)), dtype=float)
    for column, name in enumerate(families):
        values = waveforms.drives.get(name)
        if values is not None:
            drive[:, column] = np.nan_to_num(values)[samples] * float(weights[name])

    columns = [model.families.index(name) for name in families]
    target = families.index(family)
    response = model.response[:, columns]
    standoff_all = model.standoff[:, columns]
    excited = excited_families(waveforms, model.families)
    screened = [model.families.index(name) for name in excited] or columns
    screen_all = model.standoff[:, screened].min(axis=1)
    index = {probe.channel: row for row, probe in enumerate(model.targets)}

    channels: list[str] = []
    rows: list[int] = []
    gram: list[np.ndarray] = []
    moment: list[np.ndarray] = []
    square: list[float] = []
    stand: list[float] = []
    screen: list[float] = []
    share: list[float] = []
    used: list[int] = []
    noise: list[float] = []
    for channel, signal in sorted(waveforms.probes.items()):
        row = index.get(channel)
        if row is None or channel not in scatter:
            continue
        finite = np.isfinite(signal)
        if not finite[quiet].any():
            continue
        take = finite[samples]
        if int(take.sum()) < minimum_baseline:
            continue
        seen = drive[take, :]
        observed = signal[samples][take] - float(np.mean(signal[quiet & finite]))
        power = (seen * response[row, :]) ** 2
        total = float(power.sum())
        channels.append(channel)
        rows.append(row)
        gram.append(seen.T @ seen)
        moment.append(seen.T @ observed)
        square.append(float(observed @ observed))
        stand.append(float(standoff_all[row, target]))
        screen.append(float(screen_all[row]))
        share.append(float(power[:, target].sum() / total) if total > 0.0 else 0.0)
        used.append(int(take.sum()))
        noise.append(float(scatter[channel]))
    if not channels:
        raise LatticeError(f"shot {waveforms.shot} kept no probe the store recorded")

    return ShotMoments(
        shot=waveforms.shot,
        family=family,
        families=families,
        channels=tuple(channels),
        rows=np.asarray(rows, dtype=int),
        drive_gram=np.stack(gram),
        probe_moment=np.vstack(moment),
        probe_square=np.asarray(square, dtype=float),
        standoff=np.asarray(stand, dtype=float),
        screen_standoff=np.asarray(screen, dtype=float),
        target_share=np.asarray(share, dtype=float),
        samples_used=np.asarray(used, dtype=int),
        sample_count=int(samples.size),
        scatter=np.asarray(noise, dtype=float),
    )


@dataclass(frozen=True)
class HypothesisScore:
    """How well one turn layout reproduces one shot's pattern across the array.

    ``amplitude`` is the single scale the shot's whole prediction was allowed, so
    a layout is never credited for getting the level right; ``residual`` is what
    is left after that scale, whitened by each channel's own quiescent scatter and
    reported per probe in ``per_channel`` so a promotion can be checked for having
    paid for one probe with another.
    """

    shot: int
    family: str
    channels: tuple[str, ...]
    amplitude: float
    residual: float
    signal: float
    samples: int = 0
    per_channel: dict[str, float] = field(default_factory=dict)

    @property
    def probe_count(self) -> int:
        """Return how many probes the pattern was scored on."""

        return len(self.channels)


def _projections(
    moments: ShotMoments,
    columns: Mapping[str, np.ndarray],
    rows: np.ndarray,
    gains: Mapping[str, float] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return each selected probe's model-observation cross term and model power."""

    coupling = np.column_stack([columns[name] for name in moments.families])[
        moments.rows[rows], :
    ]
    if gains is not None:
        scale = np.asarray(
            [gains.get(moments.channels[row], 1.0) for row in rows], dtype=float
        )
        coupling = coupling * scale[:, None]
    return (
        np.einsum("pc,pc->p", moments.probe_moment[rows], coupling),
        np.einsum("pc,pcd,pd->p", coupling, moments.drive_gram[rows], coupling),
    )


def score_hypothesis(
    moments: ShotMoments,
    columns: Mapping[str, np.ndarray],
    *,
    select: np.ndarray | None = None,
    gains: Mapping[str, float] | None = None,
    amplitude: float | None = None,
) -> HypothesisScore:
    """Score one layout on one shot, with the shot's amplitude divided out.

    ``columns`` maps each energised coil to its field per ampere-turn at every
    probe in the model's order; the layout under test supplies the driven coil's
    column and the rest keep whatever the caller gave them, which is how a change
    confined to one coil is measured without re-deriving the others.

    ``gains`` applies a per-channel multiplier and ``amplitude`` fixes the shot's
    scale instead of fitting it.  Passing both is how a layout is judged with
    nothing left free: any pattern the layout does not explain then shows up in the
    residual rather than being absorbed by a rescaling.
    """

    keep = np.ones(len(moments.channels), dtype=bool) if select is None else select
    if not keep.any():
        raise LatticeError(f"shot {moments.shot} keeps no probe to score")
    rows = np.flatnonzero(keep)
    weight = 1.0 / moments.scatter[rows] ** 2

    predicted_moment, predicted_power = _projections(moments, columns, rows, gains)
    cross = float(weight @ predicted_moment)
    power = float(weight @ predicted_power)
    observed = float(weight @ moments.probe_square[rows])
    if power <= 0.0:
        raise LatticeError(f"shot {moments.shot} predicts no signal at its probes")

    amplitude = cross / power if amplitude is None else float(amplitude)
    samples = float(moments.samples_used[rows].sum())
    total = observed - 2.0 * amplitude * cross + amplitude**2 * power
    per_channel = {
        moments.channels[row]: float(
            math.sqrt(
                max(
                    moments.probe_square[row]
                    - 2.0 * amplitude * predicted_moment[position]
                    + amplitude**2 * predicted_power[position],
                    0.0,
                )
                / moments.samples_used[row]
            )
        )
        for position, row in enumerate(rows)
    }
    return HypothesisScore(
        shot=moments.shot,
        family=moments.family,
        channels=tuple(moments.channels[row] for row in rows),
        amplitude=amplitude,
        residual=math.sqrt(max(total, 0.0) / samples),
        signal=math.sqrt(observed / samples),
        samples=int(samples),
        per_channel=per_channel,
    )


@dataclass(frozen=True)
class ArrayCalibration:
    """Per-channel gains and per-shot amplitudes, measured off the far field.

    Solved with the uniform columns and no layout in the model, on the probes that
    stand clear of every coil a shot excited -- where a layout is invisible and any
    disagreement therefore belongs to the channel or the shot.  ``channel_shots``
    records how many shots each gain rests on, because a gain from one shot is
    that shot's amplitude wearing a channel's name.
    """

    gains: dict[str, float]
    amplitudes: dict[int, float]
    channel_shots: dict[str, int]
    residual: float
    iterations: int

    def constrained(self, minimum_shots: int) -> dict[str, float]:
        """Return the gains resting on enough shots, others left at unity."""

        return {
            channel: value
            if self.channel_shots.get(channel, 0) >= minimum_shots
            else 1.0
            for channel, value in self.gains.items()
        }

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical JSON representation."""

        return {
            "amplitudes": {
                str(k): float(v) for k, v in sorted(self.amplitudes.items())
            },
            "channel_shots": dict(sorted(self.channel_shots.items())),
            "gains": {k: float(v) for k, v in sorted(self.gains.items())},
            "iterations": self.iterations,
            "residual": float(self.residual),
        }


def calibrate_array(
    moments: Sequence[ShotMoments],
    columns: Mapping[str, np.ndarray],
    *,
    standoff: float = NEAR_FIELD_STANDOFF,
    iterations: int = 40,
    tolerance: float = 1.0e-9,
) -> ArrayCalibration:
    """Solve per-channel gains and per-shot amplitudes on the far field.

    The product of a channel gain and a shot amplitude is all the data constrains,
    so the gains are normalised to unit median after every sweep and the overall
    level lives in the amplitudes.  Alternating the two updates is exact in each
    one and converges in a handful of sweeps because the system is rank one.
    """

    if not moments:
        raise LatticeError("no shot to calibrate the array on")
    selections = {
        record.shot: record.far_field(standoff=standoff) for record in moments
    }
    usable = [record for record in moments if selections[record.shot].any()]
    if not usable:
        raise LatticeError(
            f"no probe stands {standoff} pack widths clear of any excited coil"
        )
    projections = {
        record.shot: _projections(
            record, columns, np.flatnonzero(selections[record.shot]), None
        )
        for record in usable
    }
    channels = sorted(
        {
            record.channels[row]
            for record in usable
            for row in np.flatnonzero(selections[record.shot])
        }
    )
    counts = {
        channel: sum(
            1
            for record in usable
            if channel
            in {record.channels[row] for row in np.flatnonzero(selections[record.shot])}
        )
        for channel in channels
    }

    gains = {channel: 1.0 for channel in channels}
    amplitudes = {record.shot: 1.0 for record in usable}
    previous = math.inf
    swept = 0
    for swept in range(1, iterations + 1):
        for record in usable:
            rows = np.flatnonzero(selections[record.shot])
            weight = 1.0 / record.scatter[rows] ** 2
            scale = np.asarray(
                [gains[record.channels[row]] for row in rows], dtype=float
            )
            cross, power = projections[record.shot]
            top = float(weight @ (scale * cross))
            bottom = float(weight @ (scale**2 * power))
            if bottom > 0.0:
                amplitudes[record.shot] = top / bottom
        top = {channel: 0.0 for channel in channels}
        bottom = {channel: 0.0 for channel in channels}
        for record in usable:
            rows = np.flatnonzero(selections[record.shot])
            cross, power = projections[record.shot]
            level = amplitudes[record.shot]
            for position, row in enumerate(rows):
                channel = record.channels[row]
                top[channel] += level * cross[position]
                bottom[channel] += level**2 * power[position]
        for channel in channels:
            if bottom[channel] > 0.0:
                gains[channel] = top[channel] / bottom[channel]
        middle = float(np.median(np.asarray(list(gains.values()), dtype=float)))
        if middle > 0.0:
            gains = {channel: value / middle for channel, value in gains.items()}
            amplitudes = {shot: value * middle for shot, value in amplitudes.items()}
        scores = [
            score_hypothesis(
                record,
                columns,
                select=selections[record.shot],
                gains=gains,
                amplitude=amplitudes[record.shot],
            )
            for record in usable
        ]
        residual = pooled_residual(scores)
        if abs(previous - residual) <= tolerance * max(residual, 1.0e-12):
            previous = residual
            break
        previous = residual
    return ArrayCalibration(
        gains=gains,
        amplitudes=amplitudes,
        channel_shots=counts,
        residual=float(previous),
        iterations=swept,
    )


@dataclass(frozen=True)
class FillProfile:
    """The fill fraction a coil's near-field pattern prefers, and how sharply.

    ``interval`` is a spread over shots and not a formal error bar: the residual
    these shots leave is an order above the sensor floor, so it is dominated by
    what the description still gets wrong rather than by noise, and a curvature
    interval computed as if it were noise would claim a precision the data does
    not have.  The shot-to-shot spread of the individually preferred fills is the
    honest statement of how well the cohort pins the number.
    """

    family: str
    shape: tuple[int, int]
    fill: float
    residual: float
    uniform_residual: float
    interval: tuple[float, float]
    per_shot_fill: dict[int, float] = field(default_factory=dict)
    profile: dict[float, float] = field(default_factory=dict)

    @property
    def improvement(self) -> float:
        """Return the share of the uniform-density residual the layout removes."""

        if self.uniform_residual <= 0.0:
            return 0.0
        return float(1.0 - self.residual / self.uniform_residual)

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical JSON representation."""

        return {
            "family": self.family,
            "fill": float(self.fill),
            "improvement": float(self.improvement),
            "interval": [float(self.interval[0]), float(self.interval[1])],
            "per_shot_fill": {str(k): float(v) for k, v in self.per_shot_fill.items()},
            "profile": {f"{k:.4f}": float(v) for k, v in sorted(self.profile.items())},
            "residual": float(self.residual),
            "shape": list(self.shape),
            "uniform_residual": float(self.uniform_residual),
        }


def fill_grid(
    *,
    bounds: tuple[float, float] = FILL_BOUNDS,
    step: float = 0.005,
) -> tuple[float, ...]:
    """Return the fill fractions a profile is evaluated at."""

    lower, upper = bounds
    count = int(round((upper - lower) / step)) + 1
    return tuple(float(lower + index * step) for index in range(count))


def pooled_residual(scores: Sequence[HypothesisScore]) -> float:
    """Return the whitened residual of a set of shots, pooled by probe-sample."""

    weight = np.asarray(
        [score.samples or score.probe_count for score in scores], dtype=float
    )
    square = np.asarray([score.residual**2 for score in scores], dtype=float)
    total = float(weight.sum())
    if total <= 0.0:
        raise LatticeError("no scored probe to pool")
    return float(math.sqrt(float(weight @ square) / total))


def profile_fill(
    moments: Sequence[ShotMoments],
    targets: Sequence[ProbeTarget],
    vertices: np.ndarray,
    baseline: Mapping[str, np.ndarray],
    *,
    shape: tuple[int, int],
    fills: Sequence[float] | None = None,
    standoff: float = NEAR_FIELD_STANDOFF,
    share: float = MINIMUM_TARGET_SHARE,
    minimum_probes: int = MINIMUM_PATTERN_PROBES,
    gains: Mapping[str, float] | None = None,
    amplitudes: Mapping[int, float] | None = None,
) -> FillProfile:
    """Scan the fill fraction against a cohort's near-field patterns.

    ``baseline`` carries the uniform-density column for every coil; the driven
    coil's column is replaced at each fill and the rest are left alone, so the
    profile answers for the layout and nothing else.  ``gains`` and ``amplitudes``
    come from the far-field calibration; supplying both leaves the layout as the
    only free thing in the comparison.
    """

    if not moments:
        raise LatticeError("no shot to profile a fill fraction on")
    family = moments[0].family
    usable = [
        record
        for record in moments
        if int(record.near_field(standoff=standoff, share=share).sum())
        >= minimum_probes
    ]
    if not usable:
        raise LatticeError(
            f"no {family} shot keeps {minimum_probes} probes inside "
            f"{standoff} pack widths that it also dominates"
        )
    selections = {
        record.shot: record.near_field(standoff=standoff, share=share)
        for record in usable
    }

    def scored(
        record: ShotMoments, columns: Mapping[str, np.ndarray]
    ) -> HypothesisScore:
        return score_hypothesis(
            record,
            columns,
            select=selections[record.shot],
            gains=gains,
            amplitude=None if amplitudes is None else amplitudes.get(record.shot),
        )

    uniform = pooled_residual([scored(record, baseline) for record in usable])

    profile: dict[float, float] = {}
    per_shot: dict[int, dict[float, float]] = {record.shot: {} for record in usable}
    for value in fills if fills is not None else fill_grid():
        lattice = TurnLattice(shape[0], shape[1], value)
        columns = dict(baseline)
        columns[family] = lattice_column(targets, vertices, lattice)
        scores = [scored(record, columns) for record in usable]
        profile[value] = pooled_residual(scores)
        for record, entry in zip(usable, scores, strict=True):
            per_shot[record.shot][value] = entry.residual

    best = min(profile, key=lambda value: profile[value])
    preferred = {
        shot: min(curve, key=lambda value: curve[value])
        for shot, curve in per_shot.items()
        if curve
    }
    spread = np.asarray(sorted(preferred.values()), dtype=float)
    interval = (
        (float(np.quantile(spread, 0.16)), float(np.quantile(spread, 0.84)))
        if spread.size > 1
        else (float(best), float(best))
    )
    return FillProfile(
        family=family,
        shape=shape,
        fill=float(best),
        residual=float(profile[best]),
        uniform_residual=float(uniform),
        interval=interval,
        per_shot_fill=preferred,
        profile=profile,
    )


@dataclass(frozen=True)
class DisplacementProfile:
    """Where a coil's near probes want its current centroid, layout aside.

    A rigid displacement of the outline is a strictly larger family than any turn
    layout inside it: a layout can only move the centroid by a fraction of a
    millimetre and only along the direction its vacated position points, while this
    scan moves it anywhere.  So the scan bounds what ANY intra-pack redistribution
    could buy, and its answer is decisive both ways.  If four identically-wound
    coils agree on a displacement, something in the winding description is wrong by
    that much; if they disagree, the near-probe misfit is not in the winding at all
    and no layout will remove it.
    """

    family: str
    offset: tuple[float, float]
    residual: float
    uniform_residual: float
    reach: float
    grid: dict[tuple[float, float], float] = field(default_factory=dict)

    @property
    def improvement(self) -> float:
        """Return the share of the near-field residual the displacement removes."""

        if self.uniform_residual <= 0.0:
            return 0.0
        return float(1.0 - self.residual / self.uniform_residual)

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical JSON representation."""

        return {
            "family": self.family,
            "improvement": float(self.improvement),
            "offset_mm": [1.0e3 * self.offset[0], 1.0e3 * self.offset[1]],
            "reach_mm": 1.0e3 * self.reach,
            "residual": float(self.residual),
            "uniform_residual": float(self.uniform_residual),
        }


def profile_displacement(
    moments: Sequence[ShotMoments],
    targets: Sequence[ProbeTarget],
    vertices: np.ndarray,
    baseline: Mapping[str, np.ndarray],
    *,
    reach: float = 5.0e-3,
    steps: int = 11,
    standoff: float = NEAR_FIELD_STANDOFF,
    share: float = MINIMUM_TARGET_SHARE,
    minimum_probes: int = MINIMUM_PATTERN_PROBES,
    gains: Mapping[str, float] | None = None,
    amplitudes: Mapping[int, float] | None = None,
) -> DisplacementProfile:
    """Scan a rigid displacement of one coil's outline against its near probes."""

    if not moments:
        raise LatticeError("no shot to profile a displacement on")
    family = moments[0].family
    selections = {
        record.shot: record.near_field(standoff=standoff, share=share)
        for record in moments
    }
    usable = [
        record
        for record in moments
        if int(selections[record.shot].sum()) >= minimum_probes
    ]
    if not usable:
        raise LatticeError(f"no {family} shot keeps a near-field pattern")

    def pooled(columns: Mapping[str, np.ndarray]) -> float:
        return pooled_residual(
            [
                score_hypothesis(
                    record,
                    columns,
                    select=selections[record.shot],
                    gains=gains,
                    amplitude=None
                    if amplitudes is None
                    else amplitudes.get(record.shot),
                )
                for record in usable
            ]
        )

    uniform = pooled(baseline)
    offsets = np.linspace(-reach, reach, steps)
    grid: dict[tuple[float, float], float] = {}
    for shift_r in offsets:
        for shift_z in offsets:
            columns = dict(baseline)
            columns[family] = uniform_column(
                targets, translated_section(vertices, (shift_r, shift_z))
            )
            grid[(float(shift_r), float(shift_z))] = pooled(columns)
    best = min(grid, key=lambda key: grid[key])
    return DisplacementProfile(
        family=family,
        offset=best,
        residual=grid[best],
        uniform_residual=uniform,
        reach=reach,
        grid=grid,
    )


def channel_deltas(
    moments: Sequence[ShotMoments],
    baseline: Mapping[str, np.ndarray],
    proposed: Mapping[str, np.ndarray],
    *,
    select: str = "all",
    standoff: float = NEAR_FIELD_STANDOFF,
    share: float = MINIMUM_TARGET_SHARE,
    gains: Mapping[str, float] | None = None,
    amplitudes: Mapping[int, float] | None = None,
) -> dict[str, dict[str, float]]:
    """Return each channel's whitened residual before and after a layout change.

    ``select`` chooses the probe set the amplitude is fitted on -- ``near`` for the
    probes that read the layout, ``far`` for those that do not, ``all`` for every
    probe the shot recorded.  A promotion has to survive the ``all`` reading,
    because a layout that helps the near probes by tilting the whole array's
    amplitude has not improved the description.
    """

    rows: dict[str, dict[str, list[float]]] = {}
    for record in moments:
        if select == "near":
            mask = record.near_field(standoff=standoff, share=share)
        elif select == "far":
            mask = record.far_field(standoff=standoff)
        elif select == "all":
            mask = np.ones(len(record.channels), dtype=bool)
        else:
            raise LatticeError(f"unknown probe selection {select!r}")
        if not mask.any():
            continue
        level = None if amplitudes is None else amplitudes.get(record.shot)
        before = score_hypothesis(
            record, baseline, select=mask, gains=gains, amplitude=level
        )
        after = score_hypothesis(
            record, proposed, select=mask, gains=gains, amplitude=level
        )
        for channel in before.channels:
            entry = rows.setdefault(channel, {"before": [], "after": []})
            entry["before"].append(before.per_channel[channel])
            entry["after"].append(after.per_channel[channel])
    result: dict[str, dict[str, float]] = {}
    for channel, entry in sorted(rows.items()):
        before = float(np.sqrt(np.mean(np.asarray(entry["before"]) ** 2)))
        after = float(np.sqrt(np.mean(np.asarray(entry["after"]) ** 2)))
        result[channel] = {
            "after": after,
            "before": before,
            "delta": after - before,
            "improvement": 0.0 if before <= 0.0 else 1.0 - after / before,
            "shots": float(len(entry["before"])),
        }
    return result


def baseline_columns(
    model: ResponseModel,
    geometry: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """Return every coil's uniform-density column, keyed by family.

    Read from the response the turn fit already built rather than recomputed, so
    a layout comparison cannot accidentally change the coils it is not testing.
    """

    sections = coil_sections(geometry)
    missing = [name for name in model.families if name not in sections]
    if missing:
        raise LatticeError(f"registry carries no active component {missing}")
    return {
        name: model.response[:, column].copy()
        for column, name in enumerate(model.families)
    }


def excitation_summary(
    waveforms: ShotWaveforms,
    model: ResponseModel,
    family: str,
) -> dict[str, Any]:
    """Describe what one shot drove, for the record that reports its use."""

    excited = excited_families(waveforms, model.families)
    return {
        "drove_target": family in excited,
        "energised": list(energised_families(waveforms, model.families)),
        "excited": list(excited),
        "sample_count": int(waveforms.sample_count),
        "shot": int(waveforms.shot),
    }
