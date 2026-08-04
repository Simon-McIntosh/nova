"""What the vacuum cohort established, pinned so the artifact can be rebuilt.

The fit that produced these numbers reads seventeen thousand shots to build its
census and then several hundred waveforms, which is not something artifact
authoring can do.  The results are therefore recorded here as data, with enough
provenance attached that each one can be traced to the shots that earned it and
recomputed -- the turn counts, the sensor floor and the amplitude screen by
:mod:`nova.scripts.mast_calibration_experiments`, the passive decays by
:mod:`nova.scripts.mast_vacuum_refinement`.

Which route established a value matters more than how close two routes came, so
the module keeps them apart.  For ten of the thirteen coils the archive states the
turn count itself: the store publishes both the current in one conductor and the
same current already multiplied, and dividing them back gives one exact integer on
every shot that carries both.  That is PUBLISHED, it is what gets authored, and it
rests on four orders more data than any fit here reads.  The vacuum response
measures the same quantity independently, by predicting each probe from the
registry outlines and the measured current, and it is recorded beside the published
integer as corroboration rather than in place of it.  One coil -- the solenoid --
has no published count and is FITTED.  Two have their excitation published as
ampere turns already, which fixes the product and not the count, so they stay
UNRESOLVED however clean the shots are.

Keeping the fit visible is what makes the record useful when the two disagree, and
they do: nine of the ten fitted amplitudes round to the published integer and one
does not.  Had the fit been discarded on agreement, that one would have been
invisible.

None of this depends on the reconstruction's own machine description.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from nova.imas.machine_evidence import (
    EvidenceRecord,
    FieldEvidence,
    SourceReference,
    Uncertainty,
)
from nova.imas.mast_seed_parameters import catalog_source

VACUUM_COHORT_STORE = "/work/projects/imas_gpu/mast/level1/shots"
"""Level-1 shot store the cohort was drawn from."""

VACUUM_COHORT_SHOTS = 400
"""Plasma-free, deliberately excited, adequately instrumented shots available."""

TRAINING_SHOTS = (
    14065,
    14071,
    14074,
    14076,
    14078,
    14081,
    14086,
    14089,
    14092,
    14098,
    14099,
    14103,
    14104,
    14106,
    14107,
    14109,
    14110,
    14113,
    14115,
    14127,
    15295,
    15296,
    19227,
    19231,
    19232,
    19235,
    19238,
    19246,
    19250,
    19251,
    19252,
    24938,
    24939,
    24940,
    24941,
    24947,
    24948,
    24959,
    24965,
    24966,
    24977,
    24978,
    24979,
    24980,
    25827,
    25828,
    25835,
    25836,
)
"""Shots the multipliers were fitted on, each isolating and holding one coil."""

HELD_OUT_SHOTS = (
    14126,
    14128,
    15201,
    15299,
    15943,
    16177,
    16417,
    19254,
    19996,
    22165,
    28490,
    30436,
)
"""Shots withheld from fitting, including every shot driving all six circuits."""

HELD_OUT_FAMILY = "P1+P2+P3+P4+P5+P6"
"""Excitation family withheld in full, so no coil combination was seen twice."""

HELD_OUT_VARIANCE_EXPLAINED = 0.9889
"""Share of held-out probe signal power the fitted multipliers reproduce.

On the generic cohort's twelve-shot held-out arm.  The calibration cohort's arm is
:data:`CALIBRATION_HELD_OUT_VARIANCE_EXPLAINED`, over eighty-two shots, and the two
are not comparable: a share of explained variance depends on how much signal the
withheld shots carry, so it can only be read against its own split.
"""

NOMINAL_VARIANCE_EXPLAINED = 0.1014
"""Share the same prediction reaches with one turn per coil, the unfitted case."""

HELD_OUT_RESIDUAL = 4.3818e-3
"""Root-mean-square held-out probe residual [T] with the fitted multipliers."""

NOMINAL_RESIDUAL = 3.9498e-2
"""Root-mean-square held-out probe residual [T] before fitting."""

CALIBRATION_HELD_OUT_SHOTS = 82
"""Shots withheld from every fit on the calibration cohort's declared split."""

CALIBRATION_HELD_OUT_VARIANCE_EXPLAINED = 0.9826
"""Share of that arm's probe signal the authored weights reproduce."""

CALIBRATION_HELD_OUT_RESIDUAL = 4.2637e-3
"""Root-mean-square held-out probe residual [T] with the authored weights.

Eleven times the measured sensor floor, which is the useful reading of it: the
description is nowhere near instrument-limited on vacuum data, so the misfit that
remains is calibration and modelling error rather than noise.

Promoting the published integers over the free fit costs almost nothing here --
letting every weight float reaches 4.2367e-3 on the same shots, six parts in a
thousand better -- so the more defensible values are not the worse ones.
"""

CALIBRATION_NOMINAL_RESIDUAL = 3.0778e-2
"""The same arm's residual with one turn per coil, the unfitted case [T]."""

FIRST_SHOT = 11695
"""First shot the registry's configuration covers."""

LAST_SHOT = 30473
"""Last shot the registry's configuration covers."""

INTEGER_INTERVAL = 0.5
"""Interval half-width, in turns, inside which a turn count names one integer."""


def _derived_ampere_turn_channel(row: FittedTurns) -> SourceReference:
    """Cite the store's own pair of current channels as stating a turn count."""

    return SourceReference(
        title="MAST level-1 shot store derived ampere-turn channels",
        url="https://mastapp.site/",
        locator=(
            f"amc {row.channel} against its derived ampere-turn channel, "
            f"ratio {row.published_turns} on at least {PUBLISHED_RATIO_SHOTS} shots"
        ),
        machine="mast",
        text_verified=True,
    )


def _quiescent_measurement(locator: str) -> SourceReference:
    """Cite a direct reading of the store, with no fit between it and the number."""

    return SourceReference(
        title="MAST level-1 vacuum shot store",
        url="https://mastapp.site/",
        locator=locator,
        machine="mast",
        text_verified=True,
    )


def _vacuum_fit(locator: str) -> SourceReference:
    """Cite the vacuum-response fit itself as the origin of a fitted value."""

    return SourceReference(
        title="MAST level-1 vacuum shot store",
        url="https://mastapp.site/",
        locator=locator,
        machine="mast",
        text_verified=True,
    )


@dataclass(frozen=True)
class FittedTurns:
    """One coil's signed turn count, and which route established it.

    Two routes reach a turn count and they are not equal in strength.  The archive
    publishes, for ten of the thirteen coils, a derived channel already multiplied
    by the count, and dividing it back gives one exact integer held across every
    shot that carries it -- a statement by the source, over four orders more data
    than any fit here reads.  The vacuum response reaches the same quantity by
    measuring the field a known current produced, which is independent but limited
    by how well the geometry models the field.

    ``published_turns`` records the first route when the archive states it, and
    then it is what gets authored.  ``multiplier`` always records the second, so
    the fit stays visible as corroboration rather than being replaced by the
    number it agrees with -- and where the two disagree, the disagreement is in
    the record instead of being hidden by whichever was written.
    """

    family: str
    channel: str
    multiplier: float
    half_width: float
    shot_count: int
    turns_per_multiplier: float = 1.0
    published_turns: int | None = None

    @property
    def identified(self) -> bool:
        """Return whether any shot pinned this coil."""

        return self.shot_count > 0

    @property
    def published(self) -> bool:
        """Return whether the archive states this coil's count itself."""

        return self.published_turns is not None

    @property
    def counted(self) -> bool:
        """Return whether the interval names one integer turn count."""

        return self.identified and self.half_width < INTEGER_INTERVAL

    @property
    def fitted_turns(self) -> float:
        """Return the turn count the vacuum response alone measured.

        The multiplier scales the channel the store publishes, which is not always
        the current in one turn.  The solenoid is driven as two parallel circuits,
        so one turn carries half the feed current and the coil's turn count is
        twice the multiplier a fit against that channel returns.  Keeping the
        conversion here rather than in the fit means the fitted number stays a
        statement about the channel and the authored number a statement about the
        conductor.
        """

        return float(self.multiplier * self.turns_per_multiplier)

    @property
    def turns(self) -> float:
        """Return the signed physical turn count to author."""

        if self.published_turns is not None:
            return float(self.published_turns)
        value = self.fitted_turns
        return float(round(value)) if self.counted else float(value)

    @property
    def corroboration(self) -> float:
        """Return how far the fit sits from the published count, as a fraction."""

        if self.published_turns is None or not self.identified:
            return float("nan")
        return float(self.fitted_turns / self.published_turns - 1.0)

    @property
    def agrees_with_published(self) -> bool:
        """Return whether the fit rounds to the count the archive publishes."""

        if self.published_turns is None or not self.identified:
            return False
        return round(self.fitted_turns) == self.published_turns

    @property
    def interval(self) -> Uncertainty:
        """Return the bound this coil's count is supported to.

        A published count carries the fit's own disagreement with it as the
        interval, never something narrower: the archive's integer is exact as a
        statement, and what is uncertain is whether this description reproduces the
        field that integer implies.  Reporting the exact integer with a zero-width
        interval would assert that agreement rather than measure it.
        """

        half = self.half_width * self.turns_per_multiplier
        if self.published_turns is not None:
            reach = max(abs(self.fitted_turns - self.published_turns) + half, half)
            return Uncertainty(
                lower=float(self.published_turns - reach),
                upper=float(self.published_turns + reach),
                unit="turn",
            )
        centre = self.fitted_turns
        if self.counted:
            return Uncertainty(
                lower=float(round(centre) - max(half, INTEGER_INTERVAL)),
                upper=float(round(centre) + max(half, INTEGER_INTERVAL)),
                unit="turn",
            )
        return Uncertainty(
            lower=float(centre - half), upper=float(centre + half), unit="turn"
        )

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical JSON representation."""

        return {
            "agrees_with_published": self.agrees_with_published,
            "channel": self.channel,
            "corroboration": float(self.corroboration),
            "counted": self.counted,
            "family": self.family,
            "fitted_turns": self.fitted_turns,
            "half_width": float(self.half_width),
            "identified": self.identified,
            "multiplier": float(self.multiplier),
            "published_turns": self.published_turns,
            "shot_count": self.shot_count,
            "turns": self.turns,
        }


VACUUM_FITTED_TURNS = (
    FittedTurns("sol", "sol_current", 344.656565, 11.684209, 4, 2.0),
    FittedTurns("p2_inner_lower", "p2il_feed_current", 11.995271, 0.001789, 2, 1.0, 12),
    FittedTurns("p2_inner_upper", "p2iu_feed_current", 11.955105, 0.007343, 2, 1.0, 12),
    FittedTurns("p2_outer_lower", "p2ol_feed_current", 7.989963, 0.003854, 2, 1.0, 8),
    FittedTurns("p2_outer_upper", "p2ou_feed_current", 7.970566, 0.010332, 2, 1.0, 8),
    FittedTurns("p3_lower", "p3l_feed_current", 8.067073, 0.037685, 5, 1.0, 8),
    FittedTurns("p3_upper", "p3u_feed_current", 8.039254, 0.000000, 1, 1.0, 8),
    FittedTurns("p4_lower", "p4l_feed_current", 22.766710, 0.001115, 2, 1.0, 23),
    FittedTurns("p4_upper", "p4u_feed_current", 22.913501, 0.077527, 3, 1.0, 23),
    FittedTurns("p5_lower", "p5l_feed_current", 23.820633, 0.452745, 8, 1.0, 23),
    FittedTurns("p5_upper", "p5u_feed_current", 23.019478, 0.019990, 2, 1.0, 23),
    FittedTurns("p6_lower", "p6l_current", float("nan"), float("nan"), 0),
    FittedTurns("p6_upper", "p6u_current", float("nan"), float("nan"), 0),
)
"""Signed turn count per coil, in registry order.

Ten of the thirteen carry the count the archive states, with the vacuum response
of the designed single-coil experiments beside it as an independent measurement.
Nine of the ten round to the published integer, the largest of those disagreeing
by :data:`WORST_PUBLISHED_OFFSET`.

The tenth is the useful one.  P5 lower's experiments measure 23.82 turns, which
rounds to twenty-four rather than to the published twenty-three, and its upper
partner -- same winding design, same published count -- measures 23.02.  The
disagreement is not scatter: the two shots from the earliest campaign read 0.990
and 0.991 of the published field while the six from the later one read 1.035 to
1.037, each group agreeing internally to a part in a thousand.  So the same coil's
measured response moved four and a half percent between campaigns, by far more
than either campaign's own repeatability, and no term in this description carries
whatever moved.  That is why the published integer promotes the count and the fit
corroborates it, rather than the other way round: a route whose answer depends on
which campaign it reads cannot name an integer, however tight each campaign looks.
"""

CAMPAIGN_RESPONSE_SHIFT = 0.045
"""Fractional change in one coil's measured response between two campaigns.

P5 lower, measured on the designed single-coil experiments with the published turn
count as reference: 0.990 on the earliest campaign and 1.036 on a later one, with
each campaign internally consistent to a part in a thousand.  Larger than the
sensor floor, larger than back-to-back repeatability, and larger than the spread
of any other coil -- so it is a real change in either the coil, its sensors or
their calibration, and it is the largest single unexplained term this route has
found.

Every identified multiplier is positive, so a positive current in the store's
channel produces the field a positive ``turns_with_sign`` predicts.  The poloidal
field components are unchanged by the source-to-target coordinate transform -- it
scales the flux and flips the safety factor and leaves B alone -- so this polarity
is the target convention's polarity and needed no reinterpretation.
"""

CALIBRATION_CLASS_SHOTS = {
    "pulsed_excitation": 31,
    "quiescent": 372,
    "sustained_coil_group": 287,
    "sustained_single_coil": 39,
    "sustained_symmetric_pair": 43,
    "toroidal_field_only": 577,
}
"""Plasma-free shots the store holds of each kind of designed experiment.

Recovered by classing the whole store on excitation shape rather than by trusting
a shot list.  The single-coil class is the archive's in-situ calibration campaign:
contiguous blocks holding one coil alone, each repeated at two current levels and
at two hold durations.
"""

PUBLISHED_TURN_RATIOS = {
    "p2_inner_lower": 12,
    "p2_inner_upper": 12,
    "p2_outer_lower": 8,
    "p2_outer_upper": 8,
    "p3_lower": 8,
    "p3_upper": 8,
    "p4_lower": 23,
    "p4_upper": 23,
    "p5_lower": 23,
    "p5_upper": 23,
}
"""The turn count the archive states for each coil that carries both channels."""

PUBLISHED_RATIO_SHOTS = 15135
"""Shots supporting the least-carried of those ratios, all agreeing on one integer."""

SINGLE_COIL_AMPLITUDE_SHOTS = 29
"""Designed single-coil experiments the published counts were measured against."""

PUBLISHED_AMPLITUDE = 0.9998
"""Measured field over field predicted from the published counts, median."""

PUBLISHED_AMPLITUDE_SPREAD = 0.0159
"""Spread of that amplitude across the single-coil experiments."""

WORST_PUBLISHED_OFFSET = 0.0357
"""Largest fractional disagreement any one coil shows with its published count."""

MIS_SCALED_SHOTS = (19322, 24947, 24948, 24959, 24977, 24978, 24979, 24980)
"""Shots whose magnetics were recorded at an amplitude no turn count explains.

Seven read close to half what their own currents imply and one reads nothing.
They are matched to correctly-scaled twins in the same campaign -- the same coil at
the same current to a tenth of a percent, every excitation channel agreeing, every
probe a factor of two apart, and the field's shape across the array unchanged -- so
the discrepancy is on the magnetics side of the acquisition and not in the machine.
Pooling them with the rest is what left the twenty-three turn coils bounded at five
turns instead of counted.
"""

MIS_SCALED_AMPLITUDE = (0.420, 0.566)
"""Amplitude the seven halved shots read, against the published counts."""

SENSOR_FLOOR = 3.8698e-4
"""Root-mean-square field a poloidal probe reads with nothing driven [T].

Measured on plasma-free shots that energised no poloidal coil, as the scatter
about each channel's own drift ramp, which is what survives the pre-excitation
offset subtraction every fit here performs.
"""

SENSOR_FLOOR_BY_FAMILY = {"ccbv": 4.4411e-4, "obr": 1.2630e-4, "obv": 4.3067e-4}
"""The same floor pooled per probe family [T]."""

SENSOR_FLOOR_SHOTS = 60
"""Shots the sensor floor was measured on."""

SENSOR_FLOOR_CHANNELS = 77
"""Probe channels that contributed a floor."""

REPEAT_REPRODUCIBILITY = (0.0043, 0.0519)
"""Fractional disagreement between back-to-back repetitions of one experiment.

Measured on nine groups of shots that re-fired the same coil at the same current
within one campaign, after dividing out the current each shot's own channel
recorded, so a supply that delivered one percent less does not count against the
sensors.  The median is about three percent.

This is the number that decides how a turn count may be promoted.  Three percent of
a twenty-three turn coil is seven tenths of a turn, so a fit reading one campaign
cannot name that integer however tightly its own shots agree -- which is why the
archive's published integers promote the counts and the fit corroborates them.

Restricted to repetitions inside one campaign and with the amplitude-refused shots
removed: a shot recorded at half amplitude disagrees with its own twin by that half
and would be reported as irreproducibility, and two shots a thousand apart at a
similar current are two different machine states rather than a repetition.  One
solenoid pair survives all of that and still disagrees by essentially one hundred
percent, so it is left out of the range and named here instead of being averaged
away: shots 15679 and 15692.
"""

SOLENOID_FAMILY = "sol"
"""Coil the archive publishes no turn count for, so a fit is the only route."""

SOLENOID_ALONE_SHOTS = 4
"""Shots that held the solenoid and no other coil, and identified it."""

SOLENOID_CAMPAIGN_VALUES = (337.4, 366.4)
"""Range the solenoid's ampere-turn weight takes across those shots.

Not scatter.  One shot from the earliest campaign reads 366.4 and three from a
later one read 337.4 to 343.1, each group tight, so the same campaign-dependent
response that moves P5 lower by four and a half percent moves the solenoid by
eight.  It is the whole of this route's width.
"""

SOLENOID_CODRIVEN_VALUES = (427.5, 476.7)
"""Weight the same fit returns from shots that drove another coil as well.

The solenoid's ampere turns outweigh a few-kiloampere neighbour by a factor of
forty, so it passes every leverage and correlation screen on such a shot and still
absorbs the neighbour's misfit.  The two populations do not overlap, which is why
only solenoid-only shots contribute: pooling them would move the centre seven
percent and triple the interval, and a measurement that degrades as data is added
is measuring the wrong thing.
"""

SOLENOID_STRATUM_INTERVAL = (348.5, 361.1)
"""Independent bound on the same weight, from a pooled plasma-free stratum.

Measured elsewhere on eighty-five plasma-free shots and about a hundred and eleven
thousand slices, with no solve in the loop, as a correction to a forward model
whose own solenoid ampere turns are 328 -- so the bound is on the same physical
quantity this record carries, ampere turns per ampere of the solenoid channel.
"""

SOLENOID_GEOMETRY_AGREEMENT = 0.0018
"""How far the two descriptions' solenoid field per ampere turn differ.

The comparison that makes the two routes' numbers comparable at all.  Each ampere
turn of the other description's 656-filament solenoid stack produces a field at
these probes that agrees with this description's polygon solenoid to this fraction
in amplitude, with a shape correlation of 0.99997 over the array.  So a difference
between the two stated weights is a difference about ampere turns and not about
where the copper is or how it is modelled -- which had to be established before
either number could be said to disagree with the other.
"""

ERROR_FIELD_SHOTS = 262
"""Plasma-free shots that deliberately drove a non-axisymmetric coil."""

ERROR_FIELD_PEAK = 12054.4
"""Strongest current any error-field coil channel carried in the archive [A].

Larger than the strongest the later campaigns reach, because the store names
these coils two ways and the earlier scheme is acquired on its own clock inside
the same group: a reader that knows only the later names sees the earlier
campaigns as carrying no such coil, and those campaigns hold the two strongest
shots in the archive.  Both schemes are read in
:mod:`nova.imas.mast_error_field_screen`.
"""

ERROR_FIELD_ISOLATED_SHOTS = 26
"""Shots that drove a non-axisymmetric coil with every poloidal coil quiet.

These make the coupling measurable directly rather than inferred: with nothing
else driving, a probe's regression on the error-field waveform IS that channel's
response per ampere.
"""

ERROR_FIELD_ARRAY_RESPONSE = 2.2e-9
"""Median probe response per ampere of non-axisymmetric current [T/A].

Measured across seventy-seven channels on the isolated shots.  At the strongest
excitation the archive contains it puts twenty-seven microtesla on a typical
channel, against a pooled sensor floor of :data:`SENSOR_FLOOR` -- fourteen times
below.  So the non-axisymmetric drive present in this archive is not merely
screened out of the calibration, it is invisible to these sensors.
"""

ERROR_FIELD_COUPLED_CHANNEL = "obr17"
"""The one channel whose coupling to the excitation is real and reproducible."""

ERROR_FIELD_COUPLED_RESPONSE = 1.1238e-7
"""That channel's response per ampere [T/A], median over fourteen shots.

Fifty times its own neighbours' and stable to twenty-eight percent across shots,
tracking the current rather than its derivative.  A field cannot do that: the
array is wound at seventy-five millimetre pitch and no coil produces a field that
varies fiftyfold over that distance, so this is a conductor shared with the
excitation.  It is screened like a field anyway -- the channel reports it either
way -- but it carries no field pattern and therefore no information about where
the array sits.
"""

ERROR_FIELD_COUPLED_THRESHOLD = 685.4
"""Excitation at which the coupled channel reaches its own noise floor [A].

Every other channel needs at least five kiloamperes and the median needs
eighty-two, so the screen removes one channel from the shots that drove the
error-field coils and removes no shot at all.
"""

ERROR_FIELD_SCREENED_SHOTS = (40, 50, 3)
"""Shots losing a channel to the screen, in train, held-out and noise order.

The three noise shots matter most: the quiescent class admitted them because
error-field drive is not poloidal-coil drive, so the sensor floor was measured
with a non-axisymmetric coil running on three of its sixty shots.  Only the one
coupled channel is affected, and re-measuring under the screen moves the pooled
floor by :data:`SCREENED_FLOOR_SHIFT` -- so the floor is not merely asserted to
be free of that excitation, it is measured to be.
"""

SCREENED_FLOOR_SHIFT = 1.75e-9
"""Change in the pooled sensor floor when the screen is applied [T].

Five parts in a million of the floor itself.  The coupled channel's own floor
moves by 0.67 microtesla, upward, because the three shots removed happened to be
quiet ones -- and that is a fortieth of the channel's own shot-to-shot spread of
25.9 microtesla.  Both statements are needed: the pooled number is what the ladder
targets, and the per-channel number is what says the shift is not hiding inside
one sensor.
"""

PROBE_VERDICT_COUNTS = {"field_shape": 64, "inseparable": 12, "not_tested": 1}
"""How the pre-registered discriminant classed all seventy-seven channels.

No channel reached a probe-side verdict, so no gain and no orientation was
promoted.  The twelve inseparable ones are the probes with no orthogonal partner,
whose rigid residual is unmeasured rather than small.
"""

NEAR_FIELD_PROBE_GAIN = 0.826
"""Measured field over described field for the coil a probe stands beside.

Pooled over the two outboard axial probes that carry most of the cross-source
excess, each about one winding-pack width from the nearer member of the P5 pair.
The description over-predicts that coil's axial field at that distance by
twenty-one percent while predicting every coil beyond about five pack widths to
within :data:`DISTANT_PROBE_GAIN` of unity.
"""

DISTANT_PROBE_GAIN = 1.066
"""Measured field over described field for coils beyond the near-field cut.

The same two probes, same shots, same fit.  That this sits near one while
:data:`NEAR_FIELD_PROBE_GAIN` does not is the whole discriminant: a wrong
effective area or a rotated probe scales every coil's field alike, and only the
arrangement of current inside a nearby pack can scale one coil and leave the
others.
"""

PROBE_GAIN_SPREAD_SIGNIFICANCE = 12.4
"""Standard errors separating the smallest per-coil gain from the largest.

Measured on the quieter of the two probes; the other reads 13.5.  The criterion
called three standard errors excitation-selective before any fit ran.
"""

RIGID_RESIDUAL_FLOOR_RATIO = (58.5, 101.0)
"""Residual after the best scale and rotation, in units of the channel's floor.

The two outboard axial probes.  Their co-located radial partners reach 12.8 and
2.1 on the same shots, so the point at which the description stops describing
these sensors is specific to the axial component beside the winding pack rather
than general to the mount.
"""

PROBE_TILT_INTERVAL = ((0.194, 0.078), (-0.179, 0.060))
"""Fitted poloidal rotation and its jackknife error, per probe [rad].

Both inside the mounting tolerance and nearly equal and opposite at positions
that mirror each other about the midplane, which is what a field error
antisymmetric in height looks like and not what two independently mis-mounted
probes look like.  Neither is promoted: the criterion requires the rotation to
remove the residual, and the residual it leaves is
:data:`RIGID_RESIDUAL_FLOOR_RATIO` times the floor.
"""

CAMPAIGN_PROBE_SPREAD = 0.0183
"""Median spread of a probe's gain across campaigns, all coils pooled."""

CAMPAIGN_COIL_SPREAD = 0.0978
"""Median spread of a coil's gain across campaigns, all far-field probes pooled.

Five times the probe figure.  Whatever moves between campaigns moves in the coil
dimension and not in the sensor dimension, which is what the question was: the
sensors are stable to under two percent across the whole archive.  The P5 lower
coil itself reads 0.9875 and 0.9818 on its two campaigns -- six parts in a
thousand, not the four and a half percent an earlier route reported before the
near-field probes were excluded and the cancelling-pair shots refused.
"""

LOOP_DISPOSITION_COUNTS = {
    "agreed": 25,
    "promoted": 3,
    "dual_valued": 11,
    "no_described_counterpart": 7,
    "no_channel": 5,
}
"""How every flux-loop channel's position was dispositioned.

All three promotions went to the position the description already carries, so the
adjudication confirms the described loop set where it can decide and moves
nothing.
"""

LOOP_DECIDED_MARGIN = (0.0715, 0.1249)
"""Held-out residual advantage of the promoted loop positions, smallest to largest.

Against a declared margin of five percent.  The loops that stayed dual-valued
came in between two and sixteen parts in ten thousand, so the boundary the
criterion drew is not a close call in either direction.
"""

VERTICAL_PAIR_ALONE_SHOTS = 207
"""Plasma-free shots that sustain a vertical-control coil with contrast to read it.

The vertical pair is not wired in series -- the two coils are driven independently
and on most shots to different currents -- so a shot that holds them at unequal
magnitudes constrains each separately.  A selection requiring the partner to be
quiet finds three such shots and concludes the coil cannot be seen; requiring only
that the two differ finds this many, at up to
:data:`VERTICAL_PAIR_STRONGEST` amperes.  What remains out of reach is therefore
not the experiment.
"""

VERTICAL_PAIR_DEDICATED_SHOTS = 36
"""Shots that sustain the vertical pair and no other coil at all."""

VERTICAL_PAIR_STRONGEST = 1.89e4
"""Strongest current a vertical-control coil carries on those shots [A]."""

VERTICAL_PAIR_AMPLITUDE = 0.379
"""Best-fit amplitude of the published unit weight on a pair-only shot."""

VERTICAL_PAIR_EXPLAINED = 0.0004
"""Share of that shot's probe signal the published unit weight reproduces.

A negative result, and the useful kind: the amplitude is far from one and the
prediction is essentially uncorrelated with the reading, so the shortfall is not a
mis-scaled weight but a field the description does not carry at all.  These shots
drive only the vertical pair, so whatever dominates their signal belongs to that
pair's neighbourhood -- its case above all -- and until that is modelled the pair's
published ampere-turn semantics cannot be confirmed or refuted from response.
"""

RADIAL_PROBE_FAMILY = "obr"
"""Probe family whose sensitive axis lies along the major radius."""

AXIAL_PROBE_FAMILIES = "ccbv,obv"
"""Probe families whose sensitive axes lie along the machine axis."""

AXIS_RESIDUAL_MARGIN = 1.35
"""How much worse the next-best sensitive-axis assignment predicts the cohort."""

PASSIVE_DECAY_INTERVAL = (0.023, 0.072)
"""Seconds of effective passive decay observed across the cohort's free decays."""

PASSIVE_DOMINANT_SHARE = 0.864
"""Smallest share of post-pulse signal power carried by one decay pattern."""

PASSIVE_DECAY_SHOTS = 12
"""Shots whose free decay the passive identification read."""

PASSIVE_DECISIVE_ATTRIBUTIONS = 2
"""Shots where one conductor group explained the dominant pattern outright."""

CASE_GROUP_COUNT = 10
"""Coil-case groups the plates resolve into, one per poloidal-field coil set."""

CASE_PLATE_COUNT = 24
"""Case parts the registry publishes as a single undifferentiated family.

Fewer parts than there are plates.  Each four-plate group closes into an enclosure
around its coil, and two plates meeting along a shared face are one polygon, so a
group of four contributes one part where its corners join and four where they do
not.  What the grouping has to cover is therefore parts and not plates.
"""


def fitted_turns(family: str) -> FittedTurns:
    """Return one coil's fitted turn count."""

    for row in VACUUM_FITTED_TURNS:
        if row.family == family:
            return row
    raise KeyError(f"no fitted turn count for active component {family!r}")


def authored_turns() -> dict[str, float]:
    """Return the signed turn count to write, per coil the cohort identified."""

    return {row.family: row.turns for row in VACUUM_FITTED_TURNS if row.identified}


def _turn_statement(row: FittedTurns) -> str:
    """Describe what established one coil's turn count, and how strongly."""

    if row.published:
        return (
            f"the store publishes this coil's current twice, once in one conductor "
            f"and once already multiplied, and their ratio is exactly "
            f"{row.published_turns} on every one of the {PUBLISHED_RATIO_SHOTS} or "
            f"more shots that carry both; the vacuum response of "
            f"{row.shot_count} shots that drove this coil alone independently "
            f"measures {row.fitted_turns:.3f} turns, which is "
            f"{abs(row.corroboration):.2%} from that integer and rounds to it"
        )
    if row.counted:
        return (
            f"the signed vacuum response of {row.shot_count} shots that drove this "
            f"coil alone gives {row.multiplier:.3f} turns per ampere of "
            f"{row.channel}, which names {row.turns:+.0f} turns"
        )
    return (
        f"the signed vacuum response of {row.shot_count} shots bounds this coil at "
        f"{row.fitted_turns:.1f} turns but the shots disagree by "
        f"{row.half_width * row.turns_per_multiplier:.1f} turns, so the count is "
        "carried as an interval rather than rounded to one integer"
    )


def _turn_assumptions(row: FittedTurns) -> tuple[str, ...]:
    """State what the turn count and its corroboration depend on."""

    assumptions = [
        "the winding pack carries a uniform current density over its measured "
        "outline, which fixes the field shape a probe standing clear of the pack "
        "reads and is why probes inside two pack widths are not read at all",
        "the excitation is held long enough for the coil case and the vessel to "
        "give back the current its own ramp induced, which is what separates a "
        "coil's own field from the transient around it",
        "the shot's magnetics were recorded at the amplitude its own currents "
        "imply, which is tested rather than assumed: shots reading between "
        f"{MIS_SCALED_AMPLITUDE[0]:.2f} and {MIS_SCALED_AMPLITUDE[1]:.2f} of the "
        "predicted field with the field's shape across the array unchanged are "
        "refused, because an amplitude and a turn count enter the prediction as "
        "one product and cannot be separated inside one shot",
    ]
    if row.turns_per_multiplier != 1.0:
        assumptions.append(
            "the coil is driven as two parallel circuits, so one turn carries half "
            "the measured feed current and the turn count is twice the fitted "
            "multiplier"
        )
    if row.family == SOLENOID_FAMILY:
        low, high = SOLENOID_STRATUM_INTERVAL
        assumptions.extend(
            (
                "only shots that held this coil and nothing else contribute: it "
                "outweighs a few-kiloampere neighbour by a factor of forty, so on a "
                "shot that drove one it passes every identifiability screen and still "
                f"absorbs that neighbour's misfit, returning "
                f"{SOLENOID_CODRIVEN_VALUES[0]:.0f} to "
                f"{SOLENOID_CODRIVEN_VALUES[1]:.0f} against "
                f"{SOLENOID_CAMPAIGN_VALUES[0]:.0f} to "
                f"{SOLENOID_CAMPAIGN_VALUES[1]:.0f} for the shots that drove it alone",
                "the interval's width is a campaign-dependent response and not "
                "measurement noise: the shots split into tight groups that disagree by "
                "eight percent, so no number of further shots inside one campaign "
                "narrows it",
                f"an independent measurement of the same weight on a pooled "
                f"plasma-free stratum bounds it at [{low:.1f}, {high:.1f}], which this "
                f"interval overlaps, so the two routes are consistent rather than in "
                f"conflict",
                f"the comparison is between the same physical quantity: the two "
                f"descriptions' solenoid field per ampere turn at these probes agrees "
                f"to {SOLENOID_GEOMETRY_AGREEMENT:.2%} in amplitude and 0.99997 in "
                f"shape across the array, so neither weight is absorbing a difference "
                f"in where the conductor is",
            )
        )
    if row.published:
        assumptions.append(
            "the published ratio is a statement by the source over four orders more "
            "shots than the fit reads, so it is what the count rests on and the fit "
            "is corroboration; the interval carries the fit's disagreement with it "
            "rather than the integer's own exactness"
        )
    return tuple(assumptions)


def fitted_turn_records(
    first_shot: int = FIRST_SHOT,
    last_shot: int = LAST_SHOT,
) -> list[EvidenceRecord]:
    """Record the turn count each coil's vacuum response established."""

    records = []
    for row in VACUUM_FITTED_TURNS:
        path = f"pf_active/coil({row.family})/element/turns_with_sign"
        if not row.identified:
            records.append(
                EvidenceRecord(
                    path=path,
                    evidence=FieldEvidence.UNRESOLVED,
                    first_shot=first_shot,
                    last_shot=last_shot,
                    statement=(
                        "this coil's turn count is out of reach for a structural "
                        "reason and not for want of an experiment: the store "
                        f"publishes its excitation in ampere turns already, so the "
                        f"{VERTICAL_PAIR_DEDICATED_SHOTS} shots that hold the "
                        "vertical pair and nothing else -- reaching "
                        f"{VERTICAL_PAIR_STRONGEST / 1e3:.1f} kiloamperes -- fix the "
                        "product of turns and current and can never divide it"
                    ),
                    assumptions=(
                        "the search for a dedicated experiment is closed rather than "
                        "abandoned: classing the whole store by excitation shape finds "
                        f"{VERTICAL_PAIR_ALONE_SHOTS} plasma-free shots that sustain "
                        "one of these coils with enough contrast against its partner "
                        "to read it separately, so no shot is missing",
                        "the two coils are driven independently rather than wired in "
                        "series, which is why the pair's split is identifiable at all "
                        "and why a selection requiring the partner to be quiet finds "
                        "almost nothing",
                        "the published product is what lets the coil be driven: its "
                        "channel drive carries one ampere turn per ampere, so the "
                        "forward-model column is fixed without the count this record "
                        "leaves open",
                        "that unit weight is not yet corroborated by response, and the "
                        "attempt is a negative result rather than a gap in the data: "
                        "on a shot holding only this pair the prediction built from it "
                        f"reproduces {VERTICAL_PAIR_EXPLAINED:.2%} of the probe signal "
                        f"at a best-fit amplitude of {VERTICAL_PAIR_AMPLITUDE:.2f}, so "
                        "what these shots excite is dominated by something this "
                        "description does not carry -- the pair's own case and the "
                        "structure around it are the candidates, and the ampere-turn "
                        "semantics cannot be tested until they are modelled",
                    ),
                    blocks_axisymmetric_forward_model=True,
                )
            )
            continue
        records.append(
            EvidenceRecord(
                path=path,
                evidence=(
                    FieldEvidence.PUBLISHED if row.published else FieldEvidence.FITTED
                ),
                first_shot=first_shot,
                last_shot=last_shot,
                statement=_turn_statement(row),
                assumptions=_turn_assumptions(row),
                source=(
                    _derived_ampere_turn_channel(row)
                    if row.published
                    else _vacuum_fit(
                        f"{row.shot_count} isolating shots, channel {row.channel}"
                    )
                ),
                uncertainty=row.interval,
            )
        )
    return records


def sensor_floor_records(
    first_shot: int = FIRST_SHOT,
    last_shot: int = LAST_SHOT,
) -> list[EvidenceRecord]:
    """Record what each probe reads when nothing is driving it.

    This is the one quantity that says when the description is finished.  Without
    it a residual can only be compared against another description's residual,
    which is how a machine parameter fitted to magnetics ends up looking like a
    better description; with it, the remaining misfit is either instrument or ours.
    """

    families = ", ".join(
        f"{name} {value * 1e6:.0f}" for name, value in SENSOR_FLOOR_BY_FAMILY.items()
    )
    return [
        EvidenceRecord(
            path="magnetics/b_field_pol_probe/field/data_error_upper",
            evidence=FieldEvidence.MEASURED,
            first_shot=first_shot,
            last_shot=last_shot,
            statement=(
                f"a poloidal probe reads {SENSOR_FLOOR * 1e6:.0f} microtesla "
                f"root-mean-square over the array with no coil driven, measured on "
                f"{SENSOR_FLOOR_SHOTS} plasma-free shots across "
                f"{SENSOR_FLOOR_CHANNELS} channels; per family in microtesla, "
                f"{families}"
            ),
            assumptions=(
                "the quantity is the scatter about each channel's own drift ramp, "
                "because an integrator's zero moves smoothly and the pre-excitation "
                "offset subtraction every fit here performs already removes that "
                "ramp -- charging a model for it would overstate the floor severalfold",
                "a shot holding only the toroidal field counts as undriven, because a "
                "poloidal probe is oriented to reject that field, so what it reads "
                "there is the instrument plus whatever the orientation fails to "
                "reject and both belong in the floor a description must reach",
            ),
            source=_quiescent_measurement(
                f"{SENSOR_FLOOR_SHOTS} shots driving no poloidal coil"
            ),
            uncertainty=Uncertainty(
                lower=min(SENSOR_FLOOR_BY_FAMILY.values()),
                upper=max(SENSOR_FLOOR_BY_FAMILY.values()),
                unit="T",
            ),
        ),
        EvidenceRecord(
            path="magnetics/b_field_pol_probe/field/validity",
            evidence=FieldEvidence.MEASURED,
            first_shot=first_shot,
            last_shot=last_shot,
            statement=(
                f"{len(MIS_SCALED_SHOTS)} plasma-free shots record a field no turn "
                f"count explains: seven read between {MIS_SCALED_AMPLITUDE[0]:.2f} and "
                f"{MIS_SCALED_AMPLITUDE[1]:.2f} of what their own currents imply and "
                "one reads nothing at all"
            ),
            assumptions=(
                "each has a twin in the same campaign driving the same coil to the "
                "same current within a tenth of a percent, with every excitation "
                "channel agreeing and every probe reading a factor of two apart, so "
                "the discrepancy is an amplitude the magnetics acquisition applied "
                "and not a field the coils produced",
                "the ratio taken probe by probe has the same scatter on a halved shot "
                "as on its twin, which is what distinguishes a uniform amplitude from "
                "a conductor in the wrong place",
                "a shot like this cannot measure a turn count at all, because the "
                "amplitude and the count enter the prediction as one product",
            ),
            source=_quiescent_measurement(
                "amplitude against the published counts, shots "
                + ", ".join(str(shot) for shot in MIS_SCALED_SHOTS)
            ),
        ),
    ]


def fitted_diagnostic_records(
    first_shot: int = FIRST_SHOT,
    last_shot: int = LAST_SHOT,
) -> list[EvidenceRecord]:
    """Record the diagnostic choices the vacuum response could and could not fix."""

    return [
        EvidenceRecord(
            path=f"magnetics/b_field_pol_probe({RADIAL_PROBE_FAMILY})/poloidal_angle",
            evidence=FieldEvidence.MEASURED,
            first_shot=first_shot,
            last_shot=last_shot,
            statement=(
                "the level-1 store puts this family's sensitive axis along the major "
                "radius, and refitting the whole cohort under each assignment "
                f"predicts the probes {AXIS_RESIDUAL_MARGIN:.2f} times better in "
                "residual that way round than the other"
            ),
            assumptions=(
                "nineteen of the store's seventy-eight probe positions carry two "
                "probes, one radial and one axial, so a placement match that "
                "resolves a tie by array order takes the axial partner's angle and "
                "reports a component the radial probe never saw",
                "an axis assigned the wrong way round cannot be rescued by any "
                "multiplier, which is what lets the cohort confirm the store rather "
                "than merely prefer one reading of it",
            ),
            source=catalog_source("level-1 magnetics named probe arrays"),
            uncertainty=Uncertainty(lower=0.0, upper=0.0, unit="rad"),
        ),
        EvidenceRecord(
            path="magnetics/b_field_pol_probe/position/phi",
            evidence=FieldEvidence.UNRESOLVED,
            first_shot=first_shot,
            last_shot=last_shot,
            statement=(
                "a poloidal-field vacuum shot cannot separate the two candidate "
                "toroidal positions, because the field it produces is the same at "
                "every toroidal angle"
            ),
            assumptions=(
                "the discriminating experiment is a deliberately non-axisymmetric "
                f"excitation and the store holds {ERROR_FIELD_SHOTS} plasma-free "
                f"shots of it, reaching {ERROR_FIELD_PEAK / 1e3:.1f} kiloamperes in "
                "an error-field coil channel, so the experiment is present rather "
                "than merely hoped for",
                f"{ERROR_FIELD_ISOLATED_SHOTS} of those shots drove the "
                "non-axisymmetric coil with every poloidal coil quiet, which makes "
                "the coupling a direct measurement rather than an inference: the "
                f"array's median response is {ERROR_FIELD_ARRAY_RESPONSE * 1e9:.1f} "
                "nanotesla per ampere, so even the strongest excitation the archive "
                "contains puts twenty-seven microtesla on a typical channel against "
                f"a {SENSOR_FLOOR * 1e6:.0f} microtesla floor",
                "so the blocker is not the missing error-field winding geometry but "
                "the excitation itself: separating the candidates would need about "
                "fifty kiloamperes through these coils, four times the largest shot "
                "in the archive, and authoring their geometry would not change that",
                "the single channel that does respond reproducibly, "
                f"{ERROR_FIELD_COUPLED_CHANNEL}, responds fifty times more than the "
                "channels seventy-five "
                "millimetres either side of it, which no coil's field can do -- it is "
                "a conductor shared with the excitation and carries no field pattern "
                "to phase a toroidal position against",
            ),
            candidates=("150 degrees", "330 degrees"),
        ),
        EvidenceRecord(
            path="magnetics/b_field_phi_probe/toroidal_angle",
            evidence=FieldEvidence.UNRESOLVED,
            first_shot=first_shot,
            last_shot=last_shot,
            statement=(
                "the level-1 store publishes no toroidal field probe channel, so "
                "there is no reading to regress an orientation against"
            ),
            assumptions=(
                "the only toroidal field quantities published are reconstruction "
                "outputs evaluated at the geometric and magnetic axes, which carry "
                "no per-sensor information",
            ),
        ),
        EvidenceRecord(
            path=f"magnetics/b_field_pol_probe({AXIAL_PROBE_FAMILIES})/poloidal_angle",
            evidence=FieldEvidence.MEASURED,
            first_shot=first_shot,
            last_shot=last_shot,
            statement=(
                "these families' sensitive axes lie along the machine axis, which "
                "the catalog pose records and the vacuum response confirms"
            ),
            source=catalog_source("level-1 magnetics named probe arrays"),
        ),
        EvidenceRecord(
            path="magnetics/flux_loop(saddle)/traversal_sign",
            evidence=FieldEvidence.UNRESOLVED,
            first_shot=first_shot,
            last_shot=last_shot,
            statement=(
                "the saddle signals the store publishes have had the poloidal-field "
                "pickup removed before publication, and that pickup is the whole of "
                "the quantity a coil pulse would fix the traversal sign against"
            ),
            assumptions=(
                "the remaining saddle products are differences between toroidally "
                "opposite loops, which cancel the axisymmetric response by "
                "construction",
                "the sign becomes measurable from an uncorrected saddle voltage, "
                "so the blocker is the published signal's processing rather than "
                "the geometry or the experiment",
            ),
            candidates=("recorded traversal", "reversed traversal"),
        ),
    ]


def fitted_passive_records(
    first_shot: int = FIRST_SHOT,
    last_shot: int = LAST_SHOT,
) -> list[EvidenceRecord]:
    """Record what the free decays fixed about the passive circuit."""

    lower, upper = PASSIVE_DECAY_INTERVAL
    return [
        EvidenceRecord(
            path="pf_passive/loop/time_constant",
            evidence=FieldEvidence.FITTED,
            first_shot=first_shot,
            last_shot=last_shot,
            statement=(
                "one spatial pattern carries at least "
                f"{PASSIVE_DOMINANT_SHARE:.0%} of the post-pulse probe signal in "
                f"every transient shot and decays on {lower * 1e3:.0f} to "
                f"{upper * 1e3:.0f} milliseconds"
            ),
            assumptions=(
                "the decay window opens after the supply transient has passed, so "
                "what is left decays rather than rings",
                "the reported quantity is the effective decay of the whole passive "
                "circuit as the probes see it, not the decay of any one conductor",
            ),
            source=_vacuum_fit("post-pulse probe decay on the transient shots"),
            uncertainty=Uncertainty(lower=lower, upper=upper, unit="s"),
        ),
        EvidenceRecord(
            path="pf_passive/loop/resistance",
            evidence=FieldEvidence.UNRESOLVED,
            first_shot=first_shot,
            last_shot=last_shot,
            statement=(
                "the probes separate one decay pattern per shot, so the cohort "
                "supports one effective parameter and not a resistance for each of "
                "the sixteen passive families"
            ),
            assumptions=(
                "one conductor group explains the dominant pattern outright on "
                f"{PASSIVE_DECISIVE_ATTRIBUTIONS} of {PASSIVE_DECAY_SHOTS} shots, "
                "and which group it is changes with which coil was pulsed, so the "
                "pattern identifies a neighbourhood rather than a conductor",
                "the per-family currents the store publishes are the "
                "reconstruction's own wall-model output rather than an instrument "
                "reading, so they cannot ground a fit",
            ),
        ),
        EvidenceRecord(
            path="pf_passive/loop(coil_cases)/element/geometry/outline",
            evidence=FieldEvidence.GENERATED,
            first_shot=first_shot,
            last_shot=last_shot,
            statement=(
                f"the {CASE_PLATE_COUNT} case plates the registry publishes as one "
                f"family resolve into {CASE_GROUP_COUNT} groups, one per "
                "poloidal-field coil set, by enclosure"
            ),
            assumptions=(
                "a coil case surrounds its own coil and nothing else, so each plate "
                "belongs to the coil set it is nearest to and no plate is left "
                "further than half a metre from one",
                "the grouping is by coil set rather than by winding pack, because "
                "one case encloses both packs of a set and the store publishes one "
                "case-current channel per set",
            ),
            uncertainty=Uncertainty(
                lower=float(CASE_GROUP_COUNT),
                upper=float(CASE_GROUP_COUNT),
                unit="group",
            ),
        ),
    ]


def sensor_adjudication_records(
    first_shot: int = FIRST_SHOT,
    last_shot: int = LAST_SHOT,
) -> list[EvidenceRecord]:
    """Record what the sensor adjudication decided, and what it decided against.

    Two of the three outcomes here are negative, and they are the load-bearing
    ones.  No probe earned a gain or an orientation correction, and the reason is
    measured rather than assumed: the scale the data wants depends on which coil is
    driving, which a rigid probe error cannot produce.  And the flux-loop positions
    the fit could decide all came down on the side the description already carried,
    so an adjudication that could have moved thirteen sensor poses moves none.
    """

    return [
        EvidenceRecord(
            path="magnetics/b_field_pol_probe/field/scale",
            evidence=FieldEvidence.FITTED,
            first_shot=first_shot,
            last_shot=last_shot,
            statement=(
                "no probe carries a calibration-scale correction: fitted over the "
                "calibration cohort, the scale each channel wants is one to within "
                "its own scatter once the coil that dominates the reading is far "
                "enough away, and the excess the outboard channels carry is not a "
                "scale at all"
            ),
            assumptions=(
                "each outboard axial probe shares its position with an outboard "
                "radial one, so a scale and a poloidal rotation are both fittable "
                "against measurements at that point rather than against the model "
                "whose winding description is itself in question",
                "the scale is estimated per shot and pooled across shots, because a "
                "waveform's samples carry about one pulse of information between "
                "them and a standard error taken from the sample count would make "
                "every channel look excitation-selective",
                "a shot driving a coil pair in opposition is refused: the two fields "
                "cancel, each coil's own power becomes a large multiple of the "
                "prediction, and the scale returned is a ratio of two nearly "
                "cancelling numbers rather than an amplitude",
            ),
            uncertainty=Uncertainty(lower=0.95, upper=1.05, unit="1"),
        ),
        EvidenceRecord(
            path="pf_active/coil(p4,p5)/element/current_distribution",
            evidence=FieldEvidence.UNRESOLVED,
            first_shot=first_shot,
            last_shot=last_shot,
            statement=(
                "uniform current density over these coils' pack outlines "
                "over-predicts the axial field one pack width away by "
                f"{(1.0 / NEAR_FIELD_PROBE_GAIN - 1.0) * 100:.0f} percent, while the "
                "same fit predicts every coil beyond about five pack widths to "
                f"within {abs(DISTANT_PROBE_GAIN - 1.0) * 100:.0f} percent, so what "
                "is unresolved is how the turns sit inside the pack and not how much "
                "current they carry"
            ),
            assumptions=(
                "the two channels carrying most of the cross-source excess are "
                "placed identically by both descriptions, so no pose difference can "
                "explain it and the cause is one both descriptions inherit",
                "their per-coil scale spreads by "
                f"{PROBE_GAIN_SPREAD_SIGNIFICANCE:.1f} standard errors across the "
                "coils that drove them, against three declared beforehand as the "
                "boundary, and the largest departure is on the nearest coil in each "
                "case",
                "the best joint scale and rotation leaves "
                f"{RIGID_RESIDUAL_FLOOR_RATIO[0]:.0f} and "
                f"{RIGID_RESIDUAL_FLOOR_RATIO[1]:.0f} times each channel's own "
                "quiescent scatter, so no rigid probe transform explains them "
                "however stable its coefficients",
                "the two fitted rotations are nearly equal and opposite at positions "
                "mirroring each other about the midplane, which is a field error "
                "antisymmetric in height rather than two mis-mounted probes",
            ),
            candidates=("filament lattice", "uniform pack density"),
        ),
        EvidenceRecord(
            path="magnetics/flux_loop/position",
            evidence=FieldEvidence.MEASURED,
            first_shot=first_shot,
            last_shot=last_shot,
            statement=(
                f"{LOOP_DISPOSITION_COUNTS['agreed']} loops are placed identically by "
                "both sources and every disagreement the measured flux could decide "
                "was decided in favour of the position already described"
            ),
            assumptions=(
                "each channel is joined to its loop through the reconstruction's own "
                "family blocks rather than by proximity, because several described "
                "loops share one position and a nearest-position join hands them all "
                "the same loop",
                "each candidate position is scored with one free scale per channel, "
                "so a position is judged on how its predicted flux varies with the "
                "coils and with time rather than on an amplitude the loop's own gain "
                "could supply",
                "a candidate promotes only when it wins on shots the fit never saw, "
                "by at least a twentieth in residual, and with the same winner in and "
                "out of sample: the promoted ones win by "
                f"{LOOP_DECIDED_MARGIN[0] * 100:.1f} to "
                f"{LOOP_DECIDED_MARGIN[1] * 100:.1f} percent",
            ),
            source=catalog_source("level-2 flux-loop positions as catalogued"),
            uncertainty=Uncertainty(lower=-1.0e-3, upper=1.0e-3, unit="m"),
        ),
        EvidenceRecord(
            path="magnetics/flux_loop(displaced)/position",
            evidence=FieldEvidence.UNRESOLVED,
            first_shot=first_shot,
            last_shot=last_shot,
            statement=(
                f"{LOOP_DISPOSITION_COUNTS['dual_valued']} loops are placed "
                "differently by the two sources by three to nine millimetres and the "
                "measured flux does not choose between them"
            ),
            assumptions=(
                "the two positions differ in predicted flux by up to a hundred and "
                "fifty times the channel's own scatter, so the question is "
                "answerable in principle; what stops it is that both candidates "
                "leave a residual two orders above that scatter, and a few "
                "millimetres of position moves that residual by parts in a thousand",
                "so this stays open until the term dominating the loop residual is "
                "found, which is a statement about the field description and not "
                "about the loop survey",
            ),
            candidates=("catalogued position", "reconstruction position"),
        ),
        EvidenceRecord(
            path="magnetics/flux_loop(undescribed)/position",
            evidence=FieldEvidence.UNRESOLVED,
            first_shot=first_shot,
            last_shot=last_shot,
            statement=(
                "the reconstruction reads "
                f"{LOOP_DISPOSITION_COUNTS['no_described_counterpart']} loops the "
                "description carries no sensor for, and the description "
                f"carries {LOOP_DISPOSITION_COUNTS['no_channel']} loops no published "
                "channel reaches"
            ),
            assumptions=(
                "the gap is one-sided by family: the four lower P4 loops and the two "
                "upper P6 loops are missing while their opposite-side counterparts "
                "are carried, and each missing loop sits about a quarter of a metre "
                "from the nearest described one, so this is an absent fixture set "
                "rather than a displaced one",
                "the catalogued source the description is built from does not list "
                "them, so closing the gap needs a document and not a fit",
            ),
            blocks_axisymmetric_forward_model=False,
        ),
        EvidenceRecord(
            path="magnetics/b_field_pol_probe/non_axisymmetric_screen",
            evidence=FieldEvidence.MEASURED,
            first_shot=first_shot,
            last_shot=last_shot,
            statement=(
                "the deliberately non-axisymmetric excitation the archive contains "
                "cannot bias a calibration fit at seventy-six of seventy-seven "
                f"channels, and re-measuring the sensor floor under the screen moves "
                f"it by {SCREENED_FLOOR_SHIFT * 1e12:.1f} picotesla"
            ),
            assumptions=(
                f"the coupling is measured on {ERROR_FIELD_ISOLATED_SHOTS} shots that "
                "drove the non-axisymmetric coil with every poloidal coil quiet, and "
                "the threshold each channel is screened at is where that coupling "
                "reaches the channel's own quiescent scatter",
                f"one channel, {ERROR_FIELD_COUPLED_CHANNEL}, is refused above "
                f"{ERROR_FIELD_COUPLED_THRESHOLD:.0f} amperes; every other channel's "
                "threshold exceeds five kiloamperes and the median exceeds eighty, so "
                "the screen removes a channel and never a shot",
                "the channel that has a coupling responds fifty times more than the "
                "channels either side of it, which no coil's field does over "
                "seventy-five millimetres, so it is read as a shared conductor and "
                "not as evidence about the array's toroidal position",
            ),
            source=catalog_source("level-1 error-field coil current channels"),
            uncertainty=Uncertainty(lower=0.0, upper=SCREENED_FLOOR_SHIFT, unit="T"),
        ),
    ]


def fitted_evidence(
    first_shot: int = FIRST_SHOT,
    last_shot: int = LAST_SHOT,
) -> list[EvidenceRecord]:
    """Return every record the vacuum refinement contributes."""

    return [
        *fitted_turn_records(first_shot, last_shot),
        *fitted_diagnostic_records(first_shot, last_shot),
        *fitted_passive_records(first_shot, last_shot),
        *sensor_floor_records(first_shot, last_shot),
        *sensor_adjudication_records(first_shot, last_shot),
    ]


SUPERSEDED_SEED_PATHS = frozenset(
    {
        "pf_active/coil/element/turns_with_sign",
        "pf_active/circuit/connections",
        "magnetics/b_field_pol_probe/poloidal_angle",
        "magnetics/b_field_pol_probe/position/phi",
        "magnetics/b_field_phi_probe/toroidal_angle",
        "magnetics/flux_loop(saddle)/traversal_sign",
        "magnetics/flux_loop/position",
    }
)
"""Seed records the refinement replaces with per-component or evidenced ones.

The seed carries one record for every coil's turn count together, which is the
right statement while nothing is known and the wrong one once eleven of thirteen
coils have been measured separately.  Two diagnostic paths are replaced because
the refinement establishes WHY each is unresolved, which the seed could only
assert.

The blanket circuit-connection record goes because it says the node matrix cannot
be filled at all, which stops being true once the supply columns are dropped: the
coil-to-coil junctions the sources fix are authorable without any supply, and
per-circuit records take over from the one that denied them.

The probe sensitive axis is replaced for a different reason.  The seed records one
angle for all seventy-eight probes, which stops being true once the radial family
carries its own: the store gives nineteen outboard probes an axis along the major
radius and the rest an axial one.  A blanket record would now assert a uniformity
the machine does not have, so it is narrowed to one record per axis.

The loop positions are replaced because the seed can only say the catalog gives
them, and the refinement can say the loops' own measured flux was asked which of
two candidate positions it prefers and answered.  That is the same claim with a
test behind it, and the record has to carry the test or the next reader repeats it.
"""


def refined_evidence(
    seed: Mapping[str, Any] | tuple[EvidenceRecord, ...],
    first_shot: int = FIRST_SHOT,
    last_shot: int = LAST_SHOT,
) -> tuple[EvidenceRecord, ...]:
    """Fold the refinement's records into a seed ledger's records."""

    records = tuple(seed) if not isinstance(seed, Mapping) else tuple(seed["records"])
    kept = [row for row in records if row.path not in SUPERSEDED_SEED_PATHS]
    return tuple(kept) + tuple(fitted_evidence(first_shot, last_shot))
