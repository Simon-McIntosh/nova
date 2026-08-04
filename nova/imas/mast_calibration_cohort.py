"""Separate the archive's designed calibration experiments from ordinary shots.

The generic no-plasma selection in :mod:`nova.imas.mast_vacuum_cohort` asks only
whether a shot is usable: no plasma, some coil driven, enough probes recorded.
That admits four hundred shots of which most are operational preparation, and it
treats a shot that pulsed six circuits together for six milliseconds as the equal
of one that held a single coil at twenty kiloamperes for two thirds of a second.
Only the second kind measures anything about one coil.

The archive contains the second kind deliberately.  Reading the census by
excitation shape recovers contiguous blocks of shots that hold one coil alone,
repeated at two current levels and at two hold durations, one block per coil --
the in-situ calibration campaign the magnetic-diagnostics literature describes.
It also recovers a large class holding the toroidal field alone, a class holding
the vertical-control pair alone, and nine hundred shots where nothing was driven
at all, which is where sensor noise can be measured rather than assumed.

Classification is by what an experiment can IDENTIFY, and the two facts that
decide it are independent.  Whether a coil was held long enough for the induced
currents its own ramp created to decay, because otherwise the reading is the coil
plus its case plus the vessel.  And whether its series partner was quiet, because
a pair carrying one waveform fixes only the pair's total.  A shot satisfying both
for one coil measures that coil; a shot satisfying both for a symmetric pair
measures the pair's sum and not the split.

Turn counts are not all the same kind of claim either.  For ten of the thirteen
coils the store publishes, beside the current in one conductor, a derived channel
already multiplied by the turn count, and their ratio is one integer held to a
part in ten million across fifteen thousand shots.  That is the archive stating a
turn count, and it is stronger evidence than any fit against a few dozen shots.
Recognising it needs a tolerance on the number and not on its decimal spelling:
de-duplicating the ratios at six decimal places splits a family apart on float
noise in the store's own division and silently discards the statement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Iterable, Mapping, Sequence

from nova.imas.mast_vacuum_cohort import (
    COIL_DRIVES,
    EXCITATION_CURRENT,
    PLASMA_FREE_CURRENT,
    SUSTAINED_HOLD,
    CohortError,
    ShotSurvey,
)

TOROIDAL_EXCITATION_CURRENT = 1.0e4
"""Amperes of toroidal field current above which the field coil was driven.

The toroidal channel carries a standing reading of a few hundred amperes on a
shot that never energised it, the same pickup floor the poloidal channels sit at,
and a driven toroidal field runs to a hundred kiloamperes.  The line sits an
order clear of the floor and an order below the excitation.
"""

RATIO_INTEGER_TOLERANCE = 1.0e-3
"""How far a published ampere-turn ratio may sit from an integer and still be one.

A tolerance on the VALUE, not on its decimal spelling.  The store computes each
derived ampere-turn channel by multiplying a conductor current, so the ratio
recovered by dividing them back carries float noise of order one part in ten
million -- twenty of fifteen thousand shots land at 23.000001 rather than 23.
Grouping the ratios by rounded decimals therefore reports a family as varying
when it does not, and discards the archive's own statement of six of the ten turn
counts.  This tolerance is four orders looser than that noise and three orders
tighter than the half-turn that would confuse neighbouring integers, so no
spelling accident can split a family and no genuine disagreement can pass.
"""

SYMMETRIC_PAIR_CONTRAST = 0.9
"""Smallest ratio between a pair's two currents that counts as driving it as one.

Above this the two members carry the same excitation to within a tenth, so their
sum is what the probes constrain and the split is not identifiable at all.  Below
it the shot has contrast to exploit and belongs to the asymmetric class.
"""


class CalibrationError(CohortError):
    """Raised when a shot cannot be classed as a calibration experiment."""


class ExperimentClass(StrEnum):
    """What kind of experiment a shot is, by what its excitation can identify."""

    SUSTAINED_SINGLE_COIL = "sustained_single_coil"
    SUSTAINED_SYMMETRIC_PAIR = "sustained_symmetric_pair"
    SUSTAINED_COIL_GROUP = "sustained_coil_group"
    PULSED_EXCITATION = "pulsed_excitation"
    TOROIDAL_FIELD_ONLY = "toroidal_field_only"
    QUIESCENT = "quiescent"
    PLASMA = "plasma"
    UNREADABLE = "unreadable"


IDENTIFYING_CLASSES = (
    ExperimentClass.SUSTAINED_SINGLE_COIL,
    ExperimentClass.SUSTAINED_SYMMETRIC_PAIR,
    ExperimentClass.SUSTAINED_COIL_GROUP,
)
"""Classes whose excitation is held long enough to measure a coil's own field."""

NOISE_CLASSES = (ExperimentClass.QUIESCENT, ExperimentClass.TOROIDAL_FIELD_ONLY)
"""Classes carrying no poloidal excitation, where sensor noise is measurable.

The toroidal-field class is included because a poloidal probe is oriented to be
blind to the toroidal field, so a shot holding that field alone measures what the
probe reads when nothing it is sensitive to is happening -- which is the noise
plus whatever the orientation fails to reject, and that residue belongs in an
envelope the description is asked to reach.
"""


def _series_partner() -> dict[str, str]:
    """Map each coil to the coil its documented circuit wires it in series with."""

    families = {drive.family for drive in COIL_DRIVES}
    partners: dict[str, str] = {}
    for family in families:
        for suffix, other in (("_upper", "_lower"), ("_lower", "_upper")):
            if family.endswith(suffix):
                candidate = family[: -len(suffix)] + other
                if candidate in families:
                    partners[family] = candidate
    return partners


def integer_ampere_turn_ratios(
    surveys: Iterable[ShotSurvey],
    *,
    tolerance: float = RATIO_INTEGER_TOLERANCE,
) -> dict[str, int]:
    """Return the integer ampere-turn ratio the store publishes, per coil.

    A family is reported only when every shot that carries the ratio agrees with
    one integer to within ``tolerance``.  A family whose ratios straddle two
    integers is left out rather than averaged, because that would be a statement
    the archive does not make.
    """

    observed: dict[str, list[float]] = {}
    for survey in surveys:
        for family, ratio in survey.turn_multipliers.items():
            if math.isfinite(ratio):
                observed.setdefault(family, []).append(float(ratio))
    ratios: dict[str, int] = {}
    for family, values in observed.items():
        integer = round(sum(values) / len(values))
        if integer <= 0:
            continue
        if all(abs(value - integer) <= tolerance for value in values):
            ratios[family] = int(integer)
    return dict(sorted(ratios.items()))


def ampere_turn_ratio_support(
    surveys: Iterable[ShotSurvey],
    *,
    tolerance: float = RATIO_INTEGER_TOLERANCE,
) -> dict[str, int]:
    """Return how many shots carry each recognised integer ampere-turn ratio."""

    rows = tuple(surveys)
    recognised = integer_ampere_turn_ratios(rows, tolerance=tolerance)
    counts = {family: 0 for family in recognised}
    for survey in rows:
        for family in survey.turn_multipliers:
            if family in counts:
                counts[family] += 1
    return dict(sorted(counts.items()))


@dataclass(frozen=True, order=True)
class CalibrationExperiment:
    """One shot, the class of experiment it is, and what it can identify."""

    shot: int
    experiment: ExperimentClass
    sustained: tuple[str, ...]
    excited: tuple[str, ...]
    identifies: tuple[str, ...]
    identifies_sum: tuple[str, ...]
    peak_current: float
    hold_time: float
    toroidal_peak: float
    probe_count: int

    @property
    def measures_turns(self) -> bool:
        """Return whether this shot can measure a coil's own turn count."""

        return bool(self.identifies)

    @property
    def measures_noise(self) -> bool:
        """Return whether this shot carries no poloidal excitation to measure."""

        return self.experiment in NOISE_CLASSES

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical JSON representation."""

        return {
            "excited": list(self.excited),
            "experiment": str(self.experiment),
            "hold_time": float(self.hold_time),
            "identifies": list(self.identifies),
            "identifies_sum": list(self.identifies_sum),
            "peak_current": float(self.peak_current),
            "probe_count": self.probe_count,
            "shot": self.shot,
            "sustained": list(self.sustained),
            "toroidal_peak": float(self.toroidal_peak),
        }


def classify_experiment(
    survey: ShotSurvey,
    *,
    hold: float = SUSTAINED_HOLD,
    minimum_probes: int = 40,
) -> CalibrationExperiment:
    """Class one surveyed shot by what its excitation can identify.

    The order of the tests is the order of the disqualifications.  A shot with a
    plasma is not a vacuum experiment whatever it drove.  A shot missing a store
    group or most of its probe array cannot be read.  Only then does the shape of
    the excitation decide, and the decisive distinction inside the driven classes
    is the hold: a pulsed shot is kept as its own class rather than discarded,
    because it still constrains the passive circuit it excited.
    """

    partner = _series_partner()
    excited = tuple(survey.excited_families)
    sustained = tuple(survey.sustained_coils(hold))
    peak = max(
        (survey.coil_peaks.get(family, 0.0) for family in excited),
        default=0.0,
    )
    held = max(
        (survey.coil_hold_times.get(family, 0.0) for family in sustained), default=0.0
    )
    common = dict(
        shot=survey.shot,
        sustained=sustained,
        excited=excited,
        peak_current=peak,
        hold_time=held,
        toroidal_peak=survey.toroidal_current_peak,
        probe_count=len(survey.field_channels),
    )

    if survey.plasma_current_peak >= PLASMA_FREE_CURRENT:
        return CalibrationExperiment(
            experiment=ExperimentClass.PLASMA,
            identifies=(),
            identifies_sum=(),
            **common,
        )
    if survey.absent_groups or len(survey.field_channels) < minimum_probes:
        return CalibrationExperiment(
            experiment=ExperimentClass.UNREADABLE,
            identifies=(),
            identifies_sum=(),
            **common,
        )
    if not excited:
        experiment = (
            ExperimentClass.TOROIDAL_FIELD_ONLY
            if survey.toroidal_current_peak >= TOROIDAL_EXCITATION_CURRENT
            else ExperimentClass.QUIESCENT
        )
        return CalibrationExperiment(
            experiment=experiment, identifies=(), identifies_sum=(), **common
        )
    if not sustained:
        return CalibrationExperiment(
            experiment=ExperimentClass.PULSED_EXCITATION,
            identifies=(),
            identifies_sum=(),
            **common,
        )

    alone: list[str] = []
    paired: list[str] = []
    for family in sustained:
        other = partner.get(family)
        own = survey.coil_peaks.get(family, 0.0)
        if other is None:
            alone.append(family)
            continue
        mate = survey.coil_peaks.get(other, 0.0)
        if mate < EXCITATION_CURRENT:
            alone.append(family)
        elif own > 0.0 and min(own, mate) / max(own, mate) >= SYMMETRIC_PAIR_CONTRAST:
            paired.append(family)
        else:
            alone.append(family)

    if len(sustained) == 1:
        experiment = ExperimentClass.SUSTAINED_SINGLE_COIL
    elif len(sustained) == 2 and len(paired) == 2:
        experiment = ExperimentClass.SUSTAINED_SYMMETRIC_PAIR
    else:
        experiment = ExperimentClass.SUSTAINED_COIL_GROUP
    return CalibrationExperiment(
        experiment=experiment,
        identifies=tuple(sorted(alone)),
        identifies_sum=tuple(sorted(paired)),
        **common,
    )


def calibration_experiments(
    surveys: Iterable[ShotSurvey],
    *,
    hold: float = SUSTAINED_HOLD,
    minimum_probes: int = 40,
) -> tuple[CalibrationExperiment, ...]:
    """Class every surveyed shot, in ascending shot order."""

    return tuple(
        sorted(
            (
                classify_experiment(survey, hold=hold, minimum_probes=minimum_probes)
                for survey in surveys
            ),
            key=lambda row: row.shot,
        )
    )


def class_counts(
    experiments: Iterable[CalibrationExperiment],
) -> dict[str, int]:
    """Return how many shots fall in each experiment class."""

    counts: dict[str, int] = {}
    for row in experiments:
        counts[str(row.experiment)] = counts.get(str(row.experiment), 0) + 1
    return dict(sorted(counts.items()))


@dataclass(frozen=True)
class Identifiability:
    """Which shots can measure one coil, and on what terms."""

    family: str
    alone: tuple[int, ...]
    in_sum: tuple[int, ...]
    strongest: float

    @property
    def identifiable(self) -> bool:
        """Return whether any shot measures this coil on its own."""

        return bool(self.alone)

    @property
    def sum_only(self) -> bool:
        """Return whether the coil is reachable only through a pair's total."""

        return not self.alone and bool(self.in_sum)

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical JSON representation."""

        return {
            "alone": list(self.alone),
            "alone_count": len(self.alone),
            "family": self.family,
            "identifiable": self.identifiable,
            "in_sum": list(self.in_sum),
            "in_sum_count": len(self.in_sum),
            "strongest": float(self.strongest),
            "sum_only": self.sum_only,
        }


def identifiability_map(
    experiments: Iterable[CalibrationExperiment],
    *,
    families: Sequence[str] | None = None,
) -> tuple[Identifiability, ...]:
    """Report, per coil, which experiments can measure it and how strongly.

    This is what orders the work: a coil with a dozen shots that hold it alone is
    measurable now, a coil reachable only inside a symmetric pair has its sum
    measurable and its split structurally out of reach, and a coil in neither list
    cannot be measured from this archive however the fit is arranged.
    """

    order = (
        tuple(families)
        if families is not None
        else tuple(drive.family for drive in COIL_DRIVES)
    )
    rows = tuple(experiments)
    result = []
    for family in order:
        alone = tuple(sorted(row.shot for row in rows if family in row.identifies))
        in_sum = tuple(sorted(row.shot for row in rows if family in row.identifies_sum))
        reachable = {*alone, *in_sum}
        strongest = max(
            (row.peak_current for row in rows if row.shot in reachable),
            default=0.0,
        )
        result.append(
            Identifiability(
                family=family, alone=alone, in_sum=in_sum, strongest=strongest
            )
        )
    return tuple(result)


@dataclass(frozen=True)
class CalibrationCohort:
    """The classed experiments, the split declared over them, and the refusals."""

    by_class: Mapping[str, tuple[int, ...]]
    training: tuple[int, ...]
    held_out: tuple[int, ...]
    noise_shots: tuple[int, ...]
    identifiability: tuple[Identifiability, ...]
    published_ratios: Mapping[str, int]

    def validate(self) -> None:
        """Reject a cohort whose split leaks a shot or holds nothing back."""

        overlap = set(self.training) & set(self.held_out)
        if overlap:
            raise CalibrationError(f"shots {sorted(overlap)} are in both cohort arms")
        if not self.held_out:
            raise CalibrationError(
                "a calibration cohort must hold out at least one shot"
            )
        if not self.training:
            raise CalibrationError(
                "a calibration cohort must train on at least one shot"
            )
        if set(self.noise_shots) & (set(self.training) | set(self.held_out)):
            raise CalibrationError(
                "a noise shot carries no excitation and cannot also fit a turn count"
            )

    @property
    def shots(self) -> tuple[int, ...]:
        """Return every shot the fitting arms hold, in ascending order."""

        return tuple(sorted((*self.training, *self.held_out)))

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical JSON representation."""

        return {
            "by_class": {k: list(v) for k, v in sorted(self.by_class.items())},
            "class_sizes": {k: len(v) for k, v in sorted(self.by_class.items())},
            "held_out": list(self.held_out),
            "identifiability": [row.as_dict() for row in self.identifiability],
            "noise_shot_count": len(self.noise_shots),
            "noise_shots": list(self.noise_shots),
            "published_ampere_turn_ratios": dict(sorted(self.published_ratios.items())),
            "training": list(self.training),
        }


def select_calibration_cohort(
    surveys: Iterable[ShotSurvey],
    *,
    hold: float = SUSTAINED_HOLD,
    minimum_probes: int = 40,
    held_out_fraction: float = 0.25,
    noise_limit: int = 0,
) -> CalibrationCohort:
    """Class the store and declare the split before any parameter is fitted.

    The split holds out whole shots and is taken from the END of each coil's own
    list of experiments, so a coil measured by six shots is trained on the first
    few and challenged on the last few.  Splitting per coil rather than per
    excitation family is what keeps every coil represented in both arms: the
    calibration campaigns are contiguous blocks of shots, so a split on shot
    number alone withholds whole coils and leaves them untested rather than
    untrained.
    """

    rows = tuple(surveys)
    experiments = calibration_experiments(
        rows, hold=hold, minimum_probes=minimum_probes
    )
    by_class: dict[str, list[int]] = {}
    for row in experiments:
        by_class.setdefault(str(row.experiment), []).append(row.shot)

    identifiability = identifiability_map(experiments)
    training: list[int] = []
    held_out: list[int] = []
    for row in identifiability:
        members = list(row.alone)
        if not members:
            continue
        take = (
            max(1, round(len(members) * held_out_fraction)) if len(members) > 1 else 0
        )
        boundary = len(members) - take
        for shot in members[:boundary]:
            if shot not in training and shot not in held_out:
                training.append(shot)
        for shot in members[boundary:]:
            if shot not in training and shot not in held_out:
                held_out.append(shot)

    noise = [
        row.shot
        for row in experiments
        if row.measures_noise and row.shot not in {*training, *held_out}
    ]
    if noise_limit > 0 and len(noise) > noise_limit:
        step = max(len(noise) // noise_limit, 1)
        noise = noise[::step][:noise_limit]

    cohort = CalibrationCohort(
        by_class={k: tuple(sorted(v)) for k, v in sorted(by_class.items())},
        training=tuple(sorted(training)),
        held_out=tuple(sorted(held_out)),
        noise_shots=tuple(sorted(noise)),
        identifiability=identifiability,
        published_ratios=integer_ampere_turn_ratios(rows),
    )
    cohort.validate()
    return cohort
