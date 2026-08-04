"""What the vacuum cohort established, pinned so the artifact can be rebuilt.

The fit that produced these numbers reads seventeen thousand shots to build its
census and then several dozen waveforms, which is not something artifact
authoring can do.  The results are therefore recorded here as data, with enough
provenance attached that each one can be traced to the shots that earned it and
recomputed by :mod:`nova.scripts.mast_vacuum_refinement`.

Three dispositions appear and the difference between them is the point of the
module.  A coil whose interval sits inside half a turn has been COUNTED, and its
integer is authored.  A coil that is identified but whose shots disagree by
several turns has been BOUNDED, and the fitted value is authored with an interval
wide enough to say so -- writing the nearest integer there would assert a
precision the cohort does not have.  A coil no shot could see stays UNRESOLVED and
is not written into the IDS at all.

None of this depends on the reconstruction's own machine description.  The
archive publishes a derived ampere-turn channel for ten of the thirteen coils
whose ratio to the conductor-current channel is an exact integer, and that ratio
agrees with every counted coil here; it is recorded alongside as corroboration and
was not an input to any fit.
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
"""Share of held-out probe signal power the fitted multipliers reproduce."""

NOMINAL_VARIANCE_EXPLAINED = 0.1014
"""Share the same prediction reaches with one turn per coil, the unfitted case."""

HELD_OUT_RESIDUAL = 4.3818e-3
"""Root-mean-square held-out probe residual [T] with the fitted multipliers."""

NOMINAL_RESIDUAL = 3.9498e-2
"""Root-mean-square held-out probe residual [T] before fitting."""

FIRST_SHOT = 11695
"""First shot the registry's configuration covers."""

LAST_SHOT = 30473
"""Last shot the registry's configuration covers."""

INTEGER_INTERVAL = 0.5
"""Interval half-width, in turns, inside which a turn count names one integer."""


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
    """One coil's signed turn count as the vacuum cohort measured it."""

    family: str
    channel: str
    multiplier: float
    half_width: float
    shot_count: int
    turns_per_multiplier: float = 1.0
    archive_multiplier: float | None = None

    @property
    def identified(self) -> bool:
        """Return whether any shot pinned this coil."""

        return self.shot_count > 0

    @property
    def counted(self) -> bool:
        """Return whether the interval names one integer turn count."""

        return self.identified and self.half_width < INTEGER_INTERVAL

    @property
    def turns(self) -> float:
        """Return the signed physical turn count to author.

        The multiplier scales the channel the store publishes, which is not always
        the current in one turn.  The solenoid is driven as two parallel circuits,
        so one turn carries half the feed current and the coil's turn count is
        twice the multiplier a fit against that channel returns.  Keeping the
        conversion here rather than in the fit means the fitted number stays a
        statement about the channel and the authored number a statement about the
        conductor.
        """

        value = self.multiplier * self.turns_per_multiplier
        return float(round(value)) if self.counted else float(value)

    @property
    def interval(self) -> Uncertainty:
        """Return the bound the cohort supports on the authored turn count."""

        half = self.half_width * self.turns_per_multiplier
        centre = self.multiplier * self.turns_per_multiplier
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
            "archive_multiplier": self.archive_multiplier,
            "channel": self.channel,
            "counted": self.counted,
            "family": self.family,
            "half_width": float(self.half_width),
            "identified": self.identified,
            "multiplier": float(self.multiplier),
            "shot_count": self.shot_count,
            "turns": self.turns,
        }


VACUUM_FITTED_TURNS = (
    FittedTurns("sol", "sol_current", 344.656565, 11.684209, 4, 2.0),
    FittedTurns("p2_inner_lower", "p2il_feed_current", 11.996472, 0.007024, 2),
    FittedTurns("p2_inner_upper", "p2iu_feed_current", 11.960446, 0.009394, 2),
    FittedTurns("p2_outer_lower", "p2ol_feed_current", 7.990149, 0.003038, 2, 1.0, 8.0),
    FittedTurns("p2_outer_upper", "p2ou_feed_current", 7.973877, 0.008579, 2, 1.0, 8.0),
    FittedTurns("p3_lower", "p3l_feed_current", 8.051440, 0.042680, 4, 1.0, 8.0),
    FittedTurns("p3_upper", "p3u_feed_current", 8.068439, 0.002014, 1, 1.0, 8.0),
    FittedTurns("p4_lower", "p4l_feed_current", 21.861222, 5.331953, 8),
    FittedTurns("p4_upper", "p4u_feed_current", 22.151385, 5.914825, 8),
    FittedTurns("p5_lower", "p5l_feed_current", 21.830221, 3.877191, 6),
    FittedTurns("p5_upper", "p5u_feed_current", 22.582544, 6.070722, 6),
    FittedTurns("p6_lower", "p6l_current", float("nan"), float("nan"), 0),
    FittedTurns("p6_upper", "p6u_current", float("nan"), float("nan"), 0),
)
"""Signed turn count per coil, in registry order.

Every identified multiplier is positive, so a positive current in the store's
channel produces the field a positive ``turns_with_sign`` predicts.  The poloidal
field components are unchanged by the source-to-target coordinate transform -- it
scales the flux and flips the safety factor and leaves B alone -- so this polarity
is the target convention's polarity and needed no reinterpretation.
"""

UNIDENTIFIED_LEVERAGE = 0.006
"""Largest share of predicted signal power P6 reached in any admissible shot."""

P6_ISOLATING_SHOTS = 3
"""Plasma-free shots that both isolate and sustain a P6 coil."""

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

CASE_PLATE_COUNT = 32
"""Case plates the registry publishes as a single undifferentiated family."""


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
    """Describe what the cohort established about one coil's turn count."""

    if row.counted:
        return (
            f"the signed vacuum response of {row.shot_count} shots that drove this "
            f"coil alone gives {row.multiplier:.3f} turns per ampere of "
            f"{row.channel}, which names {row.turns:+.0f} turns"
        )
    return (
        f"the signed vacuum response of {row.shot_count} shots bounds this coil at "
        f"{row.turns:.1f} turns but the shots disagree by "
        f"{row.half_width * row.turns_per_multiplier:.1f} turns, so the count is "
        "carried as an interval rather than rounded to one integer"
    )


def _turn_assumptions(row: FittedTurns) -> tuple[str, ...]:
    """State what the fitted turn count depends on."""

    assumptions = [
        "the winding pack carries a uniform current density over its measured "
        "outline, which fixes the field shape a probe standing clear of the pack "
        "reads and is why probes inside two pack widths are not read at all",
        "the excitation is held long enough for the coil case and the vessel to "
        "give back the current its own ramp induced, which is what separates a "
        "coil's own field from the transient around it",
    ]
    if row.turns_per_multiplier != 1.0:
        assumptions.append(
            "the coil is driven as two parallel circuits, so one turn carries half "
            "the measured feed current and the turn count is twice the fitted "
            "multiplier"
        )
    if row.archive_multiplier is not None:
        assumptions.append(
            f"the archive's own derived ampere-turn channel multiplies this coil's "
            f"conductor current by {row.archive_multiplier:g}, which the fit "
            f"reproduces without having been given it"
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
                        "no plasma-free shot lets this coil be seen: the "
                        f"{P6_ISOLATING_SHOTS} shots that drive it alone and hold it "
                        "also drive a neighbouring circuit two orders harder, "
                        f"leaving it {UNIDENTIFIED_LEVERAGE:.1%} of the predicted "
                        "signal power at most"
                    ),
                    assumptions=(
                        "a coil contributing under one part in a hundred of the "
                        "predicted signal takes whatever multiplier the residual "
                        "wants, so a number fitted there would report the residual",
                        "the store publishes this coil's excitation as ampere-turns "
                        "rather than as the current in one conductor, so even a "
                        "clean shot would fix the product and not the count",
                        "that same published product is what lets the coil be driven "
                        "at all: its channel drive carries one ampere-turn per "
                        "ampere, so the forward-model column is fixed without the "
                        "count this record leaves open",
                    ),
                    blocks_axisymmetric_forward_model=True,
                )
            )
            continue
        records.append(
            EvidenceRecord(
                path=path,
                evidence=FieldEvidence.FITTED,
                first_shot=first_shot,
                last_shot=last_shot,
                statement=_turn_statement(row),
                assumptions=_turn_assumptions(row),
                source=_vacuum_fit(
                    f"{row.shot_count} isolating shots, channel {row.channel}"
                ),
                uncertainty=row.interval,
            )
        )
    return records


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
                "excitation, and the store does publish error-field coil currents",
                "predicting that excitation needs the error-field coil winding "
                "geometry, which the registry does not carry, so the blocker is a "
                "missing conductor model rather than missing shots",
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


def fitted_evidence(
    first_shot: int = FIRST_SHOT,
    last_shot: int = LAST_SHOT,
) -> list[EvidenceRecord]:
    """Return every record the vacuum refinement contributes."""

    return [
        *fitted_turn_records(first_shot, last_shot),
        *fitted_diagnostic_records(first_shot, last_shot),
        *fitted_passive_records(first_shot, last_shot),
    ]


SUPERSEDED_SEED_PATHS = frozenset(
    {
        "pf_active/coil/element/turns_with_sign",
        "pf_active/circuit/connections",
        "magnetics/b_field_pol_probe/poloidal_angle",
        "magnetics/b_field_pol_probe/position/phi",
        "magnetics/b_field_phi_probe/toroidal_angle",
        "magnetics/flux_loop(saddle)/traversal_sign",
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
