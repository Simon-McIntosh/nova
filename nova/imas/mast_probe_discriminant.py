"""Separate a probe's own calibration error from a near-conductor shape error.

Two outboard axial probes carry most of the difference between the two machine
descriptions' magnetics misfit, and the descriptions place them at the same
point to within a nanometre.  A cross-source pose difference therefore cannot be
the cause, which leaves two candidates that both descriptions would inherit: the
channel's calibration is wrong, or the field the coils are asserted to produce at
that point is wrong.  The two are separable, and this module fixes how before any
number is fitted.

What makes them separable is that each outboard axial probe shares its position
with an outboard radial probe.  One point, two orthogonal components, both
measured.  A rigid error in one probe -- a wrong effective area, or a winding
normal rotated in the poloidal plane -- is a linear map from the true field at
that point to the reported number, so it is fully described by two parameters and
predicts the residual from quantities that are *measured* rather than modelled.
A misrepresented current arrangement inside a neighbouring winding pack is not a
map on the field at a point at all: it changes the field itself, differently for
every coil, and it perturbs both components.

    reported = gain * (B_z cos(tilt) + B_r sin(tilt))

is the whole of the probe-side hypothesis.  Its two falsifiable consequences are
that one ``gain`` fits every excitation family, and that the part of the residual
the tilt explains is proportional to the *co-located radial channel's own
reading* -- no forward model enters that second test, so it survives whatever
the winding description gets wrong.

The field-shape hypothesis makes the opposite prediction on both counts.  A
uniform current density over a pack outline is the honest reading of an outline
and the wrong reading of a winding, and how wrong it is falls off with distance,
so the apparent gain it induces depends on which coil is driving: large for the
coils whose packs the probe stands beside, negligible for coils far enough away
that only their total ampere-turns reach the probe.  It also cannot leave the
co-located radial channel clean, because displacing current inside a pack moves
both field components at a nearby point.

So the discriminant is: hold the excitation family fixed and ask what gain the
data wants; then ask whether one gain served all of them.  An excitation-invariant
gain is a property of the probe.  An excitation-selective gain is a property of
the geometry.

Everything below -- the hypotheses, the statistics, the thresholds and the
mapping from statistics to verdict -- is fixed here and consumed by
:mod:`nova.imas.mast_probe_calibration`, which does the fitting.  Nothing in this
module reads data or fits anything, so the criterion cannot be tuned to the
answer it produced.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping

CO_LOCATION_TOLERANCE = 1.0e-3
"""Metres two probes may be apart and still be treated as one point.

The outboard arrays are built as radial/axial pairs on shared mounts, and the
registry records most pairs at byte-identical coordinates.  A millimetre is the
scale on which the pair's separation stops mattering: the field gradient across
the outboard array is of order a tesla per metre at the currents these
experiments drive, so a millimetre of separation moves the two probes' fields by
a millitesla at most -- comparable with the noise floor and far below the
excesses under test.  A pair further apart than this is not treated as
co-located and its probe is reported as having no orthogonal partner rather than
being paired approximately.
"""

FAMILY_LEVERAGE = 0.5
"""Share of a probe's predicted signal power one coil family must carry.

A per-family gain is only that family's gain if the family dominates the
prediction.  Below a half, the number returned is a blend of the family's gain
with whatever else was energised, and the excitation-selectivity test -- which is
the whole discriminant -- would compare blends instead of gains.
"""

SPREAD_SIGNIFICANCE = 3.0
"""Standard errors of per-family gain spread that count as excitation-selective.

The probe-side hypothesis asserts one gain for every family, so the spread of
the per-family gains is a test of it directly.  Three pooled standard errors is
the point at which the spread stops being consistent with one gain measured
several times.
"""

FLOOR_MULTIPLE = 3.0
"""Multiples of a channel's measured noise scatter a rigid fit must reach.

A probe-side verdict claims the residual IS the calibration error, so the
residual left after the best gain-and-tilt fit has to fall to the level at which
the channel stops being informative.  Three times the channel's own quiescent
scatter allows for the excitation's own reproducibility without admitting a
residual an order of magnitude above the floor.
"""

MAXIMUM_TILT = 0.2
"""Radians of poloidal rotation still attributable to how a probe was mounted.

About eleven degrees.  A fitted angle larger than this is not a mounting
tolerance on a probe screwed to a bracket; it is the fit absorbing something
else, and a promotion at such an angle would write a fiction into the
description.  A probe whose only good fit needs a larger angle stays
dual-valued.
"""

MINIMUM_FAMILIES = 3
"""Coil families a probe needs before its gain spread means anything.

Two families give a spread with no notion of scatter, so the significance test
cannot run.  Three is the smallest number at which one family disagreeing with
the others is distinguishable from the others disagreeing with it.
"""


class DiscriminantError(ValueError):
    """Raised when a discriminant statistic or verdict is inadmissible."""


class ProbeVerdict(StrEnum):
    """Which cause the data assigns a probe's excess."""

    CALIBRATION_GAIN = "calibration_gain"
    CALIBRATION_TILT = "calibration_tilt"
    FIELD_SHAPE = "field_shape"
    INSEPARABLE = "inseparable"
    NOT_TESTED = "not_tested"


@dataclass(frozen=True, order=True)
class DiscriminantStatistics:
    """The five numbers the verdict is a function of, for one probe.

    ``gain_spread`` and ``gain_standard_error`` are pooled over the excitation
    families that cleared :data:`FAMILY_LEVERAGE`; ``tilt`` regresses this
    probe's residual on the co-located orthogonal channel's measured signal, so
    it is a measurement-to-measurement coefficient and carries no forward model.
    ``rigid_residual`` and ``noise_floor`` are both in tesla.
    """

    channel: str
    family_count: int
    gain: float
    gain_standard_error: float
    gain_spread: float
    near_coil_gain: float
    distant_coil_gain: float
    tilt: float
    tilt_standard_error: float
    tilt_variance_removed: float
    partner_channel: str
    partner_excess_share: float
    rigid_residual: float
    noise_floor: float

    def validate(self) -> None:
        """Reject statistics that cannot support any verdict."""

        if not self.channel:
            raise DiscriminantError("statistics must name a channel")
        if self.family_count < 0:
            raise DiscriminantError(
                f"{self.channel!r} cannot have {self.family_count} families"
            )
        for value, name in (
            (self.gain_standard_error, "gain standard error"),
            (self.gain_spread, "gain spread"),
            (self.tilt_standard_error, "tilt standard error"),
            (self.rigid_residual, "rigid residual"),
            (self.noise_floor, "noise floor"),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise DiscriminantError(
                    f"{self.channel!r} {name} must be finite and non-negative"
                )
        for value, name in (
            (self.gain, "gain"),
            (self.tilt, "tilt"),
            (self.near_coil_gain, "near-coil gain"),
            (self.distant_coil_gain, "distant-coil gain"),
        ):
            if not math.isfinite(value):
                raise DiscriminantError(f"{self.channel!r} {name} must be finite")
        if self.noise_floor <= 0.0:
            raise DiscriminantError(
                f"{self.channel!r} needs a measured noise floor to be judged"
            )

    @property
    def excitation_selective(self) -> bool:
        """Return whether the per-family gains refuse to be one gain."""

        if self.family_count < MINIMUM_FAMILIES:
            return False
        if self.gain_standard_error <= 0.0:
            return bool(self.gain_spread > 0.0)
        return bool(self.gain_spread > SPREAD_SIGNIFICANCE * self.gain_standard_error)

    @property
    def near_field_contrast(self) -> float:
        """Return how far the adjacent coils' gain sits from the distant coils'."""

        return abs(self.near_coil_gain - self.distant_coil_gain)

    @property
    def rigid_fit_reaches_floor(self) -> bool:
        """Return whether a gain-and-tilt fit explains the channel to its floor."""

        return bool(self.rigid_residual <= FLOOR_MULTIPLE * self.noise_floor)

    @property
    def tilt_admissible(self) -> bool:
        """Return whether the fitted angle is small enough to be a mounting error."""

        return bool(abs(self.tilt) <= MAXIMUM_TILT)

    @property
    def tilt_identified(self) -> bool:
        """Return whether the angle is resolved and carries real variance."""

        if not self.tilt_admissible or self.tilt_standard_error <= 0.0:
            return False
        return bool(
            abs(self.tilt) > SPREAD_SIGNIFICANCE * self.tilt_standard_error
            and self.tilt_variance_removed > 0.5
        )

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical JSON representation."""

        return {
            "channel": self.channel,
            "distant_coil_gain": self.distant_coil_gain,
            "excitation_selective": self.excitation_selective,
            "family_count": self.family_count,
            "gain": self.gain,
            "gain_spread": self.gain_spread,
            "gain_standard_error": self.gain_standard_error,
            "near_coil_gain": self.near_coil_gain,
            "near_field_contrast": self.near_field_contrast,
            "noise_floor": self.noise_floor,
            "partner_channel": self.partner_channel,
            "partner_excess_share": self.partner_excess_share,
            "rigid_fit_reaches_floor": self.rigid_fit_reaches_floor,
            "rigid_residual": self.rigid_residual,
            "tilt": self.tilt,
            "tilt_identified": self.tilt_identified,
            "tilt_standard_error": self.tilt_standard_error,
            "tilt_variance_removed": self.tilt_variance_removed,
        }


def adjudicate(statistics: DiscriminantStatistics) -> ProbeVerdict:
    """Map one probe's statistics onto its verdict.

    The order of the tests is part of the criterion.  Excitation selectivity is
    asked first because it is the one statement the two hypotheses disagree
    about unconditionally: a rigid probe error cannot make the gain depend on
    which coil is driving, whatever else is true.  Only once the gain has been
    shown to be one number does the question of whether it is an area or an
    angle arise, and only a rigid fit that actually reaches the channel's noise
    floor earns a probe-side verdict -- a fit that leaves the residual an order
    of magnitude high has not explained the channel, however stable its
    coefficient.
    """

    statistics.validate()
    if statistics.family_count < MINIMUM_FAMILIES:
        return ProbeVerdict.NOT_TESTED
    if statistics.excitation_selective:
        return ProbeVerdict.FIELD_SHAPE
    if not statistics.rigid_fit_reaches_floor:
        return ProbeVerdict.INSEPARABLE
    if statistics.tilt_identified:
        return ProbeVerdict.CALIBRATION_TILT
    resolution = SPREAD_SIGNIFICANCE * statistics.gain_standard_error
    if abs(statistics.gain - 1.0) > resolution:
        return ProbeVerdict.CALIBRATION_GAIN
    return ProbeVerdict.INSEPARABLE


def promotable(verdict: ProbeVerdict) -> bool:
    """Return whether a verdict licenses writing a value into the description.

    Only a probe-side verdict does.  A field-shape verdict says the probe is
    fine and the geometry is not, so promoting a gain under it would bake a
    geometry error into a sensor record and hide it from the description that
    owns it.
    """

    return verdict in (ProbeVerdict.CALIBRATION_GAIN, ProbeVerdict.CALIBRATION_TILT)


@dataclass(frozen=True)
class PreRegistration:
    """The criterion as a citable, byte-stable object.

    Recording it as data rather than only as prose lets the fitting module assert
    it did not move: the thresholds a run applied are written beside the run's
    results, and a later reader can compare them with the ones declared here.
    """

    co_location_tolerance: float = CO_LOCATION_TOLERANCE
    family_leverage: float = FAMILY_LEVERAGE
    spread_significance: float = SPREAD_SIGNIFICANCE
    floor_multiple: float = FLOOR_MULTIPLE
    maximum_tilt: float = MAXIMUM_TILT
    minimum_families: int = MINIMUM_FAMILIES

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical JSON representation."""

        return {
            "co_location_tolerance": self.co_location_tolerance,
            "family_leverage": self.family_leverage,
            "floor_multiple": self.floor_multiple,
            "maximum_tilt": self.maximum_tilt,
            "minimum_families": self.minimum_families,
            "spread_significance": self.spread_significance,
        }

    def agrees_with(self, applied: Mapping[str, Any]) -> bool:
        """Return whether a recorded run used exactly these thresholds."""

        return self.as_dict() == dict(applied)


HYPOTHESES = {
    "calibration_gain": (
        "The channel reports gain times the axial field with gain not one. One "
        "gain fits every excitation family, the co-located radial channel shows "
        "no matching excess, and the gain repeats across campaigns."
    ),
    "calibration_tilt": (
        "The probe's winding normal is rotated in the poloidal plane, so the "
        "channel reports the axial field plus the sine of the angle times the "
        "radial field. The residual is then proportional to the co-located "
        "radial channel's own reading with one angle for every family, a test "
        "that reads two measurements and no forward model."
    ),
    "field_shape": (
        "Uniform current density over a neighbouring pack outline misplaces the "
        "current inside the pack, so the modelled field at a probe beside it is "
        "wrong by an amount that depends on which coil is driving. The apparent "
        "gain is then excitation-selective, larger for the adjacent coils than "
        "for distant ones, and the co-located radial channel carries a "
        "correlated excess."
    ),
}
"""What each candidate cause asserts, and the consequence that tests it."""
