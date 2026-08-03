"""Fit the signed coil-to-probe vacuum response from measured excitations.

Geometry fixes the shape of a coil's field and nothing else.  Two quantities
survive it: how many times the conductor goes round, which sets the amplitude,
and which way the current flows relative to the coordinate frame, which sets the
polarity.  Both are recovered here from shots that drove known currents with no
plasma present, by predicting each probe from the registry outlines and asking
what signed multiplier on each coil's measured channel reproduces the reading.

The prediction is linear in the multipliers, so the fit is a least-squares solve
and not a search.  For a probe ``p`` and a sample ``t``,

    y(p, t) = sum_c G(p, c) n_c I_c(t)

with ``G`` the field a single-turn ring of the coil's own cross-section produces
at the probe, per ampere, and ``I_c`` the current the store measured in coil
``c``'s channel.  ``n_c`` is what the fit returns.  When the channel measures the
current in one conductor, ``n_c`` is the coil's turn count; when it already
reports ampere-turns, ``n_c`` is one, and a fit that says so has confirmed the
channel's meaning rather than measured a turn count.  The distinction is a
property of the archive, not of the coil, so both cases are reported as found.

Nothing here reads the reconstruction's filament subdivision.  ``G`` comes from
the measured winding-pack outline through the polygon section kernel, which
spreads one ampere uniformly over the cross-section; the number of cells a
solver would cut that outline into does not appear and cannot change the answer.

Two things had to be got right for the numbers to mean anything, and both are
enforced here rather than left to the caller.

A probe close to a winding pack does not measure the pack's total current.  It
measures how that current is arranged inside the pack, and the outline says only
where the copper is, not how the turns are stacked within it.  Uniform current
density over the footprint is the honest reading of an outline and the wrong
reading of a winding, so probes within :data:`MINIMUM_STANDOFF` pack widths of any
deliberately excited coil are dropped from that shot.  Left in, they bias a
twenty-three turn coil to twenty-eight; taken out, the estimate is flat over every
standoff beyond the cut.

Coils driven in lockstep cannot be separated.  MAST's pairs are wired in series,
so one shot that energises a pair constrains the pair's total and says nothing
about the division between its members.  A solve given both columns returns two
numbers anyway, so the estimates are screened by their own covariance and a
degenerate pair is refused rather than reported.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import shapely

from nova.biot.polygon import polygon_greens
from nova.imas.mast_vacuum_cohort import (
    COIL_DRIVES,
    ENERGISED_CURRENT,
    EXCITATION_CURRENT,
    RADIAL_AXIS_FAMILIES,
    CohortError,
    ProbeChannel,
    ShotWaveforms,
)

MINIMUM_STANDOFF = 2.0
"""Pack widths a probe must stand off an excited coil to be read.

Measured, not chosen.  Held-out prediction improves as the cut widens -- 0.975 of
the variance at half a width, 0.978 at one, 0.989 at two -- and then stops
improving, while the coils that have probes in their near field move from
twenty-eight turns with every probe included to twenty-two and settle there.  Past
two widths the only thing that changes is that shots start dropping out for want
of an admissible probe, which costs held-out coverage and buys nothing.
"""

MINIMUM_LEVERAGE = 0.02
"""Share of predicted signal power a coil must carry to be read from a shot.

Below this the coil is a rounding correction on somebody else's field and its
multiplier is whatever the residual happens to want.
"""

MAXIMUM_RELATIVE_ERROR = 0.05
"""Largest standard error, relative to the value, that counts as identified.

Coils wired in series carry the same waveform, so a single shot cannot say how
the pair's total is divided between them however clean the data is; the solve
usually reports that as a large error on each, and this threshold turns it into a
refusal instead of a number.
"""

MAXIMUM_CORRELATION = 0.95
"""Largest correlation with another fitted coil that still counts as identified.

The relative-error test alone does not catch a degenerate pair, because a pair
constrained only in its difference can be handed equal and opposite multipliers of
several thousand turns each with a small error on both.  What distinguishes that
case is the off-diagonal: the two parameters come back almost perfectly
anti-correlated.  Measured on this cohort, the estimates it removes are the ones
scattering over thousands of turns while the ones it keeps agree to a few percent.
"""


class ResponseError(ValueError):
    """Raised when a vacuum response cannot be assembled or identified."""


@dataclass(frozen=True, order=True)
class ProbeTarget:
    """One field probe's position and the poloidal axis it is sensitive to."""

    channel: str
    family: str
    registry_index: int
    r: float
    z: float
    radial_cosine: float
    axial_sine: float

    def validate(self) -> None:
        """Reject a target whose sensitive axis is not a unit vector."""

        norm = math.hypot(self.radial_cosine, self.axial_sine)
        if not math.isclose(norm, 1.0, rel_tol=1e-9, abs_tol=1e-9):
            raise ResponseError(
                f"probe {self.channel!r} sensitive axis has length {norm}, not one"
            )
        if self.r <= 0.0:
            raise ResponseError(f"probe {self.channel!r} must sit at positive radius")


def probe_targets(
    probes: Sequence[Mapping[str, Any]],
    channels: Iterable[ProbeChannel],
    *,
    radial_families: frozenset[str] = RADIAL_AXIS_FAMILIES,
) -> tuple[ProbeTarget, ...]:
    """Pose each named channel and orient it along one poloidal axis.

    The registry records one poloidal angle for every probe, so it cannot say
    which families measure the radial component.  ``radial_families`` supplies
    that assignment and is a hypothesis under test: :func:`score_axis_assignment`
    refits under each candidate and reports which one the data prefers.
    """

    targets = []
    for channel in channels:
        row = probes[channel.registry_index]
        if str(row["family"]) != channel.family:
            raise ResponseError(
                f"channel {channel.channel!r} maps to registry family "
                f"{row['family']!r}, not {channel.family!r}"
            )
        radial = channel.family in radial_families
        target = ProbeTarget(
            channel=channel.channel,
            family=channel.family,
            registry_index=channel.registry_index,
            r=float(row["pose"][0]),
            z=float(row["pose"][1]),
            radial_cosine=1.0 if radial else 0.0,
            axial_sine=0.0 if radial else 1.0,
        )
        target.validate()
        targets.append(target)
    return tuple(targets)


def coil_sections(geometry: Mapping[str, Any]) -> dict[str, tuple[np.ndarray, ...]]:
    """Return each active family's winding-pack outlines as vertex arrays."""

    sections: dict[str, tuple[np.ndarray, ...]] = {}
    for family, wkb_hex in sorted(geometry["active_components"].items()):
        outline = shapely.from_wkb(bytes.fromhex(wkb_hex))
        parts = getattr(outline, "geoms", None)
        polygons = (outline,) if parts is None else tuple(parts)
        sections[family] = tuple(
            np.asarray(polygon.exterior.coords, dtype=float)[:-1]
            for polygon in polygons
        )
    return sections


def coil_response_matrix(
    geometry: Mapping[str, Any],
    targets: Sequence[ProbeTarget],
    *,
    families: Sequence[str] | None = None,
) -> np.ndarray:
    """Field each coil produces at each probe, per ampere-turn [T/(A.turn)].

    Row ``p`` column ``c`` is the component of coil ``c``'s field along probe
    ``p``'s sensitive axis when one ampere flows once round the coil's
    cross-section.  A family resolved into several outlines carries its turns in
    proportion to section area, which is what a uniformly wound pack does, so the
    family's column is the area-weighted mean of its parts.
    """

    sections = coil_sections(geometry)
    order = _order(families)
    missing = [family for family in order if family not in sections]
    if missing:
        raise ResponseError(f"registry carries no active component {missing}")
    target_r = np.asarray([target.r for target in targets], dtype=float)
    target_z = np.asarray([target.z for target in targets], dtype=float)
    cosine = np.asarray([target.radial_cosine for target in targets], dtype=float)
    sine = np.asarray([target.axial_sine for target in targets], dtype=float)

    response = np.zeros((len(targets), len(order)), dtype=float)
    for column, family in enumerate(order):
        parts = sections[family]
        areas = np.asarray(
            [abs(shapely.Polygon(vertices).area) for vertices in parts], dtype=float
        )
        total = float(areas.sum())
        if total <= 0.0:
            raise ResponseError(f"active component {family!r} has no cross-section")
        for vertices, area in zip(parts, areas, strict=True):
            _, radial, axial = polygon_greens(target_r, target_z, vertices)
            response[:, column] += (area / total) * (cosine * radial + sine * axial)
    return response


def probe_standoff(
    geometry: Mapping[str, Any],
    targets: Sequence[ProbeTarget],
    *,
    families: Sequence[str] | None = None,
) -> np.ndarray:
    """Distance from each probe to each coil, in that coil's pack widths.

    The scale is the pack's SMALLER cross-section dimension, not its diagonal.
    What a nearby probe is sensitive to, beyond the total ampere-turns, is how
    the current is distributed across the face it is looking at, and the smaller
    dimension is how far the current can be displaced in that direction.  Using
    the diagonal instead makes a long thin coil look near-field everywhere -- the
    solenoid is three metres long and fifty millimetres thick, so every probe in
    the machine sits inside one of its diagonals and none of them inside one of
    its widths.  A probe inside a pack reports zero and is excluded by any
    positive cut.
    """

    sections = coil_sections(geometry)
    order = _order(families)
    standoff = np.zeros((len(targets), len(order)), dtype=float)
    for column, family in enumerate(order):
        polygons = [shapely.Polygon(vertices) for vertices in sections[family]]
        bounds = np.asarray([polygon.bounds for polygon in polygons], dtype=float)
        width = float(
            np.max(np.minimum(bounds[:, 2] - bounds[:, 0], bounds[:, 3] - bounds[:, 1]))
        )
        if width <= 0.0:
            raise ResponseError(f"active component {family!r} has no extent")
        for row, target in enumerate(targets):
            point = shapely.Point(target.r, target.z)
            standoff[row, column] = (
                min(point.distance(polygon) for polygon in polygons) / width
            )
    return standoff


def _order(families: Sequence[str] | None) -> tuple[str, ...]:
    if families is not None:
        return tuple(families)
    return tuple(drive.family for drive in COIL_DRIVES)


def baseline_window(waveforms: ShotWaveforms) -> np.ndarray:
    """Mark the samples a probe's standing offset is measured in.

    A probe integrator reads its own offset, so the zero of every channel is
    measured on the shot rather than assumed.  The window is where every drive
    channel is still below the energised floor, which is a statement about the
    machine and not about the acquisition clock.
    """

    return waveforms.baseline_mask


def energised_families(
    waveforms: ShotWaveforms,
    families: Sequence[str],
) -> tuple[str, ...]:
    """Return the coils carrying enough current to belong in the prediction."""

    return tuple(
        family
        for family in families
        if _peak_drive(waveforms, family) >= ENERGISED_CURRENT
    )


def excited_families(
    waveforms: ShotWaveforms,
    families: Sequence[str],
) -> tuple[str, ...]:
    """Return the coils a supply deliberately drove on this shot.

    The near-field screen keys off this and not off :func:`energised_families`,
    because a coil idling at a few hundred amperes contributes a field four
    orders below the excitation whatever a probe's standoff from it is.  Screening
    on the pickup floor instead throws away every probe on every shot: with
    thirteen coils each excluding its own neighbourhood, the union is the array.
    """

    return tuple(
        family
        for family in families
        if _peak_drive(waveforms, family) >= EXCITATION_CURRENT
    )


def _peak_drive(waveforms: ShotWaveforms, family: str) -> float:
    values = waveforms.drives.get(family)
    if values is None:
        return 0.0
    finite = values[np.isfinite(values)]
    return float(np.max(np.abs(finite))) if finite.size else 0.0


@dataclass(frozen=True)
class ResponseModel:
    """The geometry-derived part of the vacuum response, built once.

    Everything here comes from the registry outlines and the probe poses, so it
    is fixed across the whole cohort and no shot can change it.  Only the
    standoff cut is per shot, because which coils were energised is.
    """

    families: tuple[str, ...]
    targets: tuple[ProbeTarget, ...]
    response: np.ndarray
    standoff: np.ndarray
    radial_families: tuple[str, ...]
    minimum_standoff: float = MINIMUM_STANDOFF

    @classmethod
    def build(
        cls,
        geometry: Mapping[str, Any],
        probes: Sequence[Mapping[str, Any]],
        channels: Sequence[ProbeChannel],
        *,
        radial_families: frozenset[str] = RADIAL_AXIS_FAMILIES,
        families: Sequence[str] | None = None,
        minimum_standoff: float = MINIMUM_STANDOFF,
    ) -> ResponseModel:
        """Pose the probes, couple every coil to them and measure the standoffs."""

        order = _order(families)
        targets = probe_targets(probes, channels, radial_families=radial_families)
        return cls(
            families=order,
            targets=targets,
            response=coil_response_matrix(geometry, targets, families=order),
            standoff=probe_standoff(geometry, targets, families=order),
            radial_families=tuple(sorted(radial_families)),
            minimum_standoff=minimum_standoff,
        )

    def select(self, families: Sequence[str]) -> ResponseModel:
        """Return the same model restricted to a subset of coils."""

        columns = [self.families.index(family) for family in families]
        return ResponseModel(
            families=tuple(families),
            targets=self.targets,
            response=self.response[:, columns],
            standoff=self.standoff[:, columns],
            radial_families=self.radial_families,
            minimum_standoff=self.minimum_standoff,
        )

    def admissible_probes(self, energised: Sequence[str]) -> np.ndarray:
        """Mark the probes far enough from every energised coil to be read.

        The test is over the energised coils together, not one at a time: a probe
        sitting inside one coil's near field carries that coil's shape whatever
        other coil the fit is trying to read, so it has to leave the shot
        entirely rather than leave one column.
        """

        keep = np.ones(len(self.targets), dtype=bool)
        for family in energised:
            if family not in self.families:
                continue
            column = self.families.index(family)
            keep &= self.standoff[:, column] >= self.minimum_standoff
        return keep

    def design(
        self,
        waveforms: ShotWaveforms,
        *,
        stride: int = 1,
        minimum_baseline: int = 20,
    ) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
        """Build one shot's design rows, offsets removed and probes screened.

        Only probes the store recorded on this shot contribute, only samples the
        window tests admitted, and only probes standing clear of every energised
        coil.  Each probe's pre-excitation mean is subtracted, so a fitted
        multiplier answers for the change in field the coils produced and not for
        an integrator's standing offset.
        """

        quiet = baseline_window(waveforms)
        if int(quiet.sum()) < minimum_baseline:
            raise ResponseError(
                f"shot {waveforms.shot} has {int(quiet.sum())} pre-excitation "
                f"samples, below the {minimum_baseline} needed to measure offsets"
            )
        keep = self.admissible_probes(excited_families(waveforms, self.families))
        allowed = {
            target.channel for target, ok in zip(self.targets, keep, strict=True) if ok
        }
        drive = np.zeros((waveforms.time.size, len(self.families)), dtype=float)
        for column, family in enumerate(self.families):
            values = waveforms.drives.get(family)
            if values is not None:
                drive[:, column] = np.nan_to_num(values)

        index = {target.channel: row for row, target in enumerate(self.targets)}
        rows: list[np.ndarray] = []
        observations: list[np.ndarray] = []
        used: list[str] = []
        samples = np.flatnonzero(waveforms.sample_mask)[::stride]
        if samples.size == 0:
            raise ResponseError(f"shot {waveforms.shot} admits no samples")
        for channel, signal in sorted(waveforms.probes.items()):
            row = index.get(channel)
            if row is None or channel not in allowed:
                continue
            finite = np.isfinite(signal)
            if not finite[quiet].any():
                continue
            offset = float(np.mean(signal[quiet & finite]))
            take = samples[finite[samples]]
            if take.size == 0:
                continue
            rows.append(drive[take] * self.response[row, :])
            observations.append(signal[take] - offset)
            used.append(channel)
        if not rows:
            raise ResponseError(
                f"shot {waveforms.shot} keeps no probe clear of its energised coils"
            )
        return np.vstack(rows), np.concatenate(observations), tuple(used)


@dataclass
class NormalEquations:
    """Accumulated least-squares system for the signed coil multipliers.

    Samples are added shot by shot rather than stacked, because a cohort holds
    more probe-samples than fit into one design matrix and the normal equations
    carry everything the solve needs.  ``residual_square`` is kept so a fit's own
    goodness can be reported without a second pass.
    """

    size: int
    gram: np.ndarray
    moment: np.ndarray
    residual_square: float = 0.0
    sample_count: int = 0
    shots: tuple[int, ...] = ()

    @classmethod
    def empty(cls, size: int) -> NormalEquations:
        """Return a system with no samples in it."""

        return cls(
            size=size,
            gram=np.zeros((size, size), dtype=float),
            moment=np.zeros(size, dtype=float),
        )

    def add(self, design: np.ndarray, observed: np.ndarray, shot: int) -> None:
        """Fold one shot's rows into the accumulated system."""

        if design.ndim != 2 or design.shape[1] != self.size:
            raise ResponseError(
                f"design has shape {design.shape}, expected (n, {self.size})"
            )
        if observed.shape != (design.shape[0],):
            raise ResponseError("design and observation row counts differ")
        self.gram += design.T @ design
        self.moment += design.T @ observed
        self.residual_square += float(observed @ observed)
        self.sample_count += int(design.shape[0])
        self.shots = (*self.shots, shot)

    def solve(self) -> tuple[np.ndarray, np.ndarray, float]:
        """Return the multipliers, their covariance and the residual variance."""

        if self.sample_count <= self.size:
            raise ResponseError(
                f"{self.sample_count} samples cannot identify {self.size} multipliers"
            )
        solution, *_ = np.linalg.lstsq(self.gram, self.moment, rcond=None)
        residual = self.residual_square - float(self.moment @ solution)
        degrees = max(self.sample_count - self.size, 1)
        variance = max(residual, 0.0) / degrees
        return solution, variance * np.linalg.pinv(self.gram), variance

    @property
    def condition(self) -> float:
        """Return the accumulated system's condition number."""

        return float(np.linalg.cond(self.gram))


@dataclass(frozen=True)
class ResponseFit:
    """The fitted multipliers, the system that produced them and its residual."""

    families: tuple[str, ...]
    multipliers: dict[str, float]
    standard_errors: dict[str, float]
    residual_rms: float
    signal_rms: float
    condition: float
    sample_count: int
    shots: tuple[int, ...]
    probe_channels: tuple[str, ...]

    @property
    def variance_explained(self) -> float:
        """Return the fraction of observed signal power the model reproduces."""

        if self.signal_rms <= 0.0:
            return 0.0
        return float(1.0 - (self.residual_rms / self.signal_rms) ** 2)

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical JSON representation."""

        return {
            "condition": float(self.condition),
            "families": list(self.families),
            "multipliers": {k: float(v) for k, v in sorted(self.multipliers.items())},
            "probe_channels": list(self.probe_channels),
            "residual_rms": float(self.residual_rms),
            "sample_count": self.sample_count,
            "shots": list(self.shots),
            "signal_rms": float(self.signal_rms),
            "standard_errors": {
                k: float(v) for k, v in sorted(self.standard_errors.items())
            },
            "variance_explained": float(self.variance_explained),
        }


def fit_response(
    waveforms: Iterable[ShotWaveforms],
    model: ResponseModel,
    *,
    stride: int = 1,
) -> ResponseFit:
    """Fit one signed multiplier per coil across a set of shots."""

    system = NormalEquations.empty(len(model.families))
    channels: set[str] = set()
    for shot in waveforms:
        design, observed, used = model.design(shot, stride=stride)
        system.add(design, observed, shot.shot)
        channels |= set(used)
    solution, covariance, variance = system.solve()
    return ResponseFit(
        families=model.families,
        multipliers=dict(zip(model.families, map(float, solution), strict=True)),
        standard_errors={
            family: float(math.sqrt(max(covariance[column, column], 0.0)))
            for column, family in enumerate(model.families)
        },
        residual_rms=float(math.sqrt(max(variance, 0.0))),
        signal_rms=float(
            math.sqrt(system.residual_square / max(system.sample_count, 1))
        ),
        condition=system.condition,
        sample_count=system.sample_count,
        shots=system.shots,
        probe_channels=tuple(sorted(channels)),
    )


@dataclass(frozen=True)
class PredictionScore:
    """How well a fitted response predicts shots it was not trained on."""

    shots: tuple[int, ...]
    residual_rms: float
    signal_rms: float
    sample_count: int
    per_shot: Mapping[int, float] = None  # type: ignore[assignment]

    @property
    def variance_explained(self) -> float:
        """Return the fraction of held-out signal power the model reproduces."""

        if self.signal_rms <= 0.0:
            return 0.0
        return float(1.0 - (self.residual_rms / self.signal_rms) ** 2)

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical JSON representation."""

        return {
            "per_shot_variance_explained": {
                str(k): float(v) for k, v in sorted((self.per_shot or {}).items())
            },
            "residual_rms": float(self.residual_rms),
            "sample_count": self.sample_count,
            "shots": list(self.shots),
            "signal_rms": float(self.signal_rms),
            "variance_explained": float(self.variance_explained),
        }


def score_prediction(
    waveforms: Iterable[ShotWaveforms],
    model: ResponseModel,
    multipliers: Mapping[str, float],
    *,
    stride: int = 1,
) -> PredictionScore:
    """Predict shots with fixed multipliers and report the residual."""

    vector = np.asarray(
        [multipliers.get(family, 0.0) for family in model.families], dtype=float
    )
    residual = 0.0
    signal = 0.0
    count = 0
    shots: list[int] = []
    per_shot: dict[int, float] = {}
    for shot in waveforms:
        try:
            design, observed, _ = model.design(shot, stride=stride)
        except ResponseError:
            continue
        error = observed - design @ vector
        power = float(observed @ observed)
        fault = float(error @ error)
        residual += fault
        signal += power
        count += int(observed.size)
        shots.append(shot.shot)
        per_shot[shot.shot] = 0.0 if power <= 0.0 else float(1.0 - fault / power)
    if count == 0:
        raise ResponseError("no held-out sample was available to score")
    return PredictionScore(
        shots=tuple(shots),
        residual_rms=float(math.sqrt(residual / count)),
        signal_rms=float(math.sqrt(signal / count)),
        sample_count=count,
        per_shot=per_shot,
    )


@dataclass(frozen=True)
class AxisScore:
    """One candidate sensitive-axis assignment and the residual it gives."""

    radial_families: tuple[str, ...]
    residual_rms: float
    variance_explained: float

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical JSON representation."""

        return {
            "radial_families": list(self.radial_families),
            "residual_rms": float(self.residual_rms),
            "variance_explained": float(self.variance_explained),
        }


def score_axis_assignment(
    geometry: Mapping[str, Any],
    probes: Sequence[Mapping[str, Any]],
    channels: Sequence[ProbeChannel],
    waveforms: Sequence[ShotWaveforms],
    candidates: Sequence[frozenset[str]],
    *,
    stride: int = 1,
    minimum_standoff: float = MINIMUM_STANDOFF,
) -> tuple[AxisScore, ...]:
    """Refit under each sensitive-axis assignment and rank the candidates.

    The registry cannot distinguish a radial probe from an axial one, so which
    families measure which component is decided here by prediction: an axis
    assigned the wrong way round predicts a field component the probe never saw
    and cannot be rescued by any multiplier.  The scores are returned in the
    order given so the margin between them is the caller's to read.
    """

    scores = []
    for candidate in candidates:
        model = ResponseModel.build(
            geometry,
            probes,
            channels,
            radial_families=frozenset(candidate),
            minimum_standoff=minimum_standoff,
        )
        fit = fit_response(waveforms, model, stride=stride)
        scores.append(
            AxisScore(
                radial_families=tuple(sorted(candidate)),
                residual_rms=fit.residual_rms,
                variance_explained=fit.variance_explained,
            )
        )
    return tuple(scores)


@dataclass(frozen=True)
class ShotEstimate:
    """One coil's multiplier as a single shot measured it."""

    shot: int
    family: str
    multiplier: float
    standard_error: float
    leverage: float
    correlation: float = 0.0

    @property
    def identified(self) -> bool:
        """Return whether this shot pinned this coil rather than merely allowed it.

        Three things have to hold and none of them implies the others.

        The coil must have moved the probes enough to be seen, which is what
        ``leverage`` measures -- the share of the predicted signal power this coil
        is responsible for.

        The solve must have pinned it, which is what the relative standard error
        measures.

        And the solve must have pinned it ON ITS OWN, which is what
        ``correlation`` measures.  The third test is not implied by the second and
        is the one that matters here: two coils carrying nearly the same waveform
        can be given equal and opposite multipliers of several thousand turns, and
        because it is their DIFFERENCE the probes constrain, each comes back with a
        small standard error and a large leverage.  Such an estimate is confidently
        wrong, survives both other tests, and is exactly the case where one
        parameter has absorbed another's residual.
        """

        if not math.isfinite(self.multiplier) or self.multiplier == 0.0:
            return False
        relative = self.standard_error / abs(self.multiplier)
        return (
            self.leverage >= MINIMUM_LEVERAGE
            and relative <= MAXIMUM_RELATIVE_ERROR
            and abs(self.correlation) <= MAXIMUM_CORRELATION
        )

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical JSON representation."""

        return {
            "correlation": float(self.correlation),
            "family": self.family,
            "identified": self.identified,
            "leverage": float(self.leverage),
            "multiplier": float(self.multiplier),
            "shot": self.shot,
            "standard_error": float(self.standard_error),
        }


def per_shot_estimates(
    waveforms: Iterable[ShotWaveforms],
    model: ResponseModel,
    *,
    stride: int = 1,
) -> tuple[ShotEstimate, ...]:
    """Fit each shot on its own, over only the coils that shot drove.

    A cohort fit reports one number per coil and hides which shots earned it.
    Fitting shot by shot answers a different and more useful question: which
    shots could see which coil, and do they agree.  Restricting each solve to the
    coils actually energised is what stops an unexcited coil from absorbing the
    residual of an excited one -- with thirteen columns and one driven circuit,
    the other twelve are free parameters fitting noise.
    """

    estimates: list[ShotEstimate] = []
    for shot in waveforms:
        driven = energised_families(shot, model.families)
        if not driven:
            continue
        try:
            reduced = model.select(driven)
            design, observed, _ = reduced.design(shot, stride=stride)
            system = NormalEquations.empty(len(driven))
            system.add(design, observed, shot.shot)
            solution, covariance, _ = system.solve()
        except ResponseError, CohortError, np.linalg.LinAlgError:
            continue
        power = (design * solution) ** 2
        total = float(power.sum())
        correlation = _correlation(covariance)
        for position, family in enumerate(driven):
            estimates.append(
                ShotEstimate(
                    shot=shot.shot,
                    family=family,
                    multiplier=float(solution[position]),
                    standard_error=float(
                        math.sqrt(max(covariance[position, position], 0.0))
                    ),
                    leverage=(
                        float(power[:, position].sum()) / total if total > 0.0 else 0.0
                    ),
                    correlation=float(correlation[position]),
                )
            )
    return tuple(estimates)


def _correlation(covariance: np.ndarray) -> np.ndarray:
    """Return each parameter's strongest correlation with any other parameter."""

    size = covariance.shape[0]
    if size < 2:
        return np.zeros(size)
    deviation = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    scale = np.outer(deviation, deviation)
    with np.errstate(divide="ignore", invalid="ignore"):
        matrix = np.where(scale > 0.0, covariance / scale, 0.0)
    np.fill_diagonal(matrix, 0.0)
    return np.max(np.abs(matrix), axis=1)


@dataclass(frozen=True)
class TurnDisposition:
    """What the cohort established about one coil's signed turn count."""

    family: str
    reports_ampere_turns: bool
    shots: tuple[int, ...]
    multiplier: float
    spread: float
    standard_error: float
    archive_multiplier: float | None = None

    @property
    def identified(self) -> bool:
        """Return whether any shot pinned this coil."""

        return bool(self.shots) and math.isfinite(self.multiplier)

    @property
    def nearest_integer(self) -> int | None:
        """Return the closest integer turn count, or nothing if unidentified."""

        if not self.identified:
            return None
        return int(round(self.multiplier))

    @property
    def integer_offset(self) -> float:
        """Return the distance from the fitted multiplier to that integer."""

        if not self.identified:
            return float("nan")
        return abs(self.multiplier - round(self.multiplier))

    @property
    def interval(self) -> tuple[float, float]:
        """Return the bound the cohort supports, widened by cross-shot spread.

        The interval carried into the evidence record is the wider of what the
        solve's own covariance says and what the shots' disagreement says, because
        a parameter that drifts between shots is not known to the precision one
        shot reports.
        """

        if not self.identified:
            return (float("nan"), float("nan"))
        half = max(self.spread, self.standard_error)
        return (self.multiplier - half, self.multiplier + half)

    @property
    def sign(self) -> int:
        """Return the polarity the cohort assigned to this coil's channel."""

        if not self.identified:
            return 0
        return 1 if self.multiplier > 0.0 else -1

    @property
    def half_width(self) -> float:
        """Return half the width of the interval the cohort supports."""

        if not self.identified:
            return float("nan")
        return max(self.spread, self.standard_error)

    @property
    def resolves_an_integer(self) -> bool:
        """Return whether the cohort names one turn count rather than a range.

        A turn count is a whole number, but a fit that lands near one has not
        established it unless the interval excludes its neighbours.  Requiring the
        interval to sit inside half a turn is what separates a coil the cohort
        counted from a coil it merely bounded -- and the difference is stark rather
        than marginal here, with the counted coils agreeing to a hundredth of a
        turn and the bounded ones scattering over five.
        """

        return self.identified and self.half_width < 0.5

    @property
    def agrees_with_archive(self) -> bool:
        """Return whether the fit lands on the archive's own multiplier."""

        if self.archive_multiplier is None or not self.identified:
            return False
        return abs(self.multiplier - self.archive_multiplier) <= 0.5

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical JSON representation."""

        return {
            "agrees_with_archive": self.agrees_with_archive,
            "archive_multiplier": self.archive_multiplier,
            "family": self.family,
            "identified": self.identified,
            "integer_offset": float(self.integer_offset),
            "interval": [float(value) for value in self.interval],
            "multiplier": float(self.multiplier),
            "nearest_integer": self.nearest_integer,
            "reports_ampere_turns": self.reports_ampere_turns,
            "resolves_an_integer": self.resolves_an_integer,
            "shot_count": len(self.shots),
            "shots": list(self.shots),
            "sign": self.sign,
            "spread": float(self.spread),
            "standard_error": float(self.standard_error),
        }


def aggregate_turns(
    estimates: Iterable[ShotEstimate],
    *,
    families: Sequence[str] | None = None,
    archive_multipliers: Mapping[str, float] | None = None,
) -> tuple[TurnDisposition, ...]:
    """Combine the shots that identified each coil into one signed turn count.

    Shots are combined by inverse-variance weighting, which is the right weight
    when the disagreement between them is measurement noise.  Whether it is stays
    visible: ``spread`` reports the shots' own scatter, and a scatter wider than
    the weighted error is a statement that something varies between shots that
    the model does not carry.
    """

    order = _order(families)
    ampere_turns = {drive.family: drive.reports_ampere_turns for drive in COIL_DRIVES}
    grouped: dict[str, list[ShotEstimate]] = {family: [] for family in order}
    for estimate in estimates:
        if estimate.family in grouped and estimate.identified:
            grouped[estimate.family].append(estimate)

    dispositions = []
    for family in order:
        rows = sorted(grouped[family], key=lambda row: row.shot)
        archive = (
            None if archive_multipliers is None else archive_multipliers.get(family)
        )
        if not rows:
            dispositions.append(
                TurnDisposition(
                    family=family,
                    reports_ampere_turns=bool(ampere_turns.get(family, False)),
                    shots=(),
                    multiplier=float("nan"),
                    spread=float("nan"),
                    standard_error=float("nan"),
                    archive_multiplier=archive,
                )
            )
            continue
        values = np.asarray([row.multiplier for row in rows], dtype=float)
        errors = np.asarray([max(row.standard_error, 1.0e-12) for row in rows])
        weight = 1.0 / errors**2
        dispositions.append(
            TurnDisposition(
                family=family,
                reports_ampere_turns=bool(ampere_turns.get(family, False)),
                shots=tuple(row.shot for row in rows),
                multiplier=float((values * weight).sum() / weight.sum()),
                spread=float(np.std(values)) if values.size > 1 else 0.0,
                standard_error=float(1.0 / math.sqrt(weight.sum())),
                archive_multiplier=archive,
            )
        )
    return tuple(dispositions)


def probe_residuals(
    waveforms: ShotWaveforms,
    model: ResponseModel,
    multipliers: Mapping[str, float],
    *,
    stride: int = 1,
) -> dict[str, tuple[float, float]]:
    """Return each probe's residual and signal amplitude for one shot.

    Reported per probe because the residual's shape over the array is what says
    whether what is left is noise or a field the model does not carry.
    """

    vector = np.asarray(
        [multipliers.get(family, 0.0) for family in model.families], dtype=float
    )
    quiet = baseline_window(waveforms)
    keep = model.admissible_probes(excited_families(waveforms, model.families))
    allowed = {
        target.channel for target, ok in zip(model.targets, keep, strict=True) if ok
    }
    drive = np.zeros((waveforms.time.size, len(model.families)), dtype=float)
    for column, family in enumerate(model.families):
        values = waveforms.drives.get(family)
        if values is not None:
            drive[:, column] = np.nan_to_num(values)
    index = {target.channel: row for row, target in enumerate(model.targets)}
    samples = np.flatnonzero(waveforms.sample_mask)[::stride]
    result: dict[str, tuple[float, float]] = {}
    for channel, signal in sorted(waveforms.probes.items()):
        row = index.get(channel)
        if row is None or channel not in allowed:
            continue
        finite = np.isfinite(signal)
        if not finite[quiet].any():
            continue
        take = samples[finite[samples]]
        if take.size == 0:
            continue
        observed = signal[take] - float(np.mean(signal[quiet & finite]))
        predicted = (drive[take] * model.response[row, :]) @ vector
        error = observed - predicted
        result[channel] = (
            float(np.sqrt(np.mean(error**2))),
            float(np.sqrt(np.mean(observed**2))),
        )
    return result
