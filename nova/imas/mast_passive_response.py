"""Identify what the vacuum transients say about the passive structures.

When a coil's current is switched off the poloidal field does not stop: induced
currents in the vessel, the coil cases and the centre column keep flowing and
decay on their own time constants.  The probes see that decay directly, and it is
the only measurement of the passive circuit in this archive -- the per-family
currents the store publishes are the reconstruction's own wall-model output, not
an instrument reading, so they can corroborate a fit but never ground one.

What a free decay can identify is bounded, and the bound is worth stating before
any fitting.  The probes measure a field, so they constrain the passive currents
only through the spatial patterns those currents produce.  Two groups whose
patterns are nearly parallel over the probe array contribute one measurable
quantity between them, however different their geometry looks in a drawing.  The
number of separable patterns is therefore a property of the diagnostic set, not
of the vessel, and it is measured here rather than assumed: the post-pulse probe
data is decomposed, the modes above the noise floor are counted, and only that
many independent parameters are allowed to be reported.

This is why the module's headline output is a count and a set of time constants
rather than a resistance per family.  A resistance per family would be sixteen
numbers fitted to a handful of separable observations, and the extra ones would
be reporting the residual.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import shapely

from nova.biot.polygon import polygon_greens
from nova.imas.mast_seed_parameters import passive_material
from nova.imas.mast_vacuum_cohort import EXCITATION_CURRENT, ShotWaveforms
from nova.imas.mast_vacuum_response import (
    ProbeTarget,
    ResponseError,
    baseline_window,
    coil_sections,
)

CASE_FAMILY = "coil_cases"
"""Registry passive family holding every poloidal-field coil's case."""

CASE_PROXIMITY = 0.5
"""Metres within which a case plate is assigned to a coil.

A coil case surrounds its own coil and nothing else, so nearest-coil assignment
is the mechanism rather than a heuristic.  The limit exists to refuse a plate that
is not near any coil instead of attaching it to the least distant one.
"""


class PassiveError(ValueError):
    """Raised when a passive grouping or decay cannot be identified."""


@dataclass(frozen=True)
class PassiveGroup:
    """One axisymmetric conductor group and the sections that make it up."""

    name: str
    family: str
    sections: tuple[np.ndarray, ...]

    @property
    def area(self) -> float:
        """Return the group's total poloidal cross-section area."""

        return float(
            sum(abs(shapely.Polygon(vertices).area) for vertices in self.sections)
        )

    @property
    def major_radius(self) -> float:
        """Return the area-weighted centroid radius of the group."""

        total = 0.0
        moment = 0.0
        for vertices in self.sections:
            polygon = shapely.Polygon(vertices)
            area = abs(polygon.area)
            total += area
            moment += area * float(polygon.centroid.x)
        if total <= 0.0:
            raise PassiveError(f"passive group {self.name!r} has no cross-section")
        return moment / total


def passive_sections(geometry: Mapping[str, Any]) -> dict[str, tuple[np.ndarray, ...]]:
    """Return each passive family's outlines as vertex arrays."""

    sections: dict[str, tuple[np.ndarray, ...]] = {}
    for family, wkb_hex in sorted(geometry["passive_components"].items()):
        outline = shapely.from_wkb(bytes.fromhex(wkb_hex))
        parts = getattr(outline, "geoms", None)
        polygons = (outline,) if parts is None else tuple(parts)
        sections[family] = tuple(
            np.asarray(polygon.exterior.coords, dtype=float)[:-1]
            for polygon in polygons
        )
    return sections


def case_side(family: str) -> str:
    """Return the coil set whose case encloses one winding pack.

    One case encloses a whole coil set, not each winding pack inside it, so the
    inner and outer packs of the same set share a case.  The store agrees: it
    publishes one case-current channel per set and none per pack.  Grouping by
    pack instead would cut one conductor into pieces that are not electrically
    separate and hand a fit parameters the machine does not have.
    """

    parts = family.split("_")
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[-1]}"
    return family


def case_grouping(geometry: Mapping[str, Any]) -> dict[str, tuple[np.ndarray, ...]]:
    """Split the coil-case family into one group per poloidal-field coil.

    The registry resolves the cases as a single family of thin plates with no
    statement about which coil each plate belongs to, which is one of the
    unresolved multi-section groupings.  Geometry settles it without a fit: a case
    encloses its own coil, so each plate joins the active component it is nearest
    to.  A plate further than :data:`CASE_PROXIMITY` from every coil is left out
    and reported, because attaching it to the least distant coil would invent a
    connection the geometry does not support.
    """

    plates = passive_sections(geometry).get(CASE_FAMILY)
    if not plates:
        raise PassiveError(f"registry carries no {CASE_FAMILY!r} family")
    coils: dict[str, list[shapely.Polygon]] = {}
    for family, parts in coil_sections(geometry).items():
        coils.setdefault(case_side(family), []).extend(
            shapely.Polygon(vertices) for vertices in parts
        )
    grouped: dict[str, list[np.ndarray]] = {}
    orphans = 0
    for vertices in plates:
        plate = shapely.Polygon(vertices)
        distances = {
            family: min(plate.distance(polygon) for polygon in polygons)
            for family, polygons in coils.items()
        }
        nearest = min(distances, key=lambda key: distances[key])
        if distances[nearest] > CASE_PROXIMITY:
            orphans += 1
            continue
        grouped.setdefault(f"{CASE_FAMILY}_{nearest}", []).append(vertices)
    if orphans:
        grouped.setdefault(f"{CASE_FAMILY}_unassigned", [])
    return {name: tuple(rows) for name, rows in sorted(grouped.items())}


def passive_groups(
    geometry: Mapping[str, Any],
    *,
    split_cases: bool = True,
) -> tuple[PassiveGroup, ...]:
    """Return the conductor groups a passive fit may treat as independent."""

    groups: list[PassiveGroup] = []
    for family, parts in sorted(passive_sections(geometry).items()):
        if family == CASE_FAMILY and split_cases:
            for name, rows in case_grouping(geometry).items():
                if rows:
                    groups.append(PassiveGroup(name=name, family=family, sections=rows))
            continue
        groups.append(PassiveGroup(name=family, family=family, sections=parts))
    return tuple(groups)


def passive_coupling(
    groups: Sequence[PassiveGroup],
    targets: Sequence[ProbeTarget],
) -> np.ndarray:
    """Field each group produces at each probe, per ampere [T/A].

    Current is spread over the group's sections in proportion to area, which is
    what a uniform current density in a single connected conductor does.  A group
    whose sections are not in fact connected in parallel would need its own
    column per section, and whether the probes can tell the difference is what
    :func:`separable_modes` answers.
    """

    target_r = np.asarray([target.r for target in targets], dtype=float)
    target_z = np.asarray([target.z for target in targets], dtype=float)
    cosine = np.asarray([target.radial_cosine for target in targets], dtype=float)
    sine = np.asarray([target.axial_sine for target in targets], dtype=float)
    coupling = np.zeros((len(targets), len(groups)), dtype=float)
    for column, group in enumerate(groups):
        areas = np.asarray(
            [abs(shapely.Polygon(v).area) for v in group.sections], dtype=float
        )
        total = float(areas.sum())
        if total <= 0.0:
            raise PassiveError(f"passive group {group.name!r} has no cross-section")
        for vertices, area in zip(group.sections, areas, strict=True):
            _, radial, axial = polygon_greens(target_r, target_z, vertices)
            coupling[:, column] += (area / total) * (cosine * radial + sine * axial)
    return coupling


def group_self_inductance(group: PassiveGroup, *, samples: int = 96) -> float:
    """Return the group's ring self-inductance [H], averaged over its section.

    A filament's self-inductance diverges, so the flux is taken as the mean over
    the conductor's own cross-section -- the quantity a uniformly distributed
    current actually links.  The mean is taken on a quasi-random point set inside
    the sections rather than a grid, so a long thin plate is sampled along its
    length instead of by whatever a grid's aspect ratio happens to allow.
    """

    points: list[np.ndarray] = []
    weights: list[float] = []
    for vertices in group.sections:
        polygon = shapely.Polygon(vertices)
        area = abs(polygon.area)
        if area <= 0.0:
            continue
        r0, z0, r1, z1 = polygon.bounds
        generator = np.random.default_rng(0)
        inside: list[tuple[float, float]] = []
        while len(inside) < samples:
            batch = generator.uniform([r0, z0], [r1, z1], size=(4 * samples, 2))
            for point in batch:
                if polygon.covers(shapely.Point(point)):
                    inside.append((float(point[0]), float(point[1])))
                if len(inside) >= samples:
                    break
        points.append(np.asarray(inside, dtype=float))
        weights.append(area)
    if not points:
        raise PassiveError(f"passive group {group.name!r} has no interior")
    total_area = float(sum(weights))
    flux = 0.0
    for vertices in group.sections:
        for block, weight in zip(points, weights, strict=True):
            psi, _, _ = polygon_greens(block[:, 0], block[:, 1], vertices)
            flux += (weight / total_area) * float(np.mean(psi))
    return flux


def group_resistance(group: PassiveGroup) -> float | None:
    """Return the group's nominal ring resistance [ohm], or nothing if unsourced."""

    material = passive_material(group.family)
    if material is None:
        return None
    return material.loop_resistance(group.area, group.major_radius)


def decay_window(
    waveforms: ShotWaveforms,
    *,
    settle: float = 2.0e-3,
    span: float = 0.06,
) -> np.ndarray:
    """Mark the samples after the last deliberate excitation stopped.

    The window opens ``settle`` after the final switch-off, which lets the supply
    transient and the fastest structures pass so what remains decays rather than
    rings, and closes ``span`` later, before the slowest mode has fallen into the
    probe noise.  Both are window boundaries, not fitted quantities: the time
    constants come out of the data inside the window.

    Switch-off is judged against the deliberate-excitation threshold rather than
    the pickup floor.  The vertical-control coils hold a few hundred amperes
    between pulses, so a test at the pickup floor finds the machine still driven
    at the end of the record and opens no window at all.
    """

    driven = np.zeros(waveforms.time.shape, dtype=bool)
    for values in waveforms.drives.values():
        driven |= np.abs(np.nan_to_num(values)) >= EXCITATION_CURRENT
    indices = np.flatnonzero(driven)
    if indices.size == 0:
        raise PassiveError(f"shot {waveforms.shot} drove no coil")
    stop = float(waveforms.time[indices[-1]])
    return (
        (waveforms.time >= stop + settle)
        & (waveforms.time <= stop + settle + span)
        & waveforms.sample_mask
    )


@dataclass(frozen=True)
class DecayMode:
    """One separable spatial pattern in a free decay, and how fast it decayed."""

    index: int
    singular_value: float
    signal_fraction: float
    time_constant: float
    fit_quality: float
    pattern: np.ndarray

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical JSON representation."""

        return {
            "fit_quality": float(self.fit_quality),
            "index": self.index,
            "signal_fraction": float(self.signal_fraction),
            "singular_value": float(self.singular_value),
            "time_constant": float(self.time_constant),
        }


@dataclass(frozen=True)
class DecaySpectrum:
    """The separable content of one shot's free decay."""

    shot: int
    channels: tuple[str, ...]
    modes: tuple[DecayMode, ...]
    noise_floor: float
    sample_count: int

    @property
    def separable_count(self) -> int:
        """Return how many patterns rose above the noise floor."""

        return len(self.modes)

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical JSON representation."""

        return {
            "channels": list(self.channels),
            "modes": [row.as_dict() for row in self.modes],
            "noise_floor": float(self.noise_floor),
            "sample_count": self.sample_count,
            "separable_count": self.separable_count,
            "shot": self.shot,
        }


def separable_modes(
    waveforms: ShotWaveforms,
    targets: Sequence[ProbeTarget],
    *,
    modes: int = 6,
    noise_multiple: float = 3.0,
) -> DecaySpectrum:
    """Decompose one shot's free decay and count what the probes can separate.

    The noise floor is measured on the same channels in the shot's own
    pre-excitation window, so it is this shot's noise and not a nominal figure.  A
    mode is kept when its amplitude exceeds that floor by ``noise_multiple``,
    which is what makes the reported count a measurement.
    """

    window = decay_window(waveforms)
    quiet = baseline_window(waveforms)
    if int(window.sum()) < 8:
        raise PassiveError(f"shot {waveforms.shot} has too short a decay window")
    poses = {target.channel for target in targets}
    usable: list[tuple[str, np.ndarray, np.ndarray]] = []
    for channel, signal in sorted(waveforms.probes.items()):
        if channel not in poses:
            continue
        finite = np.isfinite(signal)
        covered = float(np.count_nonzero(finite & window)) / max(int(window.sum()), 1)
        if covered < 0.5 or not finite[quiet].any():
            continue
        usable.append((channel, signal, finite))
    if len(usable) < 4:
        raise PassiveError(f"shot {waveforms.shot} has too few clean probes")

    # The store pads a probe record with absent samples once its acquisition ends,
    # and the ends differ between channels.  Narrowing the window to where every
    # admitted channel is present keeps the decomposition on one common time base,
    # which is what makes a spatial pattern across the array mean anything.
    shared = window.copy()
    for _, _, finite in usable:
        shared &= finite
    if int(shared.sum()) < 8:
        raise PassiveError(
            f"shot {waveforms.shot} has {int(shared.sum())} samples common to "
            "every probe in its decay window"
        )
    window = shared

    channels: list[str] = []
    rows: list[np.ndarray] = []
    floors: list[float] = []
    for channel, signal, finite in usable:
        offset = float(np.mean(signal[quiet & finite]))
        channels.append(channel)
        rows.append(signal[window] - offset)
        floors.append(float(np.std(signal[quiet & finite])))
    data = np.vstack(rows)
    floor = float(np.median(floors))
    left, values, right = np.linalg.svd(data, full_matrices=False)
    time = waveforms.time[window]
    total = float((values**2).sum())
    kept: list[DecayMode] = []
    for index in range(min(modes, values.size)):
        amplitude = values[index] / math.sqrt(data.shape[1])
        if amplitude < noise_multiple * floor:
            break
        constant, quality = _exponential_time_constant(
            time, right[index] * values[index]
        )
        kept.append(
            DecayMode(
                index=index,
                singular_value=float(values[index]),
                signal_fraction=float(values[index] ** 2 / total) if total > 0 else 0.0,
                time_constant=constant,
                fit_quality=quality,
                pattern=left[:, index].copy(),
            )
        )
    return DecaySpectrum(
        shot=waveforms.shot,
        channels=tuple(channels),
        modes=tuple(kept),
        noise_floor=floor,
        sample_count=int(window.sum()),
    )


def _exponential_time_constant(
    time: np.ndarray,
    values: np.ndarray,
) -> tuple[float, float]:
    """Fit a single exponential to a temporal mode and report how well it fitted.

    The fit is a straight line through the log of the magnitude, which is exact
    for a single decaying mode and visibly poor for a superposition -- so the
    returned quality is what says whether calling this one time constant was
    legitimate.  Samples that have fallen into the noise are dropped rather than
    logged, because the log of noise is a slope of its own.
    """

    magnitude = np.abs(values)
    scale = float(magnitude.max()) if magnitude.size else 0.0
    if scale <= 0.0:
        return (float("nan"), 0.0)
    keep = magnitude > 0.05 * scale
    if int(keep.sum()) < 4:
        return (float("nan"), 0.0)
    span = time[keep] - float(time[keep][0])
    logged = np.log(magnitude[keep] / scale)
    slope, intercept = np.polyfit(span, logged, 1)
    if slope >= 0.0:
        return (float("inf"), 0.0)
    predicted = slope * span + intercept
    residual = float(np.sum((logged - predicted) ** 2))
    spread = float(np.sum((logged - logged.mean()) ** 2))
    quality = 0.0 if spread <= 0.0 else float(1.0 - residual / spread)
    return (float(-1.0 / slope), quality)


@dataclass(frozen=True)
class GroupAttribution:
    """Which conductor group a decay pattern is consistent with, and how well."""

    mode_index: int
    time_constant: float
    best_group: str
    alignment: float
    runner_up: str
    runner_up_alignment: float

    @property
    def decisive(self) -> bool:
        """Return whether one group explains the pattern and the next does not.

        Both tests matter.  The winner has to explain the pattern at all, and it
        has to explain it better than the next candidate by a margin -- two groups
        with nearly parallel patterns produce a high alignment each and identify
        neither.
        """

        return (
            self.alignment >= 0.9 and self.alignment - self.runner_up_alignment >= 0.1
        )

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical JSON representation."""

        return {
            "alignment": float(self.alignment),
            "best_group": self.best_group,
            "decisive": self.decisive,
            "mode_index": self.mode_index,
            "runner_up": self.runner_up,
            "runner_up_alignment": float(self.runner_up_alignment),
            "time_constant": float(self.time_constant),
        }


def attribute_modes(
    spectrum: DecaySpectrum,
    groups: Sequence[PassiveGroup],
    coupling: np.ndarray,
    targets: Sequence[ProbeTarget],
) -> tuple[GroupAttribution, ...]:
    """Match each measured decay pattern against every group's field pattern.

    Alignment is the cosine between the measured spatial pattern and the pattern
    a unit current in one group would produce, so it is scale-free and answers
    only the question geometry can answer: does the field look like it came from
    there.  It deliberately says nothing about how much current flowed, because a
    free decay's amplitude and the group's resistance are not separable from the
    probes alone.
    """

    index = {target.channel: row for row, target in enumerate(targets)}
    rows = [index[channel] for channel in spectrum.channels if channel in index]
    if len(rows) < 4:
        raise PassiveError(f"shot {spectrum.shot} shares too few probes with the model")
    patterns = coupling[rows, :]
    norms = np.linalg.norm(patterns, axis=0)
    attributions = []
    for mode in spectrum.modes:
        observed = mode.pattern[: len(rows)]
        scale = float(np.linalg.norm(observed))
        if scale <= 0.0:
            continue
        alignment = np.zeros(len(groups))
        for column in range(len(groups)):
            if norms[column] <= 0.0:
                continue
            alignment[column] = abs(
                float(observed @ patterns[:, column]) / (scale * norms[column])
            )
        order = np.argsort(-alignment)
        attributions.append(
            GroupAttribution(
                mode_index=mode.index,
                time_constant=mode.time_constant,
                best_group=groups[order[0]].name,
                alignment=float(alignment[order[0]]),
                runner_up=groups[order[1]].name if len(groups) > 1 else "",
                runner_up_alignment=(
                    float(alignment[order[1]]) if len(groups) > 1 else 0.0
                ),
            )
        )
    return tuple(attributions)


def effective_resistance(time_constant: float, inductance: float) -> float:
    """Return the resistance a measured decay implies for a known inductance."""

    if not math.isfinite(time_constant) or time_constant <= 0.0:
        raise PassiveError("a resistance needs a finite positive time constant")
    if not math.isfinite(inductance) or inductance <= 0.0:
        raise PassiveError("a resistance needs a finite positive inductance")
    return inductance / time_constant


def survey_decays(
    waveforms: Iterable[ShotWaveforms],
    targets: Sequence[ProbeTarget],
    groups: Sequence[PassiveGroup],
    coupling: np.ndarray,
) -> tuple[tuple[DecaySpectrum, tuple[GroupAttribution, ...]], ...]:
    """Decompose and attribute every shot's decay that admits one."""

    results = []
    for shot in waveforms:
        try:
            spectrum = separable_modes(shot, targets)
            attributions = attribute_modes(spectrum, groups, coupling, targets)
        except PassiveError, ResponseError, np.linalg.LinAlgError:
            continue
        results.append((spectrum, attributions))
    return tuple(results)
