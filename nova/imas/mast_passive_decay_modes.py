"""Identify passive resistance from free decays, with the inductance in the loop.

A decay pattern is a **mode**: a specific mixture of every circuit at once, set
by the inductance and the resistance together.  So scoring an observed pattern
against one conductor group's field -- asking which conductor it came from -- is
asking a question the data cannot answer.  Two groups whose fields look alike
score alike and identify neither, the mixture that actually decayed is
unrepresentable as any single group, and the amplitudes are thrown away.

This module fits the mixture.  With ``L`` exact from geometry, a candidate
resistance fixes the whole mode set at once through
``R v = (1/tau) L v``: every mode's decay time AND its probe pattern come out of
the same eigenproblem, so the two are no longer independent things to match.  The
prediction for one shot's decay is then

    ``B(t) = sum_k a_k (C v_k) exp(-t / tau_k)``

with ``C`` the geometry's probe coupling and ``a_k`` the only free numbers per
shot -- the amplitudes the switch-off happened to leave behind.  Because the
amplitudes are free and linear, they are projected out exactly, and what is
actually being fitted is the resistance model shared by every shot.

**Turn counts do not enter.**  The initial condition is fitted per shot rather
than propagated from the drive, so nothing here depends on how many turns a coil
has or on what its supply did.  That is deliberate: the excitation sets the
amplitudes and the passive circuit sets everything else, and only the second is
under test.

**Free parameters are the resistivity of a material, not the resistance of a
circuit.**  Fifty-seven circuit resistances fitted to a handful of separable
patterns would report the residual.  Each circuit's resistance is its own
measured section geometry times its material's resistivity, so the unknowns are a
few resistivity multipliers -- one per class of conductor, declared before
fitting -- and the geometry does the rest.  Whether a multiplier is identified at
all is measured by profiling it, not assumed from the fit converging.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize

from nova.imas.mast_block_scale import BRACKETED, MEASURED
from nova.imas.mast_passive_inductance import Linkage, PassiveTurn
from nova.imas.mast_passive_response import PassiveError, decay_window
from nova.imas.mast_vacuum_cohort import EXCITATION_CURRENT, ShotWaveforms
from nova.imas.mast_vacuum_response import ProbeTarget, baseline_window

RESISTIVITY_CLASSES = {
    "coil_cases": "coil_case",
    "incon": "centre_column",
    "rodgr": "centre_rod",
}
"""Which resistivity multiplier each registry family draws on.

Declared before any fit, by what the conductor *is* rather than by what improves
the residual.  Families absent from this mapping fall to ``vessel_shell``: the
welded vacuum vessel, its end crowns, the divertor arms and the support rings are
one fabrication in one material, and nothing published distinguishes their
grades.  The two named exceptions are the ones a source does distinguish -- the
Inconel centre tube -- and the one no source resolves at all, the rod and
ground-return family, whose material may be copper rather than vessel steel and
which therefore has to carry its own multiplier or contaminate the vessel's.
"""

DEFAULT_RESISTIVITY_CLASS = "vessel_shell"
"""The class a family with no distinguishing source belongs to."""

SLOWEST_RESOLVABLE_TIME = 0.5
"""Seconds above which a mode is indistinguishable from a standing offset.

A mode this slow changes by about a tenth across the decay window, and the
window's own offset was already removed against the pre-excitation baseline, so
admitting it would let the fit absorb a baseline error as a physical circuit.
"""

FASTEST_RESOLVABLE_TIME = 5.0e-4
"""Seconds below which a mode has died before the window opens.

The window opens two milliseconds after switch-off to let the supply transient
pass, by which time a mode this fast has fallen by more than 98 per cent.  Its
amplitude is then whatever the noise in the first samples happens to be.
"""

RESOLVED_MODE_COUNT = 3
"""How many modes the reconstruction is allowed to carry.

Three is how many patterns a single free decay puts above the probe noise floor
on the cleanest shots this archive holds, and it is declared before fitting so the
fit cannot buy residual by adding basis functions.
:func:`mode_count_sensitivity` reports what changes at two and at four, which is
what shows whether three was the binding choice.
"""

MULTIPLIER_BOUNDS = (0.2, 20.0)
"""Bounds on a resistivity multiplier.

Wide, and deliberately not the material interval.  An axisymmetric ring standing
in for a welded three-dimensional shell has a longer current path and cut-outs
the ring model does not carry, so its effective resistance is expected ABOVE the
bulk value; bounding the fit at the material interval would hide that by
construction.  The material interval is applied afterwards, as the promotion
test, where a value outside it has to be explained rather than prevented.
"""

SETTLE_DRIVE_FRACTION = 0.02
"""Fraction of the shot's peak drive still allowed inside the decay window.

The vertical-control coils hold a few hundred amperes between pulses, so a test
at zero opens no window at all.  What must be excluded is a drive still *moving*,
because a ramp inside the window injects a term the free-decay model has no
place for.
"""


class DecayModeError(PassiveError):
    """Raised when a transient or a resistance model cannot be identified."""


@dataclass(frozen=True)
class DecayTransient:
    """One shot's free decay, offset-removed and whitened by its own noise.

    ``signal`` is ``(channels, samples)`` in tesla with each channel's
    pre-excitation offset subtracted; ``noise`` is that channel's scatter in the
    same window, so the residual can be reported in units of the noise the shot
    itself measured rather than a nominal figure.

    ``scale_dispositions`` carries the acquisition-range warrant of every channel
    read.  It is part of the fit's provenance: a channel read unscaled because
    nothing measured its setting and one read unscaled because its setting was
    measured to be the ordinary one are different statements about the same
    number.
    """

    shot: int
    channels: tuple[str, ...]
    time: np.ndarray
    signal: np.ndarray
    noise: np.ndarray
    excitation_family: str
    driven_families: tuple[str, ...]
    peak_drive: float
    residual_drive: float
    scale_dispositions: Mapping[str, str] = field(default_factory=dict)
    refused_channels: tuple[str, ...] = ()

    @property
    def sample_count(self) -> int:
        """Return how many samples the decay window admitted."""

        return int(self.time.size)

    @property
    def signal_to_noise(self) -> float:
        """Return the whitened amplitude of the transient."""

        return float(np.sqrt(np.mean((self.signal / self.noise[:, None]) ** 2)))

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical JSON representation."""

        return {
            "channel_count": len(self.channels),
            "driven_families": list(self.driven_families),
            "excitation_family": self.excitation_family,
            "noise_median": float(np.median(self.noise)),
            "peak_drive": self.peak_drive,
            "refused_channels": list(self.refused_channels),
            "residual_drive": self.residual_drive,
            "sample_count": self.sample_count,
            "scaled_channels": sorted(
                channel
                for channel, disposition in self.scale_dispositions.items()
                if disposition in (MEASURED, BRACKETED)
            ),
            "shot": self.shot,
            "signal_to_noise": self.signal_to_noise,
            "window_span": float(self.time[-1] - self.time[0])
            if self.time.size
            else 0.0,
        }


def read_transient(
    waveforms: ShotWaveforms,
    targets: Sequence[ProbeTarget],
    *,
    excitation_family: str,
    refused_channels: Iterable[str] = (),
    minimum_channels: int = 8,
) -> DecayTransient:
    """Extract one shot's free decay on the channels the model can predict.

    The window, the offset and the noise all come from this shot: the offset and
    scatter are measured in its own pre-excitation baseline, and the window is
    narrowed to the samples every admitted channel is present for, because a
    spatial pattern across the array only means something on a common time base.
    """

    window = decay_window(waveforms)
    quiet = baseline_window(waveforms)
    if int(window.sum()) < 8:
        raise DecayModeError(f"shot {waveforms.shot} has too short a decay window")
    refused = set(refused_channels)
    posed = {target.channel for target in targets}

    admitted: list[tuple[str, np.ndarray, np.ndarray]] = []
    for channel, signal in sorted(waveforms.probes.items()):
        if channel not in posed or channel in refused:
            continue
        finite = np.isfinite(signal)
        covered = float(np.count_nonzero(finite & window)) / max(int(window.sum()), 1)
        if covered < 0.5 or int(np.count_nonzero(finite & quiet)) < 8:
            continue
        admitted.append((channel, signal, finite))
    if len(admitted) < minimum_channels:
        raise DecayModeError(
            f"shot {waveforms.shot} admits {len(admitted)} channels, "
            f"fewer than the {minimum_channels} a pattern needs"
        )

    shared = window.copy()
    for _, _, finite in admitted:
        shared &= finite
    if int(shared.sum()) < 8:
        raise DecayModeError(
            f"shot {waveforms.shot} has {int(shared.sum())} samples common to "
            "every admitted channel"
        )

    channels: list[str] = []
    rows: list[np.ndarray] = []
    floors: list[float] = []
    for channel, signal, finite in admitted:
        offset = float(np.mean(signal[quiet & finite]))
        scatter = float(np.std(signal[quiet & finite]))
        if not math.isfinite(scatter) or scatter <= 0.0:
            continue
        channels.append(channel)
        rows.append(signal[shared] - offset)
        floors.append(scatter)
    if len(channels) < minimum_channels:
        raise DecayModeError(
            f"shot {waveforms.shot} measures a noise floor on only "
            f"{len(channels)} channels"
        )

    time = waveforms.time[shared]
    peak, residual = _drive_activity(waveforms, shared)
    return DecayTransient(
        shot=waveforms.shot,
        channels=tuple(channels),
        time=time - float(time[0]),
        signal=np.vstack(rows),
        noise=np.asarray(floors, dtype=float),
        excitation_family=excitation_family,
        driven_families=_driven_families(waveforms),
        peak_drive=peak,
        residual_drive=residual,
        scale_dispositions={
            row.channel: row.disposition for row in waveforms.scale_corrections
        },
        refused_channels=tuple(sorted(refused & posed)),
    )


def _driven_families(waveforms: ShotWaveforms) -> tuple[str, ...]:
    """Return the coil families this shot drove above the excitation threshold."""

    return tuple(
        sorted(
            family
            for family, values in waveforms.drives.items()
            if np.nanmax(np.abs(values)) >= EXCITATION_CURRENT
        )
    )


def _drive_activity(
    waveforms: ShotWaveforms, window: np.ndarray
) -> tuple[float, float]:
    """Return the shot's peak drive and how much of it still moves in the window.

    The second number is the swing of the largest drive across the decay window,
    relative to that drive's own peak.  A free-decay model has no place for a
    drive that is still ramping, so this is the quantity that disqualifies a
    window rather than the standing current, which merely offsets it.
    """

    peak = 0.0
    swing = 0.0
    for values in waveforms.drives.values():
        finite = np.nan_to_num(values)
        peak = max(peak, float(np.max(np.abs(finite))))
        inside = finite[window]
        if inside.size:
            swing = max(swing, float(np.max(inside) - np.min(inside)))
    return (peak, swing / peak if peak > 0.0 else 0.0)


def resistivity_class(family: str) -> str:
    """Return which multiplier a registry family's resistance is scaled by."""

    return RESISTIVITY_CLASSES.get(family, DEFAULT_RESISTIVITY_CLASS)


def class_names(turns: Sequence[PassiveTurn]) -> tuple[str, ...]:
    """Return the resistivity classes the carried circuits actually populate."""

    return tuple(sorted({resistivity_class(turn.family) for turn in turns}))


def circuit_multipliers(
    turns: Sequence[PassiveTurn],
    names: Sequence[str],
    values: np.ndarray,
) -> np.ndarray:
    """Expand per-class multipliers onto one multiplier per circuit."""

    index = {name: position for position, name in enumerate(names)}
    return np.asarray(
        [values[index[resistivity_class(turn.family)]] for turn in turns], dtype=float
    )


@dataclass(frozen=True)
class ModeSet:
    """The decay modes one candidate resistance model predicts.

    ``tau`` are decay times [s] slowest first and ``signature`` ``(probes,
    modes)`` the field each mode produces per unit mode amplitude [T].  Both come
    from the same eigenproblem, which is what stops a fit trading a time constant
    against a pattern.
    """

    tau: np.ndarray
    vectors: np.ndarray
    signature: np.ndarray

    @property
    def mode_count(self) -> int:
        """Return how many modes the set carries."""

        return int(self.tau.size)


def mode_set(
    linkage: Linkage,
    resistance: np.ndarray,
    coupling: np.ndarray,
    *,
    multipliers: np.ndarray | None = None,
) -> ModeSet:
    """Solve ``R v = (1/tau) L v`` and give every mode its probe signature."""

    diagonal = np.asarray(resistance, dtype=float)
    if multipliers is not None:
        multipliers = np.asarray(multipliers, dtype=float)
        if multipliers.shape != diagonal.shape:
            raise DecayModeError(
                f"multiplier shape {multipliers.shape} does not match "
                f"{diagonal.shape} circuits"
            )
        if not np.all(np.isfinite(multipliers)) or np.any(multipliers <= 0.0):
            raise DecayModeError("resistivity multipliers must be finite and positive")
        diagonal = diagonal * multipliers
    rate, vectors = eigh(np.diag(diagonal), linkage.matrix)
    if np.any(rate <= 0.0):
        raise DecayModeError("the resistance model produced a non-decaying mode")
    order = np.argsort(rate)
    rate = rate[order]
    vectors = vectors[:, order]
    return ModeSet(
        tau=1.0 / rate,
        vectors=vectors,
        signature=coupling @ vectors,
    )


def resolvable_modes(
    modes: ModeSet,
    *,
    fastest: float = FASTEST_RESOLVABLE_TIME,
    slowest: float = SLOWEST_RESOLVABLE_TIME,
) -> np.ndarray:
    """Return the mode indices whose decay time the window can resolve."""

    return np.flatnonzero((modes.tau >= fastest) & (modes.tau <= slowest))


def visible_modes(
    modes: ModeSet,
    transient: DecayTransient,
    rows: np.ndarray,
    *,
    count: int = RESOLVED_MODE_COUNT,
    fastest: float = FASTEST_RESOLVABLE_TIME,
    slowest: float = SLOWEST_RESOLVABLE_TIME,
) -> np.ndarray:
    """Return the modes this shot's channels can actually see, strongest first.

    A mode is ranked by how long it lives times how strongly it shows up in
    whitened probe units, which is the same ranking a reduced circuit model uses
    to decide whose history a per-slice fit cannot absorb.  Ranking on the
    channels of THIS shot matters: an array that lost half its channels sees a
    different set of modes from the full one.
    """

    admitted = resolvable_modes(modes, fastest=fastest, slowest=slowest)
    if admitted.size == 0:
        raise DecayModeError("no predicted mode decays inside the resolvable band")
    whitened = modes.signature[np.ix_(rows, admitted)] / transient.noise[:, None]
    ranking = modes.tau[admitted] * np.linalg.norm(whitened, axis=0)
    order = admitted[np.argsort(-ranking)]
    return np.sort(order[: min(count, order.size)])


@dataclass(frozen=True)
class Reconstruction:
    """How well one candidate mode set explains one shot's decay."""

    shot: int
    mode_indices: tuple[int, ...]
    tau: tuple[float, ...]
    amplitudes: tuple[float, ...]
    whitened_residual: float
    whitened_signal: float
    condition: float

    @property
    def variance_explained(self) -> float:
        """Return the whitened variance the mode set accounts for."""

        if self.whitened_signal <= 0.0:
            return 0.0
        return float(1.0 - (self.whitened_residual / self.whitened_signal) ** 2)

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical JSON representation."""

        return {
            "amplitudes": list(self.amplitudes),
            "condition": self.condition,
            "mode_indices": list(self.mode_indices),
            "shot": self.shot,
            "tau": list(self.tau),
            "variance_explained": self.variance_explained,
            "whitened_residual": self.whitened_residual,
            "whitened_signal": self.whitened_signal,
        }


def channel_rows(transient: DecayTransient, channels: Sequence[str]) -> np.ndarray:
    """Return where each of a transient's channels sits in the coupling matrix."""

    index = {channel: row for row, channel in enumerate(channels)}
    missing = [channel for channel in transient.channels if channel not in index]
    if missing:
        raise DecayModeError(
            f"shot {transient.shot} carries unposed channels {missing[:4]}"
        )
    return np.asarray([index[channel] for channel in transient.channels], dtype=int)


def reconstruct(
    transient: DecayTransient,
    modes: ModeSet,
    rows: np.ndarray,
    selection: np.ndarray,
) -> Reconstruction:
    """Fit the free mode amplitudes and report what is left over.

    The amplitudes enter linearly, so they are solved exactly rather than
    searched.  Everything the resistance model controls -- the decay times and the
    spatial patterns -- is held fixed here, which is what makes the leftover a
    statement about the resistance rather than about the fit.
    """

    noise = transient.noise[:, None]
    observed = (transient.signal / noise).ravel()
    envelope = np.exp(-transient.time[None, :] / modes.tau[selection][:, None])
    patterns = modes.signature[np.ix_(rows, selection)] / noise
    design = np.stack(
        [
            np.outer(patterns[:, column], envelope[column]).ravel()
            for column in range(selection.size)
        ],
        axis=1,
    )
    amplitudes, *_ = np.linalg.lstsq(design, observed, rcond=None)
    residual = observed - design @ amplitudes
    scale = np.sqrt(observed.size)
    return Reconstruction(
        shot=transient.shot,
        mode_indices=tuple(int(index) for index in selection),
        tau=tuple(float(value) for value in modes.tau[selection]),
        amplitudes=tuple(float(value) for value in amplitudes),
        whitened_residual=float(np.linalg.norm(residual) / scale),
        whitened_signal=float(np.linalg.norm(observed) / scale),
        condition=float(np.linalg.cond(design)),
    )


@dataclass(frozen=True)
class MisfitReport:
    """The pooled misfit of one resistance model over a set of transients."""

    misfit: float
    reconstructions: tuple[Reconstruction, ...]

    @property
    def variance_explained(self) -> float:
        """Return the pooled whitened variance the model accounts for."""

        residual = sum(row.whitened_residual**2 for row in self.reconstructions)
        signal = sum(row.whitened_signal**2 for row in self.reconstructions)
        return 0.0 if signal <= 0.0 else float(1.0 - residual / signal)


def decay_misfit(
    transients: Sequence[DecayTransient],
    linkage: Linkage,
    resistance: np.ndarray,
    coupling: np.ndarray,
    channels: Sequence[str],
    turns: Sequence[PassiveTurn],
    names: Sequence[str],
    values: np.ndarray,
    *,
    mode_count: int = RESOLVED_MODE_COUNT,
) -> MisfitReport:
    """Return the pooled whitened misfit of one candidate resistivity model.

    Every transient is weighted alike, in units of its own measured noise, so a
    loud shot does not outvote a quiet one and the pooled number reads as a
    multiple of the noise floor.
    """

    modes = mode_set(
        linkage,
        resistance,
        coupling,
        multipliers=circuit_multipliers(turns, names, values),
    )
    rows_by_shot = {
        transient.shot: channel_rows(transient, channels) for transient in transients
    }
    reconstructions = []
    for transient in transients:
        rows = rows_by_shot[transient.shot]
        selection = visible_modes(modes, transient, rows, count=mode_count)
        reconstructions.append(reconstruct(transient, modes, rows, selection))
    if not reconstructions:
        raise DecayModeError("no transient to fit")
    misfit = math.sqrt(
        float(np.mean([row.whitened_residual**2 for row in reconstructions]))
    )
    return MisfitReport(misfit=misfit, reconstructions=tuple(reconstructions))


@dataclass(frozen=True)
class ResistivityFit:
    """A fitted resistivity model, and everything needed to judge it."""

    names: tuple[str, ...]
    multipliers: tuple[float, ...]
    misfit: float
    nominal_misfit: float
    variance_explained: float
    iterations: int
    converged: bool
    reconstructions: tuple[Reconstruction, ...] = ()

    @property
    def improvement(self) -> float:
        """Return the fractional misfit reduction against the nominal model."""

        if self.nominal_misfit <= 0.0:
            return 0.0
        return float(1.0 - self.misfit / self.nominal_misfit)

    def multiplier(self, name: str) -> float:
        """Return one class's fitted multiplier."""

        return self.multipliers[self.names.index(name)]

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical JSON representation."""

        return {
            "converged": self.converged,
            "improvement": self.improvement,
            "iterations": self.iterations,
            "misfit": self.misfit,
            "multipliers": dict(zip(self.names, self.multipliers, strict=True)),
            "names": list(self.names),
            "nominal_misfit": self.nominal_misfit,
            "variance_explained": self.variance_explained,
        }


def fit_resistivity(
    transients: Sequence[DecayTransient],
    linkage: Linkage,
    resistance: np.ndarray,
    coupling: np.ndarray,
    channels: Sequence[str],
    turns: Sequence[PassiveTurn],
    *,
    names: Sequence[str] | None = None,
    mode_count: int = RESOLVED_MODE_COUNT,
    bounds: tuple[float, float] = MULTIPLIER_BOUNDS,
    start: Sequence[float] | None = None,
    maxiter: int = 200,
) -> ResistivityFit:
    """Fit one resistivity multiplier per conductor class on the given transients.

    The search runs in the logarithm of the multiplier, so a factor of two up and
    a factor of two down are the same distance and the positivity the physics
    requires is structural rather than enforced by a bound.  ``start`` seeds the
    search away from the nominal model, which is what makes a refit from a known
    optimum -- a leave-one-out or a profile point -- affordable.
    """

    names = tuple(names if names is not None else class_names(turns))
    if not names:
        raise DecayModeError("no resistivity class to fit")

    def misfit_of(logarithms: np.ndarray) -> float:
        return decay_misfit(
            transients,
            linkage,
            resistance,
            coupling,
            channels,
            turns,
            names,
            np.exp(logarithms),
            mode_count=mode_count,
        ).misfit

    nominal = decay_misfit(
        transients,
        linkage,
        resistance,
        coupling,
        channels,
        turns,
        names,
        np.ones(len(names)),
        mode_count=mode_count,
    )
    limits = [(math.log(bounds[0]), math.log(bounds[1]))] * len(names)
    seed = (
        np.zeros(len(names))
        if start is None
        else np.log(np.clip(np.asarray(start, dtype=float), *bounds))
    )
    result = minimize(
        misfit_of,
        seed,
        method="L-BFGS-B",
        bounds=limits,
        options={"maxiter": maxiter, "eps": 1.0e-3, "ftol": 1.0e-10},
    )
    fitted = decay_misfit(
        transients,
        linkage,
        resistance,
        coupling,
        channels,
        turns,
        names,
        np.exp(result.x),
        mode_count=mode_count,
    )
    return ResistivityFit(
        names=names,
        multipliers=tuple(float(value) for value in np.exp(result.x)),
        misfit=fitted.misfit,
        nominal_misfit=nominal.misfit,
        variance_explained=fitted.variance_explained,
        iterations=int(result.nit),
        converged=bool(result.success),
        reconstructions=fitted.reconstructions,
    )


@dataclass(frozen=True)
class ProfileInterval:
    """One class's identifiability, measured by profiling it against the rest.

    ``lower`` and ``upper`` bracket the multipliers whose best achievable misfit
    stays within ``tolerance`` of the optimum, every other class re-optimised at
    each point.  ``identified`` is the verdict: an interval that runs to the
    search bounds means the data does not constrain this class at all, however
    confidently the joint fit reported a number for it.
    """

    name: str
    fitted: float
    lower: float
    upper: float
    tolerance: float
    curvature: float
    samples: tuple[tuple[float, float], ...]

    @property
    def identified(self) -> bool:
        """Return whether the profile closes inside the search bounds."""

        return (
            self.lower > MULTIPLIER_BOUNDS[0] * 1.01
            and self.upper < MULTIPLIER_BOUNDS[1] * 0.99
            and self.upper / self.lower < 100.0
        )

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical JSON representation."""

        return {
            "curvature": self.curvature,
            "fitted": self.fitted,
            "identified": self.identified,
            "lower": self.lower,
            "name": self.name,
            "samples": [[float(x), float(y)] for x, y in self.samples],
            "tolerance": self.tolerance,
            "upper": self.upper,
        }


def profile_class(
    name: str,
    fit: ResistivityFit,
    transients: Sequence[DecayTransient],
    linkage: Linkage,
    resistance: np.ndarray,
    coupling: np.ndarray,
    channels: Sequence[str],
    turns: Sequence[PassiveTurn],
    *,
    tolerance_fraction: float = 0.02,
    points: int = 13,
    mode_count: int = RESOLVED_MODE_COUNT,
) -> ProfileInterval:
    """Profile one class's multiplier, re-optimising every other class.

    Re-optimising the others is what makes this an identifiability statement
    rather than a sensitivity one: a class whose effect another class can undo
    has a flat profile even though the joint fit found a sharp minimum, and that
    flatness is precisely the thing that must stop a promotion.
    """

    position = fit.names.index(name)
    others = [index for index in range(len(fit.names)) if index != position]
    fitted = fit.multipliers[position]
    grid = np.unique(
        np.concatenate(
            [
                np.geomspace(MULTIPLIER_BOUNDS[0], MULTIPLIER_BOUNDS[1], points),
                [fitted],
            ]
        )
    )
    tolerance = fit.misfit * (1.0 + tolerance_fraction)

    def best_at(value: float) -> float:
        if not others:
            trial = np.array([value])
            return decay_misfit(
                transients,
                linkage,
                resistance,
                coupling,
                channels,
                turns,
                fit.names,
                trial,
                mode_count=mode_count,
            ).misfit

        def objective(logarithms: np.ndarray) -> float:
            trial = np.empty(len(fit.names))
            trial[position] = value
            trial[others] = np.exp(logarithms)
            return decay_misfit(
                transients,
                linkage,
                resistance,
                coupling,
                channels,
                turns,
                fit.names,
                trial,
                mode_count=mode_count,
            ).misfit

        start = np.log([fit.multipliers[index] for index in others])
        limits = [
            (math.log(MULTIPLIER_BOUNDS[0]), math.log(MULTIPLIER_BOUNDS[1]))
        ] * len(others)
        outcome = minimize(
            objective,
            start,
            method="L-BFGS-B",
            bounds=limits,
            options={"maxiter": 60, "eps": 2.0e-3, "ftol": 1.0e-9},
        )
        return float(outcome.fun)

    samples = tuple((float(value), best_at(float(value))) for value in grid)
    lower, upper = _tolerance_crossing(samples, fitted, tolerance)
    return ProfileInterval(
        name=name,
        fitted=fitted,
        lower=lower,
        upper=upper,
        tolerance=tolerance,
        curvature=_profile_curvature(samples, fitted),
        samples=samples,
    )


def _tolerance_crossing(
    samples: Sequence[tuple[float, float]], fitted: float, tolerance: float
) -> tuple[float, float]:
    """Return where the profile crosses the tolerance either side of the optimum.

    The crossing is interpolated in the logarithm of the multiplier rather than
    snapped to whichever grid point happened to fall inside.  Snapping would make
    a sharply identified class report a degenerate interval whenever the optimum
    sits between two grid points -- the tighter the constraint, the narrower the
    band, and so the more likely no grid point lands in it at all.
    """

    logarithms = np.log([value for value, _ in samples])
    scores = np.asarray([score for _, score in samples])
    centre = int(np.argmin(np.abs(logarithms - math.log(fitted))))

    def walk(step: int, limit: float) -> float:
        index = centre
        while 0 <= index + step < scores.size:
            if scores[index + step] > tolerance:
                span = logarithms[index + step] - logarithms[index]
                rise = scores[index + step] - scores[index]
                if rise <= 0.0:
                    return float(np.exp(logarithms[index + step]))
                share = float(np.clip((tolerance - scores[index]) / rise, 0.0, 1.0))
                return float(np.exp(logarithms[index] + span * share))
            index += step
        return limit

    return (walk(-1, MULTIPLIER_BOUNDS[0]), walk(1, MULTIPLIER_BOUNDS[1]))


def _profile_curvature(samples: Sequence[tuple[float, float]], fitted: float) -> float:
    """Return the profile's second derivative in log-multiplier at the optimum.

    A flat profile has no curvature and identifies nothing; the number is
    reported so a marginal interval can be read against how sharp the minimum
    was rather than only against a tolerance.
    """

    logarithms = np.log([value for value, _ in samples])
    scores = np.asarray([score for _, score in samples])
    if logarithms.size < 3:
        return 0.0
    weight = np.exp(-(((logarithms - math.log(fitted)) / 1.5) ** 2))
    coefficients = np.polyfit(logarithms, scores, 2, w=weight)
    return float(2.0 * coefficients[0])


def mode_count_sensitivity(
    fit: ResistivityFit,
    transients: Sequence[DecayTransient],
    linkage: Linkage,
    resistance: np.ndarray,
    coupling: np.ndarray,
    channels: Sequence[str],
    turns: Sequence[PassiveTurn],
    *,
    counts: Sequence[int] = (2, 3, 4),
) -> dict[str, Any]:
    """Refit at other mode counts to show whether the declared count was binding."""

    rows = {}
    for count in counts:
        trial = fit_resistivity(
            transients,
            linkage,
            resistance,
            coupling,
            channels,
            turns,
            names=fit.names,
            mode_count=count,
        )
        rows[str(count)] = {
            "misfit": trial.misfit,
            "multipliers": dict(zip(trial.names, trial.multipliers, strict=True)),
        }
    return rows


def held_out_score(
    fit: ResistivityFit,
    transients: Sequence[DecayTransient],
    linkage: Linkage,
    resistance: np.ndarray,
    coupling: np.ndarray,
    channels: Sequence[str],
    turns: Sequence[PassiveTurn],
    *,
    mode_count: int = RESOLVED_MODE_COUNT,
) -> dict[str, Any]:
    """Score a fitted model and the nominal one on transients it never saw.

    Both numbers are reported because only their difference is a result: a
    fitted model that predicts held-out decays no better than the nominal one has
    fitted the training shots, and the promotion contract turns on exactly that
    comparison.
    """

    fitted = decay_misfit(
        transients,
        linkage,
        resistance,
        coupling,
        channels,
        turns,
        fit.names,
        np.asarray(fit.multipliers),
        mode_count=mode_count,
    )
    nominal = decay_misfit(
        transients,
        linkage,
        resistance,
        coupling,
        channels,
        turns,
        fit.names,
        np.ones(len(fit.names)),
        mode_count=mode_count,
    )
    return {
        "fitted_misfit": fitted.misfit,
        "fitted_variance_explained": fitted.variance_explained,
        "improvement": (
            0.0
            if nominal.misfit <= 0.0
            else float(1.0 - fitted.misfit / nominal.misfit)
        ),
        "nominal_misfit": nominal.misfit,
        "nominal_variance_explained": nominal.variance_explained,
        "shots": [row.shot for row in fitted.reconstructions],
        "transient_count": len(transients),
    }


def leave_one_out(
    transients: Sequence[DecayTransient],
    linkage: Linkage,
    resistance: np.ndarray,
    coupling: np.ndarray,
    channels: Sequence[str],
    turns: Sequence[PassiveTurn],
    *,
    names: Sequence[str],
    mode_count: int = RESOLVED_MODE_COUNT,
    start: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Refit with each shot dropped in turn, to measure cross-shot stability.

    The promotion contract asks for a value that does not depend on which shots
    happened to be in the set.  Dropping one shot at a time and reporting the
    spread of each multiplier answers that directly, and a class whose value
    swings by more than its own profile interval is not stable whatever the joint
    fit said.  Each refit starts from the joint optimum, which is what keeps a
    full leave-one-out sweep affordable without changing where it lands.
    """

    spread: dict[str, list[float]] = {name: [] for name in names}
    for dropped in range(len(transients)):
        subset = [row for index, row in enumerate(transients) if index != dropped]
        if len(subset) < 2:
            continue
        trial = fit_resistivity(
            subset,
            linkage,
            resistance,
            coupling,
            channels,
            turns,
            names=names,
            mode_count=mode_count,
            start=start,
            maxiter=60,
        )
        for name, value in zip(trial.names, trial.multipliers, strict=True):
            spread[name].append(value)
    return {
        name: {
            "maximum": float(np.max(values)) if values else math.nan,
            "median": float(np.median(values)) if values else math.nan,
            "minimum": float(np.min(values)) if values else math.nan,
            "relative_spread": (
                float((np.max(values) - np.min(values)) / np.median(values))
                if values and np.median(values) > 0.0
                else math.nan
            ),
        }
        for name, values in spread.items()
    }
