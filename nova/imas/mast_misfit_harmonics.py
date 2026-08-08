"""Harmonic maps of the MAST vacuum misfit, per drive group.

The misfit is what a drive produces at the sensors that the described windings
cannot produce anywhere.  Away from the conductor carrying it, that field solves
the homogeneous Grad-Shafranov operator, so it expands exactly in the ring
functions of :mod:`nova.biot.toroidalharmonic` and the expansion is a map: a
flux everywhere, not a score at candidate source points.  This module turns the
banked per-shot couplings into the constraint sets that expansion is fitted to,
and the fits into maps.

Two sensor classes, and why both are needed
-------------------------------------------
A pickup probe reads one field component along its own axis, which is the CURL
of a harmonic column.  A flux loop is closed about the machine axis and reads
the total poloidal flux through its own contour, which IS a harmonic column.
The distinction is what makes a mixed set worth assembling: the described
thirteen-coil column space absorbs a third of any filament's probe pattern and
leaves the probe array able to identify only ten of its thirteen directions, and
a probe row cannot escape that because every probe reads the same functional of
the field.  A loop row is a different functional and is not confined to the same
null space.  The two classes arrive in different units -- tesla per ampere-turn
and weber per ampere-turn -- and are made commensurable by whitening each row
with its own measured noise, which is the only weighting under which a
least-squares solve of a mixed set means anything.

Never fit the projected survivor alone
--------------------------------------
The tempting reduction is to project the described coil columns out of the
coupling first and fit the harmonics to what survives.  It is wrong, and
measurably so: the projection removes a third of a filament's own pattern, so
the surviving vector is no longer any filament's field and the position read
off it is biased by up to 202 mm with no noise at all.  The coil currents are
therefore carried as unknowns ALONGSIDE the harmonic coefficients and both are
fitted to the raw coupling.  The projected fit is retained only as the
comparison that measures what the shortcut costs on real data.

The two families and the shell between the bands
------------------------------------------------
About a focal circle placed in the candidate source region, every sensor sits
at a smaller focal distance than the source, so the source enters through the
first-kind (``INNER``) family -- the enclosed side.  The second-kind (``OUTER``)
family carries whatever sits FARTHER from the focal circle than the sensors do:
the solenoid, the far field, anything the description leaks in from outside.
Fitting each family alone tests the two hypotheses against each other, and the
two-family fit is the reconciled map.  Where the one-family maps agree, the
measurement determines the flux whichever side the current is on; where they
diverge beyond the resampling band, it does not, and the divergence is centred
on the current that separates them.

The sensor set falls into two bands that face each other across the machine --
the centre column at 0.18 m and the outboard wall beyond 1.4 m, with a single
loop between them at 0.66 m.  Fitting one band and predicting the other is the
sharpest available statement about the source: one expansion serves both bands
only if the current it represents is placed where both bands see it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from nova.biot import toroidalharmonic as th

CENTRE_COLUMN_RADIUS = 0.5
"""Radius separating the two sensor bands [m].

The centre-column sensors sit at 0.178-0.180 m and the nearest sensor off the
column at 0.662 m, so the split is a gap in the geometry rather than a chosen
threshold; the margin either side is a factor of 2.8 and 1.3.
"""

SIGNIFICANCE = 3.0
"""Standard deviations a whitened singular direction must carry to be kept."""

FOCAL_OFFSET = 0.15
"""Displacement of the focal circle from the source it is placed around [m].

Far enough that the source's focal distance stays inside the range the ladders
are conditioned over, close enough that the coefficient modulus still falls fast
with degree.  The recovery study measured 0.12-0.20 m to be equivalent.
"""

MEDIAN_EFFICIENCY = 1.2533
"""Ratio of a median's standard error to a mean's for a normal sample."""

ROBUST_SCALE = 1.4826
"""Factor taking a median absolute deviation to a normal standard deviation."""


class MisfitMapError(ValueError):
    """A constraint set or fit the geometry or the data cannot support."""


@dataclass(frozen=True)
class SensorClass:
    """One sensor class's poses, per-shot couplings and described response.

    ``coupling`` is ``(n_shot, n_channel)`` in the class's own units per
    ampere-turn -- tesla for a probe, weber for a loop -- and carries NaN where a
    shot could not serve a channel.  ``described`` is the same class's response to
    each described coil family, in the same units per ampere-turn, and is what
    the joint fit gives the coil-current unknowns to act through.
    """

    channel: tuple[str, ...]
    r: np.ndarray
    z: np.ndarray
    radial_cosine: np.ndarray
    axial_sine: np.ndarray
    reads_flux: bool
    coupling: np.ndarray
    described: np.ndarray
    floor: np.ndarray

    def __post_init__(self):
        """Reject a class whose arrays disagree about how many channels it has."""
        count = len(self.channel)
        shapes = {
            "r": self.r.shape,
            "z": self.z.shape,
            "radial_cosine": self.radial_cosine.shape,
            "axial_sine": self.axial_sine.shape,
            "floor": self.floor.shape,
        }
        wrong = {name: shape for name, shape in shapes.items() if shape != (count,)}
        if wrong:
            raise MisfitMapError(f"{count} channels but {wrong}")
        if self.coupling.ndim != 2 or self.coupling.shape[1] != count:
            raise MisfitMapError(
                f"coupling {self.coupling.shape} against {count} channels"
            )
        if self.described.shape[0] != count:
            raise MisfitMapError(
                f"described response {self.described.shape} against {count} channels"
            )


@dataclass(frozen=True)
class ConstraintSet:
    """One drive group's pooled misfit, ready to fit.

    ``value`` and ``noise`` are the pooled coupling and its standard error in
    each row's own units per ampere-turn; ``sample`` keeps the per-shot rows the
    pooling came from so a resampling band can be built without returning to the
    store.
    """

    group: str
    channel: tuple[str, ...]
    r: np.ndarray
    z: np.ndarray
    radial_cosine: np.ndarray
    axial_sine: np.ndarray
    reads_flux: np.ndarray
    value: np.ndarray
    noise: np.ndarray
    described: np.ndarray
    sample: np.ndarray
    shots: tuple[int, ...] = ()

    @property
    def rows(self) -> int:
        """Return the number of constrained rows."""
        return self.value.size

    @property
    def weight(self) -> np.ndarray:
        """Return the whitening weight, the reciprocal of each row's noise."""
        return 1.0 / self.noise

    @property
    def centre_column(self) -> np.ndarray:
        """Return the mask of rows mounted on the centre column."""
        return self.r < CENTRE_COLUMN_RADIUS

    @property
    def outboard(self) -> np.ndarray:
        """Return the mask of rows on the outboard wall or the outer coils."""
        return self.r >= CENTRE_COLUMN_RADIUS

    def select(self, mask) -> ConstraintSet:
        """Return the constraint set restricted to the rows ``mask`` keeps."""
        mask = np.asarray(mask, dtype=bool)
        if not mask.any():
            raise MisfitMapError(f"{self.group}: no row survives the selection")
        return ConstraintSet(
            group=self.group,
            channel=tuple(np.asarray(self.channel)[mask]),
            r=self.r[mask],
            z=self.z[mask],
            radial_cosine=self.radial_cosine[mask],
            axial_sine=self.axial_sine[mask],
            reads_flux=self.reads_flux[mask],
            value=self.value[mask],
            noise=self.noise[mask],
            described=self.described[mask],
            sample=self.sample[:, mask],
            shots=self.shots,
        )


def pooled_noise(sample: np.ndarray, floor: np.ndarray) -> tuple[np.ndarray, ...]:
    """Return the pooled coupling, its standard error and the count per channel.

    Channels are pooled by median because a single shot's regression can be
    thrown by a window the drive left, and the median's standard error is its
    own robust scatter over the root of the count, inflated by the efficiency a
    median loses against a mean.  ``floor`` is the instrument's own contribution
    to one shot, and enters as a floor rather than as the estimate: the measured
    scatter carries the shot-to-shot systematics as well and is the larger of the
    two wherever the drive was not the only thing moving.
    """
    sample = np.asarray(sample, dtype=np.float64)
    floor = np.asarray(floor, dtype=np.float64)
    count = np.isfinite(sample).sum(axis=0)
    with np.errstate(invalid="ignore"):
        value = np.nanmedian(sample, axis=0)
        spread = ROBUST_SCALE * np.nanmedian(np.abs(sample - value[None, :]), axis=0)
    live = count >= 2
    # A channel whose shots agree to the last bit reports no scatter, which is a
    # count artefact rather than a perfect measurement; fall back to the set's
    # own median scatter so it cannot dominate the whitened solve.
    typical = float(np.nanmedian(spread[live & (spread > 0.0)])) if live.any() else 0.0
    spread = np.where(np.isfinite(spread) & (spread > 0.0), spread, typical)
    error = MEDIAN_EFFICIENCY * spread / np.sqrt(np.maximum(count, 1))
    return value, np.maximum(error, floor / np.sqrt(np.maximum(count, 1))), count


def sensor_class(
    archive,
    group: str,
    *,
    reads_flux: bool,
    floor=None,
) -> SensorClass:
    """Return one sensor class's rows for one drive group from a coupling bank.

    The bank holds one row per (shot, drive group) with the whole channel set on
    each, which is what lets a resampling draw take a shot and keep every channel
    that shot served.  A field class carries the sensitive axis the pose states;
    a flux class has none, and the axis arrays are filled with the radial
    direction so the array shapes match without ever being read.
    """
    groups = [str(name) for name in np.asarray(archive["groups"])]
    rows = [index for index, name in enumerate(groups) if name == group]
    if not rows:
        raise MisfitMapError(f"the bank carries no rows for drive group {group!r}")
    channel = tuple(str(name) for name in np.asarray(archive["channel"]))
    count = len(channel)
    if reads_flux:
        cosine, sine = np.ones(count), np.zeros(count)
    else:
        cosine = np.asarray(archive["cos"], dtype=np.float64)
        sine = np.asarray(archive["sin"], dtype=np.float64)
    spans = np.asarray(archive["spans"], dtype=np.float64)[rows]
    scatter = np.zeros(count) if floor is None else np.asarray(floor, dtype=np.float64)
    return SensorClass(
        channel=channel,
        r=np.asarray(archive["r"], dtype=np.float64),
        z=np.asarray(archive["z"], dtype=np.float64),
        radial_cosine=cosine,
        axial_sine=sine,
        reads_flux=bool(reads_flux),
        coupling=np.asarray(archive["coupling"], dtype=np.float64)[rows],
        described=np.asarray(archive["response"], dtype=np.float64),
        floor=scatter * float(np.sqrt(np.mean(1.0 / spans**2))),
    )


def bank_shots(archive, group: str) -> tuple[int, ...]:
    """Return the shots a coupling bank holds for one drive group."""
    groups = [str(name) for name in np.asarray(archive["groups"])]
    shots = np.asarray(archive["shots"], dtype=int)
    return tuple(
        int(shots[index]) for index, name in enumerate(groups) if name == group
    )


def assemble(
    group: str,
    classes,
    *,
    excluded=(),
    minimum_shots: int = 2,
    shots=(),
) -> ConstraintSet:
    """Return one drive group's constraint set from its sensor classes.

    A channel is carried when at least ``minimum_shots`` of the group's shots
    served it, so a channel pooled from a single regression -- whose scatter
    cannot be measured and whose weight would therefore be invented -- never
    enters the solve.
    """
    excluded = set(excluded)
    channels, position, height = [], [], []
    cosine, sine, flux, values, noises, described, samples = [], [], [], [], [], [], []
    for sensor in classes:
        value, error, count = pooled_noise(sensor.coupling, sensor.floor)
        keep = (
            (count >= minimum_shots)
            & np.isfinite(value)
            & (error > 0.0)
            & np.asarray([name not in excluded for name in sensor.channel])
        )
        if not keep.any():
            continue
        channels.extend(np.asarray(sensor.channel)[keep].tolist())
        position.append(sensor.r[keep])
        height.append(sensor.z[keep])
        cosine.append(sensor.radial_cosine[keep])
        sine.append(sensor.axial_sine[keep])
        flux.append(np.full(int(keep.sum()), sensor.reads_flux, dtype=bool))
        values.append(value[keep])
        noises.append(error[keep])
        described.append(sensor.described[keep])
        samples.append(sensor.coupling[:, keep])
    if not channels:
        raise MisfitMapError(f"{group}: no channel clears the pooling requirement")
    return ConstraintSet(
        group=group,
        channel=tuple(channels),
        r=np.concatenate(position),
        z=np.concatenate(height),
        radial_cosine=np.concatenate(cosine),
        axial_sine=np.concatenate(sine),
        reads_flux=np.concatenate(flux),
        value=np.concatenate(values),
        noise=np.concatenate(noises),
        described=np.concatenate(described, axis=0),
        sample=np.concatenate(samples, axis=1),
        shots=tuple(int(shot) for shot in shots),
    )


def harmonic_design(basis: th.ToroidalHarmonics, constraint: ConstraintSet):
    """Return the harmonic columns each row reads, in that row's own units.

    A flux row takes the column itself and a field row takes its curl projected
    onto the row's sensitive axis, which is the whole difference between the two
    sensor classes as far as the solve is concerned.
    """
    design = np.empty((constraint.rows, len(basis.labels)), dtype=np.float64)
    flux = constraint.reads_flux
    if flux.any():
        design[flux] = basis.flux(constraint.r[flux], constraint.z[flux])
    if (~flux).any():
        design[~flux] = basis.project(
            constraint.r[~flux],
            constraint.z[~flux],
            constraint.radial_cosine[~flux],
            constraint.axial_sine[~flux],
        )
    return design


@dataclass(frozen=True)
class MisfitFit:
    """One fitted expansion of a drive group's misfit."""

    group: str
    order: int
    families: tuple[str, ...]
    coefficients: np.ndarray
    """Harmonic coefficients alone, whatever the fit carried beside them."""

    currents: np.ndarray = field(default_factory=lambda: np.empty(0))
    """Described-coil ampere-turn corrections, empty for a projected fit."""

    labels: tuple[str, ...] = ()
    rank: int = 0
    condition: float = float("nan")
    residual: float = float("nan")
    """Root-mean-square whitened residual; one is a fit at the noise level."""

    explained: float = float("nan")
    """Fraction of the whitened data power the fit reproduces."""

    coil_explained: float = float("nan")
    """Fraction the described coil columns alone reproduce, at the same rank."""

    def family_slice(self, family: str) -> slice:
        """Return the coefficient span of one radial family."""
        if family not in self.families:
            raise MisfitMapError(f"fit carries {self.families}, not {family!r}")
        width = self.coefficients.size // len(self.families)
        start = self.families.index(family) * width
        return slice(start, start + width)


def _explained(design, data, weight, coefficients) -> float:
    """Return the whitened variance fraction a solution reproduces."""
    residual = weight * (design @ coefficients - data)
    total = float(np.sum((weight * data) ** 2))
    if not total > 0.0:
        return float("nan")
    return float(1.0 - np.sum(residual**2) / total)


def fit_jointly(
    basis: th.ToroidalHarmonics,
    constraint: ConstraintSet,
    *,
    significance: float = SIGNIFICANCE,
) -> MisfitFit:
    """Fit harmonics and described coil currents together to the raw coupling.

    The default, and the only fit whose source read is unbiased: the coil columns
    are free rather than projected out, so the harmonic block keeps the whole of
    a source's pattern instead of the third that survives the projection.
    """
    harmonic = harmonic_design(basis, constraint)
    design = np.column_stack([harmonic, constraint.described])
    solution = th.solve_equilibrated(
        design, constraint.value, weight=constraint.weight, significance=significance
    )
    width = harmonic.shape[1]
    coil_only = th.solve_equilibrated(
        constraint.described,
        constraint.value,
        weight=constraint.weight,
        significance=significance,
    )
    return MisfitFit(
        group=constraint.group,
        order=basis.order,
        families=tuple(basis.families),
        coefficients=solution.coefficients[:width],
        currents=solution.coefficients[width:],
        labels=tuple(basis.labels),
        rank=solution.rank,
        condition=solution.equilibrated_condition,
        residual=solution.residual,
        explained=_explained(
            design, constraint.value, constraint.weight, solution.coefficients
        ),
        coil_explained=_explained(
            constraint.described,
            constraint.value,
            constraint.weight,
            coil_only.coefficients,
        ),
    )


def fit_projected(
    basis: th.ToroidalHarmonics,
    constraint: ConstraintSet,
    *,
    significance: float = SIGNIFICANCE,
) -> MisfitFit:
    """Fit harmonics to the coupling with the described column space removed.

    The shortcut the joint fit exists to avoid, kept so the two can be compared
    on the same data.  Both the design and the data are projected, so the fit is
    consistent -- it is the REPRESENTATION that is damaged, because the projector
    removes directions the source itself occupies.
    """
    harmonic = harmonic_design(basis, constraint)
    described = constraint.described * constraint.weight[:, None]
    orthogonal = np.eye(constraint.rows) - described @ np.linalg.pinv(described)
    whitened = orthogonal * constraint.weight[None, :]
    solution = th.solve_equilibrated(
        whitened @ harmonic, whitened @ constraint.value, significance=significance
    )
    return MisfitFit(
        group=constraint.group,
        order=basis.order,
        families=tuple(basis.families),
        coefficients=solution.coefficients,
        labels=tuple(basis.labels),
        rank=solution.rank,
        condition=solution.equilibrated_condition,
        residual=solution.residual,
        explained=_explained(
            whitened @ harmonic,
            whitened @ constraint.value,
            np.ones(constraint.rows),
            solution.coefficients,
        ),
    )


def select_degree(
    focus: th.FocalCircle,
    constraint: ConstraintSet,
    orders,
    families=(th.INNER,),
    *,
    folds: int = 5,
    significance: float = SIGNIFICANCE,
    seed: int = 0,
):
    """Return the degree that predicts held-out CHANNELS best, and every score.

    Rows are held out, not shots: the question a degree answers is whether the
    added angular structure is seen by sensors that did not set it, and the
    in-sample residual cannot answer it because it falls with degree regardless.

    Two departures from a plain held-out root-mean-square, both forced by this
    row set.  The design and data are whitened FIRST, because the set mixes tesla
    with weber and an unweighted score is set by whichever class carries the
    larger numbers -- it would answer a question about unit choice.  And the
    score is the MEDIAN absolute whitened error rather than its
    root-mean-square, because leverage is wildly uneven here: one loop is mounted
    on a coil whose described column it alone pins, and holding that single row
    out leaves its own prediction free to run to twenty times the data, which
    swamps a squared score at every degree equally.  A median measures the degree;
    a root-mean-square measures which fold got the leverage row.

    The degree returned is the LOWEST whose score sits within one standard error
    of the best, the spread being taken across the folds themselves.  Above about
    degree six the score curve here is flat to a few percent, and taking its bare
    minimum would buy a doubling of the column count with a difference the fold
    spread cannot resolve.
    """
    weight = constraint.weight
    data = constraint.value * weight
    index = np.arange(data.size)
    np.random.default_rng(seed).shuffle(index)
    parts = [index[k::folds] for k in range(folds)]
    scores = {}
    for order in orders:
        basis = th.ToroidalHarmonics(focus, order=int(order), families=tuple(families))
        design = (
            np.column_stack([harmonic_design(basis, constraint), constraint.described])
            * weight[:, None]
        )
        errors = []
        for held in parts:
            train = np.setdiff1d(index, held)
            solution = th.solve_equilibrated(
                design[train], data[train], significance=significance
            )
            errors.append(
                float(
                    np.median(np.abs(design[held] @ solution.coefficients - data[held]))
                )
            )
        scores[int(order)] = (float(np.mean(errors)), float(np.std(errors, ddof=1)))
    lowest = min(scores, key=lambda key: scores[key][0])
    reach = scores[lowest][0] + scores[lowest][1] / np.sqrt(folds)
    best = min(order for order in scores if scores[order][0] <= reach)
    return best, {order: value for order, (value, _) in scores.items()}


def place_focus(source_r: float, source_z: float, *, offset: float = FOCAL_OFFSET):
    """Return a focal circle displaced vertically from a source position."""
    return th.FocalCircle(float(source_r), float(source_z) + float(offset))


@dataclass(frozen=True)
class SourceRead:
    """A source position read off a fitted one-family expansion, with its focus."""

    focus: th.FocalCircle
    fit: MisfitFit
    estimate: th.SourceEstimate | None
    track: tuple[tuple[float, float], ...] = ()
    convergent: bool = False
    """Whether every constrained row sits on the side the expansion converges on.

    An inner-family expansion converges strictly farther from the focal circle
    than its source.  A read placing the source nearer the focal circle than some
    sensor is not merely imprecise: the series it was fitted with diverges at
    that sensor, so the read refutes its own fit and must not be reported as a
    position.
    """

    rows: int = 0
    """Rows the settled read was fitted on, after divergent ones were dropped."""


def rows_converge(
    basis: th.ToroidalHarmonics, constraint: ConstraintSet, distance: float
) -> bool:
    """Return whether every constrained row lies where the expansion converges."""
    return bool(th.convergent_points(basis, constraint.r, constraint.z, distance).all())


def _inside(constraint: ConstraintSet, r: float, z: float) -> bool:
    """Return whether a position lies within the sensor set's own extent.

    The bound is the array rather than a stated machine size: a source the
    sensors surround is one they can place, and a read outside the box they span
    is an extrapolation the same measurement cannot support whatever the vessel
    happens to be shaped like.
    """
    return bool(
        constraint.r.min() <= r <= constraint.r.max()
        and constraint.z.min() <= z <= constraint.z.max()
    )


def iterate_focus(
    constraint: ConstraintSet,
    seed: th.FocalCircle,
    *,
    order: int = 6,
    family: str = th.INNER,
    offset: float = FOCAL_OFFSET,
    rounds: int = 4,
    significance: float = SIGNIFICANCE,
) -> SourceRead:
    """Return the focal circle and fit a source read settles on from ``seed``.

    A single-family read is only as good as the focal circle it is expanded
    about: too far from the source and the coefficient modulus barely falls with
    degree, so the distance the modulus encodes is poorly determined.  Re-placing
    the circle at a fixed offset from each read and refitting converges in a few
    rounds and is what took the synthetic position error from 31 mm to 5.4 mm.

    A read and its row set have to be made self-consistent, not merely fitted: a
    one-family expansion DIVERGES at any sensor lying on the far side of the
    source it implies, so such a row cannot be down-weighted, only dropped, and
    dropping it changes the read that decided to drop it.  Each round therefore
    refits on the rows the previous read leaves convergent and stops when both
    the focus and the row set stop moving.  ``convergent`` records whether the
    settled read kept every row; where it did not, the count it kept is the
    honest support of the position.
    """
    focus, held, estimate = seed, None, None
    keep = np.ones(constraint.rows, dtype=bool)
    track: list[tuple[float, float]] = []
    for _ in range(int(rounds)):
        basis = th.ToroidalHarmonics(focus, order=order, families=(family,))
        if keep.sum() <= len(basis.labels):
            break
        rows = constraint.select(keep)
        fit = fit_jointly(basis, rows, significance=significance)
        try:
            read = th.locate_source(basis, fit.coefficients)
        except ValueError:
            break
        held, estimate = (focus, fit, keep), read
        track.append((read.r, read.z))
        if not _inside(constraint, read.r, read.z):
            break
        moved = place_focus(read.r, read.z, offset=offset)
        step = np.hypot(moved.radius - focus.radius, moved.height - focus.height)
        settled = th.convergent_points(
            th.ToroidalHarmonics(moved, order=order, families=(family,)),
            constraint.r,
            constraint.z,
            th.focal_frame(np.array([read.r]), np.array([read.z]), moved).distance[0],
        )
        stable = bool((settled == keep).all())
        focus, keep = moved, settled
        if step < 1.0e-4 and stable:
            break
    if held is None:
        basis = th.ToroidalHarmonics(seed, order=order, families=(family,))
        return SourceRead(
            focus=seed,
            fit=fit_jointly(basis, constraint, significance=significance),
            estimate=None,
            track=tuple(track),
        )
    return SourceRead(
        focus=held[0],
        fit=held[1],
        estimate=estimate,
        track=tuple(track),
        convergent=bool(held[2].all()),
        rows=int(held[2].sum()),
    )


def flux_map(basis: th.ToroidalHarmonics, coefficients, grid_r, grid_z):
    """Return the expansion's flux on a ``(n_z, n_r)`` raster [Wb per ampere-turn]."""
    mesh_r, mesh_z = np.meshgrid(
        np.asarray(grid_r, dtype=np.float64), np.asarray(grid_z, dtype=np.float64)
    )
    columns = basis.flux(mesh_r.ravel(), mesh_z.ravel())
    return (columns @ np.asarray(coefficients, dtype=np.float64)).reshape(mesh_r.shape)


def supported_mask(focus: th.FocalCircle, constraint: ConstraintSet, grid_r, grid_z):
    """Return the raster mask of points the sensor set brackets in focal distance.

    A harmonic expansion is an interpolation in the focal coordinate as much as a
    map in the plane: between the nearest and farthest focal distance any sensor
    holds, the fitted coefficients are pinned by measurement.  Closer to the
    focal circle than every sensor, the inner family's own singularity is being
    extrapolated toward with nothing to hold it; farther out than every sensor,
    the outer family's is.  Masking there says which part of the contour map the
    measurement is responsible for and which part is the basis talking to itself.
    """
    mesh_r, mesh_z = np.meshgrid(
        np.asarray(grid_r, dtype=np.float64), np.asarray(grid_z, dtype=np.float64)
    )
    distance = th.focal_frame(mesh_r.ravel(), mesh_z.ravel(), focus).distance
    span = th.focal_frame(constraint.r, constraint.z, focus).distance
    inside = (distance >= span.min()) & (distance <= span.max())
    return inside.reshape(mesh_r.shape)


def convergence_mask(basis: th.ToroidalHarmonics, grid_r, grid_z, distance: float):
    """Return the raster mask of points a one-family expansion converges at."""
    mesh_r, mesh_z = np.meshgrid(
        np.asarray(grid_r, dtype=np.float64), np.asarray(grid_z, dtype=np.float64)
    )
    inside = th.convergent_points(basis, mesh_r.ravel(), mesh_z.ravel(), distance)
    return inside.reshape(mesh_r.shape)


def resample_maps(
    basis: th.ToroidalHarmonics,
    constraint: ConstraintSet,
    grid_r,
    grid_z,
    *,
    draws: int = 200,
    significance: float = SIGNIFICANCE,
    seed: int = 0,
):
    """Return per-draw flux maps from resampling the group's SHOTS.

    Shots are the independent unit, not channels: every channel of one shot
    shares that shot's drive waveform, its window and its baseline, so resampling
    channels would understate the band by the amount those are correlated.  A
    draw that leaves a channel with fewer than two live shots drops the channel,
    which is the same rule the pooling uses.
    """
    generator = np.random.default_rng(seed)
    count = constraint.sample.shape[0]
    if count < 2:
        raise MisfitMapError(f"{constraint.group}: {count} shot rows cannot resample")
    floor = constraint.noise * np.sqrt(count)
    maps = []
    for _ in range(int(draws)):
        rows = generator.integers(0, count, count)
        drawn = constraint.sample[rows]
        value, error, live = pooled_noise(drawn, floor)
        keep = (live >= 2) & np.isfinite(value) & (error > 0.0)
        if keep.sum() < len(basis.labels):
            continue
        design = np.column_stack(
            [
                harmonic_design(basis, constraint.select(keep)),
                constraint.described[keep],
            ]
        )
        solution = th.solve_equilibrated(
            design, value[keep], weight=1.0 / error[keep], significance=significance
        )
        maps.append(
            flux_map(basis, solution.coefficients[: len(basis.labels)], grid_r, grid_z)
        )
    if not maps:
        raise MisfitMapError(f"{constraint.group}: every resampling draw was refused")
    return np.stack(maps)


def family_contrast(
    constraint: ConstraintSet,
    focus: th.FocalCircle,
    *,
    order: int = 6,
    significance: float = SIGNIFICANCE,
) -> float:
    """Return how far the enclosed family out-explains the external one here.

    Both families are fitted about the SAME focal circle, jointly with the coil
    currents, and the score is the difference of the whitened variance fractions
    they reach.  It compares two source hypotheses that differ only in which side
    of the circle the current sits on: the first kind is singular nowhere but the
    circle itself, so a positive contrast says the misfit looks like current AT
    this circle, and the second kind is singular only on the axis and at
    infinity, so a negative one says it looks like current out there.

    Scanned over the placement of the circle this localises, and more generally
    than a filament scan: at degree ``n`` the inner family represents any current
    distribution near the circle rather than one ring.  What it must not be
    confused with is the difference between the two families' flux MAPS.  That
    difference peaks beside whatever circle it is given -- measured here, its
    peak tracks the focal circle across the whole machine and reaches thirteen
    resampling sigma wherever it is put -- because it measures where the two
    bases disagree, which is next to their own singular sets.  It moves with the
    focus, not with the machine, and localises nothing.
    """
    scores = [
        fit_jointly(
            th.ToroidalHarmonics(focus, order=int(order), families=(family,)),
            constraint,
            significance=significance,
        ).explained
        for family in (th.INNER, th.OUTER)
    ]
    return float(scores[0] - scores[1])


def scan_focus(
    constraint: ConstraintSet,
    grid_r,
    grid_z,
    *,
    order: int = 6,
    significance: float = SIGNIFICANCE,
) -> np.ndarray:
    """Return :func:`family_contrast` on an ``(n_z, n_r)`` raster of placements."""
    grid_r = np.asarray(grid_r, dtype=np.float64)
    grid_z = np.asarray(grid_z, dtype=np.float64)
    return np.asarray(
        [
            [
                family_contrast(
                    constraint,
                    th.FocalCircle(float(radius), float(height)),
                    order=order,
                    significance=significance,
                )
                for radius in grid_r
            ]
            for height in grid_z
        ],
        dtype=np.float64,
    )


def cross_prediction(
    basis: th.ToroidalHarmonics, fit: MisfitFit, target: ConstraintSet
) -> dict[str, float]:
    """Return how a fit does on rows it never saw, against saying nothing.

    Reported as MEDIAN absolute whitened error rather than as an explained
    fraction.  An expansion fitted to one part of this array and evaluated on
    another is extrapolating, and an extrapolating harmonic series does not miss
    by a factor -- it misses by orders of magnitude on the few rows nearest its
    own singularity, which sends a squared score to minus ten-to-the-fourteen and
    stops it distinguishing bad from catastrophic.  The median says what a
    typical unseen row suffers, and ``baseline`` is what predicting zero there
    would have cost, so a ratio above one is an expansion worse than silence.
    """
    design = np.column_stack([harmonic_design(basis, target), target.described])
    coefficients = np.concatenate([fit.coefficients, fit.currents])
    error = float(
        np.median(np.abs(target.weight * (design @ coefficients - target.value)))
    )
    baseline = float(np.median(np.abs(target.weight * target.value)))
    return {
        "error": error,
        "baseline": baseline,
        "ratio": error / baseline if baseline > 0.0 else float("nan"),
    }


def band_agreement(
    basis: th.ToroidalHarmonics,
    constraint: ConstraintSet,
    *,
    significance: float = SIGNIFICANCE,
):
    """Return each band's fit and how well each predicts the band it never saw.

    The two bands face each other across the machine, so one expansion serves
    both only if the current it represents is placed where both bands see it.
    Where a source sits between them, each band's fit absorbs it into its own
    near field and predicts the far band no better than silence.
    """
    bands = {
        "centre_column": constraint.centre_column,
        "outboard": constraint.outboard,
    }
    fits, scores = {}, {}
    for name, mask in bands.items():
        if mask.sum() <= len(basis.labels):
            continue
        fits[name] = fit_jointly(
            basis, constraint.select(mask), significance=significance
        )
        scores[name] = cross_prediction(basis, fits[name], constraint.select(~mask))
    return fits, scores


__all__ = [
    "CENTRE_COLUMN_RADIUS",
    "FOCAL_OFFSET",
    "SIGNIFICANCE",
    "ConstraintSet",
    "MisfitFit",
    "MisfitMapError",
    "SensorClass",
    "assemble",
    "band_agreement",
    "bank_shots",
    "convergence_mask",
    "cross_prediction",
    "family_contrast",
    "fit_jointly",
    "fit_projected",
    "flux_map",
    "harmonic_design",
    "iterate_focus",
    "place_focus",
    "pooled_noise",
    "SourceRead",
    "resample_maps",
    "rows_converge",
    "select_degree",
    "supported_mask",
    "scan_focus",
    "sensor_class",
]
