"""Read MAST EFIT geometry after reconstruction and score it as a referee.

The reconstruction chain deliberately has no reference-equilibrium input.  This
module preserves that boundary: :func:`run_refereed_parity_chain` completes the
ordinary corrected-input solve before it opens the ``efm`` group, and the lower
level :func:`score_with_efit_referee` accepts an already-completed result.

EFIT publishes one magnetic axis, a fixed-width LCFS point string, and up to two
x-points at each catalogue time.  The absence of both x-points denotes a limited
slice.  Geometry comparisons are reduced with the median, matching the frozen
benchmark aggregation used to register the geometry tolerances.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from nova.imas.mast_parity_chain import (
    ParityChainResult,
    SliceScorecard,
    TopologyLabels,
    run_parity_chain,
)
from nova.imas.mast_vacuum_cohort import SHOT_STORE
from nova.imas.parity_tolerances import ScorecardField

FROZEN_SHOTS = (21978, 21983, 21985, 21986, 21989, 22086)
EFIT_GROUP = "efm"


class EfitRefereeError(ValueError):
    """The catalogue cannot provide a complete reference comparison."""


def _readonly(values: Any, *, dtype: Any = float) -> np.ndarray:
    """Return an owned immutable array."""

    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _valid_points(points: np.ndarray) -> np.ndarray:
    """Identify finite poloidal-plane positions within a generous machine bound."""

    return (
        np.all(np.isfinite(points), axis=-1)
        & (points[..., 0] > 0.0)
        & (points[..., 0] < 10.0)
        & (np.abs(points[..., 1]) < 10.0)
    )


@dataclass(frozen=True)
class EfitReferee:
    """Immutable geometry read directly from one shot's ``efm`` group."""

    shot: int
    time_s: np.ndarray
    magnetic_axis_m: np.ndarray
    lcfs_m: np.ndarray
    x_points_m: np.ndarray
    diverted: np.ndarray
    usable: np.ndarray

    def __post_init__(self) -> None:
        """Reject arrays that cannot describe the same catalogue slices."""

        count = int(np.asarray(self.time_s).size)
        expected = {
            "magnetic_axis_m": (count, 2),
            "x_points_m": (count, 2, 2),
            "diverted": (count,),
            "usable": (count,),
        }
        observed = {name: np.asarray(getattr(self, name)).shape for name in expected}
        if observed != expected:
            raise EfitRefereeError(
                "EFIT referee shapes disagree: "
                f"expected {expected}, observed {observed}"
            )
        lcfs = np.asarray(self.lcfs_m)
        if lcfs.ndim != 3 or lcfs.shape[0] != count or lcfs.shape[2] != 2:
            raise EfitRefereeError(
                "EFIT referee LCFS must have shape (slice_count, point_count, 2)"
            )

    @property
    def slice_count(self) -> int:
        """Return the number of EFIT catalogue rows."""

        return int(self.time_s.size)

    @property
    def usable_slice_count(self) -> int:
        """Return the number of rows carrying an axis and an LCFS."""

        return int(np.count_nonzero(self.usable))


@dataclass(frozen=True)
class ReferenceGeometryScores:
    """Per-slice comparisons and their shot-level reductions."""

    reference_index: np.ndarray
    reference_time_s: np.ndarray
    usable_reference: np.ndarray
    magnetic_axis_distance_m: np.ndarray
    lcfs_distance_m: np.ndarray
    x_point_distance_m: np.ndarray
    x_point_distance_failure_cause: np.ndarray
    topology_class_agreement: np.ndarray

    @property
    def usable_slice_count(self) -> int:
        """Return how many solve slices aligned to usable EFIT geometry."""

        return int(np.count_nonzero(self.usable_reference))

    @property
    def aggregates(self) -> dict[str, float]:
        """Return the four registered geometry metrics."""

        values = {
            ScorecardField.MAGNETIC_AXIS_DISTANCE_M.value: _finite_median(
                self.magnetic_axis_distance_m, "magnetic-axis distance"
            ),
            ScorecardField.LCFS_DISTANCE_M.value: _finite_median(
                self.lcfs_distance_m, "LCFS distance"
            ),
            ScorecardField.X_POINT_DISTANCE_M.value: _finite_median(
                self.x_point_distance_m,
                "x-point distance",
                failure_causes=self.x_point_distance_failure_cause,
            ),
            ScorecardField.TOPOLOGY_CLASS_AGREEMENT_FRACTION.value: _finite_mean(
                self.topology_class_agreement, "topology-class agreement"
            ),
        }
        return values


@dataclass(frozen=True)
class RefereedParityResult:
    """A completed reconstruction plus its score-only EFIT comparison."""

    chain: ParityChainResult
    referee: EfitReferee
    geometry_scores: ReferenceGeometryScores

    @property
    def scorecard(self) -> SliceScorecard:
        """Return the scorecard whose reference fields have been resolved."""

        return self.chain.scorecard

    @property
    def usable_reference_slices(self) -> int:
        """Return the number of solved slices with an aligned usable reference."""

        return self.geometry_scores.usable_slice_count


def read_efit_referee(
    shot: int,
    *,
    store: Path | str = SHOT_STORE,
) -> EfitReferee:
    """Read immutable scoring geometry from one MAST ``efm`` catalogue group."""

    import zarr

    root = zarr.open_group(f"{Path(store)}/{int(shot)}.zarr", mode="r")
    if EFIT_GROUP not in root:
        raise EfitRefereeError(f"shot {shot} carries no {EFIT_GROUP!r} group")
    group = root[EFIT_GROUP]
    required = {
        "time",
        "magnetic_axis_r",
        "magnetic_axis_z",
        "lcfs_r",
        "lcfs_z",
        "xpoint1_rc",
        "xpoint1_zc",
        "xpoint2_rc",
        "xpoint2_zc",
    }
    missing = sorted(required.difference(group.keys()))
    if missing:
        raise EfitRefereeError(
            f"shot {shot} EFIT group is missing geometry arrays {missing}"
        )

    time_s = np.asarray(group["time"][...], dtype=float)
    magnetic_axis = np.column_stack(
        [group["magnetic_axis_r"][...], group["magnetic_axis_z"][...]]
    ).astype(float)
    lcfs = np.stack([group["lcfs_r"][...], group["lcfs_z"][...]], axis=-1).astype(float)
    x_points = np.stack(
        [
            np.column_stack([group["xpoint1_rc"][...], group["xpoint1_zc"][...]]),
            np.column_stack([group["xpoint2_rc"][...], group["xpoint2_zc"][...]]),
        ],
        axis=1,
    ).astype(float)

    axis_valid = _valid_points(magnetic_axis)
    lcfs_valid = _valid_points(lcfs)
    x_point_valid = _valid_points(x_points)
    magnetic_axis = np.where(axis_valid[:, None], magnetic_axis, np.nan)
    lcfs = np.where(lcfs_valid[..., None], lcfs, np.nan)
    x_points = np.where(x_point_valid[..., None], x_points, np.nan)
    diverted = np.any(x_point_valid, axis=1)
    usable = np.isfinite(time_s) & axis_valid & (np.sum(lcfs_valid, axis=1) >= 3)

    order = np.argsort(time_s, kind="stable")
    return EfitReferee(
        shot=int(shot),
        time_s=_readonly(time_s[order]),
        magnetic_axis_m=_readonly(magnetic_axis[order]),
        lcfs_m=_readonly(lcfs[order]),
        x_points_m=_readonly(x_points[order]),
        diverted=_readonly(diverted[order], dtype=bool),
        usable=_readonly(usable[order], dtype=bool),
    )


def _nearest_reference_indices(
    reference_time_s: np.ndarray,
    solve_time_s: np.ndarray,
    tolerance_s: float | None,
) -> np.ndarray:
    """Match solve times to the closest catalogue row within half a clock tick."""

    reference_time = np.asarray(reference_time_s, dtype=float)
    solve_time = np.asarray(solve_time_s, dtype=float)
    if reference_time.size == 0:
        return np.full(solve_time.shape, -1, dtype=int)
    if tolerance_s is None:
        spacing = np.diff(reference_time)
        spacing = spacing[np.isfinite(spacing) & (spacing > 0.0)]
        if spacing.size == 0:
            tolerance_s = 1.0e-9
        else:
            tolerance_s = 0.5001 * float(np.median(spacing)) + 1.0e-9
    if tolerance_s < 0.0:
        raise EfitRefereeError("reference time tolerance must be non-negative")

    upper = np.searchsorted(reference_time, solve_time, side="left")
    upper = np.clip(upper, 0, reference_time.size - 1)
    lower = np.clip(upper - 1, 0, reference_time.size - 1)
    upper_distance = np.abs(reference_time[upper] - solve_time)
    lower_distance = np.abs(reference_time[lower] - solve_time)
    index = np.where(lower_distance <= upper_distance, lower, upper)
    distance = np.abs(reference_time[index] - solve_time)
    return np.where(np.isfinite(solve_time) & (distance <= tolerance_s), index, -1)


def _boundary_distance(first: np.ndarray, second: np.ndarray) -> float:
    """Return the symmetric mean nearest-point separation of two boundaries."""

    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    first = first[_valid_points(first)]
    second = second[_valid_points(second)]
    if first.shape[0] < 3 or second.shape[0] < 3:
        return float("nan")
    separation = np.linalg.norm(first[:, None, :] - second[None, :, :], axis=2)
    return float(
        0.5
        * (np.mean(np.min(separation, axis=0)) + np.mean(np.min(separation, axis=1)))
    )


def _finite_median(
    values: np.ndarray,
    name: str,
    *,
    failure_causes: np.ndarray | None = None,
) -> float:
    """Reduce a finite metric or refuse to manufacture a missing score."""

    if failure_causes is not None and np.any(np.asarray(failure_causes, dtype=str)):
        return float("nan")
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise EfitRefereeError(f"no comparable slices provide {name}")
    return float(np.median(finite))


def _finite_mean(values: np.ndarray, name: str) -> float:
    """Average a finite categorical metric or refuse an empty comparison."""

    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise EfitRefereeError(f"no comparable slices provide {name}")
    return float(np.mean(finite))


def compare_reference_geometry(
    time_s: np.ndarray,
    topology: TopologyLabels,
    referee: EfitReferee,
    *,
    time_tolerance_s: float | None = None,
) -> ReferenceGeometryScores:
    """Compare completed topology labels with the nearest EFIT catalogue rows."""

    time = np.asarray(time_s, dtype=float)
    topology.validate(time.size)
    index = _nearest_reference_indices(referee.time_s, time, time_tolerance_s)
    usable = index >= 0
    usable[usable] &= referee.usable[index[usable]]

    reference_time = np.full(time.shape, np.nan)
    axis_distance = np.full(time.shape, np.nan)
    lcfs_distance = np.full(time.shape, np.nan)
    x_point_distance = np.full(time.shape, np.nan)
    x_point_failure_cause = np.full(time.shape, "", dtype="U32")
    topology_agreement = np.full(time.shape, np.nan)
    solved_axis = np.asarray(topology.magnetic_axis_m, dtype=float)
    solved_lcfs = np.asarray(topology.lcfs_m, dtype=float)
    solved_x_point = np.asarray(topology.x_point_m, dtype=float)
    solved_diverted = np.asarray(topology.diverted, dtype=bool)

    for solve_row in np.flatnonzero(usable):
        reference_row = int(index[solve_row])
        reference_time[solve_row] = referee.time_s[reference_row]
        if _valid_points(solved_axis[solve_row]):
            axis_distance[solve_row] = np.linalg.norm(
                solved_axis[solve_row] - referee.magnetic_axis_m[reference_row]
            )
        lcfs_distance[solve_row] = _boundary_distance(
            solved_lcfs[solve_row], referee.lcfs_m[reference_row]
        )
        reference_diverted = bool(referee.diverted[reference_row])
        topology_agreement[solve_row] = float(
            bool(solved_diverted[solve_row]) == reference_diverted
        )
        candidates = referee.x_points_m[reference_row]
        candidates = candidates[_valid_points(candidates)]
        solved_x_point_valid = bool(_valid_points(solved_x_point[solve_row]))
        if not solved_x_point_valid and candidates.size:
            x_point_failure_cause[solve_row] = "reconstruction-x-point-absent"
        elif solved_x_point_valid and not candidates.size:
            x_point_failure_cause[solve_row] = "referee-x-point-absent"
        elif not solved_diverted[solve_row] and not reference_diverted:
            x_point_distance[solve_row] = 0.0
        elif solved_diverted[solve_row] and reference_diverted:
            x_point_distance[solve_row] = float(
                np.min(np.linalg.norm(candidates - solved_x_point[solve_row], axis=1))
            )

    return ReferenceGeometryScores(
        reference_index=_readonly(index, dtype=int),
        reference_time_s=_readonly(reference_time),
        usable_reference=_readonly(usable, dtype=bool),
        magnetic_axis_distance_m=_readonly(axis_distance),
        lcfs_distance_m=_readonly(lcfs_distance),
        x_point_distance_m=_readonly(x_point_distance),
        x_point_distance_failure_cause=_readonly(x_point_failure_cause, dtype=str),
        topology_class_agreement=_readonly(topology_agreement),
    )


def score_with_efit_referee(
    result: ParityChainResult,
    referee: EfitReferee,
    *,
    time_tolerance_s: float | None = None,
) -> RefereedParityResult:
    """Resolve reference metrics on an already-completed reconstruction result."""

    if int(result.scorecard.shot) != int(referee.shot):
        raise EfitRefereeError(
            f"scorecard shot {result.scorecard.shot} does not match "
            f"referee shot {referee.shot}"
        )
    geometry_scores = compare_reference_geometry(
        result.scorecard.time_s,
        result.topology,
        referee,
        time_tolerance_s=time_tolerance_s,
    )
    metrics = dict(result.scorecard.registered_metrics)
    metrics.update(geometry_scores.aggregates)
    scorecard = replace(result.scorecard, registered_metrics=metrics)
    chain = replace(result, scorecard=scorecard)
    return RefereedParityResult(chain, referee, geometry_scores)


def run_refereed_parity_chain(
    shot: int,
    *,
    referee_store: Path | str | None = None,
    referee_time_tolerance_s: float | None = None,
    **solve_arguments: Any,
) -> RefereedParityResult:
    """Run the ordinary chain, then open EFIT solely to score its completed output."""

    result = run_parity_chain(int(shot), **solve_arguments)
    store = solve_arguments.get("store", SHOT_STORE)
    referee = read_efit_referee(
        int(shot), store=store if referee_store is None else referee_store
    )
    return score_with_efit_referee(
        result, referee, time_tolerance_s=referee_time_tolerance_s
    )


EMPTY_SENTINEL = 1.0e35


def _imas_uri(source: Path | str) -> str:
    """Return an IMAS-python URI for a filesystem path or an existing URI."""

    text = str(source)
    if text.startswith("imas:"):
        return text
    return f"imas:hdf5?path={Path(text).expanduser().resolve()}"


@dataclass(frozen=True)
class EquilibriumState:
    """One reconstructed state read from an IMAS equilibrium, with its digest."""

    source: str
    dd_version: str
    time_s: float
    magnetic_axis_m: np.ndarray
    boundary_m: np.ndarray
    flux_axis: float
    flux_boundary: float
    plasma_current_a: float
    q95: float
    wall_loops: tuple[np.ndarray, ...]

    def __post_init__(self) -> None:
        """Reject a state that cannot be scored as a closed plasma boundary."""

        boundary = np.asarray(self.boundary_m, dtype=float)
        if boundary.ndim != 2 or boundary.shape[0] < 3 or boundary.shape[1] != 2:
            raise EfitRefereeError(
                "an equilibrium state must carry a boundary outline of at least "
                f"three (R, Z) points, observed shape {boundary.shape}"
            )
        axis = np.asarray(self.magnetic_axis_m, dtype=float)
        if axis.shape != (2,) or not np.all(np.isfinite(axis)):
            raise EfitRefereeError(
                f"an equilibrium state must carry one finite magnetic axis, "
                f"observed {axis}"
            )
        for name in ("flux_axis", "flux_boundary", "plasma_current_a"):
            value = float(getattr(self, name))
            if np.isfinite(value) and abs(value) >= EMPTY_SENTINEL:
                raise EfitRefereeError(
                    f"equilibrium state carries sentinel fill as {name}: {value}"
                )

    @property
    def flux_span(self) -> float:
        """Return the axis-to-boundary flux span, in the state's own units."""

        span = float(self.flux_boundary) - float(self.flux_axis)
        return abs(span) if np.isfinite(span) else float("nan")

    @property
    def magnetic_axis(self) -> np.ndarray:
        """Return the read-only axis position."""

        return _readonly(self.magnetic_axis_m)

    @property
    def boundary(self) -> np.ndarray:
        """Return the read-only closed outline."""

        return _readonly(self.boundary_m)

    def digest(self) -> str:
        """Return a sha256 over every quantity this state is scored by."""

        import hashlib

        digest = hashlib.sha256()
        digest.update(f"dd_version={self.dd_version}\n".encode())
        digest.update(np.float64(self.time_s).tobytes())
        digest.update(np.asarray(self.magnetic_axis_m, dtype=float).tobytes())
        digest.update(np.asarray(self.boundary_m, dtype=float).tobytes())
        for value in (
            self.flux_axis,
            self.flux_boundary,
            self.plasma_current_a,
            self.q95,
        ):
            digest.update(np.float64(value).tobytes())
        for loop in self.wall_loops:
            digest.update(np.asarray(loop, dtype=float).tobytes())
        return "sha256:" + digest.hexdigest()


def _equilibrium_q95(equilibrium, index: int) -> float:
    """Read the state's own safety factor at normalised flux 0.95, or nan."""

    frame = equilibrium.time_slice[index]
    try:
        profiles = frame.profiles_1d
        q = np.asarray(profiles.q, dtype=float)
        psi_norm = np.asarray(profiles.psi_norm, dtype=float)
    except AttributeError, TypeError:
        return float("nan")
    if psi_norm.size != q.size:
        psi = np.asarray(profiles.psi, dtype=float)
        if psi.size != q.size:
            return float("nan")
        span = float(psi[-1]) - float(psi[0])
        if not np.isfinite(span) or span == 0.0:
            return float("nan")
        psi_norm = (psi - float(psi[0])) / span
    if psi_norm.size < 2:
        return float("nan")
    usable = (
        np.isfinite(psi_norm)
        & np.isfinite(q)
        & (np.abs(q) < EMPTY_SENTINEL)
        & (np.abs(psi_norm) < EMPTY_SENTINEL)
    )
    if np.count_nonzero(usable) < 2:
        return float("nan")
    order = np.argsort(psi_norm[usable])
    q95 = float(np.interp(0.95, psi_norm[usable][order], q[usable][order]))
    return q95 if np.isfinite(q95) else float("nan")


def read_equilibrium_wall(source: Path | str) -> tuple[np.ndarray, ...]:
    """Read the vessel or limiter rings a machine description carries."""

    import imas

    entry = imas.DBEntry(_imas_uri(source), "r")
    try:
        wall = entry.get("wall")
    except Exception as error:  # noqa: BLE001 - imas raises a backend-specific type
        raise EfitRefereeError(
            f"no wall IDS is readable at {source}: {error}"
        ) from error
    rings: list[np.ndarray] = []
    for description in wall.description_2d:
        try:
            units = description.limiter.unit
        except AttributeError:
            continue
        for unit in units:
            radius = np.asarray(unit.outline.r, dtype=float)
            height = np.asarray(unit.outline.z, dtype=float)
            if radius.size < 3:
                continue
            points = np.column_stack([radius, height])
            points = points[_valid_points(points)]
            if points.shape[0] >= 3:
                rings.append(_readonly(points))
    if not rings:
        raise EfitRefereeError(f"the wall IDS at {source} carries no limiter ring")
    return tuple(rings)


def read_equilibrium_state(
    source: Path | str,
    *,
    occurrence: int = 0,
    time_index: int | None = None,
    time_s: float | None = None,
    wall_source: Path | str | None = None,
) -> EquilibriumState:
    """Read one reconstructed state from any IMAS equilibrium.

    The entry is opened without a data dictionary version so that imas-python
    selects the version the dataset was written at; that version is recorded on
    the returned state and named by its digest.  A frame carrying sentinel fill
    is refused rather than scored.
    """

    import imas

    uri = _imas_uri(source)
    entry = imas.DBEntry(uri, "r")
    equilibrium = entry.get("equilibrium", occurrence)
    count = len(equilibrium.time_slice)
    if count == 0:
        raise EfitRefereeError(f"the equilibrium at {source} carries no slice")
    times = np.asarray(equilibrium.time, dtype=float)
    if time_index is None:
        if time_s is not None:
            finite = np.isfinite(times) & (np.abs(times) < EMPTY_SENTINEL)
            if not np.any(finite):
                raise EfitRefereeError(
                    f"the equilibrium at {source} carries no time base"
                )
            candidates = np.flatnonzero(finite)
            time_index = int(candidates[np.argmin(np.abs(times[candidates] - time_s))])
        else:
            time_index = count - 1
    index = int(time_index)
    if not 0 <= index < count:
        raise EfitRefereeError(
            f"time index {index} lies outside the {count} slices at {source}"
        )

    frame = equilibrium.time_slice[index]
    radius = np.asarray(frame.boundary.outline.r, dtype=float)
    height = np.asarray(frame.boundary.outline.z, dtype=float)
    outline = np.column_stack([radius, height])[
        _valid_points(np.column_stack([radius, height]))
    ]
    quantities = frame.global_quantities
    axis = np.array(
        [float(quantities.magnetic_axis.r), float(quantities.magnetic_axis.z)],
        dtype=float,
    )
    wall_loops = read_equilibrium_wall(
        wall_source if wall_source is not None else source
    )

    def _scalar(value: Any) -> float:
        """Return a stored scalar, mapping sentinel fill onto an absent measure."""

        number = float(value)
        return number if abs(number) < EMPTY_SENTINEL else float("nan")

    state = EquilibriumState(
        source=str(source),
        dd_version=str(equilibrium.ids_properties.version_put.data_dictionary.value),
        time_s=float(times[index]),
        magnetic_axis_m=_readonly(axis),
        boundary_m=_readonly(outline),
        flux_axis=_scalar(quantities.psi_axis),
        flux_boundary=_scalar(quantities.psi_boundary),
        plasma_current_a=_scalar(quantities.ip),
        q95=_equilibrium_q95(equilibrium, index),
        wall_loops=wall_loops,
    )
    return state


@dataclass(frozen=True)
class StateComparison:
    """One candidate state scored against one reference state."""

    reference: EquilibriumState
    candidate: EquilibriumState
    gauge_wb: float
    magnetic_axis_distance_m: float
    lcfs_distance_m: float
    plasma_current_fraction: float
    axis_flux_fraction: float
    boundary_flux_fraction: float
    q95_difference: float

    @property
    def metrics(self) -> dict[str, float]:
        """Return the registered metrics keyed by their registry names."""

        return {
            ScorecardField.MAGNETIC_AXIS_DISTANCE_M.value: (
                self.magnetic_axis_distance_m
            ),
            "plasma_current_fraction": self.plasma_current_fraction,
            "axis_flux_fraction": self.axis_flux_fraction,
            "boundary_flux_fraction": self.boundary_flux_fraction,
            "q95_difference": self.q95_difference,
        }

    def verdicts(self) -> Mapping[str, bool]:
        """Return the registered verdicts; a metric absent from the state fails."""

        from nova.imas.parity_tolerances import state_scoreboard_verdicts

        return state_scoreboard_verdicts(self.metrics)


def compare_states(
    reference: EquilibriumState,
    candidate: EquilibriumState,
) -> StateComparison:
    """Score a candidate state against a reference with the registered bands.

    Each state's stored flux sign is read as it stands, and the boundary flux is
    compared after the one additive constant that aligns the candidate's axis
    flux onto the reference's.  Every flux measure is a fraction of the
    reference's axis-to-boundary span, so the bands are on the span rather than
    on absolute flux.
    """

    span = reference.flux_span
    gauge = float("nan")
    current = float(reference.plasma_current_a)
    flux_span_usable = np.isfinite(span) and span > 0.0
    if flux_span_usable:
        gauge = float(reference.flux_axis) - float(candidate.flux_axis)
    q95_difference = float("nan")
    if np.isfinite(reference.q95) and np.isfinite(candidate.q95):
        q95_difference = abs(abs(candidate.q95) - abs(reference.q95))
    return StateComparison(
        reference=reference,
        candidate=candidate,
        gauge_wb=gauge,
        magnetic_axis_distance_m=float(
            np.linalg.norm(
                np.asarray(candidate.magnetic_axis_m)
                - np.asarray(reference.magnetic_axis_m)
            )
        ),
        lcfs_distance_m=_boundary_distance(candidate.boundary_m, reference.boundary_m),
        plasma_current_fraction=(
            abs(candidate.plasma_current_a - current) / abs(current)
            if np.isfinite(current) and current != 0.0
            else float("nan")
        ),
        axis_flux_fraction=(
            abs(candidate.flux_axis - reference.flux_axis) / span
            if flux_span_usable
            else float("nan")
        ),
        boundary_flux_fraction=(
            abs(candidate.flux_boundary + gauge - reference.flux_boundary) / span
            if flux_span_usable
            else float("nan")
        ),
        q95_difference=q95_difference,
    )


def _polygon_area(outline: np.ndarray) -> float:
    """Return the unsigned shoelace area of one closed outline."""

    points = np.asarray(outline, dtype=float)
    radius = points[:, 0]
    height = points[:, 1]
    return float(
        abs(np.dot(radius, np.roll(height, -1)) - np.dot(height, np.roll(radius, -1)))
        * 0.5
    )


def _point_in_loops(point: np.ndarray, loops: tuple[np.ndarray, ...]) -> bool:
    """Return whether a point lies inside any of the supplied closed rings."""

    from nova.equilibrium.wall_mask import inside_polygon

    px = np.array([float(point[0])])
    py = np.array([float(point[1])])
    return any(
        bool(np.asarray(inside_polygon(px, py, loop[:, 0], loop[:, 1])).ravel()[0])
        for loop in loops
    )


@dataclass(frozen=True)
class ReferenceFreeChecks:
    """The three checks a state must pass without any reference state."""

    axis_inside_wall: bool
    axis_inside_boundary: bool
    boundary_area_fraction: float

    @property
    def values(self) -> dict[str, float | bool]:
        """Return the checks keyed by the reference-free registry names."""

        return {
            "axis_inside_wall": self.axis_inside_wall,
            "axis_inside_boundary": self.axis_inside_boundary,
            "boundary_area_fraction": self.boundary_area_fraction,
        }

    def verdicts(self) -> Mapping[str, bool]:
        """Return the registered verdicts, refusing a missing measure."""

        from nova.imas.parity_tolerances import reference_free_verdicts

        return reference_free_verdicts(self.values)


def reference_free_checks(state: EquilibriumState) -> ReferenceFreeChecks:
    """Run the reference-free geometry checks on one state."""

    outline = np.asarray(state.boundary_m, dtype=float)
    bounding_box = float((np.ptp(outline[:, 0])) * (np.ptp(outline[:, 1])))
    if not np.isfinite(bounding_box) or bounding_box <= 0.0:
        raise EfitRefereeError(
            f"the boundary at {state.source} has no positive bounding box"
        )
    axis = np.asarray(state.magnetic_axis_m, dtype=float)
    return ReferenceFreeChecks(
        axis_inside_wall=_point_in_loops(axis, state.wall_loops),
        axis_inside_boundary=bool(_point_in_loops(axis, (_readonly(outline),))),
        boundary_area_fraction=_polygon_area(outline) / bounding_box,
    )


__all__ = [
    "EFIT_GROUP",
    "EMPTY_SENTINEL",
    "EquilibriumState",
    "FROZEN_SHOTS",
    "EfitReferee",
    "EfitRefereeError",
    "ReferenceGeometryScores",
    "ReferenceFreeChecks",
    "RefereedParityResult",
    "StateComparison",
    "compare_reference_geometry",
    "compare_states",
    "read_efit_referee",
    "read_equilibrium_state",
    "reference_free_checks",
    "read_equilibrium_wall",
    "run_refereed_parity_chain",
    "score_with_efit_referee",
]
