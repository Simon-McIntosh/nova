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
from typing import Any

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
    """Per-slice comparisons and their finite shot-level reductions."""

    reference_index: np.ndarray
    reference_time_s: np.ndarray
    usable_reference: np.ndarray
    magnetic_axis_distance_m: np.ndarray
    lcfs_distance_m: np.ndarray
    x_point_distance_m: np.ndarray
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
                self.x_point_distance_m, "x-point distance"
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


def _finite_median(values: np.ndarray, name: str) -> float:
    """Reduce a finite metric or refuse to manufacture a missing score."""

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
        if not solved_diverted[solve_row] and not reference_diverted:
            x_point_distance[solve_row] = 0.0
        elif solved_diverted[solve_row] and reference_diverted:
            candidates = referee.x_points_m[reference_row]
            candidates = candidates[_valid_points(candidates)]
            if _valid_points(solved_x_point[solve_row]) and candidates.size:
                x_point_distance[solve_row] = float(
                    np.min(
                        np.linalg.norm(candidates - solved_x_point[solve_row], axis=1)
                    )
                )

    return ReferenceGeometryScores(
        reference_index=_readonly(index, dtype=int),
        reference_time_s=_readonly(reference_time),
        usable_reference=_readonly(usable, dtype=bool),
        magnetic_axis_distance_m=_readonly(axis_distance),
        lcfs_distance_m=_readonly(lcfs_distance),
        x_point_distance_m=_readonly(x_point_distance),
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


__all__ = [
    "EFIT_GROUP",
    "FROZEN_SHOTS",
    "EfitReferee",
    "EfitRefereeError",
    "ReferenceGeometryScores",
    "RefereedParityResult",
    "compare_reference_geometry",
    "read_efit_referee",
    "run_refereed_parity_chain",
    "score_with_efit_referee",
]
