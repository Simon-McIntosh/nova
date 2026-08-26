"""Host-only comparison of closed plasma boundaries in physical coordinates."""

from dataclasses import dataclass

import numpy as np

from nova.equilibrium.topology import BoundaryMode


__all__ = [
    "BoundaryMode",
    "ClosedBoundaryComparison",
    "classify_boundary_mode",
    "compare_closed_boundaries",
]


@dataclass(frozen=True)
class ClosedBoundaryComparison:
    """Fail-closed topology and geometry comparison for one boundary pair."""

    achieved_mode: BoundaryMode | None
    reference_mode: BoundaryMode | None
    topology_class_agreement: bool | None
    symmetric_sup_distance_m: float | None
    symmetric_rms_distance_m: float | None
    x_point_distance_m: float | None
    failures: tuple[str, ...]


def classify_boundary_mode(class_margin: float) -> BoundaryMode:
    """Classify an achieved topology from its saddle-aware signed margin."""

    if class_margin is None:
        raise ValueError("class margin is absent")
    try:
        margin = float(class_margin)
    except (TypeError, ValueError) as error:
        raise ValueError("class margin must be a scalar") from error
    if np.isnan(margin):
        raise ValueError("class margin is indeterminate")
    return BoundaryMode.DIVERTED if margin >= 0.0 else BoundaryMode.LIMITED


def _closed_polyline(points: object) -> np.ndarray | None:
    """Return a finite, explicitly closed polyline or fail closed."""

    if points is None:
        return None
    try:
        polyline = np.asarray(points, dtype=float)
    except TypeError, ValueError:
        return None
    if polyline.ndim != 2 or polyline.shape[1] != 2 or not np.isfinite(polyline).all():
        return None

    if len(polyline) > 1 and np.array_equal(polyline[0], polyline[-1]):
        polyline = polyline[:-1]
    if len(polyline) < 3 or len(np.unique(polyline, axis=0)) < 3:
        return None

    segment_lengths = np.linalg.norm(np.roll(polyline, -1, axis=0) - polyline, axis=1)
    retained = segment_lengths > np.finfo(float).eps
    polyline = polyline[retained]
    if len(polyline) < 3 or len(np.unique(polyline, axis=0)) < 3:
        return None
    return np.vstack((polyline, polyline[0]))


def _sample_closed_polyline(polyline: np.ndarray, count: int) -> np.ndarray:
    """Sample an explicitly closed polyline uniformly in arc length."""

    lengths = np.linalg.norm(np.diff(polyline, axis=0), axis=1)
    distance = np.concatenate(([0.0], np.cumsum(lengths)))
    query = np.linspace(0.0, distance[-1], count, endpoint=False)
    return np.column_stack(
        tuple(np.interp(query, distance, polyline[:, axis]) for axis in range(2))
    )


def _point_to_segment_distances(points: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    """Return each point's distance to the nearest polyline segment."""

    starts = polyline[:-1]
    vectors = polyline[1:] - starts
    squared_lengths = np.einsum("ij,ij->i", vectors, vectors)
    result = np.empty(len(points), dtype=float)
    chunk_size = 256
    for start in range(0, len(points), chunk_size):
        chunk = points[start : start + chunk_size]
        relative = chunk[:, None, :] - starts[None, :, :]
        fraction = np.einsum("csi,si->cs", relative, vectors) / squared_lengths
        fraction = np.clip(fraction, 0.0, 1.0)
        offset = relative - fraction[:, :, None] * vectors[None, :, :]
        squared_distance = np.einsum("csi,csi->cs", offset, offset)
        result[start : start + len(chunk)] = np.sqrt(np.min(squared_distance, axis=1))
    return result


def _reference_boundary_mode(value: object) -> BoundaryMode | None:
    """Normalize a reference topology class without guessing an absent class."""

    if value is None:
        return None
    try:
        return BoundaryMode(value)
    except TypeError, ValueError:
        return None


def _finite_point(point: object) -> np.ndarray | None:
    """Return one finite physical point or fail closed."""

    if point is None:
        return None
    try:
        result = np.asarray(point, dtype=float)
    except TypeError, ValueError:
        return None
    if result.shape != (2,) or not np.isfinite(result).all():
        return None
    return result


def _finite_point_set(points: object) -> np.ndarray | None:
    """Return the finite members of an unordered physical point set."""

    if points is None:
        return None
    try:
        result = np.asarray(points, dtype=float)
    except TypeError, ValueError:
        return None
    if result.ndim != 2 or result.shape[1] != 2:
        return None
    result = result[np.isfinite(result).all(axis=1)]
    return result if len(result) else None


def compare_closed_boundaries(
    predicted_closed_rz_m: object,
    reference_closed_rz_m: object,
    *,
    class_margin: float | None,
    reference_mode: BoundaryMode | str | None,
    predicted_saddle_rz_m: object,
    reference_x_points_rz_m: object,
    sample_count: int = 2000,
) -> ClosedBoundaryComparison:
    """Compare closed boundaries, achieved topology, and unordered X points.

    Boundary distances use fixed arc-length populations and the opposing
    polyline's segments. Inputs that cannot support a metric produce a stable
    failure name and ``None`` rather than a non-finite numeric result.
    """

    if isinstance(sample_count, bool) or not isinstance(sample_count, int):
        raise TypeError("sample_count must be an integer")
    if sample_count < 3:
        raise ValueError("sample_count must be at least three")

    failures: list[str] = []
    predicted = _closed_polyline(predicted_closed_rz_m)
    reference = _closed_polyline(reference_closed_rz_m)
    if predicted is None:
        failures.append("missing_predicted_closed_boundary")
    if reference is None:
        failures.append("missing_reference_closed_boundary")

    try:
        achieved_mode = classify_boundary_mode(class_margin)
    except ValueError:
        achieved_mode = None
        failures.append("missing_achieved_topology_class")
    normalized_reference_mode = _reference_boundary_mode(reference_mode)
    if normalized_reference_mode is None:
        failures.append("missing_reference_topology_class")

    saddle = _finite_point(predicted_saddle_rz_m)
    if saddle is None:
        failures.append("missing_predicted_saddle")
    reference_x_points = _finite_point_set(reference_x_points_rz_m)
    if reference_x_points is None:
        failures.append("missing_reference_x_points")

    sup_distance = None
    rms_distance = None
    if predicted is not None and reference is not None:
        predicted_sample = _sample_closed_polyline(predicted, sample_count)
        reference_sample = _sample_closed_polyline(reference, sample_count)
        predicted_sample = np.vstack((predicted_sample, predicted_sample[0]))
        reference_sample = np.vstack((reference_sample, reference_sample[0]))
        predicted_to_reference = _point_to_segment_distances(
            predicted_sample[:-1], reference_sample
        )
        reference_to_predicted = _point_to_segment_distances(
            reference_sample[:-1], predicted_sample
        )
        distances = np.concatenate((predicted_to_reference, reference_to_predicted))
        sup_distance = float(np.max(distances))
        rms_distance = float(np.sqrt(np.mean(distances**2)))

    topology_agreement = (
        achieved_mode is normalized_reference_mode
        if achieved_mode is not None and normalized_reference_mode is not None
        else None
    )
    x_distance = (
        float(np.min(np.linalg.norm(reference_x_points - saddle, axis=1)))
        if saddle is not None and reference_x_points is not None
        else None
    )
    return ClosedBoundaryComparison(
        achieved_mode=achieved_mode,
        reference_mode=normalized_reference_mode,
        topology_class_agreement=topology_agreement,
        symmetric_sup_distance_m=sup_distance,
        symmetric_rms_distance_m=rms_distance,
        x_point_distance_m=x_distance,
        failures=tuple(failures),
    )
