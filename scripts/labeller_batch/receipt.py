#!/usr/bin/env python3
"""Aggregate forward-labeller shot manifests into a completion receipt."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np


QUALIFICATION_FIELD = "ForwardSolveReceipt.qualified"
QUALIFICATION_COMPUTATION = (
    "result.converged and equilibrium.finite.passed and "
    "terminal_residual <= policy.qualification_tolerance"
)

#: Fixed width of the absolute time-in-shot conditioned-fraction bins.
_TIME_BIN_SECONDS = 0.050

#: Session-frame fields that, when present, carry the per-slice topology
#: class the topology-pinned read requested.
_FRAME_CLASS_FIELDS = ("topology_class", "requested_class", "class")

#: Run-level verdict on whether the session-frame layout carries a class
#: channel.  The layout is written once per corpus, so the first session that
#: either decodes without a class field or fails to decode settles it and
#: later sessions are not re-read.
_SESSION_SCHEMA: dict[str, str] = {}


def _fraction(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _finite_value(value: Any) -> bool:
    """Return whether an optional scalar is finite."""
    return value is not None and math.isfinite(float(value))


def _unqualified_reason(row: dict[str, Any], tolerance: float) -> str:
    """Return one mutually exclusive explanation for an unqualified slice."""
    if row.get("exception"):
        return "slice_exception"
    if not row.get("converged"):
        return "not_converged"
    residual = float(row.get("terminal_residual", math.nan))
    if not math.isfinite(residual):
        return "non_finite_terminal_residual"
    if residual > tolerance:
        return "residual_above_qualification_tolerance"
    return "receipt_finite_check_failed"


def _distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    """Return the requested robust summary of a finite sample."""
    finite = np.asarray([value for value in values if math.isfinite(value)])
    if not finite.size:
        return {"count": 0, "median": None, "p90": None, "maximum": None}
    return {
        "count": int(finite.size),
        "median": float(np.median(finite)),
        "p90": float(np.percentile(finite, 90)),
        "maximum": float(np.max(finite)),
    }


def _load_companion(path: Path) -> dict[str, np.ndarray] | None:
    """Return the per-slice companion arrays of one shot, or None."""
    try:
        with np.load(str(path)) as store:
            return {name: store[name] for name in store.files}
    except OSError, ValueError:
        return None


def _companion_rows(item: dict[str, Any]) -> dict[str, np.ndarray | None] | None:
    """Return the diagnostic rows that are also written manifest slices.

    The companion npz carries one record per admitted slice; the result is
    restricted to rows the manifest also records as written so the table
    populations cannot drift from the receipt's written-slice accounting.
    """
    companion = item.get("companion")
    if not companion:
        return None
    path = Path(companion)
    if not path.is_file():
        return None
    data = _load_companion(path)
    if data is None:
        return None
    for key in ("row", "time", "conditioned"):
        if key not in data:
            return None
    rows = np.asarray(data["row"]).astype(np.int64)
    written = np.asarray(
        sorted(
            int(record["row"])
            for record in item.get("slices", ())
            if record.get("written") and record.get("row") is not None
        ),
        dtype=np.int64,
    )
    keep = np.isin(rows, written) & np.isfinite(np.asarray(data["time"]))
    if not keep.any():
        return None
    free_error = data.get("free_centroid_error_m")
    conditioned_error = data.get("conditioned_centroid_error_m")
    return {
        "row": rows[keep],
        "time": np.asarray(data["time"], dtype=float)[keep],
        "conditioned": np.asarray(data["conditioned"], dtype=bool)[keep],
        "free_error_m": (
            np.asarray(free_error, dtype=float)[keep]
            if free_error is not None
            else None
        ),
        "conditioned_error_m": (
            np.asarray(conditioned_error, dtype=float)[keep]
            if conditioned_error is not None
            else None
        ),
    }


def _time_in_shot(times: np.ndarray) -> np.ndarray:
    """Return seconds since the shot's first written slice."""
    return np.asarray(times, dtype=float) - float(np.min(times))


def _decile_labels(times: np.ndarray) -> np.ndarray:
    """Map one shot's written-slice times onto deciles of its own span."""
    offset = _time_in_shot(times)
    span = float(np.max(offset))
    if span <= 0.0:
        return np.zeros(offset.size, dtype=int)
    normalised = np.clip(offset / span, 0.0, 1.0)
    return np.minimum(9, np.floor(10.0 * normalised).astype(int))


def _time_bin_labels(times: np.ndarray) -> np.ndarray:
    """Map time-in-shot onto fixed 50 ms bin indices."""
    offset = _time_in_shot(times)
    return np.floor(offset / _TIME_BIN_SECONDS).astype(int)


def _classes_from_session(session_path: Path) -> list[int] | None:
    """Per-frame topology classes from a session, or None when absent."""
    if _SESSION_SCHEMA.get("verdict") == "no_class_field":
        return None
    try:
        from nova.equilibrium.steering_frames import (
            frames_from_session,
            read_session,
        )
    except ImportError:
        _SESSION_SCHEMA["verdict"] = "no_class_field"
        return None
    try:
        dataset = read_session(
            filename=session_path.name, dirname=str(session_path.parent)
        )
        frames = frames_from_session(dataset)
    except Exception:
        _SESSION_SCHEMA["verdict"] = "no_class_field"
        return None
    for field in _FRAME_CLASS_FIELDS:
        if any(getattr(frame, field, None) is not None for frame in frames):
            _SESSION_SCHEMA["verdict"] = "has_class_field"
            return [int(getattr(frame, field)) for frame in frames]
    _SESSION_SCHEMA["verdict"] = "no_class_field"
    return None


def _classes_from_manifest(item: dict[str, Any]) -> list[int] | None:
    """Per-written-slice classes from the manifest records, or None."""
    written = [
        record
        for record in item.get("slices", ())
        if record.get("written") and record.get("row") is not None
    ]
    if not written:
        return None
    values = [record.get("requested_class") for record in written]
    if any(value is None for value in values):
        return None
    return [int(value) for value in values]


_TOPOLOGY_NAMES: dict[int, str] | None = None


def _class_label(value: int) -> str:
    """Name a topology-class integer, the decimal form when unknown."""
    global _TOPOLOGY_NAMES
    if _TOPOLOGY_NAMES is None:
        try:
            from nova.equilibrium.topology import TopologyClass
        except Exception:
            _TOPOLOGY_NAMES = {0: "limited", 1: "diverted"}
        else:
            _TOPOLOGY_NAMES = {
                int(member): member.name.lower() for member in TopologyClass
            }
    return _TOPOLOGY_NAMES.get(int(value), str(int(value)))


def _resolve_class_source(complete: Sequence[dict[str, Any]]) -> tuple[str, str]:
    """Decide the corpus-wide topology-class source and an absent reason.

    Prefers a class channel on the session frames read through
    ``steering_frames``, then ``requested_class`` on the manifest slice
    records.  The session layout is written once per corpus, so the first
    shot's verdict settles the frame side.
    """
    for item in complete:
        session = item.get("session")
        if session and Path(session).is_file():
            if _classes_from_session(Path(session)) is not None:
                return "session_frame", ""
            break
    for item in complete:
        if any(
            record.get("requested_class") is not None
            for record in item.get("slices", ())
            if record.get("written")
        ):
            return "manifest_requested_class", ""
    return (
        "unavailable",
        "the session frames carry no decodable topology-class field and the "
        "manifest slice records carry no requested_class, so the split cannot "
        "be structured by topology class under this layout",
    )


def _shot_class_labels(
    rows: np.ndarray,
    item: dict[str, Any],
    source: str,
) -> np.ndarray | None:
    """Per-companion-row class labels for one shot, or None when unclassed."""
    if source == "unavailable":
        return None
    written = [
        record
        for record in item.get("slices", ())
        if record.get("written") and record.get("row") is not None
    ]
    if source == "manifest_requested_class":
        values = [record.get("requested_class") for record in written]
    else:
        session = item.get("session")
        classes = _classes_from_session(Path(session)) if session else None
        if classes is None or len(classes) != len(written):
            return None
        values = classes
    if any(value is None for value in values):
        return None
    labels_by_row = {
        int(record["row"]): _class_label(int(value))
        for record, value in zip(written, values, strict=True)
    }
    return np.asarray([labels_by_row.get(int(row)) for row in rows], dtype=object)


def _conditioned_bins(
    labels: np.ndarray,
    conditioned: np.ndarray,
    label_order: Sequence[int] | None = None,
) -> list[tuple[int, int, int]]:
    """Return ``(label, slices, conditioned)`` per bin, label-sorted."""
    order = (
        list(label_order) if label_order is not None else sorted(set(labels.tolist()))
    )
    totals: list[tuple[int, int, int]] = []
    for label in order:
        mask = labels == label
        totals.append(
            (
                label,
                int(np.count_nonzero(mask)),
                int(np.count_nonzero(conditioned & mask)),
            )
        )
    return totals


def _adjudication_tables(complete: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Return the two imas-ambix adjudication tables over the diagnostics npz."""
    class_source, class_reason = _resolve_class_source(complete)
    decile_pool: list[np.ndarray] = []
    bin_pool: list[np.ndarray] = []
    conditioned_pool: list[np.ndarray] = []
    class_buckets: dict[str, list[bool]] = {}
    displacements: list[float] = []
    displacement_by_class: dict[str, list[float]] = {}
    companion_shots = 0
    classed_shots = 0
    for item in complete:
        rows = _companion_rows(item)
        if rows is None:
            continue
        companion_shots += 1
        time = np.asarray(rows["time"])
        conditioned = np.asarray(rows["conditioned"], dtype=bool)
        decile_pool.append(_decile_labels(time))
        bin_pool.append(_time_bin_labels(time))
        conditioned_pool.append(conditioned)
        row_numbers = np.asarray(rows["row"], dtype=np.int64)
        labels = _shot_class_labels(row_numbers, item, class_source)
        if labels is not None:
            classed_shots += 1
            for label, flag in zip(labels.tolist(), conditioned.tolist(), strict=True):
                if label is not None:
                    class_buckets.setdefault(str(label), []).append(bool(flag))
        free_error = rows["free_error_m"]
        conditioned_error = rows["conditioned_error_m"]
        if free_error is None or conditioned_error is None:
            continue
        both = np.isfinite(free_error) & np.isfinite(conditioned_error)
        if not both.any():
            continue
        values = 1000.0 * (conditioned_error[both] - free_error[both])
        displacements.extend(float(value) for value in values)
        if labels is not None:
            for value, label in zip(
                values.tolist(), labels[both].tolist(), strict=True
            ):
                if label is not None:
                    displacement_by_class.setdefault(str(label), []).append(value)
    deciles = np.concatenate(decile_pool) if decile_pool else np.asarray([], dtype=int)
    time_bins = np.concatenate(bin_pool) if bin_pool else np.asarray([], dtype=int)
    conditioned_all = (
        np.concatenate(conditioned_pool)
        if conditioned_pool
        else np.asarray([], dtype=bool)
    )
    table: dict[str, Any] = {
        "conditioned_by_time_in_shot": {
            "coordinate": "seconds since the shot's first written slice",
            "deciles": {
                "definition": (
                    "decile of each shot's own written-slice time span, "
                    "pooled across shots"
                ),
                "bins": [
                    {
                        "decile": label,
                        "slices": count,
                        "conditioned": conditioned_count,
                        "fraction": _fraction(conditioned_count, count),
                    }
                    for label, count, conditioned_count in _conditioned_bins(
                        deciles, conditioned_all, label_order=range(10)
                    )
                ],
            },
            "absolute_50ms": {
                "bin_width_ms": int(_TIME_BIN_SECONDS * 1000.0),
                "bins": [
                    {
                        "bin_ms": int(round(label * _TIME_BIN_SECONDS * 1000.0)),
                        "slices": count,
                        "conditioned": conditioned_count,
                        "fraction": _fraction(conditioned_count, count),
                    }
                    for label, count, conditioned_count in _conditioned_bins(
                        time_bins, conditioned_all
                    )
                ],
            },
        },
        "conditioned_by_topology_class": {
            "source": class_source,
            "classes": {
                label: {
                    "slices": len(flags),
                    "conditioned": sum(flags),
                    "fraction": _fraction(sum(flags), len(flags)),
                }
                for label, flags in sorted(class_buckets.items())
            },
        },
        "pin_displacement_mm": {
            "definition": (
                "1000 * (conditioned_centroid_error_m - free_centroid_error_m) "
                "over written slices with both finite"
            ),
            "unit": "mm",
            "overall": _distribution(displacements),
            "by_topology_class": {
                label: _distribution(values)
                for label, values in sorted(displacement_by_class.items())
            },
        },
    }
    if class_source == "unavailable":
        table["conditioned_by_topology_class"]["reason"] = class_reason
    table["conditioned_by_time_in_shot"]["shot_coverage"] = {
        "companion_shots": companion_shots,
        "classed_shots": classed_shots,
    }
    return table


def _shot_metrics(item: dict[str, Any]) -> dict[str, Any]:
    """Return complete per-shot quality, branch and throughput evidence."""
    rows = [row for row in item.get("slices", ()) if row.get("written")]
    excluded = [row for row in item.get("slices", ()) if row.get("excluded")]
    tolerance = float(item.get("policy", {}).get("qualification_tolerance", math.nan))
    reasons: dict[str, int] = {}
    for row in rows:
        if row.get("qualified"):
            continue
        reason = _unqualified_reason(row, tolerance)
        reasons[reason] = reasons.get(reason, 0) + 1
    differences = [
        1000.0
        * (
            float(row.get("achieved_current_centroid_z", math.nan))
            - float(row.get("target_current_centroid_z", math.nan))
        )
        for row in rows
        if row.get("achieved_current_centroid_z") is not None
        and row.get("target_current_centroid_z") is not None
    ]
    slice_wall_seconds = sum(float(row.get("wall_seconds", 0.0)) for row in rows)
    conditioned = [row for row in rows if row.get("conditioned")]
    free_guarded = [row for row in rows if row.get("free_branch_guard_ok") is not None]
    conditioned_guarded = [
        row for row in conditioned if row.get("conditioned_branch_guard_ok") is not None
    ]
    return {
        "shot": int(item["shot"]),
        "admitted_slices": len(rows),
        "excluded_slices": len(excluded),
        "converged_slices": sum(bool(row.get("converged")) for row in rows),
        "unconverged_slices": sum(not bool(row.get("converged")) for row in rows),
        "qualified_slices": sum(bool(row.get("qualified")) for row in rows),
        "unqualified_slices": sum(not bool(row.get("qualified")) for row in rows),
        "qualification": {
            "field": QUALIFICATION_FIELD,
            "computation": QUALIFICATION_COMPUTATION,
            "tolerance": tolerance,
        },
        "unqualified_by_reason": dict(sorted(reasons.items())),
        "branch_guard_agreement_slices": sum(
            bool(row.get("branch_guard_ok")) for row in rows
        ),
        "free_guard_evaluated_slices": len(free_guarded),
        "free_guard_agreement_slices": sum(
            bool(row.get("free_branch_guard_ok")) for row in free_guarded
        ),
        "free_slices": len(rows) - len(conditioned),
        "conditioned_slices": len(conditioned),
        "conditioned_guard_evaluated_slices": len(conditioned_guarded),
        "conditioned_guard_agreement_slices": sum(
            bool(row.get("conditioned_branch_guard_ok")) for row in conditioned_guarded
        ),
        "exception_slices": sum(bool(row.get("exception")) for row in rows),
        "vertical_centroid_difference_mm": {
            "definition": (
                "1000 * (achieved_current_centroid_z - efm/current_centrd_z)"
            ),
            "signed": _distribution(differences),
            "absolute": _distribution([abs(value) for value in differences]),
        },
        "slice_wall_seconds": slice_wall_seconds,
        "mean_slice_wall_seconds": _fraction(slice_wall_seconds, len(rows)),
        "slices_per_second_per_card": _fraction(len(rows), slice_wall_seconds),
        "warm_rate_eligible": float(item.get("setup_wall_seconds", 0.0)) == 0.0,
    }


def aggregate(root: Path, *, expected_shots: int | None = None) -> dict[str, Any]:
    """Return completion and throughput metrics for one sessions directory."""
    _SESSION_SCHEMA.clear()
    manifests = []
    for path in sorted(root.glob("*.manifest.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        record["_path"] = str(path.resolve())
        manifests.append(record)
    complete = [item for item in manifests if item.get("status") == "complete"]
    failed = [item for item in manifests if item.get("status") != "complete"]
    slices = [
        row for item in complete for row in item.get("slices", ()) if row.get("written")
    ]
    excluded = [
        row
        for item in complete
        for row in item.get("slices", ())
        if row.get("excluded")
    ]
    converged = sum(bool(row.get("converged")) for row in slices)
    unconverged = len(slices) - converged
    qualified = sum(bool(row.get("qualified")) for row in slices)
    guarded = [row for row in slices if row.get("free_branch_guard_ok") is not None]
    guard_agreement = sum(bool(row.get("branch_guard_ok")) for row in guarded)
    conditioned = [row for row in slices if row.get("conditioned")]
    conditioned_guarded = [
        row for row in conditioned if row.get("conditioned_branch_guard_ok") is not None
    ]
    centroid_slices = sum(
        _finite_value(row.get("achieved_current_centroid_r"))
        and _finite_value(row.get("achieved_current_centroid_z"))
        for row in slices
    )
    companion_records = [
        item
        for item in complete
        if item.get("companion") and Path(item["companion"]).is_file()
    ]
    companion_slices = sum(
        int(item.get("companion_slice_count", 0)) for item in companion_records
    )
    shot_metrics = [_shot_metrics(item) for item in complete]
    adjudication = _adjudication_tables(complete)
    warm_shots = [item for item in shot_metrics if item["warm_rate_eligible"]]
    warm_slices = sum(int(item["admitted_slices"]) for item in warm_shots)
    warm_seconds = sum(float(item["slice_wall_seconds"]) for item in warm_shots)
    gpu_seconds = sum(
        float(item.get("setup_wall_seconds", 0.0))
        + float(item.get("shot_wall_seconds", 0.0))
        for item in complete
    )
    expected = len(manifests) if expected_shots is None else int(expected_shots)
    return {
        "schema": "nova-forward-labeller-completion",
        "root": str(root.resolve()),
        "expected_shots": expected,
        "manifest_count": len(manifests),
        "completed_shots": len(complete),
        "failed_shots": len(failed),
        "completion_fraction": _fraction(len(complete), expected),
        "slices": len(slices),
        "admitted_slices": len(slices),
        "excluded_slices": len(excluded),
        "converged_slices": converged,
        "unconverged_slices": unconverged,
        "qualified_slices": qualified,
        "converged_fraction": _fraction(converged, len(slices)),
        "qualified_fraction": _fraction(qualified, len(slices)),
        "branch_guard_evaluated_slices": len(guarded),
        "branch_guard_agreement_slices": guard_agreement,
        "branch_guard_agreement_fraction": _fraction(guard_agreement, len(guarded)),
        "free_guard_agreement_slices": sum(
            bool(row.get("free_branch_guard_ok")) for row in guarded
        ),
        "free_guard_agreement_fraction": _fraction(
            sum(bool(row.get("free_branch_guard_ok")) for row in guarded),
            len(guarded),
        ),
        "conditioned_slices": len(conditioned),
        "conditioned_guard_evaluated_slices": len(conditioned_guarded),
        "conditioned_guard_agreement_slices": sum(
            bool(row.get("conditioned_branch_guard_ok")) for row in conditioned_guarded
        ),
        "conditioned_guard_agreement_fraction": _fraction(
            sum(
                bool(row.get("conditioned_branch_guard_ok"))
                for row in conditioned_guarded
            ),
            len(conditioned_guarded),
        ),
        "exception_slices": sum(bool(row.get("exception")) for row in slices),
        "current_centroid_slices": centroid_slices,
        "companion_shots": len(companion_records),
        "companion_slices": companion_slices,
        "companion_completion_fraction": _fraction(
            len(companion_records), len(complete)
        ),
        "flux_function_grid_points": sorted(
            {
                int(item["flux_function_grid_points"])
                for item in companion_records
                if item.get("flux_function_grid_points") is not None
            }
        ),
        "slices_per_second_per_card": _fraction(len(slices), gpu_seconds),
        "warm_rate_definition": (
            "admitted slices in shots after the setup and compilation-bearing "
            "first shot of each shard, divided by their summed per-slice wall time"
        ),
        "warm_shots": len(warm_shots),
        "warm_slices": warm_slices,
        "warm_wall_seconds": warm_seconds,
        "warm_slices_per_second_per_card": _fraction(warm_slices, warm_seconds),
        "gpu_hours": gpu_seconds / 3600.0,
        "nova_revisions": sorted({str(item.get("nova_revision")) for item in complete}),
        "carrier_identities": sorted(
            {str(item.get("carrier_identity")) for item in complete}
        ),
        "policy_digests": sorted({str(item.get("policy_digest")) for item in complete}),
        **adjudication,
        "shots": shot_metrics,
        "failed": [
            {
                "shot": item.get("shot"),
                "failure": item.get("failure"),
                "traceback": item.get("traceback"),
                "manifest": item["_path"],
            }
            for item in failed
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-shots", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    receipt = aggregate(arguments.root, expected_shots=arguments.expected_shots)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True))
    return 0 if receipt["failed_shots"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
