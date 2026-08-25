"""Registered terminal-observable acceptance for one throughput batch.

The acceptance compares a scalar-route reference with the labels emitted by
one measured batch.  Every registered observable remains a separate verdict;
an aggregate pass can therefore never hide which label or held-out case moved.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def _finite_difference(
    reference: np.ndarray, candidate: np.ndarray
) -> tuple[float, float]:
    """Return absolute and reference-scaled differences for one batch member."""

    finite = np.isfinite(reference) & np.isfinite(candidate)
    if not np.array_equal(np.isnan(reference), np.isnan(candidate)):
        return float("inf"), float("inf")
    if not np.any(finite):
        if np.array_equal(reference, candidate, equal_nan=True):
            return 0.0, 0.0
        return float("inf"), float("inf")
    absolute = float(
        np.max(
            np.abs(
                candidate[finite].astype(np.float64)
                - reference[finite].astype(np.float64)
            )
        )
    )
    scale = max(float(np.max(np.abs(reference[finite]))), np.finfo(np.float64).tiny)
    return absolute, absolute / scale


def _bound_ratio(value: float, bound: float) -> float:
    """Return envelope utilisation while retaining exact-zero envelopes."""

    if bound > 0.0:
        return value / bound
    return 0.0 if value == 0.0 else float("inf")


def _validate_registration(
    registration: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    """Return the unique observable registration indexed by name."""

    by_name = {str(row["observable"]): row for row in registration}
    if len(by_name) != len(registration):
        raise ValueError("observable registration contains duplicate names")
    for name, row in by_name.items():
        kind = row.get("criterion_kind")
        if kind not in {
            "exact_equality",
            "banked_absolute_envelope",
            "derived_absolute_envelope",
            "banked_dual_envelope",
        }:
            raise ValueError(f"{name} has unsupported criterion kind {kind!r}")
        if (
            kind == "banked_dual_envelope"
            and not {
                "absolute_bound",
                "relative_bound",
            }
            <= row.keys()
        ):
            raise ValueError(f"{name} has an incomplete dual envelope")
        if (
            row.get("has_nonzero_continuum_value") is False
            and "absolute_bound" in row
            and "relative_bound" in row
        ):
            raise ValueError(
                f"{name} has no nonzero continuum value and cannot carry a "
                "relative criterion alongside its absolute bound"
            )
        if (
            kind
            in {
                "banked_absolute_envelope",
                "derived_absolute_envelope",
            }
            and "absolute_bound" not in row
        ):
            raise ValueError(f"{name} has an incomplete absolute envelope")
    return by_name


def _observable_result(
    name: str,
    reference: Any,
    candidate: Any,
    registration: Mapping[str, Any],
    case_ids: Sequence[str],
    batch_size: int,
) -> dict[str, Any]:
    """Score one registered observable over every case and batch member."""

    left = np.asarray(reference)
    right = np.asarray(candidate)
    declared_shape = tuple(int(size) for size in registration["shape"])
    expected_shape = (len(case_ids), batch_size, *declared_shape)
    if left.shape != expected_shape or right.shape != expected_shape:
        raise ValueError(
            f"{name} requires reference and candidate shape {expected_shape}; "
            f"received {left.shape} and {right.shape}"
        )
    if left.dtype != right.dtype:
        raise ValueError(
            f"{name} changes dtype from {left.dtype.name} to {right.dtype.name}"
        )
    expected_dtype = np.dtype(str(registration["dtype"]))
    if left.dtype != expected_dtype:
        raise ValueError(
            f"{name} requires dtype {expected_dtype.name}; received {left.dtype.name}"
        )

    flattened_left = left.reshape(len(case_ids), batch_size, -1)
    flattened_right = right.reshape(len(case_ids), batch_size, -1)
    case_rows = []
    maximum_absolute = 0.0
    maximum_relative = 0.0
    maximum_ratio = 0.0
    for case_index, case_id in enumerate(case_ids):
        member_rows = []
        for member_index in range(batch_size):
            member_left = flattened_left[case_index, member_index]
            member_right = flattened_right[case_index, member_index]
            absolute, relative = _finite_difference(member_left, member_right)
            if registration["criterion_kind"] == "exact_equality":
                passes = bool(np.array_equal(member_left, member_right, equal_nan=True))
                ratio = 0.0 if passes else float("inf")
            elif registration["criterion_kind"] in {
                "banked_absolute_envelope",
                "derived_absolute_envelope",
            }:
                absolute_bound = float(registration["absolute_bound"])
                ratio = _bound_ratio(absolute, absolute_bound)
                passes = bool(absolute <= absolute_bound)
            else:
                absolute_bound = float(registration["absolute_bound"])
                relative_bound = float(registration["relative_bound"])
                absolute_ratio = _bound_ratio(absolute, absolute_bound)
                relative_ratio = _bound_ratio(relative, relative_bound)
                ratio = max(absolute_ratio, relative_ratio)
                passes = bool(absolute <= absolute_bound and relative <= relative_bound)
            maximum_absolute = max(maximum_absolute, absolute)
            maximum_relative = max(maximum_relative, relative)
            maximum_ratio = max(maximum_ratio, ratio)
            member_rows.append(
                {
                    "member_index": member_index,
                    "passes": passes,
                    "maximum_absolute_difference": absolute,
                    "maximum_relative_difference": relative,
                    "maximum_bound_ratio": ratio if np.isfinite(ratio) else None,
                }
            )
        case_rows.append(
            {
                "case_id": case_id,
                "passes": all(row["passes"] for row in member_rows),
                "members": member_rows,
            }
        )

    result = {
        "observable": name,
        "criterion_kind": registration["criterion_kind"],
        "passes": all(row["passes"] for row in case_rows),
        "case_pass_count": sum(row["passes"] for row in case_rows),
        "case_fail_count": sum(not row["passes"] for row in case_rows),
        "member_pass_count": sum(
            member["passes"] for row in case_rows for member in row["members"]
        ),
        "member_fail_count": sum(
            not member["passes"] for row in case_rows for member in row["members"]
        ),
        "maximum_absolute_difference": maximum_absolute,
        "maximum_relative_difference": maximum_relative,
        "maximum_bound_ratio": maximum_ratio if np.isfinite(maximum_ratio) else None,
        "cases": case_rows,
    }
    if registration["criterion_kind"] in {
        "banked_absolute_envelope",
        "derived_absolute_envelope",
    }:
        result.update(absolute_bound=float(registration["absolute_bound"]))
    elif registration["criterion_kind"] == "banked_dual_envelope":
        result.update(
            absolute_bound=float(registration["absolute_bound"]),
            relative_bound=float(registration["relative_bound"]),
        )
    return result


def evaluate_observable_bound_acceptance(
    *,
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    registration: Sequence[Mapping[str, Any]],
    case_ids: Sequence[str],
    batch_size: int,
) -> dict[str, Any]:
    """Apply every registered bound to one measured throughput batch.

    Values carry leading ``(case, batch_member)`` axes followed by each
    observable's registered shape.  A registered bound passes only when every
    batch member passes on every case, while the receipt retains both the
    six-case count and the expanded member count.
    """

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    identities = tuple(str(case_id) for case_id in case_ids)
    if not identities or len(set(identities)) != len(identities):
        raise ValueError("case_ids must be non-empty and unique")
    by_name = _validate_registration(registration)
    missing_reference = sorted(by_name.keys() - reference.keys())
    missing_candidate = sorted(by_name.keys() - candidate.keys())
    if missing_reference or missing_candidate:
        raise ValueError(
            "registered observables are absent: "
            f"reference={missing_reference}, candidate={missing_candidate}"
        )

    rows = [
        _observable_result(
            name,
            reference[name],
            candidate[name],
            by_name[name],
            identities,
            batch_size,
        )
        for name in sorted(by_name)
    ]
    observable_pass_count = sum(row["passes"] for row in rows)
    case_pass_count = sum(row["case_pass_count"] for row in rows)
    member_pass_count = sum(row["member_pass_count"] for row in rows)
    return {
        "acceptance_entry_point": (
            "nova.equilibrium.observable_acceptance."
            "evaluate_observable_bound_acceptance"
        ),
        "batch_size": batch_size,
        "case_count": len(identities),
        "case_ids": list(identities),
        "registered_bound_count": len(rows),
        "observable_pass_count": observable_pass_count,
        "observable_fail_count": len(rows) - observable_pass_count,
        "case_observable_evaluation_pass_count": case_pass_count,
        "case_observable_evaluation_fail_count": len(rows) * len(identities)
        - case_pass_count,
        "member_observable_evaluation_pass_count": member_pass_count,
        "member_observable_evaluation_fail_count": (
            len(rows) * len(identities) * batch_size - member_pass_count
        ),
        "passes": observable_pass_count == len(rows),
        "per_observable": rows,
    }


__all__ = ["evaluate_observable_bound_acceptance"]
