"""Price margin-armed branching from banked MAST catalog-frame margins."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parents[1]
DEFAULT_MARGIN_SOURCE = (
    HERE / "docs/figures/dual-branch-selection/pinned-branch-contrast.json"
)
DEFAULT_CENSUS_SOURCE = HERE / "docs/figures/mast-catalog-gpu-solve/slice-census.json"
DEFAULT_OUTPUT = HERE / "docs/figures/dual-branch-selection/fork-arming-economics.json"
DEFAULT_THRESHOLDS = (0.0025, 0.005, 0.01, 0.02)


def _read_json(path: Path) -> dict[str, Any]:
    """Read one banked evidence input."""

    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    """Return the content identity of one evidence input."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite_distribution(values: list[float]) -> dict[str, Any]:
    """Summarize finite absolute margins without extrapolating the cohort."""

    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "minimum": float(np.min(array)),
        "quantiles": {
            "0.25": float(np.quantile(array, 0.25)),
            "0.50": float(np.quantile(array, 0.50)),
            "0.75": float(np.quantile(array, 0.75)),
        },
        "maximum": float(np.max(array)),
        "mean": float(np.mean(array)),
        "values_sorted": [float(value) for value in np.sort(array)],
    }


def _policy_price(
    *,
    threshold: float | None,
    finite_margins: list[float],
    unavailable_count: int,
    frame_count: int,
    catalog_slices: int,
    goal_seconds: float,
    single_solve_seconds: float,
) -> dict[str, Any]:
    """Price one fork policy in single-branch solve equivalents."""

    if threshold is None:
        finite_armed_count = len(finite_margins)
        armed_count = frame_count
        policy = "always_fork"
    else:
        finite_armed_count = sum(value <= threshold for value in finite_margins)
        armed_count = finite_armed_count + unavailable_count
        policy = "margin_armed"

    armed_fraction = armed_count / frame_count
    single_solve_equivalents = catalog_slices * (1.0 + armed_fraction)
    always_fork_equivalents = 2.0 * catalog_slices
    assumed_aggregate_solve_seconds = single_solve_equivalents * single_solve_seconds
    return {
        "policy": policy,
        "absolute_margin_threshold": threshold,
        "finite_armed_frame_count": finite_armed_count,
        "unavailable_margin_fallback_frame_count": unavailable_count,
        "armed_frame_count": armed_count,
        "armed_fraction": armed_fraction,
        "projected_armed_catalog_slices": catalog_slices * armed_fraction,
        "single_branch_solve_equivalents": single_solve_equivalents,
        "additional_solve_equivalents_above_single_branch": (
            single_solve_equivalents - catalog_slices
        ),
        "fraction_of_single_branch_workload_budget": (
            single_solve_equivalents / catalog_slices
        ),
        "fraction_of_always_fork_workload": (
            single_solve_equivalents / always_fork_equivalents
        ),
        "assumed_aggregate_solve_seconds": assumed_aggregate_solve_seconds,
        "fraction_of_one_hour_aggregate_time_budget_under_assumed_cost": (
            assumed_aggregate_solve_seconds / goal_seconds
        ),
        "required_aggregate_branch_solves_per_second_for_goal": (
            single_solve_equivalents / goal_seconds
        ),
    }


def run(
    *,
    margin_source: Path = DEFAULT_MARGIN_SOURCE,
    census_source: Path = DEFAULT_CENSUS_SOURCE,
    output: Path = DEFAULT_OUTPUT,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
    single_solve_seconds: float = 1.0,
) -> dict[str, Any]:
    """Bank the observed margin distribution and its parameterized economics."""

    if single_solve_seconds <= 0.0 or not math.isfinite(single_solve_seconds):
        raise ValueError("single_solve_seconds must be finite and positive")
    if not thresholds or any(
        threshold < 0.0 or not math.isfinite(threshold) for threshold in thresholds
    ):
        raise ValueError("thresholds must be non-empty, finite, and non-negative")
    thresholds = tuple(sorted(set(thresholds)))

    margin_receipt = _read_json(margin_source)
    census_receipt = _read_json(census_source)
    references = margin_receipt["references"]
    if len(references) != 6:
        raise RuntimeError(
            f"expected six frozen MAST references, found {len(references)}"
        )
    machines = {record["reference"]["machine"] for record in references}
    if machines != {"MAST"}:
        raise RuntimeError(f"expected only MAST frames, found {sorted(machines)}")

    frames = []
    finite_margins = []
    unavailable_count = 0
    for record in references:
        reference = record["reference"]
        arm = record["pure_arm"]
        margin = arm.get("class_margin")
        nonfinite = arm.get("class_margin_nonfinite")
        if margin is None or not math.isfinite(float(margin)):
            unavailable_count += 1
            absolute_margin = None
            availability = "nonfinite" if nonfinite is not None else "absent"
        else:
            absolute_margin = abs(float(margin))
            finite_margins.append(absolute_margin)
            availability = "finite"
        frames.append(
            {
                "machine": reference["machine"],
                "shot": int(reference["shot"]),
                "slice_index": int(reference["slice_index"]),
                "time_s": float(reference["time_s"]),
                "class_margin": margin,
                "absolute_class_margin": absolute_margin,
                "margin_availability": availability,
                "class_margin_nonfinite": nonfinite,
            }
        )

    if not finite_margins:
        raise RuntimeError("the frozen cohort contains no finite margins")

    catalog_slices = int(census_receipt["totals"]["equilibrium_slices"])
    goal_seconds = float(census_receipt["scope"]["target_wall_seconds"])
    required_rate = float(
        census_receipt["totals"]["required_aggregate_slices_per_second"]
    )
    if catalog_slices != 1_341_435 or not math.isclose(
        required_rate, catalog_slices / goal_seconds, rel_tol=1.0e-12
    ):
        raise RuntimeError("catalog census does not match the declared one-hour budget")

    prices = [
        _policy_price(
            threshold=None,
            finite_margins=finite_margins,
            unavailable_count=unavailable_count,
            frame_count=len(frames),
            catalog_slices=catalog_slices,
            goal_seconds=goal_seconds,
            single_solve_seconds=single_solve_seconds,
        )
    ]
    prices.extend(
        _policy_price(
            threshold=threshold,
            finite_margins=finite_margins,
            unavailable_count=unavailable_count,
            frame_count=len(frames),
            catalog_slices=catalog_slices,
            goal_seconds=goal_seconds,
            single_solve_seconds=single_solve_seconds,
        )
        for threshold in thresholds
    )

    receipt = {
        "artifact": "fork arming economics from production branch margins",
        "measurement_contract": {
            "machine": "MAST",
            "frame_count": len(frames),
            "cohort": margin_receipt["measurement_contract"]["cohort"],
            "frame_selection": margin_receipt["measurement_contract"]["selection"],
            "margin_source": (
                "terminal class_margin from the pure DIVERTED branch of "
                "ForwardProfile.solve_portfolio, one value per frozen catalog frame"
            ),
            "arming_rule": "fork when abs(class_margin) <= threshold",
            "unavailable_margin_fallback": (
                "always fork when class_margin is absent or non-finite"
            ),
            "extrapolation": (
                "the observed armed fraction of the six-frame frozen cohort is "
                "projected onto the full catalog denominator; no representativeness "
                "claim or confidence interval is made"
            ),
        },
        "provenance": {
            "margin_receipt": str(margin_source.relative_to(HERE)),
            "margin_receipt_sha256": _sha256(margin_source),
            "margin_receipt_source_commit": margin_receipt["source_commit"],
            "catalog_census": str(census_source.relative_to(HERE)),
            "catalog_census_sha256": _sha256(census_source),
            "catalog_index": census_receipt["provenance"]["catalog_index"],
            "catalog_identity_sha256": census_receipt["provenance"][
                "catalog_identity_sha256"
            ],
            "slice_counts_sha256": census_receipt["provenance"]["slice_counts_sha256"],
            "driver_sha256": _sha256(Path(__file__)),
        },
        "catalog_budget": {
            "slice_count": catalog_slices,
            "goal_seconds": goal_seconds,
            "required_aggregate_slices_per_second": required_rate,
            "required_rate_status": (
                "derived requirement from slice_count / goal_seconds, not measured "
                "throughput"
            ),
            "single_branch_solve_cost_parameter": {
                "value": single_solve_seconds,
                "units": "aggregate solve-seconds per slice for one pinned branch",
                "status": "assumed normalization parameter, not a measurement",
            },
            "workload_denominator": (
                "1,341,435 single-branch solve equivalents; policy fractions compare "
                "branch-solve work against processing every catalog slice once"
            ),
        },
        "margin_availability": {
            "frame_count": len(frames),
            "finite_count": len(finite_margins),
            "nonfinite_or_absent_count": unavailable_count,
            "nonfinite_or_absent_fraction": unavailable_count / len(frames),
        },
        "absolute_margin_distribution": _finite_distribution(finite_margins),
        "frames": frames,
        "policies": prices,
        "decision_status": {
            "fork_policy_locked": False,
            "reason": (
                "armed fraction alone is insufficient; the separate accelerator-lane "
                "measurement of two-branch batch cost is still required"
            ),
            "out_of_scope": "two-branch batch cost on the accelerator lane",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return receipt


def main() -> None:
    """Run the banked evidence calculation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--margin-source", type=Path, default=DEFAULT_MARGIN_SOURCE)
    parser.add_argument("--census-source", type=Path, default=DEFAULT_CENSUS_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--threshold",
        dest="thresholds",
        action="append",
        type=float,
        help="absolute class-margin threshold; may be repeated",
    )
    parser.add_argument(
        "--single-branch-solve-seconds",
        type=float,
        default=1.0,
        help=(
            "assumed aggregate solve-seconds per slice for one pinned branch; "
            "this is a cost parameter, not measured throughput"
        ),
    )
    args = parser.parse_args()
    receipt = run(
        margin_source=args.margin_source,
        census_source=args.census_source,
        output=args.output,
        thresholds=(tuple(args.thresholds) if args.thresholds else DEFAULT_THRESHOLDS),
        single_solve_seconds=args.single_branch_solve_seconds,
    )
    print(
        json.dumps(
            {
                "frame_count": receipt["measurement_contract"]["frame_count"],
                "margin_availability": receipt["margin_availability"],
                "policies": receipt["policies"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
