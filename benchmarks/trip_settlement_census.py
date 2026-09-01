"""Bank active-set settlement counts and their fixed-shape cost projection."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAST_BANK = (
    ROOT / "docs/figures/primary-xpoint-evidence/efit-topology-corroboration.json"
)
DEFAULT_DIIID_BANK = (
    ROOT / "docs/figures/diiid-forward-onboarding/forward-gs/forward_gs_receipt.json"
)
DEFAULT_JSON = ROOT / "docs/figures/solver-trip-orchestration/settlement-histogram.json"
DEFAULT_PNG = ROOT / "docs/figures/solver-trip-orchestration/settlement-histogram.png"
TRIP_LIMIT = 16
FULL_TRIP_FLOOR_MS_PER_SLICE = 1.75
TARGET_MS_PER_SLICE = 1.0
MAST_CAVEAT = (
    "the MAST telemetry comes from a regeneration whose semantic diff changed "
    "10/12 selected primaries vs the committed bank and flipped 22086/43 pure "
    "to active_set_stagnated, so mark any non-settling arm accordingly rather "
    "than folding it into the settled statistics, and cite the bank SHA as your "
    "provenance."
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return payload


def _settlement_trip(mask_differences: list[int]) -> int | None:
    """Return the one-based start of the final observed all-zero suffix."""

    if not mask_differences:
        return None
    last_change = max(
        (index for index, difference in enumerate(mask_differences) if difference),
        default=-1,
    )
    first_stable_zero = last_change + 1
    if first_stable_zero >= len(mask_differences):
        return None
    return first_stable_zero + 1


def _projection(mean_trips: float) -> dict[str, float]:
    projected = FULL_TRIP_FLOOR_MS_PER_SLICE * mean_trips / TRIP_LIMIT
    return {
        "mean_trips": mean_trips,
        "projected_ms_per_slice": projected,
        "target_ms_per_slice": TARGET_MS_PER_SLICE,
        "margin_below_target_ms_per_slice": TARGET_MS_PER_SLICE - projected,
    }


def _record(
    *,
    identity: str,
    mask_differences: list[int],
    active_set_residuals: list[float | None],
    termination: str,
    retained_iteration_history: list[float | None] | None = None,
) -> dict[str, Any]:
    if len(mask_differences) != len(active_set_residuals):
        raise ValueError(f"{identity}: mask and residual histories differ in length")
    settlement_trip = _settlement_trip(mask_differences)
    return {
        "identity": identity,
        "recorded_trips": len(mask_differences),
        "mask_differences": mask_differences,
        "active_set_residuals": active_set_residuals,
        "retained_iteration_history": retained_iteration_history,
        "termination": termination,
        "settlement": "settled" if settlement_trip is not None else "non_settling",
        "settlement_trip_count": settlement_trip,
    }


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    settled = [
        int(record["settlement_trip_count"])
        for record in records
        if record["settlement_trip_count"] is not None
    ]
    non_settling = [
        record["identity"]
        for record in records
        if record["settlement_trip_count"] is None
    ]
    if not settled:
        raise ValueError("the bank contains no observed settlements")
    histogram = Counter(settled)
    settled_mean = float(np.mean(settled))
    fallback_mean = float(
        (sum(settled) + TRIP_LIMIT * len(non_settling)) / len(records)
    )
    return {
        "observed_records": len(records),
        "settled_records": len(settled),
        "non_settling_records": len(non_settling),
        "non_settling_identities": non_settling,
        "settled_histogram": {str(trip): histogram[trip] for trip in sorted(histogram)},
        "settled_only_projection": _projection(settled_mean),
        "full_trip_fallback_projection": _projection(fallback_mean),
    }


def _mast_census(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != 12:
        raise ValueError("the MAST bank must carry exactly twelve rows")

    pure_rows = [row for row in rows if row.get("arm") == "pure"]
    mixed_rows = [row for row in rows if row.get("arm") == "mixed"]
    if len(pure_rows) != 6 or len(mixed_rows) != 6:
        raise ValueError("the MAST bank must carry six pure and six mixed arms")

    records: list[dict[str, Any]] = []
    for row in pure_rows:
        iterations = int(row["active_set_iterations"])
        differences = [int(value) for value in row["active_set_mask_differences"]]
        residuals = [
            None if value is None else float(value)
            for value in row["active_set_residuals"]
        ]
        if iterations != len(differences) or iterations != len(residuals):
            raise ValueError(f"{row['identity']} pure: incomplete trip telemetry")
        records.append(
            _record(
                identity=f"{row['identity']} pure",
                mask_differences=differences,
                active_set_residuals=residuals,
                termination=str(row["termination_reason"]),
            )
        )

    mixed_lengths = {
        f"{row['identity']} mixed": int(row["active_set_iterations"])
        for row in mixed_rows
    }
    for row in mixed_rows:
        if int(row["active_set_iterations"]) != 0:
            raise ValueError(f"{row['identity']} mixed: expected zero-length history")
        if row["active_set_mask_differences"] or row["active_set_residuals"]:
            raise ValueError(
                f"{row['identity']} mixed: inconsistent zero-length history"
            )

    summary = _summarize(records)
    if sum(record["recorded_trips"] for record in records) != 66:
        raise ValueError("the MAST pure-arm telemetry must contain 66 trip records")
    return {
        "source": {
            "path": str(path.relative_to(ROOT)),
            "sha256": _sha256(path),
            "declared_rows": len(rows),
            "censused_pure_arms": len(pure_rows),
            "pure_arm_trip_records": sum(
                record["recorded_trips"] for record in records
            ),
        },
        "limitation": {
            "mixed_arms_excluded": len(mixed_rows),
            "reason": (
                "all six mixed arms carry valid zero-length active-set histories; "
                "the settlement census therefore covers the six pure arms only"
            ),
            "recorded_lengths": mixed_lengths,
        },
        "required_caveat": MAST_CAVEAT,
        "records": records,
        "summary": summary,
    }


def _diiid_census(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    rows = payload.get("result", {}).get("frame_records")
    if not isinstance(rows, list) or len(rows) != 5:
        raise ValueError("the DIII-D bank must carry exactly five frame records")

    records: list[dict[str, Any]] = []
    for row in rows:
        iterations = int(row["active_set_iterations"])
        differences = [
            int(value) for value in row["active_set_mask_differences"][:iterations]
        ]
        residuals = [
            None if value is None else float(value)
            for value in row["active_set_residuals"][:iterations]
        ]
        retained_iteration_history = [
            None if value is None else float(value) for value in row["residual_history"]
        ]
        if iterations != len(differences) or iterations != len(residuals):
            raise ValueError(f"DIII-D frame {row['frame']}: incomplete trip telemetry")
        records.append(
            _record(
                identity=f"{row['shot']} frame {row['frame']}",
                mask_differences=differences,
                active_set_residuals=residuals,
                termination=str(row["solver_termination"]),
                retained_iteration_history=retained_iteration_history,
            )
        )

    summary = _summarize(records)
    expected_histogram = {"3": 2, "4": 1, "5": 1, "6": 1}
    if summary["settled_histogram"] != expected_histogram:
        raise ValueError(
            "the DIII-D settlement histogram changed from the banked partial"
        )
    if summary["settled_only_projection"]["mean_trips"] != 4.2:
        raise ValueError("the DIII-D mean changed from the banked partial")
    return {
        "source": {
            "path": str(path.relative_to(ROOT)),
            "sha256": _sha256(path),
            "declared_frames": len(rows),
        },
        "records": records,
        "summary": summary,
    }


def _test_surface() -> dict[str, Any]:
    return {
        "decision": {
            "key": "settled-exit-policy",
            "status": "open_lead_decision",
            "implemented_by_this_census": False,
        },
        "fixed_shape_contract": (
            "a settled exit must keep the compiled trip extent fixed and turn "
            "post-settlement trips into masked no-ops, never a dynamic shape"
        ),
        "required_tests": [
            {
                "node_id": (
                    "tests/test_own_mask_acceptance.py::"
                    "test_own_mask_acceptance_is_inert_when_candidates_keep_the_mask"
                ),
                "preserves": (
                    "paired eager/JIT bit identity when own-mask acceptance sees "
                    "no mask change"
                ),
            },
            {
                "node_id": (
                    "tests/test_own_mask_acceptance.py::"
                    "test_candidate_worsening_by_measured_own_mask_ratio_is_refused"
                ),
                "preserves": "candidate acceptance on each candidate's own mask",
            },
            {
                "node_id": (
                    "tests/test_fixed_point.py::"
                    "test_newton_returns_the_best_observed_relative_sup_iterate"
                ),
                "preserves": (
                    "best-iterate retention rather than terminal-iterate return"
                ),
            },
            {
                "node_id": (
                    "tests/test_fixed_point.py::"
                    "test_newton_refuses_equal_own_mask_residual_and_keeps_settled_mask"
                ),
                "preserves": "equal own-mask refusal and settled-mask retention",
            },
            {
                "node_id": (
                    "tests/test_fixed_point.py::"
                    "test_newton_with_a_constant_active_set_preserves_the_smooth_solve"
                ),
                "preserves": (
                    "bit identity against the smooth solve for a constant mask"
                ),
            },
            {
                "node_id": (
                    "tests/test_newton_trajectory_continuation.py::"
                    "test_monotone_solve_is_bit_identical_with_continuation_on_and_off"
                ),
                "preserves": "paired result identity on a motionless mask",
            },
            {
                "node_id": (
                    "tests/test_globalization_state_continuation.py::"
                    "test_monotone_solve_is_bit_identical_eager_and_jit"
                ),
                "preserves": "eager/JIT identity while globalization state is retained",
            },
        ],
    }


def _combined_summary(mast: dict[str, Any], diiid: dict[str, Any]) -> dict[str, Any]:
    settled_counts = []
    fallback_counts = []
    for machine in (mast, diiid):
        for record in machine["records"]:
            count = record["settlement_trip_count"]
            if count is not None:
                settled_counts.append(int(count))
                fallback_counts.append(int(count))
            else:
                fallback_counts.append(TRIP_LIMIT)
    return {
        "settled_records": len(settled_counts),
        "non_settling_records": len(fallback_counts) - len(settled_counts),
        "settled_only_projection": _projection(float(np.mean(settled_counts))),
        "full_trip_fallback_projection": _projection(float(np.mean(fallback_counts))),
    }


def _draw(payload: dict[str, Any], output: Path) -> None:
    mast = payload["machines"]["MAST"]
    diiid = payload["machines"]["DIII-D"]
    trip_bins = list(range(1, 7))
    mast_hist = mast["summary"]["settled_histogram"]
    diiid_hist = diiid["summary"]["settled_histogram"]

    figure, (histogram_axis, projection_axis) = plt.subplots(
        2,
        1,
        figsize=(9.4, 7.2),
        gridspec_kw={"height_ratios": (1.8, 1.0)},
        constrained_layout=True,
    )
    x = np.arange(len(trip_bins) + 1)
    width = 0.36
    mast_counts = [mast_hist.get(str(trip), 0) for trip in trip_bins] + [
        mast["summary"]["non_settling_records"]
    ]
    diiid_counts = [diiid_hist.get(str(trip), 0) for trip in trip_bins] + [
        diiid["summary"]["non_settling_records"]
    ]
    histogram_axis.bar(
        x - width / 2, mast_counts, width, label="MAST pure", color="#512b81"
    )
    histogram_axis.bar(
        x + width / 2, diiid_counts, width, label="DIII-D", color="#1696a7"
    )
    histogram_axis.set_xticks(x, [*[str(trip) for trip in trip_bins], "not\nsettled"])
    histogram_axis.set_ylabel("bank records")
    histogram_axis.set_xlabel(
        "outer trips consumed before the final all-zero mask suffix"
    )
    histogram_axis.set_title("Active-set settlement census")
    histogram_axis.legend(frameon=False)
    histogram_axis.spines[["top", "right"]].set_visible(False)
    histogram_axis.text(
        0.99,
        0.98,
        "MAST mixed arms: 6 unobserved (valid zero-length histories)",
        transform=histogram_axis.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="#555555",
    )

    labels = ["MAST settled only", "MAST with 16-trip fallback", "DIII-D"]
    projections = [
        mast["summary"]["settled_only_projection"]["projected_ms_per_slice"],
        mast["summary"]["full_trip_fallback_projection"]["projected_ms_per_slice"],
        diiid["summary"]["settled_only_projection"]["projected_ms_per_slice"],
    ]
    colors = ["#512b81", "#8a6cab", "#1696a7"]
    y = np.arange(len(labels))
    projection_axis.barh(y, projections, color=colors)
    projection_axis.axvline(
        TARGET_MS_PER_SLICE,
        color="#b33a3a",
        linestyle="--",
        linewidth=1.5,
        label="1 ms target",
    )
    for index, value in enumerate(projections):
        projection_axis.text(value + 0.02, index, f"{value:.3f}", va="center")
    projection_axis.set_yticks(y, labels)
    projection_axis.set_xlim(0.0, 1.1)
    projection_axis.set_xlabel("projected ms/slice from the 1.75 ms, 16-trip floor")
    projection_axis.invert_yaxis()
    projection_axis.spines[["top", "right", "left"]].set_visible(False)
    projection_axis.legend(frameon=False, loc="lower right")

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def run(
    mast_bank: Path,
    diiid_bank: Path,
    output_json: Path,
    output_png: Path,
) -> dict[str, Any]:
    mast = _mast_census(mast_bank)
    diiid = _diiid_census(diiid_bank)
    payload = {
        "schema": "nova.trip-settlement-census/1",
        "recorded_at": "2026-09-01",
        "question": (
            "how many outer trips are observed before each active-set mask enters "
            "its final all-zero difference suffix"
        ),
        "method": {
            "settlement_definition": (
                "the one-based first trip in the final observed all-zero mask-"
                "difference suffix; an earlier zero followed by a change is not "
                "settlement"
            ),
            "non_settling_definition": (
                "no trailing zero-difference observation before telemetry ends"
            ),
            "settled_statistics": "exclude non-settling and unobserved records",
            "fallback_projection": (
                "charge each observed non-settling record the configured 16-trip "
                "limit; unobserved zero-length histories remain excluded"
            ),
            "projection_formula": "1.75 ms/slice * mean trips / 16 trips",
        },
        "constants": {
            "configured_trip_limit": TRIP_LIMIT,
            "full_trip_floor_ms_per_slice": FULL_TRIP_FLOOR_MS_PER_SLICE,
            "target_ms_per_slice": TARGET_MS_PER_SLICE,
        },
        "machines": {"MAST": mast, "DIII-D": diiid},
        "combined_observed_cohort": _combined_summary(mast, diiid),
        "semantic_preservation_test_surface": _test_surface(),
        "solver_source_modified": False,
        "project_absolute_figure_src": (
            "/nova/figures/solver-trip-orchestration/settlement-histogram.png"
        ),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    _draw(payload, output_png)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mast-bank", type=Path, default=DEFAULT_MAST_BANK)
    parser.add_argument("--diiid-bank", type=Path, default=DEFAULT_DIIID_BANK)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-png", type=Path, default=DEFAULT_PNG)
    arguments = parser.parse_args()
    payload = run(
        arguments.mast_bank.resolve(),
        arguments.diiid_bank.resolve(),
        arguments.output_json.resolve(),
        arguments.output_png.resolve(),
    )
    mast = payload["machines"]["MAST"]["summary"]
    diiid = payload["machines"]["DIII-D"]["summary"]
    mast_fallback_ms = mast["full_trip_fallback_projection"]["projected_ms_per_slice"]
    print(
        "SETTLEMENT_CENSUS "
        f"mast_settled={mast['settled_records']}/6 "
        f"mast_mean={mast['settled_only_projection']['mean_trips']:.3f} "
        f"mast_fallback_ms={mast_fallback_ms:.6f} "
        f"diiid_settled={diiid['settled_records']}/5 "
        f"diiid_mean={diiid['settled_only_projection']['mean_trips']:.3f} "
        f"diiid_ms={diiid['settled_only_projection']['projected_ms_per_slice']:.6f}"
    )


if __name__ == "__main__":
    main()
