"""Render paired direct-probe latency and modeled per-trip differences."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


EXPECTED_COMPONENTS = {
    "wall_reachability",
    "flood_fills",
    "separatrix",
    "spline_fits",
    "census",
    "limiter_tangency_along_spline",
    "census_values_by_spline",
    "line_of_sight_rule",
    "host_sync_remainder",
    "forward_evaluation",
    "jacobian_vector_product",
    "line_search",
    "gmres_orthogonalisation",
}


def _read(path: Path, arm: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("complete"):
        raise RuntimeError(f"{path} is not a complete profile")
    if payload.get("arm") != arm:
        raise RuntimeError(f"{path} records arm={payload.get('arm')!r}, not {arm!r}")
    components = {row["component"] for row in payload["per_trip_components"]}
    if components != EXPECTED_COMPONENTS:
        raise RuntimeError(
            f"{path} component mismatch: "
            f"missing={sorted(EXPECTED_COMPONENTS - components)} "
            f"extra={sorted(components - EXPECTED_COMPONENTS)}"
        )
    return payload


def _read_tip(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "nova.production_trip_tip_attribution":
        raise RuntimeError(f"{path} is not a production-tip attribution")
    if not payload.get("complete"):
        raise RuntimeError(f"{path} is incomplete")
    required = {
        "counts",
        "terminal",
        "revision",
        "base_revision",
        "held_commit",
        "scheduler",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise RuntimeError(f"{path} missing fields: {missing}")
    return payload


def _paired_rows(main: dict, candidate: dict) -> list[dict]:
    main_rows = {row["component"]: row for row in main["per_trip_components"]}
    candidate_rows = {row["component"]: row for row in candidate["per_trip_components"]}
    rows = []
    for component in EXPECTED_COMPONENTS:
        baseline = main_rows[component]
        changed = candidate_rows[component]
        rows.append(
            {
                "component": component,
                "main_direct_wall_s_per_call": baseline["direct_wall_s_per_call"],
                "candidate_direct_wall_s_per_call": changed["direct_wall_s_per_call"],
                "delta_direct_wall_s_per_call": changed["direct_wall_s_per_call"]
                - baseline["direct_wall_s_per_call"],
                "main_calls_per_trip": baseline["calls_per_trip"],
                "candidate_calls_per_trip": changed["calls_per_trip"],
                "delta_calls_per_trip": changed["calls_per_trip"]
                - baseline["calls_per_trip"],
                "main_direct_product_s_per_trip": baseline["direct_product_s_per_trip"],
                "candidate_direct_product_s_per_trip": changed[
                    "direct_product_s_per_trip"
                ],
                "delta_direct_product_s_per_trip": changed["direct_product_s_per_trip"]
                - baseline["direct_product_s_per_trip"],
                "main_persistent_cache": baseline["persistent_compilation_cache"],
                "candidate_persistent_cache": changed["persistent_compilation_cache"],
            }
        )
    return sorted(
        rows,
        key=lambda row: row["delta_direct_product_s_per_trip"],
        reverse=True,
    )


def render(
    main_path: Path,
    candidate_path: Path,
    receipt_path: Path,
    figure: Path,
    tip_paths: list[Path],
    trip_attribution_path: Path,
):
    main = _read(main_path, "main")
    candidate = _read(candidate_path, "candidate")
    rows = _paired_rows(main, candidate)
    payload = {
        "schema": "nova.paired_trip_component_attribution",
        "captured_at": datetime.now(UTC).isoformat(),
        "main_revision": main["revision"],
        "candidate_revision": candidate["revision"],
        "scheduler": candidate["timer_runs"][0]["scheduler"],
        "candidate_resume_provenance": candidate.get("resume_provenance"),
        "main_promotion_count_observation": main["promotion_count_observation"],
        "candidate_promotion_count_observation": candidate[
            "promotion_count_observation"
        ],
        "main_measured_full_solve_wall_s": main["trip"]["measured_full_solve_wall_s"],
        "candidate_measured_full_solve_wall_s": candidate["trip"][
            "measured_full_solve_wall_s"
        ],
        "main_wall_s_per_active_set_trip": main["trip"]["wall_s"],
        "candidate_wall_s_per_active_set_trip": candidate["trip"]["wall_s"],
        "method": (
            "rank by candidate-minus-main isolated direct-probe median multiplied "
            "by modeled calls per active-set trip; rows overlap and are not additive"
        ),
        "rows": rows,
    }
    receipt_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    display = [row for row in reversed(rows)]
    labels = [row["component"].replace("_", " ") for row in display]
    delta_ms = [1.0e3 * row["delta_direct_product_s_per_trip"] for row in display]
    main_ms = [1.0e3 * row["main_direct_wall_s_per_call"] for row in display]
    candidate_ms = [1.0e3 * row["candidate_direct_wall_s_per_call"] for row in display]
    positions = np.arange(len(display))

    figure_object, axes = plt.subplots(
        1,
        2,
        figsize=(15.5, 8.2),
        gridspec_kw={"width_ratios": (1.15, 1.0)},
        constrained_layout=True,
    )
    colors = ["#b64238" if value > 0.0 else "#2c6aa0" for value in delta_ms]
    axes[0].barh(positions, delta_ms, color=colors)
    axes[0].axvline(0.0, color="#202020", linewidth=0.8)
    axes[0].set_yticks(positions, labels)
    axes[0].set_xlabel("candidate - main [ms / active-set trip]")
    axes[0].set_title("Modeled per-trip difference")
    axes[0].grid(axis="x", alpha=0.22)

    height = 0.36
    axes[1].barh(positions - height / 2, main_ms, height, label="main", color="#566b7a")
    axes[1].barh(
        positions + height / 2,
        candidate_ms,
        height,
        label="candidate",
        color="#d07a31",
    )
    axes[1].set_yticks(positions, [])
    axes[1].set_xscale("symlog", linthresh=1.0e-3)
    axes[1].set_xlabel("direct synchronized median [ms / call]")
    axes[1].set_title("Isolated call latency")
    axes[1].grid(axis="x", alpha=0.22)
    axes[1].legend(loc="lower right")

    wall_delta = (
        candidate["trip"]["measured_full_solve_wall_s"]
        - main["trip"]["measured_full_solve_wall_s"]
    )
    figure_object.suptitle(
        "Paired H200 component attribution\n"
        f"full solve: {main['trip']['measured_full_solve_wall_s']:.3f} s main → "
        f"{candidate['trip']['measured_full_solve_wall_s']:.3f} s candidate "
        f"({wall_delta:+.3f} s)"
    )
    figure_object.text(
        0.5,
        0.005,
        "Per-trip products are diagnostic and overlap; "
        "they are not an additive wall decomposition.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    figure_object.savefig(figure, dpi=180)
    plt.close(figure_object)
    tips = [_read_tip(path) for path in tip_paths]
    if len(tips) != 2:
        raise RuntimeError(f"expected two intermediate tips, found {len(tips)}")
    bases = {tip["base_revision"] for tip in tips}
    jobs = {tip["scheduler"]["job_id"] for tip in tips}
    if len(bases) != 1:
        raise RuntimeError(f"intermediate tips use different bases: {sorted(bases)}")
    if len(jobs) != 1:
        raise RuntimeError(f"intermediate tips use different jobs: {sorted(jobs)}")
    main_trips = int(main["trip"]["counts"]["active_set_trips"])
    candidate_trips = int(candidate["trip"]["counts"]["active_set_trips"])
    doubling_tips = [
        tip for tip in tips if int(tip["counts"]["active_set_trips"]) == 2 * main_trips
    ]
    attribution = {
        "schema": "nova.production_trip_count_attribution",
        "captured_at": datetime.now(UTC).isoformat(),
        "main": {
            "revision": main["revision"],
            "active_set_trips": main_trips,
            "measured_full_solve_wall_s": main["trip"]["measured_full_solve_wall_s"],
        },
        "candidate": {
            "revision": candidate["revision"],
            "active_set_trips": candidate_trips,
            "measured_full_solve_wall_s": candidate["trip"][
                "measured_full_solve_wall_s"
            ],
        },
        "intermediate_tips": tips,
        "attribution": {
            "doubling_is_behavior_change": True,
            "main_to_candidate_trip_ratio": candidate_trips / main_trips,
            "held_tips_matching_candidate_trip_count": [
                {
                    "label": tip["label"],
                    "held_commit": tip["held_commit"],
                    "merge_revision": tip["revision"],
                    "terminal_residual": tip["terminal"]["residual"],
                    "termination_reason": tip["terminal"]["termination_reason"],
                    "achieved_class": tip["terminal"]["achieved_class"],
                }
                for tip in doubling_tips
                if int(tip["counts"]["active_set_trips"]) == candidate_trips
            ],
            "unique_held_tip": (
                {
                    "label": doubling_tips[0]["label"],
                    "held_commit": doubling_tips[0]["held_commit"],
                    "merge_revision": doubling_tips[0]["revision"],
                }
                if len(doubling_tips) == 1
                else None
            ),
            "gate": (
                "attributed"
                if len(doubling_tips) == 1 and candidate_trips == 2 * main_trips
                else "ambiguous"
            ),
        },
        "method": (
            "each held commit was merged independently into the same main revision; "
            "both detached tips ran the same production arm in this H200 job"
        ),
    }
    trip_attribution_path.write_text(
        json.dumps(attribution, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"PAIRED_RECEIPT_WRITTEN={receipt_path}")
    print(f"PAIRED_FIGURE_WRITTEN={figure}")
    print(f"TRIP_ATTRIBUTION_WRITTEN={trip_attribution_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--tip", type=Path, action="append", required=True)
    parser.add_argument("--trip-attribution", type=Path, required=True)
    arguments = parser.parse_args()
    render(
        arguments.main,
        arguments.candidate,
        arguments.receipt,
        arguments.figure,
        arguments.tip,
        arguments.trip_attribution,
    )


if __name__ == "__main__":
    main()
