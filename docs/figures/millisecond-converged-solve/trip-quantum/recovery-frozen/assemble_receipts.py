"""Assemble paired production timing and semantic receipts."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _active_trip_rows(receipt: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for row in receipt["outer_iterations"] if row["active_at_entry"]]


def _public_summary(row: dict[str, Any]) -> dict[str, Any]:
    route = row["public_route"]
    result = route["result"]
    topology = route["terminal_topology"]
    return {
        "terminal_residual": result["terminal_residual"],
        "trips": result["active_set_iterations"],
        "termination": result["termination_reason"],
        "class": topology["achieved_class"],
    }


def assemble(
    *,
    timing_baseline: Path,
    timing_measurement: Path,
    production_measurement: Path,
    timing_output: Path,
    rows_baseline: Path,
    rows_measurement: Path,
    rows_output: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    baseline = _read(timing_baseline)
    measured = _read(timing_measurement)
    production = _read(production_measurement)
    baseline_trips = _active_trip_rows(baseline)
    measured_trips = _active_trip_rows(measured)
    if len(baseline_trips) != len(measured_trips):
        raise RuntimeError(
            "paired timing requires the same number of active outer trips: "
            f"baseline={len(baseline_trips)} measured={len(measured_trips)}"
        )
    paired_trips = [
        {
            "trip": int(before["outer_index"]) + 1,
            "before_s": before["outer_wall_s"],
            "after_s": after["outer_wall_s"],
            "difference_s": after["outer_wall_s"] - before["outer_wall_s"],
            "ratio": after["outer_wall_s"] / before["outer_wall_s"],
            "attempted_promotions": after["attempted_promotions"],
            "accepted_promotions": after["accepted_promotions"],
            "backtrack_sum": sum(after["backtrack_counts"]),
        }
        for before, after in zip(baseline_trips, measured_trips, strict=True)
    ]
    timing = {
        "schema": "nova.recovery_frozen_partition_timing",
        "captured_at": datetime.now(UTC).isoformat(),
        "measurement_revision": measured["measurement_revision"],
        "scheduler": measured["scheduler"],
        "persistent_compilation_cache": measured["compile"]["persistent_cache"],
        "baseline": {
            "path": str(timing_baseline),
            "measurement_revision": baseline["measurement_revision"],
            "stated_solve_wall_s": 41.1,
            "banked_uninstrumented_s": baseline["uninstrumented_execution"]["wall_s"],
            "stated_active_trip_range_s": [3.4, 12.2],
        },
        "measurement": {
            "outer_trace_path": str(timing_measurement),
            "production_path": str(production_measurement),
            "outer_uninstrumented_s": measured["uninstrumented_execution"]["wall_s"],
            "production_synchronized_s": production["production_probe"]["steady"][
                "median_s"
            ],
            "summary": measured["uninstrumented_execution"]["summary"],
        },
        "per_trip": paired_trips,
        "verdict": {
            "semantic_summary_identical": (
                measured["uninstrumented_execution"]["summary"]
                == baseline["uninstrumented_execution"]["summary"]
            ),
            "active_trip_count_identical": len(baseline_trips) == len(measured_trips),
            "solve_difference_from_stated_baseline_s": (
                production["production_probe"]["steady"]["median_s"] - 41.1
            ),
        },
    }
    _write(timing_output, timing)

    before_rows = _read(rows_baseline)
    after_rows = _read(rows_measurement)
    before_by_identity = {row["identity"]: row for row in before_rows["rows"]}
    after_by_identity = {row["identity"]: row for row in after_rows["rows"]}
    if before_by_identity.keys() != after_by_identity.keys():
        raise RuntimeError("four-row identities changed")
    paired_rows = []
    for identity in before_by_identity:
        before = _public_summary(before_by_identity[identity])
        after = _public_summary(after_by_identity[identity])
        paired_rows.append(
            {
                "identity": identity,
                "before": before,
                "after": after,
                "difference": {
                    "terminal_residual": (
                        after["terminal_residual"] - before["terminal_residual"]
                    ),
                    "trips": after["trips"] - before["trips"],
                    "termination_changed": after["termination"]
                    != before["termination"],
                    "class_changed": after["class"] != before["class"],
                },
                "identical": after == before,
            }
        )
    rows = {
        "schema": "nova.recovery_frozen_partition_four_rows",
        "captured_at": datetime.now(UTC).isoformat(),
        "measurement_revision": after_rows["source_commit"],
        "baseline_path": str(rows_baseline),
        "measurement_path": str(rows_measurement),
        "rows": paired_rows,
        "verdict": {
            "row_count": len(paired_rows),
            "all_identical": all(row["identical"] for row in paired_rows),
        },
    }
    _write(rows_output, rows)
    return timing, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timing-baseline", type=Path, required=True)
    parser.add_argument("--timing-measurement", type=Path, required=True)
    parser.add_argument("--production-measurement", type=Path, required=True)
    parser.add_argument("--timing-output", type=Path, required=True)
    parser.add_argument("--rows-baseline", type=Path, required=True)
    parser.add_argument("--rows-measurement", type=Path, required=True)
    parser.add_argument("--rows-output", type=Path, required=True)
    arguments = parser.parse_args()
    timing, rows = assemble(
        timing_baseline=arguments.timing_baseline,
        timing_measurement=arguments.timing_measurement,
        production_measurement=arguments.production_measurement,
        timing_output=arguments.timing_output,
        rows_baseline=arguments.rows_baseline,
        rows_measurement=arguments.rows_measurement,
        rows_output=arguments.rows_output,
    )
    print(json.dumps({"timing": timing["verdict"], "rows": rows["verdict"]}))
    if not timing["verdict"]["semantic_summary_identical"]:
        raise SystemExit("production semantic summary changed")
    if not rows["verdict"]["all_identical"]:
        raise SystemExit("public-route semantic rows changed")


if __name__ == "__main__":
    main()
