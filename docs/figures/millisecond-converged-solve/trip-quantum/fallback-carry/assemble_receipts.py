"""Assemble fallback-carry timing and four-row semantic receipts."""

from __future__ import annotations

import argparse
import csv
from datetime import UTC, datetime
import json
from pathlib import Path


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_csv(path: Path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _summary(row):
    result = row["public_route"]["result"]
    topology = row["public_route"]["terminal_topology"]
    return {
        "terminal_residual": result["terminal_residual"],
        "trips": result["active_set_iterations"],
        "termination": result["termination_reason"],
        "class": topology["achieved_class"],
    }


def assemble(arguments):
    baseline = _read(arguments.timing_baseline)
    production = _read(arguments.production_measurement)
    outer = _read(arguments.outer_measurement)
    promotions = _read(arguments.promotion_measurement)
    after_outer = {
        int(row["outer_index"]) + 1: row
        for row in outer["outer_iterations"]
        if row["active_at_entry"]
    }
    after_promotion_trips = {row["trip"]: row for row in promotions["per_trip"]}
    per_trip = []
    for before in baseline["trips"]:
        trip = before["trip"]
        before_wall = (
            before["banked_trip_wall_s"]
            if before["banked_trip_wall_s"] is not None
            else before["instrumented_promotion_wall_s"]
        )
        after_wall = (
            after_outer[trip]["outer_wall_s"]
            if trip in after_outer
            else after_promotion_trips[trip]["promotion_wall_s"]
        )
        per_trip.append(
            {
                "trip": trip,
                "before_s": before_wall,
                "after_s": after_wall,
                "difference_s": after_wall - before_wall,
                "ratio": after_wall / before_wall,
                "attempted_promotions": before["promotions"],
                "accepted_promotions_before": before["accepted_promotions"],
                "accepted_promotions_after": after_promotion_trips[trip][
                    "accepted_promotions"
                ],
            }
        )

    before_promotions = {
        (int(row["trip"]), int(row["promotion"])): row
        for row in _read_csv(arguments.promotion_baseline)
    }
    after_promotions = {
        (int(row["trip"]), int(row["promotion"])): row
        for row in _read_csv(arguments.promotion_table)
    }
    if before_promotions.keys() != after_promotions.keys():
        raise RuntimeError("promotion identities changed")
    promotion_pairs = []
    for key in before_promotions:
        before = before_promotions[key]
        after = after_promotions[key]
        before_wall = float(before["wall_s"])
        after_wall = float(after["wall_s"])
        before_accepted = before["accepted"].lower() == "true"
        after_accepted = after["accepted"].lower() == "true"
        promotion_pairs.append(
            {
                "trip": key[0],
                "promotion": key[1],
                "before_s": before_wall,
                "after_s": after_wall,
                "difference_s": after_wall - before_wall,
                "accepted_before": before_accepted,
                "accepted_after": after_accepted,
            }
        )

    before_summary = baseline["uninstrumented_execution"]["summary"]
    after_summary = promotions["summary"]
    before_solve = baseline["uninstrumented_execution"]["wall_s"]
    after_solve = production["production_probe"]["steady"]["median_s"]
    late_pairs = [row for row in promotion_pairs if row["trip"] in (6, 7)]
    repeated_late = [row for row in late_pairs if row["promotion"] > 1]
    timing = {
        "schema": "nova.unchanged_state_fallback_carry_timing",
        "captured_at": datetime.now(UTC).isoformat(),
        "measurement_revision": outer["measurement_revision"],
        "scheduler": outer["scheduler"],
        "persistent_compilation_cache": outer["compile"]["persistent_cache"],
        "baseline": {
            "path": str(arguments.timing_baseline),
            "measurement_revision": baseline["measurement_revision"],
            "production_synchronized_s": before_solve,
            "stated_solve_wall_s": 34.96,
            "stated_trip_range_s": [0.12, 12.3],
        },
        "measurement": {
            "production_path": str(arguments.production_measurement),
            "outer_path": str(arguments.outer_measurement),
            "promotion_path": str(arguments.promotion_measurement),
            "production_synchronized_s": after_solve,
            "summary": after_summary,
        },
        "per_trip": per_trip,
        "per_promotion": promotion_pairs,
        "verdict": {
            "semantic_summary_identical": before_summary == after_summary,
            "acceptance_sequence_identical": all(
                row["accepted_before"] == row["accepted_after"]
                for row in promotion_pairs
            ),
            "solve_saving_s": before_solve - after_solve,
            "solve_saving_fraction": (before_solve - after_solve) / before_solve,
            "repeated_late_promotion_saving_s": sum(
                row["before_s"] - row["after_s"] for row in repeated_late
            ),
        },
    }
    _write(arguments.timing_output, timing)

    before_rows = _read(arguments.rows_baseline)
    after_rows = _read(arguments.rows_measurement)
    before_by_identity = {row["identity"]: row for row in before_rows["rows"]}
    after_by_identity = {row["identity"]: row for row in after_rows["rows"]}
    if before_by_identity.keys() != after_by_identity.keys():
        raise RuntimeError("four-row identities changed")
    paired_rows = []
    for identity in before_by_identity:
        before = _summary(before_by_identity[identity])
        after = _summary(after_by_identity[identity])
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
        "schema": "nova.unchanged_state_fallback_carry_four_rows",
        "captured_at": datetime.now(UTC).isoformat(),
        "measurement_revision": after_rows["source_commit"],
        "baseline_path": str(arguments.rows_baseline),
        "measurement_path": str(arguments.rows_measurement),
        "rows": paired_rows,
        "verdict": {
            "row_count": len(paired_rows),
            "all_identical": all(row["identical"] for row in paired_rows),
        },
    }
    _write(arguments.rows_output, rows)
    return timing, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timing-baseline", type=Path, required=True)
    parser.add_argument("--production-measurement", type=Path, required=True)
    parser.add_argument("--outer-measurement", type=Path, required=True)
    parser.add_argument("--promotion-baseline", type=Path, required=True)
    parser.add_argument("--promotion-measurement", type=Path, required=True)
    parser.add_argument("--promotion-table", type=Path, required=True)
    parser.add_argument("--timing-output", type=Path, required=True)
    parser.add_argument("--rows-baseline", type=Path, required=True)
    parser.add_argument("--rows-measurement", type=Path, required=True)
    parser.add_argument("--rows-output", type=Path, required=True)
    arguments = parser.parse_args()
    timing, rows = assemble(arguments)
    print(json.dumps({"timing": timing["verdict"], "rows": rows["verdict"]}))
    if not timing["verdict"]["semantic_summary_identical"]:
        raise SystemExit("production semantic summary changed")
    if not timing["verdict"]["acceptance_sequence_identical"]:
        raise SystemExit("promotion acceptance sequence changed")


if __name__ == "__main__":
    main()
