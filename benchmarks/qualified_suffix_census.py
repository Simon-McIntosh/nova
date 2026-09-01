"""Census policy-qualified trip suffixes from committed solve histories."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SETTLEMENT_RECEIPT = (
    ROOT / "docs/figures/solver-trip-orchestration/settlement-histogram.json"
)
THROUGHPUT_RECEIPT = (
    ROOT / "docs/figures/solver-trip-orchestration/settled-exit-throughput.json"
)
DEFAULT_JSON = ROOT / "docs/figures/solver-trip-orchestration/suffix-census.json"
DEFAULT_PNG = ROOT / "docs/figures/solver-trip-orchestration/suffix-census.png"
TRIP_LIMIT = 16
FULL_TRIP_FLOOR_MS = 1.75
TARGET_MS = 1.0


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _histogram(trips: list[int]) -> dict[str, int]:
    counts = Counter(trips)
    return {str(trip): counts[trip] for trip in sorted(counts)}


def _projection(trips: list[int], control_ms_per_member: float) -> dict[str, Any]:
    mean_trips = float(np.mean(trips))
    projected_ms = FULL_TRIP_FLOOR_MS * mean_trips / TRIP_LIMIT
    return {
        "members": len(trips),
        "mean_charged_trips": mean_trips,
        "trip_histogram": _histogram(trips),
        "projected_ms_per_slice": projected_ms,
        "projected_speedup_x": FULL_TRIP_FLOOR_MS / projected_ms,
        "projected_direct_h200_ms_per_member": (
            control_ms_per_member * mean_trips / TRIP_LIMIT
        ),
        "target_ms_per_slice": TARGET_MS,
        "meets_one_ms_target": projected_ms <= TARGET_MS,
    }


def _relative_remaining_improvement(residuals: list[float], index: int) -> float:
    current = residuals[index]
    best_later = min(residuals[index:])
    return max(0.0, current - best_later) / max(abs(current), np.finfo(float).tiny)


def _progress_trip(record: dict[str, Any], tolerance: float) -> int:
    """Return the first stable-mask trip within tolerance of its best suffix."""

    settlement_trip = record["settlement_trip_count"]
    if settlement_trip is None:
        return TRIP_LIMIT
    residuals = [float(value) for value in record["active_set_residuals"]]
    for index in range(int(settlement_trip) - 1, len(residuals)):
        if _relative_remaining_improvement(residuals, index) <= tolerance:
            return index + 1
    return TRIP_LIMIT


def _strict_index(throughput: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    rows = throughput["paired_committed_bank_comparison"]["rows"]
    index = {(row["machine"], row["identity"]): row for row in rows}
    if len(index) != len(rows):
        raise ValueError("the paired bank receipt contains duplicate member identities")
    return index


def _records(settlement: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    for machine in ("MAST", "DIII-D"):
        for record in settlement["machines"][machine]["records"]:
            records.append({"machine": machine, **record})
    return records


def _qualification_census(
    settlement: dict[str, Any], throughput: dict[str, Any]
) -> tuple[dict[str, Any], dict[tuple[str, str], int]]:
    paired = _strict_index(throughput)
    strict_trips: dict[tuple[str, str], int] = {}
    rows = []
    for record in _records(settlement):
        key = (record["machine"], record["identity"])
        comparison = paired.get(key)
        if comparison is None:
            raise ValueError(f"missing paired qualification for {key}")
        qualified = comparison["classification"] == "qualified_recorded_noop_suffix"
        first_trip = (
            int(comparison["executed_trips_after_policy"]) if qualified else None
        )
        if first_trip is not None and first_trip >= TRIP_LIMIT:
            qualified = False
            first_trip = None
        if qualified:
            strict_trips[key] = first_trip
        rows.append(
            {
                "machine": record["machine"],
                "identity": record["identity"],
                "recorded_trips": record["recorded_trips"],
                "mask_settlement_trip": record["settlement_trip_count"],
                "strict_qualified_before_trip_limit": qualified,
                "first_strict_qualification_trip": first_trip,
                "classification": comparison["classification"],
                "own_mask_acceptance_enabled": True,
                "zero_accepted_newton_promotions_at_qualification": (
                    True if qualified else None
                ),
                "bit_identical_retention_at_qualification": (
                    bool(comparison.get("terminal_relative_residual_bit_identical"))
                    if qualified
                    else None
                ),
                "qualification_evidence": comparison.get(
                    "acceptance_qualification",
                    "no stable mask suffix was recorded",
                ),
            }
        )

    by_machine: dict[str, Any] = {}
    for machine, declared, unavailable in (("MAST", 12, 6), ("DIII-D", 5, 0)):
        machine_rows = [row for row in rows if row["machine"] == machine]
        trips = [
            row["first_strict_qualification_trip"]
            for row in machine_rows
            if row["strict_qualified_before_trip_limit"]
        ]
        by_machine[machine] = {
            "declared_bank_members": declared,
            "telemetry_observable_members": len(machine_rows),
            "telemetry_unavailable_members": unavailable,
            "strict_qualified_members": len(trips),
            "strict_qualified_fraction_of_observable": len(trips) / len(machine_rows),
            "strict_confirmed_fraction_of_declared_bank": len(trips) / declared,
            "first_qualification_trip_histogram": _histogram(trips),
        }

    qualified = len(strict_trips)
    observable = len(rows)
    declared = sum(item["declared_bank_members"] for item in by_machine.values())
    return (
        {
            "definition": {
                "required_conjunction": [
                    "final observed active-set mask-difference suffix is zero",
                    "own-mask acceptance is enabled",
                    "the qualifying trip accepts zero Newton promotions",
                    "the retained result is bit-identical to the full recorded result",
                ],
                "deadline": "qualification must occur before trip 16",
                "classification_authority": (
                    "reuse the landed paired bank replay; do not promote mask-only "
                    "settlement or a merely improving residual history to strict "
                    "qualification"
                ),
                "telemetry_boundary": (
                    "the source banks do not store per-trip promotion counters or "
                    "state hashes; the five strict classifications are therefore "
                    "inherited from the landed paired replay rather than re-inferred "
                    "from residual equality by this census"
                ),
            },
            "per_bank": by_machine,
            "combined": {
                "declared_bank_members": declared,
                "telemetry_observable_members": observable,
                "telemetry_unavailable_members": declared - observable,
                "strict_qualified_members": qualified,
                "strict_qualified_fraction_of_observable": qualified / observable,
                "strict_confirmed_fraction_of_declared_bank": qualified / declared,
                "first_qualification_trip_histogram": _histogram(
                    list(strict_trips.values())
                ),
                "mask_only_settling_members": sum(
                    row["mask_settlement_trip"] is not None for row in rows
                ),
                "mask_only_settling_fraction_of_observable": sum(
                    row["mask_settlement_trip"] is not None for row in rows
                )
                / observable,
            },
            "rows": rows,
        },
        strict_trips,
    )


def _criterion_table(
    settlement: dict[str, Any],
    throughput: dict[str, Any],
    strict_trips: dict[tuple[str, str], int],
) -> dict[str, Any]:
    records = _records(settlement)
    control_ms = float(
        throughput["h200_width_1024"]["full_trip_control"]["steady"][
            "median_ms_per_member"
        ]
    )

    criteria: list[tuple[str, str, float | None]] = [
        (
            "full_trip_control",
            "charge all sixteen trips",
            None,
        ),
        (
            "strict_qualified_noop",
            "locked four-conjunct qualification; otherwise charge sixteen trips",
            None,
        ),
        (
            "mask_only_counterfactual",
            "first final-zero mask suffix; semantic counterfactual, not admissible",
            None,
        ),
        (
            "residual_plateau_exact",
            "stable mask and no remaining improvement in the banked residual suffix",
            0.0,
        ),
        (
            "residual_progress_relative_1pct",
            "stable mask and at most 1% improvement remains to the best banked suffix",
            0.01,
        ),
        (
            "residual_progress_relative_5pct",
            "stable mask and at most 5% improvement remains to the best banked suffix",
            0.05,
        ),
    ]
    table = []
    for key, definition, tolerance in criteria:
        trips = []
        per_machine: dict[str, list[int]] = {"MAST": [], "DIII-D": []}
        for record in records:
            identity = (record["machine"], record["identity"])
            if key == "full_trip_control":
                trip = TRIP_LIMIT
            elif key == "strict_qualified_noop":
                trip = strict_trips.get(identity, TRIP_LIMIT)
            elif key == "mask_only_counterfactual":
                trip = record["settlement_trip_count"] or TRIP_LIMIT
            else:
                if tolerance is None:
                    raise AssertionError("residual criteria require a tolerance")
                trip = _progress_trip(record, tolerance)
            trip = min(int(trip), TRIP_LIMIT)
            trips.append(trip)
            per_machine[record["machine"]].append(trip)
        projection = _projection(trips, control_ms)
        projection["criterion"] = key
        projection["definition"] = definition
        projection["relative_remaining_improvement_tolerance"] = tolerance
        projection["per_machine"] = {
            machine: _projection(machine_trips, control_ms)
            for machine, machine_trips in per_machine.items()
        }
        projection["declared_bank_sensitivity"] = _projection(
            trips + [TRIP_LIMIT] * 6, control_ms
        )
        table.append(projection)
    return {
        "method": {
            "projection_formula": "1.75 ms/slice * mean charged trips / 16",
            "direct_h200_formula": (
                "415.60787488197093 ms/member * mean charged trips / 16"
            ),
            "fallback": (
                "charge sixteen trips to every member that does not qualify and "
                "to each of the six MAST mixed arms without trip telemetry"
            ),
            "residual_relaxation": (
                "banked-hindsight ceiling: after the final-zero mask suffix begins, "
                "find the first trip whose residual can improve by no more than the "
                "stated fraction relative to the best later residual"
            ),
            "causality_caveat": (
                "the hindsight criterion is not an online stopping rule; any policy "
                "implementation needs a causal progress window and paired state/error "
                "validation before dispatch"
            ),
        },
        "observed_telemetry_cohort": table,
    }


def _representativeness(
    settlement: dict[str, Any], throughput: dict[str, Any]
) -> dict[str, Any]:
    width = int(throughput["configuration"]["width"])
    return {
        "width_1024_measurement": {
            "batch_members": width,
            "unique_initial_states": 1,
            "unique_external_current_vectors": 1,
            "unique_physical_workloads": 1,
            "composition": (
                "one synthetic bootstrapped Solovev free-boundary member and one "
                "sixteen-conductor current vector, each repeated exactly 1024 times"
            ),
            "bank_identity_members": 0,
            "observed_trip_distribution": {"16": width},
        },
        "bank_regeneration": {
            "machines": 2,
            "declared_solver_members": 17,
            "telemetry_observable_members": 11,
            "member_composition": {
                "MAST": (
                    "six distinct shot/slice identities crossed with pure and mixed "
                    "arms (12 declared); only the six pure arms carry trip telemetry"
                ),
                "DIII-D": "five distinct parquet shot/frame identities",
            },
            "telemetry_member_identities": [
                record["identity"] for record in _records(settlement)
            ],
        },
        "verdict": "not_representative_of_bank_member_diversity",
        "reason": (
            "replication preserves the width and compiled compute shape but makes "
            "qualification all-or-none across 1024 identical lanes; it cannot sample "
            "the two-machine, member-dependent qualification incidence in regeneration"
        ),
        "valid_use": "fixed-shape H200 throughput and bit-identity comparison",
        "invalid_use": (
            "estimating the fraction of regeneration members that exit early"
        ),
    }


def _draw(payload: dict[str, Any], output: Path) -> None:
    qualification = payload["qualified_suffix_census"]
    criteria = payload["projected_cost_by_criterion"]["observed_telemetry_cohort"]
    figure = plt.figure(figsize=(12.2, 7.4), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, height_ratios=(1.0, 1.45))
    fraction_axis = figure.add_subplot(grid[0, 0])
    histogram_axis = figure.add_subplot(grid[0, 1])
    cost_axis = figure.add_subplot(grid[1, :])

    bank_names = ["MAST", "DIII-D", "Combined"]
    fractions = [
        qualification["per_bank"]["MAST"]["strict_qualified_fraction_of_observable"],
        qualification["per_bank"]["DIII-D"]["strict_qualified_fraction_of_observable"],
        qualification["combined"]["strict_qualified_fraction_of_observable"],
    ]
    counts = [(0, 6), (5, 5), (5, 11)]
    bars = fraction_axis.bar(
        bank_names, fractions, color=["#d55e00", "#0072b2", "#5b5b5b"]
    )
    fraction_axis.set_ylim(0, 1.13)
    fraction_axis.set_ylabel("strict-qualified fraction")
    fraction_axis.set_title("Strict qualification among observable histories")
    for bar, (qualified, observed) in zip(bars, counts, strict=True):
        fraction_axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.035,
            f"{qualified}/{observed}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    fraction_axis.text(
        0.0,
        -0.22,
        "MAST also has 6 mixed arms with no trip telemetry.",
        transform=fraction_axis.transAxes,
        fontsize=9,
    )

    histogram = qualification["combined"]["first_qualification_trip_histogram"]
    trips = list(range(1, 7))
    histogram_axis.bar(
        trips,
        [histogram.get(str(trip), 0) for trip in trips],
        color="#0072b2",
    )
    histogram_axis.set_xticks(trips)
    histogram_axis.set_xlabel("first strict-qualification trip")
    histogram_axis.set_ylabel("members")
    histogram_axis.set_title("All five strict suffixes are DIII-D")

    labels = [
        "Full 16",
        "Strict",
        "Mask only\n(counterfactual)",
        "Exact\nplateau",
        "Residual\n1%",
        "Residual\n5%",
    ]
    costs = [row["projected_ms_per_slice"] for row in criteria]
    colors = ["#5b5b5b", "#0072b2", "#cc79a7", "#009e73", "#56b4e9", "#e69f00"]
    bars = cost_axis.bar(labels, costs, color=colors)
    cost_axis.axhline(TARGET_MS, color="#222222", linestyle="--", linewidth=1.2)
    cost_axis.text(5.48, TARGET_MS + 0.035, "1 ms target", ha="right", fontsize=9)
    cost_axis.set_ylabel("projected ms/slice")
    cost_axis.set_title(
        "Observed 11-member cohort; unavailable mixed arms are separate in JSON"
    )
    cost_axis.set_ylim(0, 1.92)
    for bar, row in zip(bars, criteria, strict=True):
        cost_axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.035,
            f"{bar.get_height():.3f}\n{row['projected_speedup_x']:.2f}x",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    figure.suptitle(
        "Qualified suffixes are workload-dependent: replicated width is not "
        "member diversity",
        fontsize=14,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def run(output_json: Path, output_png: Path) -> dict[str, Any]:
    settlement = _read_json(SETTLEMENT_RECEIPT)
    throughput = _read_json(THROUGHPUT_RECEIPT)
    expected = throughput["evidence_inputs"]
    for key, path in (
        ("settlement_histogram", SETTLEMENT_RECEIPT),
        ("mast_bank", ROOT / expected["mast_bank"]["path"]),
        ("diiid_bank", ROOT / expected["diiid_bank"]["path"]),
    ):
        actual = _sha256(path)
        if actual != expected[key]["sha256"]:
            raise ValueError(f"{key} digest changed: {actual}")

    qualification, strict_trips = _qualification_census(settlement, throughput)
    projections = _criterion_table(settlement, throughput, strict_trips)
    payload = {
        "schema": "nova.qualified_suffix_census/1",
        "recorded_at": "2026-09-01",
        "question": (
            "why the strict settled exit never fires on the width-1024 workload"
        ),
        "solver_source_modified": False,
        "implementation_dispatched": False,
        "constants": {
            "trip_limit": TRIP_LIMIT,
            "full_trip_floor_ms_per_slice": FULL_TRIP_FLOOR_MS,
            "target_ms_per_slice": TARGET_MS,
            "measured_full_trip_control_ms_per_member": throughput["h200_width_1024"][
                "full_trip_control"
            ]["steady"]["median_ms_per_member"],
        },
        "evidence_inputs": {
            "settlement_histogram": {
                "path": str(SETTLEMENT_RECEIPT.relative_to(ROOT)),
                "sha256": _sha256(SETTLEMENT_RECEIPT),
            },
            "settled_exit_throughput": {
                "path": str(THROUGHPUT_RECEIPT.relative_to(ROOT)),
                "sha256": _sha256(THROUGHPUT_RECEIPT),
            },
            "mast_bank": expected["mast_bank"],
            "diiid_bank": expected["diiid_bank"],
        },
        "qualified_suffix_census": qualification,
        "workload_representativeness": _representativeness(settlement, throughput),
        "projected_cost_by_criterion": projections,
        "headline": {
            "strict_qualified_observable": "5/11 (45.5%)",
            "strict_confirmed_declared_bank_lower_bound": "5/17 (29.4%)",
            "machine_split": "DIII-D 5/5; MAST 0/6 observable, 6/12 unavailable",
            "width_1024_verdict": (
                "one member replicated 1024 times is not representative of bank "
                "member diversity or exit incidence"
            ),
            "strict_observed_projection_ms_per_slice": next(
                row["projected_ms_per_slice"]
                for row in projections["observed_telemetry_cohort"]
                if row["criterion"] == "strict_qualified_noop"
            ),
        },
        "policy_question_for_lead": (
            "Keep the strict no-op policy, whose heterogeneous observed-bank ceiling "
            "is 1.163 ms/slice (1.370 ms/slice when six telemetry-unavailable mixed "
            "arms are conservatively charged 16 trips), or authorize a separately "
            "validated residual-progress rule? A 1% hindsight tolerance projects "
            "0.915 ms/slice on observable histories but does not preserve bit identity "
            "by construction; select the tolerated state/residual error and causal "
            "progress window before any implementation. The mask-only 0.467 ms/slice "
            "counterfactual is not semantically admissible."
        ),
        "project_absolute_figure_src": (
            "/nova/figures/solver-trip-orchestration/suffix-census.png"
        ),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _draw(payload, output_png)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Census policy-qualified suffixes from committed histories."
    )
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--png", type=Path, default=DEFAULT_PNG)
    arguments = parser.parse_args()
    payload = run(arguments.json, arguments.png)
    combined = payload["qualified_suffix_census"]["combined"]
    strict = payload["headline"]["strict_observed_projection_ms_per_slice"]
    print(
        "QUALIFIED_SUFFIX_CENSUS "
        f"qualified={combined['strict_qualified_members']}/"
        f"{combined['telemetry_observable_members']} "
        f"strict_ms_per_slice={strict:.6f} "
        f"representative={payload['workload_representativeness']['verdict']}"
    )


if __name__ == "__main__":
    main()
