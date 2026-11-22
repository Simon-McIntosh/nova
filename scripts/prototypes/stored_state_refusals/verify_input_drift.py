"""Check every quantitative claim this record makes against the files it cites.

The record states the current drift, the two positive controls, the negative
controls and the seed-sensitivity outcome. This script recomputes each from the
arm records and current maps on disk and asserts it, so a number that drifts
fails here rather than silently outliving its measurement.

    verify_input_drift.py <run directory> [record-era current map]

Prints one ``CLAIM <name> PASS|FAIL <detail>`` line per claim and exits nonzero
when any claim fails. Run it through ``submit.sh``.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

RECORD_CURRENT_MAP = Path(
    "/home/ITER/mcintos/.cache/nova-inverse-equilibrium-baseline/"
    "efit-comparison-run-copy/fitted-physical-measurement-1318907/"
    "physical-tcv/tcv-fitted-current-map.json"
)

FAILURES: list[str] = []


def claim(name: str, condition: bool, detail: str) -> None:
    print(f"CLAIM {name} {'PASS' if condition else 'FAIL'} {detail}", flush=True)
    if not condition:
        FAILURES.append(name)


def _arm(run: Path, relative: str) -> dict:
    return json.loads((run / relative).read_text())


def current_drift(run: Path, record_map: Path) -> dict:
    """Recompute the TCV fitted-current drift from the two published maps."""
    today = json.loads(
        (run / "attribution-fixed/tcv-fitted-current-map.json").read_text()
    )
    record = json.loads(record_map.read_text())
    today_values = today["effective_current_a"]
    record_values = record["effective_current_a"]
    names = list(today_values)
    missing = [name for name in names if name not in record_values]
    if missing:
        raise SystemExit(f"the record map is missing {len(missing)} conductors")
    pairs = [(float(today_values[name]), float(record_values[name])) for name in names]
    difference = [abs(today - record) for today, record in pairs]
    relative = [abs(today - record) / max(abs(record), 1.0) for today, record in pairs]
    record_norm = math.sqrt(sum(float(record_values[name]) ** 2 for name in names))
    difference_norm = math.sqrt(sum(value * value for value in difference))
    return dict(
        conductors=len(names),
        identical=int(sum(1 for value in difference if value < 1.0e-6)),
        median_relative=statistics.median(relative),
        max_relative=max(relative),
        max_absolute=max(difference),
        difference_l2=difference_norm,
        record_l2=record_norm,
        l2_fraction=difference_norm / record_norm,
    )


def main() -> int:
    run = Path(sys.argv[1]).resolve()
    record_map = (
        Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else RECORD_CURRENT_MAP
    )

    drift = current_drift(run, record_map)
    print("CURRENT_DRIFT " + json.dumps(drift, sort_keys=True), flush=True)
    claim("conductors", drift["conductors"] == 288, f"{drift['conductors']} conductors")
    claim(
        "no_conductor_identical_within_1e_6_a",
        drift["identical"] == 0,
        f"{drift['identical']} of {drift['conductors']} identical",
    )
    claim(
        "median_relative_difference",
        abs(drift["median_relative"] - 0.049392) < 5.0e-5,
        f"{100 * drift['median_relative']:.4f} percent",
    )
    claim(
        "max_absolute_difference_a",
        abs(drift["max_absolute"] - 7000.24) < 10.0,
        f"{drift['max_absolute']:.2f} A",
    )
    claim(
        "l2_fraction_of_record_vector",
        abs(drift["l2_fraction"] - 0.0782) < 5.0e-4,
        f"{drift['difference_l2']:.2f} A of {drift['record_l2']:.2f} A",
    )

    record_seed = _arm(run, "record-seed/tcv-record-seed.json")
    recorded = record_seed["recorded"]
    factor = record_seed["factor_of_recorded_residual"]
    claim(
        "positive_control_tcv_within_factor_three",
        bool(record_seed["within_factor_three"]) and 1.0 / 3.0 <= factor <= 3.0,
        f"residual {record_seed['residual']} against {recorded['residual']}, "
        f"factor {factor}",
    )
    claim(
        "positive_control_tcv_reason_code",
        record_seed["termination_reason"] == recorded["termination_reason"] == 12,
        f"reason {record_seed['termination_reason']}",
    )

    comparison = _arm(run, "attribution-fixed/tcv-input-comparison.json")
    stored = _arm(run, "attribution-fixed/tcv-input-stored.json")
    for arm, entry in (
        ("tcv_input_comparison", comparison),
        ("tcv_input_stored", stored),
    ):
        claim(
            f"negative_control_{arm}_outside_factor_three",
            not entry["within_factor_three"],
            f"residual {entry['residual']} against {recorded['residual']}, "
            f"factor {entry['factor_of_recorded_residual']}",
        )

    iter_stored = _arm(run, "attribution-fixed/iter-input-stored.json")
    claim(
        "iter_today_differs_from_record",
        iter_stored["termination_reason"] == 12
        and iter_stored["axis"]["admitted"] is True,
        f"reason {iter_stored['termination_reason']}, axis admitted "
        f"{iter_stored['axis'].get('admitted')}",
    )
    iter_record = _arm(run, "iter-record-seed-json/iter-record-seed.json")
    claim(
        "positive_control_iter_reproduces_refusal",
        iter_record["termination_reason"] == 3
        and iter_record["axis"]["admitted"] is False
        and iter_record["failure_trajectory"]["exception"] == "NoQualifiedAxisError",
        f"reason {iter_record['termination_reason']}, exception "
        f"{iter_record['failure_trajectory']['exception']}",
    )
    claim(
        "iter_positive_control_same_trip_count",
        iter_record["failure_trajectory"]["trips"] == 1,
        f"{iter_record['failure_trajectory']['trips']} trip",
    )

    nominal = _arm(run, "measurement/tcv-seed-nominal.json")
    drifted = _arm(run, "measurement/tcv-seed-drifted-generating-failure.json")
    claim(
        "seed_nudge_changes_outcome_class",
        nominal["termination_reason"] == 12
        and drifted["termination_reason"] == 3
        and drifted["exception"] == "NoQualifiedAxisError",
        f"nominal reason {nominal['termination_reason']}, nudged reason "
        f"{drifted['termination_reason']}",
    )

    if FAILURES:
        print("INPUT_DRIFT_CLAIMS_FAILED " + json.dumps(FAILURES), flush=True)
        return 1
    print("INPUT_DRIFT_CLAIMS_OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
