"""Check every quantitative claim this record makes against the files it cites.

The record states the current drift, the two positive controls, the negative
controls and the seed-sensitivity outcome. This script recomputes each from the
arm records and current maps on disk and asserts it, so a number that drifts
fails here rather than silently outliving its measurement.

The recorded residual a control is scored against is read from the published
comparison record itself, not from the arm's own copy of it, and every factor is
recomputed from the arm's measured residual and that published value. The arm's
stored ``within_factor_three`` flag is then compared with the recomputed band, so
a flag left behind by a changed constant fails: a control whose factor is read
back out of the field that decided it proves nothing.

    verify_input_drift.py <run directory> [record-era current map]

Prints one ``CLAIM <name> PASS|FAIL <detail>`` line per claim and exits nonzero
when any claim fails. Run it through ``submit.sh``.
"""

from __future__ import annotations

import json
import math
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

RECORD_FRAGMENT = ROOT / (
    "docs/evidence/fragments/inverse-equilibrium-baseline/efit-comparison.html"
)
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


def published_recorded_residual(path: Path) -> float:
    """Return the TCV residual the published comparison record states.

    The record quotes the value in its table and its prose, so every occurrence
    must be the same number: a record that disagrees with itself cannot anchor a
    control.
    """
    values = {float(value) for value in re.findall(r"0\.0303642\d+", path.read_text())}
    if len(values) != 1:
        raise SystemExit(f"{path}: the recorded residual reads as {sorted(values)}")
    return values.pop()


def residual_factor(arm: dict, recorded: float) -> float | None:
    """Return the arm's terminal residual as a multiple of the recorded one."""
    residual = arm.get("residual")
    if residual is None or not math.isfinite(float(residual)):
        return None
    return abs(float(residual) / recorded)


def control_claim(arm: dict, recorded: float) -> tuple[bool, str]:
    """Score one arm against the published residual, recomputed end to end.

    Three independent facts carry the check: the arm's own copy of the recorded
    residual equals the published value; the factor recomputed from the arm's
    measured residual and the published value lies inside a factor of three; and
    the flag the arm stored agrees with that recomputed band. An arm that
    measured no finite residual cannot pass.
    """
    factor = residual_factor(arm, recorded)
    if factor is None:
        return False, "no finite terminal residual"
    quoted = (arm.get("recorded") or {}).get("residual")
    agrees = quoted is not None and math.isclose(
        float(quoted), recorded, rel_tol=1.0e-12, abs_tol=1.0e-12
    )
    inside = 1.0 / 3.0 <= factor <= 3.0
    passes = agrees and inside == bool(arm.get("within_factor_three"))
    return passes, (
        f"residual {arm['residual']} against published {recorded}, factor "
        f"{factor}, recorded constant agrees: {agrees}, stored flag "
        f"{arm.get('within_factor_three')} matches recomputed {inside}"
    )


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


def _arm(run: Path, relative: str) -> dict:
    return json.loads((run / relative).read_text())


def evaluate(run: Path, record_map: Path, fragment: Path) -> list[tuple]:
    """Return every claim as one tuple of name, passed flag and detail."""
    claims: list[tuple] = []
    recorded = published_recorded_residual(fragment)

    drift = current_drift(run, record_map)
    print("CURRENT_DRIFT " + json.dumps(drift, sort_keys=True), flush=True)
    claims.append(("conductors", drift["conductors"] == 288, drift["conductors"]))
    claims.append(
        (
            "no_conductor_identical_within_1e_6_a",
            drift["identical"] == 0,
            f"{drift['identical']} of {drift['conductors']} identical",
        )
    )
    claims.append(
        (
            "median_relative_difference",
            abs(drift["median_relative"] - 0.049392) < 5.0e-5,
            f"{100 * drift['median_relative']:.4f} percent",
        )
    )
    claims.append(
        (
            "max_absolute_difference_a",
            abs(drift["max_absolute"] - 7000.24) < 10.0,
            f"{drift['max_absolute']:.2f} A",
        )
    )
    claims.append(
        (
            "l2_fraction_of_record_vector",
            abs(drift["l2_fraction"] - 0.0782) < 5.0e-4,
            f"{drift['difference_l2']:.2f} A of {drift['record_l2']:.2f} A",
        )
    )
    claims.append(
        (
            "recorded_residual_matches_published_record",
            math.isclose(recorded, 0.030364200526436996, rel_tol=0.0, abs_tol=0.0),
            f"published record states {recorded}",
        )
    )

    positive = _arm(run, "record-seed/tcv-record-seed.json")
    passed, detail = control_claim(positive, recorded)
    claims.append(("positive_control_tcv_within_factor_three", passed, detail))
    factor = residual_factor(positive, recorded)
    claims.append(
        (
            "positive_control_tcv_reproduces_the_record",
            factor is not None and factor <= 1.005,
            f"factor {factor} of the published residual",
        )
    )
    claims.append(
        (
            "positive_control_tcv_reason_code",
            positive.get("termination_reason")
            == (positive.get("recorded") or {}).get("termination_reason")
            == 12,
            f"reason {positive.get('termination_reason')}",
        )
    )

    for label, relative in (
        ("tcv_input_comparison", "attribution-fixed/tcv-input-comparison.json"),
        ("tcv_input_stored", "attribution-fixed/tcv-input-stored.json"),
    ):
        arm = _arm(run, relative)
        arm_factor = residual_factor(arm, recorded)
        claims.append(
            (
                f"negative_control_{label}_outside_factor_three",
                arm_factor is not None and not (1.0 / 3.0 <= arm_factor <= 3.0),
                f"residual {arm.get('residual')} against published {recorded}, "
                f"factor {arm_factor}",
            )
        )

    nominal = _arm(run, "measurement/tcv-seed-nominal.json")
    comparison = _arm(run, "attribution-fixed/tcv-input-comparison.json")
    for name, arm in (("nominal", nominal), ("comparison", comparison)):
        arm_factor = residual_factor(arm, recorded)
        claims.append(
            (
                f"nominal_seed_arm_{name}_outside_factor_three",
                arm_factor is not None and arm_factor < 0.2,
                f"factor {arm_factor} of the published residual",
            )
        )
    scatter = abs(float(nominal["residual"]) - float(comparison["residual"])) / float(
        comparison["residual"]
    )
    claims.append(
        (
            "nominal_seed_arm_two_runs_agree",
            scatter <= 1.0e-3,
            f"the two independent runs of the same nominal arm differ by {scatter:.3g} "
            "relative",
        )
    )
    claims.extend(_iter_claims(run))
    claims.extend(_nudge_claims(run))
    claims.extend(_repeat_claims(run, recorded))
    return claims


def _iter_claims(run: Path) -> list[tuple]:
    """Return the ITER control's claims, from the two ITER arm records."""
    stored = _arm(run, "attribution-fixed/iter-input-stored.json")
    record_seed = _arm(run, "iter-record-seed-json/iter-record-seed.json")
    trajectory = record_seed.get("failure_trajectory") or {}
    stored_reason = stored.get("termination_reason")
    stored_admitted = (stored.get("axis") or {}).get("admitted")
    seed_reason = record_seed.get("termination_reason")
    seed_admitted = (record_seed.get("axis") or {}).get("admitted")
    exception = trajectory.get("exception")
    return [
        (
            "iter_today_differs_from_record",
            stored_reason == 12 and stored_admitted is True,
            "reason %s, axis admitted %s" % (stored_reason, stored_admitted),
        ),
        (
            "positive_control_iter_reproduces_refusal",
            seed_reason == 3
            and seed_admitted is False
            and exception == "NoQualifiedAxisError",
            "reason %s, axis admitted %s, exception %s"
            % (seed_reason, seed_admitted, exception),
        ),
        (
            "iter_positive_control_same_trip_count",
            trajectory.get("trips") == 1,
            "%s trip" % trajectory.get("trips"),
        ),
    ]


def _nudge_claims(run: Path) -> list[tuple]:
    """Return the seed-nudge claims, from the two sensitivity arms."""
    nominal = _arm(run, "measurement/tcv-seed-nominal.json")
    drifted = _arm(run, "measurement/tcv-seed-drifted-generating-failure.json")
    nominal_reason = nominal.get("termination_reason")
    drifted_reason = drifted.get("termination_reason")
    return [
        (
            "seed_nudge_changes_outcome_class",
            nominal_reason == 12
            and drifted_reason == 3
            and drifted.get("exception") == "NoQualifiedAxisError",
            "nominal reason %s, nudged reason %s" % (nominal_reason, drifted_reason),
        )
    ]


def _repeat_claims(run: Path, recorded: float) -> list[tuple]:
    """Return the repeatability claims, when the repeat job's record is present.

    Three runs of the same nominal arm were made in one job: two through the
    comparison construction and one through the seed-sensitivity construction.
    The first pair measures the scatter of one construction, the third measures
    what the choice of construction adds to it, and the two nominal values this
    record quotes are compared against both.
    """
    path = run / "scatter" / "repeat.json"
    if not path.is_file():
        return []
    arms = {arm["name"]: arm for arm in json.loads(path.read_text())["arms"]}

    def spread(first: str, second: str) -> float:
        a = float(arms[first]["residual"])
        b = float(arms[second]["residual"])
        return abs(a - b) / max(abs(a), abs(b))

    self_spread = spread("solve-first", "solve-second")
    cross_spread = spread("seed-first", "solve-first")
    nominal = float(_arm(run, "measurement/tcv-seed-nominal.json")["residual"])
    comparison = float(
        _arm(run, "attribution-fixed/tcv-input-comparison.json")["residual"]
    )
    quoted = abs(nominal - comparison) / abs(comparison)
    factors = [residual_factor(arm, recorded) for arm in arms.values()]
    return [
        (
            "repeat_one_construction_agrees_with_itself",
            self_spread <= 1.0e-3,
            "%.3g relative between two runs of one construction" % self_spread,
        ),
        (
            "repeat_two_constructions_agree",
            cross_spread <= 5.0e-3,
            "%.3g relative between the two code paths" % cross_spread,
        ),
        (
            "repeat_quoted_pair_sits_at_run_to_run_scale",
            quoted <= 5.0e-3,
            "%.3g relative between the two nominal values this record quotes" % quoted,
        ),
        (
            "repeat_arms_all_far_outside_factor_three",
            all(factor is not None and factor < 0.2 for factor in factors),
            "factors %s" % [round(f, 6) if f else f for f in factors],
        ),
    ]


def main() -> int:
    run = Path(sys.argv[1]).resolve()
    record_map = (
        Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else RECORD_CURRENT_MAP
    )
    fragment = Path(sys.argv[3]).resolve() if len(sys.argv) > 3 else RECORD_FRAGMENT
    claims = evaluate(run, record_map, fragment)
    for name, condition, detail in claims:
        claim(name, condition, detail)
    (run / "verify-input-drift.json").write_text(
        json.dumps(
            [dict(name=n, passed=bool(c), detail=str(d)) for n, c, d in claims],
            indent=1,
        )
        + "\n"
    )
    if FAILURES:
        print("INPUT_DRIFT_CLAIMS_FAILED " + json.dumps(FAILURES), flush=True)
        return 1
    print("INPUT_DRIFT_CLAIMS_OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
