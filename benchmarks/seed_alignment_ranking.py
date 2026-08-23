"""Retest the locked seed diagnostic on a preregistered enlarged corpus."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import numpy as np
from scipy.stats import norm, spearmanr

from benchmarks.diiid_constrained_cold_start import _solve_public_seam
from benchmarks.efit_parity_warm_neighbour import _prepare_frame
from benchmarks.parity_divergence_attribution import (
    NONLINEAR_UPDATES,
    _build_case,
    _case_rows,
    _difference,
    _solve_pair,
)
from nova.equilibrium.seed_alignment import residual_action_amplification
from nova.imas.mast_solve_inputs import SHOT_STORE
from nova.jax.config import configure_dtypes


OUTPUT = Path("docs/figures/topology-preserving-continuation/seed-alignment-power.json")
LOCKED_RECEIPT = Path(
    "docs/figures/topology-preserving-continuation/seed-alignment-ranking.json"
)
GROWTH_BANK = Path(
    "docs/figures/mast-catalog-gpu-solve/parity-divergence-attribution.json"
)
TERMINAL_BANK = Path("docs/figures/dual-basin-solve/newton-warm-ladder-replay.json")
SELECTION_BANK = Path(
    "docs/figures/efit-flux-decomposition/native-grid-decomposition.json"
)
BANKED_CASES = (
    (21978, 35),
    (21983, 35),
    (21985, 51),
    (21986, 46),
    (21989, 55),
)
LOCKED_DEFINITION = (
    "For seed s, fixed-point map F, residual r = F(s) - s, and local "
    "action A = I - J_F(s), score q(s,F) = ||r||_2 / ||A r||_2."
)
BANKED_ASSOCIATIONS = {
    "spearman_against_cumulative_growth": {
        "statistic": 0.60,
        "two_sided_pvalue": 0.28475697986529386,
        "sample_count": 5,
    },
    "spearman_against_terminal_residual": {
        "statistic": 0.00,
        "two_sided_pvalue": 1.0,
        "sample_count": 5,
    },
}
ALPHA = 0.05


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _text_digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], text=True, stderr=subprocess.DEVNULL
    ).strip()


def _write(output: Path, payload: dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def _selected_cases(store: Path) -> tuple[tuple[int, int], ...]:
    frozen = tuple(sorted(_case_rows(store)))
    if set(BANKED_CASES) - set(frozen):
        raise RuntimeError("the banked five are not contained in the frozen cohort")
    following = tuple((shot, slice_index + 1) for shot, slice_index in frozen[:2])
    selected = tuple(sorted(set(BANKED_CASES) | set(frozen) | set(following)))
    if len(frozen) != 6 or len(selected) != 8:
        raise RuntimeError("the declared enlargement must yield eight unique frames")
    return selected


def _preregistration(
    selected: tuple[tuple[int, int], ...], recorded_at: str
) -> dict[str, Any]:
    added = tuple(case for case in selected if case not in BANKED_CASES)
    return {
        "recorded_utc": recorded_at,
        "recorded_before_any_added_frame_score": True,
        "recorded_before_any_new_outcome_measurement": True,
        "frame_selection_rule": (
            "Retain the locked five-frame corpus; take the complete six-shot "
            "frozen held-out cohort selected by native-grid-decomposition; then "
            "sort that cohort by shot and add the immediately following slice "
            "from its first two shots. Sort unique (shot, slice_index) pairs. "
            "No observable score, growth, or terminal residual enters selection."
        ),
        "selection_authority": {
            "path": str(SELECTION_BANK),
            "sha256": _digest(SELECTION_BANK),
        },
        "banked_cases": [
            {"shot": shot, "slice_index": slice_index}
            for shot, slice_index in BANKED_CASES
        ],
        "added_cases": [
            {"shot": shot, "slice_index": slice_index} for shot, slice_index in added
        ],
        "enlarged_corpus": [
            {"shot": shot, "slice_index": slice_index} for shot, slice_index in selected
        ],
        "banked_sample_count": len(BANKED_CASES),
        "added_sample_count": len(added),
        "enlarged_sample_count": len(selected),
        "observable": {
            "name": "seed residual-action amplification",
            "definition": LOCKED_DEFINITION,
            "definition_sha256": _text_digest(LOCKED_DEFINITION),
            "ranking_direction": (
                "ascending is preferred; larger predicts worse outcome"
            ),
            "inputs": ["candidate seed", "fixed-point map at seed", "map tangent"],
            "outcome_exclusions": [
                "converged solve",
                "terminal state",
                "terminal residual",
                "cumulative growth",
                "reference equilibrium",
            ],
            "singular_convention": (
                "an exact fixed point scores zero; nonzero residual with zero "
                "action scores positive infinity"
            ),
            "locked_source": {
                "path": str(LOCKED_RECEIPT),
                "sha256": _digest(LOCKED_RECEIPT),
            },
            "changed_from_locked_source": False,
        },
        "association_test": (
            "two-sided Spearman rank correlation on all selected frames"
        ),
        "significance_alpha": ALPHA,
        "power_definition": (
            "two-sided Fisher-z normal approximation at alpha 0.05; reported "
            "both at the banked effect sizes and at the enlarged observed effects"
        ),
    }


def _validate_locked_receipt(receipt: dict[str, Any]) -> None:
    locked = receipt["preregistration"]["observable"]
    if locked["definition"] != LOCKED_DEFINITION:
        raise RuntimeError("the carried observable differs from its locked definition")
    for name, expected in BANKED_ASSOCIATIONS.items():
        actual = receipt["ranking"][name]
        if actual != expected:
            raise RuntimeError(f"the banked association changed for {name}")


def _banked_rows() -> dict[tuple[int, int], dict[str, Any]]:
    receipt = _read(LOCKED_RECEIPT)
    _validate_locked_receipt(receipt)
    rows = {
        (int(row["shot"]), int(row["slice_index"])): {
            "shot": int(row["shot"]),
            "slice_index": int(row["slice_index"]),
            "time_s": float(row["time_s"]),
            "cumulative_growth": float(row["cumulative_growth"]),
            "terminal_residual": float(row["terminal_residual"]),
            "outcome_source": "locked five-frame receipt",
        }
        for row in receipt["ranking"]["rows"]
    }
    if set(rows) != set(BANKED_CASES):
        raise RuntimeError(
            "the locked receipt does not contain exactly the banked five"
        )
    return rows


def _sixth_banked_row() -> dict[str, Any]:
    growth_receipt = _read(GROWTH_BANK)
    terminal_receipt = _read(TERMINAL_BANK)
    growth_rows = {
        (int(row["shot"]), int(row["slice_index"])): row
        for row in growth_receipt["current_source_measurement"]["cases"]
    }
    terminal_rows = {
        (
            int(row["reference"]["shot"]),
            int(row["reference"]["slice_index"]),
        ): row
        for row in terminal_receipt["references"]
    }
    key = (22086, 43)
    growth = growth_rows[key]
    terminal = terminal_rows[key]["cold_newton_control"]
    first = float(growth["single_map_application"]["maximum_absolute_difference"])
    last = float(growth["solved_flux"]["maximum_absolute_difference"])
    return {
        "shot": key[0],
        "slice_index": key[1],
        "time_s": float(growth["time_s"]),
        "cumulative_growth": math.inf if first == 0.0 else last / first,
        "terminal_residual": float(terminal["terminal_fixed_point_residual"]),
        "outcome_source": "previously banked frozen-six receipts",
    }


def _measure_growth(store: Path, shot: int, slice_index: int) -> dict[str, Any]:
    profile, seed, target_current, time_s = _build_case(store, shot, slice_index)
    mapped = profile.flux_map(target_current=target_current)
    eager_map = mapped(seed.flux)
    compiled_map = jax.jit(mapped)(seed.flux)
    jax.block_until_ready(compiled_map)
    first = _difference(eager_map, compiled_map)
    eager, compiled = _solve_pair(profile, seed.flux, target_current, NONLINEAR_UPDATES)
    last = _difference(eager.flux, compiled.flux)
    initial_difference = float(first["maximum_absolute_difference"])
    terminal_difference = float(last["maximum_absolute_difference"])
    if initial_difference == 0.0:
        raise RuntimeError(
            f"zero initial eager-compiled difference for {shot}/{slice_index}"
        )
    return {
        "time_s": time_s,
        "cumulative_growth": terminal_difference / initial_difference,
        "growth_measurement": {
            "nonlinear_updates": NONLINEAR_UPDATES,
            "initial_eager_compiled_maximum_absolute_difference": initial_difference,
            "terminal_eager_compiled_maximum_absolute_difference": terminal_difference,
            "eager_terminal_residual": float(eager.fixed_point.residual),
            "compiled_terminal_residual": float(compiled.fixed_point.residual),
        },
    }


def _measure_terminal(
    store: Path, shot: int, slice_index: int, cache_box: list[Any]
) -> dict[str, Any]:
    frame, _mast_case, _context = _prepare_frame(store, shot, slice_index, cache_box)
    outcome = _solve_public_seam(frame, frame.seed)
    return {
        "terminal_residual": float(outcome.residual),
        "terminal_measurement": {
            "iterations": int(outcome.iterations),
            "termination": outcome.termination,
            "achieved_current_a": float(outcome.achieved_current_a),
        },
    }


def _measure_added_outcomes(
    output: Path,
    store: Path,
    preregistration: dict[str, Any],
    selected: tuple[tuple[int, int], ...],
) -> dict[tuple[int, int], dict[str, Any]]:
    outcomes = _banked_rows()
    sixth = _sixth_banked_row()
    outcomes[(sixth["shot"], sixth["slice_index"])] = sixth
    dynamic = [case for case in selected if case not in outcomes]
    cache_box: list[Any] = [None]
    for index, (shot, slice_index) in enumerate(dynamic, start=1):
        growth = _measure_growth(store, shot, slice_index)
        terminal = _measure_terminal(store, shot, slice_index, cache_box)
        outcomes[(shot, slice_index)] = {
            "shot": shot,
            "slice_index": slice_index,
            **growth,
            **terminal,
            "outcome_source": "current-tree preregistered measurement",
        }
        _write(
            output,
            {
                "schema": "nova-seed-alignment-power/1.0",
                "status": "outcomes_in_progress",
                "preregistration": preregistration,
                "outcomes_banked_before_scoring": [
                    outcomes[key] for key in sorted(outcomes)
                ],
            },
        )
        print(
            f"banked added outcome {index}/{len(dynamic)}: {shot}/{slice_index}",
            flush=True,
        )
    if set(outcomes) != set(selected):
        raise RuntimeError("the enlarged outcome bank is incomplete")
    return outcomes


def _score_cases(
    store: Path, selected: tuple[tuple[int, int], ...]
) -> dict[tuple[int, int], dict[str, Any]]:
    locked = _read(LOCKED_RECEIPT)
    _validate_locked_receipt(locked)
    rows: dict[tuple[int, int], dict[str, Any]] = {
        (int(row["shot"]), int(row["slice_index"])): {
            "seed_residual_action_amplification": float(
                row["seed_residual_action_amplification"]
            ),
            "score_time_s": float(row["time_s"]),
            "score_source": "locked five-frame receipt",
        }
        for row in locked["ranking"]["rows"]
    }
    if set(rows) != set(BANKED_CASES):
        raise RuntimeError("the locked score bank does not match its five cases")
    added = [case for case in selected if case not in rows]
    cache_box: list[Any] = [None]
    for index, (shot, slice_index) in enumerate(added, start=1):
        frame, mast_case, _context = _prepare_frame(store, shot, slice_index, cache_box)
        target_current = frame.selected.recorded_plasma_current_a
        map_fn = frame.profile.flux_map(target_current=target_current)
        score = residual_action_amplification(map_fn, frame.seed)
        rows[(shot, slice_index)] = {
            "seed_residual_action_amplification": float(score),
            "score_time_s": float(mast_case["reference"]["time_s"]),
            "score_source": "current-tree unchanged observable",
        }
        print(
            f"computed added locked score {index}/{len(added)}: {shot}/{slice_index}",
            flush=True,
        )
    return rows


def _correlation(rows: list[dict[str, Any]], outcome: str) -> dict[str, Any]:
    result = spearmanr(
        [row["seed_residual_action_amplification"] for row in rows],
        [row[outcome] for row in rows],
    )
    return {
        "statistic": float(result.statistic),
        "two_sided_pvalue": float(result.pvalue),
        "sample_count": len(rows),
    }


def _fisher_power(effect: float, sample_count: int) -> float:
    bounded = min(abs(effect), 1.0 - np.finfo(float).eps)
    noncentrality = np.arctanh(bounded) * math.sqrt(sample_count - 3)
    critical = float(norm.ppf(1.0 - ALPHA / 2.0))
    return float(
        norm.sf(critical - noncentrality) + norm.cdf(-critical - noncentrality)
    )


def _scientific_verdict(
    growth: dict[str, Any], residual: dict[str, Any], planned_growth_power: float
) -> dict[str, Any]:
    growth_passes = bool(
        growth["statistic"] > 0.0 and growth["two_sided_pvalue"] < ALPHA
    )
    residual_passes = bool(
        residual["statistic"] > 0.0 and residual["two_sided_pvalue"] < ALPHA
    )
    if growth_passes and residual_passes:
        status = "SUPPORTED_ON_BOTH_OUTCOMES"
    elif growth_passes or residual_passes:
        status = "PARTIAL_ONE_OUTCOME"
    elif planned_growth_power < 0.8:
        status = "UNDERPOWERED_NEGATIVE"
    else:
        status = "NEGATIVE_WITH_ADEQUATE_POWER"
    return {
        "status": status,
        "promote_as_seed_selector": growth_passes and residual_passes,
        "growth_reaches_positive_significance": growth_passes,
        "residual_reaches_positive_significance": residual_passes,
        "failing_outcomes": [
            name
            for name, passes in (
                ("cumulative_growth", growth_passes),
                ("terminal_residual", residual_passes),
            )
            if not passes
        ],
        "reason": (
            "Promotion still requires a positive, significant rank association "
            "against both outcomes. The enlarged result is closed at its measured "
            "power even when that requirement is not met."
        ),
    }


def measure(output: Path, store: Path) -> dict[str, Any]:
    """Bank selected outcomes, then score the unchanged observable once."""

    configure_dtypes()
    selected = _selected_cases(store)
    prior = _read(output) if output.exists() else None
    selected_records = [
        {"shot": shot, "slice_index": slice_index} for shot, slice_index in selected
    ]
    can_resume_scoring = bool(
        prior is not None
        and prior.get("status") == "outcomes_banked_before_scoring"
        and prior["preregistration"]["enlarged_corpus"] == selected_records
        and prior["preregistration"]["observable"]["definition"] == LOCKED_DEFINITION
        and len(prior["outcomes"]) == len(selected)
    )
    if can_resume_scoring:
        preregistration = prior["preregistration"]
        preregistered_at = preregistration["recorded_utc"]
        outcomes_banked_at = prior["outcomes_banked_utc"]
        outcomes = {
            (int(row["shot"]), int(row["slice_index"])): row
            for row in prior["outcomes"]
        }
        print("resuming from complete outcome bank; scoring begins", flush=True)
    else:
        preregistered_at = datetime.now(UTC).isoformat()
        preregistration = _preregistration(selected, preregistered_at)
        _write(
            output,
            {
                "schema": "nova-seed-alignment-power/1.0",
                "status": "preregistered",
                "preregistration": preregistration,
            },
        )
        print(f"preregistered {len(selected)} frames before scoring", flush=True)
        outcomes = _measure_added_outcomes(output, store, preregistration, selected)
        outcomes_banked_at = datetime.now(UTC).isoformat()
        _write(
            output,
            {
                "schema": "nova-seed-alignment-power/1.0",
                "status": "outcomes_banked_before_scoring",
                "preregistration": preregistration,
                "outcomes_banked_utc": outcomes_banked_at,
                "outcomes": [outcomes[key] for key in sorted(outcomes)],
            },
        )
        print("all enlarged-corpus outcomes banked; scoring begins", flush=True)

    scores = _score_cases(store, selected)
    rows = [
        {
            **outcomes[key],
            **scores[key],
            "corpus_role": "banked_five" if key in BANKED_CASES else "added",
        }
        for key in selected
    ]
    growth = _correlation(rows, "cumulative_growth")
    residual = _correlation(rows, "terminal_residual")
    planned_growth_power = _fisher_power(
        BANKED_ASSOCIATIONS["spearman_against_cumulative_growth"]["statistic"],
        len(rows),
    )
    planned_residual_power = _fisher_power(
        BANKED_ASSOCIATIONS["spearman_against_terminal_residual"]["statistic"],
        len(rows),
    )
    old_commit = _read(LOCKED_RECEIPT)["source_identity"]["commit"]
    production_diff = [
        path
        for path in _git(
            "diff", "--name-only", old_commit, "HEAD", "--", "nova"
        ).splitlines()
        if path
    ]
    receipt = {
        "schema": "nova-seed-alignment-power/1.0",
        "status": "banked",
        "completed_utc": datetime.now(UTC).isoformat(),
        "preregistration": preregistration,
        "measurement_sequence": {
            "preregistered_utc": preregistered_at,
            "all_outcomes_banked_utc": outcomes_banked_at,
            "added_scores_computed_only_after_outcome_bank_completed": True,
        },
        "observable_execution": {
            "definition_carried_forward_unchanged": True,
            "definition": LOCKED_DEFINITION,
            "converged_solve_enters_observable": False,
            "statement": (
                "Every score uses only the cold moment seed, one fixed-point map "
                "evaluation, and one tangent action at that seed. The outcome bank "
                "was completed before any added-frame score was computed; the five "
                "locked scores are carried verbatim from their banked receipt."
            ),
            "backend": jax.default_backend(),
            "precision": "float64",
        },
        "outcome_definitions": {
            "cumulative_growth": (
                "maximum eager-compiled flux separation after twelve production "
                "nonlinear updates divided by the separation after one map application"
            ),
            "terminal_residual": (
                "terminal fixed-point residual from the cold "
                "target-current-constrained public host-Krylov solve"
            ),
        },
        "ranking": {
            "rows": rows,
            "banked_five_readings": BANKED_ASSOCIATIONS,
            "spearman_against_cumulative_growth": growth,
            "spearman_against_terminal_residual": residual,
        },
        "statistical_power": {
            "alpha": ALPHA,
            "sample_count": len(rows),
            "method": (
                "two-sided Fisher-z normal approximation; descriptive achieved "
                "power under a continuous bivariate alternative"
            ),
            "at_banked_effect_sizes": {
                "cumulative_growth": {
                    "effect_size": BANKED_ASSOCIATIONS[
                        "spearman_against_cumulative_growth"
                    ]["statistic"],
                    "power": planned_growth_power,
                },
                "terminal_residual": {
                    "effect_size": BANKED_ASSOCIATIONS[
                        "spearman_against_terminal_residual"
                    ]["statistic"],
                    "power": planned_residual_power,
                },
            },
            "at_enlarged_observed_effect_sizes": {
                "cumulative_growth": {
                    "effect_size": growth["statistic"],
                    "power": _fisher_power(growth["statistic"], len(rows)),
                },
                "terminal_residual": {
                    "effect_size": residual["statistic"],
                    "power": _fisher_power(residual["statistic"], len(rows)),
                },
            },
            "adequate_power_threshold": 0.8,
        },
        "scientific_verdict": _scientific_verdict(
            growth, residual, planned_growth_power
        ),
        "comparison_scope": {
            "claim": "rank association only; no causal performance attribution",
            "locked_score_source_commit": old_commit,
            "added_measurement_source_commit": _git("rev-parse", "HEAD"),
            "production_paths_changed_since_locked_score_tree": production_diff,
            "qualification": (
                "The five historical outcomes, the previously banked sixth frame, "
                "and six current-tree additions do not share one fully identified "
                "production tree. Multiple production paths changed, so the pooled "
                "association answers ranking power only and cannot attribute an "
                "outcome delta to the observable or to any solver mechanism."
            ),
        },
        "source_identity": {
            "commit": _git("rev-parse", "HEAD"),
            "tree": _git("rev-parse", "HEAD^{tree}"),
        },
        "authority_inputs": {
            str(path): {"sha256": _digest(path)}
            for path in (
                LOCKED_RECEIPT,
                GROWTH_BANK,
                TERMINAL_BANK,
                SELECTION_BANK,
            )
        },
    }
    _write(output, receipt)
    return receipt


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output", type=Path, default=OUTPUT)
    result.add_argument("--store", type=Path, default=SHOT_STORE)
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    result = measure(arguments.output, arguments.store)
    print(
        json.dumps(
            {
                "status": result["status"],
                "sample_count": result["statistical_power"]["sample_count"],
                "growth": result["ranking"]["spearman_against_cumulative_growth"],
                "terminal_residual": result["ranking"][
                    "spearman_against_terminal_residual"
                ],
                "power": result["statistical_power"]["at_banked_effect_sizes"],
                "verdict": result["scientific_verdict"]["status"],
            },
            indent=2,
        )
    )
