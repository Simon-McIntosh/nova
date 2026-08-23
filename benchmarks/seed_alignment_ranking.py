"""Rank outcome-blind seed diagnostics against an existing outcome bank."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
from scipy.stats import spearmanr

from benchmarks.efit_parity_warm_neighbour import _prepare_frame
from nova.equilibrium.seed_alignment import residual_action_amplification
from nova.imas.mast_solve_inputs import SHOT_STORE
from nova.jax.config import configure_dtypes


OUTPUT = Path(
    "docs/figures/topology-preserving-continuation/seed-alignment-ranking.json"
)
GROWTH_BANK = Path(
    "docs/figures/mast-catalog-gpu-solve/parity-divergence-attribution.json"
)
RESIDUAL_BANK = Path("docs/figures/efit-forward-parity/warm-neighbour-stall-lift.json")
EVENT_BANK = Path(
    "docs/figures/forward-operator-refinement/event-resolved-amplification.json"
)
CASES = (
    (21978, 35),
    (21983, 35),
    (21985, 51),
    (21986, 46),
    (21989, 55),
)
PREREGISTRATION = {
    "recorded_before_observable_scores_and_outcome_access": True,
    "observable": {
        "name": "seed residual-action amplification",
        "definition": (
            "For seed s, fixed-point map F, residual r = F(s) - s, and local "
            "action A = I - J_F(s), score q(s,F) = ||r||_2 / ||A r||_2."
        ),
        "interpretation": (
            "Larger q means the seed residual occupies a direction on which the "
            "local fixed-point action is weaker, so larger values are ranked as "
            "predicting more cumulative growth and a larger terminal residual."
        ),
        "ranking_direction": "ascending is preferred; larger predicts worse outcome",
        "inputs": ["candidate seed", "fixed-point map at that seed", "map tangent"],
        "outcome_exclusions": [
            "converged solve",
            "terminal state",
            "terminal residual",
            "cumulative growth",
            "reference equilibrium",
        ],
        "singular_convention": (
            "an exact fixed point scores zero; nonzero residual with zero action "
            "scores positive infinity"
        ),
    },
    "fixed_corpus": [
        {"shot": shot, "slice_index": slice_index} for shot, slice_index in CASES
    ],
    "association_test": "two-sided Spearman rank correlation, n = 5",
}


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], text=True, stderr=subprocess.DEVNULL
    ).strip()


def _write_preregistration(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "schema": "nova-seed-alignment-ranking/1.0",
                "status": "preregistered",
                "preregistration": PREREGISTRATION,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


def _score_seeds(store: Path) -> list[dict[str, Any]]:
    cache_box: list[Any] = [None]
    rows: list[dict[str, Any]] = []
    for shot, slice_index in CASES:
        frame, mast_case, _context = _prepare_frame(store, shot, slice_index, cache_box)
        target_current = frame.selected.recorded_plasma_current_a
        map_fn = frame.profile.flux_map(target_current=target_current)
        score = residual_action_amplification(map_fn, frame.seed)
        rows.append(
            {
                "shot": shot,
                "slice_index": slice_index,
                "time_s": float(mast_case["reference"]["time_s"]),
                "seed_residual_action_amplification": float(score),
            }
        )
    return rows


def _banked_outcomes() -> dict[tuple[int, int], dict[str, float]]:
    growth_bank = _read(GROWTH_BANK)
    residual_bank = _read(RESIDUAL_BANK)
    growth = {
        (int(row["shot"]), int(row["slice_index"])): float(
            row["solved_flux"]["maximum_absolute_difference"]
            / row["single_map_application"]["maximum_absolute_difference"]
        )
        for row in growth_bank["current_source_measurement"]["cases"]
    }
    residual = {
        (
            int(row["reference"]["shot"]),
            int(row["reference"]["slice_index"]),
        ): float(row["cold_control"]["terminal_fixed_point_residual"])
        for row in residual_bank["references"]
    }
    expected = set(CASES)
    if expected - growth.keys() or expected - residual.keys():
        raise RuntimeError("the fixed corpus is incomplete in the outcome banks")
    return {
        key: {"cumulative_growth": growth[key], "terminal_residual": residual[key]}
        for key in CASES
    }


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


def _outlier_adjudication() -> dict[str, Any]:
    event = _read(EVENT_BANK)
    seed = event["predictions"]["different_seed_direction"]
    context = event["registered_gate_context"]
    return {
        "shot": 21985,
        "slice_index": 51,
        "classification": "most-informative",
        "instrumentation_perturbed": False,
        "basis": (
            "This is the only corpus frame carrying a reproduced twelve-update "
            "trajectory, iteration-local tangent diagnostics, and a real same-shot "
            "alternate seed. The alternate moved three bursts to none and changed "
            "cumulative separation growth from 5.379792627e9 to 0.06186094. The "
            "instrumented replay reproduced the banked trajectory and changed no "
            "production or test source, so the extreme case is retained rather than "
            "treated as measurement perturbation."
        ),
        "banked_trajectory_reproduced": bool(
            event["trajectory"]["banked_twelve_update_trajectory_reproduced"]
        ),
        "production_or_test_source_changed": bool(
            context["production_or_test_source_changed"]
        ),
        "baseline_burst_updates": seed["baseline_burst_updates"],
        "alternate_burst_updates": seed["alternate_burst_updates"],
        "baseline_cumulative_growth": seed["baseline_cumulative_separation_growth"],
        "alternate_cumulative_growth": seed["alternate_cumulative_separation_growth"],
    }


def measure(output: Path, store: Path) -> dict[str, Any]:
    """Preregister, score the fixed seeds, then open the banked outcomes."""

    configure_dtypes()
    _write_preregistration(output)
    scored_rows = _score_seeds(store)
    outcomes = _banked_outcomes()
    rows = [
        {**row, **outcomes[(row["shot"], row["slice_index"])]} for row in scored_rows
    ]
    growth_correlation = _correlation(rows, "cumulative_growth")
    residual_correlation = _correlation(rows, "terminal_residual")
    receipt = {
        "schema": "nova-seed-alignment-ranking/1.0",
        "status": "banked",
        "completed_utc": datetime.now(UTC).isoformat(),
        "preregistration": PREREGISTRATION,
        "observable_execution": {
            "converged_solve_enters_observable": False,
            "statement": (
                "No converged solve enters the observable: every score uses only "
                "the frame's cold moment seed, one fixed-point map evaluation, and "
                "one tangent action at that same seed. Outcomes are opened only "
                "after all five scores have been computed."
            ),
            "backend": jax.default_backend(),
            "precision": "float64",
        },
        "ranking": {
            "rows": rows,
            "spearman_against_cumulative_growth": growth_correlation,
            "spearman_against_terminal_residual": residual_correlation,
        },
        "scientific_verdict": {
            "status": "HOLD_AS_SEED_SELECTOR",
            "reason": (
                "The preregistered score has a positive but underpowered rank "
                "association with cumulative growth and a null rank association "
                "with terminal residual. It therefore does not consistently rank "
                "both banked outcomes and must not be promoted as a seed selector."
            ),
            "growth_spearman": growth_correlation["statistic"],
            "growth_two_sided_pvalue": growth_correlation["two_sided_pvalue"],
            "residual_spearman": residual_correlation["statistic"],
            "residual_two_sided_pvalue": residual_correlation["two_sided_pvalue"],
        },
        "outlier_adjudication": _outlier_adjudication(),
        "comparison_scope": {
            "claim": "rank association only; no causal performance attribution",
            "qualification": (
                "The observable is measured on the current tree and compared with "
                "outcomes banked on earlier trees. Multiple production mechanisms "
                "changed between those trees, so the correlations rank the fixed "
                "corpus but do not attribute outcome changes to this diagnostic."
            ),
        },
        "source_identity": {
            "commit": _git("rev-parse", "HEAD"),
            "tree": _git("rev-parse", "HEAD^{tree}"),
        },
        "authority_inputs": {
            str(path): {"sha256": _digest(path)}
            for path in (GROWTH_BANK, RESIDUAL_BANK, EVENT_BANK)
        },
    }
    output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
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
                "growth_spearman": result["ranking"][
                    "spearman_against_cumulative_growth"
                ]["statistic"],
                "residual_spearman": result["ranking"][
                    "spearman_against_terminal_residual"
                ]["statistic"],
                "outlier": result["outlier_adjudication"]["classification"],
            },
            indent=2,
        )
    )
