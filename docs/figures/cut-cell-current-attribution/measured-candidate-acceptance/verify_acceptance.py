"""Replay measured candidate scores through the production admission rules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nova.jax.config import configure_dtypes

configure_dtypes()

import jax.numpy as jnp  # noqa: E402

from nova.equilibrium.fixed_point import _select_backtracking_candidate  # noqa: E402


def _production_selection(
    factors,
    merits,
    residuals,
    incumbent_merit,
    incumbent_residual,
    acceptance_reference,
    predicted_merits,
    predicted_current_merit,
):
    return _select_backtracking_candidate(
        jnp.asarray(factors),
        jnp.asarray(merits),
        jnp.asarray(residuals),
        jnp.asarray(incumbent_merit),
        jnp.asarray(incumbent_residual),
        jnp.asarray(acceptance_reference),
        jnp.asarray(predicted_merits),
        jnp.asarray(predicted_current_merit),
        True,
        True,
    )


def _historical_checks(coarse):
    scores = coarse["historical_reproduction"]["scores"]
    analytic = scores["toward_analytic"]
    analytic_selection = _production_selection(
        analytic["factors"],
        analytic["merits"],
        analytic["residuals"],
        analytic["incumbent_merit"],
        analytic["incumbent_residual"],
        analytic["acceptance_reference"],
        analytic["predicted_merits"],
        analytic["predicted_current_merit"],
    )
    assert int(analytic_selection.selected) == 0
    assert bool(analytic_selection.accepted)
    assert bool(analytic_selection.selected_model_unreliable)

    half = scores["map_defect"]
    half_selection = _production_selection(
        half["factors"],
        half["merits"],
        half["residuals"],
        half["incumbent_merit"],
        half["incumbent_residual"],
        half["acceptance_reference"],
        half["predicted_merits"],
        half["predicted_current_merit"],
    )
    selected = int(half_selection.selected)
    assert half["factors"][selected] == 0.5
    assert bool(half_selection.accepted)
    assert not bool(half_selection.selected_model_unreliable)
    return {
        "full_analytic": {
            "actual_merit": analytic["merits"][0],
            "predicted_merit": analytic["predicted_merits"][0],
            "accepted": True,
            "model_unreliable": True,
            "selected_index": int(analytic_selection.selected),
        },
        "half_defect": {
            "actual_merit": half["merits"][selected],
            "predicted_merit": half["predicted_merits"][selected],
            "accepted": True,
            "model_unreliable": False,
            "selected_index": selected,
        },
    }


def _authoritative_checks(receipt):
    incumbent = receipt["incumbent"]
    analytic_ladder = receipt["directions"]["analytic"]["ladder"]
    analytic_selection = _production_selection(
        [item["fraction"] for item in analytic_ladder],
        [item["actual"]["merit"] for item in analytic_ladder],
        [item["actual"]["relative_sup"] for item in analytic_ladder],
        incumbent["merit"],
        incumbent["relative_sup"],
        incumbent["merit"],
        [item["predicted"]["merit"] for item in analytic_ladder],
        incumbent["merit"],
    )
    analytic = analytic_ladder[int(analytic_selection.selected)]
    assert int(analytic_selection.selected) == 0
    assert bool(analytic_selection.accepted)
    assert bool(analytic_selection.selected_model_unreliable)

    newton = receipt["directions"]["newton"]["ladder"]
    newton_verdicts = []
    for item in newton:
        selection = _production_selection(
            [item["fraction"]],
            [item["actual"]["merit"]],
            [item["actual"]["relative_sup"]],
            incumbent["merit"],
            incumbent["relative_sup"],
            incumbent["merit"],
            [item["predicted"]["merit"]],
            incumbent["merit"],
        )
        assert not bool(selection.accepted)
        newton_verdicts.append({"fraction": item["fraction"], "accepted": False})
    return {
        "cells": receipt["realised_cells"],
        "analytic": {
            "actual_merit": analytic["actual"]["merit"],
            "predicted_merit": analytic["predicted"]["merit"],
            "residual": analytic["actual"]["relative_sup"],
            "accepted": True,
            "model_unreliable": True,
            "selected_index": int(analytic_selection.selected),
        },
        "newton_candidates_refused": len(newton),
        "newton_production_selector_verdicts": newton_verdicts,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    coarse = json.loads((args.receipts / "cells-135.json").read_text())
    fine = json.loads((args.receipts / "cells-342.json").read_text())
    result = {
        "historical": _historical_checks(coarse),
        "authoritative": [
            _authoritative_checks(coarse),
            _authoritative_checks(fine),
        ],
        "production_direction_note": (
            "Only measured analytic witnesses use pessimistic-model admission; "
            "all sampled Newton candidates worsen actual merit and remain refused."
        ),
        "selector_kernel": (
            "nova.equilibrium.fixed_point._select_backtracking_candidate"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
