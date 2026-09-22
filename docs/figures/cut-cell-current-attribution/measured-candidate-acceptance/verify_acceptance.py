"""Replay measured candidate scores through the production admission rules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nova.jax.config import configure_dtypes

configure_dtypes()

from nova.equilibrium.fixed_point import (  # noqa: E402
    _measured_decrease_accepts_pessimistic_model,
    _model_decrease_is_trusted,
)


def _actual_checks(incumbent, candidate, fraction):
    required = incumbent["merit"] * (1.0 - 1.0e-4 * fraction)
    return (
        candidate["actual"]["merit"] <= required
        and candidate["actual"]["relative_sup"] < incumbent["relative_sup"]
    )


def _pessimistic_admission(incumbent, candidate):
    return bool(
        _measured_decrease_accepts_pessimistic_model(
            candidate["predicted"]["merit"],
            candidate["actual"]["merit"],
            incumbent["merit"],
            incumbent["merit"],
            candidate["actual"]["relative_sup"],
            incumbent["relative_sup"],
            candidate["fraction"],
        )
    )


def _historical_checks(coarse):
    scores = coarse["historical_reproduction"]["scores"]
    analytic = scores["toward_analytic"]
    selected = analytic["ladder_selected"]
    full_candidate = {
        "fraction": analytic["factors"][selected],
        "actual": {
            "merit": analytic["merits"][selected],
            "relative_sup": analytic["residuals"][selected],
        },
        "predicted": {"merit": analytic["predicted_merits"][selected]},
    }
    incumbent = {
        "merit": analytic["incumbent_merit"],
        "relative_sup": analytic["incumbent_residual"],
    }
    assert _actual_checks(incumbent, full_candidate, full_candidate["fraction"])
    assert _pessimistic_admission(incumbent, full_candidate)

    half = scores["map_defect"]
    selected = half["ladder_selected"]
    assert half["factors"][selected] == 0.5
    assert half["merits"][selected] <= half["incumbent_merit"] * (
        1.0 - 1.0e-4 * half["factors"][selected]
    )
    assert half["residuals"][selected] < half["incumbent_residual"]
    assert bool(
        _model_decrease_is_trusted(
            half["predicted_merits"][selected],
            half["merits"][selected],
            half["incumbent_merit"],
            half["predicted_current_merit"],
        )
    )
    return {
        "full_analytic": {
            "actual_merit": full_candidate["actual"]["merit"],
            "predicted_merit": full_candidate["predicted"]["merit"],
            "accepted": True,
            "model_unreliable": True,
        },
        "half_defect": {
            "actual_merit": half["merits"][selected],
            "predicted_merit": half["predicted_merits"][selected],
            "accepted": True,
            "model_unreliable": False,
        },
    }


def _authoritative_checks(receipt):
    incumbent = receipt["incumbent"]
    analytic = receipt["directions"]["analytic"]["ladder"][0]
    assert _actual_checks(incumbent, analytic, analytic["fraction"])
    assert _pessimistic_admission(incumbent, analytic)

    newton = receipt["directions"]["newton"]["ladder"]
    assert all(not _actual_checks(incumbent, item, item["fraction"]) for item in newton)
    assert all(not _pessimistic_admission(incumbent, item) for item in newton)
    return {
        "cells": receipt["realised_cells"],
        "analytic": {
            "actual_merit": analytic["actual"]["merit"],
            "predicted_merit": analytic["predicted"]["merit"],
            "residual": analytic["actual"]["relative_sup"],
            "accepted": True,
            "model_unreliable": True,
        },
        "newton_candidates_refused": len(newton),
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
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
