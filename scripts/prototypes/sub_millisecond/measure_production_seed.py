"""Probe production cold discs and independence of explicit warm seeds."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import jax
import numpy as np
import nova

from nova.equilibrium.forward import ReconstructMoment
from nova.equilibrium.solve_request import ExplicitSolveSeed
from tests.test_two_class_recovery import _limited_problem


def run(output):
    assert jax.config.jax_enable_x64 is True
    print(
        f"nova={nova.__file__} module={__file__} cwd={Path.cwd().resolve()}", flush=True
    )
    profile, original, _receipt, _root = _limited_problem("coarse")
    centroid = np.asarray(original.branches.centroid[0])
    target = float(original.branches.plasma_current[0])
    warm = np.asarray(original.branches.flux[0]).copy()
    recorded = []
    uniform_disc = ReconstructMoment.uniform_disc

    def capture(self, *args, **kwargs):
        value = uniform_disc(self, *args, **kwargs)
        recorded.append(np.asarray(value).copy())
        return value

    with patch.object(ReconstructMoment, "uniform_disc", capture):
        positive = profile.cold_seed_portfolio(target, centroid)
        zero = profile.cold_seed_portfolio(0.0, centroid)

    positive_current, zero_current = recorded
    explicit = ExplicitSolveSeed(warm)
    resolved = np.asarray(explicit.resolve(profile))
    assert np.count_nonzero(positive_current) > 0
    assert np.sum(zero_current) == 0
    assert np.array_equal(resolved, warm)
    assert np.count_nonzero(np.asarray(positive.branches.flux[0])) > 0
    result = {
        "target_current": target,
        "positive_seed_current_sum": float(np.sum(positive_current)),
        "positive_seed_nonzero_cells": int(np.count_nonzero(positive_current)),
        "zero_seed_current_sum": float(np.sum(zero_current)),
        "zero_seed_nonzero_cells": int(np.count_nonzero(zero_current)),
        "explicit_warm_seed_bitwise_preserved_after_zero_cold_seed": bool(
            np.array_equal(resolved, warm)
        ),
        "positive_cold_supported_cells": int(positive.branches.supported_cells[0]),
        "zero_cold_supported_cells": int(zero.branches.supported_cells[0]),
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print("RESULT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.output)
