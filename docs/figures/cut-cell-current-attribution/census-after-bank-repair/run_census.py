"""Run the fixed-state amplitude census with a caller-owned output directory."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from benchmarks import unit_amplitude_current_census as census
from nova.equilibrium import clip_quadrature, forward_operator


output_root = Path(os.environ["CENSUS_OUTPUT_ROOT"])
census.OUTPUT_ROOT = output_root
census.RECEIPT = output_root / "receipt.json"
overflow_signatures = [0]
original_moments = clip_quadrature.clipped_support_current_moments


def checked_moments(*args, **kwargs):
    result = original_moments(*args, **kwargs)
    if not np.all(np.isfinite(np.asarray(result))):
        overflow_signatures[0] += 1
    return result


clip_quadrature.clipped_support_current_moments = checked_moments
forward_operator.clipped_support_current_moments = checked_moments
result = census.measure()
exact_totals = {
    name: state["unit_amplitude_totals_a"]["exact"] for name, state in result["states"].items()
}
chord_totals = {
    name: state["unit_amplitude_totals_a"]["chord"] for name, state in result["states"].items()
}
print(
    "realised_cells=%d exact_overflow_signatures=%d exact_total_finite=%s"
    % (
        result["analytic_cells"]["exact_cut_count"]
        + result["analytic_cells"]["interior_count"]
        + result["analytic_cells"]["exterior_count"],
        overflow_signatures[0],
        all(np.isfinite(value) for value in exact_totals.values()),
    ),
    flush=True,
)
print("exact_totals_a=%r" % exact_totals, flush=True)
print("chord_totals_a=%r" % chord_totals, flush=True)