"""Run the fixed-state amplitude census with a caller-owned output directory.

The bank overflow count is read off the census's own per-cell currents, not by
wrapping the moment integrator: that integrator runs inside a JAX scan, so a
wrapper cannot inspect its output without a tracer conversion error.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from benchmarks import unit_amplitude_current_census as census


output_root = str(Path(os.environ["CENSUS_OUTPUT_ROOT"]))
census.OUTPUT_ROOT = Path(output_root)
census.RECEIPT = Path(output_root) / "receipt.json"
result = census.measure()


def nonfinite_count(values) -> int:
    array = np.asarray(
        [np.nan if value is None else float(value) for value in values],
        dtype=np.float64,
    )
    return int(np.count_nonzero(~np.isfinite(array)))


overflow = {
    state_name: {
        mode: nonfinite_count(
            [row[mode + "_current_a"] for row in rows]
        )
        for mode in census.CLIP_MODES
    }
    for state_name, rows in result["per_cell"].items()
}
total_overflow = sum(
    count for per_mode in overflow.values() for count in per_mode.values()
)
exact_totals = {
    name: state["unit_amplitude_totals_a"]["exact"]
    for name, state in result["states"].items()
}
chord_totals = {
    name: state["unit_amplitude_totals_a"]["chord"]
    for name, state in result["states"].items()
}
analytic_totals = {
    name: float(np.sum([row["analytic_exact_current_a"] for row in rows]))
    for name, rows in result["per_cell"].items()
}
print(
    "realised_cells=%d nonfinite_cell_current_signatures=%d exact_total_finite=%s"
    % (
        result["analytic_cells"]["exact_cut_count"]
        + result["analytic_cells"]["interior_count"]
        + result["analytic_cells"]["exterior_count"],
        total_overflow,
        all(
            value is not None and np.isfinite(value) for value in exact_totals.values()
        ),
    ),
    flush=True,
)
print("overflow_by_state_mode=%r" % overflow, flush=True)
print("analytic_totals_a=%r" % analytic_totals, flush=True)
print("chord_totals_a=%r" % chord_totals, flush=True)
print("exact_totals_a=%r" % exact_totals, flush=True)
print("analytic_target_current_a=%r" % result["target_current_a"], flush=True)
print("amplitude_gate=%r" % result["verification"]["amplitude_gate"], flush=True)
print(
    "seed_digest_matches_commit=%r"
    % result["seed"]["digest_matches_commit"],
    flush=True,
)