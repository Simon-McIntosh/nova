"""Guards for the numbers this record publishes.

The fragment's arm table is the record's claim, and every cell of it must be the
value the arm's own ``summary.json`` holds; the lag-one convention the table
uses has to be reproducible from the text, and a constant divisor must not move
a correlation.  The arms live under the operator's reckon runs directory, so the
table check skips where those runs are absent rather than inventing a value.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

import lag_table

RUNS = Path(os.environ.get("DRIFT_RUNS", str(Path.home() / ".config/reckon/crew/runs")))
FRAGMENT = (
    Path(__file__).resolve().parents[4]
    / "docs/evidence/fragments/inverse-equilibrium-baseline"
    / "drifting-sensor-probe-rows.html"
)


def test_fragment_arm_table_matches_the_arms() -> None:
    """Every row of the arm table carries its own summary's lag-one triplet."""

    if not (RUNS / lag_table.CURRENT_RUN / "study/summary.json").exists():
        pytest.skip("the arms' summaries are not on this host")
    assert FRAGMENT.exists(), FRAGMENT
    assert lag_table.failures(FRAGMENT, RUNS) == []


def test_a_constant_divisor_leaves_a_correlation_unchanged() -> None:
    """The table's fixed-span divisor cannot move a lag-one figure."""

    rng = np.random.default_rng(20261119)
    series = np.cumsum(rng.normal(size=64))
    reference = float(np.corrcoef(series[:-1], series[1:])[0, 1])
    scaled = float(np.corrcoef((series / 76.70379)[:-1], (series / 76.70379)[1:])[0, 1])
    assert scaled == pytest.approx(reference, abs=1e-12)
