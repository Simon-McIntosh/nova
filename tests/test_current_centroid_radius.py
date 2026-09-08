"""Guard the solved current-centroid major radius against EFIT.

The forward labeller labels each slice with the plasma current centroid its own
moment observation reads off the converged flux.  The height component has
always been guarded, by the branch guard and the conditioning target; the major
radius has never been compared with anything, so a several-centimetre inboard
offset survived a corpus of 923 shots.  This test makes the major-radius
channel a checked one.

Tolerance argument (from the grid, never from the present miss): the solved
centroid is a cell-current-weighted mean of the lattice cell centres.  Placing
each cell's current at its centre rather than at its true in-cell distribution
displaces the moment by at most half a cell, so the discretisation floor on the
centroid major radius is ``dr / 2``, where ``dr`` is the radial step of the
stride-2 lattice the labeller solves on.  A maintained-sign displacement beyond
that floor is a resolved systematic, not discretisation scatter.  The floor is
computed here from the stored efm axis, not hard-coded, so it stays with the grid.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from benchmarks import centroid_radius_validation as validation
from nova.imas.mast_solve_inputs import SHOT_STORE


def _availability() -> tuple[bool, str]:
    """Return (works, reason) for the measurement basis on this host."""
    if not (SHOT_STORE / "22086.zarr").exists():
        return False, f"the MAST {SHOT_STORE} zarr store is not present"
    if not (validation.DEFAULT_CORPUS_ROOT / "22086.manifest.json").is_file():
        return False, f"the solved corpus {validation.DEFAULT_CORPUS_ROOT} is absent"
    return True, ""


_AVAILABLE, _REASON = _availability()


@pytest.mark.skipif(not _AVAILABLE, reason=_REASON)
def test_centroid_major_radius_within_the_grid_discretisation_floor() -> None:
    """The solved centroid major radius agrees with EFIT to dr/2 or better.

    The tolerance is argued from the lattice spacing rather than fitted to the
    observed miss: a guard set to pass today's number would ratify the defect.
    Today the measured bias exceeds the floor on the diverted body rows, so
    this protection is expected to report FAIL until the systematic is
    attributed and repaired; the per-row deltas below name the offenders.
    """
    measurements = {
        shot: validation.load_measurements(validation.DEFAULT_CORPUS_ROOT, shot)
        for shot in validation.CARRIER_SHOTS
    }
    radial_step = validation.grid_radial_step(22086)
    floor_m = validation.discretisation_tolerance_m(radial_step)
    floor_cm = floor_m * 100.0
    assert floor_cm > 0.0
    rows = validation._bank_rows(measurements, floor_m)  # noqa: SLF001
    measured = [row for row in rows if row["measured"]]
    assert len(measured) >= 1, "no bank row carried a solved centroid"
    failures = [row["row"] for row in measured if not row["pass"]]
    assert not failures, (
        "solved centroid major radius leaves the grid discretisation floor "
        f"(dr/2 = {floor_cm:.2f} cm): rows {failures} sit beyond it; "
        "per-row deltas cm = "
        + ", ".join(f"{row['row']}:{row['delta_cm']:+.2f}" for row in measured)
    )


@pytest.mark.skipif(not _AVAILABLE, reason=_REASON)
def test_carrier_sample_reproportion_is_recorded(tmp_path: Path) -> None:
    """The carrier sample's measured distribution is finite and recorded.

    The receipt carries median, mean, standard deviation, per-shot medians and
    the inboard fraction; this test forces the benchmark to recompute them from
    the corpus so a stale or empty receipt cannot silently pass.
    """
    receipt = validation.measure(
        validation.DEFAULT_CORPUS_ROOT,
        tmp_path,
    )
    distribution = receipt["carrier_distribution"]
    assert distribution["count"] > 0
    for key in ("median_cm", "mean_cm", "std_cm", "inboard_fraction"):
        assert np.isfinite(distribution[key]), key
    assert 0.0 <= distribution["inboard_fraction"] <= 1.0
    for shot in validation.CARRIER_SHOTS:
        per_shot = receipt["carrier_per_shot"][str(shot)]
        assert per_shot["count"] > 0
        assert np.isfinite(per_shot["median_cm"])
    bank_measured = [row for row in receipt["bank_rows"] if row["measured"]]
    assert len(bank_measured) >= 1


@pytest.mark.skipif(not _AVAILABLE, reason=_REASON)
def test_bank_rows_without_converged_solves_are_named_not_hidden(
    tmp_path: Path,
) -> None:
    """Bank rows lacking a solved centroid are reported, not silently dropped.

    Rows the labeller did not solve to a finite centroid cannot be asserted
    against EFIT; the guard must surface them as missing evidence rather than
    pretend the twelve rows are all measured.
    """
    measurements = {
        shot: validation.load_measurements(validation.DEFAULT_CORPUS_ROOT, shot)
        for shot in validation.CARRIER_SHOTS
    }
    radial_step = validation.grid_radial_step(22086)
    floor_m = validation.discretisation_tolerance_m(radial_step)
    rows = validation._bank_rows(measurements, floor_m)  # noqa: SLF001
    reported = {row["row"] for row in rows}
    assert reported == set(validation.MAST_BANK_ROWS)
    assert all(row["measured"] or row["reason"] for row in rows)
    missing = [row["row"] for row in rows if not row["measured"]]
    assert [row["row"] for row in rows] == list(validation.MAST_BANK_ROWS)
    assert all(row["reason"] for row in rows if not row["measured"])
    # Every unmeasured bank row must be recorded in the receipt, never dropped:
    receipt = validation.measure(validation.DEFAULT_CORPUS_ROOT, tmp_path)
    receipt_rows = {row["row"]: row for row in receipt["bank_rows"]}
    assert set(receipt_rows) == set(validation.MAST_BANK_ROWS)
    for missing_row in missing:
        assert receipt_rows[missing_row]["measured"] is False
        assert receipt_rows[missing_row]["reason"]


if __name__ == "__main__":
    pytest.main([__file__])
