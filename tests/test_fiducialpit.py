"""Unit tests for the pit-integration core (nova.assembly.fiducialpit).

Pure-function tests run everywhere; the pit-driver tests rebuild sector
workbooks from the in-repo canonical units via the characterization fixture
machinery and skip visibly when those fixtures cannot be materialized.
"""

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas
import pytest

from nova.assembly.fiducialpit import (
    DEFAULT_PIT_SECTORS,
    FiducialPit,
    PitAnalysis,
    analyze_pit,
    measured_gap_profiles,
    rotate_to_angle,
)
from tests.characterization import _fixtures

FIXTURES_AVAILABLE = _fixtures.sector_modules_available()

requires_fixtures = pytest.mark.skipif(
    not FIXTURES_AVAILABLE,
    reason="sector-module fixtures cannot be materialized: in-repo canonical "
    "units or the workbook transcoder are unavailable",
)


class TestRotateToAngle:
    """rotate_to_angle spins a vector about the z-axis by a radian angle."""

    def test_identity(self):
        vector = np.array([1.0, 2.0, 3.0])
        assert np.allclose(rotate_to_angle(vector, 0.0), vector)

    def test_quarter_turn(self):
        rotated = rotate_to_angle(np.array([1.0, 0.0, 0.0]), np.pi / 2)
        assert np.allclose(rotated, [0.0, 1.0, 0.0])

    def test_z_unchanged(self):
        rotated = rotate_to_angle(np.array([3.0, -1.0, 5.0]), 1.3)
        assert rotated[2] == pytest.approx(5.0)


class TestMeasuredGapProfiles:
    """The recorded feeler-gauge tables are well-formed, testable data."""

    def test_expected_pairs(self):
        profiles = measured_gap_profiles()
        assert set(profiles) == {(8, 9), (12, 13), (13, 8)}

    def test_columns_and_lengths(self):
        for frame in measured_gap_profiles().values():
            assert list(frame.columns) == ["z", "gap"]
            assert len(frame["z"]) == len(frame["gap"])
            assert len(frame) > 0

    def test_gaps_are_physical(self):
        for frame in measured_gap_profiles().values():
            assert (frame["gap"] > 0).all()
            assert (frame["gap"] < 5).all()

    def test_fresh_frames_per_call(self):
        """Each call returns independent frames so callers may mutate freely."""
        first = measured_gap_profiles()[(8, 9)]
        first.loc[0, "gap"] = 999.0
        assert measured_gap_profiles()[(8, 9)].loc[0, "gap"] != 999.0


class TestDefaultSectors:
    def test_contiguous_cluster(self):
        assert DEFAULT_PIT_SECTORS == {6: [12, 13], 7: [8, 9], 5: [16, 5], 8: [4, 11]}


@pytest.fixture(scope="module")
def analysis() -> PitAnalysis:
    """Run analyze_pit (no plotting) over an adjacent two-sector subset."""
    if not FIXTURES_AVAILABLE:
        pytest.skip(
            "sector-module fixtures cannot be materialized: in-repo canonical "
            "units or the workbook transcoder are unavailable"
        )
    _fixtures.ensure_sector_cache()
    return analyze_pit(
        sectors={6: [12, 13], 7: [8, 9]},
        phase="latest",
        pcr=True,
        private=False,
        plot=False,
    )


@requires_fixtures
class TestAnalyzePit:
    def test_returns_pit(self, analysis):
        assert isinstance(analysis.pit, FiducialPit)

    def test_no_figures_when_plot_disabled(self, analysis):
        assert analysis.figures == {}

    def test_gaps_frame(self, analysis):
        gaps = analysis.pit.gaps
        assert isinstance(gaps, pandas.DataFrame)
        assert not gaps.empty
        assert set(gaps["gap_type"]) <= {"intra-sector", "inter-sector"}

    def test_position_statistics_shape(self, analysis):
        stats = analysis.position_statistics
        assert isinstance(stats, pandas.DataFrame)
        assert {"parameter", "n", "mean", "std"} <= set(stats.columns)

    def test_position_summary_columns(self, analysis):
        summary = analysis.position_summary
        assert isinstance(summary, pandas.DataFrame)
        assert {"parameter", "mean", "tolerance", "margin"} <= set(summary.columns)

    def test_summary_reducer(self, analysis):
        summary = analysis.pit.summary()
        assert isinstance(summary, pandas.DataFrame)
        assert set(summary["type"]) <= {"intra-sector", "inter-sector", "combined"}


@requires_fixtures
class TestAnalyzePitPlots:
    """A plotting run renders and returns the full chart set."""

    def test_figures_rendered(self):
        _fixtures.ensure_sector_cache()
        result = analyze_pit(
            sectors={6: [12, 13], 7: [8, 9]},
            phase="latest",
            plot=True,
            predict=False,
        )
        try:
            assert "position" in result.figures
            assert "gaps" in result.figures
            assert "statistics" in result.figures
            # One gap-profile figure per measured coil pair.
            profile_keys = {k for k in result.figures if k.startswith("gap_profile_")}
            assert profile_keys == {
                f"gap_profile_{a}_{b}" for a, b in measured_gap_profiles()
            }
        finally:
            plt.close("all")
