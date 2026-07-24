"""Unit tests for the fiducial-sector analyses (nova.assembly.fiducialsector).

Pure tests (chart builders, lazy-import guard) run everywhere; the driver
tests build a real sector from the in-repo canonical units via the
characterization fixture machinery and skip visibly when those fixtures cannot
be materialized.
"""

import matplotlib

matplotlib.use("Agg", force=True)

import numpy as np
import pandas as pd
import pytest

import nova.assembly.fiducialsector as fiducialsector
from nova.assembly.fiducialsector import (
    FiducialSector,
    ilis_scatter_chart,
    sector_offset_chart,
)
from tests.characterization import _fixtures

FIXTURES_AVAILABLE = _fixtures.sector_modules_available()

requires_fixtures = pytest.mark.skipif(
    not FIXTURES_AVAILABLE,
    reason="sector-module fixtures cannot be materialized: in-repo canonical "
    "units or the workbook transcoder are unavailable",
)


def test_altair_import_is_lazy():
    """Importing the module must not bind altair at module scope.

    altair is optional; the chart builders import it lazily so the module (and
    everything that constructs a sector) imports without the charting stack.
    """
    assert not hasattr(fiducialsector, "alt")


class TestSectorOffsetChart:
    """sector_offset_chart returns a constructed (unshown) altair chart."""

    def _plot_data(self):
        rng = np.random.default_rng(3)
        rows = []
        for feature in ["ILIS +1", "Gap", "ILIS -1"]:
            for _ in range(5):
                rows.append(
                    {
                        "Name": "P",
                        "r": rng.uniform(2000, 3200),
                        "z": rng.uniform(-4000, 4000),
                        "offset": rng.normal(),
                        "coil": 8,
                        "feature": feature,
                    }
                )
        return pd.DataFrame(rows)

    def test_returns_chart(self):
        chart = sector_offset_chart(
            self._plot_data(), ["ILIS +1", "Gap", "ILIS -1"], "Sector 7"
        )
        assert type(chart).__module__.startswith("altair")
        # a valid spec serializes without error
        assert isinstance(chart.to_dict(), dict)


class TestIlisScatterChart:
    """ilis_scatter_chart returns a constructed (unshown) altair chart."""

    def _data(self):
        rng = np.random.default_rng(5)
        rows = []
        for feature in ["ILIS +1", "ILIS -1"]:
            for _ in range(6):
                rows.append(
                    {
                        "Name": "P",
                        "r": rng.uniform(2000, 3200),
                        "y": rng.uniform(-3000, 3000),
                        "z": rng.uniform(-4000, 4000),
                        "ro_phi": rng.normal(scale=10),
                        "offset": rng.normal(),
                        "outlier": False,
                        "coil": 8,
                        "feature": feature,
                        "type": np.nan,
                    }
                )
        for name in ["A'", "B'", "H'"]:
            rows.append(
                {
                    "Name": name,
                    "r": rng.uniform(2000, 3200),
                    "y": rng.uniform(-3000, 3000),
                    "z": rng.uniform(-4000, 4000),
                    "ro_phi": rng.normal(scale=10),
                    "offset": np.nan,
                    "outlier": np.nan,
                    "coil": 8,
                    "feature": "CCL",
                    "type": "projected",
                }
            )
        return pd.DataFrame(rows)

    def test_returns_chart(self):
        chart = ilis_scatter_chart(self._data(), "Sector 7")
        assert type(chart).__module__.startswith("altair")
        assert isinstance(chart.to_dict(), dict)

    def test_handles_all_outlier_mask(self):
        """A degenerate (no clean points) frame still builds a chart."""
        data = self._data()
        data.loc[data.feature.str.startswith("ILIS"), "outlier"] = True
        chart = ilis_scatter_chart(data, "Sector 7")
        assert isinstance(chart.to_dict(), dict)


@pytest.fixture(scope="module")
def sector():
    """Build the sector-7 coil pair from the rebuilt SSAT BR workbook.

    Built with ``augment=False`` so the lifted analyses are exercised on the
    measured surfaces without the reference-version search.
    """
    if not FIXTURES_AVAILABLE:
        pytest.skip(
            "sector-module fixtures cannot be materialized: in-repo canonical "
            "units or the workbook transcoder are unavailable"
        )
    _fixtures.ensure_sector_cache()
    return FiducialSector(
        phase="SSAT BR", sectors={7: [8, 9]}, private=False, augment=False
    )


@requires_fixtures
class TestCclCylindrical:
    def test_columns_and_filtering(self, sector):
        ccl = sector.ccl_cylindrical()
        assert set(ccl.columns) == {
            "Name",
            "x",
            "y",
            "z",
            "r",
            "phi",
            "ro_phi",
            "coil",
            "feature",
        }
        # only the inboard CCL fiducials, all tagged CCL
        assert set(ccl.Name) <= {"A", "B", "H"}
        assert set(ccl.feature) == {"CCL"}
        # restricted to coils that carry ILIS surfaces
        assert set(ccl.coil) <= set(sector.ilis.coil.unique())

    def test_cylindrical_columns_consistent(self, sector):
        ccl = sector.ccl_cylindrical()
        assert np.allclose(ccl.r, np.linalg.norm(ccl[["x", "y"]].values, axis=1))
        assert np.allclose(ccl.ro_phi, fiducialsector.ILIS_RADIUS * ccl.phi)

    def test_custom_targets(self, sector):
        ccl = sector.ccl_cylindrical(targets=("A",))
        assert set(ccl.Name) <= {"A"}


@requires_fixtures
class TestSectorOffsetGrid:
    def test_two_coil_gap(self, sector):
        plot_data, facet_sort = sector.sector_offset_grid()
        assert facet_sort == ["ILIS +1", "Gap", "ILIS -1"]
        assert set(plot_data.feature) == {"ILIS +1", "Gap", "ILIS -1"}
        assert set(plot_data.columns) == {
            "Name",
            "r",
            "z",
            "offset",
            "coil",
            "feature",
        }
        # the gap plane samples the requested (r, z) grid resolution
        gap = plot_data[plot_data.feature == "Gap"]
        assert len(gap) == 20 * 40

    def test_grid_resolution_kwargs(self, sector):
        plot_data, _ = sector.sector_offset_grid(n_r=8, n_z=12)
        gap = plot_data[plot_data.feature == "Gap"]
        assert len(gap) == 8 * 12

    def test_single_coil_no_gap(self, sector):
        """A single-coil sector measures both surfaces without forming a gap."""
        single = FiducialSector(
            phase="SSAT BR", sectors={7: [8]}, private=False, augment=False
        )
        plot_data, facet_sort = single.sector_offset_grid()
        assert facet_sort == ["ILIS +1", "ILIS -1"]
        assert "Gap" not in set(plot_data.feature)


@requires_fixtures
class TestProjectCclToMidplane:
    def test_original_and_projected(self, sector):
        data = sector.project_ccl_to_midplane()
        types = set(data["type"].dropna().unique())
        assert {"original", "projected"} <= types
        # projected inboard fiducials are renamed with a trailing prime
        projected = data[data["type"] == "projected"]
        assert all(name.endswith("'") for name in projected.Name)
        assert set(projected.Name) <= {"A'", "B'", "H'"}

    def test_projection_recomputes_radius(self, sector):
        data = sector.project_ccl_to_midplane()
        projected = data[data["type"] == "projected"]
        assert np.allclose(
            projected.r, np.linalg.norm(projected[["x", "y"]].values, axis=1)
        )


@requires_fixtures
class TestChartsFromRealData:
    """The chart builders accept the frames the sector methods emit."""

    def test_offset_chart(self, sector):
        plot_data, facet_sort = sector.sector_offset_grid()
        chart = sector_offset_chart(plot_data, facet_sort, "Sector 7")
        assert isinstance(chart.to_dict(), dict)

    def test_scatter_chart(self, sector):
        data = sector.project_ccl_to_midplane()
        chart = ilis_scatter_chart(data, "Sector 7")
        assert isinstance(chart.to_dict(), dict)
