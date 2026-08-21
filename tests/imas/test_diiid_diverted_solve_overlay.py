"""Tests for the prescribed-input diverted-solve publication figure."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from benchmarks.diiid_poloidal_figures import CoilGeometry, StaticGeometry


MODULE_PATH = (
    Path(__file__).parents[2] / "benchmarks" / "diiid_diverted_solve_overlay.py"
)
SPEC = importlib.util.spec_from_file_location(
    "diiid_diverted_solve_overlay", MODULE_PATH
)
overlay = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = overlay
SPEC.loader.exec_module(overlay)


def _coil(name: str, radius: float) -> CoilGeometry:
    return CoilGeometry(
        name=name,
        elements=(
            np.asarray(
                [
                    [radius - 0.02, -0.04],
                    [radius + 0.02, -0.04],
                    [radius + 0.02, 0.04],
                    [radius - 0.02, 0.04],
                ]
            ),
        ),
    )


def test_contour_levels_come_only_from_the_named_given_map() -> None:
    given = np.arange(65 * 65, dtype=float).reshape(65, 65) / 1000.0
    levels = overlay.contour_levels_from_given(given, count=9)

    assert len(levels) == 9
    assert np.all(np.diff(levels) > 0.0)
    assert levels[0] == pytest.approx(np.quantile(given, 0.01))
    assert levels[-1] == pytest.approx(np.quantile(given, 0.99))


def test_flux_pair_uses_unfilled_contours_and_the_identical_level_object() -> None:
    class Axis:
        def __init__(self) -> None:
            self.calls = []

        def contour(self, *args, **kwargs):
            self.calls.append((args, kwargs))

    axis = Axis()
    radius = np.linspace(1.0, 2.0, 4)
    height = np.linspace(-1.0, 1.0, 5)
    given = np.add.outer(height, radius)
    solved = given.T + 0.1
    levels = np.linspace(0.2, 2.8, 7)

    overlay._draw_flux_pair(axis, radius, height, given, radius, height, solved, levels)

    assert len(axis.calls) == 2
    assert axis.calls[0][1]["levels"] is levels
    assert axis.calls[1][1]["levels"] is levels
    assert axis.calls[0][1]["colors"] != axis.calls[1][1]["colors"]
    assert "cmap" not in axis.calls[0][1]
    assert "cmap" not in axis.calls[1][1]


def test_gauge_match_removes_only_a_spatially_constant_offset() -> None:
    given = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    solved = given - 0.75
    interior = np.ones_like(given, dtype=bool)

    result = overlay.gauge_match(given, solved, interior)

    np.testing.assert_allclose(result["aligned_total_wb"], given)
    assert result["fractional_rms"] == pytest.approx(0.0)
    expected_correction = float(overlay.nova_total_flux_to_corpus(0.75))
    assert result["additive_correction_wb_per_radian"] == pytest.approx(
        expected_correction
    )
    assert result["removed_solve_minus_label_offset_wb_per_radian"] == (
        pytest.approx(-expected_correction)
    )


def test_conductor_classification_separates_released_and_netcdf_only() -> None:
    shipped_names = tuple(overlay.POLOIDAL_CONDUCTORS[:2])
    geometry = StaticGeometry(
        limiter=np.asarray([[1.0, -1.0], [2.0, -1.0], [2.0, 1.0]]),
        coils=(
            _coil(shipped_names[0], 0.8),
            _coil(shipped_names[1], 0.9),
            _coil("ECOILB", 2.2),
        ),
        probe_positions=np.empty((0, 2)),
        flux_loop_positions=np.empty((0, 2)),
        dd_versions={},
        source_path="machine.nc",
    )

    result = overlay.classify_conductors(geometry)

    assert [coil.name for coil in result["shipped"]] == list(shipped_names)
    assert [coil.name for coil in result["netcdf_only"]] == ["ECOILB"]
    assert result["shipped_elements"] == 2
    assert result["netcdf_only_elements"] == 1


def test_generated_receipt_preserves_reproduction_and_comparison_contracts() -> None:
    receipt_path = overlay.DEFAULT_OUTPUT / overlay.RECEIPT_NAME
    receipt = json.loads(receipt_path.read_text())
    solve = receipt["solve_reproduction"]
    comparison = receipt["comparison"]
    contours = receipt["contours"]
    machine = receipt["machine_geometry"]

    assert solve["relative_residual"] == pytest.approx(
        overlay.EXPECTED_RELATIVE_RESIDUAL,
        rel=overlay.RESIDUAL_REPRODUCTION_RELATIVE_TOLERANCE,
        abs=overlay.RESIDUAL_REPRODUCTION_ABSOLUTE_TOLERANCE,
    )
    assert solve["iterations"] == 4
    assert solve["terminal_topology"] == "diverted"
    assert solve["current_relative_error"] <= solve["current_relative_error_tolerance"]
    assert "not a fit" in comparison["interpretation"]
    assert comparison["label_representability_ceiling_fractional_rms"] == 0.0429
    assert contours["kind"] == "unfilled line contours"
    assert contours["computed_once"]
    assert contours["applied_verbatim_to_both_maps_in_both_panels"]
    assert len(contours["level_values_wb_per_radian"]) == overlay.CONTOUR_COUNT
    assert machine["limiter_vertices"] == 82
    assert machine["shipped_conductor_groups"] == 19
    assert machine["netcdf_only_conductor_groups"] == 5
    assert machine["released_grid_shape"] == [65, 65]
    assert (overlay.DEFAULT_OUTPUT / overlay.FIGURE_NAME).is_file()


def test_source_contains_no_filled_contour_or_equilibrium_edit_path() -> None:
    source = Path(overlay.__file__).read_text()

    assert "contourf" not in source
    assert "nova/equilibrium" not in source
    assert "self-consistent equilibrium" in source
