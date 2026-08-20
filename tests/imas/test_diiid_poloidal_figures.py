"""Tests for the DIII-D poloidal contour publication figures."""

import importlib.util
import sys
from pathlib import Path

import matplotlib.axes
import numpy as np


MODULE_PATH = Path(__file__).parents[2] / "benchmarks" / "diiid_poloidal_figures.py"
SPEC = importlib.util.spec_from_file_location("diiid_poloidal_figures", MODULE_PATH)
figures = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = figures
SPEC.loader.exec_module(figures)


def _geometry():
    limiter = np.asarray(
        [[0.8, -1.4], [2.6, -1.4], [2.6, 1.4], [0.8, 1.4]], dtype=float
    )
    coils = (
        figures.CoilGeometry(
            "OH",
            (
                np.asarray([[0.65, -0.5], [0.75, -0.5], [0.75, 0.0], [0.65, 0.0]]),
                np.asarray([[0.65, 0.0], [0.75, 0.0], [0.75, 0.5], [0.65, 0.5]]),
            ),
        ),
        figures.CoilGeometry(
            "PF",
            (np.asarray([[2.7, 0.3], [2.8, 0.3], [2.8, 0.5], [2.7, 0.5]]),),
        ),
    )
    return figures.StaticGeometry(
        limiter=limiter,
        coils=coils,
        probe_positions=np.asarray([[0.9, -0.4], [0.9, 0.4]]),
        flux_loop_positions=np.asarray([[1.0, -0.6], [1.0, 0.6]]),
        dd_versions={"wall": "3.41.0", "pf_active": "3.41.0", "magnetics": "3.41.0"},
        source_path="/data/DIII-D/200000.nc",
    )


def _frame():
    radius = np.linspace(0.8, 2.6, 25)
    height = np.linspace(-1.4, 1.4, 31)
    rr, zz = np.meshgrid(radius, height)
    label = (rr - 1.7) ** 2 + 0.6 * zz**2
    names = (
        *(f"TS_core_r+0_{index}" for index in range(4)),
        *(f"TS_divertor_r+1_{index}" for index in range(4)),
        *(f"TS_tangential_r+0_{index}" for index in range(4)),
    )
    positions = np.asarray(
        [
            [1.9, -0.6],
            [1.9, -0.2],
            [1.9, 0.2],
            [1.9, 0.6],
            [1.4, -1.0],
            [1.4, -0.8],
            [1.4, 0.8],
            [1.4, 1.0],
            [1.5, -0.03],
            [1.7, -0.04],
            [1.9, -0.05],
            [2.1, -0.06],
        ]
    )
    separatrix = np.column_stack(
        [
            1.7 + 0.45 * np.cos(np.linspace(0, 2 * np.pi, 80)),
            0.9 * np.sin(np.linspace(0, 2 * np.pi, 80)),
        ]
    )
    return figures.CompetitionFrame(
        shot="shot.parquet",
        frame=12,
        time_ms=420.0,
        recorded_r2=0.995,
        radius=radius,
        height=height,
        label=label,
        separatrix=separatrix,
        thomson_names=tuple(names),
        thomson_positions=positions,
        row={},
        additive_gauge=0.2,
    )


def _fields(frame):
    return {
        "label": frame.label,
        "plasma": 0.7 * frame.label,
        "remainder": 0.3 * frame.label,
        "coil": 0.28 * frame.label + 0.01,
        "residual": 0.02 * frame.label - 0.01,
    }


def test_house_style_matches_imas_ink_values():
    style = figures.STYLE
    assert style.contour_color == "#999999"
    assert style.contour_linewidth == 0.35
    assert style.coil_edgecolor == "#888888"
    assert style.coil_facecolor == "none"
    assert style.wall_color == "#000000"
    assert style.wall_linewidth == 1.0
    assert style.separatrix_color == "#cc0000"
    assert style.separatrix_linewidth == 1.5
    assert style.figure_dpi == 120


def test_coordinate_line_extends_to_limiter():
    geometry = _geometry()
    points = np.asarray([[1.4, -0.8], [1.4, 0.0], [1.4, 0.8]])
    anchor, direction, residual = figures._fit_collinear_line(points)
    line = figures._extend_line_to_limiter(anchor, direction, geometry.limiter)
    assert residual < 1e-12
    np.testing.assert_allclose(line[:, 0], 1.4)
    np.testing.assert_allclose(np.sort(line[:, 1]), [-1.4, 1.4])


def test_grid_axis_preserves_endpoints_and_removes_spacing_jitter():
    stored = np.asarray([0.8399999738, 1.4066666365, 1.9733333588, 2.5399999619])
    canonical = figures._canonical_axis(stored)
    assert canonical[0] == stored[0]
    assert canonical[-1] == stored[-1]
    np.testing.assert_allclose(np.diff(canonical), np.diff(canonical)[0])


def test_four_figures_use_only_grey_unfilled_contours_and_equal_axes(
    tmp_path, monkeypatch
):
    contour_calls = []
    aspect_calls = []
    original_contour = matplotlib.axes.Axes.contour
    original_aspect = matplotlib.axes.Axes.set_aspect

    def capture_contour(axis, *args, **kwargs):
        contour_calls.append(kwargs.copy())
        return original_contour(axis, *args, **kwargs)

    def capture_aspect(axis, aspect, *args, **kwargs):
        aspect_calls.append(aspect)
        return original_aspect(axis, aspect, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "contour", capture_contour)
    monkeypatch.setattr(matplotlib.axes.Axes, "set_aspect", capture_aspect)
    frame = _frame()
    receipt = figures.write_figures(_geometry(), frame, _fields(frame), tmp_path)

    assert [Path(path).name for path in receipt["figures"]] == [
        "machine_geometry.png",
        "competition_psi.png",
        "plasma_subtraction_sequence.png",
        "thomson_and_magnetics_geometry.png",
    ]
    assert all(Path(path).stat().st_size > 1_000 for path in receipt["figures"])
    assert contour_calls
    assert all(call["colors"] == "#999999" for call in contour_calls)
    assert all(call["linewidths"] == 0.35 for call in contour_calls)
    assert all(aspect == "equal" for aspect in aspect_calls)
    assert receipt["rendering"]["contourf_used"] is False
    assert receipt["rendering"]["imshow_used"] is False
    assert receipt["rendering"]["colormap_used"] is False


def test_receipt_counts_coil_groups_and_geometry_only_diagnostics(tmp_path):
    frame = _frame()
    receipt = figures.write_figures(_geometry(), frame, _fields(frame), tmp_path)

    assert receipt["machine_geometry"]["active_coils"] == 2
    assert receipt["machine_geometry"]["active_elements"] == 3
    assert receipt["machine_geometry"]["coil_labels"] == 2
    assert receipt["machine_geometry"]["label_unit"] == "one label per coil"
    assert receipt["diagnostic_geometry"]["derived_sightlines"] == 2
    assert receipt["diagnostic_geometry"]["tangential_sightline"].startswith(
        "not recoverable"
    )
    assert receipt["content_fence"]["netcdf_magnetics_signal_arrays_read"] == []
    assert receipt["content_fence"]["netcdf_equilibrium_ids_opened"] is False
    source = MODULE_PATH.read_text()
    assert ".contourf(" not in source
    assert ".imshow(" not in source
