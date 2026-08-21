"""Tests for the DIII-D state-of-play publication figures."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib.axes
import numpy as np

from benchmarks.diiid_poloidal_figures import CoilGeometry


MODULE_PATH = (
    Path(__file__).parents[2] / "benchmarks" / "diiid_state_of_play_figures.py"
)
SPEC = importlib.util.spec_from_file_location(
    "diiid_state_of_play_figures", MODULE_PATH
)
figures = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = figures
SPEC.loader.exec_module(figures)


def _geometry():
    limiter = np.asarray(
        [[0.8, -1.4], [2.6, -1.4], [2.6, 1.4], [0.8, 1.4]], dtype=float
    )
    coils = tuple(
        CoilGeometry(
            name,
            (
                np.asarray(
                    [
                        [radius - 0.04, height - 0.08],
                        [radius + 0.04, height - 0.08],
                        [radius + 0.04, height + 0.08],
                        [radius - 0.04, height + 0.08],
                    ]
                ),
            ),
        )
        for name, radius, height in (
            ("F1A", 0.72, 0.8),
            ("F1B", 0.72, -0.8),
            ("ECOILA", 0.62, 0.1),
            ("ECOILB", 0.62, -0.1),
            ("E567UP", 2.72, 1.2),
            ("E567DN", 2.72, -1.2),
            ("E89UP", 1.7, 1.7),
            ("E89DN", 1.7, -1.7),
        )
    )
    return figures.StaticGeometry(
        limiter=limiter,
        coils=coils,
        probe_positions=np.asarray([[0.9, -0.4], [0.9, 0.4]]),
        flux_loop_positions=np.asarray([[1.0, -0.6], [1.0, 0.6]]),
        dd_versions={"wall": "3.41.0", "pf_active": "3.41.0", "magnetics": "3.41.0"},
        source_path="/data/DIII-D/200000.nc",
    )


def _frame(index: int):
    radius = np.linspace(0.8, 2.6, 25)
    height = np.linspace(-1.4, 1.4, 31)
    rr, zz = np.meshgrid(radius, height)
    given = (rr - (1.65 + 0.02 * index)) ** 2 + 0.5 * zz**2
    tared = 0.28 * given + 0.02 * rr - 0.01 * zz
    radius_map, height_map = np.meshgrid(radius, height, indexing="ij")
    given_current = 2.2e5 * np.exp(
        -((radius_map - 1.65) ** 2 / 0.18**2 + height_map**2 / 0.55**2)
    )
    given_current -= 1.1e5 * np.exp(
        -((radius_map - 2.25) ** 2 / 0.12**2 + (height_map + 0.8) ** 2 / 0.2**2)
    )
    tared_current = 0.2 * given_current
    boundary_angle = np.linspace(0.0, 2.0 * np.pi, 80, endpoint=False)
    given_lcfs = np.column_stack(
        (1.65 + 0.48 * np.cos(boundary_angle), 0.9 * np.sin(boundary_angle))
    )
    extracted_lcfs = given_lcfs + np.asarray([0.01, -0.005])
    topology = figures.TopologyOverlay(
        given_axis=np.asarray([1.65, 0.0]),
        given_x_point=np.asarray([1.65, -0.9]),
        given_lcfs=given_lcfs,
        given_axis_flux_wb=0.0,
        given_x_flux_wb=0.4,
        given_boundary_flux_wb=0.4,
        extracted_axis=np.asarray([1.66, -0.005]),
        extracted_x_point=np.asarray([1.66, -0.89]),
        extracted_lcfs=extracted_lcfs,
        extracted_axis_flux_wb=0.001,
        extracted_x_flux_wb=0.398,
        extracted_boundary_flux_wb=0.399,
        extracted_diverted=True,
        separations_m={
            "magnetic_axis": float(np.hypot(0.01, 0.005)),
            "x_point": float(np.hypot(0.01, 0.01)),
            "lcfs_symmetric_mean": float(np.hypot(0.01, 0.005)),
        },
    )
    return figures.StateFrame(
        spec=figures.FrameSpec(
            name=f"synthetic_{index}",
            shot=f"shot_{index}.parquet",
            frame=10 + index,
            expected_time_ms=100.0 + index,
        ),
        time_ms=100.0 + index,
        radius=radius,
        height=height,
        given_total_zr=given,
        tared_total_zr=tared,
        given_current_density_rz=given_current,
        tared_current_density_rz=tared_current,
        current_valid_rz=np.ones_like(given_current, dtype=bool),
        core_rz=(radius_map - 1.65) ** 2 + height_map**2 < 0.6**2,
        exact_plasma_current_a=1.0e6,
        density_threshold_a_per_m2=2.0e4,
        topology=topology,
    )


def test_named_frames_are_distinct_diverted_publication_inputs():
    assert len(figures.FRAME_SPECS) == 3
    assert len({item.shot for item in figures.FRAME_SPECS}) == 3
    assert [(item.frame, item.expected_time_ms) for item in figures.FRAME_SPECS] == [
        (179, 3740.0),
        (44, 1080.0),
        (36, 980.0),
    ]


def test_boundary_gradient_minimum_uses_shipped_curve_and_map():
    radius = np.linspace(1.0, 2.0, 41)
    height = np.linspace(-0.5, 0.5, 41)
    rr, zz = np.meshgrid(radius, height)
    field = (rr - 1.5) ** 2 + zz**2
    boundary = np.asarray([[1.2, 0.0], [1.5, 0.0], [1.8, 0.0], [1.5, 0.3]], dtype=float)
    point = figures.boundary_gradient_minimum(radius, height, field, boundary)
    np.testing.assert_allclose(point, [1.5, 0.0])


def test_symmetric_boundary_separation_is_in_physical_metres():
    left = np.asarray([[1.0, -1.0], [2.0, -1.0], [2.0, 1.0], [1.0, 1.0]])
    right = left + np.asarray([0.02, 0.0])
    assert 0.009 < figures._boundary_separation(left, right) < 0.021


def test_flux_pairs_share_unfilled_line_levels_and_current_masks_are_transparent(
    tmp_path, monkeypatch
):
    contour_calls = []
    mesh_calls = []
    original_contour = matplotlib.axes.Axes.contour
    original_mesh = matplotlib.axes.Axes.pcolormesh

    def capture_contour(axis, *args, **kwargs):
        contour_calls.append(kwargs.copy())
        return original_contour(axis, *args, **kwargs)

    def capture_mesh(axis, *args, **kwargs):
        mesh_calls.append((np.ma.asarray(args[2]), kwargs.copy()))
        return original_mesh(axis, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "contour", capture_contour)
    monkeypatch.setattr(matplotlib.axes.Axes, "pcolormesh", capture_mesh)
    frames = [_frame(index) for index in range(3)]
    receipt = figures.write_figures(_geometry(), frames, tmp_path)

    assert len(contour_calls) == 9
    assert all(call["colors"] == "#999999" for call in contour_calls)
    assert all(call["linewidths"] == 0.35 for call in contour_calls)
    for row in range(3):
        left = np.asarray(contour_calls[3 + 2 * row]["levels"])
        right = np.asarray(contour_calls[4 + 2 * row]["levels"])
        np.testing.assert_array_equal(left, right)
    density_meshes = [call for call in mesh_calls if call[0].shape == (31, 25)]
    assert len(density_meshes) == 6
    assert all(np.any(np.ma.getmaskarray(values)) for values, _kwargs in density_meshes)
    assert all(
        kwargs["cmap"]._rgba_bad[-1] == 0.0 for _values, kwargs in density_meshes
    )
    assert all(kwargs["norm"].linthresh == 2.0e4 for _values, kwargs in density_meshes)
    assert receipt["rendering"]["contourf_used"] is False
    assert receipt["rendering"]["tared_pairs_use_shared_levels"] is True
    assert receipt["rendering"]["current_colormap_bad_alpha"] == 0.0


def test_receipt_names_every_artifact_quantity_source_and_surviving_count(tmp_path):
    receipt = figures.write_figures(
        _geometry(), [_frame(index) for index in range(3)], tmp_path
    )

    assert [Path(path).name for path in receipt["artifacts"]["figures"]] == list(
        figures.FIGURE_NAMES
    )
    assert all(
        Path(path).stat().st_size > 1_000 for path in receipt["artifacts"]["figures"]
    )
    assert receipt["selection"]["named_frame_count"] == 3
    assert len(receipt["selection"]["frames"]) == 3
    assert len(receipt["figures"]["tared_flux.png"]["panels"]) == 6
    current_panels = receipt["figures"]["current_density_patches.png"]["panels"]
    assert len(current_panels) == 6
    assert all(panel["source"] for panel in current_panels)
    assert all(panel["quantity"] for panel in current_panels)
    assert all(
        panel["absolute_threshold_a_per_m2"] == 2.0e4 for panel in current_panels
    )
    assert all(panel["surviving_cell_count"] > 0 for panel in current_panels)
    assert all(
        panel["subthreshold_cells"] == "masked transparent" for panel in current_panels
    )
    source = MODULE_PATH.read_text()
    assert ".contourf(" not in source
