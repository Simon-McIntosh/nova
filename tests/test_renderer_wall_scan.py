"""Tests for the scattered-field renderer scanner.

The scanner reports renderer calls that contour a field interpolated from
scattered nodes without masking it to the wall. The expectations below pin the
rule on named control sites and on crafted sources; they assert no total count
of unguarded sites, so a repair that fixes a call site cannot turn this file
red. The corpus-wide reconciliation against the reviewed inventory lives in the
report script under
``docs/figures/figure-and-solver-audit/fsa-renderer-wall-scanner-guard-form/``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import renderer_wall_scan as scan  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
ROOTS = ["benchmarks", "docs/figures"]


def _records():
    return scan.scan_paths([str(REPO / root) for root in ROOTS])


def _sites(records):
    return {(record.path, record.line) for record in records}


def _scan_source(tmp_path, source: str):
    """Scan one crafted module and return the reported (base name, line) set.

    A crafted module lives outside the repository, so the scanner reports its
    absolute path; the base name identifies it here.
    """
    (tmp_path / "module.py").write_text(source)
    records = scan.scan_paths([str(tmp_path)])
    return {(Path(record.path).name, record.line) for record in records}


def test_scanner_is_importable_and_typed():
    record = scan.UnmaskedCall("a.py", 3, "tricontour")
    assert str(record) == "a.py:3: tricontour"


def test_flags_a_painter_over_a_scattered_raster_without_wall(tmp_path):
    # A certificate painter drawing a _raster_field raster with no wall was a
    # real unguarded site; it is now repaired, so this control is crafted and
    # must redden if the painter rule is removed, rather than depending on an
    # unrepaired site remaining in the tree.
    sites = _scan_source(
        tmp_path,
        "radial, height, raster = certificate._raster_field(coords, values, wall)\n"
        "driver.poloidal.draw_flux_contours(\n"
        "    axis, radial, height, raster, levels, color=color\n"
        ")\n",
    )
    assert sites == {("module.py", 2)}


def test_flags_a_direct_tricontour_on_scattered_nodes(tmp_path):
    # A ledger panel drawing an Axes tricontour over scattered nodes straight to
    # the figure was a real unguarded site; it is now repaired, so this control
    # is crafted and must redden if the tricontour rule is removed.
    sites = _scan_source(
        tmp_path,
        "axes.tricontour(node[:, 0], node[:, 1], grid_flux, levels)\n",
    )
    assert sites == {("module.py", 1)}


def test_control_does_not_flag_painter_that_passes_a_wall_argument():
    # Control site: the closed-forms render route draws through
    # draw_scattered_contours, whose wall is required.
    assert (
        "docs/figures/centroid-constrained-oracle-solve/root-cause/closed-forms/render.py",
        52,
    ) not in _sites(_records())


def test_control_does_not_flag_painter_fed_by_a_pre_blanked_raster():
    # Control site: benchmarks/contour_tree_explanatory_figure.py:359 draws a
    # raster produced by _masked_field, already blanked to the wall.
    assert (
        "benchmarks/contour_tree_explanatory_figure.py",
        359,
    ) not in _sites(_records())


def test_control_ignores_post_filtered_probe():
    # Control site: benchmarks/plasma_cell_trip_panels.py:406 triangulates only
    # to extract polylines, rejects every curve outside the wall, and closes
    # the figure without drawing it. Exempt by path and line.
    assert ("benchmarks/plasma_cell_trip_panels.py", 406) not in _sites(_records())


def test_treats_a_positionally_passed_wall_as_guarded(tmp_path):
    # draw_flux_contours declares wall as its ninth positional parameter
    # (nova/media/poloidal.py), so a call that hands the wall positionally is
    # guarded and must not be reported. The raster is a scattered producer, so
    # without positional-wall recognition the call would be flagged: this
    # control reddens if that recognition is removed.
    sites = _scan_source(
        tmp_path,
        "raster = _raster_field(coordinates, values)\n"
        "draw_flux_contours(\n"
        "    axes,\n"
        "    radius,\n"
        "    height,\n"
        "    raster,\n"
        "    levels,\n"
        "    style,\n"
        "    color,\n"
        "    linewidth,\n"
        "    wall,\n"
        ")\n",
    )
    assert sites == set()


def test_reports_a_wall_less_painter(tmp_path):
    sites = _scan_source(
        tmp_path,
        "raster = _raster_field(coordinates, values)\n"
        "draw_flux_contours(axes, radius, height, raster, levels)\n",
    )
    assert sites == {("module.py", 2)}


def test_reports_a_painter_that_passes_wall_none(tmp_path):
    # An explicit wall=None does not guard: the painter blanks nothing, so the
    # call is reported exactly as if the argument were absent. This control
    # reddens if wall=None is treated as a wall.
    sites = _scan_source(
        tmp_path,
        "raster = _raster_field(coordinates, values)\n"
        "draw_flux_contours(axes, radius, height, raster, levels, wall=None)\n",
    )
    assert sites == {("module.py", 2)}


def test_does_not_report_a_painter_that_passes_a_real_wall(tmp_path):
    # The companion to the wall=None control: a non-null wall is a guard.
    sites = _scan_source(
        tmp_path,
        "raster = _raster_field(coordinates, values)\n"
        "draw_flux_contours(axes, radius, height, raster, levels, wall=units)\n",
    )
    assert sites == set()


def test_clean_tree_reports_nothing(tmp_path):
    (tmp_path / "render.py").write_text(
        "from nova.media.poloidal import draw_scattered_contours\n"
        "draw_scattered_contours(ax, r, z, f, levels, wall=wall)\n"
    )
    assert scan.scan_paths([str(tmp_path)]) == []


def test_cli_exits_nonzero_on_a_file_with_a_site(tmp_path):
    target = tmp_path / "site.py"
    target.write_text(
        "raster = _raster_field(coordinates, values)\n"
        "draw_flux_contours(axes, radius, height, raster, levels)\n"
    )
    process = subprocess.run(
        [sys.executable, str(REPO / "tests" / "renderer_wall_scan.py"), str(target)],
        capture_output=True,
        text=True,
    )
    assert process.returncode == 1
    assert "site.py:2" in process.stdout


def test_cli_exits_zero_on_a_clean_tree(tmp_path):
    (tmp_path / "clean.py").write_text("x = np.ones(3)\n")
    process = subprocess.run(
        [sys.executable, str(REPO / "tests" / "renderer_wall_scan.py"), str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0
    assert process.stdout == ""
