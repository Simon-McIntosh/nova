"""Regression checks for the native ITER mesh drawing and its save refusals."""

from pathlib import Path
import os
import subprocess
import xml.etree.ElementTree as ET

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
from PIL import Image
import pytest

from scripts.prototypes.iter_bank import render

ROOT = Path(__file__).resolve().parent


@pytest.fixture
def geometry():
    return render.read_geometry(ROOT / "geometry.json")


@pytest.fixture
def scene(geometry):
    return render.build_figure(geometry)


def expected_labels(geometry):
    names = [geometry["coil_names"][owner] for owner in geometry["owners"]]
    names[2:4] = ["CS1\nupper", "CS1\nlower"]
    names[-2:] = ["VS3\nupper", "VS3\nlower"]
    return names


def plasma_collections(scene):
    return [
        c
        for ax in scene.axes
        for c in ax.collections
        if (c.get_gid() or "").startswith("plasma-cells-")
    ]


def test_native_scene_is_readable_and_identities_are_complete(
    scene, geometry, tmp_path
):
    """Measure the native scene independently of the renderer's report."""
    canvas = FigureCanvasAgg(scene)
    canvas.draw()
    rgb = np.asarray(canvas.buffer_rgba())[:, :, :3].astype(float) / 255
    mesh = (rgb[:, :, 2] > rgb[:, :, 0] + 0.02) & (rgb[:, :, 0] > rgb[:, :, 1] + 0.02)
    mesh[:55] = False
    assert mesh.sum() > 10000
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    ratio = 1.05 / (linear @ np.array([0.2126, 0.7152, 0.0722]) + 0.05)
    assert np.median(ratio[mesh]) >= 3
    assert np.mean(rgb.min(axis=2) < 0.90) > 0.045
    labels = expected_labels(geometry)
    assert [text.get_text() for text in scene.axes[0].texts] == labels
    assert len(set(labels)) == len(geometry["coils"])
    report = render.check_scene(scene, labels)
    assert report["sections"] == len(geometry["coils"])
    assert report["labelled"] == len(scene.axes[0].texts)
    assert report["detail_coils"] == 0
    assert all(not ax.axison for ax in scene.axes)
    for ax in scene.axes:
        scene.canvas.draw()
        initial = ax.get_position(original=True)
        actual = ax.get_position()
        assert np.allclose(initial.bounds, actual.bounds)
    output = tmp_path / "mesh.svg"
    render.checked_save(scene, output)
    root = ET.parse(output).getroot()
    ids = [element.get("id") for element in root.iter() if element.get("id")]
    assert len(ids) == len(set(ids))
    assert len([name for name in ids if name.startswith("direct-label")]) == len(labels)
    png = tmp_path / "mesh.png"
    completed = subprocess.run(
        [
            "convert",
            "-background",
            "white",
            "-density",
            "140",
            str(output),
            "-resize",
            "1400",
            str(png),
        ],
        env={**os.environ, "TMPDIR": str(tmp_path), "MAGICK_TMPDIR": str(tmp_path)},
        capture_output=True,
        text=True,
        check=True,
    )
    assert not completed.stderr
    rgb = np.array(Image.open(png).convert("RGB")).astype(float) / 255
    mesh = (rgb[:, :, 2] > rgb[:, :, 0] + 0.02) & (rgb[:, :, 0] > rgb[:, :, 1] + 0.02)
    mesh[:55] = False
    assert mesh.sum() > 10000
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    ratio = 1.05 / (linear @ np.array([0.2126, 0.7152, 0.0722]) + 0.05)
    assert np.median(ratio[mesh]) >= 3
    assert np.mean(rgb.min(axis=2) < 0.90) > 0.045


@pytest.mark.parametrize("damage", ["missing", "wrong_name", "missing_id"])
def test_missing_or_changed_label_is_refused(scene, geometry, damage):
    label = scene.axes[0].texts[-1]
    if damage == "missing":
        label.remove()
    elif damage == "wrong_name":
        label.set_text("unidentified")
    else:
        label.set_gid(None)
    with pytest.raises(ValueError, match="coil labels"):
        render.check_scene(scene, expected_labels(geometry))


def test_missing_coil_polygon_is_refused(scene, geometry):
    coils = next(c for c in scene.axes[0].collections if c.get_gid() == "coil-sections")
    coils.set_verts([path.vertices for path in coils.get_paths()][:-1])
    with pytest.raises(ValueError, match="coil geometry"):
        render.check_scene(scene, expected_labels(geometry))


def test_coil_in_mesh_detail_is_refused(scene, geometry):
    render.draw_coils(scene.axes[1], geometry["coils"][-1:], gid="coil-sections")
    with pytest.raises(ValueError, match="coil geometry"):
        render.check_scene(scene, expected_labels(geometry))


def test_faint_mesh_is_refused(scene, geometry):
    for collection in plasma_collections(scene):
        collection.set_alpha(0.45)
    with pytest.raises(ValueError, match="plasma contrast below 3:1"):
        render.check_scene(scene, expected_labels(geometry))


def test_darker_reference_wall_is_refused(scene, geometry):
    for ax in scene.axes:
        for line in ax.lines:
            line.set_color("#666666")
    with pytest.raises(ValueError, match="reference geometry is darker"):
        render.check_scene(scene, expected_labels(geometry))


@pytest.mark.parametrize("painter", ["coil", "plasma", "line"])
@pytest.mark.parametrize("tagged", [False, True])
@pytest.mark.parametrize("placement", ["axes", "figure"])
def test_label_over_geometry_is_refused(tmp_path, painter, tagged, placement):
    """Both polygon collection branches and lines refuse labels with any ID."""
    figure = Figure(figsize=(14, 7.2), dpi=100)
    ax = figure.add_axes((0.1, 0.1, 0.8, 0.8))
    ax.set_axis_off()
    ax.set(xlim=(0, 1), ylim=(0, 1))
    polygon = np.array([[0.1, 0.5], [0.9, 0.5], [0.9, 0.9], [0.1, 0.9]])
    if painter == "coil":
        render.draw_coils(ax, [polygon])
    elif painter == "plasma":
        render.draw_plasma_cells(ax, [polygon])
    else:
        ax.plot([0.1, 0.9], [0.5, 0.5])
    label = ax.text if placement == "axes" else figure.text
    label(0.5, 0.5, "over data", gid="direct-label-test" if tagged else None)
    with pytest.raises(ValueError, match="label over data"):
        render.checked_save(figure, tmp_path / "refused.svg")
    assert not (tmp_path / "refused.svg").exists()


@pytest.mark.parametrize("damage", ["clipping", "collision"])
def test_invalid_text_is_refused(tmp_path, damage):
    figure = Figure(figsize=(14, 7.2), dpi=100)
    if damage == "clipping":
        figure.text(1.1, 0.5, "outside")
    else:
        figure.text(0.5, 0.5, "one")
        figure.text(0.5, 0.5, "two")
    with pytest.raises(ValueError, match="text outside canvas|text overlaps"):
        render.checked_save(figure, tmp_path / "refused.svg")
    assert not (tmp_path / "refused.svg").exists()


def test_wall_is_drawn_above_the_cell_outlines(scene, geometry):
    render.check_scene(scene, expected_labels(geometry))
    for ax in scene.axes:
        for line in ax.lines:
            if (line.get_gid() or "").startswith(render.WALL_GID):
                line.set_zorder(1)
    with pytest.raises(ValueError, match="first wall is not drawn above"):
        render.check_scene(scene, expected_labels(geometry))


def test_wall_mark_duplicating_the_coil_grey_is_refused(scene, geometry):
    render.check_scene(scene, expected_labels(geometry))
    for ax in scene.axes:
        for line in ax.lines:
            if (line.get_gid() or "").startswith(render.WALL_GID):
                line.set_color("#929292")
    with pytest.raises(ValueError, match="first wall mark duplicates"):
        render.check_scene(scene, expected_labels(geometry))


@pytest.mark.parametrize("damage", ["colour", "style", "label"])
def test_legend_entry_must_carry_its_component_mark(scene, geometry, damage):
    render.check_scene(scene, expected_labels(geometry))
    handle = scene.legends[0].legend_handles[-1]
    if damage == "colour":
        handle.set_color("#123456")
    elif damage == "style":
        handle.set_linestyle("-")
    else:
        handle.set_label("wall")
    with pytest.raises(ValueError, match="legend"):
        render.check_scene(scene, expected_labels(geometry))
