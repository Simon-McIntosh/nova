"""Check rendered axis flux content and shared figure-save ownership."""

import argparse
import importlib.util
from pathlib import Path
from unittest.mock import patch

import numpy as np


def check_axis(module, scene, destination):
    """Require real contours from both fields, beyond their null markers."""
    captured = []
    with patch.object(
        module, "checked_save", side_effect=lambda fig, *_: captured.append(fig)
    ):
        module.poloidal_figure(
            scene, destination / "poloidal.svg", destination / "poloidal.png"
        )
    fig = captured[0]
    full_counts = [
        sum(len(path.vertices) >= 4 for path in item.get_paths())
        for item in fig.axes[0].collections
        if hasattr(item, "levels")
    ]
    assert min(full_counts) >= 4, "full-view positive control must see both fields"
    axis = fig.axes[1]
    contours = [item for item in axis.collections if hasattr(item, "levels")]
    assert len(contours) == 2
    np.testing.assert_array_equal(contours[0].levels, contours[1].levels)
    counts = [
        sum(len(path.vertices) >= 4 for path in contour.get_paths())
        for contour in contours
    ]
    fig.canvas.draw()
    pixels = np.asarray(fig.canvas.buffer_rgba())
    box = axis.get_window_extent()
    crop = pixels[
        pixels.shape[0] - int(box.y1) : pixels.shape[0] - int(box.y0),
        int(box.x0) : int(box.x1),
        :3,
    ]
    blue = np.rint(
        np.asarray(module.matplotlib.colors.to_rgb(module.BRANCH)) * 255
    ).astype(int)
    blue_count = int(np.sum(np.max(np.abs(crop.astype(int) - blue), axis=-1) < 30))
    print(
        f"AXIS_FLUX nonempty_paths={counts} blue_pixels={blue_count} "
        f"full_view_paths={full_counts}",
        flush=True,
    )
    assert min(counts) >= 4, (
        "axis close-up must draw at least four contours of each field"
    )
    assert blue_count >= 1000, "axis field must occupy more than its triangle marker"
    module.plt.close(fig)


def check_reuse(module, destination):
    """Exercise the existing save owner through the prototype's wrapper."""
    from scripts.prototypes.bank_solve_profile.plot import save_checked

    assert getattr(module, "save_checked", None) is save_checked, (
        "figure save policy must use the shared owner"
    )
    with patch.object(module, "save_checked", wraps=save_checked) as owner:
        fig = module.plt.figure(figsize=(14, 7.2))
        module.checked_save(fig, destination / "shared.svg", destination / "shared.png")
        assert owner.call_count == 1
    fig = module.plt.figure(figsize=(14, 7.2))
    fig.text(-0.2, 0.5, "outside canvas")
    try:
        module.checked_save(
            fig, destination / "refused.svg", destination / "refused.png"
        )
    except ValueError as error:
        assert "outside canvas" in str(error)
        print("SHARED_SAVE owner_calls=1 clipped_text=refused", flush=True)
    else:
        raise AssertionError("shared canvas guard must refuse clipped text")
    finally:
        module.plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--case", choices=("axis", "reuse"), required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location(
        "arclength_plot_under_test", args.source
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    print(f"MODULE={module.__file__} CWD={Path.cwd().resolve()}", flush=True)
    module.plt.style.use("data-ink")
    destination = args.scene.parent / "plot-check-artifacts"
    destination.mkdir(exist_ok=True)
    if args.case == "axis":
        check_axis(module, args.scene, destination)
    else:
        check_reuse(module, destination)


if __name__ == "__main__":
    main()
