"""Render the persisted map residual and closed-form field comparisons."""

import json
from pathlib import Path
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmarks.plasma_cell_terminal_state import _draw_nulls
from nova.media import poloidal
from nova.media.ink import DEFAULT_INK, poloidal_axes
from nova.media.sources.frame import WallUnit


OUTPUT = Path(__file__).resolve().parent


def main():
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    print(f"revision={revision} tree={Path.cwd()} command={sys.argv!r}", flush=True)
    print(f"module.__file__={poloidal.__file__} cwd={Path.cwd()}", flush=True)
    receipt = json.loads((OUTPUT / "map-residual.json").read_text())
    fields = json.loads((OUTPUT / "field-response.json").read_text())
    coordinates = np.asarray(receipt["coordinates_rz_m"])
    wall = np.asarray(receipt["wall_rz_m"])
    offsets = receipt["wall_unit_offsets"]
    units = tuple(
        WallUnit(wall[a:b, 0], wall[a:b, 1], closed=closed, kind="vessel")
        for a, b, closed in zip(
            offsets[:-1], offsets[1:], receipt["wall_unit_closed"], strict=True
        )
    )
    # This fixture declares one closed vessel, with no material obstacles.
    assert len(units) == 1 and units[0].closed
    analytic = np.asarray(receipt["analytic_flux_wb"])
    mapped = np.asarray(receipt["mapped_flux_wb"])
    residual = np.asarray(receipt["residual_wb"])
    style = DEFAULT_INK.variant(wall_linewidth=1.2, label_fontsize=20)
    reference_style = style.variant(
        axis_color="#222222", xpoint_color="#222222", axis_markersize=13
    )
    mapped_style = style.variant(
        axis_color=style.flux_color, xpoint_color=style.flux_color, axis_markersize=8
    )

    def contours(axis, values, levels, color="#222222", dashed=False):
        drawn = poloidal.draw_scattered_contours(
            axis,
            coordinates[:, 0],
            coordinates[:, 1],
            values,
            levels,
            units,
            style=style,
            color=color,
            linewidth=2.6 if dashed else 3.0,
        )
        if dashed:
            drawn.set_linestyle("dashed")
        assert any(len(part) > 1 for group in drawn.allsegs for part in group)
        axis.clabel(drawn, inline=True, fontsize=20, fmt="%.3g")
        return drawn

    def finish(axis, label):
        poloidal.draw_wall(axis, units=units, style=style)
        poloidal_axes(axis, style)
        axis.text(0.5, 1.01, label, ha="center", transform=axis.transAxes, fontsize=20)

    def save(figure, name):
        for suffix in ("png", "svg"):
            figure.savefig(OUTPUT / f"{name}.{suffix}", dpi=100)
        plt.close(figure)

    # Level direction and extent come from the stored reference anchors.
    nulls = receipt["reference_nulls"]
    levels = np.linspace(nulls["axis_flux_wb"], nulls["boundary_flux_wb"], 8)[1:-1]
    extent = float(np.max(np.abs(residual)))
    error_levels = np.linspace(-extent, extent, 11)[1:-1]
    figure, axes = plt.subplots(1, 3, figsize=(14, 7), constrained_layout=True)
    for axis, values, label in zip(
        axes,
        (analytic, mapped, residual),
        ("Analytic flux [Wb]", "One-map flux [Wb]", "Map residual [Wb]"),
        strict=True,
    ):
        drawn = contours(axis, values, error_levels if values is residual else levels)
        if values is residual:
            drawn.set_linestyle(
                ["dashed" if level < 0 else "solid" for level in drawn.levels]
            )
        if values is not residual:
            _draw_nulls(axis, receipt["reference_nulls"], units, reference_style)
            _draw_nulls(axis, receipt["mapped_nulls"], units, mapped_style)
        finish(axis, label)
    save(figure, "analytic-map-residual")
    figure, axes = plt.subplots(1, 2, figsize=(14, 8), constrained_layout=True)
    actual = np.asarray(fields["actual_unit_flux_wb"])
    expected = np.asarray(fields["expected_unit_flux_wb"])
    field_levels = []
    for index, axis in enumerate(axes):
        level = np.linspace(expected[:, index].min(), expected[:, index].max(), 10)[
            1:-1
        ]
        field_levels.append(level.tolist())
        contours(axis, actual[:, index], level, color=style.flux_color)
        contours(axis, expected[:, index], level, dashed=True)
        finish(axis, ("Vertical field: +1 T", "Radial field: +1 T")[index])
        axis.text(
            0.5,
            -0.04,
            "Code (blue); closed form (dashed)",
            transform=axis.transAxes,
            ha="center",
            fontsize=20,
        )
    save(figure, "unit-field-closed-forms")
    (OUTPUT / "panels.json").write_text(
        json.dumps(
            {
                "map_shared_levels_wb": levels.tolist(),
                "residual_levels_wb": error_levels.tolist(),
                "unit_field_levels_wb": field_levels,
                "interpolation": (
                    "linear triangulation of persisted samples; "
                    "triangles masked against wall units"
                ),
                "unit_field_nulls": (
                    "No stationary points at R>0: vertical gradient 2*pi*R; "
                    "radial gradient (-2*pi*Z,-2*pi*R)."
                ),
                "map_null_markers": (
                    "Black large triangle/cross: reference; blue small triangle/cross: "
                    "one-map field; both sets in both flux panels."
                ),
                "map_caption": (
                    "135 cells, exact clipping, analytic start, one map application; "
                    "no terminal solve. Sup residual and prior comparison are in "
                    "map-residual.json. The residual panel is a difference field, "
                    "not equilibrium flux; dashed contours are negative."
                ),
            },
            indent=2,
        )
        + "\n"
    )
    print("RENDER_COMPLETE analytic-map-residual.png unit-field-closed-forms.png")


if __name__ == "__main__":
    main()
