"""Render the multi-unit wall figure from a committed receipt.

Two poloidal panels: a synthetic two-unit fixture (one closed vessel and one
open blade) beside the JT-60SA machine description, which carries one closed
vessel and four open strike-point first-wall structures. Both panels are drawn
through the shared poloidal painters in :mod:`nova.media.poloidal`, so the axis
is off, no gridlines are drawn, and each unit keeps its own closure.

The figure is regenerated from a receipt, never from a live solve: the receipt
records every unit's vertices, kind and closure, and
:func:`render_only` draws it without opening an IDS. The IDS read that FILLS the
receipt happens once, in :func:`build_receipt`, and is a plain IDS read (no
solve).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np

from nova.equilibrium.wall_mask import WallUnit, material_unit, vessel_unit
from nova.media import poloidal
from nova.media.ink import DEFAULT_INK

if TYPE_CHECKING:
    import matplotlib

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "figures" / "multi-unit-limiter-wall"
RECEIPT_PATH = OUT_DIR / "wall-units-receipt.json"
PNG_PATH = OUT_DIR / "wall-units.png"
SVG_PATH = OUT_DIR / "wall-units.svg"

JT60SA_PATH = "/home/ITER/mcintos/public/imasdb/jt-60sa_md"
JT60SA_DD_VERSION = "4.1.0"

# One colour per unit index so five structures cannot read as one ring. The
# closure decides the linestyle, so an open segment stays dashed whatever
# colour it carries.
UNIT_COLORS = (
    "#3366cc",
    "#cc0000",
    "#cc7722",
    "#2e7d32",
    "#7b1fa2",
)
UNIT_LINESTYLE_CLOSED = "solid"
UNIT_LINESTYLE_OPEN = "dashed"
OPEN_DASH_PATTERN = (7.0, 4.0)


def unit_style(index: int, closed: bool) -> tuple[str, str]:
    """Return the (colour, linestyle) a unit at ``index`` is drawn with."""

    colour = UNIT_COLORS[index % len(UNIT_COLORS)]
    linestyle = UNIT_LINESTYLE_CLOSED if closed else UNIT_LINESTYLE_OPEN
    return colour, linestyle


def synthetic_units() -> tuple[WallUnit, WallUnit]:
    """The two-unit fixture: one closed vessel and one open blade."""

    vessel = vessel_unit([0.0, 2.8, 2.8, 0.0], [-1.7, -1.7, 1.7, 1.7], name="vessel")
    blade = material_unit([0.6, 2.4], [0.0, 0.0], closed=False, name="blade")
    return vessel, blade


def jt60sa_units() -> tuple[WallUnit, ...]:
    """Read every limiter unit from the single JT-60SA wall description."""

    from nova.equilibrium.wall_mask import wall_units_from_ids

    import imas

    with imas.DBEntry(
        f"imas:hdf5?path={JT60SA_PATH}", "r", dd_version=JT60SA_DD_VERSION
    ) as entry:
        return wall_units_from_ids(entry.get("wall", 0, lazy=False, autoconvert=False))


def build_receipt(path: Path = RECEIPT_PATH) -> dict:
    """Read the descriptions once and write the receipt the figure renders from."""

    panels = (
        _panel_row("synthetic", "synthetic-two-unit", None, None, synthetic_units()),
        _panel_row(
            "jt60sa",
            "JT-60SA",
            JT60SA_PATH,
            JT60SA_DD_VERSION,
            jt60sa_units(),
        ),
    )
    payload = {
        "subject": "wall-units",
        "panels": panels,
        "renderer": "benchmarks/wall_units_render.py",
        "figures": {"png": PNG_PATH.name, "svg": SVG_PATH.name},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return payload


def _panel_row(name, machine, source, dd_version, units: Sequence[WallUnit]) -> dict:
    return {
        "name": name,
        "machine": machine,
        "source": source,
        "dd_version": dd_version,
        "units": [
            {
                "index": index,
                "name": unit.name,
                "kind": unit.kind,
                "closed": bool(unit.closed),
                "r": [float(value) for value in unit.r],
                "z": [float(value) for value in unit.z],
            }
            for index, unit in enumerate(units)
        ],
    }


def load_receipt(path: Path = RECEIPT_PATH) -> dict:
    return json.loads(Path(path).read_text())


def panel_units(panel: dict) -> tuple[WallUnit, ...]:
    """Rebuild the typed unit collection a receipt panel recorded."""

    return tuple(
        WallUnit(
            r=row["r"],
            z=row["z"],
            kind=row["kind"],
            closed=bool(row["closed"]),
            name=row.get("name", ""),
        )
        for row in panel["units"]
    )


def draw_panel(
    axes: matplotlib.axes.Axes,
    units: Sequence[WallUnit],
    style=DEFAULT_INK,
) -> list:
    """Draw every unit in its own colour and closure, with endpoint markers."""

    handles = []
    for index, unit in enumerate(units):
        colour, linestyle = unit_style(index, bool(unit.closed))
        poloidal.draw_wall(
            axes,
            units=(unit,),
            style=style,
            color=colour,
            linestyle=linestyle,
            linewidth=1.8,
            label=f"unit {index} {unit.kind} · {unit.r.size} pts · "
            + ("closed" if unit.closed else "open"),
        )
        line = axes.lines[-1]
        if not unit.closed:
            line.set_dashes(OPEN_DASH_PATTERN)
        points = np.column_stack((unit.r, unit.z))
        axes.plot(
            points[[0, -1], 0],
            points[[0, -1], 1],
            marker="o",
            markersize=3.6,
            linestyle="none",
            color=colour,
            zorder=style.zorder_markers,
        )
        handles.append(line)
    return handles


def render(receipt: dict) -> tuple[matplotlib.figure.Figure, np.ndarray]:
    """Draw the two panels from a receipt and return the figure and its axes."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from nova.media.ink import poloidal_axes

    figure, axes = plt.subplots(1, 2, figsize=(11.0, 5.4))
    legend_handles: list = []
    legend_labels: list[str] = []
    for axes_row, panel in zip(axes, receipt["panels"], strict=True):
        poloidal_axes(axes_row, DEFAULT_INK)
        units = panel_units(panel)
        handles = draw_panel(axes_row, units)
        all_r = np.concatenate([unit.r for unit in units])
        all_z = np.concatenate([unit.z for unit in units])
        pad_r = 0.08 * max(float(np.ptp(all_r)), 1e-6) + 0.15
        pad_z = 0.08 * max(float(np.ptp(all_z)), 1e-6) + 0.15
        axes_row.set_xlim(float(all_r.min()) - pad_r, float(all_r.max()) + pad_r)
        axes_row.set_ylim(float(all_z.min()) - pad_z, float(all_z.max()) + pad_z)
        axes_row.set_title(
            f"{panel['machine']}: {len(units)} units",
            fontsize=DEFAULT_INK.label_fontsize,
        )
        if panel["name"] == "jt60sa":
            legend_handles = handles
            legend_labels = [handle.get_label() for handle in handles]
    if legend_handles:
        axes[-1].legend(
            legend_handles,
            legend_labels,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            fontsize=DEFAULT_INK.label_fontsize - 1.0,
            frameon=False,
        )
    figure.tight_layout()
    return figure, axes


def save(figure, png_path: Path = PNG_PATH, svg_path: Path = SVG_PATH) -> None:
    for path in (png_path, svg_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        png_path,
        dpi=DEFAULT_INK.figure_dpi,
        bbox_inches="tight",
        facecolor=DEFAULT_INK.figure_facecolor,
    )
    figure.savefig(
        svg_path, bbox_inches="tight", facecolor=DEFAULT_INK.figure_facecolor
    )


def render_only(
    receipt_path: Path = RECEIPT_PATH,
    png_path: Path = PNG_PATH,
    svg_path: Path = SVG_PATH,
) -> dict:
    """Render the committed receipt to PNG and SVG with no solve and no IDS."""

    import matplotlib.pyplot as plt

    receipt = load_receipt(receipt_path)
    figure, axes = render(receipt)
    save(figure, png_path, svg_path)
    plt.close(figure)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--build",
        action="store_true",
        help="read the machine descriptions and rewrite the receipt",
    )
    parser.add_argument("--receipt", type=Path, default=RECEIPT_PATH)
    parser.add_argument("--png", type=Path, default=PNG_PATH)
    parser.add_argument("--svg", type=Path, default=SVG_PATH)
    args = parser.parse_args()

    if args.build:
        build_receipt(args.receipt)
    receipt = render_only(args.receipt, args.png, args.svg)
    print(
        "WALL_UNITS_RENDER "
        + json.dumps(
            {
                "panels": [
                    {
                        "name": panel["name"],
                        "units": len(panel["units"]),
                        "closures": [row["closed"] for row in panel["units"]],
                    }
                    for panel in receipt["panels"]
                ],
                "png": str(args.png),
                "svg": str(args.svg),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
