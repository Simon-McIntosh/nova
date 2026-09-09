"""Render the JT-60SA first wall as its five typed units.

JT-60SA's machine description carries one closed vessel outline and four
open strike-point first-wall structures. This panel draws every unit as its
own polyline with the unit-aware wall painter, so the atlas shows that a
five-unit wall is drawn as five structures rather than being collapsed into
one polygon. The receipt records each unit's vertex count and closure and the
number of polylines drawn.
"""

from __future__ import annotations

import json
from pathlib import Path

import imas
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nova.equilibrium.wall_mask import WallUnit, wall_units_from_ids
from nova.media import poloidal
from nova.media.ink import DEFAULT_INK

OUT_DIR = Path(__file__).resolve().parent
PNG_PATH = OUT_DIR / "jt60sa-wall-units.png"
RECEIPT_PATH = OUT_DIR / "jt60sa-wall-units.json"

IMAS_PATH = "/home/ITER/mcintos/public/imasdb/jt-60sa_md"
DD_VERSION = "4.1.0"


def read_units() -> tuple[WallUnit, ...]:
    """Read every limiter unit from the single JT-60SA wall description."""

    with imas.DBEntry(
        f"imas:hdf5?path={IMAS_PATH}", "r", dd_version=DD_VERSION
    ) as entry:
        return wall_units_from_ids(entry.get("wall", 0, lazy=False, autoconvert=False))


def draw_units(units: tuple[WallUnit, ...]) -> int:
    """Draw the wall as one polyline per unit and return the polyline count."""

    figure, axes = plt.subplots(figsize=(5.2, 5.2))
    axes.set_aspect("equal")
    poloidal.draw_wall(axes, units=units, style=DEFAULT_INK, linewidth=1.6)
    for index, unit in enumerate(units):
        centroid = (float(np.mean(unit.r)), float(np.mean(unit.z)))
        closure = "closed" if unit.closed else "open"
        axes.annotate(
            f"{index} {unit.kind} / {unit.r.size} pts / {closure}",
            xy=centroid,
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7.5,
            color="#222222",
            zorder=DEFAULT_INK.zorder_wall + 2,
        )
    all_r = np.concatenate([unit.r for unit in units])
    all_z = np.concatenate([unit.z for unit in units])
    axes.set_xlabel("R [m]")
    axes.set_ylabel("Z [m]")
    axes.set_xlim(float(all_r.min()) - 0.2, float(all_r.max()) + 0.2)
    axes.set_ylim(float(all_z.min()) - 0.2, float(all_z.max()) + 0.2)
    figure.savefig(PNG_PATH, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return len(units)


def write_receipt(units: tuple[WallUnit, ...], polyline_count: int) -> None:
    """Record the unit table and the number of polylines the panel draws."""

    payload = {
        "subject": "jt60sa-wall-units",
        "machine": "JT-60SA",
        "source": IMAS_PATH,
        "dd_version": DD_VERSION,
        "units": [
            {
                "index": index,
                "name": unit.name,
                "kind": unit.kind,
                "vertex_count": int(unit.r.size),
                "closed": bool(unit.closed),
            }
            for index, unit in enumerate(units)
        ],
        "drawn_as_polylines": int(polyline_count),
        "figure": PNG_PATH.name,
        "renderer": Path(__file__).name,
    }
    RECEIPT_PATH.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def main() -> None:
    units = read_units()
    polyline_count = draw_units(units)
    write_receipt(units, polyline_count)
    print(
        "JT60SA_WALL_UNITS "
        + json.dumps(
            {
                "count": len(units),
                "vertex_counts": [int(unit.r.size) for unit in units],
                "closures": [bool(unit.closed) for unit in units],
                "polyline_count": polyline_count,
                "png": str(PNG_PATH),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
