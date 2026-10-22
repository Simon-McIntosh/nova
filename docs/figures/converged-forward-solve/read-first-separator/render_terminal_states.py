"""Render retained forward states alongside their analytic certificate fields."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import runpy

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import nova

from nova.media import poloidal
from nova.media.ink import DEFAULT_INK, poloidal_axes
from nova.equilibrium.wall_mask import WallUnit

RETAINED = "#1f6f8b"
REFERENCE = "#333333"


def render(receipts: Path, output: Path, preview: Path) -> None:
    """Use saved field samples; never rerun or imply a converged correction."""
    print(f"nova.__file__={nova.__file__}", flush=True)
    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    handles = {}
    levels_by_row = []
    for ax, (kind, cells) in zip(
        axes.flat,
        [("limited", 550), ("limited", 2000), ("diverted", 550), ("diverted", 2000)],
        strict=True,
    ):
        row = json.loads((receipts / f"row_{kind}_{cells}.json").read_text())
        panel = row["panel"]
        coordinates = np.asarray(panel["coordinates"])
        wall = np.asarray(panel["wall"])
        units = (WallUnit(wall[:, 0], wall[:, 1], kind="vessel", closed=True),)
        poloidal_axes(ax)
        triangulation = mtri.Triangulation(coordinates[:, 0], coordinates[:, 1])
        reference = np.asarray(panel["analytic_flux"])
        retained = np.asarray(panel["solved_flux"])
        assert np.isfinite(reference).all() and np.isfinite(retained).all()
        # The limited certificate carries a large flux scale; the diverted
        # certificate includes open exterior flux below its zero separatrix.
        levels = np.linspace(0, max(reference) * 0.95, 8)
        levels_by_row.append(dict(kind=kind, cells=cells, levels=levels.tolist()))
        ax.tricontour(
            triangulation,
            reference,
            levels=levels,
            colors=REFERENCE,
            linewidths=2.4,
            linestyles="--",
            zorder=2,
        )
        ax.tricontour(
            triangulation,
            retained,
            levels=levels,
            colors=RETAINED,
            linewidths=2.0,
            zorder=3,
        )
        poloidal.draw_wall(ax, units=units)
        for name, color, roots, size in (
            ("reference", REFERENCE, panel["analytic_roots"], 9.5),
            ("retained", RETAINED, panel["solved_roots"], 6.5),
        ):
            roots = np.asarray(roots)
            before = len(ax.lines)
            counts = poloidal.draw_nulls(
                ax,
                magnetic_axis=roots[0],
                x_points=roots[1:],
                contain=units,
                style=replace(
                    DEFAULT_INK,
                    axis_color=color,
                    xpoint_color=color,
                    axis_markersize=size,
                    xpoint_markersize=size + 1,
                ),
            )
            print(kind, cells, name, counts)
            artists = ax.lines[before:]
            if artists:
                handles[name + " axis"] = artists[0]
            if len(artists) > 1:
                handles[name + " saddle"] = artists[1]
        for name, color, linestyle in [
            ("retained field", RETAINED, "-"),
            ("reference field", REFERENCE, "--"),
        ]:
            (handles[name],) = ax.plot(
                [], [], color=color, linestyle=linestyle, linewidth=2.4
            )
        ax.text(
            0.5,
            1.01,
            f"{kind.capitalize()}, {row['cells']:,} cells",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=14,
        )
        ax.set_xlim(
            wall[:, 0].min() - 0.04 * np.ptp(wall[:, 0]),
            wall[:, 0].max() + 0.04 * np.ptp(wall[:, 0]),
        )
        ax.set_ylim(
            wall[:, 1].min() - 0.04 * np.ptp(wall[:, 1]),
            wall[:, 1].max() + 0.04 * np.ptp(wall[:, 1]),
        )
    names = [
        "retained field",
        "reference field",
        "retained axis",
        "reference axis",
        "retained saddle",
        "reference saddle",
    ]
    figure.legend(
        [handles[n] for n in names],
        names,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=3,
        frameon=False,
        fontsize=13,
    )
    figure.subplots_adjust(
        left=0.02, right=0.98, bottom=0.10, top=0.96, wspace=0.06, hspace=0.12
    )
    check = runpy.run_path(
        str(Path(__file__).parents[1] / "tie-state-current-gap/render_tie_pair.py")
    )["check_text"]
    check(figure)
    figure.savefig(output, format="svg")
    output.write_text(
        "\n".join(line.rstrip() for line in output.read_text().splitlines()) + "\n"
    )
    figure.savefig(preview, dpi=100)
    assert output.stat().st_size <= 300_000
    print("LEVELS=" + json.dumps(levels_by_row))
    print("FIGURE=" + str(output) + " BYTES=" + str(output.stat().st_size))
    print("TEXT_CHECK=passed PREVIEW=" + str(preview))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview", type=Path, required=True)
    args = parser.parse_args()
    render(args.receipts, args.output, args.preview)
