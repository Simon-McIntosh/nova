"""Row-by-row margin of the sampled-arc cut moment route against its budget.

Every measured row carries two receipts under the run's report directory: the
row itself (``parts/<case>-<cells>.json``) with the route's moment error against
the retained fan, and its density/region discriminator with the row's other
error term, whose tenth is the budget.  This figure plots the ratio, so a row
that fails to clear its budget reads immediately as a mark below the unit line.
"""

from __future__ import annotations

import json
from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
REPORTS = Path(
    "/home/ITER/mcintos/.config/reckon/crew/reports/nova/s19-local/exact-gauss"
)
MOMENTS = ("current", "radial", "vertical")
SHORT = {
    "weak-rotation-reactor-static": "weak",
    "moderate-rotation-conventional-static": "moderate",
    "strong-rotation-compact-static": "strong",
}
CELLS = (-110, -300, -1000)


def _discriminator_path(case: str, cells: int) -> Path:
    if (case, cells) == ("weak-rotation-reactor-static", -110):
        return REPORTS / "density-region-discriminator.json"
    return REPORTS / f"density-region-discriminator-{case}-{abs(cells)}.json"


rows = []
for case in SHORT:
    for cells in CELLS:
        if not (REPORTS / "parts" / f"{case}-{abs(cells)}.json").exists():
            continue
        row = json.loads((REPORTS / "parts" / f"{case}-{abs(cells)}.json").read_text())
        discriminator = json.loads(_discriminator_path(case, cells).read_text())
        if (discriminator["case"], discriminator["requested_cells"]) != (case, cells):
            raise SystemExit(f"discriminator does not belong to {case} {cells}")
        other = discriminator["arms"][1]["moment_relative_l2_against_fan"]
        route = row["moment_relative_l2_boundary_minus_fan"]
        rows.append(
            {
                "label": f"{SHORT[case]}\n{abs(cells)}",
                "case": case,
                "cells": cells,
                "order": row["per_edge_gauss_order"],
                "evaluations": row["live_evaluations_per_cut_cell"],
                "route": np.array([route[name] for name in MOMENTS]),
                "budget": np.array([0.1 * other[name] for name in MOMENTS]),
                "exact_arm": np.array(
                    [
                        discriminator["arms"][0]["moment_relative_l2_against_fan"][name]
                        for name in MOMENTS
                    ]
                ),
            }
        )

margin = np.array([row["budget"] / row["route"] for row in rows])
position = np.arange(len(rows))
colour = {"current": "#1f4e79", "radial": "#b03030", "vertical": "#4a7c59"}
marker = {"current": "o", "radial": "s", "vertical": "^"}

figure, axis = plt.subplots(figsize=(9.2, 4.6), dpi=160)
figure.subplots_adjust(left=0.09, right=0.98, top=0.90, bottom=0.30)
for index, name in enumerate(MOMENTS):
    axis.semilogy(
        position,
        margin[:, index],
        marker=marker[name],
        color=colour[name],
        markersize=7,
        linewidth=1.0,
        label=f"{name} moment",
    )
axis.axhline(1.0, color="#333333", linewidth=1.0, linestyle="--")
axis.annotate(
    "budget: one tenth of the row's other error term",
    (0.02, 1.0),
    textcoords="offset points",
    xytext=(0, 6),
    horizontalalignment="left",
    fontsize=7.5,
)
axis.set_xticks(position, [row["label"] for row in rows])
axis.set_ylabel("budget / route error")
axis.set_ylim(0.5, 4000.0)
axis.legend(fontsize=8, loc="upper right", frameon=False, ncols=3)
for side in ("top", "right"):
    axis.spines[side].set_visible(False)

caption = (
    "Budget cleared by every measured row: the ratio of the row's budget (one "
    "tenth of its other error term) to the route's own moment error against the "
    "retained fan, on a logarithmic axis with the unit line dashed. Cells are "
    "the requested count and the three cases are the rotation branches. Every "
    "row runs 149 fixed arc edges at per-edge Gauss order "
    f"{rows[0]['order']}, {rows[0]['evaluations']} live evaluations per cut cell; "
    "the smallest margin is "
    f"{margin.min():.0f}x."
)
figure.text(
    0.5,
    0.02,
    "\n".join(textwrap.wrap(caption, 118)),
    ha="center",
    va="bottom",
    fontsize=7.5,
    color="#333333",
)
for suffix in (".svg", ".png"):
    figure.savefig(OUT / f"rows{suffix}")
print("wrote", OUT / "rows.svg")
worst_row, worst_moment = np.unravel_index(np.argmin(margin), margin.shape)
print(
    "margin min",
    float(margin.min()),
    "at",
    rows[worst_row]["label"].replace("\n", " "),
    MOMENTS[worst_moment],
)
print("exact arm max", float(max(row["exact_arm"].max() for row in rows)))
