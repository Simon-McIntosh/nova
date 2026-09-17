"""Figure for the sampled-arc cut moment route against its row budget.

Numbers are read from the measurement receipts written by
``benchmarks/exact_clip_moment_floor.py`` under the run's report directory:
``parts/weak-rotation-reactor-static-110.json`` for the route's own error
against the fan, and ``density-region-discriminator.json`` for the row's other
error term and for the two arms that separate the arc representation from the
density model.
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
ROW = json.loads(
    (REPORTS / "parts" / "weak-rotation-reactor-static-110.json").read_text()
)
DISCRIMINATOR = json.loads((REPORTS / "density-region-discriminator.json").read_text())

MOMENTS = ("current", "radial", "vertical")
LABELS = ("current", "radial", "vertical", "frozen image")
ROUTE = np.array(
    [ROW["moment_relative_l2_boundary_minus_fan"][name] for name in MOMENTS]
)
ARMS = {
    arm["name"]: np.array(
        [arm["moment_relative_l2_against_fan"][name] for name in MOMENTS]
    )
    for arm in DISCRIMINATOR["arms"]
}
EXACT_ARM = ARMS["sampled_polygon_exact_density_boundary"]
MODEL_ARM = ARMS["sampled_polygon_quadratic_density_fan"]
#: The row's other error term is the density model's own separation from the fan,
#: so one tenth of it is the budget the route's moment error must meet.
BUDGET = 0.1 * MODEL_ARM
IMAGE_ROUTE = ROW["frozen_current_image"]["boundary_minus_fan_sup_over_span"]
IMAGE_BUDGET = 0.1 * DISCRIMINATOR["arms"][1]["frozen_image_delta_sup_over_span"]

figure, (left, right) = plt.subplots(1, 2, figsize=(12.0, 4.2), dpi=160)
figure.subplots_adjust(left=0.07, right=0.98, top=0.88, bottom=0.30, wspace=0.30)

position = np.arange(len(MOMENTS))
image_position = len(MOMENTS) + 0.6
left.semilogy(
    position,
    BUDGET,
    "_",
    color="#b03030",
    markersize=24,
    label="budget, one tenth of the row's other term",
)
left.semilogy(
    position,
    ROUTE,
    "o",
    color="#1f4e79",
    markersize=8,
    label="sampled-arc route against the fan",
)
left.semilogy([image_position], [IMAGE_ROUTE], "o", color="#1f4e79", markersize=8)
left.semilogy([image_position], [IMAGE_BUDGET], "_", color="#b03030", markersize=24)
for index, budget in enumerate(BUDGET):
    left.annotate(
        f"{budget:.1e}",
        (index, budget),
        textcoords="offset points",
        xytext=(0, 7),
        ha="center",
        fontsize=7,
    )
    left.annotate(
        f"{ROUTE[index]:.1e}",
        (index, ROUTE[index]),
        textcoords="offset points",
        xytext=(0, -14),
        ha="center",
        fontsize=7,
    )
left.annotate(
    f"{IMAGE_ROUTE:.1e}",
    (image_position, IMAGE_ROUTE),
    textcoords="offset points",
    xytext=(0, -14),
    ha="center",
    fontsize=7,
)
left.set_xticks(list(position) + [image_position], LABELS)
left.set_xlim(-0.6, image_position + 0.6)
left.set_ylim(2e-10, 4e-5)
left.set_ylabel("relative $L_2$ against the fan")
left.set_title("(a) the route meets the row budget", fontsize=9)
left.legend(fontsize=7, loc="upper left", frameon=False)
left.grid(axis="y", which="both", alpha=0.25)

FLOOR = 3e-11
right.bar(position, MODEL_ARM, 0.38, color="#c98b2e", label="fitted degree-4 model")
right.semilogy(
    position,
    np.maximum(EXACT_ARM, FLOOR),
    "o",
    color="#4a7c59",
    markersize=8,
    label="exact density instead of the model",
)
right.annotate(
    "0.0 exactly",
    (2.0, FLOOR),
    textcoords="offset points",
    xytext=(0, 8),
    ha="center",
    fontsize=7,
)
right.set_yscale("log")
right.set_xticks(position, MOMENTS)
right.set_ylim(FLOOR * 0.6, MODEL_ARM.max() * 8.0)
right.set_ylabel("relative $L_2$ against the fan")
right.set_title("(b) the arc representation contributes nothing", fontsize=9)
right.legend(fontsize=7, loc="upper left", frameon=False)
right.grid(axis="y", which="both", alpha=0.25)

caption = (
    f"Weak rotation-reactor row, {ROW['requested_cells']} cells requested and "
    f"{ROW['realised_cells']} realised, {ROW['cut_cells']} cut cells, "
    f"{ROW['fixed_edges_per_cut_cell']} fixed arc edges at per-edge Gauss order "
    f"{ROW['per_edge_gauss_order']} ({ROW['live_evaluations_per_cut_cell']} live "
    "evaluations per cut cell).  (a) The route's own error against the retained "
    "fan, against the row's other error term and one tenth of it.  (b) Supplying "
    "the exact density in place of the fitted degree-4 model drives the same "
    "route to exactly zero in all three moments, so the residual is the model's "
    "fit rather than the arc's representation."
)
figure.text(
    0.5,
    0.035,
    "\n".join(textwrap.wrap(caption, 132)),
    ha="center",
    va="bottom",
    fontsize=7.5,
    color="#333333",
)

OUT.mkdir(parents=True, exist_ok=True)
for suffix in (".svg", ".png"):
    figure.savefig(OUT / f"comparison{suffix}")
print("wrote", OUT / "comparison.svg")
print("route", ROUTE.tolist())
print("budget", BUDGET.tolist())
print("image", IMAGE_ROUTE, "image budget", IMAGE_BUDGET)
