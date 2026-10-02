"""Re-render the axis-configuration, keyframes-throughput and compensator figures.

Each panel is drawn from its committed receipt. The axis-attribution panels read
axis-configuration-receipt.json and draw the machine wall it carries; the
throughput panel reads h200-press-stage-receipt.json and plots the measured
per-press rate against the two human-response fences; the compensator panel reads
compensator-authority.json and draws the early and flat-top measured row groups
as two separate segments, with no line across the seventy-seven rows that were
never measured.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np

from nova.media import poloidal
from nova.media.ink import poloidal_axes, trace_axes

LABELLER = "#3366cc"
PARITY = "#cc6633"
REFERENCE = "#666666"
REFUSED = "#996a1f"

ARM_FAMILY = {"L": LABELLER, "S": LABELLER, "R": LABELLER, "P": PARITY, "W": PARITY}


def save(figure, path):
    figure.savefig(path, dpi=170)
    plt.close(figure)
    return path


def render_axis_configuration(root):
    axis_dir = root / "docs/figures/playable-forward-solve/axis-configuration"
    receipt = json.loads((axis_dir / "axis-configuration-receipt.json").read_text())
    wall = np.asarray(receipt["machine_wall_rz_m"], dtype=float)
    efit = np.asarray(receipt["efit_axis_rz_m"], dtype=float)
    arms = receipt["arms"]

    figure, panels = plt.subplots(1, 2, figsize=(13.0, 5.4))
    left, right = panels

    poloidal_axes(left)
    poloidal.draw_wall(left, wall[:, 0], wall[:, 1], color="#202020")
    for arm in arms:
        position = np.asarray(arm["axis_rz_m"], dtype=float)
        left.plot(
            position[0],
            position[1],
            marker="o",
            markersize=8,
            markerfacecolor="none",
            markeredgecolor=ARM_FAMILY[arm["arm"]],
            markeredgewidth=1.6,
            linestyle="none",
            zorder=9,
        )
    poloidal.draw_nulls(left, magnetic_axis=efit, contain=wall)
    left.annotate(
        "L, S, R (labeller config)",
        (0.8648, -0.0148),
        xytext=(16, -34),
        textcoords="offset points",
        color=LABELLER,
        fontsize=11,
        ha="left",
        arrowprops={"arrowstyle": "->", "color": LABELLER, "lw": 1.0},
    )
    left.annotate(
        "P, W (wide-grid parity)",
        (0.9061, 0.0187),
        xytext=(-14, 44),
        textcoords="offset points",
        color=PARITY,
        fontsize=11,
        ha="right",
        arrowprops={"arrowstyle": "->", "color": PARITY, "lw": 1.0},
    )
    left.annotate(
        "EFIT axis",
        (efit[0], efit[1]),
        xytext=(-70, 22),
        textcoords="offset points",
        color="#202020",
        fontsize=11,
        ha="center",
        arrowprops={"arrowstyle": "->", "color": "#202020", "lw": 1.0},
    )
    left.set_xlim(float(wall[:, 0].min()) - 0.05, float(wall[:, 0].max()) + 0.05)
    left.set_ylim(float(wall[:, 1].min()) - 0.05, float(wall[:, 1].max()) + 0.05)
    left.set_title(
        "MAST %d, row %d, t = %.3f s - magnetic axis per arm"
        % (receipt["shot"], receipt["slice_index"], receipt["time_s"]),
        fontsize=11,
    )

    trace_axes(right)
    codes = [arm["arm"] for arm in arms]
    offsets = [arm["offset_from_efit_cm"] for arm in arms]
    positions = np.arange(len(codes))
    right.bar(positions, offsets, width=0.62, color=[ARM_FAMILY[c] for c in codes])
    for position, offset in zip(positions, offsets):
        right.annotate(
            "%+.2f cm" % offset,
            (position, offset),
            xytext=(0, 6 if offset >= 0.0 else -14),
            textcoords="offset points",
            ha="center",
            fontsize=11,
            color="#202020" if abs(offset) >= 0.5 else REFUSED,
        )
    right.axhline(0.0, color=REFERENCE, linewidth=1.2, linestyle=":")
    right.set_xticks(positions, codes)
    right.set_ylabel("axis R offset from EFIT [cm]")
    right.set_title(
        "Axis R displacement from EFIT by single-factor arm (P and W are 0.02 cm)",
        fontsize=11,
    )

    figure.tight_layout()
    return save(figure, axis_dir / "axis-configuration.png")


def render_keyframes_throughput(root):
    key_dir = root / "docs/figures/playable-forward-solve/keyframes"
    receipt = json.loads((key_dir / "h200-press-stage-receipt.json").read_text())
    presses = receipt["presses"]
    labels = []
    for press in presses:
        if press.get("press") is None:
            labels.append("prime (cold)")
        else:
            labels.append("first moved (warm)")
    index = [press["index"] for press in presses]
    rate = [1.0 / press["wall"] for press in presses]

    figure, axes = plt.subplots(figsize=(9.0, 5.2))
    trace_axes(axes)
    axes.plot(index, rate, marker="o", markersize=9, color=LABELLER, linewidth=2.6)
    for x, y, label in zip(index, rate, labels):
        axes.annotate(
            "%s\n%.4f" % (label, y),
            (x, y),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=11,
            color=LABELLER,
        )
    axes.axhline(10.0, color=REFERENCE, linewidth=1.2, linestyle="dotted")
    axes.axhline(1.0, color=REFERENCE, linewidth=1.2, linestyle="dotted")
    axes.text(
        index[-1],
        10.0,
        "10 keyframes/s warm fence",
        color=REFERENCE,
        fontsize=10,
        va="bottom",
        ha="right",
    )
    axes.set_yscale("log")
    axes.set_xticks(index, ["prime (cold)", "first moved (warm)"])
    axes.set_ylim(1e-3, 30.0)
    axes.set_ylabel("keyframes / second   (1 / press wall)")
    axes.set_title(
        "H200 measured keyframe throughput, one press per state "
        "(%d warm press measured; no chain throughput)"
        % receipt["verdict"]["presses_measured"],
        fontsize=11,
    )
    return save(figure, key_dir / "h200-press-stage-throughput.png")


def render_compensator_authority(root):
    comp_dir = root / "docs/figures/playable-forward-solve/compensator-authority"
    receipt = json.loads((comp_dir / "compensator-authority.json").read_text())
    rows = {}
    for row in receipt["rows"]:
        rows[row["row"]] = row
    target = receipt["authority_target_m_per_a"]

    figure, axes = plt.subplots(figsize=(10.0, 5.4))
    trace_axes(axes)
    groups = [
        (receipt["measured_early_rows"], LABELLER),
        (receipt["flat_top_rows"], PARITY),
    ]
    for group, colour in groups:
        xs = list(group)
        ys = [abs(rows[row]["selected_derivative_m_per_a"]) for row in group]
        axes.plot(xs, ys, marker="o", markersize=8, color=colour, linewidth=2.4)
        for x, y in zip(xs, ys):
            axes.annotate(
                "%.2e" % y,
                (x, y),
                xytext=(0, 9),
                textcoords="offset points",
                ha="center",
                fontsize=10,
                color=colour,
            )
    axes.axhline(target, color=REFERENCE, linewidth=1.2, linestyle="dotted")
    axes.text(
        receipt["flat_top_rows"][-1],
        target,
        "authority target 1e-5 m/A",
        color=REFERENCE,
        fontsize=10,
        va="top",
        ha="right",
    )
    row_96 = rows[receipt["flat_top_rows"][1]]
    axes.annotate(
        "row 96 selects %s" % row_96["selected_circuit"],
        (96, abs(row_96["selected_derivative_m_per_a"])),
        xytext=(52, 1.6e-6),
        fontsize=10,
        color=PARITY,
        arrowprops={"arrowstyle": "->", "color": PARITY, "lw": 1.1},
    )
    axes.set_yscale("log")
    axes.set_xlabel("MAST 27079 EFM row (measured groups only; no rows 18-94)")
    axes.set_ylabel("| dz_centroid / dI |   [m A$^{-1}$]")
    axes.set_title(
        "Centroid compensator authority, two measured row groups "
        "(selection rule picks the strongest drivable circuit every row)",
        fontsize=11,
    )
    return save(figure, comp_dir / "compensator-authority.png")


if __name__ == "__main__":
    root = Path(sys.argv[1]).resolve()
    print("wrote", render_axis_configuration(root).relative_to(root))
    print("wrote", render_keyframes_throughput(root).relative_to(root))
    print("wrote", render_compensator_authority(root).relative_to(root))