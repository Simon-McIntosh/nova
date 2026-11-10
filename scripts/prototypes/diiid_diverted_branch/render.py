"""Draw shared-level paired fields with lower-saddle distance endpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/prototypes/diiid_warmup_admission"))
import render as painter  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402


original_nulls = painter.draw_nulls


def draw_nulls(ax, **kwargs):
    start = len(ax.lines)
    original_nulls(ax, **kwargs)
    if kwargs["style"].xpoint_color == painter.FLUX:
        for line in ax.lines[start:]:
            line.set_marker("o" if line.get_marker() == "^" else "+")


def paint(ax, *args, **kwargs):
    painter.paint(ax, *args, **kwargs)
    for contour in ax.collections:
        width = float(contour.get_linewidths()[0])
        contour.set_linewidth(3.0 if width > 1.5 else 2.6)


def render(source, output, previews, frame):
    records = json.loads((source / "comparison.json").read_text())
    name = f"frame-{frame}-diverted"
    rows = {r["arm"]: r for r in records if r["case"] == name}
    pair = [painter.load(Path(rows[arm]["source_path"])) for arm in ("base", "head")]
    reference = painter.reference_nulls(pair[0][0])
    points = [reference[1]] + [
        np.asarray(record["terminal_topology"]["lower_saddle_rz_m"])
        for _data, record in pair
        if record["terminal_topology"]["lower_saddle_qualified"]
    ]
    points = np.asarray(points)
    lower = points.min(axis=0) - 0.08
    upper = points.max(axis=0) + 0.08
    span = max(float(np.max(upper - lower)), 0.3)
    center = (lower + upper) / 2
    levels = np.unique(
        np.r_[
            np.linspace(
                min(float(d[k].min()) for d, _r in pair for k in ("flux", "label")),
                max(float(d[k].max()) for d, _r in pair for k in ("flux", "label")),
                11,
            )[1:-1],
            [float(d[k][2]) for d, _r in pair for k in ("boundary", "saddle")],
            [
                float(d["continued_saddle"][2])
                for d, _r in pair
                if "continued_saddle" in d
            ],
        ]
    )
    fig = plt.figure(figsize=(14, 7.2), dpi=100)
    axes = [
        fig.add_axes(rect)
        for rect in (
            (0.02, 0.17, 0.28, 0.59),
            (0.31, 0.25, 0.16, 0.38),
            (0.52, 0.17, 0.28, 0.59),
            (0.81, 0.25, 0.16, 0.38),
        )
    ]
    notes = []
    for index, (arm, (data, record)) in enumerate(
        zip(("base", "head"), pair, strict=True)
    ):
        main, inset = axes[2 * index : 2 * index + 2]
        paint(main, data, levels, reference)
        paint(
            inset,
            data,
            levels,
            reference,
            xlim=(center[0] - span / 2, center[0] + span / 2),
            ylim=(center[1] - span / 2, center[1] + span / 2),
        )
        topology = record["terminal_topology"]
        distance = topology["lower_saddle_distance_m"]
        gap = (
            f"lower saddle gap {100 * distance:.2f} cm"
            if distance is not None
            else "lower saddle unresolved"
        )
        for x, y, text in (
            (0.25, 0.81, arm),
            (0.39, 0.68, "lower X-point zoom"),
            (
                0.25,
                0.12,
                f"support {topology['support']}; {topology['boundary_class']} boundary",
            ),
            (
                0.25,
                0.075,
                f"residual {record['residual']:.5g}; "
                f"converged {str(record['converged']).lower()}",
            ),
            (0.25, 0.025, gap),
        ):
            notes.append(fig.text(x + 0.5 * index, y, text, ha="center", fontsize=14))
    handles = [
        Line2D([], [], color=painter.POSITIVE, lw=2.6, label="solved flux"),
        Line2D([], [], color=painter.FLUX, ls="--", lw=2.6, label="EFIT flux"),
        Line2D([], [], color=painter.FLUX, ls=":", lw=2.4, label="EFIT boundary"),
        Line2D([], [], color=painter.DEFAULT_INK.wall_color, lw=2.4, label="limiter"),
        Line2D(
            [], [], color=painter.POSITIVE, marker="^", ls="none", label="solved axis"
        ),
        Line2D(
            [],
            [],
            color=painter.POSITIVE,
            marker="X",
            ls="none",
            label="solved saddles",
        ),
        Line2D(
            [],
            [],
            color=painter.FLUX,
            marker="o",
            mfc="none",
            ls="none",
            label="EFIT axis",
        ),
        Line2D([], [], color=painter.FLUX, marker="+", ls="none", label="EFIT X-point"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=4,
        frameon=False,
        fontsize=13,
    )
    painter.shared.save(
        fig, axes, notes, output / (name + ".svg"), previews / (name + ".png")
    )
    print(
        json.dumps(
            dict(
                machine="DIII-D",
                frame=frame,
                levels_wb=levels.tolist(),
                lower_zoom_endpoints=points.tolist(),
            )
        ),
        flush=True,
    )
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previews", type=Path, required=True)
    args = parser.parse_args()
    painter.configure_dtypes()
    painter.validate_palette()
    painter.draw_nulls = draw_nulls
    plt.rcParams["svg.fonttype"] = "none"
    for frame in (137, 146):
        render(args.input, args.output, args.previews, frame)


if __name__ == "__main__":
    main()
