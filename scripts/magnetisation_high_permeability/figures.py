"""Plot the shell receipt with the established iron-study visual roles."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from nova.media.ink import trace_axes  # noqa: E402

SUBJECT = "#7b4fb3"
MODEL = "#222222"
NEUTRAL = "#929292"


def checked_save(fig, svg, preview):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = list(fig.legends[0].get_texts())
    for ax in fig.axes:
        texts.extend([ax.xaxis.label, ax.yaxis.label, *ax.texts])
        for axis in (ax.xaxis, ax.yaxis):
            lo, hi = sorted(axis.get_view_interval())
            for tick in axis.get_major_ticks():
                if lo <= tick.get_loc() <= hi:
                    texts.extend([tick.label1, tick.label2])
    texts = [t for t in texts if t.get_visible() and t.get_text().strip()]
    problems = []
    for text in texts:
        box = text.get_window_extent(renderer)
        if not fig.bbox.contains(box.x0, box.y0) or not fig.bbox.contains(
            box.x1, box.y1
        ):
            problems.append(f"clipped {text.get_text()!r}")
    for first, second in itertools.combinations(texts, 2):
        if first.get_window_extent(renderer).overlaps(
            second.get_window_extent(renderer)
        ):
            problems.append(f"overlap {first.get_text()!r}/{second.get_text()!r}")
    for ax in fig.axes:
        for text in ax.texts:
            box = text.get_window_extent(renderer).expanded(1.04, 1.1)
            for line in ax.lines:
                path = line.get_transform().transform_path(line.get_path())
                if path.intersects_bbox(box, filled=False):
                    problems.append(f"label over data {text.get_text()!r}")
    if problems:
        raise ValueError("; ".join(problems))
    svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(svg)
    contents = svg.read_text()
    contents = re.sub(
        r'width="[\d.]+pt" height="[\d.]+pt"',
        'width="31.1111cm" height="16cm"',
        contents,
        count=1,
    )
    svg.write_text(contents)
    fig.savefig(preview, dpi=100)
    print(f"FIGURE checks=passed svg={svg} preview={preview}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--svg", type=Path, required=True)
    parser.add_argument("--preview", type=Path, required=True)
    args = parser.parse_args()
    record = json.loads(args.receipt.read_text())
    assert record["summary"]["controls_pass"]
    plt.style.use(Path(__file__).with_name("data-ink.mplstyle"))
    fig, axes = plt.subplots(2, 2, figsize=(14, 7.2), dpi=100)
    fig.subplots_adjust(
        left=0.09, right=0.98, bottom=0.12, top=0.88, wspace=0.23, hspace=0.35
    )
    for ax, mu in zip(axes.flat, [100, 300, 1000, 3000]):
        trace_axes(ax)
        for source, color, dash, marker, label, zorder in [
            ("dipole", MODEL, "--", "x", "ideal dipole", 2),
            ("loop", SUBJECT, "-", "o", "finite loop", 3),
        ]:
            rows = [
                r for r in record["shell"] if r["mu"] == mu and r["source"] == source
            ]
            ax.loglog(
                [r["h"] * 1000 for r in rows],
                [100 * max(np.abs(r["signed_error"])) for r in rows],
                color=color,
                linestyle=dash,
                marker=marker,
                linewidth=3.0 if source == "loop" else 2.6,
                markersize=6,
                label=label,
                zorder=zorder,
            )
        exact = record["shell"][[r["mu"] for r in record["shell"]].index(mu)]["exact"]
        prediction = []
        for geometry in record["geometry"]:
            fraction = 1 - geometry["volume"] / (4 * np.pi / 3)
            implied = ((2 * mu + 1) * (mu + 2) - 2 * (mu - 1) ** 2 * fraction) / (
                9 * mu
            )
            prediction.append(100 * abs(implied / exact - 1))
        ax.loglog(
            [g["h"] * 1000 for g in record["geometry"]],
            prediction,
            color=NEUTRAL,
            linestyle=":",
            linewidth=1.2,
            label="volume error (outer radius held)",
            zorder=1,
        )
        ax.set(
            xlim=(8, 50),
            ylim=(1e-4, 100),
            xlabel="cell size [mm]",
            ylabel="largest shielding error [%]",
        )
        ax.set_xticks([10, 20, 40], ["10", "20", "40"])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.tick_params(labelsize=14, width=1.2)
        for side in ("left", "bottom"):
            ax.spines[side].set_linewidth(1.2)
        ax.text(0.05, 0.82, rf"$\mu_r = {mu}$", transform=ax.transAxes, fontsize=14)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.0),
        fontsize=14,
    )
    checked_save(fig, args.svg, args.preview)


if __name__ == "__main__":
    main()
