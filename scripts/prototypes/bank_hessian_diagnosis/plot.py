"""Plot the physical trace test with shared scales and checked text extents."""

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
import numpy as np

from nova.media.ink import DEFAULT_INK
from scripts.prototypes.bank_hessian_diagnosis.measure import ROOT


def save(figure, path):
    """Refuse clipped or colliding text before writing the slide-sized canvas."""
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    text = []
    for artist in figure.findobj(Text):
        if artist.get_visible() and artist.get_text().strip():
            box = artist.get_window_extent(renderer)
            if box.width > 0 and box.height > 0:
                if not figure.bbox.contains(box.x0, box.y0) or not figure.bbox.contains(
                    box.x1, box.y1
                ):
                    raise ValueError(f"clipped text: {artist.get_text()}")
                text.append((artist.get_text(), box))
    for i, (label, box) in enumerate(text):
        for other, other_box in text[i + 1 :]:
            if box.overlaps(other_box):
                raise ValueError(f"text overlap: {label} / {other}")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, format="svg", metadata={"Date": None})
    tree = ET.parse(path)
    tree.getroot().set("width", "31.1cm")
    tree.getroot().set("height", "16cm")
    tree.write(path, encoding="utf-8", xml_declaration=True)
    preview = ROOT / "trace-preview.png"
    figure.savefig(preview, dpi=100)
    print("FIGURE_TEXT_CHECK_PASS", len(text), "artists", flush=True)
    print("FIGURE_PREVIEW", preview, flush=True)


def main():
    plt.style.use("data-ink")
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "axes.linewidth": 1.2,
        }
    )
    receipt = json.loads((ROOT / "receipt.json").read_text())
    arrays = dict(np.load(ROOT / "comparison.npz"))
    figure, axes = plt.subplots(1, 3, figsize=(14, 7.2), sharex=True, sharey=True)
    figure.subplots_adjust(left=0.09, right=0.98, bottom=0.15, top=0.82, wspace=0.12)
    grey, failed = "#929292", DEFAULT_INK.separatrix_color
    all_values = []
    for ax, row in zip(axes, receipt["states"], strict=True):
        for name, marker in (("reference", "o"), ("bank", "x")):
            values = (
                abs(np.array(row["trace"][name]["residual"]))
                / row["trace"][name]["scale"]
            )
            assert np.all(values > 0), "a zero trace requires explicit treatment"
            all_values.extend(values)
            if name == "reference":
                ax.scatter(
                    arrays["points"][:, 0],
                    values,
                    marker=marker,
                    s=28,
                    facecolors="none",
                    edgecolors=grey,
                    linewidths=1.2,
                    zorder=2,
                )
            else:
                ax.scatter(
                    arrays["points"][:, 0],
                    values,
                    marker=marker,
                    s=34,
                    c=np.where(values > 1e-8, failed, grey),
                    linewidths=1.4,
                    zorder=3,
                )
        ax.axhline(1e-8, color=grey, linestyle=":", linewidth=1.2, zorder=1)
        ax.set_yscale("log")
        ax.set_xlabel("radius [m]")
        ax.text(0.5, 1.04, f"state {row['state']}", ha="center", transform=ax.transAxes)
        ax.spines[["right", "top"]].set_visible(False)
        ax.grid(False)
    axes[0].set_ylabel("scaled absolute trace residual")
    axes[0].set_ylim(min(all_values) / 3, max(all_values) * 3)
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            color=grey,
            markerfacecolor="none",
            linestyle="none",
            label="source stencil",
        ),
        Line2D(
            [],
            [],
            marker="x",
            color=failed,
            linestyle="none",
            label="bank above budget",
        ),
        Line2D([], [], color=grey, linestyle=":", linewidth=1.2, label="trace budget"),
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.54, 0.98),
    )
    path = Path(
        "docs/figures/sub-millisecond-forward-path/bank-hessian-diagnosis/trace.svg"
    )
    save(figure, path)


if __name__ == "__main__":
    main()
