"""Draw the sweep's axis error against slice, per arm, with the nova slices.

One thin trace per size arm of the reference spectrum, the white arm among
them, and the two slices nova converged drawn as markers on their own
sequences: the chart says how far each reconstruction's magnetic axis sits
from the generating state, slice by slice, and where the other code was
measured.  The style is :func:`nova.media.ink.trace_axes`, so the panel is
despined and never gridded.

Run as a script over one sweep study:

    python series.py --output <svg> --raster <png>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

STUDY = Path(__file__).resolve().parents[1] / "probe_rows"
SIZE_ARMS = (0.01, 0.02, 0.05, 0.10)
INK = ("#3366cc", "#1f5fa8", "#c06a1b", "#cc0000", "#333333")


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def axis_error(rows: list[dict], truth: list[float]) -> np.ndarray:
    centre = np.array(truth, dtype=float)
    return np.array(
        [
            float(np.linalg.norm(np.array(row["axis"], dtype=float) - centre))
            if row.get("axis") is not None
            else float("nan")
            for row in rows
        ]
    )


def draw(axes, study: Path, nova: Path) -> list:
    import matplotlib.lines as lines

    from nova.media.ink import DEFAULT_INK, trace_axes

    truth = json.loads((study / "generating.json").read_text())["axis"]
    trace_axes(axes, DEFAULT_INK)
    handles = []
    for colour, factor in zip(INK, SIZE_ARMS):
        path = study / ("efit-drift-100-%.4f-index1.0-corner2.0000.jsonl" % factor)
        rows = load(path)
        error = axis_error(rows, truth)
        axes.plot(
            np.arange(error.size),
            error,
            color=colour,
            linewidth=0.9,
        )
        handles.append(
            lines.Line2D([], [], color=colour, lw=0.9, label="size %.2f" % factor)
        )
    white = study / "efit-white-100-0.0100.jsonl"
    rows = load(white)
    error = axis_error(rows, truth)
    axes.plot(np.arange(error.size), error, color=INK[-1], linewidth=0.6, alpha=0.7)
    handles.append(
        lines.Line2D([], [], color=INK[-1], lw=0.6, label="white, size 0.01")
    )
    for index, sequence in enumerate(("white", "drift")):
        path = nova / ("nova-%s-3-0.0100.jsonl" % sequence)
        if not path.exists():
            continue
        rows = load(path)
        slices = np.array([row["slice"] for row in rows])
        error = axis_error(rows, truth)
        axes.plot(
            slices,
            error,
            linestyle="none",
            marker="o" if sequence == "drift" else "s",
            markersize=5.0,
            markerfacecolor="none",
            markeredgecolor=INK[3],
            markeredgewidth=1.1,
        )
    handles.append(
        lines.Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markerfacecolor="none",
            markeredgecolor=INK[3],
            label="nova, drift sequence",
        )
    )
    handles.append(
        lines.Line2D(
            [],
            [],
            linestyle="none",
            marker="s",
            markerfacecolor="none",
            markeredgecolor=INK[3],
            label="nova, white sequence",
        )
    )
    axes.set_xlabel("slice", fontsize=DEFAULT_INK.label_fontsize)
    axes.set_ylabel("axis error [m]", fontsize=DEFAULT_INK.label_fontsize)
    axes.set_xlim(0, 99)
    return handles


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", default=str(STUDY))
    parser.add_argument(
        "--nova-source",
        default="/home/ITER/mcintos/.config/reckon/crew/runs"
        "/r-20261119T065220776269-drifting-sensor-probe-rows/study",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--raster", default="")
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(6.4, 3.6))
    handles = draw(axes, Path(args.study), Path(args.nova_source))
    axes.legend(handles=handles, loc="upper left", frameon=False, fontsize=7)
    figure.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200)
    if args.raster:
        figure.savefig(Path(args.raster), dpi=200)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
