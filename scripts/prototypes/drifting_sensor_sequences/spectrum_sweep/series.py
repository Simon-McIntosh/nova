"""Draw the sweep's axis error against slice, per arm, with the nova slices.

One thin trace per size arm of the reference spectrum, the white arm among
them, and the two slices nova converged drawn as markers on their own
sequences: the chart says how far each reconstruction's magnetic axis sits
from the generating state, slice by slice, and where the other code was
measured.

The traces are read through ``sweep``'s own arm definitions and ``analyse``'s
own reader, series extraction and axis distance, and each line is drawn with
:func:`nova.media.traces.draw_trace` in :class:`nova.media.ink` colours, so the
figure and the tables cannot describe different slices and no visual constant
is restated here.

Run as a script over one sweep study:

    python series.py --output <svg>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import sweep

SIZE_ARMS = (0.01, 0.02, 0.05, 0.10)


def palette(ink):
    """The five line colours, taken from the one ink style Nova draws with."""
    return (
        ink.flux_color,
        ink.flux_loop_color,
        ink.probe_color,
        ink.separatrix_color,
        ink.contour_color,
    )


def arm_rows(study: Path, analyse, sequence: str, factor: float) -> list[dict]:
    """The rows of the arm the sweep declares for this sequence and factor."""
    for arm in sweep.arms():
        if arm["sequence"] == sequence and arm["factor"] == factor:
            return analyse.load(study / ("efit-" + sweep.label(arm) + ".jsonl"))
    raise KeyError("no %s arm at factor %.4f" % (sequence, factor))


def draw(axes, study: Path, nova: Path, analyse) -> list:
    import matplotlib.lines as lines

    from nova.media.ink import DEFAULT_INK, trace_axes
    from nova.media.traces import draw_trace

    truth = json.loads((study / "generating.json").read_text())["axis"]
    trace_axes(axes, DEFAULT_INK)
    colours = palette(DEFAULT_INK)
    handles = []
    for colour, factor in zip(colours, SIZE_ARMS):
        rows = arm_rows(study, analyse, "drift", factor)
        error = analyse.axis_distance(rows, truth)
        draw_trace(
            axes,
            np.arange(error.size),
            error,
            DEFAULT_INK,
            label="size %.2f" % factor,
            color=colour,
        )
        handles.append(
            lines.Line2D(
                [],
                [],
                color=colour,
                lw=DEFAULT_INK.trace_linewidth,
                label="size %.2f" % factor,
            )
        )
    white = arm_rows(study, analyse, "white", 0.01)
    error = analyse.axis_distance(white, truth)
    draw_trace(
        axes,
        np.arange(error.size),
        error,
        DEFAULT_INK,
        label="white, size 0.01",
        color=colours[-1],
        linewidth=0.6,
        alpha=0.7,
    )
    handles.append(
        lines.Line2D([], [], color=colours[-1], lw=0.6, label="white, size 0.01")
    )
    for sequence, marker in (("white", "s"), ("drift", "o")):
        path = nova / ("nova-%s-3-0.0100.jsonl" % sequence)
        if not path.exists():
            continue
        rows = analyse.load(path)
        draw_trace(
            axes,
            np.array([row["slice"] for row in rows]),
            analyse.axis_distance(rows, truth),
            DEFAULT_INK,
            linestyle="none",
            marker=marker,
            markersize=5.0,
            markerfacecolor="none",
            markeredgecolor=DEFAULT_INK.separatrix_color,
            markeredgewidth=1.1,
        )
    for sequence, marker in (("drift", "o"), ("white", "s")):
        handles.append(
            lines.Line2D(
                [],
                [],
                linestyle="none",
                marker=marker,
                markerfacecolor="none",
                markeredgecolor=DEFAULT_INK.separatrix_color,
                label="nova, %s sequence" % sequence,
            )
        )
    axes.set_xlabel("slice", fontsize=DEFAULT_INK.label_fontsize)
    axes.set_ylabel("axis error [m]", fontsize=DEFAULT_INK.label_fontsize)
    axes.set_xlim(0, sweep.SLICES - 1)
    return handles


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", default=str(sweep.STUDY))
    parser.add_argument("--nova-source", default=str(sweep.NOVA_SOURCE))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from nova.media.ink import DEFAULT_INK

    _, analyse = sweep.import_study()
    figure, axes = plt.subplots(figsize=(6.4, 3.6))
    handles = draw(axes, Path(args.study), Path(args.nova_source), analyse)
    axes.legend(handles=handles, loc="upper left", frameon=False, fontsize=7)
    figure.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=DEFAULT_INK.figure_dpi)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
