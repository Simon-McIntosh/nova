"""Draw the iron's share of the WEST 62751 stray field and diagnostics.

Reads the uniform-80 mm ``field.npz`` written by the ``measure`` stage
and draws three one-dimensional panels in the imas-ink style: the outboard
stray field on the three sampling planes with and without the iron, the iron's
relative contribution to it, and the iron contribution at every in-vessel
poloidal probe.

Usage: figure.py <output.png>
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from nova.media.ink import DEFAULT_INK, trace_axes  # noqa: E402

DEFAULT_FIELD = Path(
    "/work/projects/imas_gpu/nova/evidence/iron-core-magnetisation/"
    "field-source/west-62751-uniform-0.08-field.npz"
)
COLORS = ("#3366cc", "#cc7722", "#009988")
# The three planes converge in the far field, so each iron curve takes its own
# dash pattern and a fixed draw order: a reader sees all three where they
# coincide rather than only the last one drawn.
IRON_DASHES = ((None, None), (6.0, 1.6), (1.5, 1.4))


def draw(axes, data):
    style = DEFAULT_INK
    radius = data["radius"]
    edge = float(data["edge"])
    distance = radius - edge
    names = [str(name) for name in data["line_names"]]
    air = data["bmag_air"]
    iron = data["bmag_iron"]
    iron_handles = []
    air_handle = None
    upper = axes[0]
    trace_axes(upper, style)
    for index, name in enumerate(names):
        (line_air,) = upper.semilogy(
            distance,
            air[index] * 1e3,
            color=style.contour_color,
            linewidth=0.8,
            linestyle=(0, (4.0, 2.0)) if index == 0 else (0, (1.0, 2.0)),
            zorder=2,
            label="air-core (all three planes)" if index == 0 else "_air",
        )
        if air_handle is None:
            air_handle = line_air
        (line_iron,) = upper.semilogy(
            distance,
            iron[index] * 1e3,
            color=COLORS[index],
            linewidth=style.trace_linewidth,
            dashes=IRON_DASHES[index],
            zorder=4,
            label=name,
        )
        iron_handles.append(line_iron)
    upper.set_ylabel(r"$|B|$ [mT]", fontsize=style.label_fontsize)
    upper.set_xlabel(
        r"$R - R_{\mathrm{edge}}$ [m] (symlog)", fontsize=style.label_fontsize
    )
    # A symmetric log x axis gives the near-edge spike at 0.4 m real width
    # instead of the sliver a linear axis leaves it.
    upper.set_xscale("symlog", linthresh=0.1)
    upper.set_xlim(0, float(distance[-1]))

    middle = axes[1]
    trace_axes(middle)
    share = (iron - air) / air * 100
    # One quadrature point beside the edge carries a few-thousand-percent share,
    # and on a linear axis it flattens the -96 % share held further out into a
    # line indistinguishable from zero - the reader then reads the opposite sign.
    # A symmetric log axis keeps both the outlier and the plateau legible, and the
    # zero rule is drawn under the curves so it cannot hide them.
    middle.axhline(0.0, color="#888888", linewidth=0.6, zorder=0)
    for index in range(len(names)):
        middle.plot(
            distance,
            share[index],
            color=COLORS[index],
            linewidth=style.trace_linewidth,
            dashes=IRON_DASHES[index],
            zorder=3,
        )
    middle.set_yscale("symlog", linthresh=1.0)
    middle.set_xscale("symlog", linthresh=0.1)
    peak = float(np.max(share))
    middle.annotate(
        f"peak {peak:+.0f} %",
        xy=(float(distance[int(np.argmax(share))]), peak),
        xytext=(6.0, peak),
        fontsize=style.label_fontsize - 1,
        color="#444444",
        va="center",
    )
    middle.set_ylim(-1.0e3, 1.0e4)
    middle.set_xlim(0, float(distance[-1]))
    middle.set_ylabel("iron share of $|B|$ [%] (symlog)", fontsize=style.label_fontsize)
    middle.set_xlabel(
        r"$R - R_{\mathrm{edge}}$ [m] (symlog)", fontsize=style.label_fontsize
    )

    lower = axes[2]
    trace_axes(lower)
    probes = data["probes"]
    contribution = np.hypot(*data["frame_probe"].T) * 1e3
    air_probe = np.hypot(*data["direct_probe"].T) * 1e3
    limits = np.maximum(air_probe, 1e-12)
    (air_marks,) = lower.semilogy(
        probes[:, 0],
        limits,
        "_",
        color=style.contour_color,
        markersize=style.trace_markersize,
    )
    lower.semilogy(
        probes[:, 0],
        contribution,
        "o",
        color=COLORS[0],
        markersize=style.trace_markersize,
    )
    lower.set_xlabel("probe $R$ [m]", fontsize=style.label_fontsize)
    lower.set_ylabel(
        r"$|\Delta B|$ (circles) and $|B_{\mathrm{air}}|$ (dashes) [mT]",
        fontsize=style.label_fontsize,
    )

    figure = upper.figure
    handles = iron_handles + [air_handle]
    labels = names + ["air-core (all three planes)"]
    figure.legend(
        handles,
        labels,
        fontsize=style.label_fontsize,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
    )
    return style


def main():
    output = Path(sys.argv[1])
    field = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_FIELD
    output.parent.mkdir(parents=True, exist_ok=True)
    data = np.load(field)
    figure, axes = plt.subplots(
        3, 1, figsize=(6.4, 7.2), height_ratios=(2.0, 1.0, 1.2), constrained_layout=True
    )
    draw(axes, data)
    figure.savefig(output, dpi=DEFAULT_INK.figure_dpi)
    print(output, round(output.stat().st_size / 1e3, 1), "kB")


if __name__ == "__main__":
    main()
