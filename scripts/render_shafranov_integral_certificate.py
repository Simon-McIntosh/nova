"""Regenerate the Shafranov integral certificate receipt and its figure.

The analytic route is measured by the diagnostics contract tests; this script
is the one path that writes the tracked receipt and the served figure, so a
test run never mutates tracked evidence.  Every number the figure carries is
read back from the receipt this script writes.

Run it from the repository root:

    PYTHONPATH=. JAX_PLATFORMS=cpu uv run --no-sync \\
        scripts/render_shafranov_integral_certificate.py
"""

import importlib.util
import json
from pathlib import Path

import matplotlib
import matplotlib.text

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402

from nova.media.ink import DEFAULT_INK, trace_axes  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "docs/figures/constraint-augmented-newton-krylov/shafranov-integral"
RECEIPT = FIGURE_DIR / "receipt.json"
SVG = FIGURE_DIR / "shafranov-integral-certificate.svg"
PNG = FIGURE_DIR / "shafranov-integral-certificate.png"

#: Slide canvas: 14 in wide at an aspect near 1.94:1.
WIDTH_IN = 14.0
HEIGHT_IN = 7.2
DPI = 200


def _load_test_module():
    """Import the diagnostics contract module by path."""
    spec = importlib.util.spec_from_file_location(
        "nova_diagnostics_contract", ROOT / "tests/test_equilibrium_diagnostics.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _outside_view(artist, axes):
    """Report whether ``artist`` is a tick label for a tick outside the view.

    Matplotlib keeps every locator tick in the tree and gives it a label, so a
    tick beyond the axis limits carries a text artist that is never drawn and
    whose extent is far off the canvas.  Filtering on the tick's own location
    keeps the clip check about text that is actually painted.
    """
    for axis, limits in ((axes.xaxis, axes.get_xlim()), (axes.yaxis, axes.get_ylim())):
        for tick in axis.get_major_ticks() + axis.get_minor_ticks():
            if artist in (tick.label1, tick.label2):
                low, high = min(limits), max(limits)
                return not (low <= tick.get_loc() <= high)
    return False


def _check_artists(figure, direct_labels, series):
    """Refuse a figure whose text is clipped, overlapping, or over its data."""
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    canvas = figure.get_window_extent(renderer)
    texts = [
        artist
        for artist in figure.findobj(matplotlib.text.Text)
        if artist.axes is not None
        and artist.get_visible()
        and artist.get_text().strip()
        and not _outside_view(artist, artist.axes)
    ]
    clipped = []
    boxes = []
    for artist in texts:
        extent = artist.get_window_extent(renderer)
        boxes.append((artist.get_text(), extent))
        if (
            extent.x0 < canvas.x0
            or extent.x1 > canvas.x1
            or extent.y0 < canvas.y0
            or extent.y1 > canvas.y1
        ):
            clipped.append(artist.get_text())
    overlaps = [
        (first[0], second[0])
        for index, first in enumerate(boxes)
        for second in boxes[index + 1 :]
        if first[1].overlaps(second[1])
    ]
    covered = []
    for label in direct_labels:
        extent = label.get_window_extent(renderer)
        for line in series:
            points = line.get_path().vertices
            screen = line.axes.transData.transform(points)
            if any(
                extent.x0 <= x <= extent.x1 and extent.y0 <= y <= extent.y1
                for x, y in screen
            ):
                covered.append(label.get_text())
                break
    if clipped or overlaps or covered:
        raise SystemExit(
            f"figure refused: clipped text {clipped}, overlapping text "
            f"{overlaps}, labels over data {covered}"
        )


def _quiet_log_axis(axes):
    """Drop log minor labels and matplotlib's offset multiplier."""
    axes.xaxis.set_minor_formatter(NullFormatter())
    axes.yaxis.set_minor_formatter(NullFormatter())
    axes.xaxis.offsetText.set_visible(False)
    axes.yaxis.offsetText.set_visible(False)


def _plot(receipt):
    ink = DEFAULT_INK
    figure, (grid_axes, contour_axes) = plt.subplots(
        1, 2, figsize=(WIDTH_IN, HEIGHT_IN)
    )
    direct_labels = []
    series = []

    resolutions = receipt["grid_resolutions"]
    gaps = receipt["grid_gap"]
    trace_axes(grid_axes, ink)
    (grid_line,) = grid_axes.plot(
        resolutions,
        gaps,
        color=ink.flux_color,
        linewidth=ink.trace_linewidth,
        marker="o",
        markersize=ink.trace_markersize,
        zorder=ink.zorder_markers,
    )
    series.append(grid_line)
    grid_axes.set_yscale("log")
    _quiet_log_axis(grid_axes)
    grid_axes.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0e}"))
    grid_axes.set_xlim(resolutions[0] - 10, resolutions[-1] + 26)
    for resolution, gap in zip(resolutions[:-1], gaps[:-1], strict=True):
        direct_labels.append(
            grid_axes.annotate(
                f"{resolution:.0f}: {gap:.2e}",
                (resolution, gap),
                textcoords="offset points",
                xytext=(11, 0),
                ha="left",
                va="center",
                fontsize=ink.label_fontsize,
                color=ink.flux_color,
                zorder=ink.zorder_label,
            )
        )
    direct_labels.append(
        grid_axes.annotate(
            f"{resolutions[-1]:.0f}: {gaps[-1]:.2e}",
            (resolutions[-1], gaps[-1]),
            textcoords="offset points",
            xytext=(0, -14),
            ha="right",
            va="center",
            fontsize=ink.label_fontsize,
            color=ink.flux_color,
            zorder=ink.zorder_label,
        )
    )
    grid_axes.set_xlabel("grid nodes per axis", fontsize=ink.label_fontsize)
    grid_axes.set_ylabel("relative gap to analytic route", fontsize=ink.label_fontsize)
    grid_axes.set_xticks(resolutions)
    grid_axes.set_yticks([1.0e-6, 2.0e-6])
    grid_axes.set_title(
        f"grid ladder: gap at most {receipt['grid_gap_max']:.2e}",
        fontsize=ink.label_fontsize,
    )

    samplings = receipt["contour_samplings"]
    errors = receipt["errors"]
    limit = receipt["limit"]
    low, high = receipt["quadrature_order_range"]
    trace_axes(contour_axes, ink)
    (contour_line,) = contour_axes.plot(
        samplings,
        errors,
        color=ink.flux_color,
        linewidth=ink.trace_linewidth,
        marker="o",
        markersize=ink.trace_markersize,
        label="contour route",
        zorder=ink.zorder_markers,
    )
    series.append(contour_line)
    contour_axes.axhline(
        limit,
        color="0.5",
        linewidth=1.2,
        linestyle=":",
        label="plateau",
        zorder=ink.zorder_flux,
    )
    contour_axes.set_xscale("log")
    contour_axes.set_yscale("log")
    _quiet_log_axis(contour_axes)
    contour_axes.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{value:.2f}")
    )
    contour_axes.set_ylim(limit * 0.85, max(errors) * 1.25)
    ticks = sorted(
        {
            round(limit, 2),
            round((limit + max(errors)) / 2.0, 2),
            round(max(errors), 2),
        }
    )
    contour_axes.set_yticks(ticks)
    contour_axes.set_xlim(samplings[0] / 1.6, samplings[-1] * 1.6)
    contour_axes.set_xlabel("contour points", fontsize=ink.label_fontsize)
    contour_axes.set_ylabel(
        "relative error of beta_p + l_i/2", fontsize=ink.label_fontsize
    )
    contour_axes.set_xticks(samplings)
    contour_axes.set_xticklabels([str(value) for value in samplings])
    contour_axes.legend(frameon=False, fontsize=ink.label_fontsize, loc="upper right")
    direct_labels.append(
        contour_axes.annotate(
            f"plateau {limit:.3e}",
            (samplings[0], limit),
            textcoords="offset points",
            xytext=(0, 9),
            fontsize=ink.label_fontsize,
            color="0.35",
            ha="left",
        )
    )
    direct_labels.append(
        contour_axes.annotate(
            f"contour order {receipt['quadrature_order']:.2f} over {low}-{high}",
            (samplings[0], errors[0]),
            textcoords="offset points",
            xytext=(12, 0),
            ha="left",
            va="center",
            fontsize=ink.label_fontsize,
            color=ink.flux_color,
        )
    )

    figure.tight_layout()
    return figure, direct_labels, series


def main():
    module = _load_test_module()
    receipt = module._grid_route_ladder()
    receipt["identity"] = module._identity_ladder()
    RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n")

    figure, direct_labels, series = _plot(receipt)
    _check_artists(figure, direct_labels, series)
    figure.savefig(SVG, format="svg")
    figure.savefig(PNG, format="png", dpi=DPI)
    print(json.dumps({"receipt": str(RECEIPT), "svg": str(SVG), "png": str(PNG)}))


if __name__ == "__main__":
    main()
