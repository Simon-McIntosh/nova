"""The VS3 conductor and jacket cross-sections: machine description against builder.

Left panel: the machine description's two annuli, which are concentric with the
jacket outside the conductor.  Right panel: the area-equivalent squares the
builder collapsed them to, which put the jacket inside the conductor it should
surround.

Both leaders are routed so they touch only the band they name.  The two bands
are concentric rings, so a straight leader from outside the jacket ring to the
conductor band must cross the jacket ring wherever it is aimed; the conductor
caption therefore sits inside the annuli's inner hole, and both endpoints of its
leader are within radius 20.5 mm.  A segment between two points inside a disc
stays inside it, so that leader cannot reach the jacket ring at 26.5 mm at any
point along its length.  The jacket caption sits outside both rings and its
leader approaches inward, so it stops at the ring it names.

Every save runs through ``save_checked``, which refuses to write a figure whose
text is clipped by the canvas, whose text artists overlap each other, or whose
direct label sits over a filled patch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Annulus, Rectangle
from matplotlib.text import Text

CONDUCTOR = "#2f6f9f"
JACKET = "#b4553f"
CONDUCTOR_ANNULUS = (16.95, 23.00)
JACKET_ANNULUS = (26.5, 29.5)
CONDUCTOR_SIDE = 27.556
JACKET_SIDE = 22.974
LIMIT = 46.0


def panel(axis) -> None:
    axis.set_aspect("equal")
    axis.set_xlim(-LIMIT, LIMIT)
    axis.set_ylim(-LIMIT, LIMIT)
    axis.set_xlabel("r [mm]")
    axis.set_ylabel("z [mm]")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(labelsize=14)
    axis.set_xticks([-30, -20, -10, 0, 10, 20, 30])
    axis.set_yticks([-30, -20, -10, 0, 10, 20, 30])


def _text_artists(axes) -> list:
    artists = []
    for axis in axes:
        artists += list(axis.texts)
        legend = axis.get_legend()
        if legend is not None:
            artists += list(legend.get_texts())
        artists += [axis.xaxis.label, axis.yaxis.label]
        artists += list(axis.get_xticklabels()) + list(axis.get_yticklabels())
    return [artist for artist in artists if artist.get_text().strip()]


def _raster(figure):
    figure.canvas.draw()
    return np.asarray(figure.canvas.buffer_rgba()).copy()


def check_text(figure, axes) -> list[str]:
    """Clipping, text-on-text overlap and label-on-data findings, if any.

    The data test renders the figure twice: once whole, once with every text
    artist hidden.  A text artist sits on data when the second render carries a
    non-background pixel inside that text's box, which asks the question of the
    picture rather than of the patch geometry.
    """
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    artists = _text_artists(axes)
    # text only: an annotation's own get_window_extent spans its leader too,
    # and a leader that reaches the band it names is the point of the figure
    items = [(artist, Text.get_window_extent(artist, renderer)) for artist in artists]

    problems = []
    frame = figure.bbox
    tolerance = 1.0
    for artist, box in items:
        if (
            box.x0 < frame.x0 - tolerance
            or box.y0 < frame.y0 - tolerance
            or box.x1 > frame.x1 + tolerance
            or box.y1 > frame.y1 + tolerance
        ):
            problems.append("clipped: " + repr(artist.get_text()))

    for index in range(len(items)):
        for other in range(index + 1, len(items)):
            first, second = items[index][1], items[other][1]
            if (
                first.x0 < second.x1
                and second.x0 < first.x1
                and first.y0 < second.y1
                and second.y0 < first.y1
            ):
                problems.append(
                    "overlap: "
                    + repr(items[index][0].get_text())
                    + " and "
                    + repr(items[other][0].get_text())
                )

    _raster(figure)
    for artist in artists:
        artist.set_visible(False)
    bare = _raster(figure)
    for artist in artists:
        artist.set_visible(True)

    backgrounds = {tuple(int(value) for value in _raster(figure)[0, 0][:3])}
    for axis in axes:
        backgrounds.add(
            tuple(int(round(255 * value)) for value in axis.get_facecolor()[:3])
        )
    is_background = np.zeros(bare.shape[:2], dtype=bool)
    for colour in backgrounds:
        is_background |= np.all(bare[:, :, :3] == np.array(colour), axis=2)
    height = bare.shape[0]
    for artist, box in items:
        top = max(0, int(height - box.y1))
        bottom = min(height, int(height - box.y0) + 1)
        left = max(0, int(box.x0))
        right = min(bare.shape[1], int(box.x1) + 1)
        if top >= bottom or left >= right:
            continue
        if not is_background[top:bottom, left:right].all():
            problems.append("over data: " + repr(artist.get_text()))
    return problems


def save_checked(figure, axes, destination: Path) -> None:
    problems = check_text(figure, axes)
    if problems:
        for problem in problems:
            print("CHECK " + problem, file=sys.stderr)
        raise SystemExit("refusing to write " + str(destination))
    figure.savefig(destination, format="svg")
    print("wrote", destination)


def main() -> None:
    destination = Path(sys.argv[1])
    plt.rcParams.update({"font.size": 14, "axes.labelsize": 16})
    figure, axes = plt.subplots(1, 2, figsize=(14, 6.4))
    for axis in axes:
        panel(axis)

    left = axes[0]
    left.add_patch(
        Annulus(
            (0, 0),
            CONDUCTOR_ANNULUS[1],
            CONDUCTOR_ANNULUS[1] - CONDUCTOR_ANNULUS[0],
            color=CONDUCTOR,
            lw=0,
        )
    )
    left.add_patch(
        Annulus(
            (0, 0),
            JACKET_ANNULUS[1],
            JACKET_ANNULUS[1] - JACKET_ANNULUS[0],
            color=JACKET,
            lw=0,
        )
    )
    left.annotate(
        "jacket annulus\n26.5–29.5 mm",
        xy=(19.8, 19.8),
        xytext=(30.0, 40.0),
        color=JACKET,
        va="top",
        ha="left",
        arrowprops=dict(arrowstyle="-", color=JACKET, lw=1.2, shrinkA=2, shrinkB=2),
    )
    left.annotate(
        "conductor\n16.95–23.0 mm",
        xy=(14.5, 14.5),
        xytext=(0.0, 8.0),
        color=CONDUCTOR,
        va="top",
        ha="center",
        arrowprops=dict(arrowstyle="-", color=CONDUCTOR, lw=1.2, shrinkA=2, shrinkB=2),
    )

    right = axes[1]
    for side, colour in ((CONDUCTOR_SIDE, CONDUCTOR), (JACKET_SIDE, JACKET)):
        right.add_patch(
            Rectangle(
                (-side / 2, -side / 2),
                side,
                side,
                fill=False,
                edgecolor=colour,
                lw=2.4,
                zorder=3,
            )
        )
    right.text(
        0,
        26.0,
        "conductor 27.56 mm square",
        color=CONDUCTOR,
        fontsize=14,
        ha="center",
        va="bottom",
    )
    right.text(
        0,
        19.0,
        "jacket 22.97 mm square",
        color=JACKET,
        fontsize=14,
        ha="center",
        va="bottom",
    )
    right.annotate(
        "the jacket square lies inside\nthe conductor square it should surround",
        xy=(11.5, 0),
        xytext=(6.0, -33.0),
        color=JACKET,
        fontsize=14,
        ha="left",
        va="top",
        arrowprops=dict(arrowstyle="->", color=JACKET, lw=1.2, shrinkA=2, shrinkB=2),
    )

    figure.tight_layout()
    save_checked(figure, axes, destination)


if __name__ == "__main__":
    main()
