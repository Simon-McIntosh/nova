"""Draw the titan fixture wall, cold against warm, on one device.

The certificate booking gate measures six rows, each of which needs a carrier
fixture and an exterior fixture.  On one device those fixtures cost a build the
first time and a load every time after, and because the store is addressed by an
identity computed on the device, the two arms have to come from the same card:
a build measured on one device and a load measured on another are not the same
quantity and this figure refuses to draw them together any more than it invents
a wall.

Two directories carry the cold arm rather than one.  Job 1288657 measured the
four smaller rows on titan and was cut on its two-hour wall before the two
5,000-cell rows reported; job 1288792 built those two in its own first pass on
the same card.  Both are this device, so the six cold walls pair with the six
warm walls job 1288792's second pass measured.

Usage:
    build_figure.py COLD_WIDE_DIR COLD_5000_DIR WARM_DIR OUT.svg

A row missing from any of the three directories is reported on stderr and left
out of the figure, so an incomplete gate draws short rather than drawing a wall
that was never measured.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# The cell counts naming the rows, in the order the gate parametrises them, so
# the figure, the receipts and the test log read alike.
ROW_ORDER = [
    (550, "limited"),
    (550, "diverted"),
    (2000, "limited"),
    (2000, "diverted"),
    (5000, "limited"),
    (5000, "diverted"),
]

# The rows the one directory carries and the rows the other does.  A row is in
# the wide directory when the job was not cut before reaching it.
WIDE_CELLS = (550, 2000)

# The build is the cost the cache removes, so it draws in the neutral grey the
# reader reads as background; the load is what the cache delivers and takes the
# one feature hue.  Red is reserved for a directory that fails, and no mark here
# is a failure.
BUILD_COLOR = "#929292"
LOAD_COLOR = "#2f7fa0"
CONNECT_COLOR = "#c9c9c9"

PANELS = [
    ("carrier", "carrier fixture"),
    ("exterior", "exterior fixture"),
]


def read_rows(directory: Path) -> dict[tuple[int, str], dict]:
    rows = {}
    for path in sorted(directory.glob("*.json")):
        row = json.loads(path.read_text())
        rows[(int(row["requested_cells"]), str(row["case"]))] = row
    return rows


def check_text(fig, axes) -> list[str]:
    """Return the text artists that leave the canvas or overlap another.

    A clipped or colliding label is a defect in the figure, not a caveat on it,
    so the caller refuses to write the file while this returns anything.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.bbox
    boxes = []
    for legend in fig.legends:
        for artist in legend.get_texts():
            boxes.append((artist.get_text(), artist.get_window_extent(renderer)))
    for axis in axes:
        artists = list(axis.texts)
        artists += [axis.xaxis.label, axis.yaxis.label]
        artists += axis.get_xticklabels() + axis.get_yticklabels()
        for artist in artists:
            if not artist.get_visible():
                continue
            text = artist.get_text() if hasattr(artist, "get_text") else ""
            if not text:
                continue
            try:
                box = artist.get_window_extent(renderer)
            except ValueError, RuntimeError:
                continue
            boxes.append((text, box))

    offenders = []
    for text, box in boxes:
        if (
            box.x0 < canvas.x0
            or box.y0 < canvas.y0
            or box.x1 > canvas.x1
            or box.y1 > canvas.y1
        ):
            offenders.append(
                f"clipped: {text!r} box=[{box.x0:.1f},{box.y0:.1f},"
                f"{box.x1:.1f},{box.y1:.1f}] canvas=[{canvas.x0:.1f},"
                f"{canvas.y0:.1f},{canvas.x1:.1f},{canvas.y1:.1f}]"
            )

    for index, (text, box) in enumerate(boxes):
        for other_text, other in boxes[index + 1 :]:
            if box.overlaps(other):
                offenders.append(f"overlap: {text!r} and {other_text!r}")
    return offenders


def main(argv: list[str]) -> int:
    wide_dir, large_dir, warm_dir, out = (Path(a) for a in argv[1:5])
    wide, large, warm = read_rows(wide_dir), read_rows(large_dir), read_rows(warm_dir)

    rows = []
    for key in ROW_ORDER:
        cold = wide if key[0] in WIDE_CELLS else large
        if key not in cold or key not in warm:
            print(
                f"row {key[0]}-{key[1]} omitted: cold={key in cold} warm={key in warm}",
                file=sys.stderr,
            )
            continue
        if cold[key]["carrier_cache_hit"] is not False:
            print(f"row {key[0]}-{key[1]}: the cold arm reports a hit", file=sys.stderr)
        rows.append(key)
    if not rows:
        print("no row is present in every required directory", file=sys.stderr)
        return 1

    plt.style.use("data-ink")
    figure, panels = plt.subplots(
        1,
        len(PANELS),
        figsize=(14, 0.62 * len(rows) + 2.1),
        dpi=100,
        sharey=True,
        layout="constrained",
    )
    figure.get_layout_engine().set(wspace=0.14, w_pad=1.2)

    positions = list(range(len(rows)))
    labels = [
        f"{cells} {case}\n({warm[(cells, case)]['realised_cells']} cells)" for cells, case in rows
    ]

    for axis, (prefix, title) in zip(panels, PANELS, strict=True):
        marks = []
        for index, key in enumerate(rows):
            cold = (wide if key[0] in WIDE_CELLS else large)[key]
            build = float(cold[f"{prefix}_build_seconds"])
            load = float(warm[key][f"{prefix}_load_seconds"])
            if not warm[key][f"{prefix}_cache_hit"]:
                print(
                    f"row {key[0]}-{key[1]}: the warm arm reports no {prefix} hit",
                    file=sys.stderr,
                )
            marks += [max(load, 1e-3), max(build, 1e-3)]
            axis.plot(
                [max(load, 1e-3), max(build, 1e-3)],
                [index, index],
                color=CONNECT_COLOR,
                linewidth=1.2,
                zorder=1,
            )
            axis.plot(
                [max(build, 1e-3)],
                [index],
                marker="o",
                markersize=6,
                color=BUILD_COLOR,
                linestyle="none",
                zorder=2,
            )
            axis.plot(
                [max(load, 1e-3)],
                [index],
                marker="o",
                markersize=6,
                color=LOAD_COLOR,
                linestyle="none",
                zorder=3,
            )
        axis.set_xscale("log")
        # One third of a decade of margin keeps the extreme marks clear of the
        # spines, so a decade label never lands on the panel edge.
        axis.set_xlim(min(marks) / 3.0, max(marks) * 3.0)
        axis.set_yticks(positions)
        axis.set_xlabel(f"{title} wall [s]")
        axis.set_ylim(-0.6, len(rows) - 0.4)
        axis.invert_yaxis()

    panels[0].set_yticklabels(labels)
    panels[0].set_ylabel("booking row")

    handle = [
        Line2D([], [], color=BUILD_COLOR, marker="o", linestyle="none", markersize=6),
        Line2D([], [], color=LOAD_COLOR, marker="o", linestyle="none", markersize=6),
    ]
    figure.legend(
        handle,
        ["built (cold, published)", "loaded (warm, hit)"],
        loc="outside upper center",
        ncol=2,
        frameon=False,
    )

    offenders = check_text(figure, list(panels))
    if offenders:
        for line in offenders:
            print(f"figure rejected: {line}", file=sys.stderr)
        return 2

    figure.savefig(out, format="svg")
    print(f"wrote {out} with {len(rows)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))