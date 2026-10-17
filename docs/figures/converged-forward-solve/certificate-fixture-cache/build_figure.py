"""Draw the certificate fixture build-versus-load figure from the gate receipts.

Two sequential runs of the certificate booking gate write one JSON row per
parametrisation into separate directories. The first run is cold: it builds each
fixture and publishes it. The second run is warm: it loads the object the first
run published. This script reads both receipt directories and draws, for the
carrier and for the exterior, the wall each row spent building against the wall
its counterpart spent loading.

Usage:
    build_figure.py RUN1_DIR RUN2_DIR OUT.svg

A row present in only one of the two directories is drawn with the mark it has
and reported on stderr, so an incomplete gate shows as a short figure rather
than a figure that silently invents a wall.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# A row is identified by the requested cell count and the case. The order is the
# order the gate parametrises them, so the figure and the test log read alike.
ROW_ORDER = [
    (550, "limited"),
    (550, "diverted"),
    (2000, "limited"),
    (2000, "diverted"),
    (5000, "limited"),
    (5000, "diverted"),
]

# The build is the displaced cost, so it draws in the neutral grey the reader
# reads as background; the load is what the cache delivers and takes the one
# feature hue. Red is reserved for a gate that fails, and no mark here is a
# failure.
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


def walls(row: dict, prefix: str) -> tuple[float, float, bool]:
    build = float(row[f"{prefix}_build_seconds"])
    load = float(row[f"{prefix}_load_seconds"])
    return build, load, bool(row[f"{prefix}_cache_hit"])


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
    run1_dir, run2_dir, out = Path(argv[1]), Path(argv[2]), Path(argv[3])
    cold, warm = read_rows(run1_dir), read_rows(run2_dir)

    missing = []
    rows = []
    for key in ROW_ORDER:
        if key not in cold or key not in warm:
            missing.append(key)
            continue
        rows.append(key)
    if missing:
        print(
            "incomplete gate, rows omitted: "
            + ", ".join(f"{cells}-{case}" for cells, case in missing),
            file=sys.stderr,
        )
    if not rows:
        print("no row is present in both runs", file=sys.stderr)
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
        f"{cells} {case}\n({cold[(cells, case)]['realised_cells']} cells)"
        for cells, case in rows
    ]

    for axis, (prefix, title) in zip(panels, PANELS, strict=True):
        marks = []
        for index, key in enumerate(rows):
            build, _, cold_hit = walls(cold[key], prefix)
            _, load, warm_hit = walls(warm[key], prefix)
            # The build must come from the run that was cold and the load from
            # the run that was warm; reading both from one run would draw the
            # clamp floor in place of a wall and name the wrong run.
            if cold_hit:
                print(
                    f"{key[0]}-{key[1]}: the first run loaded the {prefix}, so it "
                    "reports no build wall",
                    file=sys.stderr,
                )
            if not warm_hit:
                print(
                    f"{key[0]}-{key[1]}: the second run reports no {prefix} cache hit",
                    file=sys.stderr,
                )
            marks.append(max(load, 1e-3))
            marks.append(max(build, 1e-3))
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
        # One decade of margin keeps the extreme marks clear of the spines,
        # so a decade label never lands on the panel edge.
        low, high = min(marks), max(marks)
        axis.set_xlim(low / 3.0, high * 3.0)
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
        ["built (first run)", "loaded (second run)"],
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
