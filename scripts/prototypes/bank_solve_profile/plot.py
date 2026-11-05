"""Draw the per-iterate stage split from a stage-profile receipt."""

import argparse
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.ticker import StrMethodFormatter

ROOT = Path(__file__).resolve().parents[3]
FIGURE = (
    ROOT
    / "docs/figures/sub-millisecond-forward-path"
    / "bank-solve-profile/per-iterate-stage.svg"
)

# Related read stages are lightness steps of one hue; the step stages are one
# hue each; the remainder is neutral grey.
STAGES = [
    ("read census", "#1f5d92"),
    ("read decision", "#3f8ecb"),
    ("read geometry", "#8fc0e4"),
    ("step residual", "#2e8b57"),
    ("step assembly", "#e07b39"),
    ("factor", "#7d5ba6"),
    ("back-solve", "#a05195"),
    ("remainder", "#929292"),
]


def records(log):
    """Every stage and solve receipt, keyed by lane count."""
    stages = {}
    solves = {}
    for line in Path(log).read_text().splitlines():
        if line.startswith("STAGE "):
            row = json.loads(line.partition(" ")[2])
            stages[(row["lanes"], row["name"])] = row["median_ms"]
        elif line.startswith("SOLVE "):
            row = json.loads(line.partition(" ")[2])
            solves[row["lanes"]] = row
    return stages, solves


def split(stages, solves, lanes):
    """Per-iterate stage wall in ms for one lane count."""
    named = {
        "read census": stages[(lanes, "census")],
        "read decision": stages[(lanes, "topology")]
        - stages[(lanes, "support_geometry")],
        "read geometry": stages[(lanes, "fit")] + stages[(lanes, "support_geometry")],
        "step residual": 2.0 * stages[(lanes, "residual")],
        "step assembly": stages[(lanes, "assembly")],
        "factor": stages[(lanes, "factor")],
        "back-solve": stages[(lanes, "backsolve")],
    }
    whole = solves[lanes]
    iterate = stages[(lanes, "whole_solve")] / whole["max_steps"]
    named["remainder"] = iterate - sum(named.values())
    return named, iterate, whole


def save_checked(fig, output):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.bbox
    texts = [
        item for item in fig.findobj(Text) if item.get_visible() and item.get_text()
    ]
    boxes = [(item, item.get_window_extent(renderer)) for item in texts]
    for item, box in boxes:
        if not all(canvas.contains(*corner) for corner in box.get_points()):
            raise ValueError(f"text outside canvas: {item.get_text()}")
    for index, (left, left_box) in enumerate(boxes):
        for right, right_box in boxes[index + 1 :]:
            if left_box.overlaps(right_box):
                raise ValueError(
                    f"text overlap: {left.get_text()} / {right.get_text()}"
                )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    root = ET.parse(output)
    root.getroot().set("width", "31.1cm")
    root.getroot().set("height", "16cm")
    root.write(output, encoding="unicode", xml_declaration=True)
    print(f"FIGURE={output} TEXT_CHECK=passed CANVAS=1400x720", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--output", type=Path, default=FIGURE)
    args = parser.parse_args()
    stages, solves = records(args.log)
    arm_one, iterate_one, whole_one = split(stages, solves, 1)
    arm_sixteen, iterate_sixteen, whole_sixteen = split(stages, solves, 16)
    report = {
        "per_iterate": {"one": arm_one, "sixteen": arm_sixteen},
        "iterate_ms": {"one": iterate_one, "sixteen": iterate_sixteen},
        "closure": {
            "one": sum(arm_one[name] for name, _ in STAGES[:-1]) / iterate_one,
            "sixteen": sum(arm_sixteen[name] for name, _ in STAGES[:-1])
            / iterate_sixteen,
        },
        "solve": {"one": whole_one, "sixteen": whole_sixteen},
    }
    print("REPORT " + json.dumps(report, sort_keys=True), flush=True)

    plt.style.use("data-ink")
    fig, ax = plt.subplots(figsize=(14, 7.2), dpi=100)
    rows = [
        ("16 slices", arm_sixteen, iterate_sixteen, 1),
        ("1 slice", arm_one, iterate_one, 0),
    ]
    totals = [
        sum(max(value, 0.0) for value in values.values()) for _, values, _, _ in rows
    ]
    top = max(max(totals), iterate_one, iterate_sixteen) * 1.12
    for label, values, iterate, y in rows:
        left = 0.0
        bars = [(name, colour, max(values[name], 0.0)) for name, colour in STAGES]
        for name, colour, width in bars:
            ax.barh(
                y,
                width,
                left=left,
                height=0.42,
                color=colour,
                edgecolor="white",
                linewidth=0.8,
                zorder=2,
            )
            if width >= 0.05 * top:
                ax.text(
                    left + width / 2.0,
                    y,
                    f"{width:.1f}" if width < 100 else f"{width:.0f}",
                    ha="center",
                    va="center",
                    fontsize=14,
                    color="white" if name != "read geometry" else "#12324a",
                    zorder=4,
                )
            left += width
        ax.text(
            left,
            y - 0.34,
            f"{iterate:.1f} ms per iterate",
            ha="right",
            va="top",
            fontsize=14,
            color="#333333",
            zorder=4,
        )
    ax.set_yticks([0, 1], [row[0] for row in rows])
    ax.set_ylim(-0.6, 1.55)
    ax.set_xlim(0, top)
    step = 10 ** math.floor(math.log10(top / 5.0))
    step = math.ceil(top / 5.0 / step) * step
    ax.set_xticks([step * index for index in range(int(top / step) + 1)])
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
    ax.set_autoscalex_on(False)
    ax.set_xlabel("Per-iterate wall by stage [ms]")
    ax.tick_params(axis="both", labelsize=14)
    ax.xaxis.label.set_size(16)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colour) for _, colour in STAGES]
    ax.legend(
        handles,
        [name for name, _ in STAGES],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=4,
        frameon=False,
        fontsize=14,
    )
    fig.subplots_adjust(left=0.14, right=0.97, bottom=0.14, top=0.80)
    save_checked(fig, args.output)
    plt.close(fig)


if __name__ == "__main__":
    main()
