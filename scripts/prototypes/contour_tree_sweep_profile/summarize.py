"""Fit contour-tree stage scaling and draw the H200 warm-wall figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.text import Text  # noqa: E402
import numpy as np  # noqa: E402


CASES = ("limited", "diverted")
SIZES = (550, 2000, 5000, 10000)
STAGES = (
    "full build",
    "join sweep",
    "split sweep",
    "join append estimate",
    "split append estimate",
    "adjacency",
    "split insertion",
    "other build",
)


def exponent(vertices, values):
    if any(value <= 0 for value in values):
        return None
    return float(np.polyfit(np.log(vertices), np.log(values), 1)[0])


def summarize(main_dir: Path, comparison_dir: Path) -> dict:
    result: dict = {"main": {}, "comparison": {}}
    for label, directory in (("main", main_dir), ("comparison", comparison_dir)):
        for case in CASES:
            rows = [
                json.loads((directory / f"{case}-{size}.json").read_text())
                for size in SIZES
            ]
            vertices = np.array([row["vertices"] for row in rows], dtype=float)
            stages = {
                stage: {
                    "exponent": exponent(
                        vertices,
                        [row["stage_seconds"][stage] for row in rows],
                    ),
                    "wall_10000": rows[-1]["stage_seconds"][stage],
                }
                for stage in STAGES
            }
            output: dict = {"rows": rows, "stages": stages}
            if label == "main":
                count_rows = [
                    json.loads((directory / f"counts-{case}-{size}.json").read_text())
                    for size in SIZES
                ]
                counts = {}
                for name in ("join", "split"):
                    rounds = [
                        item["counts"][name]["roots_total"] for item in count_rows
                    ]
                    maxima = [item["counts"][name]["roots_max"] for item in count_rows]
                    walls = [
                        row["stage_seconds"][f"{name} sweep"] / row["vertices"]
                        for row in rows
                    ]
                    counts[name] = {
                        "visits": [
                            item["counts"][name]["visits"] for item in count_rows
                        ],
                        "roots_total": rounds,
                        "roots_max": maxima,
                        "wall_per_visit": walls,
                        "round_exponent": exponent(vertices, rounds),
                        "maximum_exponent": exponent(vertices, maxima),
                        "per_visit_exponent": exponent(vertices, walls),
                    }
                no_append = sum(
                    rows[-1]["passes"][name]["no_append_wall"]
                    for name in ("join", "split")
                )
                total_rounds = sum(counts[name]["roots_total"][-1] for name in counts)
                seconds_per_round = no_append / total_rounds
                retained = sum(
                    stages[name]["wall_10000"]
                    for name in (
                        "join append estimate",
                        "split append estimate",
                        "adjacency",
                        "split insertion",
                        "other build",
                    )
                )
                output["counts"] = counts
                output["repair_estimate"] = {
                    "seconds_per_round": seconds_per_round,
                    "depth_bound": 14,
                    "bounded_root_wall": seconds_per_round * 2 * vertices[-1] * 14,
                    "retained_stage_wall": retained,
                    "predicted_tree_wall": seconds_per_round * 2 * vertices[-1] * 14
                    + retained,
                }
            result[label][case] = output
    return result


def check_text(fig) -> None:
    fig.canvas.draw()
    canvas = fig.bbox
    artists = [
        artist
        for artist in fig.findobj(match=Text)
        if artist.get_visible() and artist.get_text().strip()
    ]
    boxes = []
    for artist in artists:
        box = artist.get_window_extent(fig.canvas.get_renderer())
        if not canvas.contains(*box.get_points()[0]) or not canvas.contains(
            *box.get_points()[1]
        ):
            raise ValueError(f"text outside canvas: {artist.get_text()}")
        boxes.append((artist, box))
    for index, (left, left_box) in enumerate(boxes):
        for right, right_box in boxes[index + 1 :]:
            if left_box.overlaps(right_box):
                raise ValueError(
                    f"text overlap: {left.get_text()} / {right.get_text()}"
                )


def draw(summary: dict, svg: Path, preview: Path) -> None:
    plt.style.use("data-ink")
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.2), sharey=True)
    styles = {
        "full build": ("#343434", "--", "o", 2.6),
        "join sweep": ("#46778a", "-", "o", 3.0),
        "split sweep": ("#46778a", "--", "s", 2.6),
        "join append estimate": ("#46778a", ":", "o", 1.8),
        "split append estimate": ("#46778a", ":", "s", 1.8),
        "adjacency": ("#929292", "-", "o", 2.0),
        "split insertion": ("#929292", "--", "s", 2.0),
        "other build": ("#929292", ":", "^", 2.0),
    }
    for axis, case in zip(axes, CASES, strict=True):
        rows = summary["main"][case]["rows"]
        vertices = [row["vertices"] for row in rows]
        for name in STAGES:
            color, pattern, marker, width = styles[name]
            axis.plot(
                vertices,
                [row["stage_seconds"][name] for row in rows],
                color=color,
                linestyle=pattern,
                marker=marker,
                markersize=4.5,
                linewidth=width,
                label=name,
                zorder=5 if name in ("join sweep", "split sweep") else 2,
            )
        axis.text(0.04, 0.96, case, transform=axis.transAxes, va="top", fontsize=14)
        axis.set(xscale="log", yscale="log", xlabel="carrier vertices")
        axis.set_xticks((550, 2000, 5000, 10000))
        axis.set_xticklabels(("550", "2,000", "5,000", "10,000"))
        axis.set_yticks((0.01, 0.1, 1, 10, 100))
        axis.set_ylim(0.006, 180)
        axis.set_xlim(400, 14000)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(False)
    axes[0].set_ylabel("warm wall [s]")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.82))
    check_text(fig)
    svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(svg)
    fig.savefig(preview, dpi=100)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--svg", type=Path, required=True)
    parser.add_argument("--preview", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.main, args.comparison)
    args.summary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    draw(result, args.svg, args.preview)
    print(
        "SUMMARY "
        + json.dumps(
            {
                case: {
                    "full_exponent": result["main"][case]["stages"]["full build"][
                        "exponent"
                    ],
                    "repair_estimate": result["main"][case]["repair_estimate"],
                }
                for case in CASES
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
