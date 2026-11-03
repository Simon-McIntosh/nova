"""Draw paired warm stage walls from the device-arm receipts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.text import Text

from scripts.prototypes.bank_step_speed.plot_stage import stages as read_stages

STAGES = (
    ("read", "read"),
    ("residual_and_booking", "residual and booking"),
    ("local_booking_tangent", "booking tangent"),
    ("geometry_tangent_one_direction", "moving geometry"),
    ("whole_step", "whole step"),
)


def main(base, head, output):
    before, after = read_stages(Path(base)), read_stages(Path(head))
    plt.style.use(Path.home() / "Code/dotfiles/matplotlib/stylelib/data-ink.mplstyle")
    fig, ax = plt.subplots(figsize=(14, 7.2), layout="constrained")
    rows = np.arange(len(STAGES))
    labels = [label for _, label in STAGES]
    first = [before[key] for key, _ in STAGES]
    second = [after[key] for key, _ in STAGES]
    ax.barh(rows - 0.18, first, height=0.28, color="#929292", label="base")
    ax.barh(rows + 0.18, second, height=0.28, color="#2e8b57", label="bucketed")
    for positions, values, colour in (
        (rows - 0.18, first, "#555555"),
        (rows + 0.18, second, "#216d45"),
    ):
        for y, value in zip(positions, values, strict=True):
            ax.text(value + 0.5, y, f"{value:.2f}", va="center", color=colour)
    ax.set_yticks(rows, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, max(first + second) * 1.23)
    ax.set_xticks(np.arange(0, 51, 10))
    ax.set_xlabel("warm wall [ms]")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=2, frameon=False)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    labels = [
        artist
        for artist in fig.findobj(match=Text)
        if artist.get_visible() and artist.get_text()
    ]
    bounds = [(artist, artist.get_window_extent(renderer)) for artist in labels]
    canvas = fig.bbox
    for artist, bounds_box in bounds:
        if not canvas.contains(*bounds_box.get_points()[0]) or not canvas.contains(
            *bounds_box.get_points()[1]
        ):
            raise ValueError(f"text outside canvas: {artist.get_text()}")
    for index, (first_artist, first_box) in enumerate(bounds):
        for second_artist, second_box in bounds[index + 1 :]:
            if first_box.overlaps(second_box):
                raise ValueError(
                    f"text collision: {first_artist.get_text()} / "
                    f"{second_artist.get_text()}"
                )
    fig.savefig(output)
    svg = Path(output)
    content = svg.read_text()
    content = content.replace(
        'width="1008pt" height="518.4pt"', 'width="31.1cm" height="16cm"'
    )
    svg.write_text(content)
    print(json.dumps({"base": before, "bucketed": after}, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--output", required=True)
    main(**vars(parser.parse_args()))
