"""Fit warm-wall stage scaling and render the two-carrier comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


CASES = ("limited", "diverted")
RUNG = (550, 2000, 5000, 10000)
INK = "#46778a"
GREY = "#929292"


def summarize(input_dir: Path) -> dict:
    rows = {
        case: [
            json.loads((input_dir / f"{case}-{cells}.json").read_text())
            for cells in RUNG
        ]
        for case in CASES
    }
    result: dict = {"rows": rows, "fits": {}}
    for case, family in rows.items():
        cells = np.asarray([row["realised_cells"] for row in family], dtype=float)
        walls = np.asarray([row["warm_total_seconds"] for row in family])
        stage_names = list(family[-1]["stage_seconds"])
        fits = {
            "whole read": {
                "exponent": float(np.polyfit(np.log(cells), np.log(walls), 1)[0]),
                "wall_10000": float(walls[-1]),
                "share_10000": 1.0,
            }
        }
        for name in stage_names:
            times = np.asarray([row["stage_seconds"][name] for row in family])
            fits[name] = {
                "exponent": float(np.polyfit(np.log(cells), np.log(times), 1)[0]),
                "wall_10000": float(times[-1]),
                "share_10000": float(times[-1] / walls[-1]),
            }
        for name in family[-1]["substage_seconds"]:
            times = np.asarray([row["substage_seconds"][name] for row in family])
            fits[name] = {
                "exponent": float(np.polyfit(np.log(cells), np.log(times), 1)[0]),
                "wall_10000": float(times[-1]),
                "share_10000": float(times[-1] / walls[-1]),
                "nested": True,
            }
        result["fits"][case] = fits
    return result


def draw(summary: dict, destination: Path, preview: Path) -> None:
    plt.style.use("data-ink")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.8), sharey=True)
    dominant = max(
        summary["fits"]["limited"].items(),
        key=lambda item: (
            item[1]["wall_10000"]
            if item[0] != "whole read" and not item[1].get("nested")
            else -1
        ),
    )[0]
    styles = {
        "sampled field": "-.",
        "contour tree": "-",
        "null census": ":",
        "support selection": "--",
    }
    for axis, case in zip(axes, CASES, strict=True):
        rows = summary["rows"][case]
        cells = [row["realised_cells"] for row in rows]
        axis.plot(
            cells,
            [row["warm_total_seconds"] for row in rows],
            color="#4e4e4e",
            linewidth=2.6,
            linestyle="-",
            label="whole read",
            zorder=2,
        )
        for name, style in styles.items():
            axis.plot(
                cells,
                [row["stage_seconds"][name] for row in rows],
                color=INK if name == dominant else GREY,
                linewidth=3.0 if name == dominant else 2.6,
                linestyle=style,
                label=name,
                zorder=4 if name == dominant else 3,
            )
        axis.axhline(
            10.0, color="#696969", linestyle=":", linewidth=1.2, label="10 s target"
        )
        axis.set(xscale="log", yscale="log", xlabel="cells", ylabel="warm wall [s]")
        axis.set_title(case)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.0),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination)
    fig.savefig(preview, dpi=100)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--preview", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize(args.input_dir)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    draw(summary, args.figure, args.preview)
    print("SUMMARY " + json.dumps(summary["fits"], sort_keys=True))


if __name__ == "__main__":
    main()
