"""Render measured program costs and terminal poloidal fields."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.prototypes.fast_read.render_contours import panel


def render_costs(before, after, output):
    plt.style.use("data-ink")
    widths = np.array((1, 16, 128))
    figure, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=100)
    for axis, key, divisor, limit, ylabel in (
        (axes[0], "executable_bytes", 1e6, 50, "executable [MB]"),
        (axes[1], "per_state_ms", 1, 0.90, "two corrections per state [ms]"),
    ):
        original = np.array([before[f"batch_{n}"][key] / divisor for n in widths])
        runtime = np.array([after[f"batch_{n}"][key] / divisor for n in widths])
        axis.plot(widths, original, color="0.45", linestyle="--", linewidth=2.6)
        axis.plot(widths, runtime, color="#265f8a", linewidth=3.0)
        axis.axhline(limit, color="0.45", linewidth=1.2, linestyle=":")
        axis.text(widths[-1] * 1.04, original[-1], "closure", color="0.35", va="center")
        axis.text(
            widths[-1] * 1.04, runtime[-1], "runtime", color="#265f8a", va="center"
        )
        axis.text(widths[0], limit, f"bound {limit:g}", color="0.35", va="bottom")
        axis.set_xscale("log", base=2)
        axis.set_xticks(widths, labels=[str(x) for x in widths])
        axis.set_xlim(0.8, 220)
        axis.set_xlabel("batch width")
        axis.set_ylabel(ylabel)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(False)
    figure.tight_layout()
    figure.savefig(output / "program-costs.svg", metadata={"Date": None})
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    before = json.loads(args.before.read_text())
    after = json.loads(args.after.read_text())
    render_costs(before, after, args.output)
    for width, payload in after["panels"].items():
        panel(payload, args.output / f"limited-{width}-terminal.svg")
        print(f"PANEL width={width} residual={payload['terminal_residual']}")


if __name__ == "__main__":
    main()
