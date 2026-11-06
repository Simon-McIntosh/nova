"""Compare device time by source owner on a shared linear axis."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.prototypes.bank_solve_profile.plot import save_checked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    rows = json.loads((args.run / "attribution.json").read_text())["solves"]
    values = []
    for row in rows:
        groups = dict(row["groups"])
        shared = sum(
            v["device_ms"] for k, v in groups.items() if k.startswith("shared:")
        )
        groups = {
            k: v["device_ms"] for k, v in groups.items() if not k.startswith("shared:")
        }
        groups["shared fused operations"] = shared
        values.append(groups)
    names = sorted(values[1], key=lambda name: -values[1][name])
    plt.style.use("data-ink")
    fig, ax = plt.subplots(figsize=(14, 7.2), dpi=100)
    y = np.arange(len(names))
    ax.plot(
        [values[0].get(name, 0.0) for name in names],
        y + 0.18,
        "s",
        color="#929292",
        markersize=6,
        label="1 slice",
        zorder=2,
    )
    ax.plot(
        [values[1].get(name, 0.0) for name in names],
        y - 0.18,
        "o",
        color="#1f5d92",
        markersize=6,
        label="16 slices",
        zorder=3,
    )
    ax.set_yticks(y, names, fontsize=13)
    ax.invert_yaxis()
    ax.set_xlim(-8, 450)
    ax.set_xticks([0, 100, 200, 300, 400])
    ax.set_xlabel("Device time per solve [ms]", fontsize=16)
    ax.tick_params(axis="y", length=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(False)
    fig.legend(loc="upper center", bbox_to_anchor=(0.65, 0.99), ncol=2, frameon=False)
    fig.subplots_adjust(left=0.34, right=0.97, top=0.90, bottom=0.13)
    output = (
        root
        / "docs/figures/sub-millisecond-forward-path/bank-solve-trace/device-time.svg"
    )
    save_checked(fig, output)
    fig.savefig(args.run / "device-time.png", dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    main()
