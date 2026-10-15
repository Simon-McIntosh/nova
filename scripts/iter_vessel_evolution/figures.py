"""Render the vessel-evolution receipt as data-ink figures.

Each figure reads a scenario receipt written by :mod:`evolve` and writes one
SVG under ``docs/figures/iter-vessel-evolution``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "docs" / "figures" / "iter-vessel-evolution"

#: component colours, kept consistent across the study
DINA = "#315f78"
EVOLVED = "#a8593a"
BARE = "#8a8a8a"
PASSIVE_ONLY = "#5b8c5a"
FULL = "#315f78"


def _style() -> None:
    try:
        plt.style.use("data-ink")
    except OSError:
        pass


def loop_residual(receipt: dict, output: Path) -> None:
    """Per-loop peak residual against each loop's own tolerance."""
    rows = [
        row for row in receipt["group_comparison"]["rows"] if "residual_peak_a" in row
    ]
    rows.sort(key=lambda row: row["residual_peak_a"])
    peak = np.array([row["residual_peak_a"] for row in rows]) / 1e3
    tolerance = np.array([row["tolerance_a"] for row in rows]) / 1e3
    index = np.arange(len(rows))
    outside = peak > tolerance

    fig, ax = plt.subplots(figsize=(14, 5.4))
    ax.plot(
        index[~outside], peak[~outside], "o", color=DINA, ms=5, label="within tolerance"
    )
    ax.plot(
        index[outside],
        peak[outside],
        "o",
        color="#b03030",
        ms=7,
        label="outside tolerance",
    )
    ax.plot(index, tolerance, "-", color="#8a8a8a", lw=1.2, label="tolerance")
    worst = rows[-1]
    position = len(rows) - 1
    ax.annotate(
        worst["name"],
        (position, peak[position]),
        textcoords="offset points",
        xytext=(-10, 10),
        fontsize=15,
        color="#b03030",
        ha="right",
    )
    ax.set_xlabel("DINA pf_passive loop")
    ax.set_ylabel("peak residual [kA]")
    ax.set_xlim(-1, len(rows))
    ax.legend(frameon=False, fontsize=16, loc="upper left")
    ax.set_title("")
    fig.tight_layout()
    fig.savefig(output, format="svg")
    plt.close(fig)


def voltage_residual(receipt: dict, output: Path) -> None:
    """Per-coil voltage rms residual across the coupling arms."""
    arms = [
        ("bare coils", "voltage_bare", BARE),
        ("coils + passive", "voltage_passive", PASSIVE_ONLY),
        ("coils + passive + plasma", "voltage_full", FULL),
    ]
    names = list(receipt["voltage_full"]["coils"])
    positions = np.arange(len(names))
    width = 0.26
    fig, ax = plt.subplots(figsize=(14, 5.4))
    for offset, (label, key, colour) in enumerate(arms):
        values = [receipt[key]["coils"][name]["rms_v"] for name in names]
        ax.bar(
            positions + (offset - 1) * width, values, width, color=colour, label=label
        )
    ax.set_xticks(positions)
    ax.set_xticklabels(names, rotation=0)
    ax.set_ylabel("per-coil voltage residual rms [V]")
    ax.legend(frameon=False, fontsize=16)
    fig.tight_layout()
    fig.savefig(output, format="svg")
    plt.close(fig)


def vs3_arm(receipt: dict, output: Path) -> None:
    """The psi-map residual at both VS3 sections, with and without the passive flux."""
    points = receipt["vs3"]["points"]
    labels = [f"VS3 section {row['point']}" for row in points]
    positions = np.arange(len(points))
    width = 0.2
    series = [
        ("psi-map residual", [row["section2_peak_wb"] for row in points], BARE),
        (
            "DINA passive flux",
            [row["dina_passive_flux_peak_wb"] for row in points],
            DINA,
        ),
        (
            "evolved passive flux",
            [row["evolved_passive_flux_peak_wb"] for row in points],
            EVOLVED,
        ),
        (
            "residual after",
            [row["residual_after_peak_wb"] for row in points],
            "#b03030",
        ),
    ]
    fig, ax = plt.subplots(figsize=(14, 5.4))
    for offset, (label, values, colour) in enumerate(series):
        ax.bar(
            positions + (offset - 1.5) * width, values, width, color=colour, label=label
        )
    ax.axhline(receipt["vs3"]["tolerance_wb"], color="#8a8a8a", ls=":", lw=1.2)
    ax.text(
        positions[0] - 0.45,
        receipt["vs3"]["tolerance_wb"],
        "psi-map tolerance",
        fontsize=15,
        color="#555555",
        va="bottom",
        ha="left",
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("flux [Wb]")
    ax.legend(frameon=False, fontsize=16)
    fig.tight_layout()
    fig.savefig(output, format="svg")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pulse", type=int, required=True)
    parser.add_argument("--tag", default="12mm")
    parser.add_argument("--receipt", type=Path, default=None)
    args = parser.parse_args()
    receipt_path = args.receipt or (
        Path("/home/ITER/mcintos/Code/nova/.cache/iter-passive-plasma-circuit")
        / "gate-logs/iter-vessel-evolution"
        / f"result-{args.pulse}-{args.tag}.json"
    )
    receipt = json.loads(receipt_path.read_text())
    _style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    loop_residual(receipt, FIGURES / f"vessel-loop-residual-{args.pulse}.svg")
    voltage_residual(receipt, FIGURES / f"vessel-voltage-residual-{args.pulse}.svg")
    vs3_arm(receipt, FIGURES / f"vessel-vs3-residual-{args.pulse}.svg")
    print(f"wrote 3 figures for {args.pulse} to {FIGURES}")


if __name__ == "__main__":
    main()
