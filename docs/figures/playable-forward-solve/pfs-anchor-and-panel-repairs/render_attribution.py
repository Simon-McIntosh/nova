"""Re-render the centroid-attribution and batched-labeller panels.

Each panel is drawn from its committed receipt: the class comparison from the
per-row requested and emergent class fields, the throughput panel from the
receipt's measured arms rather than the exception path, and the parity panel
from the verified corpus fractions, so no panel carries a series its receipt
does not hold.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from nova.media.ink import trace_axes  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
CENTROID = ROOT / "docs/figures/playable-forward-solve/centroid-attribution"
BATCHED = ROOT / "docs/figures/playable-forward-solve/batched-labeller"

BLUE = "#3366cc"
ORANGE = "#cc6633"
GREEN = "#00916e"
RED = "#cc3344"
GREY = "#666666"


def render_centroid_attribution() -> Path:
    receipt = json.loads((CENTROID / "centroid-radius-attribution.json").read_text())
    frozen = [row for row in receipt["rows"] if row["operator_anchor"] == "frozen"]
    figure, (left, right) = plt.subplots(1, 2, figsize=(13.0, 5.0))

    trace_axes(left)
    for name, color, marker in (
        ("class", BLUE, "o"),
        ("support", ORANGE, "s"),
        ("weighting", GREEN, "^"),
    ):
        values = [row["fraction_accounted"][name] for row in frozen]
        rows = [row["row"] for row in frozen]
        left.plot(rows, values, marker=marker, linestyle="", markersize=6, color=color)
        left.annotate(
            name,
            (rows[-1], values[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            color=color,
            fontsize=11,
            va="center",
        )
    left.axhline(0.0, color="#999999", linewidth=1.0, linestyle="dashed")
    left.set_xlabel("frame row")
    left.set_ylabel("fraction of centroid offset accounted")
    left.set_title(
        "Offset attribution by candidate (frozen anchor; %d rows)" % len(frozen)
    )

    trace_axes(right)
    counts = np.zeros((2, 2), dtype=int)
    for row in frozen:
        counts[
            int(bool(row["requested_diverted"])), int(bool(row["emergent_diverted"]))
        ] += 1
    for i in range(2):
        for j in range(2):
            right.scatter(i, j, s=2400, color=BLUE if counts[i, j] else "#dddddd")
            right.text(
                i,
                j,
                str(counts[i, j]),
                ha="center",
                va="center",
                fontsize=18,
                color="white" if counts[i, j] else "#333333",
                fontweight="bold",
            )
    right.set_xticks([0, 1], ["no", "yes"])
    right.set_yticks([0, 1], ["no", "yes"])
    right.set_xlabel("emergent diverted class")
    right.set_ylabel("requested diverted class")
    agree = sum(int(row["classes_agree"]) for row in frozen)
    right.set_title(
        "Requested vs emergent class (frozen anchor; %d/%d agree)"
        % (agree, len(frozen))
    )
    right.set_xlim(-0.6, 1.5)
    right.set_ylim(-0.6, 1.5)

    figure.tight_layout()
    path = CENTROID / "centroid-radius-attribution.png"
    figure.savefig(path, dpi=170)
    plt.close(figure)
    return path


def render_h200_throughput() -> Path:
    receipt = json.loads((BATCHED / "h200-throughput-receipt.json").read_text())
    figure, axes = plt.subplots(figsize=(8.4, 5.2))
    trace_axes(axes)
    arms = sorted(receipt["arms"], key=lambda arm: arm["batch_per_device"])
    batches = [arm["batch_per_device"] for arm in arms]
    rates = [arm["attempted_slices_per_second"] for arm in arms]
    axes.plot(batches, rates, marker="o", markersize=8, color=BLUE, linewidth=2.6)
    for batch, rate in zip(batches, rates):
        axes.annotate(
            "%.2f" % rate,
            (batch, rate),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=11,
            color=BLUE,
        )
    axes.set_xscale("log", base=2)
    axes.set_xticks(batches, [str(batch) for batch in batches])
    axes.set_xlabel("elements per device")
    axes.set_ylabel("batched throughput [slices/s]")
    reference = receipt["sequential_compiled_reference"]
    axes.set_title(
        "H200 batched-labeller throughput (measured arms; %d-slice reference)"
        % reference["slice_count"]
    )
    path = BATCHED / "h200-throughput-receipt.png"
    figure.savefig(path, dpi=170)
    plt.close(figure)
    return path


def render_terminal_state_parity() -> Path:
    receipt = json.loads((BATCHED / "acceptance-receipt.json").read_text())
    parity = receipt["fraction_parity"]
    labels = ["converged", "guard", "conditioned"]
    batched = [
        parity["batched"]["converged_fraction"],
        parity["batched"]["guard_fraction"],
        parity["batched"]["conditioned_fraction"],
    ]
    sequential = [
        parity["sequential"]["converged_fraction"],
        parity["sequential"]["guard_fraction"],
        parity["sequential"]["conditioned_fraction"],
    ]
    figure, axes = plt.subplots(figsize=(7.6, 5.0))
    trace_axes(axes)
    positions = np.arange(len(labels))
    axes.bar(positions - 0.2, sequential, width=0.4, color=GREY, label="sequential")
    axes.bar(positions + 0.2, batched, width=0.4, color=RED, label="batched")
    axes.set_xticks(positions, labels)
    axes.set_ylabel("slice fraction")
    axes.set_title(
        "Batched vs sequential fraction parity (%d slices; parity holds: %s)"
        % (parity["denominator"], receipt["acceptance_contract"]["fraction_parity"])
    )
    axes.legend(frameon=False, fontsize=11)
    path = BATCHED / "terminal-state-parity.png"
    figure.savefig(path, dpi=170)
    plt.close(figure)
    return path


if __name__ == "__main__":
    print("wrote", render_centroid_attribution().relative_to(ROOT))
    print("wrote", render_h200_throughput().relative_to(ROOT))
    print("wrote", render_terminal_state_parity().relative_to(ROOT))
