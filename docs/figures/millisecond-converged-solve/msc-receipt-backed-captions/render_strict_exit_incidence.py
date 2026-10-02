"""Re-render strict-exit-incidence.png from its committed receipt alone.

Two presentation defects are repaired here; neither touches the measurement:

* The receipt records ``strict_qualification_firing_trip`` as ``None`` for every
  harvested member, so the two former incidence panels plotted no measured
  value at all -- each marker sat on the "not persisted" line. Those panels are
  dropped; the figure now draws only the paired per-member latency the receipt
  carries.
* The former latency panels joined members in index order, inviting a trend
  line between categorical members. The members are drawn as markers only.

The rendering reads ``strict-exit-incidence.json`` and nothing else: no solve
runs, no data re-measured. Run on the login node.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from nova.media.ink import DEFAULT_INK, trace_axes  # noqa: E402

RECEIPT = Path("docs/figures/millisecond-converged-solve/strict-exit-incidence.json")
OUTPUT = Path("docs/figures/millisecond-converged-solve/strict-exit-incidence.png")

DISABLED_COLOR = DEFAULT_INK.contour_color
ENABLED_COLOR = DEFAULT_INK.flux_color


def _members(payload: dict, machine: str) -> list[dict]:
    return payload["machines"][machine]["members"]


def _latency_seconds(member: dict, arm: str) -> float:
    return member[arm]["timing"]["compile_warm_solve_ms"] / 1000.0


def _draw(payload: dict, output: Path) -> dict:
    plt.style.use("data-ink")
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5.5),
        gridspec_kw={"width_ratios": (12, 3)},
        constrained_layout=True,
    )
    machines = ("MAST", "DIII-D")
    summary: dict[str, dict] = {}
    for axis, machine in enumerate(machines):
        panel = axes[axis]
        rows = _members(payload, machine)
        labels = [row["identity"].replace(" frame ", "\nframe ") for row in rows]
        positions = np.arange(len(rows))
        control = [_latency_seconds(row, "without_exit") for row in rows]
        exited = [_latency_seconds(row, "with_exit") for row in rows]
        panel.scatter(
            positions,
            control,
            s=42,
            marker="o",
            color=DISABLED_COLOR,
            label="strict exit disabled" if axis == 0 else None,
            zorder=3,
        )
        panel.scatter(
            positions,
            exited,
            s=42,
            marker="x",
            color=ENABLED_COLOR,
            label="strict exit enabled" if axis == 0 else None,
            zorder=3,
        )
        panel.set_xticks(positions)
        panel.set_xticklabels(labels, rotation=55, ha="right")
        panel.set_ylim(bottom=0)
        trace_axes(panel)
        if axis == 1:
            summary[machine] = {"members": len(rows)}
    axes[0].set_ylabel("compile-warm solve [s]")
    # Direct labels: one per series, in the series colour, beside the first point.
    axes[0].annotate(
        "exit disabled",
        (0, _latency_seconds(_members(payload, "MAST")[0], "without_exit")),
        textcoords="offset points",
        xytext=(-4, 8),
        ha="left",
        fontsize=DEFAULT_INK.label_fontsize,
        color=DISABLED_COLOR,
    )
    axes[0].annotate(
        "exit enabled",
        (0, _latency_seconds(_members(payload, "MAST")[0], "with_exit")),
        textcoords="offset points",
        xytext=(-4, -14),
        ha="left",
        fontsize=DEFAULT_INK.label_fontsize,
        color=ENABLED_COLOR,
    )

    completion = payload.get("completion", {})
    declared = completion.get("declared_member_count")
    missing = completion.get("missing_member_count")
    note = ""
    if declared is not None and missing is not None:
        note = f"{missing} of {declared} declared members missing"
    axes[1].annotate(
        note,
        xy=(0.5, -0.42),
        xycoords="axes fraction",
        ha="center",
        va="top",
        fontsize=DEFAULT_INK.label_fontsize,
        color="#555555",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    summary["disclosure"] = {
        "declared_member_count": declared,
        "missing_member_count": missing,
        "note": note,
    }
    return summary


def main() -> int:
    payload = json.loads(RECEIPT.read_text())
    summary = _draw(payload, OUTPUT)
    print("RENDERED", OUTPUT)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
