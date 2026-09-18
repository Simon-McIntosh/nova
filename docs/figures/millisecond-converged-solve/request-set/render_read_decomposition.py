"""Render the read-request copy decomposition for the compile-cost evidence.

Reads the decomposition receipt's own JSON parts and writes one PNG with an
SVG twin and a data receipt beside it; no solver code runs and nothing is
compiled. The chain depth of a copy is the count of caller edges from the
computation that carries the read's marker up to the entry computation, which
is the number of segments in the parent chain the instrument records.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RECEIPT = Path(
    "/home/ITER/mcintos/.config/reckon/crew/reports/nova/s19-local/request-set/decomposition"
)
PARENTS = RECEIPT / "parents-solve.json"
NARRATIVE = "read-body-multiplication.md"
OUTPUT = HERE / "read-request-decomposition.png"

ACCENT = "#b5563b"
NEUTRAL = "#555555"

# Trace-flavour split of the 74 topology copies as the decomposition receipt
# states it. The parent-chain JSON records the enclosing region per copy and
# not the trace flavour, so these four counts are carried from that table.
TOPOLOGY_FLAVOURS = (
    ("primal jit(read_qualification)", 39),
    ("jvp twin", 25),
    ("transpose(jvp) twin", 3),
    ("second consumer (polish)", 7),
)


def depth_histogram(chains: list[dict]) -> dict[int, int]:
    histogram: collections.Counter = collections.Counter()
    for record in chains:
        histogram[len(str(record["chain"]).split("/"))] += int(record["count"])
    return dict(sorted(histogram.items()))


def style(axes) -> None:
    axes.set_facecolor("#fbfbf9")
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_linewidth(0.6)
        axes.spines[side].set_color("#888888")
    axes.tick_params(labelsize=7.5, color="#888888")


def receipt(moment: dict, topology: dict) -> dict:
    return {
        "schema": "nova.read-request-decomposition-figure",
        "sources": {"parents": str(PARENTS), "narrative": NARRATIVE},
        "chain_depth_definition": (
            "number of caller edges from the computation carrying the read's "
            "marker up to the entry computation, i.e. the parent chain's "
            "segment count"
        ),
        "moment_copies_by_chain_depth": {str(k): v for k, v in moment.items()},
        "topology_copies_by_chain_depth": {str(k): v for k, v in topology.items()},
        "moment_depth_mode": max(moment, key=lambda d: moment[d]),
        "topology_depth_mode": max(topology, key=lambda d: topology[d]),
        "moment_total": sum(moment.values()),
        "topology_total": sum(topology.values()),
        "topology_flavours": {name: value for name, value in TOPOLOGY_FLAVOURS},
        "topology_flavour_source": (
            "read-body-multiplication.md, the decomposition receipt's own table"
        ),
        "reading": (
            "the count is a fan of distinct enclosing compiled computations, "
            "one copy each, not a call site count; the moment histogram's mode "
            "is depth 8 and the topology read carries 28 derivative twins"
        ),
    }


def main() -> int:
    parts = json.loads(PARENTS.read_text(encoding="utf-8"))
    moment = depth_histogram(parts["current-moment path"]["by_parent_chain"])
    topology = depth_histogram(parts["topology read"]["by_parent_chain"])
    depths = sorted(set(moment) | set(topology))
    figure, (left, right) = plt.subplots(
        1, 2, figsize=(11.0, 4.4), gridspec_kw={"width_ratios": [1.7, 1.0]}
    )
    style(left)
    style(right)
    width = 0.4
    left.bar(
        [d - width / 2 for d in depths],
        [moment.get(d, 0) for d in depths],
        width,
        color=ACCENT,
        label="current-moment read (120 copies)",
    )
    left.bar(
        [d + width / 2 for d in depths],
        [topology.get(d, 0) for d in depths],
        width,
        color=NEUTRAL,
        label="topology read (74 copies)",
    )
    left.set_xlabel("chain depth (caller edges to the entry computation)", fontsize=8)
    left.set_ylabel("copies at that depth", fontsize=8)
    left.set_title("copies by enclosing-computation chain depth", fontsize=9, loc="left")
    left.set_xticks(depths)
    left.legend(frameon=False, fontsize=7.5, loc="upper left")
    labels = [name for name, _ in TOPOLOGY_FLAVOURS]
    values = [value for _, value in TOPOLOGY_FLAVOURS]
    right.bar(range(len(labels)), values, 0.62, color=ACCENT)
    for position, value in zip(range(len(labels)), values):
        right.text(position, value + 0.5, str(value), ha="center", fontsize=8)
    right.set_xticks(range(len(labels)))
    right.set_xticklabels(labels, fontsize=7.5, rotation=18, ha="right")
    right.set_ylabel("topology copies", fontsize=8)
    right.set_title("the 74 topology copies by trace flavour", fontsize=9, loc="left")
    right.set_ylim(0, max(values) * 1.2)
    figure.tight_layout()
    figure.savefig(OUTPUT, dpi=160)
    figure.savefig(OUTPUT.with_suffix(".svg"))
    (HERE / "read-request-decomposition.json").write_text(
        json.dumps(receipt(moment, topology), indent=1) + "\n", encoding="utf-8"
    )
    print("wrote", OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())