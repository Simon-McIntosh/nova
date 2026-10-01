"""Per-trip live relative residual of the diverted rung, one line per run."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

receipt = json.loads(Path(sys.argv[1]).read_text())
output = Path(sys.argv[2])
traces = dict(receipt["per_trip_live_relative_residual"])
traces.update(receipt.get("premerge_per_trip_live_relative_residual", {}))
label = {
    "slice1": "capacity scan, merged operator (e62c8110e solver)",
    "exit": "exit loop, merged operator (63a32bb4a)",
    "premerge-slice1": "capacity scan, pre-merge operator (8d02dd0f)",
    "premerge-exit": "exit loop, pre-merge operator",
}
style = {
    "slice1": dict(color="#1f5fa8", marker="o", lw=2.5),
    "exit": dict(color="#e0892b", marker="x", lw=1.2, ls="--"),
    "premerge-slice1": dict(color="#5a5a5a", marker="s", lw=2.5),
    "premerge-exit": dict(color="#c23b3b", marker="+", lw=1.2, ls="--"),
}
figure, axis = plt.subplots(figsize=(7.0, 4.2))
for name, values in traces.items():
    axis.semilogy(range(1, len(values) + 1), values, label=label[name], **style[name])
axis.set_xlabel("active-set trip")
axis.set_ylabel("live relative residual")
axis.set_title("Diverted certificate rung, 300 cells: per-trip residual")
for spine in ("top", "right"):
    axis.spines[spine].set_visible(False)
axis.legend(frameon=False, fontsize=8)
figure.tight_layout()
for suffix in (".png", ".svg"):
    figure.savefig(output.with_suffix(suffix), dpi=150)
