"""Draw where the 300-cell weak certificate program's instructions sit."""

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

receipt = json.load(open(sys.argv[1]))
rows = receipt["rows"]
total = receipt["optimized_instructions"]
kinds = {"primal": 0, "jvp": 0, "transpose": 0}
for row in rows:
    if not row["shared_body"]:
        for kind, count in row["kinds"].items():
            kinds[kind] += count
parts = [
    ("outside the live map", total - receipt["map_instructions"], "#9aa5b1"),
    ("shared request bodies", receipt["shared_body_instructions"], "#2a7fbd"),
    ("direct primal map calls", kinds["primal"], "#e0a030"),
    ("direct tangent (linearize/jvp)", kinds["jvp"], "#d0602a"),
    ("direct transpose", kinds["transpose"], "#8a3a9a"),
]
fig, ax = plt.subplots(figsize=(8, 2.6))
left = 0
for label, width, colour in parts:
    ax.barh(0, width, left=left, color=colour, label=f"{label}: {width:,}")
    left += width
ax.axvline(203078, color="black", ls="--", lw=1.2)
ax.text(203078, 0.47, " ceiling 203,078", va="bottom", fontsize=9)
ax.axvline(total, color="black", lw=0.6)
ax.text(total, -0.55, f"{total:,}", ha="right", va="top", fontsize=9)
ax.set_ylim(-0.8, 0.8)
ax.set_yticks([])
ax.set_xlabel("optimized HLO instructions, 300-cell weak certificate program")
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.legend(
    loc="upper center", bbox_to_anchor=(0.5, -0.45), ncol=2, fontsize=8, frameon=False
)
fig.tight_layout()
for suffix in ("png", "svg"):
    fig.savefig(sys.argv[2] + "." + suffix, dpi=150, bbox_inches="tight")
print({label: width for label, width, _ in parts})
