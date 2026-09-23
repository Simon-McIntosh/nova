"""Plot memory-map count against elapsed time for the fresh-process arms."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
CEILING = 65530
ARMS = sys.argv[1:] or [
    "whole-file",
    "whole-file-no-persistent-cache",
    "tests-before",
    "certificate-then-abort",
    "abort-alone",
]

fig, ax = plt.subplots(figsize=(10, 4.2))
for name in ARMS:
    rows = [
        line.split()
        for line in (HERE / f"{name}.memory.txt").read_text().splitlines()[1:]
    ]
    rows = [row for row in rows if row[3] not in ("na", "0")]
    start = float(rows[0][0])
    ax.plot(
        [(float(row[0]) - start) / 60 for row in rows],
        [int(row[3]) for row in rows],
        lw=1.2,
        label=name,
    )
ax.axhline(CEILING, color="k", ls="--", lw=0.8)
ax.text(0.5, CEILING + 600, "vm.max_map_count 65530", fontsize=8)
ax.set_xlabel("elapsed (min)")
ax.set_ylabel("entries in /proc/<pid>/maps")
ax.set_ylim(0, 70000)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
ax.set_title("tests/test_reduced_newton.py, fresh processes at main HEAD", fontsize=9)
fig.tight_layout()
stem = "map-count-bisect" if sys.argv[1:] else "map-count"
fig.savefig(HERE / f"{stem}.png", dpi=130)
fig.savefig(HERE / f"{stem}.svg")
