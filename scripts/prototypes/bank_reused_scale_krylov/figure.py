"""Draw one product's device time against one assembly's, in counts of an assembly.

The decision this figure serves: a later Newton update needs about forty
products, so the Krylov route pays only where one product costs a fraction of
one assembly. Each row is an arm from ``measure.py``; the two bars are the
device time of one call, and the annotation gives what forty products would
cost in assemblies, which is the quantity the decision turns on.
"""

import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

# (label, assembly device ms, product device ms) — medians over the two fresh
# processes of each arm, from measure.py's own trace figures.
ARMS = [
    ("16 slices, fused", 36.47, 107.35),
    ("16 slices, lane-sequential", 148.71, 184.32),
    ("128 slices, lane-sequential", 1189.72, 1484.09),
]
ASSEMBLY = "#929292"
PRODUCT = "#2e6f9e"

plt.rcParams.update(
    {
        "font.size": 14,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "axes.linewidth": 1.2,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "svg.fonttype": "none",
    }
)

figure, axes = plt.subplots(figsize=(14.0, 5.2), layout="constrained")
positions = range(len(ARMS))
height = 0.36
assembly = [row[1] for row in ARMS]
product = [row[2] for row in ARMS]

axes.barh(
    [p + height / 2 for p in positions],
    assembly,
    height=height,
    color=ASSEMBLY,
    label="one assembly",
    zorder=2,
)
axes.barh(
    [p - height / 2 for p in positions],
    product,
    height=height,
    color=PRODUCT,
    label="one product",
    zorder=3,
)

for position, (_, assembled, applied) in zip(positions, ARMS):
    axes.text(
        assembled * 1.12,
        position + height / 2,
        f"{assembled:.0f} ms",
        va="center",
        ha="left",
        color="#3a3a3a",
        fontsize=14,
    )
    axes.text(
        applied * 1.12,
        position - height / 2,
        f"{applied:.0f} ms  ·  40 products ≈ {40 * applied / assembled:.0f} assemblies",
        va="center",
        ha="left",
        color=PRODUCT,
        fontsize=14,
    )

axes.set_xscale("log")
axes.set_xlim(20, 40000)
axes.set_yticks(list(positions))
axes.set_yticklabels([row[0] for row in ARMS])
axes.set_xlabel("device time per call [ms]")
axes.invert_yaxis()
for side in ("top", "right"):
    axes.spines[side].set_visible(False)
axes.legend(frameon=False, loc="lower right")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
TARGET = os.path.join(
    ROOT,
    "docs",
    "figures",
    "sub-millisecond-forward-path",
    "bank-reused-scale-krylov",
    "break-even.svg",
)
os.makedirs(os.path.dirname(TARGET), exist_ok=True)
figure.savefig(TARGET)
print(TARGET)