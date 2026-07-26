"""Evidence figure for the partial-range primitives the finite arc needs."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from numpy.polynomial.legendre import leggauss

from nova.biot.arcamplitude import arc_limits
from nova.biot.completeelliptic import complete_kind
from nova.biot.incompleteelliptic import TRIPS, incomplete_kind
from nova.biot.incompletemoments import harmonic_moments

LONG_PI = np.longdouble("3.14159265358979323846264338327950288")
QUARTER = 0.5 * np.pi


def reference_first(co_amplitude, complement):
    """Longdouble first kind on a composite sinh map, as the test builds it."""
    complement = np.longdouble(complement)
    co_amplitude = np.longdouble(co_amplitude)
    root = np.sqrt(complement)
    if root == 0.0:
        return -np.log(np.tan(0.5 * co_amplitude))
    node, weight = (np.longdouble(term) for term in leggauss(60))
    lower, upper = np.arcsinh(co_amplitude / root), np.arcsinh(LONG_PI / (2 * root))
    panels = max(int(np.ceil(float(upper - lower))), 1)
    edge = lower + (upper - lower) * np.arange(panels + 1, dtype=np.longdouble) / panels
    width = 0.5 * (edge[1] - edge[0])
    stretch = edge[:-1, None] + width * (node[None, :] + 1.0)
    offset = root * np.sinh(stretch)
    modulus = np.sqrt(np.sin(offset) ** 2 + complement * np.cos(offset) ** 2)
    return width * ((root * np.cosh(stretch) / modulus) @ weight).sum()


def reference_plain(amplitude, complement, order, panels=1500, nodes=48):
    node, weight = leggauss(nodes)
    edge = np.linspace(0.0, amplitude, panels + 1)
    half = 0.5 * (edge[1] - edge[0])
    angle = edge[:-1, None] + half * (node[None, :] + 1.0)
    radical = np.sqrt(np.cos(angle) ** 2 + complement * np.sin(angle) ** 2)
    return (half * ((np.cos(2 * order * angle) / radical) @ weight)).sum()


fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
fig.subplots_adjust(wspace=0.30, left=0.055, right=0.985, top=0.86, bottom=0.145)

# ---------------------------------------------------------------- the fold
ax = axes[0]
separation = np.linspace(-4.0 * np.pi, 4.0 * np.pi, 4001)
lower, _ = arc_limits(separation, 0.0, 1.0)
amplitude = 0.5 * (np.pi + separation)
ax.plot(
    amplitude / np.pi,
    amplitude / np.pi,
    color="#bbbbbb",
    lw=1.0,
    ls="--",
    label="raw amplitude",
)
ax.plot(
    amplitude / np.pi,
    lower.amplitude / np.pi,
    color="#0a7d33",
    lw=1.8,
    label="folded amplitude",
)
ax.plot(
    amplitude / np.pi,
    0.25 * lower.turns,
    color="#1f4e9c",
    lw=1.3,
    label="half-turn count / 4",
)
ax.plot(
    amplitude / np.pi,
    0.35 * lower.parity,
    color="#b3541e",
    lw=1.0,
    label="parity × 0.35",
)
for case, edge in (("A", 0.5), ("B", 1.5)):
    ax.axvline(edge, color="#999999", lw=0.8, ls=":")
ax.text(0.22, 1.62, "case A", fontsize=9, color="#555555")
ax.text(0.85, 1.62, "case B", fontsize=9, color="#555555")
ax.text(1.72, 1.62, "case C", fontsize=9, color="#555555")
ax.set_xlim(-1.6, 2.3)
ax.set_ylim(-0.6, 1.85)
ax.set_xlabel(r"amplitude $\alpha/\pi$")
ax.set_ylabel(r"folded quantities ($/\pi$ where an angle)")
ax.set_title("One formula where the paper prints three cases", fontsize=11)
ax.legend(fontsize=8, loc="lower right", framealpha=0.9)

# ------------------------------------------------- first kind accuracy map
ax = axes[1]
co_amplitudes = np.array(
    [1.4, 1.0, 0.6, 0.3, 1e-1, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14]
)
complements = np.array(
    [
        1e-2,
        1e-4,
        1e-6,
        1e-9,
        1e-12,
        1e-16,
        1e-20,
        1e-25,
        1e-30,
        1e-40,
        1e-80,
        1e-160,
        1e-300,
    ]
)
error = np.zeros((len(co_amplitudes), len(complements)))
for row, co in enumerate(co_amplitudes):
    sine, cosine = np.cos(co), np.sin(co)
    for column, complement in enumerate(complements):
        got = float(
            incomplete_kind(QUARTER - co, complement, sine=sine, cosine=cosine)[0]
        )
        want = float(reference_first(co, complement))
        error[row, column] = max(abs(got - want) / abs(want), 1e-17)
mesh = ax.pcolormesh(
    np.arange(len(complements) + 1),
    np.arange(len(co_amplitudes) + 1),
    error,
    norm=LogNorm(vmin=1e-16, vmax=1e-3),
    cmap="viridis",
    shading="flat",
)
ax.set_xticks(np.arange(len(complements)) + 0.5)
ax.set_xticklabels(
    [f"{v:.0e}".replace("e-0", "e-") for v in complements], rotation=90, fontsize=7
)
ax.set_yticks(np.arange(len(co_amplitudes)) + 0.5)
ax.set_yticklabels(
    [f"{v:g}" if v >= 0.1 else f"{v:.0e}".replace("e-0", "e-") for v in co_amplitudes],
    fontsize=7,
)
ax.set_xlabel(r"modulus complement $k'^2$")
ax.set_ylabel(r"$\pi/2 - \alpha$  (half-separation from the arc end)")
ax.set_title("First kind: machine-exact but for one measured corner", fontsize=11)
plt.colorbar(mesh, ax=ax, label="relative error", pad=0.02)

# ------------------------------------------ the moment family's two directions
ax = axes[2]
co_amplitude = 0.2
amplitude = QUARTER - co_amplitude
sine, cosine = np.cos(co_amplitude), np.sin(co_amplitude)
complements = np.logspace(-16, np.log10(0.6), 40)
routes = {
    "tridiagonal solve alone": 0.0,
    "upward recursion alone": 1.0,
    "the pair, switching at 0.99": None,
}
colours = {
    "tridiagonal solve alone": "#1f4e9c",
    "upward recursion alone": "#b3541e",
    "the pair, switching at 0.99": "#0a7d33",
}
truth = {}
for complement in complements:
    truth[complement] = [
        reference_plain(amplitude, complement, order) for order in range(10)
    ]
for label, switch in routes.items():
    worst = []
    for complement in complements:
        kwargs = {} if switch is None else {"switch": switch}
        got = harmonic_moments(
            amplitude,
            1.0 - complement,
            10,
            complement=complement,
            sine=sine,
            cosine=cosine,
            **kwargs,
        )
        scale = abs(float(got[0]))
        worst.append(
            max(
                max(abs(float(one) - other) / scale, 1e-17)
                for one, other in zip(got, truth[complement])
            )
        )
    ax.loglog(
        complements,
        worst,
        color=colours[label],
        lw=1.8 if switch is None else 1.2,
        label=label,
    )
ax.axvline(1.0 - 0.99, color="#999999", lw=0.9, ls=":")
ax.text(1.05e-2, 2e-6, "switch", fontsize=8, color="#555555", rotation=90)
ax.axhline(5e-14, color="#cccccc", lw=0.9, ls="--")
ax.text(2e-16, 7e-14, "asserted bound 5e-14", fontsize=8, color="#777777")
ax.set_xlabel(r"modulus complement $k'^2$")
ax.set_ylabel("worst error over ten orders, relative to $P_0$")
ax.set_title("Partial-range moments: neither direction spans the range", fontsize=11)
ax.set_ylim(1e-17, 1e2)
ax.legend(fontsize=8, loc="upper right", framealpha=0.9)

fig.suptitle(
    "What a finite arc needs that the full turn never formed — "
    f"complement-native, fixed at {TRIPS} trips, and traceable throughout",
    fontsize=12.5,
    y=0.965,
)
out = (
    "/home/ITER/mcintos/Code/nova/docs/figures/polybow-arc-section/"
    "partial_range_primitives.png"
)
fig.savefig(out, dpi=150)
print("wrote", out)
print("quarter turn vs complete_kind, worst over the complement sweep:")
worst = 0.0
for complement in complements:
    got = float(incomplete_kind(QUARTER, complement, sine=1.0, cosine=0.0)[0])
    want = float(complete_kind(np.asarray(complement))[0])
    worst = max(worst, abs(got - want) / want)
print(f"   {worst:.2e}")
