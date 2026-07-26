"""Evidence and cost for the incomplete third kind the arc's pole seeds need.

Two modes, because they answer different questions and must not share a process:

``figure`` (the default)
    the three panels the rung turns on -- what the printed symmetric form costs at
    a near pole, where the reflected route holds against the extended-precision
    integral, and the two smallnesses that bound the corner it does not.

``cost <variant>``
    per-element cost of ONE primitive, so each runs in its own interpreter.  A
    second measurement in the same process is some 40 % faster -- allocator and
    branch-predictor warmup, not caching -- so a comparison taken that way invents
    a difference that is not there.  Run on a compute node; login-node timing has
    a five-fold spread.
"""

import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from numpy.polynomial.legendre import leggauss  # noqa: E402
from scipy.special import elliprf, elliprj  # noqa: E402

from nova.biot.completeelliptic import complete_kind, complete_pole  # noqa: E402
from nova.biot.incompleteelliptic import incomplete_kind, incomplete_pole  # noqa: E402

LONG_PI = np.longdouble("3.14159265358979323846264338327950288")
ELEMENTS = 4096


def reference_pole(co_amplitude, pole, complement, nodes=60):
    """Longdouble pole integral on two sinh maps, as the test builds it."""
    pole = np.longdouble(pole)
    complement = np.longdouble(complement)
    co_amplitude = np.longdouble(co_amplitude)
    node, weight = (np.longdouble(term) for term in leggauss(nodes))

    def integrand(cosine, sine):
        return 1.0 / (
            (cosine**2 + pole * sine**2) * np.sqrt(cosine**2 + complement * sine**2)
        )

    def graded(lower, top, layer, flipped):
        low, high = np.arcsinh(lower / layer), np.arcsinh(top / layer)
        panels = max(int(np.ceil(float(high - low))), 1)
        edge = low + (high - low) * np.arange(panels + 1, dtype=np.longdouble) / panels
        half = 0.5 * (edge[1] - edge[0])
        stretch = edge[:-1, None] + half * (node[None, :] + 1.0)
        angle = layer * np.sinh(stretch)
        pair = (np.sin(angle), np.cos(angle))
        if not flipped:
            pair = pair[::-1]
        return half * (layer * np.cosh(stretch) * integrand(*pair) @ weight).sum()

    half_amplitude = 0.5 * (LONG_PI / 2 - co_amplitude)
    return float(
        graded(np.longdouble(0.0), half_amplitude, 1.0 / np.sqrt(1.0 + pole), False)
        + graded(
            co_amplitude,
            LONG_PI / 4 + 0.5 * co_amplitude,
            np.sqrt(complement * pole / (pole + complement)),
            True,
        )
    )


def printed(co_amplitude, pole, complement):
    """The symmetric forms taken as printed: ``F + (n/3) sin^3 phi R_J``."""
    sine, cosine = np.cos(co_amplitude), np.sin(co_amplitude)
    squared_cosine = cosine * cosine
    radical = squared_cosine + complement * sine * sine
    return sine * elliprf(squared_cosine, radical, 1.0) + (
        (1.0 - pole) / 3.0
    ) * sine**3 * elliprj(
        squared_cosine, radical, 1.0, squared_cosine + pole * sine * sine
    )


def variants(count):
    """Return the primitives to cost, each over the same random arguments."""
    rng = np.random.default_rng(7)
    pole = 10.0 ** rng.uniform(-6.0, 6.0, count)
    complement = 10.0 ** rng.uniform(-12.0, 0.0, count)
    co_amplitude = rng.uniform(0.0, 1.5, count)
    sine, cosine = np.cos(co_amplitude), np.sin(co_amplitude)
    amplitude = 0.5 * np.pi - co_amplitude
    return {
        "complete-kind": lambda: complete_kind(complement),
        "complete-pole": lambda: complete_pole(pole, complement),
        "incomplete-kind": lambda: incomplete_kind(
            amplitude, complement, sine=sine, cosine=cosine
        ),
        "incomplete-pole": lambda: incomplete_pole(pole, complement, sine, cosine),
    }


def cost(name, count=ELEMENTS, repeats=5):
    """Print microseconds per element for one variant, median of ``repeats``."""
    call = variants(count)[name]
    call()  # one warm pass, so the measurement is of the arithmetic
    taken = []
    for _ in range(repeats):
        start = time.perf_counter()
        call()
        taken.append(time.perf_counter() - start)
    print(f"{name:20s} {1e6 * float(np.median(taken)) / count:8.4f} us/element")


def figure():
    """Write the rung's three-panel evidence figure."""
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.0))
    fig.subplots_adjust(wspace=0.32, left=0.06, right=0.985, top=0.80, bottom=0.145)

    # ------------------------------------------- what the printed form costs
    ax = axes[0]
    complement, co_amplitude = 1e-3, 0.7
    poles = 10.0 ** np.linspace(0.3, 13.0, 60)
    straight, reflected = [], []
    for pole in poles:
        want = reference_pole(co_amplitude, pole, complement)
        straight.append(abs(printed(co_amplitude, pole, complement) - want) / abs(want))
        got = float(
            incomplete_pole(
                np.asarray(pole),
                np.asarray(complement),
                np.cos(co_amplitude),
                np.sin(co_amplitude),
            )
        )
        reflected.append(abs(got - want) / abs(want))
    ax.loglog(poles, np.maximum(straight, 1e-17), color="#b2182b", lw=1.6,
              label=r"printed: $F+\frac{n}{3}\sin^3\!\phi\,R_J$")
    ax.loglog(poles, np.maximum(reflected, 1e-17), color="#2166ac", lw=1.6,
              label=r"reflected onto $k'^2/p$")
    ax.loglog(poles, 2.2e-16 * np.sqrt(poles), color="#999999", lw=1.0, ls="--",
              label=r"$\epsilon\sqrt{p}$")
    ax.set_xlabel(r"pole $p$  (denominator at the far end of the range)")
    ax.set_ylabel("relative error against the extended-precision integral")
    ax.set_title("A near root costs the printed form half the pole's decades")
    ax.legend(fontsize=8.5, loc="upper left")
    ax.grid(alpha=0.25, which="both")

    # -------------------------------------------------- where the route holds
    ax = axes[1]
    pole_grid = 10.0 ** np.linspace(-12.0, 12.0, 49)
    complement_grid = 10.0 ** np.linspace(-300.0, 0.0, 41)
    error = np.zeros((complement_grid.size, pole_grid.size))
    for row, complement in enumerate(complement_grid):
        for column, pole in enumerate(pole_grid):
            want = reference_pole(0.7, pole, complement)
            got = float(
                incomplete_pole(
                    np.asarray(pole), np.asarray(complement), np.cos(0.7), np.sin(0.7)
                )
            )
            error[row, column] = max(abs(got - want) / abs(want), 1e-17)
    mesh = ax.pcolormesh(
        pole_grid, complement_grid, error, norm=LogNorm(1e-17, 1e-14), cmap="viridis",
        shading="nearest",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axvline(1.0, color="#ffffff", lw=1.0, ls="--")
    ax.text(2e2, 3e-290, "reflected", color="#ffffff", fontsize=9.5,
            horizontalalignment="center")
    ax.text(2e-6, 3e-290, "as printed", color="#ffffff", fontsize=9.5,
            horizontalalignment="center")
    ax.set_xlabel(r"pole $p$")
    ax.set_ylabel(r"modulus complement $k'^2$")
    ax.set_title(f"Both orientations, both double ranges: worst {error.max():.1e}")
    fig.colorbar(mesh, ax=ax, label="relative error")

    # ------------------------------------------------- the corner it does not
    ax = axes[2]
    for pole, colour in ((1e6, "#2166ac"), (1e12, "#b2182b"), (1e18, "#1b7837")):
        complements = 10.0 ** np.linspace(-315.0, -280.0, 71)
        got = np.array(
            [
                abs(
                    float(
                        incomplete_pole(
                            np.asarray(pole), np.asarray(complement), 1.0, 0.0
                        )
                    )
                    / float(complete_pole(np.asarray(pole), np.asarray(complement)))
                    - 1.0
                )
                for complement in complements
            ]
        )
        ax.loglog(complements / pole, np.maximum(got, 1e-17), color=colour, lw=1.6,
                  label=f"$p = ${pole:.0e}")
    ax.axvline(2.2250738585072014e-308, color="#333333", lw=1.2, ls="--")
    ax.text(1.4e-308, 3e-7, "smallest normal double", fontsize=8.5, rotation=90,
            verticalalignment="center", horizontalalignment="right")
    ax.set_xlabel(r"reflected pole $k'^2/p$")
    ax.set_ylabel("relative error against the complete routine at a quarter turn")
    ax.set_title("The one corner: the reflected pole goes denormal")
    ax.legend(fontsize=8.5, loc="upper right")
    ax.grid(alpha=0.25, which="both")

    fig.suptitle(
        "The one primitive Bartky's rearrangement cannot carry, and the "
        "reflection that makes the symmetric forms hold at a near root",
        fontsize=12.5,
        y=0.975,
    )
    # the plan docs are served from the primary checkout's working tree
    path = (
        "/home/ITER/mcintos/Code/nova/docs/figures/polybow-arc-section/"
        "incomplete_third_kind.png"
    )
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "cost":
        cost(sys.argv[2])
    else:
        figure()
