"""Evidence figure for the sine-weighted moment family an arc's azimuthal rows need."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from numpy.polynomial.legendre import leggauss

from nova.biot.incompletemoments import (
    sine_cn_pole_moment,
    sine_moments,
    sine_sn_pole_moment,
    sn_pole_moment,
)

QUARTER = 0.5 * np.pi


def graded(amplitude, integrand, levels=60, ratio=0.6, nodes=48):
    """Return the integral with panels graded geometrically into both ends."""
    if amplitude == 0.0:
        return 0.0
    half = 0.5 * amplitude
    tail = half * ratio ** np.arange(levels, 0, -1)
    lower = np.concatenate([[0.0], tail, [half]])
    edge = np.concatenate([lower, amplitude - lower[-2::-1]])
    node, weight = leggauss(nodes)
    span = 0.5 * np.diff(edge)[:, None]
    angle = edge[:-1, None] + span * (node[None, :] + 1.0)
    return float((span * integrand(angle) @ weight).sum())


def radical(angle, complement):
    return np.sqrt(np.cos(angle) ** 2 + complement * np.sin(angle) ** 2)


def reference_seed(co_amplitude, complement, shift, mirrored):
    base = np.sin if mirrored else np.cos
    return graded(
        QUARTER - co_amplitude,
        lambda a: np.sin(2 * a) / ((base(a) ** 2 + shift) * radical(a, complement)),
    )


def reference_moment(co_amplitude, complement, order):
    return graded(
        QUARTER - co_amplitude,
        lambda a: np.sin(2 * a) * np.cos(2 * order * a) / radical(a, complement),
    )


def seed(co_amplitude, complement, shift, mirrored):
    routine = sine_sn_pole_moment if mirrored else sine_cn_pole_moment
    return float(
        routine(
            np.asarray(shift),
            1.0 - complement,
            np.cos(co_amplitude),
            np.sin(co_amplitude),
            complement=np.asarray(complement),
        )
    )


def subtracted_seed(co_amplitude, complement, shift, mirrored):
    """The same seed with ``1 - z`` formed by subtraction rather than supplied.

    What the route costs if the gap the inverse hyperbolic tangent needs is
    taken as one less a ratio: the ratio reaches one as the root reaches the
    range end, so the relative error tracks eps over the gap.
    """
    sine, cosine = np.cos(co_amplitude), np.sin(co_amplitude)
    near, far = sine * sine, cosine * cosine
    parameter = 1.0 - complement
    edge = np.sqrt(complement + parameter * far)
    if mirrored:
        pivot = near + (1.0 + edge) * shift
        root = np.sqrt(1.0 + shift * parameter)
    else:
        pivot = edge * near + (1.0 + edge) * (far + shift)
        partner = complement - shift * parameter
        if partner <= 0.0:
            return seed(co_amplitude, complement, shift, mirrored)
        root = np.sqrt(partner)
    argument = root * near / pivot
    if argument >= 1.0 or argument <= 0.0:
        return np.nan
    return 2.0 * near / pivot * np.arctanh(argument) / argument


def main():
    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.3))

    # LEFT -- the seed's gap, supplied against subtracted
    shifts = np.geomspace(1e-14, 1.0, 60)
    for mirrored, style, label in (
        (True, "-", "root past the near end (sin²)"),
        (False, "--", "root past the far end (cos²)"),
    ):
        supplied, subtracted = [], []
        for shift in shifts:
            want = reference_seed(0.5, 0.2, shift, mirrored)
            supplied.append(abs(seed(0.5, 0.2, shift, mirrored) - want) / abs(want))
            other = subtracted_seed(0.5, 0.2, shift, mirrored)
            subtracted.append(abs(other - want) / abs(want))
        axes[0].loglog(shifts, np.maximum(supplied, 1e-18), style, color="C0")
        axes[0].loglog(shifts, np.maximum(subtracted, 1e-18), style, color="C3")
    axes[0].loglog(
        shifts, 2.2e-16 / shifts, ":", color="0.5", label=r"$\epsilon$/shift"
    )
    axes[0].set_xlabel("shift — the root's distance past the range end")
    axes[0].set_ylabel("relative error against graded quadrature")
    axes[0].set_title("The gap has to be supplied, not subtracted")
    axes[0].plot([], [], "-", color="C0", label="gap in closed form")
    axes[0].plot([], [], "-", color="C3", label="gap as one less a ratio")
    axes[0].set_ylim(1e-18, 1e2)
    axes[0].legend(fontsize=8, loc="upper right")

    # MIDDLE -- the family over amplitude and complement
    co_amplitudes = np.linspace(0.0, 1.5, 24)
    complements = np.geomspace(1e-16, 0.99, 24)
    grid = np.zeros((complements.size, co_amplitudes.size))
    for row, complement in enumerate(complements):
        for column, co_amplitude in enumerate(co_amplitudes):
            got = sine_moments(
                QUARTER - co_amplitude,
                1.0 - complement,
                8,
                complement=complement,
                sine=np.cos(co_amplitude),
                cosine=np.sin(co_amplitude),
            )
            scale = abs(float(got[0]))
            worst = max(
                abs(
                    float(got[order])
                    - reference_moment(co_amplitude, complement, order)
                )
                for order in range(8)
            )
            grid[row, column] = worst / max(scale, 1e-30)
    mesh = axes[1].pcolormesh(
        co_amplitudes,
        complements,
        np.maximum(grid, 1e-18),
        norm=LogNorm(1e-17, 1e-12),
        cmap="viridis",
        shading="auto",
    )
    axes[1].set_yscale("log")
    axes[1].set_xlabel("quarter turn less the amplitude")
    axes[1].set_ylabel("modulus complement $k'^2$")
    axes[1].set_title("Eight orders, relative to the zeroth moment")
    figure.colorbar(mesh, ax=axes[1])

    # RIGHT -- what the weight buys: elementary against the third kind
    poles = np.geomspace(1e-6, 1e6, 40)
    sine_value, cosine_value = np.cos(0.5), np.sin(0.5)
    elementary = [abs(seed(0.5, 1e-6, shift, True)) for shift in poles]
    third = [
        abs(
            float(
                sn_pole_moment(
                    np.asarray(shift),
                    1.0 - 1e-6,
                    sine_value,
                    cosine_value,
                    complement=np.asarray(1e-6),
                )
            )
        )
        for shift in poles
    ]
    axes[2].loglog(poles, elementary, "-", color="C0", label="sine weight — elementary")
    axes[2].loglog(poles, third, "--", color="C1", label="cosine weight — third kind")
    axes[2].set_xlabel("shift")
    axes[2].set_ylabel("seed value")
    axes[2].set_title("Same seed, and only one of them is a special function")
    axes[2].legend(fontsize=8)

    figure.tight_layout()
    figure.savefig("docs/figures/polybow-arc-section/sine_moment_family.png", dpi=140)
    print("worst family error", grid.max())


if __name__ == "__main__":
    main()
