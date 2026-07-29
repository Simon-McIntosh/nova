"""Evidence that the rectangle kernel is continuous across its own corner planes.

The corner antiderivative of the rectangular-section ring carries an arctangent
boundary term whose limit at either end of the angle range is a SIGNED right angle,
so it changes sign across each of the section's four corner planes -- the two levels
``gamma = zs - z = 0`` and the two radii ``rs = r``.  That sign change is not a
discontinuity: the ring denominator ``gamma^2 + r^2 sin^2 phi`` has a root just past
each end of the range at a distance falling as ``gamma^2``, and the two complete
third-kind integrals over those roots diverge as ``1/|gamma|`` against numerators
vanishing as ``gamma``.  Their one-sided limits are ``+/- pi r^2/6`` and
``+/- sign(rs - r) pi r^2/6``, which is exactly minus the boundary term's own
``-(1 + sign(rs - r)) pi r^2/6``.  The three sum to zero from either side.

They only sum to zero in FLOATING POINT if each pole comes from the geometry.  This
script draws both arrangements of the same algebra:

* **subtracted** -- the characteristics as the paper prints them, ``2r/(r - c)`` and
  ``2r/(r + c)`` with ``c = sqrt(gamma^2 + r^2)``, the modulus complement as
  ``1 - 4 r rs/a^2``, and the signs taken through a dead-band.  Every one of those
  is a subtraction of near-equal quantities whose result is the very scale the
  cancellation runs at, so the branch terms are left with a relative error of order
  ``eps (r + c) r/gamma^2`` -- which passes one at a micrometre and then grows
  without bound.
* **complement** -- each pole as its exact square, ``((r + c)/gamma)^2``,
  ``(gamma/(r + c))^2`` and ``((rs - r)/b)^2``, the modulus complement as
  ``(gamma^2 + (rs - r)^2)/a^2``, the far pole's divergent characteristic never
  formed at all (one power of ``gamma`` is taken off the numerators and carried
  against it, leaving a bounded product), and ``rs - c`` as ``(rs - r)`` less
  ``gamma^2/(r + c)``.  This is what :mod:`nova.biot.greens` evaluates.

**Top row -- the crossing itself.**  ``psi``, ``B_R`` and ``B_Z`` scanned across the
lower face at a fixed radius inside the section's radial span, over three
micrometres either side, with the polygon oracle overlaid.  The subtracted
arrangement leaves a step of the full ``pi r^2/3`` in the antiderivative: the flux
crosses from ``+1.7e-04`` to ``-1.2e-04`` about a true value of ``+2.2e-05``, so it
is not merely inaccurate, it changes sign.  ``B_R`` is untouched, its own pole
numerator carrying ``gamma^2`` where the flux's carries ``gamma``.

**Bottom row -- the four corner planes and the corner line.**  Worst relative
deviation from the oracle over a ladder of standoffs spanning nine decades.  A jump
shows as a curve that RISES as the plane is approached, one decade per decade, since
the deviation is a constant against a quantity that stops changing; the fix shows as
a flat line on the oracle's own round-off.  The last panel slides along ``rs = r``
through a corner, where the modulus complement, the near pole and the third pole
reach zero together -- the configuration a plasma grid produces whenever a row lines
up with a cell's corner.

The oracle is the fully analytic polygon reduction
(:func:`nova.biot.polygonanalytic.polygon_analytic_greens`), not the quadrature one:
a target ON a face sits on the quadrature integrand's own boundary layer, where the
shipped 16x48 rule is six decades off in ``B_R`` and cannot referee its own singular
configuration.  Off the plane the two polygon kernels agree to 2e-09, which
``tests/test_biotgreens.py`` pins.

Run as ``python benchmarks/section_face_continuity.py [output.png]``.
"""

from __future__ import annotations

import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.special

from nova.biot.greens import MU0, corner_fields
from nova.biot.polygonanalytic import polygon_analytic_greens
from nova.biot.zeta import zeta

# The section the defect was found on: an ITER-scale major radius and a
# centimetre-scale cross-section, which is where the squared aspect ratio the poles
# sit at is small enough for the subtraction to lose everything.
RADIUS, LEVEL, WIDTH, HEIGHT = 4.0, 0.0, 0.1, 0.08
# a target radius strictly inside the radial span and off both corner radii, so the
# two corners on a face straddle it and the boundary term is live at one of them
INSIDE_SPAN = RADIUS - 0.0435
# a radius outside the span, where the boundary term is live at BOTH face corners
OUTSIDE_SPAN = RADIUS + 0.021
# where the corner-plane scans are read, off the section's mid-plane so no component
# is passing through zero
OFF_MIDPLANE = LEVEL + 0.017

LADDER = 10.0 ** -np.arange(2.0, 11.0)
CROSSING = np.linspace(-3e-6, 3e-6, 121)
COMPONENTS = (r"$\psi$  [Wb/A]", r"$B_R$  [T/A]", r"$B_Z$  [T/A]")
DEAD_BAND = 1e4 * 2.0 * np.finfo(float).eps


def rectangle() -> np.ndarray:
    """Return the section's four ``(r, z)`` corners, counter-clockwise."""
    half_width, half_height = WIDTH / 2.0, HEIGHT / 2.0
    return np.array(
        [
            [RADIUS - half_width, LEVEL - half_height],
            [RADIUS + half_width, LEVEL - half_height],
            [RADIUS + half_width, LEVEL + half_height],
            [RADIUS - half_width, LEVEL + half_height],
        ]
    )


def subtracted_corner_fields(rs, zs, r, z):
    """Return the corner antiderivative with every pole reached by subtraction.

    The paper's own printed arrangement, kept here as the thing the fix is measured
    against: the characteristics as ``2r/(r -/+ c)``, the modulus handed to the
    elliptic integrals as the parameter rather than its complement, ``rs - c`` taken
    as written, and the two signs passed through a dead-band.  The ``eps`` offsets
    on the first characteristic and on the parameter are what kept the divisions and
    ``ellipk`` in range; they are part of what is being measured.
    """
    eps = 2.0 * np.finfo(float).eps
    gamma = zs - z
    a2 = gamma**2 + (rs + r) ** 2
    a = np.sqrt(a2)
    b = rs + r
    c2 = gamma**2 + r**2
    c = np.sqrt(c2)
    k2 = (1.0 - eps) * 4.0 * r * rs / a2
    v = 1.0 + k2 * (gamma**2 - b * r) / (2.0 * r * rs)
    ellip_k = scipy.special.ellipk(k2)
    ellip_e = scipy.special.ellipe(k2)
    u_coef = k2 * (4.0 * gamma**2 + 3.0 * rs**2 - 5.0 * r**2) / (4.0 * r)

    np2 = {
        1: 2.0 * r / (r - c - eps),
        2: (1.0 - eps) * 2.0 * r / (r + c),
        3: (1.0 - eps) * 4.0 * r * rs / b**2,
    }

    def third_kind(n, m):
        """Carlson's ordinary arrangement, ``R_F + (n/3) R_J``."""
        x = np.zeros_like(n)
        y = 1.0 - m
        one = np.ones_like(n)
        return (
            scipy.special.elliprf(x, y, one)
            + scipy.special.elliprj(x, y, one, 1.0 - n) * n / 3.0
        )

    pi3 = {p: third_kind(np2[p], k2) for p in (1, 2, 3)}
    qr = {p: (rs - (-1.0) ** p * c) * np2[p] * gamma**2 * c / r for p in (1, 2)}
    qr[3] = np.zeros_like(r)
    qz = {p: (rs - (-1.0) ** p * c) * -2.0 * gamma * c * np2[p] for p in (1, 2)}
    qz[3] = gamma * b * (rs - r) * np2[3]
    pphi = {
        p: (rs - (-1.0) ** p * c) * np2[p] * c * (3.0 * r**2 - c2) / (2.0 * r)
        for p in (1, 2)
    }
    pphi[3] = -rs / b * (rs - r) * (3.0 * r**2 - rs**2)

    def p_sum(coef):
        return sum((-1.0) ** p * coef[p] * pi3[p] for p in (1, 2, 3))

    def banded(x):
        return np.where(np.abs(x) > DEAD_BAND, np.sign(x), 0.0)

    cphi = -1.0 / 3.0 * r**2 * np.pi / 2.0 * banded(gamma) * (banded(rs - r) + 1.0)
    zt = zeta(rs, r, gamma, (np.pi / 2.0) * np.ones_like(rs))
    return (
        cphi
        + gamma * r * zt
        + gamma * a / (6.0 * r) * (u_coef * ellip_k - 2.0 * rs * ellip_e)
        + gamma / (6.0 * a * r) * p_sum(pphi),
        r * zt
        - a / (2.0 * r) * rs * (ellip_e - v * ellip_k)
        - 1.0 / (4.0 * a * r) * p_sum(qr),
        3.0 / r * cphi
        + 2.0 * gamma * zt
        - a / (2.0 * r) * 1.5 * gamma * k2 * ellip_k
        - 1.0 / (4.0 * a * r) * p_sum(qz),
    )


def assemble(antiderivative, target_r, target_z):
    """Return ``(psi, B_R, B_Z)`` from one corner antiderivative, by the corner rule.

    The same combination :func:`nova.biot.greens.cylinder_greens` applies, spelled
    here so both arrangements go through one assembly and the only difference
    between the two curves is the antiderivative itself.
    """
    target_r = np.asarray(target_r, dtype=np.float64)
    target_z = np.asarray(target_z, dtype=np.float64)
    rs = np.stack(
        [
            np.full(target_r.shape, RADIUS + side * WIDTH / 2.0)
            for side in (-1, 1, 1, -1)
        ],
        axis=-1,
    )
    zs = np.stack(
        [
            np.full(target_r.shape, LEVEL + side * HEIGHT / 2.0)
            for side in (-1, -1, 1, 1)
        ],
        axis=-1,
    )
    stacks = (
        rs,
        zs,
        np.repeat(target_r[..., None], 4, axis=-1),
        np.repeat(target_z[..., None], 4, axis=-1),
    )
    aphi, br, bz = antiderivative(*stacks)
    area = WIDTH * HEIGHT

    def corner(data):
        return ((data[..., 2] - data[..., 3]) - (data[..., 1] - data[..., 0])) / (
            2.0 * np.pi * area
        )

    return (
        2.0 * np.pi * MU0 * target_r * corner(aphi),
        MU0 * corner(br),
        MU0 * corner(bz),
    )


def deviation(target_r, target_z):
    """Return each arrangement's worst relative deviation from the polygon oracle.

    The field components are scaled by the field MAGNITUDE rather than each by
    itself: ``B_R`` passes through zero on the section's mid-plane, where a
    per-component relative measure has no meaning.
    """
    exact = polygon_analytic_greens(target_r, target_z, rectangle())
    magnitude = np.hypot(exact[1], exact[2])
    local = (np.abs(exact[0]), magnitude, magnitude)
    out = {}
    for name, antiderivative in (
        ("subtracted", subtracted_corner_fields),
        ("complement", corner_fields),
    ):
        got = assemble(antiderivative, target_r, target_z)
        out[name] = np.max(
            [
                np.abs(one - other) / scale
                for one, other, scale in zip(got, exact, local)
            ],
            axis=0,
        )
    return out


def crossing(axes, index):
    """Draw one component across the lower face, both arrangements and the oracle."""
    level = LEVEL - HEIGHT / 2.0
    target_r = np.full(CROSSING.shape, INSIDE_SPAN)
    target_z = level + CROSSING
    exact = polygon_analytic_greens(target_r, target_z, rectangle())[index]
    subtracted = assemble(subtracted_corner_fields, target_r, target_z)[index]
    shipped = assemble(corner_fields, target_r, target_z)[index]
    micron = 1e6 * CROSSING
    axes.axvline(0.0, color="0.85", lw=3.0, zorder=0)
    axes.plot(micron, subtracted, color="#c1442e", lw=1.1, label="subtracted poles")
    axes.plot(micron, exact, color="0.35", lw=2.6, alpha=0.55, label="polygon oracle")
    axes.plot(
        micron, shipped, color="#1f6f8b", lw=1.4, ls="--", label="complement poles"
    )
    axes.set_xlabel(r"offset from the lower face  [$\mu$m]")
    axes.set_ylabel(COMPONENTS[index])
    # framed on the ORACLE's own variation, so the true curve is readable and the
    # subtracted arrangement leaves the frame -- which is the point being made
    middle = 0.5 * (exact.max() + exact.min())
    reach = max(exact.max() - middle, 1e-30)
    axes.set_ylim(middle - 40.0 * reach, middle + 40.0 * reach)
    excursion = np.max(np.abs(subtracted - exact)) / np.max(np.abs(exact))
    axes.set_title(
        f"steps by {excursion:.0f}x the true value, and changes sign"
        if excursion > 1.0
        else f"unaffected, to {excursion:.0e}",
        fontsize=9,
    )
    if index == 0:
        axes.legend(fontsize=8, loc="lower left", framealpha=0.9)


def ladder(axes, title, build):
    """Draw the deviation against standoff for one corner plane, both sides folded."""
    curves = {"subtracted": [], "complement": []}
    for standoff in LADDER:
        target_r, target_z = build(np.array([-standoff, standoff]))
        found = deviation(target_r, target_z)
        for name in curves:
            curves[name].append(float(np.max(found[name])))
    for name, colour in (("subtracted", "#c1442e"), ("complement", "#1f6f8b")):
        axes.loglog(LADDER, curves[name], "o-", ms=3.5, color=colour, label=name)
    on_plane = deviation(*build(np.array([0.0])))
    axes.axhline(
        float(on_plane["complement"][0]),
        color="#1f6f8b",
        ls=":",
        lw=1.0,
    )
    axes.invert_xaxis()
    axes.set_ylim(1e-16, 1e4)
    axes.set_xlabel("standoff  [m]")
    axes.set_title(title, fontsize=9)


def main(path: str) -> None:
    """Build the two rows and write the figure."""
    figure = plt.figure(figsize=(15.0, 8.4))
    grid = figure.add_gridspec(2, 15, hspace=0.34, wspace=1.9)
    for index in range(3):
        crossing(figure.add_subplot(grid[0, 5 * index : 5 * index + 5]), index)

    lower, upper = LEVEL - HEIGHT / 2.0, LEVEL + HEIGHT / 2.0
    inner, outer = RADIUS - WIDTH / 2.0, RADIUS + WIDTH / 2.0
    panels = (
        (
            "lower face, $r$ inside the span\n"
            f"$z = {lower:+.2f}$, $r = {INSIDE_SPAN:.4f}$",
            lambda step: (np.full(step.shape, INSIDE_SPAN), lower + step),
        ),
        (
            "upper face, $r$ outside the span\n"
            f"$z = {upper:+.2f}$, $r = {OUTSIDE_SPAN:.4f}$",
            lambda step: (np.full(step.shape, OUTSIDE_SPAN), upper + step),
        ),
        (
            f"inner radius plane\n$r = {inner:.2f}$, $z = {OFF_MIDPLANE:+.3f}$",
            lambda step: (inner + step, np.full(step.shape, OFF_MIDPLANE)),
        ),
        (
            f"outer radius plane\n$r = {outer:.2f}$, $z = {OFF_MIDPLANE:+.3f}$",
            lambda step: (outer + step, np.full(step.shape, OFF_MIDPLANE)),
        ),
        (
            "the $r_s = r$ line, through a corner\n"
            f"$r = {inner:.2f}$, $z \\to {lower:+.2f}$",
            lambda step: (np.full(step.shape, inner), lower + step),
        ),
    )
    for index, (title, build) in enumerate(panels):
        axes = figure.add_subplot(grid[1, 3 * index : 3 * index + 3])
        ladder(axes, title, build)
        if index == 0:
            axes.set_ylabel("worst relative deviation from the oracle")
            axes.legend(fontsize=8, loc="lower left")

    figure.savefig(path, dpi=140, bbox_inches="tight")
    print(path)


if __name__ == "__main__":
    default = pathlib.Path(__file__).resolve().parent.parent / ".evidence"
    default.mkdir(exist_ok=True)
    main(
        sys.argv[1]
        if len(sys.argv) > 1
        else str(default / "section_face_continuity.png")
    )
