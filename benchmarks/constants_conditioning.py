"""Evidence for how the arc reduction's ring constants reach their own poles.

:class:`nova.biot.constants.Constants` supplies three ring characteristics and a
modulus to the arc elements, and each of them approaches one at a source corner --
the two levels ``gamma = zs - z = 0`` and the two radii ``rs = r``.  What the
elliptic integrals actually want is the COMPLEMENT of each: the first kind grows
like ``-log k'`` and the third like the inverse root of its pole, so the answer's
relative accuracy is the complement's, and a complement reached as ``1 - k^2`` or
``1 - n`` carries an absolute ``eps`` however the parameter was formed.

This script draws both arrangements of the same algebra.

* **subtracted** -- the characteristics as the paper prints them, ``2r/(r -/+ c)``
  with ``c = sqrt(gamma^2 + r^2)``, the modulus complement as ``1 - 4 r rs/a^2``,
  the third pole as ``1 - 4 r rs/b^2``, the near root's numerator as ``rs - c``,
  the radial row's weight as ``1 + k^2 (gamma^2 - b r)/(2 r rs)``, and factors just
  under one holding the modulus and two of the characteristics off unity so the
  divisions and the library calls stay in range.  Those factors are part of what is
  being measured: they cap every complement at their own distance from one, which
  is 4.4e-16 whatever the geometry does.
* **complement** -- each pole as the exact square the geometry gives it, from
  ``r - c = -gamma^2/(r + c)``:

      1 - n1 = ((r + c)/gamma)^2   1 - n2 = (gamma/(r + c))^2   1 - n3 = ((rs-r)/b)^2
      k'^2 = (gamma^2 + (rs - r)^2)/a^2

  with ``rs - c`` as ``(rs - r)`` less ``gamma^2/(r + c)`` and the radial weight
  collapsed onto ``(3 gamma^2 + (rs - r)(rs + r))/a^2``.  This is what
  :class:`nova.biot.constants.Constants` evaluates.

**Top row -- the arguments.**  Relative error of each complement and of the far
characteristic itself against the extended-precision value of the same expression,
over a ladder of standoffs from a tenth of the ring radius down to the resolution a
float radius has at all.  The subtracted arrangement rises one decade per decade
until it saturates on the unity-holding factor, or passes one and keeps going; the
geometric arrangement is a flat line on round-off.  Note the third pole, which
carries no ``gamma`` at all: it is lost for any thin section at ANY standoff off the
plane, which is a wider engagement than the corner plane alone.

**Bottom row -- the integrals those arguments are handed**, one pole per panel and
both arrangements through the SAME routine, so the gap between the two curves is
the argument and nothing else.

The middle two panels are the same NEAR pole at two amplitudes and they are the
point of the row.  The third kind's denominator at the amplitude is
``cos^2 + (1 - n) sin^2``, so a mid-range amplitude dilutes a lost pole to nothing
-- both curves sit on round-off there, and a reader measuring only mid-range
targets would conclude the near pole did not matter -- while at a quarter turn the
sine's square is one, nothing is left to dilute it, and the same pole passes 100 %.
A quarter turn is a target on an arc's END PLANE, which a segmented winding
produces at every joint.  The fourth panel is the third pole at the finer corner
offset, flat in the standoff because its complement carries no ``gamma``.

The last panel does not close, and that is what it is for: the incomplete first
kind is drawn through ``ellipkinc`` in both arrangements because the parameter is
all that entry point takes, so an exact one removes the unity-holding factor's bias
and leaves the routine's own ``1 - m``.  Where it settles is the size of what a
complement-native incomplete first kind is worth to the arc rows.

References are the complement-native routines, whose own accuracy gates live in
``tests/test_biotcompleteelliptic.py`` and ``tests/test_biotincompleteelliptic.py``,
fed the exact geometric complements; the extended-precision values are longdouble.
Nothing here measures a special function -- what is measured is the ARGUMENT each
one is handed.

Run as ``python benchmarks/constants_conditioning.py [output.png]``.
"""

from __future__ import annotations

import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.special

from nova.biot.completeelliptic import complete_pole
from nova.biot.incompleteelliptic import incomplete_kind, incomplete_pole

# An ITER-scale major radius and a centimetre-scale section, which is where the
# squared aspect ratio the poles sit at is small enough for the subtraction to lose
# everything.
RADIUS = 6.2
# The corner's radial distance from the target, as a fraction of the radius: one at
# a section face's own scale and one four decades finer, because the third pole
# depends on this and not on the standoff from the plane.
OFFSETS = (1e-3, 1e-6)
# Standoff from the source corner's plane, over nine decades.
LADDER = 10.0 ** -np.arange(1.0, 13.0)
# Distance of the arc amplitude below a quarter turn.  A vanishing one is a target
# ON the arc's end plane, where the near pole's loss reaches the answer undiluted.
CO_AMPLITUDES = (0.9, 1e-9)
EPS = 2.0 * np.finfo(float).eps
FLOOR = 1e-17  # so a round-off-exact point still lands on a log axis


def geometry(offset, ratio):
    """Return the source and target radii and the standoff, as plain floats."""
    return RADIUS * (1.0 + offset), RADIUS, RADIUS * ratio


def extended(offset, ratio):
    """Return the complements and the far characteristic in longdouble."""
    rs, r, gamma = (np.longdouble(term) for term in geometry(offset, ratio))
    b = rs + r
    a2 = gamma**2 + b**2
    c = np.sqrt(gamma**2 + r**2)
    return {
        "modulus": (gamma**2 + (rs - r) ** 2) / a2,
        "far pole": ((r + c) / gamma) ** 2,
        "near pole": (gamma / (r + c)) ** 2,
        "third pole": ((rs - r) / b) ** 2,
        "far characteristic": -2 * r * (r + c) / gamma**2,
        "radial weight": (3 * gamma**2 + (rs - r) * b) / a2,
    }


def subtracted(offset, ratio):
    """Return every quantity as the paper prints it, unity-holding factors and all."""
    rs, r, gamma = geometry(offset, ratio)
    b = rs + r
    a2 = gamma**2 + b**2
    c = np.sqrt(gamma**2 + r**2)
    k2 = (1.0 - EPS) * 4.0 * r * rs / a2
    np2 = {
        1: 2.0 * r / (r - c - EPS),
        2: (1.0 - EPS) * 2.0 * r / (r + c),
        3: (1.0 - EPS) * 4.0 * r * rs / b**2,
    }
    return {
        "modulus": 1.0 - k2,
        "far pole": 1.0 - np2[1],
        "near pole": 1.0 - np2[2],
        "third pole": 1.0 - np2[3],
        "far characteristic": np2[1],
        "radial weight": 1.0 + k2 * (gamma**2 - b * r) / (2.0 * r * rs),
        "parameter": k2,
        "characteristics": np2,
    }


def complement(offset, ratio):
    """Return every quantity as the geometry gives it, with no factor held back."""
    rs, r, gamma = geometry(offset, ratio)
    b = rs + r
    a2 = gamma**2 + b**2
    c = np.sqrt(gamma**2 + r**2)
    return {
        "modulus": (gamma**2 + (rs - r) ** 2) / a2,
        "far pole": ((r + c) / gamma) ** 2,
        "near pole": (gamma / (r + c)) ** 2,
        "third pole": ((rs - r) / b) ** 2,
        "far characteristic": -2.0 * r * (r + c) / gamma**2,
        "radial weight": (3.0 * gamma**2 + (rs - r) * b) / a2,
        "parameter": 4.0 * r * rs / a2,
        "characteristics": {
            1: -2.0 * r * (r + c) / gamma**2,
            2: 2.0 * r / (r + c),
            3: 4.0 * r * rs / b**2,
        },
    }


def deviation(got, want):
    """Return relative deviation from an extended-precision value, floored."""
    want = np.longdouble(want)
    return max(float(abs(np.longdouble(got) - want) / abs(want)), FLOOR)


def amplitude_pair(co_amplitude):
    """Return the amplitude and its own ``(sine, cosine)``, from the co-amplitude.

    Formed from the distance BELOW a quarter turn, which is how a caller holding
    its geometry supplies it: the arc's amplitude is ``(pi + psi)/2`` for an
    azimuthal separation ``psi`` from one of its ends, so the pair is exact and
    exactly ``(1, 0)`` where the separation vanishes.
    """
    return (
        np.array([0.5 * np.pi - co_amplitude]),
        np.array([np.cos(co_amplitude)]),
        np.array([np.sin(co_amplitude)]),
    )


def third_kind_error(build, offset, ratio, pole_name, co_amplitude):
    """Return one pole's third-kind deviation from the complement-native reference.

    ``build`` supplies the arguments in one arrangement or the other; both go
    through the SAME routine, so what separates the two curves is the argument and
    nothing else.  A vanishing ``co_amplitude`` takes the complete integral, which
    is the limit in which the arc closes onto the ring.
    """
    terms, exact = build(offset, ratio), complement(offset, ratio)
    arguments = (np.array([terms[pole_name]]), np.array([terms["modulus"]]))
    reference = (np.array([exact[pole_name]]), np.array([exact["modulus"]]))
    if co_amplitude is None:
        got, want = complete_pole(*arguments), complete_pole(*reference)
    else:
        _, sine, cosine = amplitude_pair(co_amplitude)
        got = incomplete_pole(*arguments, sine, cosine)
        want = incomplete_pole(*reference, sine, cosine)
    return deviation(got[0], want[0]) if want[0] else FLOOR


def first_kind_error(build, offset, ratio, co_amplitude):
    """Return the incomplete first kind's deviation, through the library form.

    Drawn through ``ellipkinc`` in BOTH arrangements, because the parameter is all
    that entry point takes: an exact one removes the unity-holding factor's own
    bias and nothing else, since the routine re-derives the complement internally.
    So this panel does not close, and where it settles is the size of what a
    complement-native incomplete first kind is worth to the arc rows.
    """
    terms, exact = build(offset, ratio), complement(offset, ratio)
    amplitude, sine, cosine = amplitude_pair(co_amplitude)
    got = scipy.special.ellipkinc(amplitude[0], terms["parameter"])
    want, _ = incomplete_kind(
        amplitude,
        np.array([exact["modulus"]]),
        sine=sine,
        cosine=cosine,
        parameter=np.array([exact["parameter"]]),
    )
    return deviation(got, want[0])


ARRANGEMENTS = (
    ("subtracted", subtracted, "C3", "-"),
    ("complement", complement, "C0", "--"),
)
ARGUMENTS = ("modulus", "far pole", "near pole", "third pole", "far characteristic")


def argument_panel(axes, name, offset):
    """Draw one argument's deviation against the standoff, both arrangements."""
    for label, build, colour, style in ARRANGEMENTS:
        curve = [
            deviation(build(offset, ratio)[name], extended(offset, ratio)[name])
            for ratio in LADDER
        ]
        axes.loglog(LADDER, curve, style, color=colour, marker=".", label=label)
    axes.axhline(EPS, color="0.6", lw=0.8, ls=":")
    axes.set_title(name, fontsize=9)
    axes.set_xlabel(r"$\gamma/r$")
    axes.invert_xaxis()
    axes.grid(alpha=0.25, which="both")


def integral_panel(axes, title, curves):
    """Draw one integral's deviation against the standoff, both arrangements."""
    for (label, _, colour, style), curve in zip(ARRANGEMENTS, curves, strict=True):
        axes.loglog(LADDER, curve, style, color=colour, marker=".", label=label)
    axes.axhline(EPS, color="0.6", lw=0.8, ls=":")
    axes.set_title(title, fontsize=9)
    axes.set_xlabel(r"$\gamma/r$")
    axes.invert_xaxis()
    axes.grid(alpha=0.25, which="both")


# Which integral each of the lower panels draws.  The near pole appears at two
# amplitudes because the amplitude is what decides whether its loss reaches the
# answer, and the third pole at the finer corner offset because it is the offset and
# not the standoff that its own complement is set by.
def integral_panels():
    """Return ``(title, per-arrangement curves)`` for each lower panel."""

    def third(pole_name, offset, co_amplitude):
        return [
            [
                third_kind_error(build, offset, ratio, pole_name, co_amplitude)
                for ratio in LADDER
            ]
            for _, build, _, _ in ARRANGEMENTS
        ]

    def first(offset, co_amplitude):
        return [
            [first_kind_error(build, offset, ratio, co_amplitude) for ratio in LADDER]
            for _, build, _, _ in ARRANGEMENTS
        ]

    end_plane = rf"$\theta = \pi/2 - ${CO_AMPLITUDES[-1]:.0e}"
    mid_range = rf"$\theta = \pi/2 - ${CO_AMPLITUDES[0]:.0e}"
    return (
        (
            f"complete third kind, far pole\ncorner {OFFSETS[0]:.0e} r out",
            third("far pole", OFFSETS[0], None),
        ),
        (
            f"incomplete, near pole, mid range\n{mid_range}",
            third("near pole", OFFSETS[0], CO_AMPLITUDES[0]),
        ),
        (
            f"incomplete, near pole, on the end plane\n{end_plane}",
            third("near pole", OFFSETS[0], CO_AMPLITUDES[-1]),
        ),
        (
            f"incomplete, third pole\ncorner {OFFSETS[1]:.0e} r out, {end_plane}",
            third("third pole", OFFSETS[1], CO_AMPLITUDES[-1]),
        ),
        (
            f"incomplete first kind\ncorner {OFFSETS[1]:.0e} r out, {end_plane}",
            first(OFFSETS[1], CO_AMPLITUDES[-1]),
        ),
    )


def table():
    """Print the deviations the figure draws, so the numbers can be quoted."""
    poles = ("far pole", "near pole", "third pole")
    for offset in OFFSETS:
        print(f"\ncorner {offset:.0e} r out radially, target -> the corner plane")
        print("  arguments, then the third kind per pole on the arc's end plane")
        header = (
            "gamma/r",
            *(name[:11] for name in ARGUMENTS),
            *(f"Pi {name[:5]}" for name in poles),
            "F inc",
        )
        print(("{:>13}" * len(header)).format(*header))
        for label, build, _, _ in ARRANGEMENTS:
            print(f"  {label}")
            for ratio in LADDER:
                row = [
                    deviation(build(offset, ratio)[name], extended(offset, ratio)[name])
                    for name in ARGUMENTS
                ]
                row += [
                    third_kind_error(build, offset, ratio, name, CO_AMPLITUDES[-1])
                    for name in poles
                ]
                row.append(first_kind_error(build, offset, ratio, CO_AMPLITUDES[-1]))
                print(("{:13.1e}" * (len(row) + 1)).format(ratio, *row))


def main(path):
    """Draw the arguments and the integrals they are handed, and print the table."""
    table()
    figure = plt.figure(figsize=(16.0, 7.5), constrained_layout=False)
    grid = figure.add_gridspec(2, len(ARGUMENTS), hspace=0.65, wspace=0.3)
    for column, name in enumerate(ARGUMENTS):
        axes = figure.add_subplot(grid[0, column])
        argument_panel(axes, name, OFFSETS[0])
        if column == 0:
            axes.set_ylabel(
                f"relative deviation\ncorner {OFFSETS[0]:.0e} r out", fontsize=9
            )
            axes.legend(fontsize=8, loc="lower left")
    for column, (title, curves) in enumerate(integral_panels()):
        axes = figure.add_subplot(grid[1, column])
        integral_panel(axes, title, curves)
        if column == 0:
            axes.set_ylabel("relative deviation", fontsize=9)
    figure.suptitle(
        f"Ring constants at a source corner: {RADIUS} m ring, "
        "every pole by subtraction against every pole from the geometry",
        fontsize=11,
    )
    figure.savefig(path, dpi=140, bbox_inches="tight")
    print(path)


if __name__ == "__main__":
    default = pathlib.Path(__file__).resolve().parent.parent / ".evidence"
    default.mkdir(exist_ok=True)
    main(
        sys.argv[1]
        if len(sys.argv) > 1
        else str(default / "constants_conditioning.png")
    )
