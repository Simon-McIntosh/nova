"""What the parameter form of the complete elliptic integrals costs the point kernels.

``scipy.special.ellipk`` and ``ellipe`` take the PARAMETER ``m = k^2``, and a float
parameter cannot carry its own complement: by the time ``k'^2`` is smaller than the
spacing of the numbers next to one, ``m = 1 - k'^2`` has rounded it away entirely.
``K`` grows like ``-log k'``, so it comes back with an absolute error of about
``eps / 2 k'^2`` -- and the two point-filament kernels, the most-used in
:mod:`nova.biot.greens`, are built on exactly that call.

The complement is not hard to come by.  With ``denom = (a + R)^2 + dz^2`` and
``d2 = (a - R)^2 + dz^2`` the two differ by exactly ``4 a R``, so
``k'^2 = d2 / denom`` needs no subtraction from one and keeps every digit however
close the target sits; ``greens_bz_br`` already forms both radicals.  Carlson's
``R_F(0, k'^2, 1)`` takes it as an argument, and so -- for the first kind, which is
the only sensitive one -- does Cephes' ``ellipkm1``.

Measured against an extended-precision reference built in this module (80-bit
``longdouble``, a complement-seeded AGM, and power series for the two brackets that
cancel), with the routes differing from the shipped code in NOTHING but how they
obtain ``k2``, ``K`` and ``E``: one of the routes IS the shipped kernel, which
:func:`_pin_shipped` reports the gap for.  It was the parameter route when this was
written and is the split-pole route now, the recommendation below having landed --
so that gap is the small one and the parameter gap is what the adoption removed.

Three limits turned up.  They bind in different places and only the first is about
the elliptic integrals at all.

THE FLUX, WHICH THE COMPLEMENT FIXES COMPLETELY.  ``psi``'s relative error on the
shipped route follows ``eps / 2 k'^2 K`` over eleven decades of complement.  A target
at distance ``d`` from the filament sits at ``k'^2 = (d/2a)^2``, so the distance at
which a tolerance goes is proportional to the RING RADIUS, and the crossings collapse
when read in ring radii -- worst over three approach directions (radially in the
plane, vertically, and at 45 degrees), as ``d/a``:

    tolerance                1e-06     1e-09     1e-12
    parameter, a = 1.0 m     2.2e-06   8.5e-05   8.2e-03
    parameter, a = 6.2 m     3.5e-06   8.5e-05   8.2e-03
    complement, either       clean     clean     clean

``clean`` means the tolerance held at every sample, down to ``k'^2 = 2.5e-25``
(``d = 6.2e-12 m`` on the 6.2 m ring), where the complement routes still return
``psi`` to 1e-16 -- the double round-off floor of the assembled kernel.  The scatter
between directions is a factor of four at fixed ``k'^2``, which is where the last bit
of ``m`` happens to land, and the 1e-12 column is the least robust for that reason.

The cap, ``k2 = clip(k2, 0, 1 - 1e-12)``, engages at ``k'^2 = 4.9e-13``: 1.4
micrometres from a unit ring, 8.7 from an ITER-scale one.  It does not bound the
error, it only stops it diverging -- inside it ``K`` is frozen at ``K(1 - 1e-12)``
while the true ``K`` keeps growing, so ``psi``'s error climbs from 2.6e-02 at
engagement to 3.7e-01 at ``k'^2 = 2.2e-19``.  At the engagement point itself the cap
is a bad trade in its own right: unclipped, the same target is wrong by 3.5e-06
rather than 2.6e-02.  Removing the cap is not the fix either, because it converts
closer targets into ``inf``, or into ``nan`` for the ones where ``4 a R`` rounds above
``(a + R)^2`` -- which happens one ULP off the filament, and which way round depends
on the radius.

THE FIELD COMPONENTS, WHERE THE ELLIPTIC ROUTE IS NOT THE LIMIT.  ``B_Z`` and
``B_R`` are dominated by their ``E/d2`` pole, and two things in the shipped bracket
cost more than the modulus does:

* ``np.maximum(d2, _R_FLOOR)`` clamps that pole at ``_R_FLOOR = 1e-9``.  It is an
  absolute floor on a SQUARED LENGTH, so it engages 32 micrometres from the filament
  for every ring radius -- ``a`` does not appear -- and inside it the field stops
  diverging and is simply wrong.  It is the reason the field crossings above are the
  same distance in METRES (2.2e-05) for both ring sizes while the flux crossings are
  the same fraction of the radius.
* ``ar**2 - r**2 - dz**2`` is a difference of terms of order ``a^2`` whose value is of
  order ``a d``, so it arrives with relative error ``eps a / 2d``.  Written
  ``-d2 + 2 a (a - R)`` -- the same quantity exactly -- nothing cancels.

At 10 micrometres from an ITER-scale filament ``B_Z`` is out by 9.0e-01 on the shipped
route AND on both complement routes, and by 9.9e-16 once those two are fixed.  So the
field components need the ALGEBRA repaired, not the special function; with it
repaired they are clean at every distance sampled, and the complement then reaches
them only through ``K - E``.

THE FAR-FIELD BRACKET, WHICH NO ROUTE FIXES.  ``greens_psi`` forms
``(1 - k2/2) K - E``, whose series starts at ``m^2``:

    (1 - m/2) K - E = (pi/2) [ m^2/16 + m^3/32 + ... ]

out of terms of order one, so the bracket loses ``log10(16/m^2)`` digits wherever
``K`` and ``E`` came from -- and the measurement confirms all five routes agree there
to within rounding luck.  ``k^2`` is small for two quite different targets, and the
distinction is what decides whether this matters:

    tolerance      k^2        receding in z     approaching the axis (a = 6.2 m)
    1e-12          3.3e-02    11 ring radii     R < 5.1e-02 m
    1e-09          8.6e-04    68 ring radii     R < 1.3e-03 m
    1e-06          2.3e-05    420 ring radii    R < 3.6e-05 m

Neither branch reaches anywhere the spine evaluates.  A magnetic probe three ring
radii out sits at ``k^2 = 0.31`` and measures 5.5e-15; at ten ring radii,
``k^2 = 0.038`` and 6.4e-13; a psi grid whose inner edge is at ``R = 0.1 m`` on a
6.2 m ring sits at ``k^2 = 0.063`` and 4.5e-13.  The field components cancel only at
depth ``m/2`` and are two to four decades better again.  This is a finding to know
about rather than a defect to fix: it would take replacing the bracket with its series
below some ``m``, and nothing in the machine is far enough out to pay for that.

WHAT THE CONFIGURATIONS THAT MATTER ACTUALLY COST.  Relative error, shipped route
against complement-and-split-pole:

    configuration                        k'^2       psi                B_Z
    plasma cell, own edge (60 mm)        2.3e-05    8.4e-13 / 8.4e-17  1.5e-14 / 5.6e-16
    plasma cell, next cell (120 mm)      9.2e-05    2.7e-13 / 2.0e-16  1.1e-14 / 2.3e-16
    probe 10 mm from a coil              6.5e-07    3.0e-12 / 2.5e-16  3.2e-14 / 2.3e-16
    probe 1 mm from a coil               6.5e-09    5.3e-10 / 3.1e-16  6.4e-13 / 3.1e-16
    probe 10 um from a coil              6.5e-13    1.6e-02 / 1.0e-16  9.0e-01 / 9.9e-16
    winding pack, 1 um (a = 1 m)         2.5e-13    5.0e-02 / 6.1e-17  1.0e+00 / 1.6e-15
    grid edge near the axis (R = 0.1)    9.4e-01    4.5e-13 / 1.1e-13  2.6e-16 / 2.6e-16
    probe at 3 ring radii                6.9e-01    5.5e-15 / 5.5e-15  8.8e-16 / 8.8e-16

The second kind never needs the complement.  ``dE/dm`` diverges only logarithmically,
so ``E`` from the parameter is good to ``K eps`` in the worst case -- and better than
that in practice, because where ``m`` rounds to exactly one ``E(1) = 1`` is itself the
right value to 1e-24.  Measured, the split-pole route's ``B_Z`` holds 1e-16 at
``k'^2 = 2.5e-25`` where the ``K eps`` bound would allow 3.3e-15.

COST.  One SLURM compute core, fresh process per variant, median of three, over the
targets a grid evaluation actually hands the kernels (a box about a 6.2 m ring):

    variant                              us/element        ratio
                                      200k        2M      (2M)
    ellipk(m) + ellipe(m)             0.0296    0.0297     1.00
    ellipkm1(k'^2) + ellipe(m)        0.0303    0.0329     1.11
    R_F(0,k'^2,1) + 2 R_G(0,k'^2,1)   0.1149    0.1180     3.98
    greens_psi as it stands           0.0623    0.0675     2.28
    greens_bz_br as it stands         0.0871    0.1029     3.47
    psi and both fields, parameter    0.1210    0.1394     4.70
    ... via Carlson                   0.2110    0.2279     7.68
    ... via ellipkm1                  0.1210    0.1394     4.70

The special functions are 44 per cent of ``greens_psi`` and 29 per cent of
``greens_bz_br``, so the pair's cost is worth caring about.  ``ellipkm1`` costs 11 per
cent more than ``ellipk`` on the pair alone and NOTHING measurable on the assembled
three components -- 0.1394 either way, the two medians identical to four figures.
Carlson costs 4x on the pair and 64 per cent on the whole kernel.

DOWNSTREAM, THE CHANGE DOES NOT REACH THE BANDED SCHEME.  ``moment_filament`` takes
SECOND differences of these kernels in the SOURCE position on a step of 2e-03 section
radii and divides by ``step^2``, which multiplies any absolute kernel error by
``2.5e05 / cell^2`` -- so this had to be measured rather than argued.  The same
stencil and moments, driven by each route, against the exact hexagon, worst over 24
directions per band:

    standoff [section radii]   1.5      3.4      7.7      17.4     30.0
    psi, parameter             1.2e-04  1.1e-06  1.5e-08  9.9e-10  4.3e-10
    psi, split pole            1.2e-04  1.1e-06  1.3e-08  7.6e-10  3.3e-10

Every route agrees to 1.00x inside four section radii and to 1.3x at thirty (one band
reaches 1.9x), and ``B_Z`` agrees to 1.07x everywhere: the corrected filament is
limited by the TRUNCATION of the moment expansion, not by the kernel's round-off.
``benchmarks/nearfar_error.py``'s cutoffs are recorded under ``baseline`` in the JSON
as the before-image, and they should not move.

RECOMMENDATION.  Take ``K`` from ``ellipkm1(d2 / denom)`` in both point kernels: it
is a two-line change that costs nothing measurable end to end and removes the flux's
entire near-field error -- four orders at a plasma cell's own edge, six at a probe a
millimetre from a coil, and everything inside ten micrometres, where the cap today
silently answers for a target nine micrometres further away.  Do not adopt the
Carlson pair for this: it buys the same accuracy for four times the special-function
cost and 64 per cent of the whole kernel, and the second kind does not need the
complement at all.  The field components' real limit is not the elliptic route but
``np.maximum(d2, _R_FLOOR)`` -- an absolute floor on a squared length that leaves
``B_Z`` 90 per cent wrong inside 32 micrometres of any filament, whatever its radius
-- together with the ``ar**2 - r**2 - dz**2`` difference on top of it; both are exact
in double once written with the pole split off, and that is the change that matters
for a probe near a coil.

A COINCIDENT TARGET, AND WHAT THE CONTRACT SHOULD BE.  The field is genuinely
singular on the filament, so there is no right answer, but there is a right contract
and the routes disagree about it.  On a unit ring at ``R = a``, ``Z = 0``:

    parameter (today)     psi = 1.66e-05, B_Z = 1.52e-06   finite, silent, wrong
    parameter unclipped   psi = inf,      B_Z = inf        (nan one ULP off)
    complement            psi = inf,      B_Z = inf
    split pole            psi = inf,      B_Z = nan (0/0)

Today's numbers are the kernel evaluated at a phantom standoff of 1.4 micrometres:
whatever the caller asked for, it is answered for a target that far away with nothing
in the return to say so, and that is the one behaviour a caller cannot detect.  The
contract should be the divergence -- ``inf`` at ``d2 == 0``, and let the caller decide
-- because a coincident target means a filament model has been asked a question it
cannot answer, and the finite-section kernels (``cylinder_greens``,
``polygon_greens``, ``moment_filament``) exist for precisely that target and are
smooth through the conductor.  One ULP off the filament the complement routes return
the true finite values (``psi = 4.63e-05 Wb/A``, ``B_Z = 1.8e+09 T/A``), which is the
other half of the same contract: adjacent is not coincident, and only the complement
can tell the two apart.

    python benchmarks/near_field_elliptic.py <output.json> [--figure <path.png>]
    python benchmarks/near_field_elliptic.py <output.json> --quick   # skip timing
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time

import numpy as np
import scipy.special

from nova.biot.greens import (
    MU0,
    _R_FLOOR,
    greens_bz_br,
    greens_psi,
    second_moments,
    section_centroid,
)

COMPONENTS = ("psi", "bz", "br")
TOLERANCES = (1e-6, 1e-9, 1e-12)
RADII = (1.0, 6.2)
"""Ring radii: a unit ring, and the ITER major radius the rest of the suite uses."""

CLIP = 1.0e-12
"""The shipped kernels' cap, ``k2 = clip(k2, 0, 1 - CLIP)``.  It engages once the
complement falls below it, which for a target at distance ``d`` from the filament
is at ``d = 2 a sqrt(CLIP)`` -- two micrometres for a unit ring."""

LD = np.longdouble
PI = LD("3.14159265358979323846264338327950288")
MU0_LD = LD(MU0)
"""The DOUBLE permeability promoted, not recomputed: the routes and the reference
must share the constant exactly, or their difference would report the constant's
own rounding as a route error."""

SERIES_LIMIT = 0.5
"""Below this parameter the reference takes its cancelling combinations from their
power series instead of from K and E, which is the only way to hold them to
extended precision -- see :func:`flux_combination`."""


# --- extended-precision reference -------------------------------------


def complete_agm(kc2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(K, E)`` in extended precision from the COMPLEMENT ``kc2 = 1 - m``.

    The arithmetic-geometric mean, seeded with ``g_0 = sqrt(kc2)``, so the
    complement enters as itself and never as a difference against one.  That is
    what makes this usable as a reference for the very thing being measured: the
    seed carries every digit of ``kc2``, however small it is.

        a_0 = 1, g_0 = k';  a_+ = (a + g)/2, g_+ = sqrt(a g);  K = pi / (2 a_inf)
        E = K (1 - sum_n 2^(n-1) c_n^2),  c_0^2 = 1 - kc2,  c_n = (a_- - g_-)/2

    ``E`` is formed as ``K`` times a factor that tends to ``1/K``, so its relative
    error grows only as ``K`` times the extended-precision round-off -- 2e-18 at
    the deepest complement this study reaches.
    """
    kc2 = np.asarray(kc2, dtype=LD)
    a = np.ones_like(kc2)
    g = np.sqrt(kc2)
    total = LD(0.5) * (LD(1.0) - kc2)  # the n = 0 term, 2^(-1) c_0^2 with c_0^2 = m
    weight = LD(1.0)
    iterations = 0
    for iterations in range(1, 65):
        half_difference = LD(0.5) * (a - g)
        total = total + weight * half_difference**2
        weight = weight * LD(2.0)
        a, g = LD(0.5) * (a + g), np.sqrt(a * g)
        if np.all(a - g <= np.finfo(LD).eps * a):
            break
    complete_agm.iterations = iterations  # type: ignore[attr-defined]
    big_k = PI / (LD(2.0) * a)
    return big_k, big_k * (LD(1.0) - total)


def _hypergeometric_coefficients(count: int) -> np.ndarray:
    """Return ``a_n = ((1/2)_n / n!)^2`` for ``n < count``, the K series' weights."""
    order = np.arange(1, count, dtype=LD)
    return np.concatenate([[LD(1.0)], np.cumprod(((order - LD(0.5)) / order) ** 2)])


_SERIES_TERMS = 400
_COEFFICIENTS = _hypergeometric_coefficients(_SERIES_TERMS)


def _series(m: np.ndarray, weight) -> np.ndarray:
    """Sum ``sum_n weight(n) m^n`` in extended precision until the terms die.

    The convergence test cannot fire before order three: both series below have a
    vanishing leading coefficient -- that vanishing IS the cancellation being
    sidestepped -- and a term-against-total test would read the first zero term as
    convergence and return zero.
    """
    m = np.asarray(m, dtype=LD)
    total = np.zeros_like(m)
    power = np.ones_like(m)
    for order in range(1, _SERIES_TERMS):
        power = power * m
        term = weight(order) * power
        total = total + term
        if order > 2 and np.all(np.abs(term) <= np.finfo(LD).eps * np.abs(total)):
            break
    return total


def _blend(m: np.ndarray, direct: np.ndarray, weight) -> np.ndarray:
    """Return ``direct``, with the series substituted below :data:`SERIES_LIMIT`.

    Masked rather than selected with ``where``: the series needs hundreds of terms
    as ``m`` approaches one and never converges there, so it must not be evaluated
    on the elements that do not use it.
    """
    m = np.asarray(m, dtype=LD)
    out = np.array(direct, dtype=LD, copy=True)
    small = m < SERIES_LIMIT
    if small.any():
        out[small] = PI / LD(2.0) * _series(m[small], weight)
    return out


def flux_combination(m: np.ndarray, big_k: np.ndarray, big_e: np.ndarray) -> np.ndarray:
    """Return ``(1 - m/2) K - E``, the bracket the flux kernel forms.

    Its series starts at ``m^2``: the ``m^0`` and ``m^1`` coefficients cancel
    identically, so evaluating the bracket from K and E throws away
    ``log10(16/m^2)`` digits.  Below :data:`SERIES_LIMIT` the reference therefore
    sums the series instead,

        (1 - m/2) K - E = (pi/2) sum_(n>=2) [a_n 2n/(2n-1) - a_(n-1)/2] m^n,

    whose leading term is ``(pi/2) m^2/16`` and which carries no cancellation at
    all.  This is a property of the BRACKET, not of how K and E were obtained --
    no choice of elliptic-integral routine repairs it.
    """
    direct = (LD(1.0) - LD(0.5) * np.asarray(m, dtype=LD)) * big_k - big_e
    return _blend(m, direct, _flux_weight)


def kind_difference(m: np.ndarray, big_k: np.ndarray, big_e: np.ndarray) -> np.ndarray:
    """Return ``K - E``, which the field kernels form once their pole is split off.

    The series starts at ``m^1`` -- a shallower cancellation than the flux
    bracket's, and again one the series sidesteps:

        K - E = (pi/2) sum_(n>=1) a_n 2n/(2n-1) m^n.
    """
    return _blend(m, big_k - big_e, _difference_weight)


def _flux_weight(order: int) -> np.ndarray:
    """Return the flux bracket's series coefficient, zero at orders zero and one."""
    rising = _COEFFICIENTS[order] * LD(2 * order) / LD(2 * order - 1)
    return rising - _COEFFICIENTS[order - 1] / LD(2.0)


def _difference_weight(order: int) -> np.ndarray:
    """Return the ``K - E`` series coefficient, zero at order zero."""
    return _COEFFICIENTS[order] * LD(2 * order) / LD(2 * order - 1)


def reference(
    target_r: np.ndarray, target_z: np.ndarray, ar: float, az: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(psi, B_Z, B_R)`` in extended precision, free of cancellation.

    Same geometry as the shipped kernels, evaluated in ``longdouble`` (64-bit
    mantissa on this host) with two changes that the double kernels cannot make
    for themselves and that a reference must:

    * the complement is taken from the geometry, ``kc2 = d2 / denom``, where the
      two radicals differ by exactly ``4 a R`` -- so it needs no subtraction from
      one and keeps full relative precision however close the target sits;
    * the field brackets are rewritten with their pole split off,
      ``K + A E = (K - E) + 2 a (a - R)/d2 E`` and
      ``-K + B E = -(K - E) + 2 a R/d2 E``, which moves the whole far-field
      cancellation into ``K - E`` where the series handles it.

    The rearrangement is algebraically exact.  ``psi`` and ``B_R`` vanish on the
    axis by the same contract the shipped kernels honour.
    """
    r = np.asarray(target_r, dtype=LD)
    z = np.asarray(target_z, dtype=LD)
    ring_r, ring_z = LD(ar), LD(az)
    dz = z - ring_z
    denom = (ring_r + r) ** 2 + dz**2
    d2 = (ring_r - r) ** 2 + dz**2
    kc2 = d2 / denom
    m = LD(4.0) * ring_r * r / denom
    big_k, big_e = complete_agm(kc2)
    difference = kind_difference(m, big_k, big_e)

    with np.errstate(divide="ignore", invalid="ignore"):
        psi = (
            LD(2.0)
            * MU0_LD
            * np.sqrt(ring_r * r)
            / np.sqrt(m)
            * flux_combination(m, big_k, big_e)
        )
        pre = MU0_LD / (LD(2.0) * PI)
        root = np.sqrt(denom)
        bz = pre / root * (difference + LD(2.0) * ring_r * (ring_r - r) / d2 * big_e)
        br = pre * dz / (r * root) * (-difference + LD(2.0) * ring_r * r / d2 * big_e)
    on_axis = r < LD(_R_FLOOR)
    return (
        np.where(on_axis, LD(0.0), psi),
        bz,
        np.where(on_axis, LD(0.0), br),
    )


# --- the routes -------------------------------------------------------


def assemble(
    target_r: np.ndarray,
    target_z: np.ndarray,
    ar: float,
    az: float,
    k2: np.ndarray,
    big_k: np.ndarray,
    big_e: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(psi, B_Z, B_R)`` by the shipped algebra, from supplied K and E.

    A transcription of ``greens_psi`` and ``greens_bz_br`` with the two special
    functions and the modulus lifted out, so that every route below differs from
    today's code in NOTHING but how it obtains ``k2``, ``K`` and ``E``.  Pinned
    against the shipped functions in :func:`validate`.
    """
    r = np.asarray(target_r, dtype=np.float64)
    z = np.asarray(target_z, dtype=np.float64)
    dz = z - az
    denom = (ar + r) ** 2 + dz**2
    root = np.sqrt(np.maximum(denom, _R_FLOOR))
    d2 = (ar - r) ** 2 + dz**2
    k = np.sqrt(k2)

    with np.errstate(invalid="ignore", divide="ignore"):
        pref = (
            2.0 * MU0 * np.sqrt(ar * np.maximum(r, _R_FLOOR)) / np.maximum(k, _R_FLOOR)
        )
        psi = pref * ((1.0 - 0.5 * k2) * big_k - big_e)
        pre = MU0 / (2.0 * np.pi)
        bz = (
            pre
            / root
            * (big_k + (ar**2 - r**2 - dz**2) / np.maximum(d2, _R_FLOOR) * big_e)
        )
        br = (
            pre
            * dz
            / (np.maximum(r, _R_FLOOR) * root)
            * (-big_k + (ar**2 + r**2 + dz**2) / np.maximum(d2, _R_FLOOR) * big_e)
        )
    return (
        np.where(r < _R_FLOOR, 0.0, psi),
        bz,
        np.where(r < _R_FLOOR, 0.0, br),
    )


def assemble_split_pole(
    target_r: np.ndarray,
    target_z: np.ndarray,
    ar: float,
    az: float,
    k2: np.ndarray,
    kc2: np.ndarray,
    big_k: np.ndarray,
    big_e: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(psi, B_Z, B_R)`` with the field brackets' pole factored out.

    Two things the shipped brackets do that cost digits near the filament, both
    fixed here in DOUBLE, so this route measures what is reachable without leaving
    double precision:

    * ``ar**2 - r**2 - dz**2`` is a difference of terms of order ``a^2`` whose
      value is of order ``a d``, so it arrives with relative error ``eps a / 2d``.
      Written ``-d2 + 2 a (a - R)`` -- exactly the same quantity -- both pieces are
      already of the size of the answer and nothing cancels;
    * ``np.maximum(d2, _R_FLOOR)`` clamps the pole at ``1e-9`` -- an absolute floor
      on a SQUARED LENGTH, so it engages at 32 micrometres from the filament
      whatever the ring radius, and inside that the field stops diverging and is
      simply wrong.  Here the pole is divided by as it stands.

    The flux is unchanged in substance -- it has no pole and no such difference --
    but its bracket factor is taken as ``(1 + kc2)/2``, which is what ``1 - k2/2``
    is when the complement is the thing known exactly.
    """
    r = np.asarray(target_r, dtype=np.float64)
    z = np.asarray(target_z, dtype=np.float64)
    dz = z - az
    denom = (ar + r) ** 2 + dz**2
    root = np.sqrt(denom)
    d2 = (ar - r) ** 2 + dz**2
    difference = big_k - big_e
    with np.errstate(invalid="ignore", divide="ignore"):
        pref = 2.0 * MU0 * np.sqrt(ar * np.maximum(r, _R_FLOOR)) / np.sqrt(k2)
        psi = pref * (0.5 * (1.0 + kc2) * big_k - big_e)
        pre = MU0 / (2.0 * np.pi)
        bz = pre / root * (difference + 2.0 * ar * (ar - r) / d2 * big_e)
        br = pre * dz / (r * root) * (-difference + 2.0 * ar * r / d2 * big_e)
    return (
        np.where(r < _R_FLOOR, 0.0, psi),
        bz,
        np.where(r < _R_FLOOR, 0.0, br),
    )


def modulus(target_r: np.ndarray, target_z: np.ndarray, ar: float, az: float):
    """Return ``(k2, kc2)`` both taken straight from the geometry, in double.

    ``denom - d2 = 4 a R`` identically, so ``kc2 = d2 / denom`` is the complement
    of ``k2 = 4 a R / denom`` with no subtraction anywhere: whatever ``k2`` has
    lost to rounding, ``kc2`` still holds to full RELATIVE precision.  Both come
    out of quantities ``greens_bz_br`` already computes.
    """
    r = np.asarray(target_r, dtype=np.float64)
    z = np.asarray(target_z, dtype=np.float64)
    dz = z - az
    denom = np.maximum((ar + r) ** 2 + dz**2, _R_FLOOR)
    return 4.0 * ar * r / denom, ((ar - r) ** 2 + dz**2) / denom


def route_parameter(target_r, target_z, ar, az, *, clip: bool = True):
    """Today's path: ``ellipk``/``ellipe`` of the parameter, capped at ``1 - CLIP``.

    ``clip=False`` removes only the cap, to separate what the cap costs from what
    the parameter form costs.
    """
    k2, _ = modulus(target_r, target_z, ar, az)
    if clip:
        k2 = np.clip(k2, 0.0, 1.0 - CLIP)
    with np.errstate(invalid="ignore"):
        big_k = scipy.special.ellipk(k2)
        big_e = scipy.special.ellipe(k2)
    return assemble(target_r, target_z, ar, az, k2, big_k, big_e)


def route_parameter_unclipped(target_r, target_z, ar, az):
    """Today's path with the cap removed."""
    return route_parameter(target_r, target_z, ar, az, clip=False)


def route_complement(target_r, target_z, ar, az):
    """Carlson symmetric forms, which take the complement as an argument.

    ``K = R_F(0, kc2, 1)`` and ``E = 2 R_G(0, kc2, 1)``: both read ``kc2``
    directly, so neither can lose it, and neither can be handed a parameter above
    one.
    """
    k2, kc2 = modulus(target_r, target_z, ar, az)
    zero, one = np.zeros_like(kc2), np.ones_like(kc2)
    big_k = scipy.special.elliprf(zero, kc2, one)
    big_e = 2.0 * scipy.special.elliprg(zero, kc2, one)
    return assemble(target_r, target_z, ar, az, k2, big_k, big_e)


def route_complement_k(target_r, target_z, ar, az):
    """``ellipkm1(kc2)`` for K, ``ellipe`` for E -- the complement only where it bites.

    Only the FIRST kind is sensitive: ``K ~ -log(k')/1`` so its absolute error
    tracks ``eps / kc2``, whereas ``dE/dm = (E - K)/2m`` diverges only
    logarithmically and ``E -> 1``, so E from the parameter is good to ``K eps``
    -- 2e-15 at the deepest complement, and it enters the flux bracket divided by
    ``K``.  ``ellipkm1`` is the same Cephes kernel ``ellipk`` calls, entered
    through its complement argument.

    ``E`` is capped at ``m = 1`` rather than at ``1 - CLIP``, because ``E(1) = 1``
    exactly and the cap is then the true limit rather than a shift off it.  The cap
    is still needed: ``4 a R`` and ``(a + R)^2`` round independently, so the
    parameter lands one ULP ABOVE one for a target one ULP off the filament, and
    both ``ellipk`` and ``ellipe`` return ``nan`` there.
    """
    k2, kc2 = modulus(target_r, target_z, ar, az)
    big_k = scipy.special.ellipkm1(kc2)
    big_e = scipy.special.ellipe(np.minimum(k2, 1.0))
    return assemble(target_r, target_z, ar, az, k2, big_k, big_e)


def route_split_pole(target_r, target_z, ar, az):
    """``ellipkm1`` for K, and the field brackets with their pole split off.

    The end of the line for double precision: every avoidable cancellation in the
    kernel removed, the complement never reconstructed from a parameter, and no
    floor standing in for a divergence.
    """
    k2, kc2 = modulus(target_r, target_z, ar, az)
    big_k = scipy.special.ellipkm1(kc2)
    big_e = scipy.special.ellipe(np.minimum(k2, 1.0))
    return assemble_split_pole(target_r, target_z, ar, az, k2, kc2, big_k, big_e)


ROUTES = {
    "parameter": route_parameter,
    "parameter_unclipped": route_parameter_unclipped,
    "complement": route_complement,
    "complement_k": route_complement_k,
    "split_pole": route_split_pole,
}


# --- error bookkeeping ------------------------------------------------


def relative_error(value: np.ndarray, exact: np.ndarray) -> np.ndarray:
    """Return ``|value - exact| / |exact|``, differenced in extended precision.

    ``nan`` where the exact value vanishes -- ``B_R`` is identically zero in the
    ring's own plane, and a relative error there means nothing.
    """
    exact = np.asarray(exact, dtype=LD)
    with np.errstate(invalid="ignore", divide="ignore"):
        error = np.abs(np.asarray(value, dtype=LD) - exact) / np.abs(exact)
    return np.where(exact == 0.0, np.nan, error).astype(float)


def crossing(
    coordinate: np.ndarray, error: np.ndarray, tolerance: float
) -> float | None:
    """Return the coordinate at which ``error`` crosses ``tolerance`` for good.

    Both sweeps are ordered from their clean end to their bad end -- the near-field
    one inward, the far-field one outward -- so the crossing is the first sample
    whose whole remaining SUFFIX is also above the tolerance.  A single sample over
    the line with clean samples beyond it is noise, not a crossing.  ``nan`` errors
    count as exceeding: a route that returns ``nan`` has not met the tolerance.
    ``None`` means the tolerance held at every sample -- or that the component is
    identically zero along this ray, which ``B_R`` is in the ring's own plane and
    where a relative error is undefined rather than failing.
    """
    error = np.asarray(error, dtype=float)
    if np.all(np.isnan(error)):
        return None
    error = np.where(np.isnan(error), np.inf, error)
    above = np.flatnonzero(error >= tolerance)
    held = [index for index in above if np.all(error[index:] >= tolerance)]
    return float(np.asarray(coordinate)[held[0]]) if held else None


# --- part one: accuracy against distance to the filament --------------

DIRECTIONS = {
    "radial": lambda ar, d: (ar + d, np.zeros_like(d)),
    "vertical": lambda ar, d: (np.full_like(d, ar), d),
    "diagonal": lambda ar, d: (ar + d / np.sqrt(2.0), d / np.sqrt(2.0)),
}
"""Approach directions, all outward from the filament.  ``radial`` stays in the
ring's plane, where ``B_R`` vanishes identically; ``diagonal`` is there so all
three components are measured on a ray that is neither."""

OFFSETS = np.geomspace(0.5, 1.0e-12, 60)
"""Approach distances in RING RADII, so the two ring sizes are directly
comparable and scale invariance can be read off the abscissa."""


def near_field(radius: float) -> dict:
    """Return per-route relative error against distance for one ring radius."""
    out = {}
    for name, place in DIRECTIONS.items():
        offset = OFFSETS * radius
        target_r, target_z = place(radius, offset)
        target_r = np.asarray(target_r, dtype=np.float64)
        target_z = np.asarray(target_z, dtype=np.float64)
        # the REALISED distance: a + d rounds, and at d/a ~ 1e-12 the rounding is
        # parts in ten thousand of d, so the requested offset is not the measured one
        distance = np.hypot(
            np.asarray(target_r, dtype=LD) - LD(radius), np.asarray(target_z, dtype=LD)
        )
        exact = reference(target_r, target_z, radius, 0.0)
        k2, kc2 = modulus(target_r, target_z, radius, 0.0)
        record = {
            "distance_m": distance.astype(float).tolist(),
            "distance_ring_radii": (distance / LD(radius)).astype(float).tolist(),
            "kc2": kc2.tolist(),
            "clipped": (k2 > 1.0 - CLIP).tolist(),
            "route": {},
        }
        for route, evaluate in ROUTES.items():
            value = evaluate(target_r, target_z, radius, 0.0)
            record["route"][route] = {
                component: relative_error(value[index], exact[index]).tolist()
                for index, component in enumerate(COMPONENTS)
            }
            for key, coordinate in (
                ("crossing_m", distance.astype(float)),
                ("crossing_ring_radii", (distance / LD(radius)).astype(float)),
            ):
                record["route"][route][key] = {
                    component: {
                        f"{tolerance:.0e}": crossing(
                            coordinate,
                            np.asarray(record["route"][route][component]),
                            tolerance,
                        )
                        for tolerance in TOLERANCES
                    }
                    for component in COMPONENTS
                }
        out[name] = record
    return out


def clip_engagement(sweep: dict) -> dict:
    """Return each route's error where the shipped cap first engages, per direction."""
    out = {}
    for direction, record in sweep.items():
        clipped = np.flatnonzero(record["clipped"])
        if not clipped.size:
            continue
        index = int(clipped[0])  # distances descend, so this is the outermost
        out[direction] = {
            "distance_m": record["distance_m"][index],
            "distance_ring_radii": record["distance_ring_radii"][index],
            "kc2": record["kc2"][index],
            **{
                route: {component: values[component][index] for component in COMPONENTS}
                for route, values in record["route"].items()
            },
        }
    return out


# --- part two: the flux bracket's far-field cancellation --------------

FAR_PARAMETERS = np.geomspace(0.5, 1.0e-8, 40)
"""``k^2`` values for the far-field sweep.  A target three ring radii away sits at
``k^2 = 4/13 = 0.31``, so the physical regime is at the TOP of this range."""


def far_field(radius: float) -> dict:
    """Return the flux bracket's relative error against ``k^2``, two ways out.

    ``k^2`` is small for two quite different targets and the distinction matters
    for whether this bites: far from the ring in ``z``, where ``k^2 ~ 4 a^2/dz^2``
    and small means genuinely remote; and close to the SYMMETRY AXIS, where
    ``k^2 ~ 4 R/a`` and small means a target one ring radius away that happens to
    sit near ``R = 0``.
    """
    out = {}
    parameter = FAR_PARAMETERS
    places = {
        # r = a, dz set so that 4 a^2 / (4 a^2 + dz^2) = k^2
        "vertical": (
            np.full_like(parameter, radius),
            2.0 * radius * np.sqrt((1.0 - parameter) / parameter),
        ),
        # z = 0, R the small root of 4 a R / (a + R)^2 = k^2, in its stable form
        "axis_side": (
            radius * parameter / (2.0 - parameter + 2.0 * np.sqrt(1.0 - parameter)),
            np.zeros_like(parameter),
        ),
    }
    for name, (target_r, target_z) in places.items():
        target_r = np.asarray(target_r, dtype=np.float64)
        target_z = np.asarray(target_z, dtype=np.float64)
        exact = reference(target_r, target_z, radius, 0.0)
        k2, _ = modulus(target_r, target_z, radius, 0.0)
        distance = np.hypot(target_r - radius, target_z) / radius
        record = {
            "k2": k2.tolist(),
            "distance_ring_radii": distance.tolist(),
            "route": {},
        }
        for route, evaluate in ROUTES.items():
            value = evaluate(target_r, target_z, radius, 0.0)
            errors = {
                component: relative_error(value[index], exact[index]).tolist()
                for index, component in enumerate(COMPONENTS)
            }
            record["route"][route] = {
                **errors,
                **{
                    key: {
                        component: {
                            f"{tolerance:.0e}": crossing(
                                coordinate, np.asarray(errors[component]), tolerance
                            )
                            for tolerance in TOLERANCES
                        }
                        for component in COMPONENTS
                    }
                    for key, coordinate in (
                        ("crossing_ring_radii", distance),
                        ("crossing_k2", k2),
                    )
                },
            }
        out[name] = record
    return out


# --- named configurations ---------------------------------------------


def configurations() -> dict:
    """Return per-route error at the geometries the spine actually evaluates.

    A distance sweep answers "where does it break"; this answers "does it break
    anywhere we look".  Each entry is a target the spine forms in normal use.
    """
    cases = {
        "plasma cell, own edge (a=6.2, d=60mm)": (6.2, 6.26, 0.0),
        "plasma cell, next cell (a=6.2, d=120mm)": (6.2, 6.32, 0.0),
        "probe 10mm from a coil (a=6.2)": (6.2, 6.21, 0.0),
        "probe 1mm from a coil (a=6.2)": (6.2, 6.201, 0.0),
        "probe 10um from a coil (a=6.2)": (6.2, 6.20001, 0.0),
        "winding pack, 1um (a=1.0)": (1.0, 1.000001, 0.0),
        "grid edge near the axis (a=6.2, R=0.1)": (6.2, 0.1, 0.0),
        "sensor at 3 ring radii (a=6.2)": (6.2, 6.2, 18.6),
        "sensor at 10 ring radii (a=6.2)": (6.2, 6.2, 62.0),
    }
    out = {}
    for label, (radius, target_r, target_z) in cases.items():
        target_r = np.array([target_r])
        target_z = np.array([target_z])
        exact = reference(target_r, target_z, radius, 0.0)
        k2, kc2 = modulus(target_r, target_z, radius, 0.0)
        out[label] = {
            "ring_radius_m": radius,
            "distance_m": float(np.hypot(target_r[0] - radius, target_z[0])),
            "k2": float(k2[0]),
            "kc2": float(kc2[0]),
            "clipped": bool(k2[0] > 1.0 - CLIP),
            "reference": {
                component: float(exact[index][0])
                for index, component in enumerate(COMPONENTS)
            },
            "route": {
                route: {
                    component: float(
                        relative_error(
                            evaluate(target_r, target_z, radius, 0.0)[index],
                            exact[index],
                        )[0]
                    )
                    for index, component in enumerate(COMPONENTS)
                }
                for route, evaluate in ROUTES.items()
            },
        }
    return out


def coincident() -> dict:
    """Return what every route yields for a target ON the filament, and one ULP off.

    The field is genuinely singular there, so there is no right ANSWER -- but
    there is a right CONTRACT, and today the three routes disagree about it.  One
    ULP off is included because ``4 a R`` and ``(a + R)^2`` round independently,
    so the parameter can land ABOVE one for a target that is merely adjacent.
    """
    out = {}
    for radius in RADII:
        for label, target_r in {
            "on the filament": radius,
            "one ulp inside": np.nextafter(radius, 0.0),
            "one ulp outside": np.nextafter(radius, np.inf),
        }.items():
            target = np.array([float(target_r)])
            zero = np.zeros(1)
            k2, kc2 = modulus(target, zero, radius, 0.0)
            out[f"a={radius}, {label}"] = {
                "k2_minus_one": float(k2[0] - 1.0),
                "kc2": float(kc2[0]),
                "route": {
                    route: {
                        component: float(evaluate(target, zero, radius, 0.0)[index][0])
                        for index, component in enumerate(COMPONENTS)
                    }
                    for route, evaluate in ROUTES.items()
                },
            }
    return out


# --- the downstream stencil -------------------------------------------

CELL_RADIUS = 0.06
"""Plasma cell circumradius [m], matching ``benchmarks/nearfar_error.py``."""

MOMENT_STEP = 2.0e-3
"""Source-position step in section radii, matching ``greens.py::_MOMENT_STEP``."""


def _worst_per_band(
    value: np.ndarray, exact: np.ndarray, local: np.ndarray, bands: int
) -> list[float]:
    """Return the worst error over directions on each standoff band.

    Worst rather than mean, over all directions on the band: the finite-area
    correction is not isotropic, and a band is only as good as its worst ray.
    """
    shape = (bands, -1)
    error = np.abs(value.reshape(shape) - exact.reshape(shape)) / local.reshape(shape)
    return [float(x) for x in np.max(error, axis=1)]


def moment_sensitivity(radius: float = 6.2) -> dict:
    """Return the corrected filament's error per route, against the exact section.

    ``moment_filament`` -- the far field of the banded coupling scheme -- takes
    SECOND differences of the point kernels in the SOURCE position on a step of
    ``2e-3`` section radii, then divides by ``step^2``.  That multiplies whatever
    absolute error the kernel carries by ``2.5e5 / cell^2``, so whether a route
    change reaches the banded scheme is not something the kernel's own relative
    error settles.  This measures it directly: the same stencil and the same
    moments, driven by each route in turn, against the exact polygon.

    A regular hexagon has no third moments, so the quadrupole stencil is the whole
    correction and nine evaluations cover it.
    """
    from nova.biot.polygon import polygon_greens  # local: the figure path skips it

    angle = np.pi / 6 + np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)
    section = np.column_stack(
        [radius + CELL_RADIUS * np.cos(angle), CELL_RADIUS * np.sin(angle)]
    )
    centre = section_centroid(section)
    irr, izz, irz = second_moments(section)
    step = MOMENT_STEP * CELL_RADIUS

    ring = np.geomspace(1.5, 30.0, 12)
    direction = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
    offset = (ring[:, None] * CELL_RADIUS).repeat(direction.size, axis=1)
    target_r = (radius + offset * np.cos(direction)).ravel()
    target_z = (offset * np.sin(direction)).ravel()

    exact = dict(
        zip(("psi", "br", "bz"), polygon_greens(target_r, target_z, section, block=32))
    )
    magnitude = np.hypot(exact["br"], exact["bz"])
    local = {"psi": np.abs(exact["psi"]), "bz": magnitude, "br": magnitude}

    out = {"ring_section_radii": ring.tolist(), "route": {}}
    for route, evaluate in ROUTES.items():

        def at(dr: int, dz: int) -> np.ndarray:
            return np.array(
                evaluate(
                    target_r, target_z, centre[0] + dr * step, centre[1] + dz * step
                )
            )

        value = at(0, 0)
        correction = 0.5 * (
            irr * (at(1, 0) - 2.0 * value + at(-1, 0))
            + izz * (at(0, 1) - 2.0 * value + at(0, -1))
            + 0.5 * irz * (at(1, 1) - at(1, -1) - at(-1, 1) + at(-1, -1))
        )
        corrected = value + correction / step**2
        out["route"][route] = {
            component: _worst_per_band(
                corrected[index], exact[component], local[component], ring.size
            )
            for index, component in enumerate(COMPONENTS)
        }
    return out


# --- validation of the reference itself --------------------------------


def validate() -> dict:
    """Return the checks that license the reference, each as the number measured.

    A reference nobody checked is not a reference.  Four independent handles:
    against the Carlson forms in double where they are trustworthy; against the
    parameter forms at the easy end of the range; against the closed values at
    the endpoints; and the series against the direct combination in the band
    where BOTH are accurate, which is what licenses the series below it.
    """
    moderate = np.array([1e-8, 1e-6, 1e-4, 1e-2, 0.1, 0.3, 0.5, 0.7, 0.9])
    big_k, big_e = complete_agm(moderate)
    zero, one = np.zeros_like(moderate), np.ones_like(moderate)
    carlson_k = scipy.special.elliprf(zero, moderate, one)
    carlson_e = 2.0 * scipy.special.elliprg(zero, moderate, one)

    easy = np.array([0.5, 0.7, 0.9, 0.99])  # kc2 near one, where the parameter is safe
    parameter_k = scipy.special.ellipk(1.0 - easy)
    parameter_e = scipy.special.ellipe(1.0 - easy)
    easy_k, easy_e = complete_agm(easy)

    endpoint_k, endpoint_e = complete_agm(np.array([1.0, 1e-30]))
    # the band is named by its COMPLEMENT and the parameter derived in extended
    # precision, not the other way round: a double 1 - m would hand the check the
    # very rounding it is meant to be free of, and reports 2e-15 instead of 1e-17
    band_kc2 = LD(np.array([0.65, 0.6, 0.55, 0.51]))
    band = LD(1.0) - band_kc2
    band_k, band_e = complete_agm(band_kc2)

    def series_gap(exact_series, direct):
        return float(np.max(np.abs((exact_series - direct) / direct)))

    return {
        "against_carlson_double": {
            "kc2": moderate.tolist(),
            "K_relative": (np.abs(big_k - carlson_k) / big_k).astype(float).tolist(),
            "E_relative": (np.abs(big_e - carlson_e) / big_e).astype(float).tolist(),
            "worst": float(
                max(
                    np.max(np.abs(big_k - carlson_k) / big_k),
                    np.max(np.abs(big_e - carlson_e) / big_e),
                )
            ),
        },
        "against_parameter_double_easy_end": {
            "kc2": easy.tolist(),
            "worst": float(
                max(
                    np.max(np.abs(easy_k - parameter_k) / easy_k),
                    np.max(np.abs(easy_e - parameter_e) / easy_e),
                )
            ),
        },
        "endpoints": {
            "K(kc2=1) - pi/2": float(endpoint_k[0] - PI / 2),
            "E(kc2=1) - pi/2": float(endpoint_e[0] - PI / 2),
            "E(kc2=1e-30) - 1": float(endpoint_e[1] - 1.0),
            "K(kc2=1e-30)": float(endpoint_k[1]),
            "K(kc2=1e-30) - log(4/k')": float(
                endpoint_k[1] - np.log(LD(4.0) / np.sqrt(LD("1e-30")))
            ),
        },
        "series_against_direct_in_band": {
            "m": band.astype(float).tolist(),
            "flux_bracket_relative": series_gap(
                PI / LD(2.0) * _series(band, _flux_weight),
                (LD(1.0) - LD(0.5) * LD(band)) * band_k - band_e,
            ),
            "kind_difference_relative": series_gap(
                PI / LD(2.0) * _series(band, _difference_weight), band_k - band_e
            ),
        },
        "agm_iterations_deepest_complement": _agm_iterations(1e-30),
        "longdouble_eps": float(np.finfo(LD).eps),
        "pole_floor_engages_at_m": float(np.sqrt(_R_FLOOR)),
        "clip_engages_at_kc2": CLIP,
        "shipped_kernel_pinned": _pin_shipped(),
    }


def _agm_iterations(kc2: float) -> int:
    """Return the mean iteration count at one complement, as a convergence witness."""
    complete_agm(np.array([kc2]))
    return int(getattr(complete_agm, "iterations", 0))


def _pin_shipped() -> dict:
    """Return every route's largest gap from the shipped kernels.

    The routes below are a controlled comparison only if ONE of them is the shipped
    kernel -- otherwise a difference between two of them could be anything.  Which
    one that is has changed: it was ``route_parameter`` when this was written and is
    ``route_split_pole`` now, so both gaps are reported.  The split-pole gap says
    which route ships (round-off, not zero: the shipped kernel forms the same
    quantities in a different order); the parameter gap is the near-field error the
    adoption removed, and it is only small out here where nothing is close.
    """
    rng = np.random.default_rng(4)
    target_r = 6.2 + rng.uniform(-2.0, 2.0, 4096)
    target_z = rng.uniform(-2.0, 2.0, 4096)
    shipped_psi = greens_psi(target_r, target_z, 6.2, 0.0)
    shipped_bz, shipped_br = greens_bz_br(target_r, target_z, 6.2, 0.0)
    gaps = {}
    for name, route in (
        ("parameter", route_parameter),
        ("split_pole", route_split_pole),
    ):
        psi, bz, br = route(target_r, target_z, 6.2, 0.0)
        gaps[name] = {
            "psi_max_abs_gap": float(np.max(np.abs(psi - shipped_psi))),
            "bz_max_abs_gap": float(np.max(np.abs(bz - shipped_bz))),
            "br_max_abs_gap": float(np.max(np.abs(br - shipped_br))),
        }
    return gaps


# --- part three: cost, one variant per fresh process ------------------

SIZES = (200_000, 2_000_000)
REPEATS = 3
INNER = 5


def timing_arrays(size: int):
    """Return targets in a box about an ITER-scale ring, and their moduli.

    A box rather than a modulus distribution: the special functions' iteration
    counts follow the geometry, and this is the geometry a grid evaluation hands
    them.
    """
    rng = np.random.default_rng(7)
    target_r = 6.2 + rng.uniform(-1.5, 1.5, size)
    target_z = rng.uniform(-1.5, 1.5, size)
    k2, kc2 = modulus(target_r, target_z, 6.2, 0.0)
    return target_r, target_z, np.clip(k2, 0.0, 1.0 - CLIP), kc2


def time_variant(name: str, size: int) -> float:
    """Return microseconds per element for one variant, median of :data:`INNER`."""
    target_r, target_z, k2, kc2 = timing_arrays(size)
    zero, one = np.zeros_like(kc2), np.ones_like(kc2)
    variants = {
        "ellipk_ellipe": lambda: (scipy.special.ellipk(k2), scipy.special.ellipe(k2)),
        "ellipkm1_ellipe": lambda: (
            scipy.special.ellipkm1(kc2),
            scipy.special.ellipe(1.0 - kc2),
        ),
        "carlson_rf_rg": lambda: (
            scipy.special.elliprf(zero, kc2, one),
            2.0 * scipy.special.elliprg(zero, kc2, one),
        ),
        "greens_psi": lambda: greens_psi(target_r, target_z, 6.2, 0.0),
        "greens_bz_br": lambda: greens_bz_br(target_r, target_z, 6.2, 0.0),
        "route_parameter": lambda: route_parameter(target_r, target_z, 6.2, 0.0),
        "route_complement": lambda: route_complement(target_r, target_z, 6.2, 0.0),
        "route_complement_k": lambda: route_complement_k(target_r, target_z, 6.2, 0.0),
    }
    run = variants[name]
    run()  # warm the pages and any first-call setup out of the measurement
    elapsed = []
    for _ in range(INNER):
        start = time.perf_counter()
        run()
        elapsed.append(time.perf_counter() - start)
    return 1e6 * float(np.median(elapsed)) / size


VARIANTS = (
    "ellipk_ellipe",
    "ellipkm1_ellipe",
    "carlson_rf_rg",
    "greens_psi",
    "greens_bz_br",
    "route_parameter",
    "route_complement",
    "route_complement_k",
)


def cost() -> dict:
    """Return microseconds per element per variant, median over fresh processes.

    One variant per process, because a route that has already run has warmed the
    allocator and the special-function tables for the next one, and the effect is
    larger than the difference being measured.
    """
    out = {"sizes": list(SIZES), "repeats": REPEATS, "variant": {}}
    for name in VARIANTS:
        out["variant"][name] = {}
        for size in SIZES:
            runs = []
            for _ in range(REPEATS):
                finished = subprocess.run(
                    [sys.executable, __file__, "--time-variant", name, str(size)],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                runs.append(float(finished.stdout.strip().splitlines()[-1]))
            out["variant"][name][str(size)] = {
                "us_per_element": float(np.median(runs)),
                "runs": runs,
            }
    baseline = out["variant"]["ellipk_ellipe"]
    for name in VARIANTS:
        for size in SIZES:
            record = out["variant"][name][str(size)]
            record["ratio_to_ellipk_ellipe"] = (
                record["us_per_element"] / baseline[str(size)]["us_per_element"]
            )
    return out


# --- part four: the downstream regression baseline --------------------


def baseline(destination: pathlib.Path) -> dict:
    """Return ``benchmarks/nearfar_error.py``'s current output, unmodified.

    ``moment_filament`` -- the far field of the banded coupling scheme -- is built
    out of these same two point kernels by central differences on the SOURCE
    position, so anything done to them lands here.  Captured before any change,
    so the orchestrator has a before to compare against.
    """
    target = destination / "nearfar_error_baseline.json"
    command = [sys.executable, "benchmarks/nearfar_error.py", str(target)]
    finished = subprocess.run(
        command,
        capture_output=True,
        text=True,
        cwd=str(pathlib.Path(__file__).resolve().parent.parent),
        check=True,
    )
    record = json.loads(target.read_text())
    return {
        "command": " ".join(command),
        "stdout": finished.stdout,
        "cutoff_section_radii": record["cutoff_section_radii"],
        "correction": record["correction"],
        "ring": record["ring"],
        "scale": record["scale"],
    }


# --- assembly ---------------------------------------------------------


def measure(*, quick: bool = False, destination: pathlib.Path | None = None) -> dict:
    """Return the whole study: validation, accuracy, cost, and the baseline."""
    out = {
        "validation": validate(),
        "near_field": {str(radius): near_field(radius) for radius in RADII},
        "far_field": {str(radius): far_field(radius) for radius in RADII},
        "configurations": configurations(),
        "coincident": coincident(),
        "moment_sensitivity": moment_sensitivity(),
        "tolerances": list(TOLERANCES),
        "clip": CLIP,
    }
    out["clip_engagement"] = {
        str(radius): clip_engagement(out["near_field"][str(radius)]) for radius in RADII
    }
    if not quick:
        out["cost"] = cost()
        out["baseline"] = baseline(destination or pathlib.Path.cwd())
    return out


COLOURS = {
    "split_pole": "C4",
    "parameter": "C3",
    "parameter_unclipped": "C1",
    "complement": "C0",
    "complement_k": "C2",
}


def figure(data: dict, path: pathlib.Path) -> None:
    """Render the study: near-field error, the far-field bracket, and cost."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(21.0, 9.4))

    panels = (("psi", "radial"), ("bz", "vertical"), ("br", "diagonal"))
    for axis, (component, direction) in zip(axes[0], panels):
        for radius, style in zip(RADII, ("-", "--")):
            record = data["near_field"][str(radius)][direction]
            for route, values in record["route"].items():
                # inf and nan are plotted at the ceiling rather than dropped: a
                # route that returns them has failed, and a gap in the line reads
                # as no data
                error = np.asarray(values[component], dtype=float)
                axis.plot(
                    record["distance_ring_radii"],
                    np.clip(np.nan_to_num(error, nan=10.0, posinf=10.0), 1e-18, 10.0),
                    style,
                    color=COLOURS[route],
                    lw=1.2,
                    label=f"{route}, a={radius} m" if component == "psi" else None,
                )
        engagement = data["clip_engagement"][str(RADII[0])].get(direction)
        if engagement:
            axis.axvline(engagement["distance_ring_radii"], color="k", lw=0.8, ls="-.")
            axis.text(
                engagement["distance_ring_radii"] * 1.5,
                2e-17,
                "modulus cap",
                fontsize=7,
                rotation=90,
            )
        if component != "psi":  # the pole floor reaches only the field components
            axis.axvline(np.sqrt(_R_FLOOR) / RADII[0], color="0.4", lw=0.8, ls="--")
            axis.text(
                np.sqrt(_R_FLOOR) / RADII[0] * 1.5,
                2e-17,
                "pole floor",
                fontsize=7,
                rotation=90,
            )
        for tolerance in data["tolerances"]:
            axis.axhline(tolerance, color="k", ls=":", lw=0.8)
            axis.text(4e-13, tolerance * 1.5, f"{tolerance:.0e}", fontsize=7)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("distance to filament / ring radius [-]")
        axis.set_title(f"{component}, approaching {direction}")
        axis.grid(alpha=0.3)
    axes[0, 0].set_ylabel("relative error [-]")
    axes[0, 0].legend(fontsize=6, ncol=2, loc="upper right")

    axis = axes[0, 3]
    ring = np.asarray(data["moment_sensitivity"]["ring_section_radii"], dtype=float)
    for route, values in data["moment_sensitivity"]["route"].items():
        if route == "parameter_unclipped":
            continue  # indistinguishable from the clipped route at these standoffs
        for component, style in zip(("psi", "bz"), ("-", "--")):
            axis.plot(
                ring,
                np.maximum(np.asarray(values[component], dtype=float), 1e-18),
                style,
                color=COLOURS[route],
                lw=1.1,
                label=f"{route}, {component}",
            )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axis.set_xlabel("standoff / section radius [-]")
    axis.set_ylabel("error vs exact section, relative to local magnitude [-]")
    axis.set_title(
        "downstream: quadrupole-corrected filament\n"
        "vs exact hexagon (a=6.2 m) -- every route agrees"
    )
    axis.grid(alpha=0.3)
    axis.legend(fontsize=6, ncol=2)

    for axis, (direction, label) in zip(
        axes[1],
        (("vertical", "far in z"), ("axis_side", "near the symmetry axis")),
    ):
        record = data["far_field"][str(RADII[1])][direction]
        for component, marker in zip(COMPONENTS, ("o", "s", "^")):
            error = np.asarray(record["route"]["parameter"][component], dtype=float)
            if np.all(np.isnan(error)):
                continue  # B_R vanishes identically in the ring's own plane
            axis.plot(
                record["k2"],
                np.maximum(error, 1e-18),
                marker=marker,
                ms=3,
                lw=1.2,
                label=component,
            )
        k2 = np.asarray(record["k2"], dtype=float)
        axis.plot(
            k2,
            16.0 * np.finfo(float).eps / k2**2,
            "k--",
            lw=0.8,
            label=r"$16\,\epsilon/k^4$",
        )
        if direction == "vertical":
            # the field bracket's own O(m) cancellation, which only the receding
            # branch shows: near the axis its coefficient tends to +1 and the E term
            # no longer cancels against K at all
            axis.plot(
                k2,
                2.0 * np.finfo(float).eps / k2,
                "k:",
                lw=0.8,
                label=r"$2\epsilon/k^2$",
            )
        for tolerance in data["tolerances"]:
            axis.axhline(tolerance, color="0.6", ls=":", lw=0.8)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel(r"$k^2$ [-]")
        axis.set_title(f"far-field bracket, {label} (a=6.2 m)")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=7)
    axes[1, 0].set_ylabel("relative error [-]")

    axis = axes[1, 3]
    for radius, marker in zip(RADII, ("o", "x")):
        record = data["near_field"][str(radius)]["diagonal"]
        kc2 = np.asarray(record["kc2"], dtype=float)
        for route in ("parameter", "complement_k"):
            axis.plot(
                kc2,
                np.maximum(
                    np.asarray(record["route"][route]["psi"], dtype=float), 1e-18
                ),
                marker=marker,
                ms=3,
                lw=1.0,
                color=COLOURS[route],
                label=f"{route}, a={radius} m",
            )
    kc2 = np.geomspace(1e-24, 1e-1, 60)
    axis.plot(
        kc2,
        0.5 * np.finfo(float).eps / (kc2 * (0.5 * np.log(16.0 / kc2))),
        "k--",
        lw=0.8,
        label=r"$\epsilon\,/\,2 k'^2 K$",
    )
    axis.axvline(CLIP, color="k", lw=0.8, ls="-.")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel(r"complement $k'^2 = d^2/\mathrm{denom}$ [-]")
    axis.set_title(
        "the flux error follows $\\epsilon/k'^2$,\nand collapses on the complement"
    )
    axis.grid(alpha=0.3)
    axis.legend(fontsize=6)

    axis = axes[1, 2]
    if "cost" in data:
        size = str(SIZES[-1])
        names = list(VARIANTS)
        values = [data["cost"]["variant"][n][size]["us_per_element"] for n in names]
        axis.barh(range(len(names)), values, color="0.6")
        for index, (name, value) in enumerate(zip(names, values)):
            ratio = data["cost"]["variant"][name][size]["ratio_to_ellipk_ellipe"]
            axis.text(
                value, index, f"  {value:.4f} ({ratio:.2f}x)", va="center", fontsize=7
            )
        axis.set_yticks(range(len(names)))
        axis.set_yticklabels(names, fontsize=7)
        axis.set_xlabel(f"cost [us/element], {SIZES[-1]} elements")
        axis.set_xlim(0, 1.35 * max(values))
        axis.set_title("cost per element, fresh process, compute node")
        axis.grid(alpha=0.3, axis="x")
    else:
        axis.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    """Write the study to JSON, and the figure if one is asked for."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output",
        type=pathlib.Path,
        nargs="?",
        help="JSON destination, not needed by the timing entry point",
    )
    parser.add_argument("--figure", type=pathlib.Path, help="render the study here")
    parser.add_argument(
        "--quick", action="store_true", help="skip cost and baseline (login-node use)"
    )
    parser.add_argument(
        "--time-variant",
        nargs=2,
        metavar=("NAME", "SIZE"),
        help="time one variant and print us/element -- the fresh-process entry point",
    )
    arguments = parser.parse_args()
    if arguments.time_variant:
        name, size = arguments.time_variant
        print(f"{time_variant(name, int(size)):.6f}")
        return
    if arguments.output is None:
        parser.error("an output path is required")
    result = measure(quick=arguments.quick, destination=arguments.output.parent)
    arguments.output.write_text(json.dumps(result, indent=2))
    if arguments.figure:
        figure(result, arguments.figure)
    report(result)


def report(data: dict) -> None:
    """Print the headline numbers, so a run is readable without opening the JSON."""
    print("reference validation")
    for key, value in data["validation"].items():
        if isinstance(value, dict):
            summary = value.get("worst", value)
            print(f"  {key:38s} {summary}")
        else:
            print(f"  {key:38s} {value}")
    for radius in RADII:
        print(f"\ncrossing distances [m], diagonal approach, a = {radius} m")
        sweep = data["near_field"][str(radius)]["diagonal"]
        for route in ROUTES:
            for component in COMPONENTS:
                crossings = sweep["route"][route]["crossing_m"][component]
                print(
                    f"  {route:22s} {component:4s} "
                    + "  ".join(
                        f"{key}: {'clean' if value is None else format(value, '.3e')}"
                        for key, value in crossings.items()
                    )
                )
    if "cost" in data:
        print(f"\ncost [us/element], {SIZES[-1]} elements")
        for name in VARIANTS:
            record = data["cost"]["variant"][name][str(SIZES[-1])]
            print(
                f"  {name:22s} {record['us_per_element']:9.4f}"
                f"   {record['ratio_to_ellipk_ellipe']:6.2f} x"
            )


if __name__ == "__main__":
    main()
