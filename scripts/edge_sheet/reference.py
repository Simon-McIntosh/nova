"""Reference and strip evaluation for a toroidal sheet on a poloidal segment.

A magnetised cell edge carries a uniform toroidal sheet current ``K_phi`` along a
straight segment. Its field is a line integral of nova's exact ring Green's
functions over the segment. This module owns the reference and strip routes:

``sheet_reference``
    Gauss-Legendre along the segment, refined by panel bisection until two
    successive refinements agree to ``1e-13`` relative at the target, with the
    initial panels graded geometrically toward the target's foot on the segment.

``graded_precise_ring_reference``
    Independently refined elliptic ring kernels, source interpolation and weighted
    accumulation at fifty decimal digits, returning ``(psi, B_R, B_Z)``.

``strip_greens``
    The Urankar Part V route: the segment widened into a rectangle of width
    ``eps`` carrying sheet density ``K/eps``, evaluated by
    :func:`nova.biot.polygonanalytic.polygon_analytic_greens`.

``sheet_error``
    The relative difference between the two, per component.

``sheet_reference`` and ``strip_greens`` return ``(psi, B_Z, B_R)`` for unit
sheet current density and unit total current respectively, so the strip value is
``L`` times the per-ampere polygon
Green's function and the reference is the length integral of the per-ampere ring
Green's function -- the same physical current either way.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.polynomial.legendre import leggauss

from nova.biot.greens import greens_bz_br, greens_psi
from nova.biot.polygonanalytic import polygon_analytic_greens

__all__ = [
    "Convergence",
    "Target",
    "segment_frame",
    "sheet_error",
    "sheet_reference",
    "graded_precise_ring_reference",
    "strip_greens",
]


@dataclass(frozen=True)
class Convergence:
    """What the reference refinement did at one target."""

    psi: float
    bz: float
    br: float
    rounds: int
    panels: int
    achieved: float
    converged: bool
    history: tuple[float, ...] = field(default_factory=tuple)


def segment_frame(start: np.ndarray, end: np.ndarray):
    """Return ``(centre, unit_direction, unit_normal, length)`` of the segment."""
    a = np.asarray(start, dtype=np.float64)
    b = np.asarray(end, dtype=np.float64)
    span = b - a
    length = float(np.hypot(*span))
    direction = span / length
    # rotate the direction a quarter turn: the normal is the segment's
    # broadside, so an off-middle target sits on one side regardless of slope
    normal = np.array([direction[1], -direction[0]])
    return 0.5 * (a + b), direction, normal, length


@dataclass(frozen=True)
class Target:
    """One fixed observation point, named by distance and station."""

    distance: float
    station: str
    r: float
    z: float
    foot: float
    degenerate: dict = field(default_factory=dict)

    @property
    def point(self) -> np.ndarray:
        return np.array([self.r, self.z])


def target_set(
    centre: np.ndarray,
    direction: np.ndarray,
    normal: np.ndarray,
    length: float,
    *,
    cell_width: float,
    distances: tuple[float, ...],
) -> list[Target]:
    """Targets at ``distances`` off the middle and off each end of the segment.

    ``foot`` is the fraction along the segment nearest the target, which is
    where the reference quadrature clusters its panels.
    """
    centre = np.asarray(centre, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    normal = np.asarray(normal, dtype=np.float64)
    start = centre - 0.5 * length * direction
    end = centre + 0.5 * length * direction
    targets = []
    for distance in distances:
        for station in ("middle", "start", "end"):
            if station == "middle":
                point = centre + distance * normal
                foot = 0.5
            elif station == "start":
                point = start - distance * direction
                foot = 0.0
            else:
                point = end + distance * direction
                foot = 1.0
            targets.append(
                Target(
                    distance=float(distance),
                    station=station,
                    r=float(point[0]),
                    z=float(point[1]),
                    foot=foot,
                )
            )
    return targets


def _geometric_panels(foot: float, per_side: int, ratio: float):
    """Panels covering ``[0, 1]``, graded geometrically toward ``foot``."""
    panels = []
    for edge in (0.0, 1.0):
        span = abs(edge - foot)
        if span <= 0.0:
            continue
        # widths grow by ``ratio`` away from the foot and sum to the span
        first = span * (ratio - 1.0) / (ratio**per_side - 1.0)
        x = foot
        for k in range(per_side):
            width = first * ratio**k
            other = x + np.sign(edge - foot) * width
            panels.append((min(x, other), max(x, other)))
            x = other
    return np.array(panels, dtype=np.float64)


def _panel_integral(start, end, target, panels, nodes):
    """``(psi, B_Z, B_R)`` contributed by each panel, per unit sheet current."""
    x, w = leggauss(nodes)
    a = panels[:, 0:1]
    b = panels[:, 1:2]
    t = 0.5 * (b - a) * x[None, :] + 0.5 * (a + b)
    weight = (0.5 * (b - a) * w[None, :]) * np.hypot(*(end - start))
    points = start[None, None, :] + t[:, :, None] * (end - start)[None, None, :]
    rs = points[:, :, 0].ravel()
    zs = points[:, :, 1].ravel()
    wr = weight.ravel()
    # the ring Green's functions take the TARGET first and the source loop last;
    # this is the one place the order matters and it is not symmetric for the
    # field components
    psi = (wr * greens_psi(target.r, target.z, rs, zs)).reshape(panels.shape[0], -1)
    bz, br = greens_bz_br(target.r, target.z, rs, zs)
    bz = (wr * bz).reshape(panels.shape[0], -1)
    br = (wr * br).reshape(panels.shape[0], -1)
    return np.stack([psi.sum(axis=1), bz.sum(axis=1), br.sum(axis=1)], axis=1)


DEGENERATE = 1e-8
"""A component below this fraction of the largest is an exact symmetry zero."""


def _convergence_change(total, previous) -> float:
    """Largest relative change over the components that carry a value; else zero."""
    if previous is None:
        return float("inf")
    biggest = float(np.max(np.abs(total)))
    if biggest == 0.0:
        return 0.0
    active = np.abs(total) >= DEGENERATE * biggest
    if not np.any(active):
        return 0.0
    relative = np.abs(total - previous)[active] / np.abs(total)[active]
    return float(np.max(relative))


def sheet_reference(
    target: Target,
    start: np.ndarray,
    end: np.ndarray,
    *,
    per_side: int = 12,
    ratio: float = 1.7,
    nodes: int = 16,
    rounds: int = 24,
    tolerance: float = 1e-13,
    panel_budget: int = 200_000,
) -> Convergence:
    """Gauss-Legendre along the segment, bisected until successive rounds agree.

    The panels are graded geometrically toward the target's foot, so the
    integrand's scale there -- the target distance -- is resolved from the first
    round. Each round bisects every panel, which preserves that grading, and the
    round stops when the total agrees with the previous round to ``tolerance``
    relative, per component.
    """
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    panels = _geometric_panels(target.foot, per_side, ratio)
    values = _panel_integral(start, end, target, panels, nodes)
    total = values.sum(axis=0)
    history = []
    previous = None
    for round_index in range(rounds):
        if 2 * panels.shape[0] > panel_budget:
            break
        # split every panel at its midpoint; the children replace the parent
        middle = 0.5 * (panels[:, 0] + panels[:, 1])
        children = np.empty((2 * panels.shape[0], 2), dtype=np.float64)
        children[0::2, 0] = panels[:, 0]
        children[0::2, 1] = middle
        children[1::2, 0] = middle
        children[1::2, 1] = panels[:, 1]
        panels = children
        values = _panel_integral(start, end, target, panels, nodes)
        total = values.sum(axis=0)
        if previous is not None:
            # a component that is an exact symmetry zero (B_R on the mirror
            # plane of the segment, whose integrand is odd about the foot)
            # plateaus at a roundoff residue; its relative change is noise on
            # that residue and must not gate the refinement, so it is held out
            change = _convergence_change(total, previous)
            history.append(change)
            if change < tolerance:
                return Convergence(
                    psi=float(total[0]),
                    bz=float(total[1]),
                    br=float(total[2]),
                    rounds=round_index + 1,
                    panels=int(panels.shape[0]),
                    achieved=change,
                    converged=True,
                    history=tuple(history),
                )
        previous = total
    achieved = _convergence_change(total, previous)
    return Convergence(
        psi=float(total[0]),
        bz=float(total[1]),
        br=float(total[2]),
        rounds=rounds,
        panels=int(panels.shape[0]),
        achieved=achieved,
        converged=False,
        history=tuple(history),
    )


def strip_greens(
    target: Target,
    start: np.ndarray,
    end: np.ndarray,
    *,
    fraction: float,
    nodes: int | None = None,
) -> np.ndarray:
    """Part V strip value at ``target`` for ``eps = fraction * length``.

    The rectangle carries uniform azimuthal current density ``K/eps`` with
    ``K = 1``, so its total current is the segment length and the per-ampere
    polygon Green's function is scaled by that length, matching the reference's
    unit sheet density.
    """
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    centre, direction, normal, length = segment_frame(start, end)
    half = 0.5 * fraction * length
    along = 0.5 * length * direction
    across = half * normal
    vertices = np.array(
        [
            centre + along + across,
            centre + along - across,
            centre - along - across,
            centre - along + across,
        ]
    )
    kwargs = {} if nodes is None else {"nodes": nodes}
    psi, br, bz = polygon_analytic_greens(
        np.array([target.r]), np.array([target.z]), vertices, **kwargs
    )
    # total current = (K / eps) * (eps * length) = length, with K = 1
    return length * np.array([float(psi[0]), float(bz[0]), float(br[0])])


def sheet_error(
    strip: np.ndarray, reference: Convergence, floor: float = DEGENERATE
) -> tuple[float, dict]:
    """Worst non-degenerate component relative error, and the per-component map.

    A component whose reference value is below ``floor`` times the largest
    component magnitude is a symmetry zero: it carries no relative error to
    measure and is reported as degenerate rather than as a ratio of two roundoff
    residues.
    """
    ref = np.array([reference.psi, reference.bz, reference.br])
    biggest = float(np.max(np.abs(ref)))
    detail = {}
    worst = 0.0
    for name, value, other in zip(("psi", "bz", "br"), ref, strip, strict=True):
        if biggest == 0.0 or abs(value) < floor * biggest:
            detail[name] = None
            continue
        relative = float(abs(other - value) / abs(value))
        detail[name] = relative
        worst = max(worst, relative)
    return worst, detail


def graded_precise_ring_reference(start, end, point, *, rtol=1e-13):
    """Integrate independent 50-digit ring kernels, returning (psi, B_R, B_Z).

    Short distant edges need one panel; nearby edges grade toward the target's
    orthogonal foot. Successive Gaussian orders and panel bisection must agree.
    The axis kernel is evaluated by its elementary ring limit.
    """
    import mpmath as mp

    from nova.biot.greens import MU0

    start, end, point = map(lambda x: np.asarray(x, dtype=float), (start, end, point))
    delta = end - start
    length = np.linalg.norm(delta)
    if start[0] == end[0] == 0:
        return np.zeros(3)
    foot = float(np.clip(np.dot(point - start, delta) / length**2, 0, 1))
    distance = np.linalg.norm(point - start - foot * delta)
    if distance == 0:
        raise ValueError("the target lies on the closed source segment")
    panels = (
        np.array([[0.0, 1.0]])
        if distance >= length
        else _geometric_panels(foot, 8, 1.7)
    )
    previous = None
    with mp.workdps(50):
        r, z = map(mp.mpf, point)
        ar, az = map(mp.mpf, start)
        dr, dz = map(mp.mpf, delta)
        mu = mp.mpf(MU0)
        for refinement in range(8):
            nodes, weights = leggauss(min(4 * 2**refinement, 32))
            accumulated = [mp.mpf(0), mp.mpf(0), mp.mpf(0)]
            for lower, upper in panels:
                lo, hi = mp.mpf(lower), mp.mpf(upper)
                for node, weight in zip(nodes, weights):
                    t = (lo + hi) / 2 + (hi - lo) / 2 * mp.mpf(node)
                    a, h = ar + t * dr, az + t * dz
                    height = z - h
                    span = (r + a) ** 2 + height**2
                    if r == 0:
                        psi = br = mp.mpf(0)
                        bz = mu * a * a / (2 * span ** mp.mpf("1.5"))
                    else:
                        gap = (r - a) ** 2 + height**2
                        k = 4 * r * a / span
                        first, second = mp.ellipk(k), mp.ellipe(k)
                        psi = mu * mp.sqrt(span) * ((1 - k / 2) * first - second)
                        bz = (
                            mu
                            / (2 * mp.pi * mp.sqrt(span))
                            * (first + (a * a - r * r - height * height) / gap * second)
                        )
                        br = (
                            mu
                            * height
                            / (2 * mp.pi * r * mp.sqrt(span))
                            * (
                                -first
                                + (a * a + r * r + height * height) / gap * second
                            )
                        )
                    coefficient = (hi - lo) / 2 * mp.mpf(weight) * mp.mpf(length)
                    for i, value in enumerate((psi, br, bz)):
                        accumulated[i] += coefficient * value
            total = np.array([float(value) for value in accumulated])
            scale = np.maximum(np.abs(total), np.max(np.abs(total)) * 1e-8)
            if previous is not None and np.all(
                np.abs(total - previous) <= rtol * scale
            ):
                return total
            previous = total
            if refinement >= 3:
                middle = panels.mean(axis=1)
                panels = np.stack(
                    (
                        np.column_stack((panels[:, 0], middle)),
                        np.column_stack((middle, panels[:, 1])),
                    ),
                    axis=1,
                ).reshape(-1, 2)
    raise AssertionError(
        f"the precise graded ring reference did not converge: {start}, {end}, "
        f"{point}; value={total}; previous={previous}"
    )
