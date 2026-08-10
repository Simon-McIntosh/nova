"""Finite-area Green's functions: complete toroidal conductors of POLYGON section.

Generalises the rectangular-section kernel in :mod:`nova.biot.greens`
(:func:`~nova.biot.greens.cylinder_greens`, Urankar Part III) to an arbitrary
polygon cross-section, from L. K. Urankar, *"Vector potential and magnetic
field of current-carrying finite arc segment in analytical form -- Part V:
polygon cross section,"* IEEE Trans. Magn. 26(3), 1171-1180 (1990).

Why: a slanted (parallelogram) or trapezoidal conductor -- the vessel end
crowns and stability-plate arms, a non-rectangular coil pack -- is neither a
filled axis-aligned box (the rectangle kernel's assumption) nor cheaply
represented by a multi-filament tiling (O(N) cost, Riemann-limited accuracy).
Urankar converts the cross-section surface integral into a CONTOUR sum over
the polygon's edges (Stokes), does the edge-parameter integral in closed form,
and leaves -- for the axisymmetric full turn -- a single smooth 1-D integral
over the arc angle phi per edge.

The vector potential.  Per edge nu with endpoints (r'v1, z'v1) -> (r'v2, z'v2)
[paper eqs (7)-(9a)]: parametrise the edge r'(u) = r1 + b1 u with u = z' - z,
slope b1 = dr/dz, intercept r1 = r'v1 - b1 (z'v1 - z).  The u-integral of eq
(8) has the closed antiderivative (eq 9, with a0^2 = 1 + b1^2, G^2 = u^2 +
r^2 sin^2 phi, B^2 = (r1 - r cos phi)^2 + a0^2 r^2 sin^2 phi, D^2 = G^2 +
(r' - r cos phi)^2, Gamma = u + b1 (r' - r cos phi), beta1 = (r' - r cos phi)/G,
beta2 = Gamma/B, beta3 = [u(r' - r cos phi) - b1 G^2]/(r sin phi D)):

    g(u, phi) = Gamma D/(2 a0^2) + u r cos phi arsinh beta1
              + [B^2 + 2 a0^2 r cos phi (r1 - r cos phi)]/(2 a0^3) arsinh beta2
              - (r^2/2) sin 2phi arctan beta3

NOTE the 1990 typesetting trap: the printed "Gamma(phi)/2a0^2 D(phi)" means
Gamma D/(2 a0^2) -- D(phi) is a NUMERATOR factor.  ``g`` reproduces the raw
cross-section integral of r'/D dr'dz' edge-by-edge to machine precision
(regression-pinned against a dense 2-D quadrature and against the rectangular
kernel).

Then A_phi(r, z) = -sum_nu integral cos phi [g] dphi (eqs 3b, 10a/b, j = phi),
and the axisymmetric flux psi = 2 pi mu0 R A_phi / (4 pi A) per ampere of total
current.

The field.  Rather than transcribe the paper's closed B integrands (eq 11b --
a longer, more error-prone form), the field is the EXACT curl of the verified
vector potential, B_Z = (1/2piR) dpsi/dR and B_R = -(1/2piR) dpsi/dZ, obtained
by differentiating the SAME antiderivative ``g`` in closed form.  Every
transcendental in ``g`` -- arsinh beta1, arsinh beta2, arctan beta3, and the
three square roots -- carries a derivative that is rational in quantities ``g``
has already computed, thanks to two identities of the Urankar variables:

    B^2 + Gamma^2 = a0^2 D^2        (so d(arsinh beta2) has denominator a0 B D)
    (r sin phi D)^2 + numer(beta3)^2 = G^2 B^2   (so d(arctan beta3) is G^2 B^2)

so the value and both derivatives cost ONE real pass over the quadrature.  The
equivalent complex-step form (df/dx = Im f(x + ih)/h, h ~ 1e-30) is the
reference this reproduces to ~1e-13 relative --
``tests/test_biotpolygongradient.py`` pins it at rtol 1e-12 -- but it needs two
passes in complex arithmetic, where every transcendental costs several times its
real counterpart.  ``_psi_hat`` is retained as that reference: it accepts
complex arguments and returns psi alone.  Either way the curl is exact to
machine precision, not a finite difference, and psi<->B consistency holds by
construction; for a rectangle it reproduces ``cylinder_greens``' B to ~1e-15.

Working set.  The quadrature is evaluated as a (targets x nodes) block, so a
whole target column at the default 16x48 rule is several megabytes per
temporary and the kernel runs out of cache.  ``block`` caps the number of
targets evaluated at once; the default keeps one block's temporaries in a few
hundred kilobytes, which is the difference between an arithmetic-bound and a
bandwidth-bound kernel.  It changes nothing numerically -- each block is an
independent set of targets.

For the FULL TURN (axisymmetric ring, arc = 2pi) the phi-integrand is even
about phi = pi, so it is evaluated on [0, pi] and doubled, with composite
Gauss-Legendre panels.  The integrand is analytic for every target off the
section boundary -- including INSIDE the conductor -- because D^2 >= r^2 sin^2
phi > 0 at the interior quadrature nodes; convergence is spectral.  With the
default 16x48 rule the field is machine-precise (~1e-13 relative) for targets
more than ~1 cm off the section boundary and holds to <=1e-6 down to ~1 mm;
sub-millimetre standoffs (finer than any physical sensor) recover full accuracy
by raising ``n_panels`` / ``n_nodes`` (both exposed).  A target lying exactly on
an edge or vertex stays finite because the complex-step increment nudges the
evaluation off the real singularity.  This mirrors the rectangle kernel, which
carries a 785-point arcsinh (zeta) quadrature inside its "closed"
antiderivative, so a smooth bounded 1-D quadrature per edge is the established
cost model.

Sign/units conventions match :func:`nova.biot.greens.cylinder_greens` (and
hence the point-filament ``greens_psi``/``greens_bz_br``), per ampere of TOTAL
conductor current, with uniform azimuthal current density J_phi = 1/A:

    psi [Wb/A] = 2 pi mu0 R A_phi / (4 pi A)
    B   [T/A]  = curl of psi.
"""

from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import leggauss

MU0 = 4.0e-7 * np.pi

# Composite Gauss-Legendre rule on phi in [0, pi] (doubled by even symmetry):
# the integrand is analytic on the open interval, so a modest panel count
# converges past 1e-12.  16 panels x 48 nodes reproduce the rectangular kernel
# to ~1e-11.
_N_PANELS = 16
_N_NODES = 48

# Evaluation points per quadrature block.  ~20 real (block x nodes) temporaries
# are live at once per edge limit; at block 16 and the default 768-node rule
# that is ~2 MB, which fits L2.  Measured on a quiet compute node the cost per
# pair is flat over blocks of 8 to 32 and rises monotonically above it -- 1.9x
# by the time the whole target column is evaluated in one call.
_BLOCK = 16


def _phi_rule(n_panels: int, n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    x, w = leggauss(n_nodes)
    edges = np.linspace(0.0, np.pi, n_panels + 1)
    lo, hi = edges[:-1, None], edges[1:, None]
    phi = (0.5 * (hi - lo) * x[None, :] + 0.5 * (hi + lo)).ravel()
    wts = (0.5 * (hi - lo) * w[None, :]).ravel()
    return phi, wts


def _orientation(v: np.ndarray) -> tuple[float, float]:
    """Return the ``(sign, area)`` pair the edge sum is normalised by.

    The counter-clockwise edge sum yields -f(phi), so one orientation sign fixes
    all three components at once (pinned by the rectangle-reduction and filament
    oracles in ``tests/test_biotpolygon.py``).
    """
    # Shoelace products made from absolute coordinates lose the section area when
    # a small conductor is translated onto a large machine coordinate.  Translation
    # does not change the cross products, so take them about one represented vertex.
    local = v - v[0]
    rolled = np.roll(local, -1, axis=0)
    signed_area2 = float(
        np.sum(local[:, 0] * rolled[:, 1] - rolled[:, 0] * local[:, 1])
    )
    return -np.sign(signed_area2), 0.5 * abs(signed_area2)


def _edges(v: np.ndarray):
    """Yield the ``(ra, za, rb, zb)`` of each edge that contributes.

    Horizontal edges (dz = 0) contribute nothing -- f_nu(phi) vanishes, paper eq
    7a -- and are skipped; the tolerance is relative to the section's own height
    so it holds for a millimetre-scale section as well as a metre-scale one.
    """
    n = len(v)
    z_scale = max(float(np.ptp(v[:, 1])), 1e-6)
    for i in range(n):
        ra, za = v[i]
        rb, zb = v[(i + 1) % n]
        if abs(zb - za) >= 1e-12 * z_scale:
            yield ra, za, rb, zb


def horizontal_edges(vertices: np.ndarray) -> np.ndarray:
    """Return the mask of edges that contribute nothing (dz = 0, paper eq 7a).

    The tolerance is relative to the section's own height so it holds for a
    millimetre-scale section as well as a metre-scale one.  This is the pack's
    TOPOLOGY: which integrand the kernel evaluates per edge, a discrete property
    of the section rather than a smooth function of its vertices -- which is why
    :func:`traced_pack_section` takes it as a static input rather than forming
    it from traced values.
    """
    v = np.asarray(vertices, dtype=np.float64)
    z_scale = max(float(np.ptp(v[:, 1])), 1e-6)
    dz = np.roll(v[:, 1], -1) - v[:, 1]
    return np.abs(dz) < 1e-12 * z_scale


def pack_section(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Return ``(edge, weight, norm)`` for one polygon section.

    ``edge`` is ``(n, 4)`` of ``(ra, za, rb, zb)`` and ``weight`` is ``(n,)``.
    Horizontal edges (dz = 0) contribute nothing -- f_nu(phi) vanishes, paper eq
    7a -- and carry zero weight.  Their real endpoints remain in ``edge`` because
    they are still part of the polygon topology: the live edges on either side
    need those corners for their residual terms.  Kernel drivers replace a dead
    row with benign target-relative geometry before forming its slope.  ``norm``
    folds the
    orientation sign, the per-ampere area normalisation, the 2 pi R of the total
    flux and the [0, pi] half-turn doubling into one factor.
    """
    v = np.asarray(vertices, dtype=np.float64)
    sign, area = _orientation(v)
    if not np.isfinite(area) or area <= 0.0:
        raise ValueError("polygon section must have positive finite area")
    rolled = np.roll(v, -1, axis=0)
    edge = np.column_stack([v[:, 0], v[:, 1], rolled[:, 0], rolled[:, 1]])
    horizontal = horizontal_edges(v)
    weight = (~horizontal).astype(np.float64)
    norm = 2.0 * np.pi * sign * MU0 / (4.0 * np.pi * area) * 2.0
    return edge, weight, float(norm)


def traced_pack_section(xp, vertices, horizontal):
    """Return ``(edge, weight, norm)`` with the pack inside the trace.

    A transcription of :func:`pack_section` into whichever array namespace
    ``xp`` is, so a geometry Jacobian reaches THROUGH the pack: the vertices
    stay trace inputs, and the edge table, the orientation sign and the
    per-ampere area normalisation all differentiate with them.  The kernel's
    orientation sign is MINUS the shoelace sign, exactly as
    :func:`_orientation` takes it (the counter-clockwise edge sum yields
    -f(phi)); through ``xp.sign`` it back-propagates a zero, which is correct
    -- a perturbation small enough to differentiate does not flip a section's
    orientation.

    ``horizontal`` is the STATIC zero-weight mask from
    :func:`horizontal_edges`, computed OUTSIDE the trace on the base topology.
    An edge's weight is a discrete property of which integrand the kernel
    evaluates, so a perturbation that would tilt a dropped edge into a live one
    is a re-pack, not a derivative -- and baking the mask keeps the trace free
    of value branching.  The kernel holds a masked edge away from its target
    before using its coordinates, so it carries neither a value nor a tangent.
    """
    vertices = xp.asarray(vertices)
    mask = np.asarray(horizontal, dtype=bool)
    local = vertices - vertices[0]
    rolled_local = xp.roll(local, -1, axis=0)
    cross = local[:, 0] * rolled_local[:, 1] - rolled_local[:, 0] * local[:, 1]
    signed_area = 0.5 * xp.sum(cross)
    sign = -xp.sign(signed_area)
    area = xp.abs(signed_area)
    rolled = xp.roll(vertices, -1, axis=0)
    edge = xp.stack(
        [vertices[:, 0], vertices[:, 1], rolled[:, 0], rolled[:, 1]], axis=1
    )
    weight = xp.asarray((~mask).astype(np.float64))
    norm = 2.0 * np.pi * sign * MU0 / (4.0 * np.pi * area) * 2.0
    return edge, weight, norm


def pad_batch(
    sections: list[np.ndarray], edge_count: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(edge, weight, norm)`` for a batch of sections, one fixed shape.

    ``edge`` is ``(E, 4, S)``, ``weight`` is ``(E, S)`` and ``norm`` is ``(S,)``
    for ``S`` sections padded to a common edge count ``E``.  Pad weights are
    negative zero while real horizontal edges carry positive zero.  Both multiply
    identically, but their sign bit preserves which rows belong to the closed
    contour without adding another public array.  Packed kernels use that topology
    to close each heterogeneous section before holding every dead row at benign
    target-relative geometry.
    """
    if not sections:
        raise ValueError("at least one polygon section is required")
    packed = [pack_section(section) for section in sections]
    count = edge_count or max(len(edge) for edge, _, _ in packed)
    edge = np.zeros((count, 4, len(packed)))
    edge[..., 1, :] = 0.0
    edge[..., 3, :] = 1.0
    # IEEE negative zero is an arithmetic zero and a one-bit topology marker.  It
    # survives NumPy/JAX conversion and contiguous tile packing.
    weight = -np.zeros((count, len(packed)))
    norm = np.empty(len(packed))
    for column, (section_edge, section_weight, section_norm) in enumerate(packed):
        rows = len(section_edge)
        if rows > count:
            raise ValueError(
                f"section {column} has {rows} edges, above the batch width {count}"
            )
        edge[:rows, :, column] = section_edge
        weight[:rows, column] = section_weight
        norm[column] = section_norm
    return edge, weight, norm


def _held_edge(xp, one_edge, live, target_r, target_z):
    """Return one edge with dead lanes held away from every target singularity.

    ``one_edge`` may be scalar geometry shared by all targets or one row per
    pair.  ``live`` has the corresponding pair axes but no quadrature axis, so it
    is grown on the right until it broadcasts against ``target_r``.  A dead row
    is vertical, unit-height, and displaced from its own target in both
    coordinates; its slope and all corner reductions are therefore finite even
    on the symmetry axis.  The zero weight still removes its value and tangent.
    """
    target_r = xp.asarray(target_r)
    target_z = xp.asarray(target_z)
    active = xp.asarray(live)
    while active.ndim < target_r.ndim:
        active = active[..., None]
    held = (target_r + 1.0, target_z + 1.0, target_r + 1.0, target_z + 2.0)
    return tuple(
        xp.where(active, coordinate, substitute)
        for coordinate, substitute in zip(one_edge, held, strict=True)
    )


def _packed_topology(xp, weight):
    """Return live, present, chain, and next-live arrays for a padded contour.

    Real section rows are contiguous from zero.  Positive zero marks a real
    horizontal edge and negative zero a pad, so the final present row closes onto
    row zero independently in every trailing batch lane.  The returned arrays
    retain the fixed padded shape and contain no value-dependent Python branch.
    """
    weight = xp.asarray(weight)
    live = weight != 0.0
    present = ~xp.signbit(weight)
    sides = weight.shape[0]
    row = xp.arange(sides).reshape((sides,) + (1,) * (weight.ndim - 1))
    last = xp.sum(present, axis=0) - 1
    last_live = xp.sum(live & (row == last), axis=0) != 0
    previous_live = xp.concatenate((last_live[None, ...], live[:-1]), axis=0)
    following = xp.concatenate((live[1:], live[:1]), axis=0)
    next_live = xp.where(row == last, live[0], following)
    chain = xp.where(
        present,
        xp.asarray(live, dtype=weight.dtype)
        - xp.asarray(previous_live, dtype=weight.dtype),
        xp.zeros_like(weight),
    )
    return live, present, chain, next_live


def _psi_gradient(
    r: np.ndarray,
    z: np.ndarray,
    edge: np.ndarray,
    weight: np.ndarray,
    cosp: np.ndarray,
    sinp: np.ndarray,
    sin2p: np.ndarray,
    w_cos: np.ndarray,
    norm: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(psi, dpsi_dr, dpsi_dz)`` per ampere, all real, in one pass.

    Differentiates the edge antiderivative ``g`` of the module docstring with
    respect to the TARGET coordinates.  ``r, z`` are real ``(N, 1)``; the phi
    node arrays are ``(Q,)``.  The edge parameter enters only through
    ``u = z' - z``, so ``du/dz = -1`` on both limits and the edge radius at each
    limit (``r' = r1 + b1 u``) is z-independent -- which is why the z-derivative
    stays as short as the r-derivative.

    ``edge`` is ``(E, 4, ...)`` broadcasting against ``(N, 1)`` and ``weight``
    is ``(E, ...)`` broadcasting against ``(N,)`` -- the weight multiplies the
    already-integrated edge contribution, so it carries no node axis.  Pass
    scalars per edge for one shared section, or an ``(E, 4, N, 1)`` stack with
    an ``(E, N)`` weight to give every evaluation point its OWN section.  The
    second form is what a tiled assembly needs -- one section per pair, one
    kernel call per tile, shape fixed by padding rather than by the data.
    """
    a_hat = np.zeros(r.shape[0])
    da_dr = np.zeros(r.shape[0])
    da_dz = np.zeros(r.shape[0])
    # r cos phi, r sin phi and d(r sin phi)^2/dr are edge-independent
    rc = r * cosp
    s = r * sinp
    s2 = s * s
    dg2_dr = 2.0 * s * sinp
    for index in range(len(edge)):
        edge_weight = weight[index]
        ra, za, rb, zb = _held_edge(np, edge[index], edge_weight != 0.0, r, z)
        b1 = (rb - ra) / (zb - za)
        a02 = 1.0 + b1 * b1
        a0 = np.sqrt(a02)
        a03 = a02 * a0
        for u, endpoint_r, s_lim in (
            (zb - z, rb, 1.0),
            (za - z, ra, -1.0),
        ):
            rmc = endpoint_r - rc
            r1mc = rmc - b1 * u
            g2 = u * u + s2
            b2 = r1mc * r1mc + a02 * s2
            d = np.sqrt(g2 + rmc * rmc)
            cap_gamma = u + b1 * rmc
            ash1 = np.arcsinh(rmc / np.sqrt(g2))
            ash2 = np.arcsinh(cap_gamma / np.sqrt(b2))
            numer3 = u * rmc - b1 * g2
            at3 = np.arctan(numer3 / (s * d))
            coef2 = (b2 + 2.0 * a02 * rc * r1mc) / (2.0 * a03)
            g = (
                cap_gamma * d / (2.0 * a02)
                + u * rc * ash1
                + coef2 * ash2
                - 0.5 * r * r * sin2p * at3
            )
            # B^2 + Gamma^2 = a0^2 D^2 and (r sin phi D)^2 + numer3^2 = G^2 B^2
            # collapse both transcendental derivatives onto quantities above.
            b2_d = b2 * a0 * d
            g2_b2 = g2 * b2
            # z-derivative: du/dz = -1, dr'/dz = 0, dr1/dz = b1
            dd_dz = -u / d
            dash1_dz = rmc * u / (g2 * d)
            dash2_dz = -(b2 + cap_gamma * r1mc * b1) / b2_d
            dat3_dz = s * ((2.0 * b1 * u - rmc) * d + numer3 * u / d) / g2_b2
            dcoef2_dz = (2.0 * r1mc * b1 + 2.0 * a02 * rc * b1) / (2.0 * a03)
            dg_dz = (
                (-d + cap_gamma * dd_dz) / (2.0 * a02)
                + rc * (u * dash1_dz - ash1)
                + dcoef2_dz * ash2
                + coef2 * dash2_dz
                - 0.5 * r * r * sin2p * dat3_dz
            )
            # r-derivative: d(r cos phi)/dr = cos phi, d(r sin phi)/dr = sin phi
            dd_dr = (s * sinp - rmc * cosp) / d
            dgamma_dr = -b1 * cosp
            dash1_dr = -(cosp * g2 + rmc * s * sinp) / (g2 * d)
            db2_dr = a02 * dg2_dr - 2.0 * r1mc * cosp
            dash2_dr = (dgamma_dr * b2 - 0.5 * cap_gamma * db2_dr) / b2_d
            dnumer3_dr = -u * cosp - b1 * dg2_dr
            dat3_dr = (dnumer3_dr * s * d - numer3 * (sinp * d + s * dd_dr)) / g2_b2
            dcoef2_dr = (db2_dr + 2.0 * a02 * cosp * (r1mc - rc)) / (2.0 * a03)
            dg_dr = (
                (dgamma_dr * d + cap_gamma * dd_dr) / (2.0 * a02)
                + u * (cosp * ash1 + rc * dash1_dr)
                + dcoef2_dr * ash2
                + coef2 * dash2_dr
                - r * sin2p * (at3 + 0.5 * r * dat3_dr)
            )
            # -[g] over u limits ua..ub  ->  -g(ub)(+1) - g(ua)(-1); fold the
            # +/-1 into s_lim.
            scale = -s_lim * edge_weight
            a_hat += scale * (g @ w_cos)
            da_dr += scale * (dg_dr @ w_cos)
            da_dz += scale * (dg_dz @ w_cos)
    radius = r[:, 0]
    return (
        norm * radius * a_hat,
        norm * (a_hat + radius * da_dr),
        norm * radius * da_dz,
    )


def _psi_hat(
    r: np.ndarray,
    z: np.ndarray,
    v: np.ndarray,
    cosp: np.ndarray,
    sinp: np.ndarray,
    sin2p: np.ndarray,
    w_cos: np.ndarray,
    sign: float,
    area: float,
) -> np.ndarray:
    """Complex-analytic psi(r, z) per ampere from the verified edge antiderivative.

    ``r, z`` are ``(T, 1)`` (possibly complex, for the complex-step curl); the
    phi node arrays are ``(Q,)``.  Returns ``(T,)`` -- real for real inputs.

    Retained as the reference formulation for :func:`_psi_gradient`, whose
    closed-form derivatives are pinned against a complex step taken through
    this function; it is not on the shipped field path.
    """
    a_hat = np.zeros(r.shape[0], dtype=np.result_type(r.dtype, z.dtype))
    for ra, za, rb, zb in _edges(v):
        b1 = (rb - ra) / (zb - za)
        a02 = 1.0 + b1 * b1
        a03 = a02 * np.sqrt(a02)
        r1 = ra - b1 * (za - z)  # (T, 1) -- depends on z
        for u, s_lim in ((zb - z, 1.0), (za - z, -1.0)):
            rp = r1 + b1 * u
            rmc = rp - r * cosp
            r1mc = r1 - r * cosp
            g2 = u * u + (r * sinp) ** 2
            b2 = r1mc * r1mc + a02 * (r * sinp) ** 2
            d = np.sqrt(g2 + rmc * rmc)
            cap_gamma = u + b1 * rmc
            ash1 = np.arcsinh(rmc / np.sqrt(g2))
            ash2 = np.arcsinh(cap_gamma / np.sqrt(b2))
            at3 = np.arctan((u * rmc - b1 * g2) / (r * sinp * d))
            g = (
                cap_gamma * d / (2.0 * a02)
                + u * r * cosp * ash1
                + (b2 + 2.0 * a02 * r * cosp * r1mc) / (2.0 * a03) * ash2
                - 0.5 * r * r * sin2p * at3
            )
            # -[g] over u limits ua..ub  ->  -g(ub)(+1) - g(ua)(-1); fold the
            # +/-1 into s_lim.
            a_hat += -s_lim * (g @ w_cos)
    a_hat *= 2.0  # [0, pi] half-turn x2
    norm = sign * MU0 / (4.0 * np.pi * area)
    return 2.0 * np.pi * r[:, 0] * norm * a_hat


def polygon_greens(
    target_r: np.ndarray,
    target_z: np.ndarray,
    vertices: np.ndarray,
    *,
    n_panels: int = _N_PANELS,
    n_nodes: int = _N_NODES,
    block: int | None = _BLOCK,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(psi, B_R, B_Z) per ampere at targets, from a polygon-section ring.

    ``vertices`` -- (n, 2) array of the section's (r, z) corners, either
    orientation, no repeated closing vertex.  Returns arrays shaped like
    ``target_r``: total poloidal flux psi [Wb/A] and field components [T/A],
    smooth everywhere including inside the conductor.  Horizontal edges
    (dz = 0) contribute nothing (paper eq 7a) and are skipped.

    ``block`` caps the targets evaluated per quadrature block so the (targets x
    nodes) temporaries stay in cache; ``None`` evaluates every target at once.
    The result does not depend on it.
    """
    tr, tz = np.broadcast_arrays(
        np.asarray(target_r, dtype=np.float64),
        np.asarray(target_z, dtype=np.float64),
    )
    shape = tr.shape
    flat_r = tr.ravel()
    flat_z = tz.ravel()
    size = flat_r.size
    psi = np.empty(size)
    br = np.empty(size)
    bz = np.empty(size)

    # The quadrature field is a flux gradient divided by r.  Partition exact axis
    # targets before either operation and take the finite field directly from the
    # closed reduction, whose parity gives Br=0 without cancellation.
    axis = flat_r == 0.0
    if np.any(axis):
        from nova.biot.polygonanalytic import polygon_analytic_greens

        axis_rows = polygon_analytic_greens(flat_r[axis], flat_z[axis], vertices)
        psi[axis], br[axis], bz[axis] = axis_rows

    off_axis = ~axis
    if np.any(off_axis):
        edge, weight, norm = pack_section(vertices)
        phi, wts = _phi_rule(n_panels, n_nodes)
        cosp = np.cos(phi)
        sinp = np.sin(phi)
        sin2p = np.sin(2.0 * phi)
        w_cos = wts * cosp
        indices = np.flatnonzero(off_axis)
        step = len(indices) if block is None else min(block, max(len(indices), 1))
        for start in range(0, len(indices), max(step, 1)):
            selected = indices[start : start + step]
            one_psi, dpsi_dr, dpsi_dz = _psi_gradient(
                flat_r[selected, None],
                flat_z[selected, None],
                edge,
                weight,
                cosp,
                sinp,
                sin2p,
                w_cos,
                norm,
            )
            psi[selected] = one_psi
            two_pi_r = 2.0 * np.pi * flat_r[selected]
            bz[selected] = dpsi_dr / two_pi_r
            br[selected] = -dpsi_dz / two_pi_r
    return psi.reshape(shape), br.reshape(shape), bz.reshape(shape)


__all__ = [
    "MU0",
    "horizontal_edges",
    "pack_section",
    "pad_batch",
    "polygon_greens",
    "traced_pack_section",
]
