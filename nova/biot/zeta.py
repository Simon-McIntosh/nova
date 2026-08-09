"""Evaluate the zeta integral for finite-volume Biot methods.

``zeta`` is the one non-closed-form piece of the rectangular/bow conductor
antiderivative,

    zeta(alpha) = integral_0^alpha arcsinh(beta_1(a)) da,
    beta_1(a) = (rs - r cos phi) / sqrt(gamma^2 + r^2 sin^2 phi),
    phi = pi - 2 a,

with the arc half-angle folded into ``[0, pi/2]`` by the caller
(:class:`nova.biot.arc.Arc` maps every segment there) and the integrand even in
``alpha``, so only ``|alpha|`` matters.

Analytic structure -- what fixes the quadrature choice.  ``sin phi`` vanishes at
both ends of that interval, so where the target sits in the plane of the source
corner (``gamma -> 0``) the integrand carries a logarithmic endpoint
singularity, ``arcsinh(beta_1) ~ -ln|a|`` as ``a -> 0``.  Away from that plane
the integrand is analytic and fixed-order Gauss-Legendre converges
spectrally; on it, Gauss-Legendre degrades to algebraic convergence and no
practical order reaches full precision.  Two fixed-node rules therefore cover
the domain between them:

* **Gauss-Legendre** at :data:`GAUSS_ORDER` nodes -- the default, used wherever
  ``|gamma| >= NEAR_PLANE_RATIO * r``;
* **tanh-sinh** (double-exponential) at :data:`TANH_SINH_HALF_COUNT` steps per
  side -- the near-plane rule, whose endpoint clustering absorbs the
  logarithmic singularity at both ends.

Both rules have a node count independent of the data, so a whole target x
source array evaluates as a plain broadcast over fixed shapes -- no grouping by
per-element panel count, and static shapes for the JAX form below under
``jit``/``vmap``.  :func:`zeta_midpoint` retains the former uniform-midpoint
rule as an independent reference for the equivalence tests; it is not used in
production because its accuracy is limited to ~1e-3 in the plane of the source
and ~1e-7 far from it.

The canonical implementation is pure numpy (imports with numpy only, so the
kernel layer carries no numba/JAX requirement).  A JAX pytree form,
:class:`Zeta`, is provided for the jitted batched-evaluation path and is only
usable when JAX is installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property, lru_cache

import numpy as np

from nova.jax.config import Precision, resolve_precision

GAUSS_ORDER = 56
"""Gauss-Legendre node count of the default rule (<=1e-12 for gamma >= 0.2 r)."""

TANH_SINH_HALF_COUNT = 88
"""Steps per side of the tanh-sinh rule; total nodes are twice this plus one."""

TANH_SINH_HALF_WIDTH = 3.4
"""Half-width in the tanh-sinh parameter t; sets where the node set truncates.

Both this and the step size ``TANH_SINH_HALF_WIDTH / TANH_SINH_HALF_COUNT`` are
tuned against the equivalence scan: too narrow a half-width truncates the tails
and floors the accuracy no matter how fine the step, which is the trap this
pair was picked to avoid.
"""

NEAR_PLANE_RATIO = 0.2
"""``|gamma| / r`` below which the tanh-sinh rule replaces Gauss-Legendre.

Gauss-Legendre at :data:`GAUSS_ORDER` first misses 1e-12 relative accuracy at
``|gamma| / r = 0.084`` (measured over the full reduced-variable scan), so this
threshold engages the near-plane rule with better than a factor two in reserve.
"""

_MAX_BLOCK = 1 << 16
"""Elements x nodes per broadcast block, in doubles.

Sized so a block's temporaries stay in cache rather than to cap peak memory:
the integrand allocates several arrays of this size per block, and letting them
grow to a few megabytes costs a factor of four in throughput (and, past a few
megabytes, an order of magnitude to allocator churn).
"""


Rule = tuple[np.ndarray, np.ndarray, np.ndarray]
"""(offset fraction, from-upper-end flag, weight) node arrays of a fixed rule.

Nodes are carried as a fraction of the interval measured from the *nearer* end
rather than as an absolute position, because both ends are where the integrand
turns singular: resolving ``sin phi`` there needs the distance to the end at
full relative precision, which an absolute node position loses.  Weights sum to
one, so an integral over ``[0, alpha]`` is ``alpha * sum(w * f(...))``.
"""


@lru_cache(maxsize=None)
def _gauss_legendre_rule() -> Rule:
    """Return the Gauss-Legendre rule of order :data:`GAUSS_ORDER`."""
    nodes, weights = np.polynomial.legendre.leggauss(GAUSS_ORDER)
    return 0.5 * (1.0 - np.abs(nodes)), nodes > 0.0, 0.5 * weights


@lru_cache(maxsize=None)
def _tanh_sinh_rule() -> Rule:
    """Return the tanh-sinh (double-exponential) rule.

    The substitution ``s = (1 + tanh(pi/2 sinh t)) / 2`` maps the whole real
    line onto ``(0, 1)`` and drives the transformed integrand to zero
    double-exponentially at both ends, which is what turns the logarithmic
    endpoint singularity into a rapidly convergent trapezoidal sum in ``t``.
    """
    step = TANH_SINH_HALF_WIDTH / TANH_SINH_HALF_COUNT
    t = np.arange(-TANH_SINH_HALF_COUNT, TANH_SINH_HALF_COUNT + 1) * step
    u = 0.5 * np.pi * np.sinh(t)
    # logistic form of the distance to the nearer end: never cancels, so the
    # outermost abscissae keep full relative precision
    offset = 1.0 / (1.0 + np.exp(2.0 * np.abs(u)))
    weights = step * 0.5 * np.pi * np.cosh(t) / np.cosh(u) ** 2 / 2.0
    return offset, u > 0.0, weights


def _integrand(
    rs: np.ndarray,
    r: np.ndarray,
    gamma: np.ndarray,
    alpha: np.ndarray,
    rule: Rule,
) -> np.ndarray:
    """Return arcsinh(beta_1) at every node of ``rule``, appended as a last axis.

    ``phi = pi - 2 alpha s`` is never formed directly.  For a node in the lower
    half the angle from ``phi = pi`` is ``2 alpha s``, and for one in the upper
    half the angle from ``phi = 0`` is ``(pi - 2 alpha) + 2 alpha (1 - s)`` --
    both sums of non-negative terms, so ``sin phi`` stays accurate right down to
    the outermost tanh-sinh abscissa.
    """
    offset, upper, _ = rule
    angle = 2.0 * alpha[..., np.newaxis] * offset
    angle = np.where(upper, np.pi - 2.0 * alpha[..., np.newaxis] + angle, angle)
    sin_phi = np.sin(angle)
    cos_phi = np.where(upper, np.cos(angle), -np.cos(angle))
    rs, r, gamma = (array[..., np.newaxis] for array in (rs, r, gamma))
    return np.arcsinh((rs - r * cos_phi) / np.sqrt(gamma**2 + r**2 * sin_phi**2))


def _integrate(
    rs: np.ndarray,
    r: np.ndarray,
    gamma: np.ndarray,
    alpha: np.ndarray,
    rule: Rule,
) -> np.ndarray:
    """Apply a fixed-node rule to ``[0, alpha]``, blocked for cache locality."""
    weights = rule[2]
    result = np.empty(alpha.shape)
    stride = max(1, _MAX_BLOCK // weights.size)
    for start in range(0, alpha.size, stride):
        block = slice(start, start + stride)
        integrand = _integrand(rs[block], r[block], gamma[block], alpha[block], rule)
        result[block] = alpha[block] * (integrand @ weights)
    return result


def zeta(
    rs: np.ndarray,
    r: np.ndarray,
    gamma: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    """Evaluate the zeta integral element-wise to better than 1e-12 relative.

    Elements away from the plane of the source corner take the Gauss-Legendre
    rule; those within :data:`NEAR_PLANE_RATIO` of it -- where the integrand is
    logarithmically singular at the interval ends -- take the tanh-sinh rule.
    Both node sets are fixed, so each group evaluates as a plain broadcast.
    Inputs broadcast to a common shape, which the result takes.
    """
    broadcast = np.broadcast_shapes(
        np.shape(rs), np.shape(r), np.shape(gamma), np.shape(alpha)
    )
    rs, r, gamma, alpha = (
        np.ravel(np.broadcast_to(array, broadcast)) for array in (rs, r, gamma, alpha)
    )
    # the integrand is even in alpha, so the interval is taken as [0, |alpha|]
    alpha = np.abs(alpha)
    result = np.zeros(alpha.size)
    # a zero-length interval integrates to zero by inspection; evaluating it
    # would divide by a vanishing sin(phi) in the plane of the source corner
    extent = alpha > 0.0
    near_plane = np.abs(gamma) < NEAR_PLANE_RATIO * np.abs(r)
    for index, rule in (
        (np.flatnonzero(extent & ~near_plane), _gauss_legendre_rule()),
        (np.flatnonzero(extent & near_plane), _tanh_sinh_rule()),
    ):
        if index.size:
            result[index] = _integrate(
                rs[index], r[index], gamma[index], alpha[index], rule
            )
    return result.reshape(broadcast)


def traced_zeta(xp, rs, r, gamma, alpha):
    """Return the zeta integral in whichever array namespace ``xp`` is.

    The tanh-sinh rule UNCONDITIONALLY, as :class:`Zeta` takes it: the numpy
    :func:`zeta` picks between two rules per element, which is a host-side cost
    optimisation that a trace would have to pay for by evaluating both -- and
    the double-exponential rule alone is uniformly accurate over the whole
    domain, so the branch-free path takes it outright.  Everything else is
    :func:`zeta`'s own arithmetic, so with ``xp = numpy`` the two agree to the
    two rules' mutual accuracy wherever the host picks Gauss-Legendre and
    exactly where it picks tanh-sinh.

    A zero-length interval is held at one and the result masked, so it returns
    the zero it integrates to without dividing by a vanishing ``sin phi`` --
    and without the held element poisoning a geometry tangent.
    """
    dtype = xp.result_type(rs, r, gamma, alpha)
    offset, upper, weights = _tanh_sinh_rule()
    offset = xp.asarray(offset, dtype=dtype)
    upper = xp.asarray(upper)
    weights = xp.asarray(weights, dtype=dtype)
    rs, r, gamma, alpha = xp.broadcast_arrays(
        xp.asarray(rs), xp.asarray(r), xp.asarray(gamma), xp.asarray(alpha)
    )
    # the integrand is even in alpha, so the interval is taken as [0, |alpha|]
    alpha = xp.abs(alpha)
    extent = alpha > 0.0
    held = xp.where(extent, alpha, 1.0)[..., None]
    angle = 2.0 * held * offset
    angle = xp.where(upper, np.pi - 2.0 * held + angle, angle)
    sin_phi = xp.sin(angle)
    cos_phi = xp.where(upper, xp.cos(angle), -xp.cos(angle))
    integrand = xp.arcsinh(
        (rs[..., None] - r[..., None] * cos_phi)
        / xp.sqrt(gamma[..., None] ** 2 + r[..., None] ** 2 * sin_phi**2)
    )
    return xp.where(extent, alpha, 0.0) * (integrand @ weights)


def zeta_midpoint(
    rs: np.ndarray,
    r: np.ndarray,
    gamma: np.ndarray,
    alpha: np.ndarray,
    number: int = 500,
) -> np.ndarray:
    """Evaluate the zeta integral element-wise via uniform midpoint quadrature.

    Each element integrates over ``[0, alpha_i]`` with a per-element panel count
    scaled to its own limit (``~ |alpha_i| * number``), elements sharing a panel
    count integrating as one vectorised block.  Retained as an independent
    reference for the quadrature-equivalence tests: a uniform rule cannot
    resolve the logarithmic endpoint singularity, so its accuracy runs from
    ~1e-3 relative in the plane of the source corner to ~1e-7 far from it.
    """
    shape = np.shape(alpha)
    broadcast = np.broadcast(rs, r, gamma, alpha)
    rs, r, gamma, alpha = (
        np.ravel(np.broadcast_to(array, broadcast.shape))
        for array in (rs, r, gamma, alpha)
    )
    result = np.zeros(alpha.size)
    active = ~np.isclose(alpha, 0.0)
    num = np.maximum(3, (np.abs(alpha) * number).astype(int))
    for count in np.unique(num[active]):
        index = np.flatnonzero(active & (num == count))
        dalpha = alpha[index] / (count - 1)
        nodes = (
            np.ascontiguousarray(np.linspace(0.0, alpha[index], count, axis=-1)[:, :-1])
            + dalpha[:, np.newaxis] / 2.0
        )  # midpoints
        phi = np.pi - 2.0 * nodes
        rs_i, r_i, gamma_i = (array[index, np.newaxis] for array in (rs, r, gamma))
        g2 = gamma_i**2 + r_i**2 * np.sin(phi) ** 2
        integrand = np.arcsinh((rs_i - r_i * np.cos(phi)) / np.sqrt(g2))
        result[index] = np.abs(dalpha) * integrand.sum(axis=1)
    return result.reshape(shape)


try:  # optional: only importable where JAX is installed
    import jax
    import jax.numpy as jnp

    @jax.tree_util.register_pytree_node_class
    @dataclass
    class Zeta:
        """Evaluate the zeta integral through JAX (jittable, vmap-safe).

        Uses the tanh-sinh rule unconditionally.  The numpy :func:`zeta` picks
        between two rules per element, which under ``vmap`` would mean
        evaluating both and discarding one; the tanh-sinh rule alone is both
        cheaper than that and uniformly accurate over the whole domain, so the
        batched path takes it branch-free.
        """

        rs: np.ndarray | jnp.ndarray = field(repr=False)
        zs: np.ndarray | jnp.ndarray = field(repr=False)
        r: np.ndarray | jnp.ndarray = field(repr=False)
        z: np.ndarray | jnp.ndarray = field(repr=False)
        alpha: np.ndarray | jnp.ndarray = field(repr=False)
        precision: Precision | str = field(default=Precision.AUTOMATIC, repr=False)

        def __post_init__(self):
            """Construct device leaves in the precision chosen for this evaluator."""
            resolved = resolve_precision(self.precision, Precision.DOUBLE)
            dtype = jnp.float32 if resolved is Precision.SINGLE else jnp.float64
            for name in ("rs", "zs", "r", "z", "alpha"):
                setattr(self, name, jnp.asarray(getattr(self, name), dtype=dtype))
            self.precision = resolved

        def tree_flatten(self):
            """Return flattened pytree structure."""
            return (
                (self.rs, self.zs, self.r, self.z, self.alpha),
                self.precision,
            )

        @classmethod
        def tree_unflatten(cls, aux_data, children):
            """Rebuild instance from pytree variables."""
            return cls(*children, precision=aux_data)

        @cached_property
        @jax.jit
        def gamma(self):
            """Return gamma coefficient."""
            return self.zs - self.z

        @jax.jit
        def arcsinh_beta_1(self, sin_phi, cos_phi):
            """Return the zeta integrand from the sine and cosine of phi."""
            g2 = self.gamma**2 + self.r**2 * sin_phi**2
            return jnp.arcsinh((self.rs - self.r * cos_phi) / jnp.sqrt(g2))

        @jax.jit
        def __call__(self):
            """Return the zeta integral."""
            offset, upper, weights = _tanh_sinh_rule()
            dtype = jnp.result_type(self.rs, self.zs, self.r, self.z, self.alpha)
            # the node axis leads, so rs/r/gamma broadcast against it unchanged
            shape = (-1,) + (1,) * jnp.ndim(self.alpha)
            offset = jnp.asarray(offset, dtype=dtype).reshape(shape)
            upper = jnp.asarray(upper).reshape(shape)
            alpha = jnp.abs(self.alpha)
            angle = 2.0 * alpha * offset
            angle = jnp.where(upper, jnp.pi - 2.0 * alpha + angle, angle)
            integrand = self.arcsinh_beta_1(
                jnp.sin(angle), jnp.where(upper, jnp.cos(angle), -jnp.cos(angle))
            )
            return alpha * jnp.tensordot(
                jnp.asarray(weights, dtype=dtype), integrand, axes=1
            )

except ModuleNotFoundError:  # numpy-only environment

    class Zeta:  # type: ignore[no-redef]
        """Placeholder when JAX is unavailable; use the numpy ``zeta`` instead."""

        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError(
                "nova.biot.zeta.Zeta requires JAX; install the 'jax' extra or "
                "use the numpy zeta() function"
            )
