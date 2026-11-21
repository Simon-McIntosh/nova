"""Replace one source block of the production row program with a cheap stand-in.

A block is named by the callables that implement it, and every stand-in returns
the block's output structure filled with a constant, so the compiler sees that
block's own expression graph removed and nothing else changed.  A block whose
stand-in leaves the lowered program byte-identical contributed nothing and is
reported as erased.

``identity`` wraps a real callable in itself and must leave the program
unchanged; it is the control showing that a stub erasing nothing is reported as
erased rather than credited with a share.  ``probe`` raises when the block is
reached, so a program that lowers with it in place never calls the block.

Every patch replaces a module attribute from this script before the program is
traced; no project source is edited.
"""

from functools import partial, wraps

import jax
import jax.numpy as jnp
import numpy as np

FILL = 1e-9
ROWS = 9


def _constant(value):
    """Return ``value``'s pytree filled with the stub constant."""
    return jax.tree.map(
        lambda leaf: jnp.full_like(leaf, FILL) if hasattr(leaf, "dtype") else leaf,
        value,
    )


def _replace_packed(real):
    return packed


def packed(xp, target_r, target_z, *args, **kwargs):
    """Stand-in for the nine packed moment rows, shaped from the target."""
    shape = jnp.shape(target_r)
    return tuple(
        jnp.full(shape, FILL * (index + 1), jnp.float64) for index in range(ROWS)
    )


def plain_primal(real):
    """Reach the same primal with no custom jvp rule, so JAX differentiates it."""
    from nova.biot import polygonanalytic

    @wraps(real)
    def plain(*args):
        *head, nodes = args
        return polygonanalytic._packed_analytic_moments_primal(
            jnp, *head, nodes=nodes
        )

    return plain


def flat_tangent(real):
    """Keep a block's primal and give its jvp rule a constant tangent."""

    @partial(jax.custom_jvp, nondiff_argnums=(0,))
    def stub(xp, *args):
        return real(xp, *args)

    @stub.defjvp
    def rule(xp, primals, tangents):
        primal = stub(xp, *primals)
        return primal, jax.tree.map(jnp.zeros_like, primal)

    return stub


def identity(real):
    """Return the real callable's own result through a wrapper."""

    @wraps(real)
    def stub(*args, **kwargs):
        return real(*args, **kwargs)

    return stub


def probe(real):
    """Raise when the block is reached."""

    @wraps(real)
    def stub(*args, **kwargs):
        raise RuntimeError(f"REACHED {real.__module__}.{real.__qualname__}")

    return stub


def harmonic_root(moments, parameter, *, xp=np):
    return [_constant(moments[order]) for order in range(len(moments) - 1)]


def harmonic_pole(shift, seed, moments, count, *, mirrored=False):
    return [_constant(seed) for _ in range(count)]


def harmonic_family(parameter, count, *, complement=None, xp=np):
    return [_constant(parameter) for _ in range(count)]


def pole_seed(shift, *args, **kwargs):
    return _constant(shift)


def pole_seed_paired(shift, *args, **kwargs):
    return _constant(shift)


def _fixed(function):
    """Return a stand-in factory that ignores the callable it replaces."""
    return lambda real: function


def targets(block):
    """Return the (module, attribute, stand-in) triples of one block."""
    from nova.biot import elliptic, momentchannel, polygonanalytic
    from nova.equilibrium import fixed_points

    if block in {"control", ""}:
        return []
    if block == "value":
        return [(polygonanalytic, "_packed_analytic_moments_primal", _replace_packed)]
    if block == "gradient":
        return [(fixed_points, "packed_analytic_moments", flat_tangent)]
    if block == "hessian-height":
        return [(polygonanalytic, "_packed_moments_with_stable_height", plain_primal)]
    if block == "residual-tangents":
        return [
            (polygonanalytic, "_first_residual_tangent", identity),
            (polygonanalytic, "_second_residual_tangent", identity),
            (polygonanalytic, "_vertex_against_root_tangent", identity),
        ]
    if block == "elliptic":
        return [
            (momentchannel, "harmonic_root_moments", _fixed(harmonic_root)),
            (momentchannel, "harmonic_pole_moments", _fixed(harmonic_pole)),
            (momentchannel, "harmonic_root_moments_paired", _fixed(harmonic_root)),
            (momentchannel, "harmonic_pole_moments_paired", _fixed(harmonic_pole)),
            (polygonanalytic, "harmonic_moments", _fixed(harmonic_family)),
            (polygonanalytic, "harmonic_moments_paired", _fixed(harmonic_family)),
            (polygonanalytic, "cn_pole_moment", _fixed(pole_seed)),
            (polygonanalytic, "cn_pole_moment_paired", _fixed(pole_seed_paired)),
            (polygonanalytic, "sn_pole_moment", _fixed(pole_seed)),
            (polygonanalytic, "sn_pole_moment_paired", _fixed(pole_seed_paired)),
            (elliptic, "harmonic_root_moments", _fixed(harmonic_root)),
            (elliptic, "harmonic_pole_moments", _fixed(harmonic_pole)),
        ]
    if block == "identity":
        return [(polygonanalytic, "_packed_analytic_moments_primal", identity)]
    raise ValueError(block)


def apply(block, stand_in=None):
    """Patch one block's callables and return the names replaced."""
    replaced = []
    for module, name, default in targets(block):
        real = getattr(module, name)
        setattr(module, name, (stand_in or default)(real))
        replaced.append(f"{module.__name__}.{name}")
    return replaced

ELLIPTIC_NAMES = (
    "harmonic_moments",
    "harmonic_moments_paired",
    "harmonic_root_moments",
    "harmonic_root_moments_paired",
    "harmonic_pole_moments",
    "harmonic_pole_moments_paired",
    "cn_pole_moment",
    "cn_pole_moment_paired",
    "sn_pole_moment",
    "sn_pole_moment_paired",
    "_harmonic_rising_step",
    "_harmonic_ratio_step",
    "_harmonic_rising_step_tangent",
    "_harmonic_ratio_step_tangent",
    "_harmonic_moments_tangent",
    "_pole_forward_step",
    "_pole_backward_step",
    "_pole_forward_step_tangent",
    "_pole_backward_step_tangent",
    "_harmonic_pole_moments_tangent",
    "_harmonic_root_tangent_term",
    "_harmonic_root_moments_tangent",
    "_cn_pole_moment_tangent",
    "_sn_pole_moment_tangent",
)

REACHED = []


def record(real):
    """Wrap a callable so a traced program records that it reached it."""

    @wraps(real)
    def stub(*args, **kwargs):
        REACHED.append(f"{real.__module__}.{real.__qualname__}")
        return real(*args, **kwargs)

    return stub


def record_elliptic():
    """Wrap every elliptic-family callable in every imported project module."""
    import sys

    for name in sorted(sys.modules):
        module = sys.modules[name]
        if not name.startswith("nova.") or module is None:
            continue
        for attribute in ELLIPTIC_NAMES:
            real = getattr(module, attribute, None)
            if real is None or not callable(real):
                continue
            setattr(module, attribute, record(real))

ELLIPTIC_FAMILIES = {
    "harmonic_moments": harmonic_family,
    "harmonic_moments_paired": harmonic_family,
    "harmonic_root_moments": harmonic_root,
    "harmonic_root_moments_paired": harmonic_root,
    "harmonic_pole_moments": harmonic_pole,
    "harmonic_pole_moments_paired": harmonic_pole,
    "cn_pole_moment": pole_seed,
    "cn_pole_moment_paired": pole_seed_paired,
    "sn_pole_moment": pole_seed,
    "sn_pole_moment_paired": pole_seed_paired,
}


def apply_elliptic():
    """Replace every binding of the elliptic moment families in loaded project modules."""
    import sys

    replaced = []
    for name in sorted(sys.modules):
        module = sys.modules[name]
        if not name.startswith("nova.") or module is None:
            continue
        for attribute, stand_in in ELLIPTIC_FAMILIES.items():
            if not hasattr(module, attribute):
                continue
            setattr(module, attribute, stand_in)
            replaced.append(f"{name}.{attribute}")
    return replaced
