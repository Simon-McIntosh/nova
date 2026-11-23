"""Probe-local replacements of the packed moment differentiation rule."""

import inspect
from functools import wraps

import jax
import jax.numpy as jnp

from nova.biot import polygonanalytic as kernel


VARIANTS = ("current", "plain", "without_samples", "masked_samples")


def install(variant):
    """Patch only this process; retain the production primal in every arm."""
    namespaces = []
    if variant == "plain":

        def plain(*arguments):
            *primals, nodes = arguments
            return kernel._packed_analytic_moments_primal(jnp, *primals, nodes=nodes)

        kernel._packed_moments_with_stable_height = plain
        kernel._pole_product_traced = kernel._pole_product_traced.fun
    elif variant in {"without_samples", "masked_samples"}:
        source = inspect.getsource(kernel._packed_moments_height_tangent)
        source = source[source.index("def ") :]
        if variant == "without_samples":
            start = source.index("    target_z =")
            source = source[:start] + (
                "    value = evaluate(*primals)\n"
                "    _, ordinary = jax.jvp(tangent_expression, primals, tangents)\n"
                "    return value, ordinary\n"
            )
        else:
            start = source.index("    if staged:\n        shifted =")
            end = source.index("    result = tuple(", start)
            source = (
                source[:start]
                + """    def sample_cell(arguments):
        local = tuple(argument[..., None] for argument in arguments)
        nearest = jnp.minimum(
            jnp.min(jnp.abs(local[2][:, 1] - local[1])),
            jnp.min(jnp.abs(local[2][:, 3] - local[1])),
        )

        def near(_):
            shifted = jax.lax.map(
                lambda shift: tangent_expression(
                    local[0], local[1] + shift, *local[2:]
                ), shifts,
            )
            coarse = tuple((row[0] - row[1]) / (2 * shifts[0]) for row in shifted)
            middle = tuple((row[2] - row[3]) / (2 * shifts[2]) for row in shifted)
            fine = tuple((row[4] - row[5]) / (2 * shifts[4]) for row in shifted)
            return tuple(
                ((64.0 * a - 20.0 * b + c) / 45.0)[0]
                for a, b, c in zip(fine, middle, coarse, strict=True)
            )

        return jax.lax.cond(
            nearest <= 1e-7, near,
            lambda _: tuple(jnp.zeros((), local[1].dtype) for _ in range(9)),
            operand=None,
        )

    stable = jax.lax.map(
        sample_cell, tuple(jnp.moveaxis(argument, -1, 0) for argument in primals)
    )
"""
                + source[end:]
            )
        namespace = dict(kernel.__dict__)
        namespaces.append(namespace)
        exec(compile(source, inspect.getsourcefile(kernel), "exec"), namespace)
        kernel._packed_moments_with_stable_height.defjvp(
            namespace["_packed_moments_height_tangent"]
        )
    elif variant != "current":
        raise ValueError(variant)

    real = kernel._packed_analytic_moments_primal
    calls = []

    @wraps(real)
    def attributed(*args, **kwargs):
        name = f"packed_instance_{len(calls)}"
        calls.append(name)
        with jax.named_scope(name):
            return real(*args, **kwargs)

    kernel._packed_analytic_moments_primal = attributed
    for namespace in namespaces:
        namespace["_packed_analytic_moments_primal"] = attributed
    return calls
