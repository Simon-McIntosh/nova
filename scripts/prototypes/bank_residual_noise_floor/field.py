"""Reuse the stage probe at stationary and Bell-field construction boundaries."""

from contextlib import contextmanager
import inspect
import sys
from unittest.mock import patch

import jax.numpy as jnp

from nova.equilibrium import fixed_points, topology
from scripts.prototypes.bank_residual_noise_floor import attribute
from scripts.prototypes.bank_residual_noise_floor.measure import nudged

STAGES = (
    "none",
    "cached_jets",
    "scaled_jets",
    "taylor_terms",
    "bell_remainder",
    "bell_coefficients",
    "stationary_hessian",
    "third_derivatives",
    "fourth_derivatives",
)


@contextmanager
def injections(stage, seed, hits):
    def bump(value, name):
        hits.add(name)
        return jnp.where(stage == STAGES.index(name), nudged(value, seed), value)

    routes = {}
    for function, anchors in (
        (
            fixed_points._CachedField.evaluate,
            {"local_mesh = BellMesh": ("jets", "cached_jets")},
        ),
        (
            fixed_points.field_from_jets,
            {
                "anchor = scaled": ("scaled", "scaled_jets"),
                "residual = (scaled": ("taylor", "taylor_terms"),
                "coefficients = jnp.einsum": ("residual", "bell_remainder"),
                "return BellField": ("coefficients", "bell_coefficients"),
            },
        ),
    ):
        source, start = inspect.getsourcelines(function)
        lines = {
            start + i: item
            for i, line in enumerate(source)
            for anchor, item in anchors.items()
            if line.strip().startswith(anchor)
        }
        assert len(lines) == len(anchors)
        routes[function.__code__] = lines

    def trace(frame, event, arg):
        lines = routes.get(frame.f_code)
        if lines is None:
            return None
        if event == "line" and frame.f_lineno in lines:
            name, label = lines[frame.f_lineno]
            frame.f_locals[name] = bump(frame.f_locals[name], label)
        return trace

    original_solve = fixed_points.solve.__wrapped__
    original_curvature = topology._curvature_derivatives

    def solve(*args, **kwargs):
        result = original_solve(*args, **kwargs)
        return result._replace(
            jet=result.jet._replace(
                hessian=bump(result.jet.hessian, "stationary_hessian")
            )
        )

    def curvature(*args):
        third, fourth = original_curvature(*args)
        return bump(third, "third_derivatives"), bump(fourth, "fourth_derivatives")

    with (
        patch.object(fixed_points, "solve", solve),
        patch.object(topology, "_curvature_derivatives", curvature),
    ):
        sys.settrace(trace)
        try:
            yield
        finally:
            sys.settrace(None)


if __name__ == "__main__":
    attribute.STAGES = STAGES
    attribute.injections = injections
    attribute.main()
