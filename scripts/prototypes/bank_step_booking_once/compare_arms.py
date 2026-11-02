"""Compare one booking-tangent source arm against the archived base arm."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp

from nova.equilibrium import current, support_read
from scripts.prototypes.bank_newton_step.measure import machine


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main(base_path, out_prefix):
    base = load(base_path, "current_base_arm")
    bank, cache, roots, convention, policy, profile, quadrature, moments, _ = machine(
        "diverted"
    )
    seeds = roots + 0.03 * jnp.median(bank.geometry.pitch)
    support = jax.jit(
        lambda b, c, m: support_read.read(b, c, m, seeds, roots, convention, policy)
    )(bank, cache, moments)
    residual = jax.jit(
        lambda b, m, s: support_read.step_residual(b, m, m, s, profile, quadrature)
    )(bank, moments, support)
    booked = residual.booked

    def gradient(radius, flux):
        return jnp.zeros_like(radius)

    def arm(module, name):
        original = current.book_tangent_channels
        current.book_tangent_channels = module.book_tangent_channels
        try:
            function = lambda b, m, s, p: support_read._local_booking_sensitivities(  # noqa: E731
                b, m, s, profile, gradient, quadrature, p
            )
            lowered = jax.jit(function).lower(bank, moments, support, booked)
            text = lowered.compile().as_text()
            value = jax.jit(function)(bank, moments, support, booked)
        finally:
            current.book_tangent_channels = original
        Path(f"{out_prefix}-{name}.hlo").write_text(text)
        return text, value

    base_text, base_value = arm(base, "base")
    head_text, head_value = arm(current, "head")
    difference = float(jnp.max(jnp.abs(head_value - base_value)))
    scale = float(jnp.max(jnp.abs(base_value)))
    print(
        json.dumps(
            {
                "base_source": str(base_path),
                "head_source": current.__file__,
                "max_abs_difference": difference,
                "max_abs_base": scale,
                "relative_difference": difference / scale if scale else float("nan"),
                "head_bytes": len(head_text.encode()),
                "base_bytes": len(base_text.encode()),
                "head_value_sha256": hashlib.sha256(head_value.tobytes()).hexdigest(),
                "base_value_sha256": hashlib.sha256(base_value.tobytes()).hexdigest(),
                "shape": list(head_value.shape),
                "byte_identical": head_value.tobytes() == base_value.tobytes(),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])