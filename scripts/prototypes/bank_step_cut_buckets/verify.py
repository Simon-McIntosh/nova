"""Compare booked fields and channels with the unchanged branch source."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import importlib.util
import json
import sys

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium import current, support_read
from scripts.prototypes.bank_newton_step.measure import machine


def load(path):
    spec = importlib.util.spec_from_file_location("booking_reference", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main(path):
    base = load(path)
    bank, cache, roots, convention, policy, profile, quadrature, moments, _ = machine(
        "diverted"
    )
    seeds = roots + 0.03 * jnp.median(bank.geometry.pitch)
    support = jax.jit(
        lambda b, c, m: support_read.read(b, c, m, seeds, roots, convention, policy)
    )(bank, cache, moments)
    psi = support_read.nodal_jets(bank, moments)[bank.sample_index, 0]
    head_booked = jax.jit(
        lambda flux: current.book(
            flux, support.reading, bank.geometry, profile, quadrature
        )
    )(psi)
    base_booked = jax.jit(
        lambda flux: base.book(
            flux, support.reading, bank.geometry, profile, quadrature
        )
    )(psi)

    def gradient(radius, flux):
        return jnp.zeros_like(radius)

    def tangent(module):
        original = current.book_tangent_channels
        current.book_tangent_channels = module.book_tangent_channels
        try:
            return jax.jit(
                lambda b, m, s, p: support_read._local_booking_sensitivities(
                    b, m, s, profile, gradient, quadrature, p
                )
            )(bank, moments, support, head_booked)
        finally:
            current.book_tangent_channels = original

    values = {
        name: (
            np.asarray(getattr(base_booked, name)),
            np.asarray(getattr(head_booked, name)),
        )
        for name in base_booked._fields
    }
    values["channels"] = (np.asarray(tangent(base)), np.asarray(tangent(current)))
    relative = {
        name: float(
            np.max(np.abs(a.astype(float) - b.astype(float)))
            / max(float(np.max(np.abs(a.astype(float)))), 1.0)
        )
        for name, (a, b) in values.items()
    }
    print(json.dumps({"relative": relative, "maximum": max(relative.values())}))
    assert max(relative.values()) <= 1e-12


if __name__ == "__main__":
    main(sys.argv[1])
