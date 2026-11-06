"""Batched booking preserves each matrix and bounds the device scaling."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import json
import importlib.util
from dataclasses import replace
import os
from pathlib import Path
import sys
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nova.equilibrium import current, support_read, topology
from scripts.prototypes.bank_step_booking_batch.measure import (
    cycle,
    inputs,
    jacobians,
)


@pytest.fixture(scope="module")
def machine():
    assert jax.config.jax_enable_x64
    print(
        "MODULE", cycle.support_read.__file__, "CWD", Path.cwd().resolve(), flush=True
    )
    return cycle.reference.machine("diverted")


@pytest.mark.slow
@pytest.mark.parametrize("lanes", [1, 16])
def test_each_matrix_matches_archived_base(machine, lanes):
    path = os.environ.get("BOOKING_BASE_MATRICES")
    if not path:
        pytest.skip("paired CPU gate supplies the archived matrix receipt")
    bank, _, _, _, _, profile, quadrature, *_ = machine
    states, support, booked = inputs(lanes, machine)
    result = jax.jit(lambda b, m, s, p: jacobians(b, m, s, p, profile, quadrature))(
        bank, states, support, booked
    )
    actual = np.asarray(result)
    with np.load(path) as receipt:
        expected = receipt[f"matrix_{lanes}"]
    assert actual.shape == expected.shape
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    scale = np.linalg.norm(expected.reshape(lanes, -1), axis=1)
    assert np.all(scale > 0)
    error = np.linalg.norm((actual - expected).reshape(lanes, -1), axis=1) / scale
    print("MATRIX_RELATIVE_ERRORS", lanes, error.tolist(), flush=True)
    assert np.all(error <= 1e-12), error


@pytest.mark.slow
def test_geometry_stencil_overflow_is_refused(machine, monkeypatch):
    bank, _, _, _, _, profile, quadrature, *_ = machine
    states, support, booked = inputs(1, machine)
    # The seed uses one triangle; two slots cannot represent its three nodes.
    monkeypatch.setattr(cycle.support_read, "_FRESH_GEOMETRY_NODE_CAPACITY", 2)
    result = jax.jit(lambda b, m, s, p: jacobians(b, m, s, p, profile, quadrature))(
        bank, states, support, booked
    )
    assert np.isnan(np.asarray(result)).all()


def test_sixteen_lane_scaling():
    path = os.environ.get("BOOKING_TIMINGS")
    if not path:
        pytest.skip("paired H200 gate supplies synchronized timing measurements")
    timings = json.loads(Path(path).read_text())
    one = timings["1:assembly"]
    sixteen = timings["16:assembly"]
    assert np.isfinite(one) and one > 0
    assert np.isfinite(sixteen) and sixteen > 0
    print("SCALING", sixteen / one, "ONE_MS", one, "SIXTEEN_MS", sixteen, flush=True)
    assert sixteen <= 4 * one, (one, sixteen, sixteen / one)


def assert_jacobian_equal(actual, expected, label):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    scale = np.linalg.norm(expected)
    assert scale > 0
    relative = np.linalg.norm(actual - expected) / scale
    print("JACOBIAN_PARITY", label, "relative", relative, flush=True)
    assert relative <= 1e-12, (label, relative)


def test_conic_channel_jacobian_matches_scalar_derivatives():
    from tests.test_current_book_vectorised import _cells

    psi, reading, geometry, profile = _cells(1, slope=0.2)
    quadrature = current.Quadrature(segments=8)
    directions = jnp.eye(6, dtype=jnp.float64)[jnp.asarray([0, 5])][:, None]
    flux_rates = jnp.zeros((2, *psi.shape), dtype=jnp.float64)
    levels = jnp.zeros(2, dtype=jnp.float64)

    @jax.jit
    def compare():
        booked = current.book(psi, reading, geometry, profile, quadrature)

        def gradient(radius, flux):
            return jnp.full_like(radius, 0.2)

        actual = current.book_tangent_channels(
            psi,
            flux_rates,
            reading,
            directions,
            levels,
            levels,
            geometry,
            profile,
            gradient,
            quadrature,
            booked,
            jnp.arange(2),
        )

        def scalar(flux_rate, direction):
            return jnp.stack(
                current.book_tangent(
                    psi,
                    flux_rate,
                    reading,
                    direction,
                    levels[0],
                    levels[0],
                    geometry,
                    profile,
                    gradient,
                    quadrature,
                    booked,
                ),
                axis=-1,
            )

        expected = jax.vmap(scalar)(flux_rates, directions)
        return actual, jnp.moveaxis(expected, 0, -1)

    assert_jacobian_equal(*compare(), "conic-small")


@pytest.fixture(scope="module")
def archived_channels():
    path = os.environ.get("BOOKING_REFERENCE_CURRENT")
    if not path:
        pytest.skip("paired H200 gate supplies the base current module")
    spec = importlib.util.spec_from_file_location("booking_reference_current", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    print("REFERENCE_MODULE", module.__file__, flush=True)

    def reference(*arguments):
        arguments = list(arguments)
        arguments[9] = replace(
            arguments[9], cut_batch_width=module.Quadrature().cut_batch_width
        )
        return module.book_tangent_channels(*arguments)

    return reference


@pytest.fixture(scope="module", params=["limited", "diverted"])
def certificate_machine(request):
    if jax.default_backend() != "gpu":
        pytest.skip("550-cell Jacobian comparison runs on the H200 lane")
    return request.param, cycle.reference.machine(request.param)


@pytest.mark.slow
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("state", ["trial", "converged"])
def test_certificate_jacobian_matches_base(
    certificate_machine, archived_channels, state
):
    from scripts.prototypes.read_route.measure import sample_field

    kind, machine = certificate_machine
    bank, cache, roots, convention, policy, profile, quadrature, moments, analytic = (
        machine
    )
    assert len(bank.geometry.centre) == 550
    if state == "trial":
        reading = jax.jit(topology.read)(
            sample_field(analytic, bank.geometry), bank.geometry, convention, policy
        )
        psi = jax.vmap(jax.vmap(analytic.value))(bank.geometry.sample_points)
        moments = current.book(psi, reading, bank.geometry, profile, quadrature)[:3]
    support = jax.jit(
        lambda b, m: support_read.read(
            b,
            cache,
            m,
            roots,
            roots,
            convention,
            policy,
        )
    )(bank, moments)

    def matrix(b, m, s):
        return support_read.step_jacobian(
            b,
            m,
            s,
            profile,
            lambda r, p: jnp.zeros_like(r),
            quadrature,
            fresh_geometry=True,
        )

    actual = jax.jit(matrix)(bank, moments, support)
    with patch.object(current, "book_tangent_channels", archived_channels):
        expected = jax.jit(lambda b, m, s: matrix(b, m, s))(bank, moments, support)
    assert_jacobian_equal(actual, expected, f"{kind}-550-{state}")
