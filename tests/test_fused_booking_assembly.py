"""Primal moments and local assembly channels share one cell traversal."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

from dataclasses import dataclass, replace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nova.equilibrium import current, support_read
from tests.test_support_read import _case

assert jax.config.jax_enable_x64 is True
pytestmark = [pytest.mark.slow, pytest.mark.timeout(1800)]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SmoothDensity:
    def current_density(self, radius, normalized):
        return 1.0 + 0.2 * radius + 0.3 * normalized

    def density_gradient(self, radius, normalized):
        return jnp.full_like(radius, 0.3)


@pytest.fixture(scope="module", params=["limited", "diverted"])
def physical(request):
    bank, cache, roots, pitch, step, _, _ = _case(request.param)
    support, _ = step(bank, cache, roots + 0.03 * pitch, roots)
    cells = len(bank.geometry.centre)
    rows = jnp.asarray(
        np.random.default_rng(81).normal(
            scale=1e-5, size=(len(bank.nodes), cells, 3, 6)
        )
    )
    bank = replace(bank, rows=rows)
    state = tuple(jnp.zeros(cells, dtype=jnp.float64) for _ in range(3))
    return bank, state, support


def separate(bank, state, support, profile, rule):
    psi = support_read.nodal_jets(bank, state)[bank.sample_index, 0]
    booked = current.book(psi, support.reading, bank.geometry, profile, rule)
    raw_rule = replace(rule, target_current=None)
    raw = current.Booking(
        *(jnp.zeros_like(value) for value in booked[:3]),
        booked.raw_current,
        jnp.ones_like(booked.normalization),
        booked.valid,
    )
    local = support_read._local_booking_sensitivities(
        bank, state, support, profile, profile.density_gradient, raw_rule, raw
    )
    return booked, local


@pytest.mark.parametrize("target", [None, 3.7])
@pytest.mark.parametrize("tile", [32, 128])
def test_fused_moments_and_channels_match_separate_passes(physical, target, tile):
    bank, state, support = physical
    profile = SmoothDensity()
    rule = current.Quadrature(
        segments=8, target_current=target, scalar_cut_batch_width=tile
    )
    expected_book, expected_local = jax.jit(separate)(
        bank, state, support, profile, rule
    )
    actual_book, actual_local = jax.jit(
        lambda b, m, s, p, q: support_read.book_and_local_sensitivities(
            b, m, s, p, p.density_gradient, q
        )
    )(bank, state, support, profile, rule)
    assert bool(actual_book.valid) and bool(expected_book.valid)
    for name in current.Booking._fields[:-1]:
        expected = np.asarray(getattr(expected_book, name))
        actual = np.asarray(getattr(actual_book, name))
        scale = np.max(np.abs(expected))
        np.testing.assert_allclose(
            actual, expected, rtol=1e-14, atol=scale * 1e-14, err_msg=name
        )
    assert np.linalg.norm(np.asarray(expected_book.radial_moment)) > 1e-8
    assert np.linalg.norm(np.asarray(expected_book.vertical_moment)) > 1e-8
    np.testing.assert_allclose(actual_local, expected_local, rtol=1e-14, atol=1e-14)


def test_assembly_consumes_the_fused_channels(physical):
    bank, state, support = physical
    profile = SmoothDensity()
    rule = current.Quadrature(segments=8, target_current=3.7)
    booked, local = jax.jit(
        lambda b, m, s, p, q: support_read.book_and_local_sensitivities(
            b, m, s, p, p.density_gradient, q
        )
    )(bank, state, support, profile, rule)

    def matrix(b, m, s, p, q, booked, local):
        return support_read.step_jacobian(
            b,
            m,
            s,
            p,
            p.density_gradient,
            q,
            booked,
            local_sensitivities=local,
        )

    expected = jax.jit(
        lambda b, m, s, p, q, booked: matrix(b, m, s, p, q, booked, None)
    )(bank, state, support, profile, rule, booked)
    actual = jax.jit(matrix)(bank, state, support, profile, rule, booked, local)
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)
    assert np.linalg.norm(np.asarray(expected) - np.eye(expected.shape[0])) > 1e-7


def test_fusion_removes_a_conic_construction_pass(physical):
    bank, state, support = physical
    profile = SmoothDensity()
    rule = current.Quadrature(segments=8)
    original = current._conic_rings
    counts = []

    def observe(*args, **kwargs):
        counts.append(1)
        return original(*args, **kwargs)

    with patch.object(current, "_conic_rings", observe):
        jax.make_jaxpr(separate)(bank, state, support, profile, rule)
    assert len(counts) == 2, counts
    print("DUPLICATE_CONIC_CONSTRUCTIONS separate=2", flush=True)
    counts.clear()
    with patch.object(current, "_conic_rings", observe):
        jax.make_jaxpr(
            lambda b, m, s, p, q: support_read.book_and_local_sensitivities(
                b, m, s, p, p.density_gradient, q
            )
        )(bank, state, support, profile, rule)
    assert len(counts) == 1, counts
    print("FUSED_CONIC_CONSTRUCTIONS fused=1", flush=True)


def test_identical_programs_refused():
    from scripts.prototypes.h200_levers.run import refuse_inert

    original = dict(rows=[dict(lowered_sha256="observed-separate-program")])
    changed = dict(rows=[dict(lowered_sha256="observed-fused-program")])
    refuse_inert(original, changed)
    with pytest.raises(ValueError, match="inert lever"):
        refuse_inert(original, original)


def test_newton_step_matches_separate_booking(physical):
    bank, state, support = physical
    profile = SmoothDensity()
    rule = current.Quadrature(segments=8, target_current=3.7)

    def correction(b, m, s, p, q):
        return support_read.newton_step(b, m, s, p, p.density_gradient, q)

    def separate_booking(b, m, s, p, gradient, q):
        return separate(b, m, s, p, q)

    with patch.object(support_read, "book_and_local_sensitivities", separate_booking):
        expected = jax.jit(correction)(bank, state, support, profile, rule)
    actual = jax.jit(lambda *args: correction(*args))(
        bank, state, support, profile, rule
    )
    for a, e in zip(actual.update, expected.update, strict=True):
        np.testing.assert_allclose(a, e, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(
        actual.factor.matrix, expected.factor.matrix, rtol=1e-14, atol=1e-14
    )
    assert bool(actual.converged) and bool(expected.converged)
