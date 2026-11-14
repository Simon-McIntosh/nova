"""Continuous carried geometry reuses only certified unchanged cells."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nova.equilibrium import topology
from tests.test_tracked_read import _case, _read

assert jax.config.jax_enable_x64 is True
pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def carried_case():
    parts = _case("diverted")
    support, _ = _read(*parts[:6])
    assert bool(support.valid)
    return parts, support


@pytest.fixture(scope="module")
def fragments(carried_case):
    parts, support = carried_case
    return parts[0].geometry, parts[3].sigma, parts[4], support.reading


def _compare(got, want):
    for actual, expected in zip(got[:4], want[:4], strict=True):
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)


def _program(geometry, sigma, policy, decision):
    def refresh(coefficient, level, form):
        return topology._refresh_fragments(
            geometry, coefficient, sigma, level, form, decision, policy
        )

    def full(coefficient, level, form):
        return topology._fragments_at_level(
            geometry,
            coefficient,
            sigma,
            level,
            form,
            decision.normal_form_cells,
            policy,
        )[:4]

    return jax.jit(refresh), jax.jit(full)


def test_unchanged_read_recomputes_no_cells(fragments):
    geometry, sigma, policy, decision = fragments
    refresh, full = _program(*fragments)
    args = decision.field_coefficients, decision.boundary_flux, decision.saddle_form
    result = refresh(*args)
    assert int(result[-1]) == 0
    _compare(result, full(*args))


def test_changed_cut_cell_is_recomputed_and_matches_full_refresh(fragments):
    geometry, sigma, policy, decision = fragments
    area = np.asarray(decision.fragment_area).sum(axis=1)
    cut = (area > 1e-10) & (area < np.asarray(geometry.full_area) * (1 - 1e-6))
    cut &= ~np.asarray(decision.normal_form_cells)
    assert cut.any(), "positive control must contain a changing conic cell"
    index = int(np.flatnonzero(cut)[0])
    coefficient = decision.field_coefficients.at[index, 0].add(
        0.01 * abs(decision.boundary_flux - decision.axis_flux)
    )
    refresh, full = _program(*fragments)
    args = coefficient, decision.boundary_flux, decision.saddle_form
    result, reference = refresh(*args), full(*args)
    assert np.max(np.abs(np.asarray(reference[0] - decision.fragment_area))) > 1e-8
    assert int(result[-1]) == 1
    _compare(result, reference)


def test_boundary_motion_refreshes_uncertain_cells_and_reuses_bulk(fragments):
    geometry, sigma, policy, decision = fragments
    refresh, full = _program(*fragments)
    level = decision.boundary_flux + 1e-4 * abs(
        decision.boundary_flux - decision.axis_flux
    )
    args = decision.field_coefficients, level, decision.saddle_form
    result = refresh(*args)
    assert 0 < int(result[-1]) < len(geometry.centre)
    print("REFRESH_CELLS", int(result[-1]), "TOTAL", len(geometry.centre))
    _compare(result, full(*args))


def test_moving_saddle_refreshes_only_its_represented_cells(fragments):
    geometry, sigma, policy, decision = fragments
    form = decision.saddle_form._replace(
        position=decision.saddle_form.position + jnp.asarray([1e-5, 0.0])
    )
    assert bool(jnp.any(decision.normal_form_cells))
    refresh, full = _program(*fragments)
    args = decision.field_coefficients, decision.boundary_flux, form
    result = refresh(*args)
    assert int(result[-1]) == int(jnp.sum(decision.normal_form_cells))
    _compare(result, full(*args))


def test_batched_reuse_preserves_each_lanes_changed_population(fragments):
    geometry, sigma, policy, decision = fragments
    refresh, full = _program(*fragments)
    coefficient = jnp.stack([decision.field_coefficients] * 2)
    level = decision.boundary_flux + jnp.asarray([0.0, 1e-5])
    forms = jax.tree.map(lambda value: jnp.stack([value] * 2), decision.saddle_form)
    result = jax.jit(jax.vmap(refresh))(coefficient, level, forms)
    reference = jax.jit(jax.vmap(full))(coefficient, level, forms)
    assert int(result[-1][0]) == 0
    assert 0 < int(result[-1][1]) < len(geometry.centre)
    _compare(result, reference)


def test_identical_programs_refuse_a_timing_claim():
    from scripts.prototypes.h200_levers.run import refuse_inert

    record = {"rows": [{"lowered_sha256": None, "execution_counts": {"ops": 123}}]}
    with pytest.raises(ValueError, match="inert lever"):
        refuse_inert(record, record)


def test_support_read_defaults_to_full_refresh(carried_case, monkeypatch):
    from types import SimpleNamespace

    from nova.equilibrium import support_read

    parts, support = carried_case
    bank, _, _, convention, policy, moments, _ = parts
    calls = []
    original = topology.refresh_at_stationary_points

    def observed(*values, **kwargs):
        calls.append(kwargs.get("reuse", False))
        return original(*values, **kwargs)

    monkeypatch.setattr(topology, "refresh_at_stationary_points", observed)
    result = support_read._rebuilt_reading(
        bank,
        SimpleNamespace(solved=support.stationary),
        moments,
        convention,
        policy,
        None,
        support.reading,
    )
    jax.block_until_ready(result)
    assert calls == [False]
    assert bool(result.valid)
