"""Runtime coil and flux-function inputs on the operator-bank route."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

from dataclasses import replace
import logging
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nova.equilibrium import current, fixed_points, support_read as bank
from nova.equilibrium.constraint import VerticalPositionPin, uniform_radial_field_jet

assert jax.config.jax_enable_x64 is True


def _table(pressure=-1.0, diamagnetic=0.0):
    return current.TabulatedProfile(
        jnp.full(current.PROFILE_KNOT_COUNT, pressure, dtype=jnp.float64),
        jnp.full(current.PROFILE_KNOT_COUNT, diamagnetic, dtype=jnp.float64),
    )


@pytest.fixture(scope="module")
def small():
    from tests.test_support_read import _case

    operands, _, roots, _, _, _, _ = _case("limited")
    cells = len(operands.geometry.centre)
    rows = jnp.zeros((len(operands.nodes), cells, 3, 6), dtype=jnp.float64)
    coils = uniform_radial_field_jet(operands.nodes)[:, None] * 1e-3
    operands = replace(
        operands, rows=rows, coil_exterior=coils, coil_count=jnp.asarray(1, jnp.int32)
    )
    cache = fixed_points.RowCache(
        operands.mesh,
        operands.nodes,
        jnp.arange(len(operands.nodes)),
        rows,
        operands.exterior,
    )
    from nova.equilibrium import topology
    from nova.equilibrium.solve_request import TopologyPolicy

    convention = topology.TopologyConvention.from_cocos(17, 1.0)
    policy = TopologyPolicy()
    quadrature = current.Quadrature(sigma=convention.sigma, segments=8)
    state = tuple(jnp.zeros(cells, dtype=jnp.float64) for _ in range(3))
    read = jax.jit(
        lambda op, window, state: bank.read(
            op, window, state, roots, roots, convention, policy
        )
    )(operands, cache, state)
    assert bool(read.valid)
    return operands, cache, roots, convention, policy, quadrature, state, read


def test_certificate_fixture_unchanged_and_constant_table_reproduces_profile():
    from scripts.prototypes.bank_newton_step.measure import machine

    operands, _, _, _, _, profile, quadrature, moments, _ = machine("diverted")
    assert moments[0].size == 550
    plain = bank.nodal_jets(operands, moments)
    expected = operands.exterior + jnp.einsum(
        "ncmq,cm->nq", operands.rows, jnp.stack(moments, axis=-1)
    )
    np.testing.assert_allclose(plain, expected, rtol=1e-12, atol=0)
    supplied = replace(
        operands, coil_exterior=jnp.zeros((len(plain), 2, 6)), coil_count=jnp.asarray(2)
    )
    np.testing.assert_allclose(
        bank.nodal_jets(supplied, moments, jnp.ones(2)), plain, rtol=1e-12, atol=0
    )
    knots = jnp.linspace(0.0, 1.0, current.PROFILE_KNOT_COUNT)
    tabulated = current.TabulatedProfile(
        profile.p_prime(knots), profile.ff_prime(knots)
    )
    radius = jnp.linspace(1.0, 5.0, 99)
    normalized = jnp.linspace(0.0, 1.0, 99)
    np.testing.assert_allclose(
        tabulated.current_density(radius, normalized),
        profile.current_density(radius, normalized),
        rtol=1e-12,
        atol=0,
    )
    # Both profiles book the same field on the same connected support.
    from nova.equilibrium import topology

    _, analytic, _, axis, saddle = __import__(
        "tests.equilibrium.test_topology_read", fromlist=["_analytic_inputs"]
    )._analytic_inputs("diverted")
    from nova.equilibrium.solve_request import TopologyPolicy

    reading = jax.jit(topology.read)(
        analytic,
        operands.geometry,
        topology.TopologyConvention.from_cocos(17, 1.0),
        TopologyPolicy(normal_form_radius=0.054806712567833996),
    )
    psi = plain[operands.sample_index, 0]
    evaluate = jax.jit(
        lambda source: current.book(psi, reading, operands.geometry, source, quadrature)
    )
    constant = current.book(psi, reading, operands.geometry, profile, quadrature)
    table = evaluate(tabulated)
    assert bool(constant.valid) and bool(table.valid)
    for actual, reference in zip(table[:3], constant[:3], strict=True):
        np.testing.assert_allclose(actual, reference, rtol=1e-12, atol=1e-14)
    print("CERTIFICATE cells 550 relative tolerance 1e-12 knots 33 cubic Hermite")


def test_coil_bank_padding_storage_and_count(small, tmp_path):
    operands, cache, roots, convention, policy, quadrature, state, read = small
    padded = bank.SupportOperands.from_rows(
        operands.geometry,
        operands.rows,
        operands.exterior,
        operands.nodes,
        operands.sample_index,
        bucket=len(operands.nodes),
        coil_exterior=operands.coil_exterior,
    )
    assert padded.coil_exterior.shape == (padded.bucket, 1, 6)
    assert int(padded.coil_count) == 1
    assert not np.any(np.asarray(padded.coil_exterior)[len(operands.nodes) :])
    digest = bank.write_bank(padded, tmp_path)
    restored = bank.read_bank(tmp_path, digest, padded.geometry, padded.mesh)
    np.testing.assert_array_equal(restored.coil_exterior, padded.coil_exterior)
    np.testing.assert_array_equal(restored.coil_count, padded.coil_count)
    masked = replace(restored, coil_count=jnp.asarray(0, jnp.int32))
    np.testing.assert_array_equal(
        bank.nodal_jets(masked, state, jnp.ones(1)), bank.nodal_jets(masked, state)
    )
    with pytest.raises(ValueError, match="every bank coil"):
        bank.nodal_jets(restored, state, jnp.zeros(2))


def test_tabulated_derivative_matches_autodiff_at_and_between_knots():
    knots = jnp.linspace(0.0, 1.0, current.PROFILE_KNOT_COUNT)
    profile = current.TabulatedProfile(knots**2 - 1.0, 1e-6 * (1 - knots**3))
    points = jnp.concatenate((knots[1:-1], knots[:-1] + 0.37 / 32))
    radius = jnp.full_like(points, 2.4)
    automatic = jax.vmap(jax.grad(lambda r, x: profile.current_density(r, x), 1))(
        radius, points
    )
    np.testing.assert_allclose(
        profile.density_gradient(radius, points), automatic, rtol=1e-12, atol=1e-12
    )
    assert float(jnp.max(jnp.abs(automatic))) > 1.0
    with pytest.raises(ValueError, match="33 knots"):
        current.TabulatedProfile(jnp.ones(8), jnp.ones(8))


def _traced_booking(
    operands, moments, support, pressure, diamagnetic, currents, quadrature
):
    return bank.step_residual(
        operands,
        moments,
        moments,
        support,
        current.TabulatedProfile(pressure, diamagnetic),
        quadrature,
        coil_currents=currents,
    ).booked


def test_slice_inputs_cause_zero_new_compilations(small, caplog):
    operands, cache, roots, convention, policy, quadrature, state, read = small
    caplog.set_level(logging.WARNING, logger="jax")
    program = jax.jit(_traced_booking)
    source = _table()
    currents = jnp.asarray([0.0])
    next_currents = jnp.asarray([0.01])
    next_pressure = source.p_prime * 1.01
    captured = os.environ.get("NOVA_CAPTURE_SLICE_CURRENTS") == "1"

    def apply(values, pressure):
        if captured:
            # A distinct closure per slice bakes its currents into the program.
            program_for_slice = jax.jit(
                lambda op, st, rd, pp, ff, quad: _traced_booking(
                    op, st, rd, pp, ff, values, quad
                )
            )
            return program_for_slice(
                operands, state, read, pressure, source.ff_prime, quadrature
            )
        return program(
            operands, state, read, pressure, source.ff_prime, values, quadrature
        )

    with jax.log_compiles(True):
        first = apply(currents, source.p_prime)
        jax.block_until_ready(first)
        warm = [
            record.message
            for record in caplog.records
            if "Compiling jit(" in record.message
        ]
        assert warm, "compile-log instrument must see the cold program"
        caplog.clear()
        second = apply(next_currents, next_pressure)
        jax.block_until_ready(second)
        added = [
            record.message
            for record in caplog.records
            if "Compiling jit(" in record.message
        ]
    assert bool(first.valid) and bool(second.valid)
    assert not np.array_equal(
        np.asarray(first.cell_current), np.asarray(second.cell_current)
    )
    print(
        "SLICE_COMPILATIONS cold",
        len(warm),
        "changed",
        len(added),
        "captured",
        captured,
    )
    for message in added:
        print("COMPILE_LOG", message)
    assert len(added) == 0


def test_coils_reach_read_residual_and_jacobian(small):
    operands, cache, roots, convention, policy, quadrature, state, reading = small
    source = _table()
    currents = jnp.asarray([0.4])
    effective = bank.driven_operands(operands, currents)
    window = bank._driven_cache(effective, cache)
    actual = jax.jit(
        lambda op, win, amps: bank.read(
            op, win, state, roots, roots, convention, policy, coil_currents=amps
        )
    )(operands, cache, currents)
    expected = jax.jit(
        lambda op, win: bank.read(op, win, state, roots, roots, convention, policy)
    )(effective, window)
    np.testing.assert_allclose(actual.reading.axis, expected.reading.axis, rtol=1e-12)
    assert np.linalg.norm(np.asarray(actual.reading.axis - reading.reading.axis)) > 1e-5
    direct = bank.step_residual(effective, state, state, actual, source, quadrature)
    traced = bank.step_residual(
        operands, state, state, actual, source, quadrature, coil_currents=currents
    )
    for value, reference in zip(traced.residual, direct.residual, strict=True):
        np.testing.assert_allclose(value, reference, rtol=1e-12, atol=0)
    matrix = jax.jit(
        lambda op, amps: bank.step_jacobian(
            op, state, actual, source, None, quadrature, coil_currents=amps
        )
    )(operands, currents)
    np.testing.assert_allclose(matrix, jnp.eye(matrix.shape[0]), rtol=1e-12, atol=1e-12)


def test_pin_holds_independent_lane_centroids(small):
    operands, cache, roots, convention, policy, quadrature, state, reading = small
    source = _table()
    target = jnp.asarray([0.005, -0.008], dtype=jnp.float64)
    pin = VerticalPositionPin(target, source_scale=1e-3, tolerance=1e-8)
    moments = tuple(jnp.broadcast_to(value, (2, value.size)) for value in state)
    seeds = jnp.broadcast_to(roots, (2, *roots.shape))

    @jax.jit
    def program(bank_operands, window, state, amps, pressure, ff, target):
        return bank.solve_batch(
            bank_operands,
            window,
            state,
            seeds,
            seeds,
            convention,
            policy,
            lambda lane: current.TabulatedProfile(pressure[lane], ff[lane]),
            None,
            quadrature,
            max_steps=4,
            freeze_active_set=False,
            coil_currents=amps,
            vertical_pin=replace(pin, target_height=target),
        )

    pressure = jnp.stack((source.p_prime, source.p_prime * 1.1))
    ff = jnp.stack((source.ff_prime, source.ff_prime))
    result = program(
        operands, cache, moments, jnp.asarray([[0.0], [0.2]]), pressure, ff, target
    )
    jax.block_until_ready(result)
    height = jax.vmap(lambda *values: bank._centroid_height(operands.geometry, values))(
        *result.moments
    )
    print(
        "PIN target",
        np.asarray(target),
        "actual",
        np.asarray(height),
        "error",
        np.asarray(result.pin_error),
        "amplitude",
        np.asarray(result.pin_amplitude),
        "residual",
        np.asarray(result.residual),
        "converged",
        np.asarray(result.converged),
    )
    assert np.all(result.converged)
    np.testing.assert_allclose(height, target, rtol=0, atol=1e-8)
    assert np.max(np.abs(np.asarray(result.residual))) <= 1e-10
    assert np.all(np.abs(np.asarray(result.pin_amplitude)) > 0)
