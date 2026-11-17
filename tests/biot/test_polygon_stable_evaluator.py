"""The integrated polygon evaluator obeys the retained plasma accuracy bound."""

import json
import os
from pathlib import Path

import pytest


@pytest.mark.parametrize("compiled", (False, True), ids=("eager", "compiled"))
def test_stable_derivatives_keep_original_values(monkeypatch, compiled):
    from nova.jax.config import configure_dtypes

    configure_dtypes()
    import jax
    import jax.numpy as jnp
    import numpy as np

    from nova.biot import polygonanalytic as kernel

    assert jax.config.jax_enable_x64 is True
    calls = []

    def expression(xp, radius, height, *arguments, stable_poles=False, **options):
        calls.append(stable_poles)
        # A fifth-degree height term separates fourth- and sixth-order stencils.
        row = (
            2 * radius + height + (height / 1e-3) ** 5
            if stable_poles
            else radius - height
        )
        return tuple((index + 1) * row for index in range(9))

    monkeypatch.setattr(kernel, "_packed_analytic_moments_primal", expression)
    edge = jnp.asarray([[[1.0], [0.0], [2.0], [1.0]]])
    one = jnp.ones(1)
    centre = jnp.zeros((2, 1))
    zero = jnp.zeros(1, dtype=jnp.int32)

    def program(point):
        def rows(target):
            return jnp.stack(
                kernel.packed_analytic_moments(
                    jnp,
                    target[:1],
                    target[1:],
                    edge,
                    one,
                    one,
                    centre,
                    centre,
                    zero,
                    zero,
                )
            )

        return jax.jvp(rows, (point,), (jnp.asarray((0.37, -0.81)),))

    evaluate = jax.jit(program) if compiled else program
    value, derivative = evaluate(jnp.asarray((2.0, 0.0)))
    factors = np.arange(1, 10)[:, None]
    np.testing.assert_array_equal(value, 2.0 * factors)
    np.testing.assert_allclose(derivative, (2 * 0.37 - 0.81) * factors, atol=1e-11)
    # One original expression supplies values; every derivative uses the stable one.
    assert calls.count(False) == 1
    assert calls.count(True) > 0


@pytest.fixture(scope="module")
def integrated_receipt():
    path = os.environ.get("NOVA_POLYGON_INTEGRATED_RECEIPT")
    if not path:
        pytest.skip("requires the integrated 64-node plasma measurement")
    receipt = json.loads(Path(path).read_text())
    assert receipt["completed"] is True and receipt["count"] == 64
    assert [row["state"] for row in receipt["states"]] == [220, 333, 336]
    return receipt


@pytest.mark.parametrize("index", range(3), ids=("220", "333", "336"))
def test_integrated_plasma_trace(integrated_receipt, index):
    row = integrated_receipt["states"][index]
    assert row["reference_trace"]["relative"] < 1e-8
    assert row["trace"]["relative"] <= 1e-8


@pytest.mark.parametrize("component", ("psi_RR", "psi_RZ", "psi_ZR", "psi_ZZ"))
def test_interior_hessian_reference(integrated_receipt, component):
    for row in integrated_receipt["states"]:
        assert row["components"][component]["reference_bound"] < 1e-8
        assert row["components"][component]["relative"] <= 1e-8
        if component in row["boundary_reference"]:
            assert row["boundary_reference"][component]["relative"] <= 1e-8
