"""Independent packed-moment values and tangents on the shared point set."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nova.biot import polygonanalytic
from tests.biot.tangent_identity import relative_error
from tests.biot.test_kernel_compile_cost import _baseline_kernel, _geometry, _points

assert jax.config.jax_enable_x64 is True


@pytest.mark.skipif(
    jax.default_backend() != "gpu", reason="large identity gate requires a GPU"
)
def test_packed_moment_primal_and_tangent_identity():
    geometry = _geometry()
    edge, weight, norm, centre, reflection, partner, _ = geometry

    def program(kernel):
        def single(point):
            def rows(target):
                return jnp.stack(
                    kernel.packed_analytic_moments(
                        jnp,
                        jnp.broadcast_to(target[0], norm.shape),
                        jnp.broadcast_to(target[1], norm.shape),
                        edge,
                        weight,
                        norm,
                        centre,
                        centre,
                        reflection,
                        partner,
                    )
                )

            return jax.jvp(rows, (point,), (jnp.asarray((0.37, -0.81)),))

        return jax.jit(jax.vmap(single))

    points = _points()
    maxima = []
    with _baseline_kernel() as baseline:
        before, after = program(baseline), program(polygonanalytic)
        for chunk in np.split(points, 100):
            expected = jax.tree.map(np.asarray, before(jnp.asarray(chunk)))
            actual = jax.tree.map(np.asarray, after(jnp.asarray(chunk)))
            assert np.all(np.isfinite(expected[0]))
            assert np.all(np.isfinite(expected[1]))
            assert actual[0].tobytes() == expected[0].tobytes()
            maxima.append(float(relative_error(actual[1], expected[1]).max()))
    maximum = max(maxima)
    print(
        f"IDENTITY packed points={len(points)} primal_bit_identical=True "
        f"tangent_max_relative={maximum:.17g}"
    )
    assert maximum <= 1e-9
