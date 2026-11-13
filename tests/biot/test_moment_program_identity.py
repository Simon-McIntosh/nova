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


@pytest.mark.skipif(
    jax.default_backend() != "gpu", reason="large identity gate requires a GPU"
)
def test_production_source_rows_preserve_jet_identity():
    import ast
    from pathlib import Path
    import subprocess
    from types import SimpleNamespace

    from nova.equilibrium import fixed_points
    from tests.biot.test_kernel_compile_cost import BASE_REVISION

    root = Path(__file__).resolve().parents[2]
    source = subprocess.check_output(
        [
            "git",
            "-C",
            str(root),
            "show",
            f"{BASE_REVISION}:nova/equilibrium/fixed_points.py",
        ],
        text=True,
    )
    node = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "_source_jets"
    )
    edge, weight, norm, centre, reflection, partner, _ = _geometry()
    coupling = SimpleNamespace(
        edge=edge,
        weight=weight,
        norm=norm,
        centre=centre.T,
        reflection_axis=reflection,
        reflection_partner=partner,
        inverse_second_moment=jnp.eye(2)[None],
    )
    points = _points()
    maximum = 0.0
    with _baseline_kernel() as baseline:
        namespace = {
            "jax": jax,
            "jnp": jnp,
            "packed_analytic_moments": baseline.packed_analytic_moments,
        }
        exec(
            compile(
                ast.Module(body=[node], type_ignores=[]),
                "<reference-source-jets>",
                "exec",
            ),
            namespace,
        )
        before = jax.jit(
            jax.vmap(lambda point: namespace["_source_jets"](point, coupling))
        )
        after = jax.jit(
            jax.vmap(lambda point: fixed_points._source_jets(point, coupling))
        )
        for chunk in np.split(points, 100):
            expected = np.asarray(before(jnp.asarray(chunk)))
            actual = np.asarray(after(jnp.asarray(chunk)))
            assert np.all(np.isfinite(expected))
            assert actual[..., :3].tobytes() == expected[..., :3].tobytes()
            maximum = max(
                maximum, float(relative_error(actual[..., 3:], expected[..., 3:]).max())
            )
    print(
        f"IDENTITY production points={len(points)} "
        f"value_gradient_bit_identical=True hessian_max_relative={maximum:.17g}"
    )
    assert maximum <= 1e-9
