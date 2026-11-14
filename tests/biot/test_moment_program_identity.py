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


def _byte_differences(actual, expected, points, *, report):
    actual, expected = map(np.ascontiguousarray, (actual, expected))
    assert actual.dtype == expected.dtype == np.float64
    different = actual.view(np.uint64) != expected.view(np.uint64)
    if report:
        for index in np.argwhere(different)[:3]:
            key = tuple(index)
            observed, reference = float(actual[key]), float(expected[key])
            offset = int(np.ravel_multi_index(key, actual.shape)) * actual.itemsize

            def ordered(bits):
                bits = int(bits)
                return (~bits & ((1 << 64) - 1)) if bits >> 63 else bits | (1 << 63)

            distance = abs(
                ordered(actual.view(np.uint64)[key])
                - ordered(expected.view(np.uint64)[key])
            )
            print(
                f"BYTE_DIFFERENCE point={points[index[0]].tolist()} "
                f"component={index[1:].tolist()} byte_offset={offset} "
                f"ulp_distance={distance} actual={observed:.17g} "
                f"reference={reference:.17g} actual_hex={observed.hex()} "
                f"reference_hex={reference.hex()} "
                f"zero_sign_only={observed == reference == 0.0}"
            )
    return int(np.count_nonzero(different))


def test_byte_diagnostics_distinguish_rounding_and_signed_zero(capsys):
    points = np.asarray(((6.19, -5.71), (5.0, 0.0)))
    expected = np.zeros((2, 9, 1), dtype=np.float64)
    expected[0, 4, 0] = float.fromhex("-0x1.899fdf4452628p-36")
    actual = expected.copy()
    actual[0, 4, 0] = np.nextafter(expected[0, 4, 0], -np.inf)
    actual[1, 0, 0] = -0.0
    assert _byte_differences(actual, expected, points, report=True) == 2
    rounding, signed_zero = capsys.readouterr().out.splitlines()
    assert "point=[6.19, -5.71] component=[4, 0] byte_offset=32" in rounding
    assert "ulp_distance=1" in rounding
    assert "zero_sign_only=False" in rounding
    assert "reference_hex=-0x1.899fdf4452628p-36" in rounding
    assert "component=[0, 0] byte_offset=72" in signed_zero
    assert "zero_sign_only=True" in signed_zero


def test_byte_diagnostics_count_a_larger_departure_without_reporting(capsys):
    expected = np.asarray([[[1.0], [2.0]]])
    actual = expected.copy()
    actual[0, 1, 0] += 1e-6
    points = np.asarray(((5.0, 0.0),))
    assert _byte_differences(actual, expected, points, report=False) == 1
    assert capsys.readouterr().out == ""
    assert _byte_differences(expected, expected, points, report=True) == 0


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
    different = 0
    with _baseline_kernel() as baseline:
        before, after = program(baseline), program(polygonanalytic)
        for chunk in np.split(points, 100):
            expected = jax.tree.map(np.asarray, before(jnp.asarray(chunk)))
            actual = jax.tree.map(np.asarray, after(jnp.asarray(chunk)))
            assert np.all(np.isfinite(expected[0]))
            assert np.all(np.isfinite(expected[1]))
            different += _byte_differences(
                actual[0], expected[0], chunk, report=different == 0
            )
            maxima.append(float(relative_error(actual[1], expected[1]).max()))
    maximum = max(maxima)
    print(
        f"IDENTITY packed points={len(points)} primal_bit_identical={different == 0} "
        f"different_elements={different} "
        f"tangent_max_relative={maximum:.17g}"
    )
    assert different == 0
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
    different = 0
    point_scaled = 0.0
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
            different += _byte_differences(
                actual[..., :3], expected[..., :3], chunk, report=different == 0
            )
            maximum = max(
                maximum, float(relative_error(actual[..., 3:], expected[..., 3:]).max())
            )
            scale = np.max(np.abs(expected[..., 3:]), axis=(1, 2, 3))
            scaled = np.abs(actual[..., 3:] - expected[..., 3:]) / np.maximum(
                scale[:, None, None, None], np.finfo(float).tiny
            )
            point_scaled = max(point_scaled, float(scaled.max()))
    print(
        f"IDENTITY production points={len(points)} "
        f"value_gradient_bit_identical={different == 0} different_elements={different} "
        f"hessian_max_relative={maximum:.17g} "
        f"hessian_point_scaled_information={point_scaled:.17g}"
    )
    assert different == 0
    assert maximum <= 1e-9
