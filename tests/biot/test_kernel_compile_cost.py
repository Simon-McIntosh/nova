"""Bounded traces and independent numerical receipts for the point coupling jet."""

from contextlib import contextmanager
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nova.jax.config import configure_dtypes

configure_dtypes()
assert jax.config.jax_enable_x64

BASE_REVISION = os.environ.get(
    "NOVA_KERNEL_BASE_REVISION", "01575b6b164607271c9f1d06cfa3f64f18128646"
)
ROOT = Path(__file__).resolve().parents[2]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextmanager
def _baseline_kernel():
    """Load revision-pinned kernels without changing the production imports."""
    with tempfile.TemporaryDirectory(prefix="biot-reference-") as directory:
        replacements = {}
        for name in ("rangefunction", "elliptic", "momentchannel", "polygonanalytic"):
            source_path = ROOT / "nova" / "biot" / f"{name}.py"
            if name in ("rangefunction", "elliptic"):
                source_path = Path(directory) / f"{name}.py"
                source_path.write_bytes(
                    subprocess.check_output(
                        [
                            "git",
                            "-C",
                            str(ROOT),
                            "show",
                            f"{BASE_REVISION}:nova/biot/{name}.py",
                        ]
                    )
                )
            module = _load(f"reference_{name}", source_path)
            for key, value in tuple(vars(module).items()):
                origin = getattr(value, "__module__", "")
                if origin in replacements:
                    setattr(module, key, getattr(replacements[origin], value.__name__))
            replacements[f"nova.biot.{name}"] = module
            print(f"REFERENCE_MODULE {name}={module.__file__}", flush=True)
        print(f"MEASUREMENT_CWD={Path.cwd().resolve()}", flush=True)
        yield module


def _geometry():
    from nova.biot.greens import section_centroid, second_moments
    from nova.biot.polygon import pad_batch

    polygon = np.asarray(
        [[4.91, -0.10], [5.08, -0.14], [5.17, 0.02], [5.04, 0.15], [4.88, 0.08]]
    )
    edge, weight, norm = pad_batch([polygon])
    centre = section_centroid(polygon)[:, None]
    rr, zz, rz = second_moments(polygon)
    first = np.linalg.solve(np.asarray([[rr, rz], [rz, zz]]), [0.003, -0.002])
    coefficients = np.asarray([1.0, *first])[:, None]
    return tuple(
        map(
            jnp.asarray,
            (
                edge,
                weight,
                norm,
                centre,
                np.asarray([np.nan]),
                np.arange(len(polygon))[:, None],
                coefficients,
            ),
        )
    )


def _point_jet(kernel):
    """Use the packed moment rows and analytic gradient used by the topology read."""

    @jax.jit
    def value_gradient(point, geometry):
        edge, weight, norm, centre, reflection, partner, coefficient = geometry
        rows = kernel.packed_analytic_moments(
            jnp,
            jnp.broadcast_to(point[0], norm.shape),
            jnp.broadcast_to(point[1], norm.shape),
            edge,
            weight,
            norm,
            centre,
            centre,
            reflection,
            partner,
        )
        value = jnp.sum(jnp.stack(rows[:3]) * coefficient)
        radial = jnp.sum(jnp.stack(rows[3:6]) * coefficient)
        vertical = jnp.sum(jnp.stack(rows[6:]) * coefficient)
        gradient = 2 * jnp.pi * point[0] * jnp.stack((vertical, -radial))
        return value, gradient

    def jet(point, geometry):
        def gradient_with_value(target):
            value, gradient = value_gradient(target, geometry)
            return gradient, (value, gradient)

        hessian, (value, gradient) = jax.jacfwd(gradient_with_value, has_aux=True)(
            point
        )
        return jnp.concatenate((jnp.reshape(value, (1,)), gradient, hessian.ravel()))

    return jet


def _unique_equations(graph):
    seen = set()

    def visit(value):
        if id(value) in seen:
            return 0
        seen.add(id(value))
        if hasattr(value, "eqns"):
            return len(value.eqns) + sum(
                visit(equation.params) for equation in value.eqns
            )
        if hasattr(value, "jaxpr"):
            return visit(value.jaxpr)
        if isinstance(value, dict):
            return sum(visit(item) for item in value.values())
        if isinstance(value, (tuple, list)):
            return sum(visit(item) for item in value)
        return 0

    return visit(graph)


def test_point_jet_expression_bound():
    from nova.biot import polygonanalytic

    geometry = _geometry()
    point = jnp.asarray([5.62, 0.31])
    with _baseline_kernel() as baseline:
        before = _unique_equations(
            jax.make_jaxpr(jax.jit(_point_jet(baseline)))(point, geometry)
        )
    after = _unique_equations(
        jax.make_jaxpr(jax.jit(_point_jet(polygonanalytic)))(point, geometry)
    )
    print(
        f"EQUATIONS before={before} after={after} ratio={before / after:.9g}",
        flush=True,
    )
    assert before > 50000, "the counter must see the expanded reference kernel"
    assert after * 10 <= before


def _compile(jet, point, geometry):
    jax.clear_caches()
    cache_enabled = jax.config.jax_enable_compilation_cache
    jax.config.update("jax_enable_compilation_cache", False)
    try:
        started = time.perf_counter()
        executable = jax.jit(jet).lower(point, geometry).compile()
        elapsed = time.perf_counter() - started
    finally:
        jax.config.update("jax_enable_compilation_cache", cache_enabled)
    size = executable.memory_analysis().generated_code_size_in_bytes
    return executable, elapsed, size


@pytest.mark.skipif(
    jax.default_backend() != "gpu", reason="cold compile certificate requires a GPU"
)
def test_point_jet_cold_compile():
    from nova.biot import polygonanalytic

    point, geometry = jnp.asarray([5.62, 0.31]), _geometry()
    executable, elapsed, size = _compile(_point_jet(polygonanalytic), point, geometry)
    assert np.all(np.isfinite(executable(point, geometry)))
    print(
        f"COMPILE backend={jax.devices()[0]} seconds={elapsed:.9g} "
        f"executable_bytes={size}",
        flush=True,
    )
    assert size > 0
    assert elapsed < 30.0


def _points():
    random = np.random.default_rng(42091)
    angles = random.uniform(-np.pi, np.pi, 10000)
    distance = np.exp(random.uniform(np.log(2e-3), np.log(30.0), 10000))
    # Positive radii cover near material, its interior, and distant axial targets.
    return np.column_stack(
        (
            np.abs(5.02 + distance * np.cos(angles)) + 1e-4,
            0.013 + distance * np.sin(angles),
        )
    )


def _maximum_error(actual, expected):
    assert np.all(np.isfinite(expected))
    assert np.all(np.isfinite(actual))
    return np.max(
        np.abs(actual - expected) / np.maximum(np.abs(expected), np.finfo(float).tiny),
        axis=0,
    )


@pytest.mark.skipif(
    jax.default_backend() != "gpu", reason="large point identity gate requires a GPU"
)
def test_point_jet_identity():
    from nova.biot import polygonanalytic

    geometry, points = _geometry(), _points()
    jets = {}
    with _baseline_kernel() as baseline:
        for arm, module in (("baseline", baseline), ("head", polygonanalytic)):
            jet = _point_jet(module)
            started = time.perf_counter()
            executable = (
                jax.jit(jax.vmap(jet, in_axes=(0, None)))
                .lower(jnp.asarray(points[:100]), geometry)
                .compile()
            )
            print(
                f"IDENTITY_COMPILE arm={arm} "
                f"seconds={time.perf_counter() - started:.9g} "
                f"executable_bytes={executable.memory_analysis().generated_code_size_in_bytes}",
                flush=True,
            )
            jets[arm] = np.concatenate(
                [
                    np.asarray(executable(jnp.asarray(chunk), geometry))
                    for chunk in np.split(points, 100)
                ]
            )
    maximum = _maximum_error(jets["head"], jets["baseline"])
    print(
        f"IDENTITY kind=polygon points={len(points)} "
        f"component_maxima={json.dumps(maximum.tolist())} "
        f"maximum={maximum.max():.17g}",
        flush=True,
    )
    assert maximum.max() <= 1e-13


def test_harmonic_recurrence_identity(monkeypatch):
    """The complete recurrence agrees with the revision-pinned arithmetic."""
    from nova.biot import elliptic

    if os.environ.get("NOVA_KERNEL_TRUNCATE_RECURRENCE") == "1":
        scan = elliptic._scan

        def truncated(function, initial, values, xp, **kwargs):
            if function.__name__ == "ascend":
                state, result = scan(function, initial, values[:-1], xp, **kwargs)
                result = jax.tree.map(
                    lambda value: xp.concatenate((value, xp.zeros_like(value[:1]))),
                    result,
                )
                return state, result
            return scan(function, initial, values, xp, **kwargs)

        monkeypatch.setattr(elliptic, "_scan", truncated)
    parameter = np.linspace(0.99001, 0.999999, 10000)
    with _baseline_kernel() as kernel:
        expected = np.asarray(
            kernel.harmonic_moments(
                parameter,
                8,
                complement=1.0 - parameter,
                xp=np,
            )
        )
    actual = np.asarray(
        elliptic.harmonic_moments(
            parameter,
            8,
            complement=1.0 - parameter,
            xp=np,
        )
    )
    maximum = np.max(np.abs(actual - expected) / np.abs(expected))
    print(f"RECURRENCE_IDENTITY points=10000 maximum={maximum:.17g}", flush=True)
    assert maximum <= 1e-13


@pytest.mark.skipif(jax.default_backend() != "gpu", reason="point batch gate uses GPU")
def test_filament_point_jet_identity():
    """The filament route is outside the harmonic helpers and stays identical."""
    from nova.biot import greens

    reference = _load("reference_greens", ROOT / "nova/biot/greens.py")
    points = jnp.asarray(_points())

    def build(module):
        def value_gradient(point):
            value, radial, vertical = module.traced_filament_greens(
                jnp,
                point[0],
                point[1],
                5.02,
                0.013,
            )
            gradient = 2 * jnp.pi * point[0] * jnp.stack((vertical, -radial))
            return gradient, (value, gradient)

        def jet(point):
            hessian, (value, gradient) = jax.jacfwd(value_gradient, has_aux=True)(point)
            return jnp.concatenate(
                (jnp.reshape(value, (1,)), gradient, hessian.ravel())
            )

        return jax.jit(jax.vmap(jet))

    with _baseline_kernel():
        expected = np.asarray(build(reference)(points))
    actual = np.asarray(build(greens)(points))
    maximum = _maximum_error(actual, expected)
    print(
        f"IDENTITY kind=filament points=10000 maximum={maximum.max():.17g}", flush=True
    )
    assert np.any(expected != 0.0)
    assert maximum.max() <= 1e-13
