"""Bounded traces and independent numerical receipts for the point coupling jet."""

from contextlib import contextmanager, nullcontext
from functools import cache
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
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
        for name in (
            "rangefunction",
            "completeelliptic",
            "elliptic",
            "gradedresidual",
            "momentchannel",
            "polygonanalytic",
            "greens",
        ):
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
        polygon = replacements["nova.biot.polygonanalytic"]
        polygon._reference_modules = replacements
        yield polygon


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


@pytest.fixture(scope="module")
def point_jet_equation_counts():
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
    return before, after


def test_point_jet_expression_bound(point_jet_equation_counts):
    _, after = point_jet_equation_counts
    assert after <= 20000


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "Open question: fixed-capacity scans whose differentiated arithmetic "
        "stays identical to 1e-13 while reducing the point jet tenfold"
    ),
)
def test_point_jet_tenfold_expression_reduction(point_jet_equation_counts):
    before, after = point_jet_equation_counts
    assert after * 10 <= before


def _compile_arm(arm):
    """Observe backend compilation with cache policy fixed at startup."""
    import cache_guard
    from jax._src import compiler

    cache_guard.install()
    cache_disabled = os.environ.get("NOVA_KERNEL_COLD_NEGATIVE_CONTROL") != "1"
    if cache_disabled:
        assert jax.config.jax_enable_compilation_cache is False
        assert jax.config.jax_compilation_cache_dir is None
    else:
        assert jax.config.jax_enable_compilation_cache is True
        assert jax.config.jax_compilation_cache_dir is not None

    from nova.biot import polygonanalytic

    point, geometry = jnp.asarray([5.62, 0.31]), _geometry()
    with (
        _baseline_kernel()
        if arm == "baseline"
        else nullcontext(polygonanalytic) as kernel
    ):
        print(f"MEASUREMENT_MODULE={kernel.__file__}", flush=True)
        print(f"MEASUREMENT_CWD={Path.cwd().resolve()}", flush=True)
        original_compile = compiler.backend_compile_and_load

        def observe_compile(*args, **kwargs):
            started = time.perf_counter()
            executable = original_compile(*args, **kwargs)
            cache_guard.ledger().record(
                "jit_jet-backend-observed",
                "jit_jet",
                time.perf_counter() - started,
                "miss",
                persisted=False,
            )
            return executable

        compiler.backend_compile_and_load = observe_compile
        try:
            started = time.perf_counter()
            executable = jax.jit(_point_jet(kernel)).lower(point, geometry).compile()
            elapsed = time.perf_counter() - started
        finally:
            compiler.backend_compile_and_load = original_compile

        rows = [
            row
            for row in cache_guard.ledger().rows()
            if row["cache_key"].startswith("jit_jet-")
        ]
        print(
            f"CACHE_GUARD_ROWS arm={arm} {json.dumps(rows, sort_keys=True)}", flush=True
        )
        assert sum(row["hits"] for row in rows) == 0, "cache hit: wall is not cold"
        assert sum(row["misses"] for row in rows) == 1, (
            "jet compilation was not observed"
        )
        result = np.asarray(executable(point, geometry))
        assert np.all(np.isfinite(result))
        size = executable.memory_analysis().generated_code_size_in_bytes
        assert size > 0
        print(
            f"COMPILE arm={arm} backend={jax.devices()[0]} "
            f"seconds={elapsed:.9g} executable_bytes={size} cache=miss",
            flush=True,
        )


@pytest.mark.skipif(
    jax.default_backend() != "gpu", reason="cold compile certificate requires a GPU"
)
def test_point_jet_cold_compile():
    assert set(_cold_compile_walls()) == {"baseline", "head"}


@pytest.mark.skipif(
    jax.default_backend() != "gpu", reason="cold compile certificate requires a GPU"
)
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="H200 true cold head compile measured 80.1573431 s against the 30 s bound",
)
def test_point_jet_compile_budget():
    assert _cold_compile_walls()["head"] < 30.0


@cache
def _cold_compile_walls():
    walls = {}
    for arm in ("baseline", "head"):
        environment = os.environ.copy()
        if environment.get("NOVA_KERNEL_COLD_NEGATIVE_CONTROL") != "1":
            environment.pop("JAX_COMPILATION_CACHE_DIR", None)
            environment["JAX_ENABLE_COMPILATION_CACHE"] = "0"
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--cold-arm", arm],
            capture_output=True,
            text=True,
            env=environment,
            check=False,
        )
        print(result.stdout, end="", flush=True)
        print(result.stderr, end="", file=sys.stderr, flush=True)
        assert result.returncode == 0, f"{arm} cold-compile child failed"
        line = next(
            line
            for line in result.stdout.splitlines()
            if line.startswith(f"COMPILE arm={arm} ")
        )
        walls[arm] = float(line.split("seconds=")[1].split()[0])
    return walls


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
    receipt = os.environ.get("NOVA_KERNEL_IDENTITY_RECEIPT")
    if receipt:
        np.savez(receipt, points=points, baseline=jets["baseline"], head=jets["head"])
    relative = np.abs(jets["head"] - jets["baseline"]) / np.maximum(
        np.abs(jets["baseline"]),
        np.finfo(float).tiny,
    )
    worst = np.unravel_index(np.argmax(relative), relative.shape)
    print(
        f"IDENTITY_WORST point={points[worst[0]].tolist()} "
        f"component={worst[1]} baseline={jets['baseline'][worst]:.17g} "
        f"head={jets['head'][worst]:.17g}",
        flush=True,
    )
    maximum = _maximum_error(jets["head"], jets["baseline"])
    print(
        f"IDENTITY kind=polygon points={len(points)} "
        f"component_maxima={json.dumps(maximum.tolist())} "
        f"maximum={maximum.max():.17g}",
        flush=True,
    )
    assert maximum.max() <= 1e-13


def test_kernel_helpers_preserve_eager_derivatives():
    """Graph reuse does not change the arithmetic of unstaged differentiation."""
    from nova.biot.completeelliptic import complete_kind

    with _baseline_kernel() as baseline:
        reference = baseline._reference_modules["nova.biot.completeelliptic"]
        for complement in (0.9, 0.5, 1e-3, 1e-6, 1e-8):
            value = jnp.asarray(complement, dtype=jnp.float64)
            expected = jax.grad(lambda c: reference.complete_kind(c, xp=jnp)[1])(value)
            actual = jax.grad(lambda c: complete_kind(c, xp=jnp)[1])(value)
            np.testing.assert_array_equal(actual, expected)


def test_harmonic_arithmetic_order():
    """Cancellation-sensitive helpers retain the reference operation ordering."""
    from nova.biot import rangefunction

    random = np.random.default_rng(831)
    series = [random.normal(size=1000) for _ in range(8)]
    cases = (
        ("harmonic_multiply", (series, series[:5])),
        ("contract", (series, series)),
        ("deflate", (series, random.uniform(-2, 2, 1000))),
        ("_chebyshev_integral", (series,)),
        ("harmonic_add", (series, series[:3])),
    )
    with _baseline_kernel() as baseline:
        reference = baseline._reference_modules["nova.biot.rangefunction"]
        for name, arguments in cases:
            expected = getattr(reference, name)(*arguments)
            actual = getattr(rangefunction, name)(*arguments)
            for left, right in zip(
                jax.tree.leaves(actual),
                jax.tree.leaves(expected),
                strict=True,
            ):
                np.testing.assert_array_equal(left, right, err_msg=name)


def test_harmonic_recurrence_identity(monkeypatch):
    """The complete recurrence agrees with the revision-pinned arithmetic."""
    from nova.biot import elliptic

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
    assert np.all(np.isfinite(expected))
    assert np.all(expected != 0.0)

    if os.environ.get("NOVA_KERNEL_TRUNCATE_RECURRENCE") == "1":
        evaluate = elliptic.harmonic_moments

        def truncated(parameter, count, **kwargs):
            values = evaluate(parameter, count - 1, **kwargs)
            return values + [kwargs.get("xp", np).zeros_like(values[0])]

        monkeypatch.setattr(elliptic, "harmonic_moments", truncated)
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
    """The filament jet agrees with the revision-pinned special functions."""
    from nova.biot import greens

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

    with _baseline_kernel() as baseline:
        reference = baseline._reference_modules["nova.biot.greens"]
        expected = np.asarray(build(reference)(points))
    actual = np.asarray(build(greens)(points))
    maximum = _maximum_error(actual, expected)
    print(
        f"IDENTITY kind=filament points=10000 maximum={maximum.max():.17g}", flush=True
    )
    assert np.any(expected != 0.0)
    assert maximum.max() <= 1e-13


def test_kernel_helper_sharing_requires_staging():
    """Eager differentiation stays eager while tracing reuses helper graphs."""
    from nova.biot.rangefunction import _array_program

    @_array_program
    def square(value):
        return value * value

    traced = jax.make_jaxpr(square)(2.0)
    assert any(equation.primitive.name == "jit" for equation in traced.jaxpr.eqns)
    assert jax.grad(square)(2.0) == 4.0


if __name__ == "__main__":
    assert sys.argv[1] == "--cold-arm"
    _compile_arm(sys.argv[2])
