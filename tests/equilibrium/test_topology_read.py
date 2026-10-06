"""Total-field null tangents and regular saddle support contracts."""

# Configure precision before importing modules with array-valued defaults.
# ruff: noqa: E402
from dataclasses import dataclass, field as dataclass_field, replace
import json

from nova.jax.config import configure_dtypes

configure_dtypes()

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad

from nova.equilibrium.solve_request import ForwardSolvePolicy, TopologyPolicy
from nova.equilibrium.topology import (
    FieldJet,
    TopologyReason,
    saddle_cell_fragments,
    saddle_normal_form,
    stationary_read,
)

assert jax.config.jax_enable_x64 is True


class KernelCompileBarrier(AssertionError):
    """The native kernel compiler did not finish inside its wall budget."""


kernel_compile_expected_failure = pytest.mark.xfail(
    strict=True,
    raises=KernelCompileBarrier,
    reason=("Biot harmonic kernel compilation exceeds the per-test wall bound"),
)


@pytest.fixture(scope="module")
def kernel_compile_budget(tmp_path_factory):
    """Bound the shared native compiler prerequisite in its own process.

    A signal timeout cannot interrupt a native compiler call. A child process
    enforces the wall bound and retains its output; only that measured timeout
    is an expected failure. Other exceptions and numerical failures stay red.
    """
    import os
    from pathlib import Path
    import subprocess
    import sys

    directory = Path(
        os.environ.get(
            "NOVA_TOPOLOGY_EVIDENCE_DIR", tmp_path_factory.mktemp("kernel-compile")
        )
    )
    directory.mkdir(parents=True, exist_ok=True)
    module = str(Path(__file__).resolve())
    script = """
import json, resource, runpy, sys, time
m = runpy.run_path(sys.argv[1])
jax = m['jax']
from nova.equilibrium.topology import read, TopologyConvention
jax.config.update('jax_enable_compilation_cache', False)
o, total, wall, _, _ = m['_analytic_inputs']('limited')
g = m['_realised_hex_geometry'](wall, 132)
f = m['_kernel_backed_field']('limited', o, total, g)
print('KERNEL_COMPILE_INPUT_READY', flush=True)
start = time.perf_counter()
program = jax.jit(read).lower(
    f, g, TopologyConvention.from_cocos(17, 1.0), m['TopologyPolicy']()
).compile()
assert program.runtime_executable() is not None
receipt = {
    'wall_seconds': time.perf_counter()-start,
    'peak_host_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
}
print('KERNEL_COMPILE_PREREQUISITE_COMPLETE ' + json.dumps(receipt), flush=True)
"""
    command = [sys.executable, "-u", "-c", script, module]
    log = directory / "kernel-compile-prerequisite.log"
    environment = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false")
    with log.open("w") as stream:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
        stream.write(
            f"revision={revision} tree={Path.cwd()} command="
            + json.dumps(command)
            + "\n"
        )
        stream.flush()
        process = subprocess.Popen(
            command, stdout=stream, stderr=subprocess.STDOUT, env=environment
        )
        print("KERNEL_COMPILE_PROCESS", process.pid, str(log), flush=True)
        try:
            code = process.wait(timeout=600)
        except subprocess.TimeoutExpired:
            print("KERNEL_COMPILE_TIMEOUT", process.pid, "600 seconds", flush=True)
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            stream.write(
                "\nWALL_LIMIT_SECONDS=600\nEXIT=" + str(process.returncode) + "\n"
            )
            raise KernelCompileBarrier(
                f"native 132-cell kernel compile exceeded 600 seconds; receipt: {log}"
            ) from None
        stream.write("\nEXIT=" + str(code) + "\n")
    if code:
        raise RuntimeError(f"kernel prerequisite exited {code}; inspect {log}")
    assert "KERNEL_COMPILE_PREREQUISITE_COMPLETE" in log.read_text()


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class QuadraticPointField:
    """Point evaluator whose translating stationary point is known exactly."""

    centre: jax.Array
    hessian: jax.Array

    def evaluate(self, point):
        offset = point - self.centre
        return FieldJet(
            0.5 * offset @ self.hessian @ offset,
            self.hessian @ offset,
            self.hessian,
        )


def test_topology_policy_round_trip():
    policy = replace(
        ForwardSolvePolicy(),
        topology=TopologyPolicy(
            null_capacity=7,
            hessian_tolerance=1e-8,
            normal_form_radius=0.2,
            normal_form_pitch_floor=0.0,
        ),
    )
    assert (
        ForwardSolvePolicy.from_dict(json.loads(json.dumps(policy.to_dict()))) == policy
    )
    for invalid in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            TopologyPolicy(null_capacity=invalid)
    for invalid in (0.0, -1.0, np.nan, np.inf):
        with pytest.raises(ValueError):
            TopologyPolicy(hessian_tolerance=invalid)
    for invalid in (-1.0, np.nan, np.inf):
        with pytest.raises(ValueError):
            TopologyPolicy(normal_form_pitch_floor=invalid)
    assert TopologyPolicy().normal_form_pitch_floor == 1.5
    executable = (
        jax.jit(lambda value: value.hessian_tolerance).lower(policy.topology).compile()
    )
    assert float(executable(policy.topology)) == policy.topology.hessian_tolerance


def test_stationary_position_implicit_tangent():
    field = QuadraticPointField(
        jnp.asarray((2.0, 0.1)), jnp.asarray(((-3.0, 0.2), (0.2, 2.0)))
    )
    seed = jnp.asarray((2.05, 0.06))
    policy = TopologyPolicy()

    def position(centre):
        return stationary_read(
            replace(field, centre=centre), seed, 0.1, policy
        ).position

    direction = jnp.asarray((0.2, -0.7))
    actual, tangent = jax.jvp(position, (field.centre,), (direction,))
    np.testing.assert_allclose(actual, field.centre, atol=1e-14)
    np.testing.assert_allclose(tangent, direction, atol=1e-14)
    step = 1e-5
    central = (
        position(field.centre + step * direction)
        - position(field.centre - step * direction)
    ) / (2 * step)
    np.testing.assert_allclose(tangent, central, atol=2e-11)
    singular = replace(field, hessian=jnp.asarray(((1.0, 0.0), (0.0, 0.0))))
    refused = stationary_read(singular, seed, 0.1, policy)
    assert not refused.valid
    assert not refused.tangent_valid
    assert refused.reason == TopologyReason.SINGULAR_TANGENT
    _, singular_tangent = jax.jvp(
        lambda centre: (
            stationary_read(
                replace(singular, centre=centre), seed, 0.1, policy
            ).position
        ),
        (field.centre,),
        (direction,),
    )
    assert np.all(np.isnan(singular_tangent))


def _saddle_jet():
    # psi = x*x - y*y + 0.3*x**3 + 0.2*x*x*y.
    third = np.zeros((2, 2, 2), dtype=np.float64)
    third[0, 0, 0] = 1.8
    third[0, 0, 1] = third[0, 1, 0] = third[1, 0, 0] = 0.4
    return jnp.diag(jnp.asarray((2.0, -2.0))), jnp.asarray(third)


def _hexagon(radius):
    angle = np.arange(6) * np.pi / 3.0
    return radius * np.stack((np.cos(angle), np.sin(angle)), axis=1)


def _exact_core_area(vertices):
    def height(radius):
        heights = []
        for first, second in zip(vertices, np.roll(vertices, -1, axis=0), strict=True):
            if abs(second[0] - first[0]) < 1e-15:
                continue
            fraction = (radius - first[0]) / (second[0] - first[0])
            if 0.0 <= fraction <= 1.0:
                heights.append(first[1] + fraction * (second[1] - first[1]))
        if len(heights) < 2:
            return 0.0
        discriminant = 0.04 * radius**4 + 4 * (radius**2 + 0.3 * radius**3)
        lower = (0.2 * radius**2 - np.sqrt(discriminant)) / 2.0
        upper = (0.2 * radius**2 + np.sqrt(discriminant)) / 2.0
        return max(0.0, min(max(heights), upper) - max(min(heights), lower))

    return quad(
        height,
        0.0,
        float(vertices[:, 0].max()),
        epsabs=1e-13,
        epsrel=1e-12,
        points=[float(vertices[:, 0].max()) / 2],
        limit=200,
    )[0]


def test_saddle_curved_support_order():
    hessian, third = _saddle_jet()
    form = saddle_normal_form(jnp.zeros(2), hessian, third, 1e-10)
    assert form.valid
    np.testing.assert_allclose(
        np.einsum("ai,ij,aj->a", form.direction, hessian, form.direction),
        0.0,
        atol=1e-14,
    )
    radii = np.asarray((0.2, 0.12, 0.08, 0.05, 0.03))
    errors = []
    for radius in radii:
        vertices = _hexagon(radius)
        result = saddle_cell_fragments(jnp.asarray(vertices), 6, form, 1e-12)
        assert result.valid, result
        direction = np.asarray(form.direction)
        core = (direction[:, 0] > 0) & (np.roll(direction[:, 0], -1) > 0)
        assert np.count_nonzero(core) == 1
        area = float(np.asarray(result.area)[core][0])
        reference = _exact_core_area(vertices)
        full_area = 3 * np.sqrt(3) * radius**2 / 2
        errors.append(abs(area - reference) / full_area)
        np.testing.assert_allclose(np.sum(result.area), full_area, rtol=2e-14)
        assert np.all(np.asarray(result.area) > 0.0)
    order = float(np.polyfit(np.log(radii), np.log(errors), 1)[0])
    print(f"SADDLE_NORMAL_FORM radii={radii.tolist()} errors={errors} order={order}")
    assert order >= 2.0


def test_stationary_and_saddle_primitives_trace():
    policy = TopologyPolicy()
    field = QuadraticPointField(
        jnp.asarray((2.0, 0.1)), jnp.diag(jnp.asarray((-2.0, 3.0)))
    )
    seed = jnp.asarray((2.03, 0.08))
    eager = stationary_read(field, seed, 0.1, policy)
    traced = jax.jit(stationary_read)(field, seed, 0.1, policy)
    for left, right in zip(
        jax.tree.leaves(eager), jax.tree.leaves(traced), strict=True
    ):
        np.testing.assert_allclose(left, right, rtol=0, atol=1e-14)
    centres = jnp.stack((field.centre, field.centre + 0.01))
    batched = jax.jit(
        jax.vmap(
            lambda centre: stationary_read(
                replace(field, centre=centre), seed, 0.1, policy
            )
        )
    )(centres)
    np.testing.assert_allclose(batched.position, centres, atol=1e-14)
    hessian, third = _saddle_jet()
    form = jax.jit(saddle_normal_form)(jnp.zeros(2), hessian, third, 1e-10)
    fragments = jax.jit(saddle_cell_fragments)(
        jnp.asarray(_hexagon(0.1)), 6, form, 1e-12
    )
    assert fragments.valid
    refused = saddle_normal_form(jnp.zeros(2), jnp.eye(2), third, 1e-10)
    assert not refused.valid
    assert refused.reason == TopologyReason.SINGULAR_REPRESENTATION


@pytest.mark.parametrize("radius", (0.1, 0.3, 0.6))
def test_quadratic_fragments_circle_area(radius):
    from nova.equilibrium.topology import quadratic_cell_fragments

    vertices = jnp.asarray(((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)))
    coefficient = jnp.asarray((radius**2, 0.0, 0.0, -1.0, 0.0, -1.0))
    result = jax.jit(quadratic_cell_fragments)(vertices, 4, coefficient)
    assert result.valid
    assert result.required == 1
    np.testing.assert_allclose(result.area.sum(), np.pi * radius**2, rtol=2e-12)


def test_quadratic_fragments_saddle_touch_does_not_join():
    from nova.equilibrium.topology import quadratic_cell_fragments

    vertices = jnp.asarray(((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)))
    coefficient = jnp.asarray((0.0, 0.0, 0.0, 1.0, 0.0, -1.0))
    result = jax.jit(quadratic_cell_fragments)(vertices, 4, coefficient)
    assert result.valid
    assert result.required == 2
    np.testing.assert_allclose(result.area, (1.0, 1.0), rtol=2e-12)
    refused = quadratic_cell_fragments(vertices, 4, coefficient, capacity=1)
    assert not refused.valid
    assert refused.reason == TopologyReason.CAPACITY
    assert np.isnan(refused.area).all()


def test_traced_read_limited_quadratic():
    from nova.equilibrium.topology import TopologyConvention, TopologyGeometry, read

    vertices = _hexagon(0.5) + np.asarray((2.0, 0.0))
    geometry = TopologyGeometry.from_cells((vertices,), vertices[None], (vertices,))
    field = QuadraticPointField(jnp.asarray((2.0, 0.0)), -2 * jnp.eye(2))
    result = jax.jit(read)(
        field, geometry, TopologyConvention.from_cocos(17, 1.0), TopologyPolicy()
    )
    assert result.valid, result.reason
    assert result.qualified
    assert result.boundary_class == 0
    assert not np.any(result.x_point_valid)
    np.testing.assert_allclose(result.axis, (2.0, 0.0), atol=1e-13)
    np.testing.assert_allclose(result.membership, np.pi / (2 * np.sqrt(3)), rtol=1e-10)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ClosedFormPointField:
    """Independent closed-form total field for read-algorithm qualification."""

    coefficient: jax.Array
    radius: jax.Array
    scale: jax.Array
    source_parameter: jax.Array
    shift: jax.Array
    kind: str = dataclass_field(metadata={"static": True})

    def value(self, point):
        x, y = (point - self.shift) / self.radius
        if self.kind == "limited":
            axis, pressure, vertical = self.coefficient
            return (
                2
                * jnp.pi
                * (
                    axis
                    - 0.5 * pressure * (point[0] ** 2 - self.radius**2) ** 2
                    - vertical * (point[1] - self.shift[1]) ** 2
                )
            )
        log_x = jnp.log(x)
        x2, y2 = x * x, y * y
        x4, y4 = x2 * x2, y2 * y2
        x6, y6 = x4 * x2, y4 * y2
        basis = jnp.stack(
            (
                jnp.ones_like(x),
                x2,
                y2 - x2 * log_x,
                x4 - 4 * x2 * y2,
                2 * y4 - 9 * y2 * x2 + 3 * x4 * log_x - 12 * x2 * y2 * log_x,
                x6 - 12 * x4 * y2 + 8 * x2 * y4,
                8 * y6
                - 140 * y4 * x2
                + 75 * y2 * x4
                - 15 * x6 * log_x
                + 180 * x4 * y2 * log_x
                - 120 * x2 * y4 * log_x,
                y,
                y * x2,
                y**3 - 3 * y * x2 * log_x,
                3 * y * x4 - 4 * y**3 * x2,
                8 * y**5 - 45 * y * x4 - 80 * y**3 * x2 * log_x + 60 * y * x4 * log_x,
            )
        )
        particular = x4 / 8 + self.source_parameter * (0.5 * x2 * log_x - x4 / 8)
        return 2 * jnp.pi * self.scale * (particular + basis @ self.coefficient)

    def evaluate(self, point):
        return FieldJet(
            self.value(point),
            jax.grad(self.value)(point),
            jax.hessian(self.value)(point),
        )


def _analytic_inputs(kind):
    from nova.equilibrium.analytic_single_null import cerfon_freidberg_single_null
    from scripts.analytic_oracle_fixtures.measure import limiter_contour, offset_wall
    from tests.rotating_equilibrium_references import reference_cases

    if kind == "limited":
        oracle = reference_cases()["weak-rotation-reactor"].static_limit()
        field = ClosedFormPointField(
            jnp.asarray(
                (
                    oracle.axis_flux,
                    oracle.pressure_coefficient,
                    oracle.field_coefficient,
                )
            ),
            jnp.asarray(oracle.major_radius),
            jnp.asarray(1.0),
            jnp.asarray(0.0),
            jnp.zeros(2),
            kind,
        )
        wall = limiter_contour(oracle)
        axis = np.asarray((oracle.major_radius, 0.0))
        saddle = None
    else:
        oracle = cerfon_freidberg_single_null()
        field = ClosedFormPointField(
            jnp.asarray(oracle.coefficients),
            jnp.asarray(oracle.major_radius),
            jnp.asarray(oracle.flux_scale_per_radian_wb),
            jnp.asarray(oracle.source_parameter),
            jnp.zeros(2),
            kind,
        )
        wall = offset_wall(
            oracle.separatrix(1441), clearance=0.35 * oracle.minor_radius
        )
        axis, saddle = oracle.magnetic_axis, oracle.x_point
    return oracle, field, wall, axis, saddle


def _realised_hex_geometry(wall, target):
    from shapely.geometry import Polygon
    from nova.equilibrium.topology import TopologyGeometry

    vessel = Polygon(wall)
    origin = np.asarray(vessel.centroid.coords[0])
    bounds = np.asarray(vessel.bounds)

    def generate(radius, return_polygons=False):
        lower = np.floor((bounds[:2] - origin) / (1.5 * radius)).astype(int) - 3
        upper = np.ceil((bounds[2:] - origin) / (1.5 * radius)).astype(int) + 3
        cells, sampling = [], []
        offset = _hexagon(radius)
        for radial in range(lower[0], upper[0] + 1):
            for vertical in range(2 * lower[1], 2 * upper[1] + 1):
                centre = origin + radius * np.asarray(
                    (
                        1.5 * (radial + 0.137),
                        np.sqrt(3) * (vertical + 0.5 * (radial % 2) + 0.219),
                    )
                )
                polygon = Polygon(centre + offset).intersection(vessel)
                if polygon.is_empty or polygon.area < 1e-14 * radius**2:
                    continue
                if polygon.geom_type != "Polygon":
                    raise AssertionError(
                        "carrier cell has disconnected vessel intersection"
                    )
                cells.append(polygon)
                sampling.append(centre + offset)
        if not return_polygons:
            return len(cells)
        polygons = []
        for polygon in cells:
            points = np.asarray(polygon.exterior.coords[:-1])
            signed = np.sum(
                points[:, 0] * np.roll(points[:, 1], -1)
                - points[:, 1] * np.roll(points[:, 0], -1)
            )
            polygons.append(points if signed > 0 else points[::-1])
        return TopologyGeometry.from_cells(
            tuple(polygons), np.asarray(sampling), (wall,)
        )

    estimate = np.sqrt(vessel.area / target / (3 * np.sqrt(3) / 2))
    low, high = 0.5 * estimate, 2 * estimate
    for _ in range(60):
        radius = 0.5 * (low + high)
        count = generate(radius)
        if count == target:
            return generate(radius, True)
        if count > target:
            low = radius
        else:
            high = radius
    raise AssertionError(f"could not realise {target} cells; reached {count}")


@pytest.mark.parametrize("kind", ("limited", "diverted"))
def test_closed_form_carrier_read(kind):
    import time
    from nova.equilibrium.topology import TopologyConvention, read

    oracle, field, wall, axis, saddle = _analytic_inputs(kind)
    geometry = _realised_hex_geometry(wall, 132)
    started = time.perf_counter()
    result = jax.jit(read)(
        field, geometry, TopologyConvention.from_cocos(17, 1.0), TopologyPolicy()
    )
    jax.block_until_ready(result)
    pitch = float(np.sqrt(np.median(np.asarray(geometry.full_area))))
    print(
        f"CLOSED_FORM_READ kind={kind} cells={len(geometry.centre)} "
        f"valid={bool(result.valid)} qualified={bool(result.qualified)} "
        f"reason={int(result.reason)} axis={np.asarray(result.axis).tolist()} "
        f"membership_min={float(jnp.nanmin(result.membership))} "
        f"membership_max={float(jnp.nanmax(result.membership))} "
        f"compile_and_run_seconds={time.perf_counter() - started}"
    )
    assert result.valid, result.reason
    assert result.qualified
    np.testing.assert_allclose(result.axis, axis, atol=pitch)
    assert int(result.boundary_class) == int(saddle is not None)
    if saddle is not None:
        points = np.asarray(result.x_points)[np.asarray(result.x_point_valid)]
        assert len(points) == 1
        assert np.linalg.norm(points[0] - saddle) <= pitch
    assert np.all(np.asarray(result.membership) >= 0.0)
    assert np.all(np.asarray(result.membership) <= 1.0 + 1e-12)


@pytest.mark.parametrize("kind", ("limited", "diverted"))
def test_topology_shift_fails_position_clause(kind):
    import os
    from nova.equilibrium.topology import TopologyConvention, read

    _oracle, field, wall, axis, _saddle = _analytic_inputs(kind)
    geometry = _realised_hex_geometry(wall, 132)
    pitch = float(np.median(np.asarray(geometry.pitch)))
    shifted = replace(field, shift=jnp.asarray((0.0, 3 * pitch)))
    convention = TopologyConvention.from_cocos(17, 1.0)
    evaluate = jax.jit(read)
    head = (
        shifted if os.environ.get("NOVA_TOPOLOGY_POSITION_MUTATION") == "1" else field
    )
    result = evaluate(head, geometry, convention, TopologyPolicy())
    assert np.linalg.norm(np.asarray(result.axis) - axis) <= pitch
    shifted_result = evaluate(shifted, geometry, convention, TopologyPolicy())
    displacement = np.linalg.norm(np.asarray(shifted_result.axis) - axis)
    assert displacement >= 3 * pitch * (1 - 1e-12)
    assert displacement > pitch
    assert result.qualified


@kernel_compile_expected_failure
@pytest.mark.timeout(660)
def test_biot_moment_point_derivatives(kernel_compile_budget):
    from nova.equilibrium.clip_quadrature import ClippedCurrentMoments
    from nova.equilibrium.topology import BiotMomentCoupling

    coupling = BiotMomentCoupling.from_polygons(
        (_hexagon(0.08) + np.asarray((1.5, 0.0)),)
    )
    moments = ClippedCurrentMoments(
        jnp.asarray((1000.0,)), jnp.asarray((0.2,)), jnp.asarray((-0.3,))
    )
    point = jnp.asarray((2.0, 0.1))
    evaluate = jax.jit(lambda target, kernel, current: kernel.evaluate(target, current))
    jet = evaluate(point, coupling, moments)
    step = 1e-5
    directions = jnp.eye(2)
    upper = [
        evaluate(point + step * direction, coupling, moments)
        for direction in directions
    ]
    lower = [
        evaluate(point - step * direction, coupling, moments)
        for direction in directions
    ]
    gradient = np.asarray(
        [
            (up.value - down.value) / (2 * step)
            for up, down in zip(upper, lower, strict=True)
        ]
    )
    hessian = np.stack(
        [
            (up.gradient - down.gradient) / (2 * step)
            for up, down in zip(upper, lower, strict=True)
        ],
        axis=1,
    )
    np.testing.assert_allclose(jet.gradient, gradient, rtol=5e-7, atol=1e-11)
    np.testing.assert_allclose(jet.hessian, hessian, rtol=5e-7, atol=1e-11)
    np.testing.assert_allclose(jet.hessian, jet.hessian.T, rtol=5e-7, atol=1e-11)


def _reference_cell_fractions(kind, oracle, geometry):
    """Integrate oracle vertical roots independently of the production clip."""
    from shapely.geometry import Point, Polygon

    vertices = np.asarray(geometry.vertices)
    counts = np.asarray(geometry.vertex_count)
    if kind == "limited":
        angle = np.linspace(0, 2 * np.pi, 4097)
        extent = np.sqrt(2 * oracle.axis_flux / oracle.pressure_coefficient)
        boundary = np.column_stack(
            (
                np.sqrt(oracle.major_radius**2 + extent * np.cos(angle)),
                np.sqrt(oracle.axis_flux / oracle.field_coefficient) * np.sin(angle),
            )
        )
    else:
        boundary = oracle.separatrix(4097)
    region = Polygon(boundary)
    node = np.polynomial.chebyshev.chebpts1(7)
    inverse = np.linalg.inv(np.polynomial.polynomial.polyvander(node, 6))
    z_origin = 0.5 * (np.min(boundary[:, 1]) + np.max(boundary[:, 1]))
    z_scale = np.ptp(boundary[:, 1])
    fractions, uncertainties = [], []
    for cell, count in zip(vertices, counts, strict=True):
        polygon = cell[:count]
        shape = Polygon(polygon)
        if shape.distance(region.boundary) > 1e-3 * np.sqrt(shape.area):
            fractions.append(float(region.contains(shape.representative_point())))
            uncertainties.append(0.0)
            continue

        def length(radius):
            heights = []
            for first, second in zip(
                polygon, np.roll(polygon, -1, axis=0), strict=True
            ):
                delta = second - first
                if delta[0] == 0:
                    continue
                fraction = (radius - first[0]) / delta[0]
                if -1e-12 <= fraction <= 1 + 1e-12:
                    heights.append(first[1] + fraction * delta[1])
            if len(heights) < 2:
                return 0.0
            lower, upper = min(heights), max(heights)
            if kind == "limited":
                remaining = oracle.flux(radius, 0.0) / oracle.field_coefficient
                half = np.sqrt(max(remaining, 0.0))
                return max(0.0, min(upper, half) - max(lower, -half))
            samples = np.column_stack((np.full(7, radius), z_origin + z_scale * node))
            coefficient = inverse @ oracle.flux(samples)
            keep = np.flatnonzero(
                np.abs(coefficient) > np.max(np.abs(coefficient)) * 1e-13
            )
            roots = np.polynomial.polynomial.polyroots(coefficient[: keep[-1] + 1])
            roots = z_origin + z_scale * roots[np.abs(roots.imag) < 1e-8].real
            cuts = np.unique(
                np.r_[lower, roots[(roots > lower) & (roots < upper)], upper]
            )
            total = 0.0
            for first, last in zip(cuts[:-1], cuts[1:], strict=True):
                midpoint = (first + last) / 2
                if oracle.flux(np.asarray((radius, midpoint))) > 0 and region.covers(
                    Point(radius, midpoint)
                ):
                    total += last - first
            return total

        breaks = np.unique(polygon[:, 0])
        value = error = 0.0
        for first, last in zip(breaks[:-1], breaks[1:], strict=True):
            integral, estimate = quad(
                length, first, last, epsabs=shape.area * 1e-10, epsrel=1e-9, limit=100
            )
            value += integral
            error += estimate
        fractions.append(value / shape.area)
        uncertainties.append(error / shape.area)
    return np.asarray(fractions), np.asarray(uncertainties)


def _render_read_panel(
    directory,
    kind,
    field,
    geometry,
    result,
    wall,
    axis,
    saddle,
    worst_cell=None,
    suffix="",
):
    from pathlib import Path
    import matplotlib.pyplot as plt
    from shapely import STRtree, points
    from shapely.geometry import Polygon
    from nova.media import poloidal
    from nova.media.ink import DEFAULT_INK, poloidal_axes

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    radial = np.linspace(wall[:, 0].min(), wall[:, 0].max(), 180)
    vertical = np.linspace(wall[:, 1].min(), wall[:, 1].max(), 220)
    rr, zz = np.meshgrid(radial, vertical)
    locations = np.column_stack((rr.ravel(), zz.ravel()))
    polygons = [
        Polygon(np.asarray(poly)[: int(count)])
        for poly, count in zip(geometry.vertices, geometry.vertex_count, strict=True)
    ]
    matches = STRtree(polygons).query(points(locations), predicate="within")
    sampled = np.full(len(locations), np.nan)
    index, cell = matches
    local = (locations[index] - np.asarray(geometry.centre)[cell]) / np.asarray(
        geometry.pitch
    )[cell, None]
    x, y = local.T
    basis = np.column_stack((np.ones_like(x), x, y, x * x, x * y, y * y))
    sampled[index] = np.sum(basis * np.asarray(result.field_coefficients)[cell], axis=1)
    reference = np.asarray(jax.vmap(field.value)(jnp.asarray(locations))).reshape(
        rr.shape
    )
    levels = np.linspace(float(result.boundary_flux), float(result.axis_flux), 10)[:-1]
    figure, axes = plt.subplots(figsize=(14, 9), dpi=100)
    poloidal_axes(axes)
    reference_style = replace(
        DEFAULT_INK,
        axis_color="#333333",
        xpoint_color="#333333",
        axis_markersize=12,
        xpoint_markersize=13,
    )
    read_style = replace(
        DEFAULT_INK,
        axis_color=DEFAULT_INK.flux_color,
        xpoint_color=DEFAULT_INK.flux_color,
        axis_markersize=8,
        xpoint_markersize=9,
    )
    poloidal.draw_flux_contours(
        axes,
        radial,
        vertical,
        reference,
        levels,
        wall=(wall,),
        color="#444444",
        linewidth=2.6,
    )
    physical_radius = float(result.normal_form_radius)
    poloidal.draw_flux_contours(
        axes,
        radial,
        vertical,
        sampled.reshape(rr.shape),
        levels[1:] if physical_radius > 0 else levels,
        wall=(wall,),
        color=DEFAULT_INK.flux_color,
        linewidth=2.6,
    )
    if physical_radius > 0:
        from matplotlib.patches import Circle

        represented = np.asarray(result.normal_form_cells)
        boundary_values = sampled.copy()
        boundary_values[index[represented[cell]]] = np.nan
        poloidal.draw_flux_contours(
            axes,
            radial,
            vertical,
            boundary_values.reshape(rr.shape),
            levels[:1],
            wall=(wall,),
            color=DEFAULT_INK.flux_color,
            linewidth=2.6,
        )
        form = result.saddle_form
        origin = np.asarray(form.position)
        vertices = np.asarray(geometry.vertices)
        live = (
            np.arange(vertices.shape[1])[None, :]
            < np.asarray(geometry.vertex_count)[:, None]
        )
        reach = 1.1 * np.max(
            np.linalg.norm(vertices[represented[:, None] & live] - origin, axis=-1)
        )
        tree = STRtree(polygons)
        for direction, curvature, cubic in zip(
            form.direction, form.curvature, form.cubic, strict=True
        ):
            parameter = np.linspace(0, reach, 1025)[:, None]
            curve = (
                origin
                + parameter * np.asarray(direction)
                + parameter**2 * np.asarray(curvature)
                + parameter**3 * np.asarray(cubic)
            )
            hit_point, hit_cell = tree.query(points(curve), predicate="within")
            active = np.zeros(len(curve), dtype=bool)
            active[hit_point[represented[hit_cell]]] = True
            curve[~active] = np.nan
            axes.plot(
                *curve.T, color=DEFAULT_INK.flux_color, linewidth=3.0, linestyle="--"
            )
        axes.add_patch(
            Circle(
                origin,
                physical_radius,
                fill=False,
                edgecolor="#666666",
                linewidth=1.2,
                linestyle=":",
            )
        )
        figure.text(
            0.06,
            0.73,
            f"normal-form radius\n{physical_radius * 1000:.2f} mm",
            transform=figure.transFigure,
            color="#666666",
            fontsize=20,
            va="top",
        )
    poloidal.draw_wall(axes, units=(wall,))
    reference_x = np.empty((0, 2)) if saddle is None else np.asarray(saddle)[None]
    poloidal.draw_nulls(axes, axis, reference_x, style=reference_style, contain=(wall,))
    poloidal.draw_nulls(
        axes,
        np.asarray(result.axis),
        np.asarray(result.x_points)[np.asarray(result.x_point_valid)],
        style=read_style,
        contain=(wall,),
    )
    figure.text(
        0.06,
        0.89,
        "analytic field /\nlarger nulls",
        transform=figure.transFigure,
        color="#333333",
        fontsize=20,
        va="top",
    )
    figure.text(
        0.06,
        0.81,
        "cell field /\nread nulls",
        transform=figure.transFigure,
        color=DEFAULT_INK.flux_color,
        fontsize=20,
        va="top",
    )
    if worst_cell is not None:
        polygon = np.asarray(geometry.vertices)[
            worst_cell, : int(geometry.vertex_count[worst_cell])
        ]
        polygon = np.vstack((polygon, polygon[0]))
        axes.plot(*polygon.T, color="#555555", linewidth=3.0, linestyle="--")
        axes.annotate(
            f"measured cell {worst_cell}",
            xy=np.asarray(geometry.centre)[worst_cell],
            xytext=(0.06, 0.22),
            textcoords="figure fraction",
            fontsize=20,
            color="#555555",
            arrowprops={"arrowstyle": "-", "color": "#555555"},
        )
    for extension in ("png", "svg"):
        figure.savefig(
            directory / f"{kind}-poloidal{suffix}.{extension}",
            dpi=100,
            facecolor="white",
        )
    plt.close(figure)


@pytest.mark.parametrize("kind", ("limited", "diverted"))
def test_closed_form_resolution_probe(kind):
    import os
    from pathlib import Path
    import resource
    import time
    from nova.equilibrium.topology import TopologyConvention, read

    oracle, field, wall, axis, saddle = _analytic_inputs(kind)
    convention = TopologyConvention.from_cocos(17, 1.0)
    evaluate = jax.jit(read)
    rows = []
    for count in (132, 300, 550, 1074, 2616):
        geometry = _realised_hex_geometry(wall, count)
        started = time.perf_counter()
        executable = evaluate.lower(
            field, geometry, convention, TopologyPolicy()
        ).compile()
        compile_seconds = time.perf_counter() - started
        result = executable(field, geometry, convention, TopologyPolicy())
        jax.block_until_ready(result)
        reference, uncertainty = _reference_cell_fractions(kind, oracle, geometry)
        errors = np.abs(np.asarray(result.membership) - reference)
        distance = (
            np.full(count, np.inf)
            if saddle is None
            else np.linalg.norm(np.asarray(geometry.centre) - saddle, axis=1)
        )
        neighbourhood = distance < 2 * np.asarray(geometry.pitch)
        worst = int(np.argmax(errors))
        row = {
            "worst_cell": worst,
            "worst_cell_area": float(geometry.full_area[worst]),
            "median_cell_area": float(np.median(geometry.full_area)),
            "worst_coefficient": np.asarray(result.field_coefficients)[worst].tolist(),
            "worst_vertices": np.asarray(geometry.vertices)[
                worst, : int(geometry.vertex_count[worst])
            ].tolist(),
            "worst_centre": np.asarray(geometry.centre)[worst].tolist(),
            "worst_read_fraction": float(result.membership[worst]),
            "worst_reference_fraction": float(reference[worst]),
            "boundary_flux": float(result.boundary_flux),
            "case": kind,
            "cells": count,
            "valid": bool(result.valid),
            "qualified": bool(result.qualified),
            "reason": int(result.reason),
            "pitch": float(np.median(np.asarray(geometry.pitch))),
            "max_fraction_error": float(np.max(errors)),
            "smooth_fraction_error": float(np.max(errors[~neighbourhood])),
            "saddle_fraction_error": float(np.max(errors[neighbourhood]))
            if np.any(neighbourhood)
            else None,
            "reference_quadrature_estimate": float(np.max(uncertainty)),
            "compile_seconds": compile_seconds,
            "peak_host_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "axis_error_m": float(np.linalg.norm(np.asarray(result.axis) - axis)),
            "backend": jax.default_backend(),
            "scope": (
                "closed-form evaluator and area-fraction difference; "
                "not kernel-backed symmetric-difference acceptance"
            ),
        }
        rows.append(row)
        print("TOPOLOGY_PROBE " + json.dumps(row), flush=True)
        root = os.environ.get("NOVA_TOPOLOGY_EVIDENCE_DIR")
        if root:
            directory = Path(root)
            directory.mkdir(parents=True, exist_ok=True)
            (directory / f"{kind}-rows.json").write_text(
                json.dumps(rows, indent=2) + "\n"
            )
        if count == 550 and os.environ.get("NOVA_TOPOLOGY_FIGURE_DIR"):
            _render_read_panel(
                os.environ["NOVA_TOPOLOGY_FIGURE_DIR"],
                kind,
                field,
                geometry,
                result,
                wall,
                axis,
                saddle,
            )
        assert result.valid, row
        assert result.qualified, row
        assert row["axis_error_m"] <= row["pitch"]
        assert int(result.boundary_class) == int(saddle is not None)


def _limited_wall_contact(oracle, wall):
    """Independent stationary maxima of the analytic quartic on wall segments."""
    from scipy.optimize import minimize_scalar

    events = []
    for index, first in enumerate(wall):
        direction = wall[(index + 1) % len(wall)] - first

        def negative_flux(parameter):
            return -2 * np.pi * oracle.flux(*(first + parameter * direction))

        optimum = minimize_scalar(
            negative_flux,
            bounds=(0.0, 1.0),
            method="bounded",
            options={"xatol": 1e-15},
        )
        for parameter in (0.0, 1.0, float(optimum.x)):
            events.append(
                (
                    -negative_flux(parameter),
                    index,
                    parameter,
                    first + parameter * direction,
                )
            )
    return max(events, key=lambda event: event[0])


def _render_limited_contact(directory, oracle, wall, result, contact, worst):
    """Show the analytic zero contour and the wall-selected level at contact."""
    from pathlib import Path
    import matplotlib.pyplot as plt
    from nova.media import poloidal
    from nova.media.ink import DEFAULT_INK, poloidal_axes

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    figure, panels = plt.subplots(1, 2, figsize=(14, 8), dpi=100)
    boundary = np.asarray(result.boundary)
    for index, axes in enumerate(panels):
        poloidal_axes(axes)
        if index == 0:
            lower, upper = wall.min(axis=0), wall.max(axis=0)
        else:
            radial_extent = abs(boundary[0] - contact[0])
            vertical_extent = abs(boundary[1] - contact[1])
            lower = np.minimum(boundary, contact) - (
                2 * radial_extent,
                0.3 * vertical_extent,
            )
            upper = np.maximum(boundary, contact) + (
                2 * radial_extent,
                0.3 * vertical_extent,
            )
            axes.set_aspect("auto")
        radial = np.linspace(lower[0], upper[0], 450)
        vertical = np.linspace(lower[1], upper[1], 450)
        rr, zz = np.meshgrid(radial, vertical)
        flux = 2 * np.pi * oracle.flux(rr, zz)
        for level, color, linestyle in (
            (0.0, "#333333", "--"),
            (float(result.boundary_flux), DEFAULT_INK.flux_color, "-"),
        ):
            contours = poloidal.draw_flux_contours(
                axes,
                radial,
                vertical,
                flux,
                np.asarray((level,)),
                color=color,
                linewidth=2.6,
            )
            if contours is not None:
                contours.set_linestyle(linestyle)
        poloidal.draw_wall(axes, units=(wall,))
        axes.plot(*contact, marker="o", markersize=9, fillstyle="none", color="#333333")
        axes.plot(*boundary, marker="+", markersize=13, color=DEFAULT_INK.flux_color)
        if index == 0:
            poloidal.draw_nulls(
                axes,
                np.asarray((oracle.major_radius, 0.0)),
                np.empty((0, 2)),
                style=replace(DEFAULT_INK, axis_color="#333333", axis_markersize=12),
            )
            poloidal.draw_nulls(
                axes,
                np.asarray(result.axis),
                np.empty((0, 2)),
                style=replace(
                    DEFAULT_INK, axis_color=DEFAULT_INK.flux_color, axis_markersize=8
                ),
            )
            polygon = np.vstack((worst, worst[0]))
            axes.plot(*polygon.T, color="#777777", linewidth=1.2)
        axes.set_xlim(lower[0], upper[0])
        axes.set_ylim(lower[1], upper[1])
    panels[0].text(
        0.02,
        1.10,
        "zero-flux region / larger axis",
        transform=panels[0].transAxes,
        color="#333333",
        fontsize=20,
        va="top",
    )
    panels[0].text(
        0.02,
        1.04,
        "wall-selected level / read axis",
        transform=panels[0].transAxes,
        color=DEFAULT_INK.flux_color,
        fontsize=20,
        va="top",
    )
    panels[1].text(
        0.02,
        1.04,
        "contact detail (radial scale expanded)",
        transform=panels[1].transAxes,
        color="#333333",
        fontsize=20,
        va="top",
    )
    for extension in ("png", "svg"):
        figure.savefig(
            directory / f"limited-contact-diagnostic.{extension}",
            dpi=100,
            facecolor="white",
        )
    plt.close(figure)


def test_limited_boundary_instrument():
    """Expose wall/region offsets independently of cell reconstruction order."""
    import os
    from pathlib import Path
    from nova.equilibrium.topology import TopologyConvention, read

    oracle, field, wall, _, _ = _analytic_inputs("limited")
    analytic_contact = np.asarray((oracle.boundary_midplane_radii()[1], 0.0))
    analytic_vertex = int(np.argmin(np.linalg.norm(wall - analytic_contact, axis=1)))
    independent_level, segment, parameter, independent_point = _limited_wall_contact(
        oracle, wall
    )
    convention = TopologyConvention.from_cocos(17, 1.0)
    evaluate = jax.jit(read)
    rows = []
    for count in (132, 300, 550):
        geometry = _realised_hex_geometry(wall, count)
        result = evaluate(field, geometry, convention, TopologyPolicy())
        jax.block_until_ready(result)
        reference, _ = _reference_cell_fractions("limited", oracle, geometry)
        membership = np.asarray(result.membership)
        vertices, counts = (
            np.asarray(geometry.vertices),
            np.asarray(geometry.vertex_count),
        )
        polygons = [cell[:size] for cell, size in zip(vertices, counts, strict=True)]
        # Bounding-box extrema bound the separable quartic on the whole cell.
        minimum, maximum = [], []
        for polygon in polygons:
            lower, upper = polygon.min(axis=0), polygon.max(axis=0)
            radial = np.asarray((lower[0] ** 2, upper[0] ** 2)) - oracle.major_radius**2
            radial_min = 0.0 if radial[0] <= 0 <= radial[1] else np.min(radial**2)
            vertical_min = (
                0.0 if lower[1] <= 0 <= upper[1] else min(lower[1] ** 2, upper[1] ** 2)
            )
            minimum.append(
                2
                * np.pi
                * (
                    oracle.axis_flux
                    - 0.5 * oracle.pressure_coefficient * max(radial**2)
                    - oracle.field_coefficient * max(lower[1] ** 2, upper[1] ** 2)
                )
            )
            maximum.append(
                2
                * np.pi
                * (
                    oracle.axis_flux
                    - 0.5 * oracle.pressure_coefficient * radial_min
                    - oracle.field_coefficient * vertical_min
                )
            )
        inside = np.flatnonzero(
            (np.asarray(minimum) > max(float(result.boundary_flux), 0.0))
            & (membership == 1.0)
            & (reference == 1.0)
        )
        outside = np.flatnonzero(
            (np.asarray(maximum) < min(float(result.boundary_flux), 0.0))
            & (membership == 0.0)
            & (reference == 0.0)
        )
        assert inside.size and outside.size, (inside, outside)
        worst = int(np.argmax(np.abs(membership - reference)))
        first, last = wall, np.roll(wall, -1, axis=0)
        directions = last - first
        projected = np.clip(
            np.sum((np.asarray(result.boundary) - first) * directions, axis=1)
            / np.sum(directions**2, axis=1),
            0.0,
            1.0,
        )
        read_segment = int(
            np.argmin(
                np.linalg.norm(
                    first
                    + projected[:, None] * directions
                    - np.asarray(result.boundary),
                    axis=1,
                )
            )
        )
        row = {
            "cells": len(geometry.centre),
            "requested_cells": count,
            "previous_realised_cells": count,
            "max_fraction_error": float(np.max(np.abs(membership - reference))),
            "read_axis_m": np.asarray(result.axis).tolist(),
            "pitch_m": float(np.median(geometry.pitch)),
            "read_level_wb": float(result.boundary_flux),
            "analytic_boundary_flux_wb": 0.0,
            "read_wall_unit": 0,
            "read_wall_segment": read_segment,
            "read_contact_m": np.asarray(result.boundary).tolist(),
            "analytic_wall_unit": 0,
            "analytic_wall_segment": int(segment),
            "nearest_wall_vertex": analytic_vertex,
            "analytic_contact_m": analytic_contact.tolist(),
            "wall_vertex_m": wall[analytic_vertex].tolist(),
            "wall_vertex_offset_m": float(
                np.linalg.norm(wall[analytic_vertex] - analytic_contact)
            ),
            "contact_offset_m": float(
                np.linalg.norm(np.asarray(result.boundary) - analytic_contact)
            ),
            "independent_wall_max_wb": float(independent_level),
            "independent_wall_segment": int(segment),
            "independent_segment_parameter": float(parameter),
            "independent_contact_m": independent_point.tolist(),
            "worst_cell": worst,
            "worst_cell_centre_m": np.asarray(geometry.centre)[worst].tolist(),
            "worst_cell_minus_analytic_contact_m": (
                np.asarray(geometry.centre)[worst] - analytic_contact
            ).tolist(),
            "worst_read_fraction": float(membership[worst]),
            "worst_reference_fraction": float(reference[worst]),
            "inside_control": {
                "cell": int(inside[0]),
                "read": float(membership[inside[0]]),
                "reference": float(reference[inside[0]]),
                "flux_lower_bound_wb": minimum[inside[0]],
            },
            "outside_control": {
                "cell": int(outside[0]),
                "read": float(membership[outside[0]]),
                "reference": float(reference[outside[0]]),
                "flux_upper_bound_wb": maximum[outside[0]],
            },
        }
        rows.append(row)
        print("LIMITED_BOUNDARY_INSTRUMENT " + json.dumps(row), flush=True)
        root = os.environ.get("NOVA_TOPOLOGY_EVIDENCE_DIR")
        if root:
            Path(root).mkdir(parents=True, exist_ok=True)
            (Path(root) / "limited-boundary-instrument.json").write_text(
                json.dumps(rows, indent=2) + "\n"
            )
        np.testing.assert_allclose(
            result.boundary_flux, independent_level, rtol=1e-9, atol=1e-12
        )
        assert result.valid and result.qualified
        if count == 550 and os.environ.get("NOVA_TOPOLOGY_FIGURE_DIR"):
            _render_limited_contact(
                os.environ["NOVA_TOPOLOGY_FIGURE_DIR"],
                oracle,
                wall,
                result,
                analytic_contact,
                polygons[worst],
            )
    assert max(abs(row["read_level_wb"]) for row in rows) <= 1e-12
    assert max(row["contact_offset_m"] for row in rows) <= 1e-12
    # Fractions remain diagnostic; convergence uses regular-cell area units.


def _normal_form_slice(cell, result):
    """Slice the read's cubic rays independently of its area integration."""
    from shapely.geometry import Point, Polygon

    form = result.saddle_form
    origin = np.asarray(form.position)
    direction, curvature, cubic = map(
        np.asarray, (form.direction, form.curvature, form.cubic)
    )
    ray_end, boundary_parameter = [], []

    def cross(a, b):
        return a[0] * b[1] - a[1] * b[0]

    for d, c, b in zip(direction, curvature, cubic, strict=True):
        hits = []
        for edge_index, (first, last) in enumerate(
            zip(cell, np.roll(cell, -1, axis=0), strict=True)
        ):
            edge = last - first
            roots = np.polynomial.polynomial.polyroots(
                (
                    cross(origin - first, edge),
                    cross(d, edge),
                    cross(c, edge),
                    cross(b, edge),
                )
            )
            for root in roots:
                if abs(root.imag) > 1e-9 or root.real <= 1e-12:
                    continue
                t = float(root.real)
                point = origin + t * d + t**2 * c + t**3 * b
                fraction = (point - first) @ edge / (edge @ edge)
                if -1e-10 <= fraction <= 1 + 1e-10:
                    hits.append((t, edge_index + np.clip(fraction, 0, 1)))
        assert hits, "normal-form ray has no geometric exit"
        t, boundary = min(hits)
        ray_end.append(t)
        boundary_parameter.append(boundary)
    curves = []
    for index, t in enumerate(ray_end):
        parameter = np.linspace(0, t, 2049)[:, None]
        curves.append(
            origin
            + parameter * direction[index]
            + parameter**2 * curvature[index]
            + parameter**3 * cubic[index]
        )
    sectors = []
    for index in np.flatnonzero(np.asarray(form.positive)):
        following = (index + 1) % 4
        start, stop = boundary_parameter[index], boundary_parameter[following]
        if stop <= start:
            stop += len(cell)
        corners = [
            cell[k % len(cell)]
            for k in range(int(np.floor(start)) + 1, int(np.ceil(stop)))
        ]
        boundary = np.vstack(
            (curves[index], np.asarray(corners).reshape(-1, 2), curves[following][::-1])
        )
        sectors.append(Polygon(boundary))
    breaks = [origin[0]]
    for index, t in enumerate(ray_end):
        breaks.extend((curves[index][-1, 0],))
        for root in np.polynomial.polynomial.polyroots(
            (direction[index, 0], 2 * curvature[index, 0], 3 * cubic[index, 0])
        ):
            if abs(root.imag) < 1e-10 and 0 < root.real < t:
                u = root.real
                breaks.append(
                    origin[0]
                    + u * direction[index, 0]
                    + u**2 * curvature[index, 0]
                    + u**3 * cubic[index, 0]
                )

    def intervals(radius, lower, upper, selected):
        cuts = [lower, upper]
        for index, maximum in enumerate(ray_end):
            roots = np.polynomial.polynomial.polyroots(
                (
                    origin[0] - radius,
                    direction[index, 0],
                    curvature[index, 0],
                    cubic[index, 0],
                )
            )
            for root in roots:
                if abs(root.imag) < 1e-9 and -1e-12 <= root.real <= maximum + 1e-12:
                    t = float(root.real)
                    z = (
                        origin[1]
                        + t * direction[index, 1]
                        + t**2 * curvature[index, 1]
                        + t**3 * cubic[index, 1]
                    )
                    if lower < z < upper:
                        cuts.append(z)
        cuts = np.unique(cuts)
        return [
            (a, b)
            for a, b in zip(cuts[:-1], cuts[1:], strict=True)
            if any(
                live and sector.covers(Point(radius, (a + b) / 2))
                for live, sector in zip(selected[:2], sectors, strict=True)
            )
        ]

    return intervals, breaks


def _symmetric_difference_measure(kind, oracle, geometry, result, tolerance=2e-10):
    """Integrate interval symmetric differences in regular-cell area units.

    Oracle roots and core containment are independent of the reconstructed
    field. The reconstruction is sliced from its coefficients or cubic rays;
    production areas never enter the quadrature or its error denominator.
    """
    from scipy.integrate import quad_vec
    from shapely.geometry import Point, Polygon
    from nova.equilibrium.topology import quadratic_cell_fragments

    vertices, counts, centres, pitches, areas = map(
        np.asarray,
        (
            geometry.vertices,
            geometry.vertex_count,
            geometry.centre,
            geometry.pitch,
            geometry.full_area,
        ),
    )
    coefficients = np.asarray(result.field_coefficients).copy()
    coefficients[:, 0] -= float(result.boundary_flux)
    selected = np.asarray(result.fragment_selected)
    membership = np.asarray(result.membership)
    fragment_area = np.asarray(result.fragment_area)
    diverted = bool(result.boundary_class)
    median_area = float(np.median(areas))
    reference_fraction, _ = _reference_cell_fractions(kind, oracle, geometry)
    if kind == "diverted":
        core = Polygon(oracle.separatrix(8193))
        lower_z, upper_z = core.bounds[1], core.bounds[3]
        origin_z, scale_z = (lower_z + upper_z) / 2, upper_z - lower_z
        nodes = np.polynomial.chebyshev.chebpts1(7)
        inverse = np.linalg.inv(np.polynomial.polynomial.polyvander(nodes, 6))
    else:
        core = None
    errors, estimates, reconstructed, analytic = [], [], [], []
    for index, size in enumerate(counts):
        cell = vertices[index, :size]
        fraction, reference = membership[index], reference_fraction[index]
        if fraction == reference and fraction in (0.0, 1.0):
            errors.append(0.0)
            estimates.append(0.0)
            reconstructed.append(fraction)
            analytic.append(reference)
            continue
        saddle_cell = diverted and bool(np.asarray(result.normal_form_cells)[index])
        if saddle_cell:
            origin = np.asarray(result.saddle_form.position)
            extent = 1.2 * np.max(np.linalg.norm(cell - origin, axis=1))
            box = origin + extent * np.asarray(
                ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0))
            )
            ray_intervals, ray_breaks = _normal_form_slice(box, result)
        else:
            ray_intervals, ray_breaks = None, []
        coefficient, centre, pitch = coefficients[index], centres[index], pitches[index]
        fragment = None
        if (
            not saddle_cell
            and np.count_nonzero(fragment_area[index] > 0) > 1
            and np.count_nonzero(selected[index]) == 1
        ):
            fragment = quadratic_cell_fragments(
                jnp.asarray((vertices[index] - centre) / pitch),
                jnp.asarray(size),
                jnp.asarray(coefficient),
                2,
            )
            fragment = jax.tree.map(np.asarray, fragment)

        def vertical(radius):
            crossing = []
            for first, last in zip(cell, np.roll(cell, -1, axis=0), strict=True):
                delta = last - first
                if delta[0] == 0:
                    continue
                t = (radius - first[0]) / delta[0]
                if -1e-12 <= t <= 1 + 1e-12:
                    crossing.append(first[1] + t * delta[1])
            if len(crossing) < 2:
                return np.zeros(3)
            lower, upper = min(crossing), max(crossing)
            if kind == "limited":
                half = np.sqrt(
                    max(float(oracle.flux(radius, 0)) / oracle.field_coefficient, 0)
                )
                oracle_intervals = (
                    [(max(lower, -half), min(upper, half))]
                    if min(upper, half) > max(lower, -half)
                    else []
                )
            else:
                values = oracle.flux(
                    np.column_stack((np.full(7, radius), origin_z + scale_z * nodes))
                )
                polynomial = inverse @ values
                keep = np.flatnonzero(
                    np.abs(polynomial) > np.max(np.abs(polynomial)) * 1e-13
                )
                roots = np.polynomial.polynomial.polyroots(polynomial[: keep[-1] + 1])
                roots = origin_z + scale_z * roots[np.abs(roots.imag) < 1e-8].real
                cuts = np.unique(
                    np.r_[lower, roots[(roots > lower) & (roots < upper)], upper]
                )
                oracle_intervals = [
                    (a, b)
                    for a, b in zip(cuts[:-1], cuts[1:], strict=True)
                    if oracle.flux(np.asarray((radius, (a + b) / 2))) > 0
                    and core.covers(Point(radius, (a + b) / 2))
                ]
            if not np.any(selected[index]):
                read_intervals = []
            elif saddle_cell:
                read_intervals = ray_intervals(radius, lower, upper, selected[index])
            else:
                x = (radius - centre[0]) / pitch
                polynomial = (
                    coefficient[0] + coefficient[1] * x + coefficient[3] * x * x,
                    coefficient[2] + coefficient[4] * x,
                    coefficient[5],
                )
                roots = np.polynomial.polynomial.polyroots(polynomial)
                roots = sorted(
                    [float(root.real) for root in roots if abs(root.imag) < 1e-10]
                )
                roots = roots + [np.inf] * (2 - len(roots))
                lo, hi = (lower - centre[1]) / pitch, (upper - centre[1]) / pitch
                cuts = np.r_[lo, np.sort(np.clip(roots, lo, hi)), hi]
                read_intervals = []
                strip = (
                    None
                    if fragment is None
                    else np.clip(
                        np.searchsorted(fragment.slice_breaks, x, side="right") - 1,
                        0,
                        fragment.slice_labels.shape[0] - 1,
                    )
                )
                for band, (a, b) in enumerate(zip(cuts[:-1], cuts[1:], strict=True)):
                    label = (
                        0
                        if fragment is None
                        else int(fragment.slice_labels[strip, band])
                    )
                    active = (
                        np.any(selected[index])
                        if fragment is None
                        else label >= 0 and selected[index, label]
                    )
                    if (
                        b > a
                        and active
                        and np.polynomial.polynomial.polyval((a + b) / 2, polynomial)
                        > 0
                    ):
                        read_intervals.append(
                            (centre[1] + pitch * a, centre[1] + pitch * b)
                        )
            read_length = sum(b - a for a, b in read_intervals)
            oracle_length = sum(b - a for a, b in oracle_intervals)
            overlap = sum(
                max(0.0, min(b, d) - max(a, c))
                for a, b in read_intervals
                for c, d in oracle_intervals
            )
            return np.asarray(
                (
                    max(0.0, read_length + oracle_length - 2 * overlap),
                    read_length,
                    oracle_length,
                )
            )

        # Split at analytic conic/edge events so a narrow cap cannot lie
        # entirely between the adaptive quadrature's initial nodes.
        from numpy.polynomial import Polynomial

        conic_breaks = []
        oracle_breaks = []
        for first, last in zip(cell, np.roll(cell, -1, axis=0), strict=True):
            dx, dy = (last - first) / pitch
            xpoly = Polynomial(((first[0] - centre[0]) / pitch, dx))
            ypoly = Polynomial(((first[1] - centre[1]) / pitch, dy))
            edge_polynomial = (
                coefficient[0]
                + coefficient[1] * xpoly
                + coefficient[2] * ypoly
                + coefficient[3] * xpoly * xpoly
                + coefficient[4] * xpoly * ypoly
                + coefficient[5] * ypoly * ypoly
            )
            for root in edge_polynomial.roots():
                if abs(root.imag) < 1e-9 and 0 < root.real < 1:
                    conic_breaks.append(first[0] + root.real * (last[0] - first[0]))
            if kind == "limited":
                radius_poly = Polynomial((first[0], last[0] - first[0]))
                height_poly = Polynomial((first[1], last[1] - first[1]))
                analytic_edge = (
                    oracle.axis_flux
                    - 0.5
                    * oracle.pressure_coefficient
                    * (radius_poly**2 - oracle.major_radius**2) ** 2
                    - oracle.field_coefficient * height_poly**2
                )
                for root in analytic_edge.roots():
                    if abs(root.imag) < 1e-8 and 0 < root.real < 1:
                        oracle_breaks.append(
                            first[0] + root.real * (last[0] - first[0])
                        )
        discriminant = Polynomial(
            (
                coefficient[2] ** 2 - 4 * coefficient[5] * coefficient[0],
                2 * coefficient[2] * coefficient[4]
                - 4 * coefficient[5] * coefficient[1],
                coefficient[4] ** 2 - 4 * coefficient[5] * coefficient[3],
            )
        )
        for root in discriminant.roots():
            if abs(root.imag) < 1e-9:
                conic_breaks.append(centre[0] + pitch * root.real)
        breaks = np.unique(np.r_[cell[:, 0], ray_breaks, conic_breaks, oracle_breaks])
        breaks = breaks[(breaks >= cell[:, 0].min()) & (breaks <= cell[:, 0].max())]
        integral, uncertainty = np.zeros(3), 0.0
        for first, last in zip(breaks[:-1], breaks[1:], strict=True):
            value, error = quad_vec(
                vertical,
                first,
                last,
                epsabs=median_area * tolerance,
                epsrel=1e-9,
                limit=1000,
            )
            integral += value
            uncertainty += error
        errors.append(integral[0])
        estimates.append(uncertainty)
        reconstructed.append(integral[1] / areas[index])
        analytic.append(integral[2] / areas[index])
    return {
        "absolute_area_error": np.asarray(errors),
        "quadrature_error": np.asarray(estimates),
        "normalised_error": np.asarray(errors) / median_area,
        "read_fraction": np.asarray(reconstructed),
        "analytic_fraction": np.asarray(analytic),
        "median_area": median_area,
    }


def _measure_membership(kind, *, pitch_floor=1.5, receipt_name="regular-cell"):
    import os
    from pathlib import Path
    from nova.equilibrium.topology import TopologyConvention, read

    oracle, field, wall, axis, saddle = _analytic_inputs(kind)
    convention = TopologyConvention.from_cocos(17, 1.0)
    evaluate = jax.jit(read)
    calibration = None
    radius = 0.0
    if saddle is not None:
        finest = _realised_hex_geometry(wall, 2616)
        calibration = _calibrate_normal_form_radius(
            oracle, field, float(np.median(finest.pitch))
        )
        radius = calibration["radius_m"]
        print("NORMAL_FORM_RADIUS " + json.dumps(calibration), flush=True)
    policy = TopologyPolicy(
        normal_form_radius=radius, normal_form_pitch_floor=pitch_floor
    )
    rows = []
    root = os.environ.get("NOVA_TOPOLOGY_EVIDENCE_DIR")
    directory = Path(root) if root else None
    if directory is not None:
        directory.mkdir(parents=True, exist_ok=True)
    worst_panel = None
    if directory is not None and calibration is not None:
        (directory / "normal-form-radius.json").write_text(
            json.dumps(calibration, indent=2) + "\n"
        )
    for count in (132, 300, 550, 1074, 2616):
        geometry = _realised_hex_geometry(wall, count)
        effective_radius = (
            max(radius, pitch_floor * float(np.median(geometry.pitch)))
            if saddle is not None
            else 0.0
        )
        rung_policy = replace(policy, normal_form_radius=effective_radius)
        result = evaluate(field, geometry, convention, rung_policy)
        jax.block_until_ready(result)
        assert result.valid and result.qualified
        np.testing.assert_allclose(
            result.normal_form_radius, effective_radius, atol=1e-14
        )
        measure = _symmetric_difference_measure(kind, oracle, geometry, result)
        pitch = float(np.median(geometry.pitch))
        near = np.asarray(result.normal_form_cells)
        normalised = measure["normalised_error"]
        smooth_worst = int(np.argmax(np.where(near, -1.0, normalised)))
        slivers = np.flatnonzero(
            np.asarray(geometry.full_area) < 0.05 * measure["median_area"]
        )
        row = {
            "case": kind,
            "cells": len(geometry.centre),
            "pitch": pitch,
            "calibrated_radius_m": radius,
            "normal_form_radius_m": effective_radius,
            "normal_form_radius_in_pitches": effective_radius / pitch,
            "normal_form_cell_count": int(np.count_nonzero(near)),
            "median_cell_area": measure["median_area"],
            "inside_control_count": int(
                np.count_nonzero(np.asarray(result.membership) == 1.0)
            ),
            "outside_control_count": int(
                np.count_nonzero(np.asarray(result.membership) == 0.0)
            ),
            "smooth_max": float(np.max(normalised[~near])),
            "saddle_neighbourhood_max": float(np.max(normalised[near]))
            if near.any()
            else None,
            "saddle_worst_cell": int(np.argmax(np.where(near, normalised, -1.0)))
            if near.any()
            else None,
            "smooth_worst_cell": smooth_worst,
            "smooth_worst_area": float(geometry.full_area[smooth_worst]),
            "smooth_worst_fraction": float(result.membership[smooth_worst]),
            "smooth_worst_reference_fraction": float(
                measure["analytic_fraction"][smooth_worst]
            ),
            "worst_quadrature_error_regular_units": float(
                np.max(measure["quadrature_error"]) / measure["median_area"]
            ),
            "read_area_check_cell": int(
                np.argmax(
                    np.abs(measure["read_fraction"] - np.asarray(result.membership))
                    * np.asarray(geometry.full_area)
                )
            ),
            "read_area_check_regular_units": float(
                np.max(
                    np.abs(measure["read_fraction"] - np.asarray(result.membership))
                    * np.asarray(geometry.full_area)
                )
                / measure["median_area"]
            ),
            "slivers": [
                {
                    "cell": int(i),
                    "occupiable_area": float(geometry.full_area[i]),
                    "absolute_area_error": float(measure["absolute_area_error"][i]),
                    "normalised_error": float(normalised[i]),
                    "read_fraction": float(result.membership[i]),
                    "analytic_fraction": float(measure["analytic_fraction"][i]),
                }
                for i in slivers
            ],
        }
        rows.append(row)
        print("REGULAR_CELL_MEMBERSHIP " + json.dumps(row), flush=True)
        if directory is not None:
            (directory / f"{kind}-{receipt_name}-rows.json").write_text(
                json.dumps(rows, indent=2) + "\n"
            )
        panel_error = (
            row["smooth_max"] if saddle is None else row["saddle_neighbourhood_max"]
        )
        panel_cell = smooth_worst if saddle is None else row["saddle_worst_cell"]
        if worst_panel is None or panel_error > worst_panel[0]:
            worst_panel = (panel_error, geometry, result, panel_cell)
        assert row["inside_control_count"] > 0 and row["outside_control_count"] > 0, row
        assert row["smooth_max"] > 1e-12, row
        assert row["read_area_check_regular_units"] < 1e-7, row
    smooth_order = float(
        np.polyfit(
            np.log([row["pitch"] for row in rows]),
            np.log([row["smooth_max"] for row in rows]),
            1,
        )[0]
    )
    saddle_order = (
        None
        if saddle is None
        else float(
            np.polyfit(
                np.log([row["pitch"] for row in rows]),
                np.log([row["saddle_neighbourhood_max"] for row in rows]),
                1,
            )[0]
        )
    )
    envelope = 2 * max(
        row["smooth_max"] / (row["pitch"] / oracle.major_radius) ** 2 for row in rows
    )
    summary = {
        "case": kind,
        "smooth_order": smooth_order,
        "saddle_order": saddle_order,
        "measured_budget_coefficient": envelope,
    }
    print("REGULAR_CELL_ORDERS " + json.dumps(summary), flush=True)
    if directory is not None:
        (directory / f"{kind}-{receipt_name}-orders.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
    if os.environ.get("NOVA_TOPOLOGY_FIGURE_DIR"):
        _, geometry, result, worst = worst_panel
        _render_read_panel(
            os.environ["NOVA_TOPOLOGY_FIGURE_DIR"],
            kind,
            field,
            geometry,
            result,
            wall,
            axis,
            saddle,
            worst_cell=worst,
            suffix="-physical-worst",
        )
    return rows, summary, oracle


@pytest.mark.parametrize("kind", ("limited", "diverted"))
def test_topology_class_and_membership(kind):
    rows, _, oracle = _measure_membership(kind)
    coefficient = 0.274714 if kind == "limited" else 2.1805365680219544
    for row in rows:
        budget = coefficient * (row["pitch"] / oracle.major_radius) ** 2
        assert row["smooth_max"] <= budget, (row, budget)
        if row["saddle_neighbourhood_max"] is not None:
            assert row["saddle_neighbourhood_max"] <= budget, (row, budget)
    print(
        "MEMBERSHIP_AMPLITUDE_PASS "
        + json.dumps({"case": kind, "coefficient": coefficient, "rungs": len(rows)}),
        flush=True,
    )


def test_normal_form_support_away_from_null():
    from nova.equilibrium.topology import _normal_form_cell_fragments

    form = saddle_normal_form(
        jnp.zeros(2),
        jnp.asarray(((0.0, 1.0), (1.0, 0.0))),
        jnp.zeros((2, 2, 2)),
        1e-12,
        jnp.zeros((2, 2, 2, 2)),
    )
    evaluate = jax.jit(_normal_form_cell_fragments)
    for lower, upper, expected in (
        ((1.0, 1.0), (3.0, 3.0), 4.0),
        ((-1.0, -1.0), (1.0, 1.0), 2.0),
        ((-2.0, 1.0), (-1.0, 2.0), 0.0),
        ((0.0, 1.0), (1.0, 2.0), 1.0),
    ):
        x, y = lower
        u, v = upper
        cell = jnp.asarray(((x, y), (u, y), (u, v), (x, v)))
        fragments = evaluate(cell, jnp.asarray(4), form, jnp.asarray(1e-12))
        assert fragments.valid, fragments
        np.testing.assert_allclose(
            jnp.sum(fragments.area[jnp.asarray(form.positive)]), expected, atol=1e-12
        )
        np.testing.assert_allclose(
            jnp.sum(fragments.area), (u - x) * (v - y), atol=1e-12
        )


def test_normal_form_curved_cells_partition_owner_support():
    from nova.equilibrium.topology import _normal_form_cell_fragments

    hessian, third = _saddle_jet()
    form = saddle_normal_form(
        jnp.zeros(2), hessian, third, 1e-12, jnp.zeros((2, 2, 2, 2))
    )
    # Partition one null-owning square into cells that also lie away from it.
    full = jnp.asarray(((-0.2, -0.2), (0.2, -0.2), (0.2, 0.2), (-0.2, 0.2)))
    reference = saddle_cell_fragments(full, jnp.asarray(4), form, jnp.asarray(1e-12))
    total = jnp.zeros(4)
    evaluate = jax.jit(_normal_form_cell_fragments)
    for x in (-0.2, -0.1, 0.0, 0.1):
        for y in (-0.2, -0.1, 0.0, 0.1):
            cell = jnp.asarray(((x, y), (x + 0.1, y), (x + 0.1, y + 0.1), (x, y + 0.1)))
            fragments = evaluate(cell, jnp.asarray(4), form, jnp.asarray(1e-12))
            assert fragments.valid, fragments
            total += fragments.area
    np.testing.assert_allclose(total, reference.area, rtol=1e-10, atol=1e-12)


def _calibrate_normal_form_radius(oracle, field, finest_pitch):
    """Freeze the physical ray-validity radius against the finest area budget."""
    from scipy.optimize import brentq

    point = jnp.asarray(oracle.x_point)
    hessian = field.evaluate(point).hessian
    third = jax.jacfwd(lambda target: field.evaluate(target).hessian)(point)
    fourth = jax.jacfwd(jax.jacfwd(lambda target: field.evaluate(target).hessian))(
        point
    )
    form = saddle_normal_form(point, hessian, third, 1e-12, fourth)
    origin, direction, curvature, cubic = map(
        np.asarray, (form.position, form.direction, form.curvature, form.cubic)
    )
    coefficient = 0.233718
    area_budget = coefficient * (finest_pitch / oracle.major_radius) ** 2
    # Two branches may cross a regular hex; convert the area envelope to a
    # conservative normal-displacement bound using its maximum chord length.
    displacement_budget = area_budget * finest_pitch / (2 * 1.2408064788027995)

    def probe(radius):
        displacement = []
        residuals = []
        for d, c, b in zip(direction, curvature, cubic, strict=True):
            endpoint = brentq(
                lambda t: np.linalg.norm(t * d + t * t * c + t**3 * b) - radius,
                0.0,
                2 * radius,
                xtol=1e-15,
            )
            t = np.linspace(endpoint / 257, endpoint, 257)[:, None]
            points = origin + t * d + t * t * c + t**3 * b
            tangent = d + 2 * t * c + 3 * t * t * b
            normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
            normal /= np.linalg.norm(normal, axis=1)[:, None]
            delta = np.zeros(len(points))
            for _ in range(8):
                location = points + delta[:, None] * normal
                derivative = np.sum(oracle.gradient(location) * normal, axis=1)
                delta -= oracle.flux(location) / derivative
            displacement.extend(np.abs(delta))
            residuals.extend(np.abs(oracle.flux(points + delta[:, None] * normal)))
        return float(np.max(displacement)), float(np.max(residuals))

    lower, upper = finest_pitch / 100, 0.5 * oracle.minor_radius
    assert probe(lower)[0] < displacement_budget
    assert probe(upper)[0] > displacement_budget, (
        "radius bracket must see a failed branch match"
    )
    for _ in range(36):
        middle = (lower + upper) / 2
        if probe(middle)[0] <= displacement_budget:
            lower = middle
        else:
            upper = middle
    measured, residual = probe(lower)
    outside, _ = probe(lower * 1.01)
    assert measured <= displacement_budget < outside
    assert residual < 1e-10
    return {
        "radius_m": lower,
        "maximum_normal_displacement_m": measured,
        "displacement_budget_m": displacement_budget,
        "outside_radius_m": lower * 1.01,
        "outside_displacement_m": outside,
        "finest_pitch_m": finest_pitch,
        "area_budget_regular_units": area_budget,
        "budget_coefficient": coefficient,
        "samples_per_ray": 257,
        "oracle_root_residual_per_radian": residual,
    }


@pytest.mark.parametrize("kind", ("limited", "diverted"))
def test_saddle_support_order(kind):
    """Fit current reads at fixed physical radius, independently of banked rows."""
    rows, summary, _ = _measure_membership(
        kind, pitch_floor=0.0, receipt_name="fixed-radius-current"
    )
    assert [row["cells"] for row in rows] == [132, 300, 550, 1074, 2616]
    assert len({row["normal_form_radius_m"] for row in rows}) == 1
    order = summary["smooth_order"]
    print("FIXED_RADIUS_ORDER", kind, order, flush=True)
    assert order >= 1.9, f"smooth reconstruction order {order} is below 1.9: {rows}"


@pytest.mark.parametrize("kind", ("limited", "diverted"))
def test_saddle_order_evaluates_current_read(kind, monkeypatch):
    """A disabled current read cannot be hidden by historical order receipts."""
    from nova.equilibrium import topology

    class ReadCalled(Exception):
        pass

    def refuse(*args, **kwargs):
        raise ReadCalled("current read reached")

    jax.clear_caches()
    monkeypatch.setattr(topology, "read", refuse)
    with pytest.raises(ReadCalled, match="current read reached"):
        test_saddle_support_order(kind)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class UnequalSaddleField:
    asymmetry: jax.Array

    def value(self, point):
        x, z = point - jnp.asarray((2.0, 0.0))
        return -x * x - z * z + 0.5 * z**4 + self.asymmetry * z**3

    def evaluate(self, point):
        return FieldJet(
            self.value(point),
            jax.grad(self.value)(point),
            jax.hessian(self.value)(point),
        )


def test_general_saddle_census_and_capacity_refusal():
    from nova.equilibrium.topology import TopologyConvention, read

    wall = np.asarray(((1.2, -1.5), (2.8, -1.5), (2.8, 1.5), (1.2, 1.5)))
    geometry = _realised_hex_geometry(wall, 131)
    field = UnequalSaddleField(jnp.asarray(0.08))
    convention = TopologyConvention.from_cocos(17, 1.0)
    result = jax.jit(read)(field, geometry, convention, TopologyPolicy())
    assert result.valid, (result.reason, result.required_nulls)
    assert result.qualified and result.required_nulls >= 3
    assert np.count_nonzero(result.x_point_valid) == 1
    roots = (-0.24 + np.asarray((-1.0, 1.0)) * np.sqrt(0.24**2 + 16.0)) / 4
    flux = -(roots**2) + 0.5 * roots**4 + 0.08 * roots**3
    np.testing.assert_allclose(
        result.boundary, (2.0, roots[np.argmax(flux)]), atol=1e-8
    )
    refused = jax.jit(read)(
        field, geometry, convention, TopologyPolicy(null_capacity=1)
    )
    assert not refused.valid and int(refused.reason) == int(TopologyReason.CAPACITY)
    assert np.all(np.isnan(refused.membership))
    print(
        "CAPACITY_REFUSAL", int(refused.reason), int(refused.required_nulls), flush=True
    )


def test_topology_read_traces():
    from nova.equilibrium.topology import TopologyConvention, read

    _, field, wall, _, _ = _analytic_inputs("diverted")
    geometry = _realised_hex_geometry(wall, 132)
    convention, policy = (
        TopologyConvention.from_cocos(17, 1.0),
        TopologyPolicy(normal_form_radius=0.054806712567833996),
    )
    fields = (field, replace(field, scale=1.01 * field.scale))
    eager = [read(item, geometry, convention, policy) for item in fields]
    compiled = [jax.jit(read)(item, geometry, convention, policy) for item in fields]
    batch = jax.tree.map(lambda *values: jnp.stack(values), *fields)
    mapped = jax.jit(jax.vmap(read, in_axes=(0, None, None, None)))(
        batch, geometry, convention, policy
    )
    for index, result in enumerate(eager):
        assert result.valid
        for left, right, both in zip(
            jax.tree.leaves(result),
            jax.tree.leaves(compiled[index]),
            jax.tree.leaves(mapped),
            strict=True,
        ):
            np.testing.assert_allclose(
                left, right, rtol=2e-12, atol=2e-12, equal_nan=True
            )
            np.testing.assert_allclose(
                left, both[index], rtol=2e-12, atol=2e-12, equal_nan=True
            )


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class AnalyticExteriorField:
    total: ClosedFormPointField
    coupling: object
    reference_moments: object

    def value(self, point):
        return (
            self.total.value(point)
            - self.coupling.value_gradient(point, self.reference_moments)[0]
        )

    def evaluate(self, point):
        reference = self.coupling.evaluate(point, self.reference_moments)
        return jax.tree.map(jnp.subtract, self.total.evaluate(point), reference)


def _kernel_backed_field(kind, oracle, total, geometry):
    from shapely.geometry import Polygon
    from nova.equilibrium.clip_quadrature import ClippedCurrentMoments
    from nova.equilibrium.topology import BiotMomentCoupling, TotalField

    polygons = [
        np.asarray(p)[: int(n)]
        for p, n in zip(geometry.vertices, geometry.vertex_count, strict=True)
    ]
    coupling = BiotMomentCoupling.from_polygons(polygons)
    if kind == "limited":
        theta = np.linspace(0, 2 * np.pi, 4097)
        core = Polygon(
            np.column_stack(
                (
                    np.sqrt(
                        oracle.major_radius**2
                        + np.sqrt(2 * oracle.axis_flux / oracle.pressure_coefficient)
                        * np.cos(theta)
                    ),
                    np.sqrt(oracle.axis_flux / oracle.field_coefficient)
                    * np.sin(theta),
                )
            )
        )
    else:
        core = Polygon(oracle.separatrix(4097))
    node, weight = np.polynomial.legendre.leggauss(10)
    u, v = np.meshgrid((node + 1) / 2, (node + 1) / 2, indexing="ij")
    product = weight[:, None] * weight[None, :] / 4
    values = np.zeros((3, len(polygons)))
    centres = np.asarray(coupling.centre)
    for index, polygon in enumerate(polygons):
        clipped = Polygon(polygon).intersection(core)
        if clipped.is_empty:
            continue
        parts = (
            list(clipped.geoms) if clipped.geom_type == "MultiPolygon" else [clipped]
        )
        for part in parts:
            if part.geom_type != "Polygon":
                continue
            corners = np.asarray(part.exterior.coords[:-1])
            signed = 0.5 * np.sum(
                corners[:, 0] * np.roll(corners[:, 1], -1)
                - corners[:, 1] * np.roll(corners[:, 0], -1)
            )
            if signed < 0:
                corners = corners[::-1]
            for j in range(1, len(corners) - 1):
                first, second, third = corners[[0, j, j + 1]]
                edge_a, edge_b = second - first, third - first
                point = first + u[..., None] * (
                    (1 - v[..., None]) * edge_a + v[..., None] * edge_b
                )
                jacobian = edge_a[0] * edge_b[1] - edge_a[1] * edge_b[0]
                radius = point[..., 0]
                density = (
                    4 * oracle.pressure_coefficient * radius
                    + 2 * oracle.field_coefficient / radius
                ) / (4e-7 * np.pi)
                weighted = product * u * jacobian * density
                offset = point - centres[index]
                values[0, index] += np.sum(weighted)
                values[1, index] += np.sum(weighted * offset[..., 0])
                values[2, index] += np.sum(weighted * offset[..., 1])
    moments = ClippedCurrentMoments(*map(jnp.asarray, values))
    assert np.count_nonzero(values[0]) > 0 and np.all(np.isfinite(values))
    return TotalField(
        moments, coupling, AnalyticExteriorField(total, coupling, moments)
    )


@pytest.mark.parametrize("kind", ("limited", "diverted"))
@pytest.mark.parametrize("count", (1074, 2616))
@kernel_compile_expected_failure
@pytest.mark.timeout(660)
def test_kernel_backed_carrier_read(kind, count, kernel_compile_budget):
    import os
    from pathlib import Path
    import resource
    import time
    from nova.equilibrium.topology import TopologyConvention, read

    oracle, total, wall, axis, saddle = _analytic_inputs(kind)
    geometry = _realised_hex_geometry(wall, count)
    field = _kernel_backed_field(kind, oracle, total, geometry)
    policy = TopologyPolicy(
        normal_form_radius=0.054806712567833996 if saddle is not None else 0.0
    )
    convention = TopologyConvention.from_cocos(17, 1.0)
    print(
        "KERNEL_INPUT_READY",
        kind,
        count,
        float(jnp.sum(field.moments.cell_current)),
        flush=True,
    )
    compiled = jax.jit(read)
    cache_enabled = jax.config.jax_enable_compilation_cache
    jax.clear_caches()
    jax.config.update("jax_enable_compilation_cache", False)
    started = time.perf_counter()
    try:
        executable = compiled.lower(field, geometry, convention, policy).compile()
    finally:
        jax.config.update("jax_enable_compilation_cache", cache_enabled)
    compile_wall = time.perf_counter() - started
    print("KERNEL_COMPILE_COMPLETE", kind, count, compile_wall, flush=True)
    started = time.perf_counter()
    result = executable(field, geometry, convention, policy)
    jax.block_until_ready(result)
    execution_wall = time.perf_counter() - started
    raw_receipt = {
        "case": kind,
        "cells": count,
        "phase": "read-complete",
        "compile_wall_seconds": compile_wall,
        "persistent_cache_enabled_during_compile": False,
        "execute_wall_seconds": execution_wall,
        "peak_host_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "valid": bool(result.valid),
        "reason": int(result.reason),
    }
    print("KERNEL_READ_RAW " + json.dumps(raw_receipt), flush=True)
    directory = os.environ.get("NOVA_TOPOLOGY_EVIDENCE_DIR")
    if directory:
        Path(directory).mkdir(parents=True, exist_ok=True)
        (Path(directory) / f"{kind}-kernel-{count}.json").write_text(
            json.dumps(raw_receipt, indent=2) + "\n"
        )
    assert result.valid and result.qualified, (result.reason, result.required_nulls)
    pitch = float(np.median(geometry.pitch))
    assert np.linalg.norm(np.asarray(result.axis) - axis) < pitch
    assert int(result.boundary_class) == int(saddle is not None)
    if saddle is not None:
        assert (
            np.max(
                np.linalg.norm(
                    np.asarray(result.x_points)[np.asarray(result.x_point_valid)]
                    - saddle,
                    axis=1,
                )
            )
            < pitch
        )
    reference = jax.jit(read)(total, geometry, convention, policy)
    np.testing.assert_allclose(
        result.membership, reference.membership, rtol=1e-6, atol=2e-7
    )
    measure = _symmetric_difference_measure(kind, oracle, geometry, result)
    near = np.asarray(result.normal_form_cells)
    smooth_error = float(np.max(measure["normalised_error"][~near]))
    saddle_error = (
        float(np.max(measure["normalised_error"][near])) if near.any() else 0.0
    )
    coefficient = 0.274714 if kind == "limited" else 2.1805365680219544
    budget = coefficient * (pitch / oracle.major_radius) ** 2
    assert smooth_error <= budget and saddle_error <= budget
    # The fixed exterior must not cancel a change in the booked plasma operand.
    point = jnp.asarray(axis) + jnp.asarray((0.2 * pitch, 0.1 * pitch))
    changed = replace(field, moments=jax.tree.map(lambda x: 1.001 * x, field.moments))
    original_value = jax.jit(lambda f, p: f.value(p))(field, point)
    changed_value = jax.jit(lambda f, p: f.value(p))(changed, point)
    assert abs(float(changed_value - original_value)) > 1e-9
    receipt = {
        "case": kind,
        "cells": count,
        "compile_wall_seconds": compile_wall,
        "persistent_cache_enabled_during_compile": False,
        "execute_wall_seconds": execution_wall,
        "peak_host_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "valid": bool(result.valid),
        "qualified": bool(result.qualified),
        "phase": "checked",
        "smooth_error_regular_units": smooth_error,
        "normal_form_error_regular_units": saddle_error,
        "amplitude_budget": budget,
        "maximum_fraction_difference": float(
            np.max(np.abs(np.asarray(result.membership - reference.membership)))
        ),
        "current_perturbation_flux": float(changed_value - original_value),
    }
    print("KERNEL_READ " + json.dumps(receipt), flush=True)
    directory = os.environ.get("NOVA_TOPOLOGY_EVIDENCE_DIR")
    if directory:
        Path(directory).mkdir(parents=True, exist_ok=True)
        (Path(directory) / f"{kind}-kernel-{count}.json").write_text(
            json.dumps(receipt, indent=2) + "\n"
        )


def test_singular_representation_is_traced_refusal():
    from nova.equilibrium.topology import TopologyConvention, read

    _, field, wall, _, _ = _analytic_inputs("limited")
    geometry = _realised_hex_geometry(wall, 132)
    broken = replace(geometry, fit_inverse=jnp.zeros_like(geometry.fit_inverse))
    result = jax.jit(read)(
        field, broken, TopologyConvention.from_cocos(17, 1.0), TopologyPolicy()
    )
    assert not result.valid
    assert int(result.reason) == int(TopologyReason.SINGULAR_REPRESENTATION)
    print("REPRESENTATION_REFUSAL", int(result.reason), flush=True)


@pytest.mark.parametrize("kind", ("limited", "diverted"))
@kernel_compile_expected_failure
@pytest.mark.timeout(660)
def test_null_position_implicit_tangent(kind, kernel_compile_budget):
    oracle, total, wall, axis, saddle = _analytic_inputs(kind)
    geometry = _realised_hex_geometry(wall, 132)
    field = _kernel_backed_field(kind, oracle, total, geometry)
    seed = jnp.asarray(axis if saddle is None else saddle)
    pitch = float(np.median(geometry.pitch))

    def position(scale):
        perturbed = replace(
            field, moments=jax.tree.map(lambda value: scale * value, field.moments)
        )
        return stationary_read(perturbed, seed, pitch, TopologyPolicy()).position

    evaluate = jax.jit(position)
    primal, tangent = jax.jit(
        lambda scale: jax.jvp(position, (scale,), (jnp.asarray(1.0),))
    )(jnp.asarray(1.0))
    step = 1e-4
    central = (evaluate(1.0 + step) - evaluate(1.0 - step)) / (2 * step)
    np.testing.assert_allclose(primal, seed, atol=1e-8)
    np.testing.assert_allclose(tangent, central, rtol=2e-5, atol=2e-7)
    assert np.linalg.norm(tangent) > 0
    print(
        "KERNEL_NULL_TANGENT",
        kind,
        np.asarray(tangent),
        np.asarray(central),
        flush=True,
    )


def _jaxpr_equation_counts(closed):
    """Count unique subprograms as well as their expanded call sites."""
    seen = set()
    unique = 0

    def walk(value):
        nonlocal unique
        if hasattr(value, "eqns"):
            fresh = id(value) not in seen
            seen.add(id(value))
            if fresh:
                unique += len(value.eqns)
            return len(value.eqns) + sum(
                walk(parameter)
                for equation in value.eqns
                for parameter in equation.params.values()
            )
        if hasattr(value, "jaxpr"):
            return walk(value.jaxpr)
        if isinstance(value, tuple | list):
            return sum(walk(item) for item in value)
        if isinstance(value, dict):
            return sum(walk(item) for item in value.values())
        return 0

    expanded = walk(closed)
    return {"unique_equations": unique, "expanded_equations": expanded}


def _jaxpr_construct_counts(closed):
    """Attribute expanded equations to each named nested call site."""
    from collections import Counter

    names = Counter()
    primitives = Counter()

    def walk(value):
        if hasattr(value, "eqns"):
            total = len(value.eqns)
            for equation in value.eqns:
                primitives[equation.primitive.name] += 1
                nested = sum(walk(item) for item in equation.params.values())
                if nested:
                    name = str(equation.params.get("name", equation.primitive.name))
                    names[name] += nested
                total += nested
            return total
        if hasattr(value, "jaxpr"):
            return walk(value.jaxpr)
        if isinstance(value, dict):
            return sum(walk(item) for item in value.values())
        if isinstance(value, tuple | list):
            return sum(walk(item) for item in value)
        return 0

    walk(closed)
    return {
        "nested_call_equations": names.most_common(20),
        "expanded_primitives": primitives.most_common(20),
    }


@pytest.mark.parametrize("count", (132, 300, 550))
@kernel_compile_expected_failure
@pytest.mark.timeout(930)
def test_kernel_compile_growth(count, tmp_path):
    """Measure each cold topology compile in a fresh, wall-bounded process."""
    _isolated_kernel_compile(count, tmp_path)


def _isolated_kernel_compile(count, tmp_path):
    import os
    from pathlib import Path
    import subprocess
    import sys

    directory = Path(os.environ.get("NOVA_TOPOLOGY_EVIDENCE_DIR", tmp_path))
    directory.mkdir(parents=True, exist_ok=True)
    script = """
import runpy, sys
m = runpy.run_path(sys.argv[1])
m['_measure_kernel_compile_growth'](int(sys.argv[2]), sys.argv[3])
"""
    command = [
        sys.executable,
        "-u",
        "-c",
        script,
        str(Path(__file__).resolve()),
        str(count),
        str(directory),
    ]
    environment = dict(
        os.environ,
        JAX_ENABLE_COMPILATION_CACHE="false",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
    )
    mutation = os.environ.get("NOVA_TOPOLOGY_WARM_CACHE_CONTROL") == "1"
    if mutation:
        environment.update(
            JAX_ENABLE_COMPILATION_CACHE="true",
            JAX_COMPILATION_CACHE_DIR=str(tmp_path / "warm-cache"),
            JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS="0",
            JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES="0",
        )
    for arm in ("warmup", "probe") if mutation else ("cold",):
        environment["NOVA_TOPOLOGY_COMPILE_ARM"] = arm
        log = directory / f"compile-growth-{count}-{arm}.log"
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
        with log.open("w") as stream:
            stream.write(
                f"revision={revision} tree={Path.cwd()} command={json.dumps(command)}\n"
            )
            stream.flush()
            process = subprocess.Popen(
                command, stdout=stream, stderr=subprocess.STDOUT, env=environment
            )
            try:
                code = process.wait(timeout=900)
            except subprocess.TimeoutExpired:
                print(
                    f"KERNEL_COMPILE_TIMEOUT pid={process.pid} cells={count} log={log}",
                    flush=True,
                )
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                stream.write(f"\nWALL_LIMIT_SECONDS=900\nEXIT={process.returncode}\n")
                raise KernelCompileBarrier(
                    f"native {count}-cell kernel compile exceeded 900 seconds; "
                    f"receipt: {log}"
                ) from None
            stream.write(f"\nEXIT={code}\n")
        print(log.read_text(), flush=True)
        assert code == 0, f"kernel compile {arm} refused: {log}"


def _measure_kernel_compile_growth(count, tmp_path):
    """Measure cold tracing, lowering and native compilation independently."""
    import os
    from pathlib import Path
    import resource
    import time
    from nova.equilibrium.topology import TopologyConvention, read

    from jax._src import compilation_cache
    from nova.equilibrium import topology

    print(
        f"MEASUREMENT_MODULE={topology.__file__} "
        f"MEASUREMENT_CWD={Path.cwd().resolve()}",
        flush=True,
    )
    assert jax.config.jax_enable_x64 is True
    assert jax.default_backend() == "gpu"
    hits = []
    cache_get = compilation_cache.get_executable_and_time

    def observe_hit(key, *args, **kwargs):
        value = cache_get(key, *args, **kwargs)
        if value[0] is not None:
            hits.append({"key": key, "stored_compile_seconds": value[1]})
            print("PERSISTENT_CACHE_HIT " + json.dumps(hits[-1]), flush=True)
        return value

    compilation_cache.get_executable_and_time = observe_hit
    oracle, total, wall, _, _ = _analytic_inputs("limited")
    geometry = _realised_hex_geometry(wall, count)
    field = _kernel_backed_field("limited", oracle, total, geometry)
    operands = (
        field,
        geometry,
        TopologyConvention.from_cocos(17, 1.0),
        TopologyPolicy(),
    )
    receipt = {
        "cells": count,
        "kernel_edges": field.coupling.edge.shape[0],
        "backend": str(jax.devices()[0]),
        "cache_enabled_at_start": bool(jax.config.jax_enable_compilation_cache),
        "arm": os.environ.get("NOVA_TOPOLOGY_COMPILE_ARM", "cold"),
    }
    hits.clear()
    directory = Path(os.environ.get("NOVA_TOPOLOGY_EVIDENCE_DIR", tmp_path))
    directory.mkdir(parents=True, exist_ok=True)

    def checkpoint(phase):
        receipt["phase"] = phase
        receipt["peak_host_rss_kib"] = resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss
        (directory / f"compile-growth-{count}.json").write_text(
            json.dumps(receipt, indent=2) + "\n"
        )
        print("COMPILE_GROWTH " + json.dumps(receipt), flush=True)

    jax.clear_caches()
    cache_enabled = jax.config.jax_enable_compilation_cache
    try:
        checkpoint("input-ready")
        start = time.perf_counter()
        graph = jax.make_jaxpr(read)(*operands)
        receipt["trace_seconds"] = time.perf_counter() - start
        receipt.update(_jaxpr_equation_counts(graph))
        receipt.update(_jaxpr_construct_counts(graph))
        checkpoint("traced")
        start = time.perf_counter()
        lowered = jax.jit(read).lower(*operands)
        receipt["lower_seconds"] = time.perf_counter() - start
        receipt["stablehlo_bytes"] = len(lowered.as_text().encode())
        checkpoint("lowered")
        start = time.perf_counter()
        executable = lowered.compile()
        receipt["compile_seconds"] = time.perf_counter() - start
        receipt["cold_compile_wall_seconds"] = (
            receipt["lower_seconds"] + receipt["compile_seconds"]
        )
        receipt["persistent_cache_hits"] = hits
        receipt["cold_verified"] = not hits
        receipt["executable_bytes"] = (
            executable.memory_analysis().generated_code_size_in_bytes
        )
        checkpoint("compiled")
        assert not hits, "persistent cache hit: refusing to call this wall cold"
        result = executable(*operands)
        jax.block_until_ready(result)
        receipt["valid"] = bool(result.valid)
        receipt["reason"] = int(result.reason)
        checkpoint("executed")
        assert result.valid and result.qualified
    finally:
        jax.config.update("jax_enable_compilation_cache", cache_enabled)
