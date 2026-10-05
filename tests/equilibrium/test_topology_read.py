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
        topology=TopologyPolicy(null_capacity=7, hessian_tolerance=1e-8),
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


def test_biot_moment_point_derivatives():
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


def _render_read_panel(directory, kind, field, geometry, result, wall, axis, saddle):
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
    poloidal.draw_flux_contours(
        axes,
        radial,
        vertical,
        sampled.reshape(rr.shape),
        levels,
        wall=(wall,),
        color=DEFAULT_INK.flux_color,
        linewidth=2.6,
    )
    poloidal.draw_wall(axes, (wall,))
    reference_x = np.empty((0, 2)) if saddle is None else np.asarray(saddle)[None]
    poloidal.draw_nulls(axes, axis, reference_x, style=reference_style, contain=(wall,))
    poloidal.draw_nulls(
        axes,
        np.asarray(result.axis),
        np.asarray(result.x_points)[np.asarray(result.x_point_valid)],
        style=read_style,
        contain=(wall,),
    )
    axes.text(
        0.02,
        0.97,
        "analytic field / larger nulls",
        transform=axes.transAxes,
        color="#333333",
        fontsize=20,
        va="top",
    )
    axes.text(
        0.02,
        0.91,
        "cell field / read nulls",
        transform=axes.transAxes,
        color=DEFAULT_INK.flux_color,
        fontsize=20,
        va="top",
    )
    figure.savefig(directory / f"{kind}-poloidal.png", dpi=100, facecolor="white")
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
        row = {
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
    zoom = max(np.linalg.norm(boundary - contact) * 2.5, 0.015)
    for index, axes in enumerate(panels):
        poloidal_axes(axes)
        if index == 0:
            lower, upper = wall.min(axis=0), wall.max(axis=0)
        else:
            lower = np.minimum(boundary, contact) - zoom
            upper = np.maximum(boundary, contact) + zoom
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
        0.98,
        "zero-flux region / larger axis",
        transform=panels[0].transAxes,
        color="#333333",
        fontsize=20,
        va="top",
    )
    panels[0].text(
        0.02,
        0.92,
        "wall-selected level / read axis",
        transform=panels[0].transAxes,
        color=DEFAULT_INK.flux_color,
        fontsize=20,
        va="top",
    )
    panels[1].text(
        0.02,
        0.98,
        "wall contact detail",
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
            "cells": count,
            "pitch_m": float(np.median(geometry.pitch)),
            "read_level_wb": float(result.boundary_flux),
            "analytic_boundary_flux_wb": 0.0,
            "read_wall_unit": 0,
            "read_wall_segment": read_segment,
            "read_contact_m": np.asarray(result.boundary).tolist(),
            "analytic_wall_unit": 0,
            "analytic_wall_vertex": analytic_vertex,
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
    assert max(row["read_level_wb"] for row in rows) == min(
        row["read_level_wb"] for row in rows
    )
