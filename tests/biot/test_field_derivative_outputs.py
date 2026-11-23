"""Primal field gradients share the polygon contour and obey Maxwell identities."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import os
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from shapely.geometry import Point, Polygon

from nova.biot import polygonanalytic as kernel
from nova.biot.polygon import MU0
from nova.equilibrium.topology import BiotMomentCoupling

assert jax.config.jax_enable_x64 is True


def _inputs(polygons, expansion_offset=(0.0, 0.0)):
    coupling = BiotMomentCoupling.from_polygons(polygons)
    centre = np.asarray(coupling.centre).T
    return (
        np.asarray(coupling.edge),
        np.asarray(coupling.weight),
        np.asarray(coupling.norm),
        centre,
        centre + np.asarray(expansion_offset)[:, None],
        np.asarray(coupling.reflection_axis),
        np.asarray(coupling.reflection_partner),
    )


def _rows(points, geometry, *, derivatives=False, xp=np):
    return xp.stack(
        kernel.packed_analytic_moments(
            xp,
            xp.asarray(points[:, 0]),
            xp.asarray(points[:, 1]),
            *map(xp.asarray, geometry),
            field_derivatives=derivatives,
        )
    )


def _targets(kind):
    random = np.random.default_rng(719)
    shapes = (
        np.array([[-0.5, -0.4], [0.5, -0.4], [0.5, 0.4], [-0.5, 0.4]]),
        np.array([[-0.5, -0.35], [0.3, -0.45], [0.55, 0.1], [0.1, 0.5], [-0.4, 0.3]]),
        0.5
        * np.column_stack(
            (np.cos(np.arange(6) * np.pi / 3), np.sin(np.arange(6) * np.pi / 3))
        ),
    )
    count = 200 if kind == "generic" else 50
    polygons, targets, pitches = [], [], []
    for index in range(count):
        shape = shapes[index % len(shapes)]
        angle = 0.37 if index % 4 == 0 else 0.0
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        polygon = shape @ rotation.T + [3.0, 0.13]
        if index % 5 == 0:
            polygon = polygon[::-1]
        pitch = np.max(np.linalg.norm(polygon - polygon.mean(axis=0), axis=1))
        if kind == "generic":
            # Include material interiors as well as exterior targets. Keep the
            # stencil inside one smooth current-density region.
            while True:
                target = random.uniform([1.8, -0.95], [4.2, 1.2])
                distance = Polygon(polygon).boundary.distance(Point(target))
                if (
                    distance > 0.12 * pitch
                    and np.min(abs(polygon[:, 1] - target[1])) > 2e-7
                ):
                    break
        else:
            level = polygon[index % len(polygon), 1]
            offset = (0.0, 1e-12, -1e-12, 9e-8, -9e-8)[index % 5]
            target = np.array([4.4 + random.uniform(0.0, 0.4), level + offset])
            assert np.min(np.linalg.norm(polygon - target, axis=1)) >= pitch
            assert np.min(abs(polygon[:, 1] - target[1])) <= 1e-7
        polygons.append(polygon)
        targets.append(target)
        pitches.append(pitch)
    return polygons, np.array(targets), np.array(pitches)


def _sixth_difference(points, geometry, steps, coordinate):
    direction = np.eye(2)[coordinate]
    estimates = []
    for factor in (1.0, 0.5, 0.25):
        shift = (steps * factor)[:, None] * direction
        estimates.append(
            (
                _rows(points + shift, geometry)[3:9]
                - _rows(points - shift, geometry)[3:9]
            )
            / (2 * steps * factor)
        )
    return (64 * estimates[2] - 20 * estimates[1] + estimates[0]) / 45


@pytest.fixture(scope="module", params=("generic", "corner_height"))
def agreement(request):
    polygons, points, pitches = _targets(request.param)
    geometry = _inputs(polygons, expansion_offset=(0.07, -0.04))
    rows = _rows(points, geometry, derivatives=True)
    np.testing.assert_array_equal(rows[:9], _rows(points, geometry))
    assert rows.shape == (15, len(points)) and np.isfinite(rows).all()
    if os.environ.get("NOVA_FIELD_DERIVATIVE_SIGN_FLIP") == "1":
        rows[9] *= -1.0
    current = np.array(
        [
            (MU0 / Polygon(polygon).area)
            * np.array([1.0, *(point - geometry[4][:, index])])
            if Polygon(polygon).contains(Point(point))
            else np.zeros(3)
            for index, (point, polygon) in enumerate(zip(points, polygons, strict=True))
        ]
    ).T
    radial = _sixth_difference(points, geometry, 0.012 * pitches, 0)
    vertical = _sixth_difference(points, geometry, 0.012 * pitches, 1)
    actual = np.stack(
        (
            rows[9:12],
            rows[12:15],
            rows[12:15] - current,
            -rows[3:6] / points[:, 0] - rows[9:12],
        )
    )
    reference = np.stack((radial[:3], vertical[:3], radial[3:], vertical[3:]))
    return request.param, actual, reference


@pytest.mark.parametrize(
    "component", range(4), ids=("d_r_Br", "d_z_Br", "d_r_Bz", "d_z_Bz")
)
def test_field_derivative_agreement(agreement, component):
    kind, actual, reference = agreement
    # Relative error is measured per target and source moment against the
    # largest component of that target's field-gradient tensor. A vanishing
    # component therefore cannot turn round-off into an infinite relative error.
    scale = np.max(abs(reference), axis=0)
    assert np.all(scale > 0) and np.all(np.isfinite(reference))
    relative = abs(actual[component] - reference[component]) / scale
    maximum = float(relative.max())
    tolerance = 1e-8 if kind == "generic" else 1e-6
    print(
        f"FIELD_DERIVATIVE cohort={kind} targets={actual.shape[-1]} "
        f"component={component} maximum_relative={maximum:.17g} bound={tolerance}"
    )
    assert maximum <= tolerance


def test_packed_field_derivative_lowering_bound():
    from tests.equilibrium.test_topology_read import (
        _analytic_inputs,
        _realised_hex_geometry,
    )

    _, _, wall, _, _ = _analytic_inputs("diverted")
    geometry = _realised_hex_geometry(wall, 24)
    polygons = [
        np.asarray(p)[: int(n)]
        for p, n in zip(geometry.vertices, geometry.vertex_count, strict=True)
    ]
    operands = tuple(map(jnp.asarray, _inputs(polygons)))
    points = jnp.broadcast_to(jnp.array([3.15, 0.23]), (len(polygons), 2))
    counts = []
    for derivatives in (False, True):

        def program(targets, *arguments):
            return _rows(targets, arguments, derivatives=derivatives, xp=jnp)

        lowered = jax.jit(program).lower(points, *operands)
        hlo = lowered.compiler_ir("hlo").as_hlo_module().to_string()
        counts.append(len(re.findall(r"^\s*(?:ROOT )?%\S+ =", hlo, re.MULTILINE)))
    assert counts[0] > 1000, "counter must see the known-present polygon kernel"
    print(
        f"FIELD_DERIVATIVE_LOWERING cells={len(polygons)} "
        f"nine_rows={counts[0]} fifteen_rows={counts[1]} "
        f"ratio={counts[1] / counts[0]:.9g}"
    )
    assert counts[1] <= 2 * counts[0]
