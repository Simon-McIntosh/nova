"""Independent extended-precision roots and polygon moments for fixed patches."""

from __future__ import annotations

from math import comb
from pathlib import Path
from typing import NamedTuple
import hashlib
import json
import os

import numpy as np

LD = np.longdouble
POWERS = ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2))
NAMES = (
    "area",
    "first_radial",
    "first_vertical",
    "second_radial",
    "second_mixed",
    "second_vertical",
)


def casteljau(coefficients, coordinate):
    """Evaluate the final Bernstein axis using only extended-precision lerps."""
    work = np.array(coefficients, dtype=LD, copy=True)
    x = np.asarray(coordinate, dtype=LD)
    for width in range(work.shape[-1] - 1, 0, -1):
        work = (1 - x[..., None]) * work[..., :width] + x[..., None] * work[
            ..., 1 : width + 1
        ]
    return work[..., 0]


def evaluate(coefficient, origin, scale, points):
    points = np.asarray(points, dtype=LD)
    local = (points - origin) / scale
    # Broadcast every patch over its point axis, then contract radial and vertical.
    blocks = np.broadcast_to(
        coefficient[:, None], points.shape[:-1] + coefficient.shape[-2:]
    )
    radial = casteljau(blocks, local[..., 0, None])
    return casteljau(radial, local[..., 1])


def gradient(coefficient, origin, scale, points):
    order = coefficient.shape[-1] - 1
    radial = order * np.diff(coefficient, axis=-1)
    vertical = order * np.diff(coefficient, axis=-2)
    local = (points - origin) / scale

    def tensor(block):
        blocks = np.broadcast_to(block[:, None], points.shape[:-1] + block.shape[-2:])
        return casteljau(casteljau(blocks, local[..., 0, None]), local[..., 1])

    return np.stack(
        (tensor(radial) / scale[..., 0], tensor(vertical) / scale[..., 1]), axis=-1
    )


def roots(coefficient, origin, scale, start, end):
    low = np.zeros(start.shape[:-1], dtype=LD)
    high = np.ones_like(low)
    sign = evaluate(coefficient, origin, scale, start) > 0
    for _ in range(80):
        mid = (low + high) / 2
        value = evaluate(
            coefficient, origin, scale, start + mid[..., None] * (end - start)
        )
        same = (value > 0) == sign
        low = np.where(same, mid, low)
        high = np.where(same, high, mid)
    fraction = (low + high) / 2
    return start + fraction[..., None] * (end - start)


def polygon_moments(vertices, centre):
    """Integrate local monomials with an independently expanded Green integral."""
    vertices = np.asarray(vertices, dtype=LD) - np.asarray(centre, dtype=LD)
    following = np.roll(vertices, -1, axis=0)
    x, y = vertices.T
    dx, dy = (following - vertices).T
    result = []
    for p, q in POWERS:
        value = LD(0)
        for i in range(p + 2):
            for j in range(q + 1):
                term = (
                    LD(comb(p + 1, i))
                    * LD(comb(q, j))
                    * x ** (p + 1 - i)
                    * dx**i
                    * y ** (q - j)
                    * dy**j
                    * dy
                    / LD((p + 1) * (i + j + 1))
                )
                value += np.sum(term, dtype=LD)
        result.append(value)
    values = np.asarray(result, dtype=LD)
    return values * (1 if values[0] >= 0 else -1)


def instrument_control():
    assert np.finfo(LD).nmant > np.finfo(np.float64).nmant
    triangle = np.asarray([[0, 0], [1, 0], [0, 1]], dtype=LD)
    expected = np.asarray(
        [LD(1) / 2, LD(1) / 6, LD(1) / 6, LD(1) / 12, LD(1) / 24, LD(1) / 12]
    )
    error = np.max(np.abs(polygon_moments(triangle, np.zeros(2, dtype=LD)) - expected))
    assert error < 8 * np.finfo(LD).eps
    coefficient = np.asarray([[[-LD(1) / 3, LD(2) / 3], [-LD(1) / 3, LD(2) / 3]]])
    origin = np.zeros((1, 1, 2), dtype=LD)
    scale = np.ones_like(origin)
    point = roots(
        coefficient,
        origin,
        scale,
        np.asarray([[[0, 0]]], dtype=LD),
        np.asarray([[[1, 0]]], dtype=LD),
    )
    root_error = abs(point[0, 0, 0] - LD(1) / 3)
    assert root_error < 8 * np.finfo(LD).eps
    return {
        "longdouble_mantissa_bits": int(np.finfo(LD).nmant),
        "double_mantissa_bits": int(np.finfo(np.float64).nmant),
        "triangle_moment_max_absolute_error": float(error),
        "linear_root_absolute_error": float(root_error),
    }


def reference_polygons(coefficient, origin, scale, vertices, counts, centres):
    maximum = vertices.shape[1]
    slot = np.arange(maximum)
    following_slot = np.where(slot[None, :] + 1 < counts[:, None], slot[None, :] + 1, 0)
    following = np.take_along_axis(vertices, following_slot[..., None], axis=1)
    values = evaluate(coefficient, origin, scale, vertices)
    valid = slot[None] < counts[:, None]
    inside = values > 0
    next_inside = np.take_along_axis(inside, following_slot, axis=1)
    crossing = valid & (inside != next_inside)
    assert np.all(crossing.sum(axis=1) == 2), crossing.sum(axis=1)
    edge_roots = roots(coefficient, origin, scale, vertices, following)
    starts, ends, interior, tails = [], [], [], []
    for cell in range(len(vertices)):
        points = []
        leaving = None
        for edge in range(counts[cell]):
            if inside[cell, edge]:
                points.append(vertices[cell, edge])
            if crossing[cell, edge]:
                if inside[cell, edge]:
                    leaving = len(points)
                points.append(edge_roots[cell, edge])
        points = points[leaving:] + points[:leaving]
        starts.append(points[0])
        ends.append(points[1])
        interior.append(points[2])
        tails.append(points[2:])
    start, end, interior = map(
        lambda x: np.asarray(x, dtype=LD), (starts, ends, interior)
    )
    parameter = np.arange(129, dtype=LD) / 128
    chord = start[:, None] + parameter[None, :, None] * (end - start)[:, None]
    delta = end - start
    normal = np.stack((-delta[:, 1], delta[:, 0]), axis=-1)
    midpoint = (start + end) / 2
    side = np.where(np.sum((interior - midpoint) * normal, axis=1) < 0, LD(1), LD(-1))
    extent = np.minimum(
        np.sqrt(np.sum((interior - midpoint) ** 2, axis=1) / np.sum(delta**2, axis=1)),
        LD(1),
    )
    signed = side * extent
    lower, upper = (
        np.minimum(signed, LD(0))[:, None],
        np.maximum(signed, LD(0))[:, None],
    )
    offset = np.zeros(chord.shape[:-1], dtype=LD)
    for _ in range(40):
        points = chord + offset[..., None] * normal[:, None]
        value = evaluate(coefficient, origin, scale, points)
        derivative = np.sum(
            gradient(coefficient, origin, scale, points) * normal[:, None], axis=-1
        )
        step = np.divide(
            value, derivative, out=np.zeros_like(value), where=derivative != 0
        )
        offset = np.clip(offset - step, lower, upper)
    offset[:, 0] = 0
    offset[:, -1] = 0
    arc = chord + offset[..., None] * normal[:, None]
    polygons = [
        np.concatenate((arc[i], np.asarray(tails[i], dtype=LD)))
        for i in range(len(vertices))
    ]
    moments = np.stack(
        [polygon_moments(p, c) for p, c in zip(polygons, centres, strict=True)]
    )
    residual = np.max(np.abs(evaluate(coefficient, origin, scale, arc)))
    return np.stack((start, end), axis=1), polygons, moments, float(residual)


def frozen_fixture():
    import jax.numpy as jnp
    from benchmarks import solovev_certificate as certificate
    from scripts.analytic_oracle_fixtures import measure as fixture
    from nova.linalg.split_spline import fit_split_spline

    case = "weak-rotation-reactor-static"
    carrier, source, exact = certificate._case(case)
    machine = certificate._case_machine(case, carrier, exact, -110)
    coordinate = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    state = certificate._exact_state(case, exact, coordinate)
    operator = fixture.forward_operator(source, machine)
    support = fixture._analytic_profile_support(exact, operator, state)
    physical = jnp.asarray(state[: operator.physical_node_number])
    grid_flux, _ = operator.topology.split_flux_map(physical)
    axis = fixture._analytic_axis_flux(exact)
    psi_norm = (grid_flux - axis) / -axis
    coordinates = jnp.asarray(operator.grid.coordinate)
    surface = fit_split_spline(
        coordinates[None, :, 0],
        coordinates[None, :, 1],
        psi_norm[None],
        psi_norm[None] - 1,
        order=6,
        regularization=1e-14,
    )
    assert bool(surface.fit_executed)
    mesh = operator.moment_geometry.atomic_mesh
    selection = np.flatnonzero(
        np.asarray(support.included) & np.asarray(support.boundary)
    )[:20]
    assert len(selection) >= 20
    vertices = np.asarray(mesh.node_coordinates)[np.asarray(mesh.cell_nodes)[selection]]
    counts = np.asarray(mesh.cell_vertex_count)[selection]
    centres = np.asarray(mesh.centroids)[selection]
    coefficient = -np.asarray(surface.level_set_coefficients)
    origin = np.asarray([surface.radial[0], surface.vertical[0]])
    scale = np.asarray(
        [
            surface.radial[1] - surface.radial[0],
            surface.vertical[1] - surface.vertical[0],
        ]
    )
    return selection, vertices, counts, centres, coefficient, origin, scale


def bicubic_blocks(coefficient, origin, scale, vertices, counts):
    """Interpolate each frozen patch on a rational four-by-four local grid."""
    from fractions import Fraction

    nodes = [Fraction(k, 3) for k in range(4)]
    augmented = [
        [Fraction(comb(3, j)) * x**j * (1 - x) ** (3 - j) for j in range(4)]
        + [Fraction(i == j) for j in range(4)]
        for i, x in enumerate(nodes)
    ]
    for column in range(4):
        pivot = next(i for i in range(column, 4) if augmented[i][column])
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale_value = augmented[column][column]
        augmented[column] = [x / scale_value for x in augmented[column]]
        for row in range(4):
            if row != column:
                weight = augmented[row][column]
                augmented[row] = [
                    a - weight * b
                    for a, b in zip(augmented[row], augmented[column], strict=True)
                ]
    inverse = np.asarray(
        [[LD(x.numerator) / LD(x.denominator) for x in row[4:]] for row in augmented],
        dtype=LD,
    )
    origins = np.stack(
        [p[:n].min(axis=0) for p, n in zip(vertices, counts, strict=True)]
    )[:, None]
    scales = np.stack(
        [np.ptp(p[:n], axis=0) for p, n in zip(vertices, counts, strict=True)]
    )[:, None]
    u, v = np.meshgrid(np.arange(4, dtype=LD) / 3, np.arange(4, dtype=LD) / 3)
    local = np.stack((u.ravel(), v.ravel()), axis=-1)
    points = origins + scales * local[None]
    values = evaluate(coefficient, origin, scale, points).reshape(-1, 4, 4)
    result = np.einsum("ij,cjk,lk->cil", inverse, values, inverse)
    # Freeze to the exact binary64 inputs consumed by both device routes.
    return (
        np.asarray(result, dtype=np.float64),
        np.asarray(origins, dtype=np.float64),
        np.asarray(scales, dtype=np.float64),
    )


def device_polygons(coefficient, origin, scale, vertices, counts, centres):
    import jax
    import jax.numpy as jnp
    from nova.linalg.tensor_spline import _tensor_bernstein
    from nova.equilibrium.separatrix_clip import _traced_clip
    from nova.equilibrium.forward_operator import _implicit_traced_level_arc

    class Polynomial(NamedTuple):
        coefficient: object
        origin: object
        scale: object

        def __call__(self, points):
            local = (points - self.origin) / self.scale
            order = self.coefficient.shape[-1] - 1
            return _tensor_bernstein(
                self.coefficient[:, None], local[..., 0], local[..., 1], order, order
            )

    size, width, _ = vertices.shape
    nodes = jnp.arange(size * width).reshape(size, width)
    coordinates = jnp.asarray(vertices.reshape(-1, 2))
    evaluator = Polynomial(
        jnp.asarray(coefficient), jnp.asarray(origin), jnp.asarray(scale)
    )
    function = jax.jit(
        lambda e: _traced_clip(
            coordinates,
            nodes,
            jnp.asarray(counts),
            jnp.asarray(centres),
            width,
            jnp.zeros(size * width),
            curve_evaluator=e,
            participating_cell=jnp.ones(size, dtype=bool),
            arc_tracer=_implicit_traced_level_arc,
        )
    )
    support = function(evaluator)
    jax.block_until_ready(support)
    assert np.all(np.asarray(support.vertex_count) >= 129)
    points = np.asarray(support.support_vertices)
    moments = np.column_stack(
        (
            support.area,
            support.first_area_moment[:, 0],
            support.first_area_moment[:, 1],
            support.second_area_moment[:, 0, 0],
            support.second_area_moment[:, 0, 1],
            support.second_area_moment[:, 1, 1],
        )
    )
    return points[:, (0, 128)], points, np.asarray(support.vertex_count), moments


def compare(
    output, name, selection, coefficient, origin, scale, vertices, counts, centres
):
    import jax
    import measure_bernstein as prior

    restore = prior.original_evaluator()
    baseline = device_polygons(coefficient, origin, scale, vertices, counts, centres)
    restore()
    jax.clear_caches()
    candidate = device_polygons(coefficient, origin, scale, vertices, counts, centres)
    reference_crossings, polygons, reference_moments, residual = reference_polygons(
        np.asarray(coefficient, dtype=LD),
        np.asarray(origin, dtype=LD),
        np.asarray(scale, dtype=LD),
        np.asarray(vertices, dtype=LD),
        counts,
        np.asarray(centres, dtype=LD),
    )
    baseline_values = [baseline[0][..., 0], baseline[0][..., 1], *baseline[3].T]
    candidate_values = [candidate[0][..., 0], candidate[0][..., 1], *candidate[3].T]
    references = [
        reference_crossings[..., 0],
        reference_crossings[..., 1],
        *reference_moments.T,
    ]
    rows = []
    for quantity, before, after, exact in zip(
        ("crossing_radial", "crossing_vertical", *NAMES),
        baseline_values,
        candidate_values,
        references,
        strict=True,
    ):
        norm = np.sqrt(np.sum(exact**2, dtype=LD))
        assert norm > 0

        def error(value):
            return (
                np.sqrt(np.sum((np.asarray(value, dtype=LD) - exact) ** 2, dtype=LD))
                / norm
            )

        old_error, candidate_error = error(before), error(after)
        rows.append(
            {
                "quantity": quantity,
                "baseline_relative_l2": float(old_error),
                "candidate_relative_l2": float(candidate_error),
                "candidate_no_worse": bool(candidate_error <= old_error),
            }
        )
    result = {
        "reference": name,
        "job": os.environ["SLURM_JOB_ID"],
        "cells": selection.tolist(),
        "cell_count": len(selection),
        "degree": coefficient.shape[-1] - 1,
        "coefficient_sha256": hashlib.sha256(
            np.asarray(coefficient).tobytes()
        ).hexdigest(),
        "maximum_reference_arc_residual": residual,
        "rows": rows,
        "passed": all(row["candidate_no_worse"] for row in rows),
    }
    np.savez(
        output / f"{name}-arrays.npz",
        cell_ids=selection,
        coefficients=coefficient,
        origin=origin,
        scale=scale,
        atomic_vertices=vertices,
        counts=counts,
        centres=centres,
        baseline_crossings=baseline[0],
        candidate_crossings=candidate[0],
        baseline_vertices=baseline[1],
        candidate_vertices=candidate[1],
        baseline_moments=baseline[3],
        candidate_moments=candidate[3],
        reference_crossings=reference_crossings,
        reference_moments=reference_moments,
        **{
            f"reference_polygon_{cell}": polygon
            for cell, polygon in zip(selection, polygons, strict=True)
        },
    )
    (output / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return result


def measure(output):
    import jax
    from nova.jax.config import configure_dtypes

    root = Path(__file__).resolve().parents[3]
    fingerprint = {
        path: hashlib.sha256((root / path).read_bytes()).hexdigest()
        for path in (
            "nova/linalg/interpolant.py",
            "nova/equilibrium/flux_surface_extraction.py",
        )
    }
    helper_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    summary_path = output / "reference-summary.json"
    if summary_path.exists():
        saved = json.loads(summary_path.read_text())
        if (
            saved.get("source_sha256") == fingerprint
            and saved.get("helper_sha256") == helper_hash
            and saved.get("job") == os.environ["SLURM_JOB_ID"]
        ):
            print("REUSE_REFERENCE_RECEIPT", str(summary_path), flush=True)
            return
    configure_dtypes()
    assert jax.config.jax_enable_x64
    controls = instrument_control()
    selection, vertices, counts, centres, coefficient, origin, scale = frozen_fixture()
    size = len(selection)
    blocks = np.broadcast_to(coefficient, (size,) + coefficient.shape).copy()
    origins = np.broadcast_to(origin, (size, 1, 2)).copy()
    scales = np.broadcast_to(scale, (size, 1, 2)).copy()
    production = compare(
        output,
        "production-patch-reference",
        selection,
        blocks,
        origins,
        scales,
        vertices,
        counts,
        centres,
    )
    cubic, cubic_origin, cubic_scale = bicubic_blocks(
        np.asarray(blocks, dtype=LD),
        np.asarray(origins, dtype=LD),
        np.asarray(scales, dtype=LD),
        np.asarray(vertices, dtype=LD),
        counts,
    )
    bicubic = compare(
        output,
        "bicubic-reference",
        selection,
        cubic,
        cubic_origin,
        cubic_scale,
        vertices,
        counts,
        centres,
    )
    result = {
        "source_sha256": fingerprint,
        "helper_sha256": helper_hash,
        "job": os.environ["SLURM_JOB_ID"],
        "completed": True,
        "controls": controls,
        "production": production,
        "bicubic": bicubic,
        "passed": production["passed"] and bicubic["passed"],
    }
    (output / "reference-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
