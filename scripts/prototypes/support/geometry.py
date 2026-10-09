"""Represent the read's selected conic and normal-form fragments as polygons."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

from nova.equilibrium import topology
from nova.equilibrium.separatrix_clip import _area_moments
from scripts.analytic_oracle_fixtures import measure as fixture


def certificate_field(field, kind):
    """Use the same flux unit as the certificate analytic state."""
    return (
        replace(field, scale=field.scale / (2 * jnp.pi))
        if kind == "diverted"
        else field
    )


def pack(polygons, template):
    arrays = []
    for polygon in polygons:
        if polygon.is_empty:
            arrays.append(np.empty((0, 2)))
            continue
        if polygon.geom_type != "Polygon" or len(polygon.interiors):
            raise ValueError(
                "certificate booker requires one hole-free polygon per cell"
            )
        arrays.append(np.asarray(orient(polygon, sign=1.0).exterior.coords[:-1]))
    size = max(24, max(map(len, arrays)))
    vertices = np.zeros((len(arrays), size, 2))
    count = np.asarray(list(map(len, arrays)), dtype=np.int32)
    area = np.zeros(len(arrays))
    first = np.zeros((len(arrays), 2))
    second = np.zeros((len(arrays), 2, 2))
    centres = np.asarray(template.centroids)
    for index, points in enumerate(arrays):
        vertices[index, : len(points)] = points
        area[index], first[index], second[index] = _area_moments(points, centres[index])
    included = area > 0
    full = np.asarray(template.full_area)
    boundary = included & (area < full * (1 - 1e-10))
    branch_vertices = np.stack((vertices, np.zeros_like(vertices)), axis=1)
    result = template._replace(
        support_vertices=jnp.asarray(vertices),
        vertex_count=jnp.asarray(count),
        included=jnp.asarray(included),
        boundary=jnp.asarray(boundary),
        area=jnp.asarray(area),
        first_area_moment=jnp.asarray(first),
        second_area_moment=jnp.asarray(second),
        contour_area=jnp.asarray(area.sum()),
        patch_area_sum=jnp.asarray(area.sum()),
        branch_support_vertices=jnp.asarray(branch_vertices),
        branch_vertex_count=jnp.asarray(
            np.stack((count, np.zeros_like(count)), axis=1)
        ),
        branch_area=jnp.asarray(np.stack((area, np.zeros_like(area)), axis=1)),
        branch_first_area_moment=jnp.asarray(
            np.stack((first, np.zeros_like(first)), axis=1)
        ),
        branch_second_area_moment=jnp.asarray(
            np.stack((second, np.zeros_like(second)), axis=1)
        ),
        saddle=jnp.zeros(len(arrays), dtype=bool),
        vertex_capacity=jnp.asarray(size),
        refused_cell_count=jnp.asarray(0),
    )
    return result


def analytic_polygons(exact, cells, shift=(0.0, 0.0), points=8193, subdivisions=0):
    boundary = fixture._analytic_separatrix(exact, points)
    for _ in range(subdivisions):
        midpoint = 0.5 * (boundary + np.roll(boundary, -1, axis=0))
        for _ in range(12):
            if hasattr(exact, "x_point"):
                value = np.asarray(exact.flux(midpoint))
                gradient = np.asarray(exact.gradient(midpoint))
            else:
                value = np.asarray(exact.flux(midpoint[:, 0], midpoint[:, 1]))
                gradient = np.column_stack(
                    exact.flux_gradient(midpoint[:, 0], midpoint[:, 1])
                )
            norm = np.sum(gradient * gradient, axis=1)
            if np.any(norm == 0) or not np.isfinite(norm).all():
                raise ValueError("analytic midpoint projection has a singular gradient")
            midpoint -= value[:, None] * gradient / norm[:, None]
        value = (
            np.asarray(exact.flux(midpoint))
            if hasattr(exact, "x_point")
            else np.asarray(exact.flux(midpoint[:, 0], midpoint[:, 1]))
        )
        if np.max(np.abs(value)) > 1e-10 * abs(float(exact.axis_flux)):
            raise ValueError(
                "analytic midpoint projection did not reach the true zero level"
            )
        boundary = np.stack((boundary, midpoint), axis=1).reshape(-1, 2)
    boundary = boundary + np.asarray(shift)
    core = Polygon(boundary)
    if not core.is_valid:
        raise ValueError("analytic separatrix polygon is invalid")
    return [Polygon(cell).intersection(core) for cell in cells]


def normal_form_sectors(form, extent):
    """Sample the read's cubic rays, with the same positive sector ordering."""
    origin = np.asarray(form.position)
    directions, curvatures, cubics = map(
        np.asarray, (form.direction, form.curvature, form.cubic)
    )
    box = origin + extent * np.asarray(
        ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0))
    )
    ends, parameters = [], []

    def cross(a, b):
        return a[0] * b[1] - a[1] * b[0]

    for direction, curvature, cubic in zip(directions, curvatures, cubics, strict=True):
        hits = []
        for edge_index, (first, last) in enumerate(
            zip(box, np.roll(box, -1, axis=0), strict=True)
        ):
            edge = last - first
            roots = np.polynomial.polynomial.polyroots(
                (
                    cross(origin - first, edge),
                    cross(direction, edge),
                    cross(curvature, edge),
                    cross(cubic, edge),
                )
            )
            for root in roots:
                if abs(root.imag) > 1e-9 or root.real <= 1e-12:
                    continue
                t = float(root.real)
                point = origin + t * direction + t * t * curvature + t**3 * cubic
                fraction = (point - first) @ edge / (edge @ edge)
                if -1e-10 <= fraction <= 1 + 1e-10:
                    hits.append((t, edge_index + np.clip(fraction, 0, 1)))
        if not hits:
            raise ValueError("normal-form ray has no enclosing-box exit")
        end, parameter = min(hits)
        ends.append(end)
        parameters.append(parameter)
    curves = []
    for index, end in enumerate(ends):
        t = np.linspace(0, end, 4097)[:, None]
        curves.append(
            origin
            + t * directions[index]
            + t * t * curvatures[index]
            + t**3 * cubics[index]
        )
    sectors = []
    for index in np.flatnonzero(np.asarray(form.positive)):
        following = (index + 1) % 4
        start, stop = parameters[index], parameters[following]
        if stop <= start:
            stop += 4
        corners = [
            box[k % 4] for k in range(int(np.floor(start)) + 1, int(np.ceil(stop)))
        ]
        points = np.vstack(
            (curves[index], np.asarray(corners).reshape(-1, 2), curves[following][::-1])
        )
        sectors.append(Polygon(points))
    return sectors


def read_polygons(geometry, reading, sigma):
    """Carry fragment connectivity into the unchanged polygon current booker."""
    vertices = np.asarray(geometry.vertices)
    count = np.asarray(geometry.vertex_count)
    centres = np.asarray(geometry.centre)
    pitches = np.asarray(geometry.pitch)
    fractions = np.asarray(reading.membership)
    selected = np.asarray(reading.fragment_selected)
    coefficients = float(sigma) * np.asarray(reading.field_coefficients).copy()
    coefficients[:, 0] -= float(sigma) * float(reading.boundary_flux)
    local = (vertices - centres[:, None, :]) / pitches[:, None, None]
    fragments = jax.jit(
        jax.vmap(topology.quadratic_cell_fragments, in_axes=(0, 0, 0, None)),
        static_argnums=(3,),
    )(
        jnp.asarray(local),
        geometry.vertex_count,
        jnp.asarray(coefficients),
        selected.shape[1],
    )
    fragments = jax.tree.map(np.asarray, fragments)
    intervals = jax.jit(topology._positive_vertical_intervals)
    represented = np.asarray(reading.normal_form_cells)
    sectors = None
    if np.any(represented):
        extent = 1.2 * np.max(
            np.linalg.norm(
                vertices[represented] - np.asarray(reading.saddle_form.position),
                axis=-1,
            )
        )
        sectors = normal_form_sectors(reading.saddle_form, extent)
    result = []
    for index in range(len(count)):
        cell = Polygon(vertices[index, : count[index]])
        if fractions[index] <= 1e-14:
            result.append(Polygon())
            continue
        if fractions[index] >= 1 - 1e-14:
            result.append(cell)
            continue
        if represented[index]:
            parts = [
                sector.intersection(cell)
                for live, sector in zip(selected[index, :2], sectors, strict=True)
                if live
            ]
            result.append(unary_union(parts))
            continue
        breaks = fragments.slice_breaks[index]
        left, right = breaks[:-1], breaks[1:]
        # Cosine spacing resolves conic square-root endpoints without a raster.
        parameter = 0.5 * (1 - np.cos(np.linspace(0, np.pi, 513)))
        x = left[:, None] + (right - left)[:, None] * parameter
        bottom, top, _ = map(
            np.asarray,
            intervals(
                jnp.asarray(local[index]),
                jnp.asarray(count[index]),
                jnp.asarray(coefficients[index]),
                jnp.asarray(x),
            ),
        )
        parts = []
        for strip in range(len(left)):
            if right[strip] - left[strip] < 1e-13:
                continue
            for interval, fragment in enumerate(fragments.slice_labels[index, strip]):
                if fragment < 0 or not selected[index, fragment]:
                    continue
                lower = np.column_stack((x[strip], bottom[strip, :, interval]))
                upper = np.column_stack((x[strip, ::-1], top[strip, ::-1, interval]))
                points = centres[index] + pitches[index] * np.vstack((lower, upper))
                polygon = Polygon(points)
                if polygon.is_valid and polygon.area > 0:
                    parts.append(polygon)
                elif polygon.area > 0:
                    parts.append(polygon.buffer(0))
        polygon = unary_union(parts)
        # Adjacent analytic strips share an edge. Roundoff-sized gaps in a
        # union are closed far below the measured polygon-area error bound.
        if polygon.geom_type == "MultiPolygon":
            tolerance = pitches[index] * 1e-11
            polygon = polygon.buffer(tolerance).buffer(-tolerance)
        result.append(polygon.intersection(cell))
    measured = np.asarray([p.area for p in result]) / np.asarray(geometry.full_area)
    error = float(np.max(np.abs(measured - fractions)))
    print(f"READ_POLYGON_FRACTION_ERROR={error:.12g}", flush=True)
    if error > 2e-5:
        raise AssertionError(f"read polygons lost fragment membership: {error}")
    return result


def analytic_moments(source, polygons, centres):
    """Integrate analytic density with an oriented high-order triangle rule."""
    nodes, weights = np.polynomial.legendre.leggauss(15)
    u, v = np.meshgrid((nodes + 1) / 2, (nodes + 1) / 2, indexing="ij")
    wu, wv = np.meshgrid(weights / 2, weights / 2, indexing="ij")
    u, v, weight = u.ravel(), v.ravel(), (wu * wv).ravel()
    values = np.zeros((3, len(polygons)))
    for index, polygon in enumerate(polygons):
        if polygon.is_empty:
            continue
        if polygon.geom_type != "Polygon" or len(polygon.interiors):
            raise ValueError("analytic support is not one hole-free polygon")
        points = np.asarray(orient(polygon, sign=1.0).exterior.coords[:-1])
        first = points[0]
        edge_a, edge_b = points[1:-1] - first, points[2:] - first
        sample = first + u[None, :, None] * (
            (1 - v)[None, :, None] * edge_a[:, None]
            + v[None, :, None] * edge_b[:, None]
        )
        jacobian = edge_a[:, 0] * edge_b[:, 1] - edge_a[:, 1] * edge_b[:, 0]
        density = np.asarray(
            source.toroidal_current_density(sample[..., 0], sample[..., 1])
        )
        weighted = density * jacobian[:, None] * u[None, :] * weight[None, :]
        offset = sample - centres[index]
        values[0, index] = weighted.sum()
        values[1:, index] = (weighted[..., None] * offset).sum(axis=(0, 1))
    return fixture.CellCurrentMoments(*map(jnp.asarray, values))
