"""Fixed-shape Bell-cell area and profile moments from one conic cut."""

import jax
import jax.numpy as jnp
from numpy.polynomial.legendre import leggauss


_OUTER_NODES, _OUTER_WEIGHTS = (jnp.asarray(row) for row in leggauss(64))
_INNER_NODES = jnp.asarray((-1 / jnp.sqrt(3.0), 1 / jnp.sqrt(3.0)))


def _quadratic(coefficients, x, y):
    a, b, c, d, e, f = coefficients
    return a + b * x + c * y + d * x * x + e * x * y + f * y * y


def _roots(a, b, c):
    radical = jnp.sqrt(jnp.maximum(b * b - 4 * a * c, 0.0))
    safe_a = jnp.where(jnp.abs(a) > 1e-14, a, 1.0)
    safe_b = jnp.where(jnp.abs(b) > 1e-14, b, 1.0)
    quadratic = jnp.stack(
        ((-b - radical) / (2 * safe_a), (-b + radical) / (2 * safe_a))
    )
    linear = -c / safe_b
    roots = jnp.where(jnp.abs(a) > 1e-14, quadratic, linear)
    valid = jnp.where(
        jnp.abs(a) > 1e-14,
        b * b - 4 * a * c >= 0,
        jnp.abs(b) > 1e-14,
    )
    return roots, valid


def integrate_cell(
    vertices,
    count,
    centre,
    pitch,
    field_coefficients,
    boundary_level,
    sigma,
    saddle,
    axis,
    density_coefficients,
):
    """Replace the sampled saddle-sector booking for one containing cell."""
    field = field_coefficients.at[0].add(-boundary_level) * sigma
    hessian = jnp.array(((2 * field[3], field[4]), (field[4], 2 * field[5])))
    eigenvalues, eigenvectors = jnp.linalg.eigh(hessian)
    plus = eigenvectors[:, 1]
    plus *= jnp.where(jnp.dot(axis - saddle, plus) < 0, -1.0, 1.0)
    minus = eigenvectors[:, 0]
    polygon = (vertices - saddle) / pitch
    polygon = jnp.stack((polygon @ plus, polygon @ minus), axis=1)
    slot = jnp.arange(vertices.shape[0])
    following = jnp.where(slot + 1 < count, slot + 1, 0)
    end = polygon[following]
    live = slot < count

    def value(u, v):
        u, v = jnp.asarray(u), jnp.asarray(v)
        local = (saddle - centre) / pitch + u[..., None] * plus + v[..., None] * minus
        return _quadratic(field, local[..., 0], local[..., 1])

    constant = value(0.0, 0.0)
    plus_value, minus_value = value(1.0, 0.0), value(-1.0, 0.0)
    upper, lower = value(0.0, 1.0), value(0.0, -1.0)
    linear_u = (plus_value - minus_value) / 2
    linear_v = (upper - lower) / 2
    squared_u = (plus_value + minus_value) / 2 - constant
    squared_v = (upper + lower) / 2 - constant
    mixed = value(1.0, 1.0) - plus_value - upper + constant
    vertex_u = jnp.where(live, polygon[:, 0], jnp.inf)
    lower_u = jnp.maximum(0.0, jnp.min(vertex_u))
    upper_u = jnp.max(jnp.where(live, polygon[:, 0], -jnp.inf))
    upper_u = jnp.maximum(lower_u, upper_u)

    start_value = value(polygon[:, 0], polygon[:, 1])
    end_value = value(end[:, 0], end[:, 1])
    middle = (polygon + end) / 2
    middle_value = value(middle[:, 0], middle[:, 1])
    edge_squared = 2 * (start_value + end_value - 2 * middle_value)
    edge_linear = end_value - start_value - edge_squared
    edge_roots, edge_valid = _roots(edge_squared, edge_linear, start_value)
    edge_u = polygon[:, 0, None] + edge_roots.T * (end[:, 0] - polygon[:, 0])[:, None]
    edge_live = (
        live[:, None] & edge_valid[:, None] & (edge_roots.T > 0) & (edge_roots.T < 1)
    )
    discriminant_squared = mixed * mixed - 4 * squared_v * squared_u
    discriminant_linear = 2 * linear_v * mixed - 4 * squared_v * linear_u
    discriminant_constant = linear_v * linear_v - 4 * squared_v * constant
    tangent_roots, tangent_valid = _roots(
        discriminant_squared, discriminant_linear, discriminant_constant
    )
    candidates = jnp.concatenate(
        (
            jnp.array((lower_u, upper_u)),
            jnp.where(live, polygon[:, 0], upper_u),
            jnp.where(edge_live, edge_u, upper_u).reshape(-1),
            jnp.where(tangent_valid, tangent_roots, upper_u),
        )
    )
    candidates = (
        jnp.where((candidates > lower_u) & (candidates < upper_u), candidates, upper_u)
        .at[0]
        .set(lower_u)
    )
    bounds = jnp.sort(candidates)
    left, right = bounds[:-1], bounds[1:]
    theta = (_OUTER_NODES + 1) * (jnp.pi / 4)
    u = left[:, None] + (right - left)[:, None] * jnp.sin(theta) ** 2
    outer_weights = (
        (right - left)[:, None] * jnp.sin(2 * theta) * (jnp.pi / 4) * _OUTER_WEIGHTS
    )

    def one_slice(u_value):
        span = end[:, 0] - polygon[:, 0]
        safe_span = jnp.where(jnp.abs(span) > 1e-14, span, 1.0)
        edge_fraction = (u_value - polygon[:, 0]) / safe_span
        cross = (
            live & (jnp.abs(span) > 1e-14) & (edge_fraction >= 0) & (edge_fraction < 1)
        )
        crossing_v = polygon[:, 1] + edge_fraction * (end[:, 1] - polygon[:, 1])
        low = jnp.min(jnp.where(cross, crossing_v, jnp.inf))
        high = jnp.max(jnp.where(cross, crossing_v, -jnp.inf))
        valid_span = jnp.isfinite(low) & jnp.isfinite(high)
        low, high = jnp.where(valid_span, low, 0.0), jnp.where(valid_span, high, 0.0)
        v_roots, root_valid = _roots(
            squared_v,
            linear_v + mixed * u_value,
            constant + linear_u * u_value + squared_u * u_value * u_value,
        )
        v_roots = jnp.where(root_valid, v_roots, high)
        v_roots = jnp.clip(v_roots, low, high)
        cuts = jnp.sort(jnp.concatenate((jnp.array((low, high)), v_roots)))
        v_left, v_right = cuts[:-1], cuts[1:]
        middle_v = (v_left + v_right) / 2
        positive = value(u_value, middle_v) > 0
        v_nodes = middle_v[:, None] + (v_right - v_left)[:, None] * _INNER_NODES / 2
        local = (saddle - centre) / pitch + u_value * plus + v_nodes[..., None] * minus
        x, y = local[..., 0], local[..., 1]
        density = _quadratic(density_coefficients, x, y)
        terms = jnp.stack(
            (jnp.ones_like(density), density, pitch * density * x, pitch * density * y),
            axis=-1,
        )
        weights = jnp.where(positive, (v_right - v_left) / 2, 0.0)
        return jnp.sum(terms * weights[:, None, None], axis=(0, 1))

    values = jax.vmap(one_slice)(u.reshape(-1)).reshape(*u.shape, 4)
    moments = pitch * pitch * jnp.sum(values * outer_weights[..., None], axis=(0, 1))
    valid = (eigenvalues[0] < 0) & (eigenvalues[1] > 0) & jnp.all(jnp.isfinite(moments))
    return moments, valid
