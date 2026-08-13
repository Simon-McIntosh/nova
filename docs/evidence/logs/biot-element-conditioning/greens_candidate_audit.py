from __future__ import annotations

import hashlib
import json
import math
import os
import time
import warnings
from decimal import Decimal, localcontext

import numpy as np
import scipy.integrate
import scipy.special

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)

from nova.biot.greens import (
    MU0,
    _ELLIPTIC_CONDITIONED_LIMIT,
    _SECTION_QUADRATURE_CLEARANCE,
    _SECTION_QUADRATURE_ORDER,
    _filament_elliptic_combinations,
    _section_quadrature_greens,
    _section_quadrature_mask,
    corner_fields,
    cylinder_greens,
    traced_corner_fields,
    traced_cylinder_greens,
    traced_filament_greens,
)
from nova.biot.completeelliptic import complete_kind
from nova.biot.zeta import (
    GAUSS_ORDER,
    NEAR_PLANE_RATIO,
    TANH_SINH_HALF_COUNT,
    _gauss_legendre_rule,
    _tanh_sinh_rule,
    traced_zeta,
    zeta,
)


def emit(kind, **payload):
    print(json.dumps({'kind': kind, **payload}, sort_keys=True, default=float), flush=True)


def component_gap(candidate, reference):
    candidate = np.asarray(candidate, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if candidate.ndim == 1:
        candidate = candidate[:, None]
        reference = reference[:, None]
    scale = np.maximum(np.max(np.abs(reference), axis=-1), np.finfo(float).tiny)
    return np.max(np.abs(candidate - reference), axis=-1) / scale


def stack(values):
    return np.stack([np.asarray(value, dtype=float) for value in values])


def gl_average(xp, parameters, order):
    target_r, target_z, a, z0, da, dz = parameters
    node, one_weight = np.polynomial.legendre.leggauss(order)
    node_r, node_z = (
        array.ravel()
        for array in np.meshgrid(0.5 * node, 0.5 * node, indexing='ij')
    )
    weight = (0.25 * one_weight[:, None] * one_weight[None, :]).ravel()
    node_r = xp.asarray(node_r)
    node_z = xp.asarray(node_z)
    weight = xp.asarray(weight)
    source_r = a + da * node_r
    source_z = z0 + dz * node_z
    values = traced_filament_greens(xp, target_r, target_z, source_r, source_z)
    return xp.stack([xp.sum(value * weight) for value in values])


def adaptive_average(parameters):
    target_r, target_z, a, z0, da, dz = map(float, parameters)
    result = []
    errors = []
    for component in range(3):
        def outer(u):
            def inner(v):
                values = traced_filament_greens(
                    np,
                    target_r,
                    target_z,
                    a + da * u,
                    z0 + dz * v,
                )
                return float(values[component])
            value, _ = scipy.integrate.quad(
                inner, -0.5, 0.5, epsabs=1e-20, epsrel=2e-13, limit=160
            )
            return value
        value, error = scipy.integrate.quad(
            outer, -0.5, 0.5, epsabs=1e-20, epsrel=2e-13, limit=160
        )
        result.append(value)
        errors.append(error)
    return np.asarray(result), np.asarray(errors)


def decimal_combinations(parameter, derivative=False):
    with localcontext() as context:
        context.prec = 140
        one = Decimal(1)
        two = Decimal(2)
        arithmetic_mean = one
        geometric_mean = (one / two).sqrt()
        correction = one / Decimal(4)
        weight = one
        for _ in range(10):
            next_mean = (arithmetic_mean + geometric_mean) / two
            geometric_mean = (arithmetic_mean * geometric_mean).sqrt()
            correction -= weight * (arithmetic_mean - next_mean) ** 2
            arithmetic_mean = next_mean
            weight *= two
        pi = (arithmetic_mean + geometric_mean) ** 2 / (Decimal(4) * correction)
        m = Decimal.from_float(float(parameter))
        complete_coefficient = one
        first_minus_second = Decimal(0)
        flux = Decimal(0)
        radial = Decimal(0)
        value_power = m
        derivative_power = one
        for degree in range(1, 2601):
            prior_coefficient = complete_coefficient
            ratio = Decimal(2 * degree - 1) / Decimal(2 * degree)
            complete_coefficient *= ratio * ratio
            power = derivative_power * degree if derivative else value_power
            first_minus_second += (
                complete_coefficient
                * Decimal(2 * degree)
                / Decimal(2 * degree - 1)
                * power
            )
            if degree >= 2:
                coefficient = prior_coefficient * Decimal(degree - 1) / Decimal(2 * degree)
                flux += coefficient * power
                radial += Decimal(2 * degree - 1) * coefficient * power
            value_power *= m
            derivative_power *= m
        scale = pi / two
        return np.asarray([float(scale * item) for item in (first_minus_second, flux, radial)])


def corner_stack(a, z0, da, dz, target_r, target_z):
    target_r = np.asarray(target_r)
    target_z = np.asarray(target_z)
    source_r = np.stack(
        [np.full(target_r.shape, a + sign * da / 2) for sign in (-1, 1, 1, -1)],
        axis=-1,
    )
    source_z = np.stack(
        [np.full(target_z.shape, z0 + sign * dz / 2) for sign in (-1, -1, 1, 1)],
        axis=-1,
    )
    return (
        source_r,
        source_z,
        np.repeat(target_r[..., None], 4, axis=-1),
        np.repeat(target_z[..., None], 4, axis=-1),
    )


def corner_only(xp, parameters):
    target_r, target_z, a, z0, da, dz = parameters
    tr = xp.asarray(target_r)
    tz = xp.asarray(target_z)
    one = xp.ones_like(tr)
    source_r = xp.stack([(a + sign * da / 2) * one for sign in (-1, 1, 1, -1)], axis=-1)
    source_z = xp.stack([(z0 + sign * dz / 2) * one for sign in (-1, -1, 1, 1)], axis=-1)
    target_r4 = xp.stack([tr for _ in range(4)], axis=-1)
    target_z4 = xp.stack([tz for _ in range(4)], axis=-1)
    data = traced_corner_fields(xp, source_r, source_z, target_r4, target_z4)
    area = da * dz
    def reduce_one(value):
        return ((value[..., 2] - value[..., 3]) - (value[..., 1] - value[..., 0])) / (2 * np.pi * area)
    return xp.stack((
        2 * np.pi * MU0 * tr * reduce_one(data[0]),
        MU0 * reduce_one(data[1]),
        MU0 * reduce_one(data[2]),
    ))


def direct_only(xp, parameters):
    return xp.stack(_section_quadrature_greens(xp, *parameters))


def public(xp, parameters):
    return xp.stack(traced_cylinder_greens(xp, *parameters))


def zeta_audit():
    assert GAUSS_ORDER == 48
    assert len(_gauss_legendre_rule()[0]) == 48
    assert len(_tanh_sinh_rule()[0]) == 2 * TANH_SINH_HALF_COUNT + 1 == 177
    integer_host = np.asarray(traced_zeta(np, 2, 1, 1, 1))
    integer_expected = np.asarray(zeta(2.0, 1.0, 1.0, 1.0))
    mixed_args = (np.float32(2), 1, np.float32(1), np.float32(1))
    mixed_host = np.asarray(traced_zeta(np, *mixed_args))
    mixed_expected = np.asarray(zeta(*(np.asarray(x, np.float32) for x in mixed_args)))
    emit(
        'zeta_dtype',
        gauss_order=GAUSS_ORDER,
        tanh_nodes=177,
        integer_dtype=str(integer_host.dtype),
        integer_gap=float(abs(integer_host - integer_expected) / abs(integer_expected)),
        mixed_dtype=str(mixed_host.dtype),
        mixed_gap=float(abs(mixed_host - mixed_expected) / abs(mixed_expected)),
    )

    rs = np.asarray([0.0, 2.0, -3.0, 1.4])
    radius = np.asarray([0.0, 1.0, 0.0, 0.8])
    gamma = np.asarray([0.0, 0.3, 0.0, -0.2])
    alpha = np.asarray([0.0, np.pi / 2, 0.0, 0.0])
    with warnings.catch_warnings(record=True) as caught, np.errstate(
        divide='raise', invalid='raise', over='raise', under='warn'
    ):
        host = np.asarray(traced_zeta(np, rs, radius, gamma, alpha))
    geometry = jnp.asarray(np.stack((rs, radius, gamma, alpha)))
    def evaluate(g):
        return traced_zeta(jnp, g[0], g[1], g[2], g[3])
    primal, tangent = jax.jit(lambda g: jax.jvp(evaluate, (g,), (jnp.ones_like(g),)))(geometry)
    primal_vjp, pullback = jax.vjp(evaluate, geometry)
    reverse = pullback(jnp.ones_like(primal_vjp))[0]
    zero_lanes = np.asarray(alpha == 0)
    emit(
        'zeta_zero_lanes',
        warnings=len(caught),
        host=host.tolist(),
        primal=np.asarray(primal).tolist(),
        jvp_zero_lanes=np.asarray(tangent)[zero_lanes].tolist(),
        vjp_zero_lane_columns=np.asarray(reverse)[:, zero_lanes].tolist(),
        all_finite=bool(np.all(np.isfinite(primal)) and np.all(np.isfinite(tangent)) and np.all(np.isfinite(reverse))),
    )


def coherent_zeta_and_cylinder_audit():
    cases = [
        ('thick', 0.45, 0.35),
        ('thin', 0.01, 0.008),
        ('radial_plate', 0.5, 0.002),
        ('vertical_plate', 0.002, 0.5),
    ]
    for label, da, dz in cases:
        a, z0 = 6.2, -0.13
        target_r0 = a + 0.31 * da
        switch = NEAR_PLANE_RATIO * target_r0
        boundaries = (z0 - dz / 2 + switch, z0 + dz / 2 + switch)
        positive = np.asarray([
            value
            for boundary in boundaries
            for value in (np.nextafter(boundary, -np.inf), boundary, np.nextafter(boundary, np.inf))
        ])
        target_z = np.concatenate((2 * z0 - positive[::-1], positive))
        target_r = np.full(target_z.shape, target_r0)
        source_r, source_z, radius4, level4 = corner_stack(a, z0, da, dz, target_r, target_z)
        gamma4 = source_z - level4
        host_zeta = zeta(source_r, radius4, gamma4, np.pi / 2, coherent_axis=-1)
        traced_np = np.asarray(traced_zeta(np, source_r, radius4, gamma4, np.pi / 2, coherent_axis=-1))
        traced_jax = np.asarray(jax.jit(lambda sr, rr, gg: traced_zeta(jnp, sr, rr, gg, np.pi / 2, coherent_axis=-1))(
            jnp.asarray(source_r), jnp.asarray(radius4), jnp.asarray(gamma4)
        ))
        adaptive = np.empty_like(host_zeta)
        for index in np.ndindex(host_zeta.shape):
            sr = float(source_r[index]); rr = float(radius4[index]); gg = float(gamma4[index])
            def integrand(x):
                phi = np.pi - 2 * x
                return np.arcsinh((sr - rr * np.cos(phi)) / np.sqrt(gg * gg + rr * rr * np.sin(phi) ** 2))
            adaptive[index] = scipy.integrate.quad(
                integrand, 0.0, np.pi / 2, epsabs=2e-13, epsrel=2e-13, limit=300
            )[0]
        host = stack(cylinder_greens(target_r, target_z, a, z0, da, dz))
        twin_np = stack(traced_cylinder_greens(np, target_r, target_z, a, z0, da, dz))
        twin_jax = np.asarray(jax.jit(lambda rr, zz: public(jnp, (rr, zz, a, z0, da, dz)))(jnp.asarray(target_r), jnp.asarray(target_z)))
        gl24 = np.stack([gl_average(np, params, 24) for params in zip(target_r, target_z, np.full_like(target_r,a), np.full_like(target_r,z0), np.full_like(target_r,da), np.full_like(target_r,dz))], axis=-1)
        route = np.asarray(_section_quadrature_mask(np, target_r, target_z, a, z0, da, dz))
        emit(
            'coherent_case', label=label, da=da, dz=dz,
            near_rows=int(np.sum(np.any(np.abs(gamma4) < NEAR_PLANE_RATIO * np.abs(radius4), axis=-1))),
            direct_rows=int(route.sum()), corner_rows=int((~route).sum()),
            zeta_np_gap=float(np.max(component_gap(traced_np.ravel(), host_zeta.ravel()))),
            zeta_jax_gap=float(np.max(component_gap(traced_jax.ravel(), host_zeta.ravel()))),
            zeta_adaptive_gap=float(np.max(component_gap(host_zeta.ravel(), adaptive.ravel()))),
            public_np_gap=component_gap(twin_np, host).tolist(),
            public_jax_gap=component_gap(twin_jax, host).tolist(),
            public_gl24_gap=component_gap(host, gl24).tolist(),
        )


def direct_reference_and_derivative_audit(start_case=0):
    cases = [
        np.asarray([6.2 + 0.05 + 8.5 * math.hypot(0.05, 0.04), 0.01, 6.2, 0.0, 0.1, 0.08]),
        np.asarray([6.2, 0.001 + 8.5 * math.hypot(0.25, 0.001), 6.2, 0.0, 0.5, 0.002]),
        np.asarray([5.8, -0.25 - 8.5 * math.hypot(0.001, 0.25), 6.2, 0.0, 0.002, 0.5]),
    ]
    directions = [
        np.asarray([0.3, -0.2, 0.1, 0.05, 0.01, -0.02]),
        np.asarray([-0.15, 0.25, 0.08, -0.03, 0.015, 0.01]),
        np.asarray([0.11, -0.17, -0.04, 0.06, -0.005, 0.02]),
    ]
    for index, (parameters, direction) in enumerate(zip(cases, directions, strict=True)):
        if index < start_case:
            continue
        assert bool(_section_quadrature_mask(np, *parameters))
        with warnings.catch_warnings(record=True) as caught, np.errstate(all='raise'):
            candidate = stack(cylinder_greens(*parameters))[:, 0] if np.asarray(cylinder_greens(*parameters)[0]).ndim else stack(cylinder_greens(*parameters)).reshape(3)
            direct = np.asarray(direct_only(np, parameters)).reshape(3)
            gl12 = np.asarray(gl_average(np, parameters, 12))
            gl24 = np.asarray(gl_average(np, parameters, 24))
            adaptive, adaptive_error = adaptive_average(parameters)

        def candidate_jax(p):
            return public(jnp, tuple(p))
        def refined_jax(p):
            return gl_average(jnp, p, 24)
        c_value, c_jvp = jax.jit(lambda p, d: jax.jvp(candidate_jax, (p,), (d,)))(jnp.asarray(parameters), jnp.asarray(direction))
        r_value, r_jvp = jax.jit(lambda p, d: jax.jvp(refined_jax, (p,), (d,)))(jnp.asarray(parameters), jnp.asarray(direction))
        _, pullback = jax.vjp(candidate_jax, jnp.asarray(parameters))
        vjp = pullback(jnp.asarray([0.7, -0.2, 0.5]))[0]
        emit(
            'direct_reference', case=index, parameters=parameters.tolist(), warnings=len(caught),
            candidate_direct_gap=component_gap(candidate, direct).tolist(),
            gl6_gl12_gap=component_gap(direct, gl12).tolist(),
            gl6_gl24_gap=component_gap(direct, gl24).tolist(),
            gl6_adaptive_gap=component_gap(direct, adaptive).tolist(),
            adaptive_error=adaptive_error.tolist(),
            jax_value_gl24_gap=component_gap(np.asarray(c_value), np.asarray(r_value)).tolist(),
            jax_jvp_gl24_gap=component_gap(np.asarray(c_jvp), np.asarray(r_jvp)).tolist(),
            finite_vjp=bool(np.all(np.isfinite(np.asarray(vjp)))),
            vjp=np.asarray(vjp).tolist(),
        )


def bracket_route(parameters, axis):
    p = np.asarray(parameters, dtype=float)
    direction = 1.0 if p[axis] >= p[2 + (axis == 1)] else -1.0
    # The caller gives a nominal threshold point. Find adjacent representable
    # points whose route booleans differ without assuming exact algebra survives.
    current = float(p[axis])
    mask = bool(_section_quadrature_mask(np, *p))
    toward = -np.inf if direction > 0 else np.inf
    away = np.inf if direction > 0 else -np.inf
    inner = current
    outer = current
    if mask:
        for _ in range(10000):
            candidate = np.nextafter(inner, toward)
            p[axis] = candidate
            if not bool(_section_quadrature_mask(np, *p)):
                return candidate, inner
            inner = candidate
    else:
        for _ in range(10000):
            candidate = np.nextafter(outer, away)
            p[axis] = candidate
            if bool(_section_quadrature_mask(np, *p)):
                return outer, candidate
            outer = candidate
    raise RuntimeError('could not bracket route')


def route_seam_audit():
    a, z0 = 6.2, -0.07
    geometries = [
        ('regular', 0.1, 0.08),
        ('radial_plate', 0.5, 0.002),
        ('vertical_plate', 0.002, 0.5),
    ]
    bearings = [('radial', 1.0, 0.0), ('vertical', 0.0, 1.0), ('three_four', 0.6, 0.8)]
    for label, da, dz in geometries:
        halfdiag = math.hypot(da / 2, dz / 2)
        clearance = _SECTION_QUADRATURE_CLEARANCE * halfdiag
        for bearing, ur, uz in bearings:
            for sign_r, sign_z in ((1.0, 1.0), (-1.0, 1.0), (1.0, -1.0)):
                target_r = a + sign_r * (da / 2 + clearance * ur)
                target_z = z0 + sign_z * (dz / 2 + clearance * uz)
                nominal = np.asarray([target_r, target_z, a, z0, da, dz])
                axis = 0 if abs(ur) >= abs(uz) else 1
                low_coordinate, high_coordinate = bracket_route(nominal, axis)
                low = nominal.copy(); low[axis] = low_coordinate
                high = nominal.copy(); high[axis] = high_coordinate
                assert not bool(_section_quadrature_mask(np, *low))
                assert bool(_section_quadrature_mask(np, *high))
                normal = np.zeros(6)
                normal[0] = sign_r * ur
                normal[1] = sign_z * uz
                with warnings.catch_warnings(record=True) as caught, np.errstate(all='raise'):
                    low_np = np.asarray(public(np, low)).reshape(3)
                    high_np = np.asarray(public(np, high)).reshape(3)
                    corner_at_high = np.asarray(corner_only(np, high)).reshape(3)
                    direct_at_high = np.asarray(direct_only(np, high)).reshape(3)
                def evaluate(p):
                    return public(jnp, tuple(p))
                low_jax, low_slope = jax.jit(lambda p, d: jax.jvp(evaluate, (p,), (d,)))(jnp.asarray(low), jnp.asarray(normal))
                high_jax, high_slope = jax.jit(lambda p, d: jax.jvp(evaluate, (p,), (d,)))(jnp.asarray(high), jnp.asarray(normal))
                delta_position = np.linalg.norm(high[:2] - low[:2])
                low_slope = np.asarray(low_slope).reshape(3)
                high_slope = np.asarray(high_slope).reshape(3)
                physical = 0.5 * (low_slope + high_slope) * delta_position
                observed = high_np - low_np
                artificial = observed - physical
                scale = np.maximum(np.maximum(np.abs(low_np), np.abs(high_np)), np.finfo(float).tiny)
                slope_scale = np.maximum(np.maximum(np.abs(low_slope), np.abs(high_slope)), np.finfo(float).tiny)
                emit(
                    'route_seam', geometry=label, bearing=bearing, signs=[sign_r, sign_z], axis=axis,
                    warnings=len(caught), delta_position=delta_position,
                    low_mask=False, high_mask=True,
                    public_numpy_jax_low=component_gap(np.asarray(low_jax).reshape(3), low_np).tolist(),
                    public_numpy_jax_high=component_gap(np.asarray(high_jax).reshape(3), high_np).tolist(),
                    arm_gap_relative=(np.abs(direct_at_high - corner_at_high) / scale).tolist(),
                    adjacent_jump_relative=(np.abs(observed) / scale).tolist(),
                    artificial_relative=(np.abs(artificial) / scale).tolist(),
                    artificial_to_physical=(np.abs(artificial) / np.maximum(np.abs(physical), np.finfo(float).tiny)).tolist(),
                    jvp_side_relative=(np.abs(high_slope - low_slope) / slope_scale).tolist(),
                    low_jvp=low_slope.tolist(), high_jvp=high_slope.tolist(),
                )


def held_arm_and_contract_audit():
    a, z0, da, dz = 6.2, -0.13, 0.1, 0.08
    target_r = np.asarray([0.0, np.nextafter(0.0, 1.0), a - da / 2, a + da / 2, a, a, a])
    target_z = np.asarray([z0, z0 + 0.31, z0 - dz / 2, z0 + dz / 2, z0, z0 + dz / 2, z0 - dz / 2])
    with warnings.catch_warnings(record=True) as caught, np.errstate(all='warn'):
        host = stack(cylinder_greens(target_r, target_z, a, z0, da, dz))
        twin = stack(traced_cylinder_greens(np, target_r, target_z, a, z0, da, dz))
    geometry = jnp.asarray(np.stack((target_r, target_z)))
    def evaluate(g):
        return public(jnp, (g[0], g[1], a, z0, da, dz))
    primal, tangent = jax.jit(lambda g: jax.jvp(evaluate, (g,), (jnp.ones_like(g),)))(geometry)
    _, pullback = jax.vjp(evaluate, geometry)
    reverse = pullback(jnp.ones_like(primal))[0]
    emit(
        'held_contracts', warnings=len(caught), warning_messages=[str(item.message) for item in caught],
        host_finite=bool(np.all(np.isfinite(host))), twin_finite=bool(np.all(np.isfinite(twin))),
        jax_primal_finite=bool(np.all(np.isfinite(np.asarray(primal)))),
        jax_jvp_finite=bool(np.all(np.isfinite(np.asarray(tangent)))),
        jax_vjp_finite=bool(np.all(np.isfinite(np.asarray(reverse)))),
        numpy_gap=component_gap(twin, host).tolist(), jax_gap=component_gap(np.asarray(primal), host).tolist(),
        axis_psi=host[0, :2].tolist(), axis_br=host[1, :2].tolist(),
    )


def elliptic_audit():
    smallest = np.nextafter(0.0, 1.0)
    parameters = np.asarray([
        smallest, 2 * smallest, 3 * smallest, 4 * smallest,
        np.nextafter(1e-300, np.inf), 1e-200, 1e-50, 1e-16, 1e-12,
        0.5, np.nextafter(_ELLIPTIC_CONDITIONED_LIMIT, 0.0),
        _ELLIPTIC_CONDITIONED_LIMIT, np.nextafter(_ELLIPTIC_CONDITIONED_LIMIT, np.inf), 0.99,
    ])
    complement = 1.0 - parameters
    first = scipy.special.ellipk(parameters)
    second = scipy.special.ellipe(parameters)
    host = np.asarray(_filament_elliptic_combinations(np, parameters, complement, first, second)).T
    expected = np.asarray([decimal_combinations(value) for value in parameters])
    def combinations(m):
        first_jax, second_jax = complete_kind(1.0 - m, xp=jnp)
        return jnp.stack(_filament_elliptic_combinations(jnp, m, 1.0 - m, first_jax, second_jax), axis=-1)
    jax_value, jax_jvp = jax.jit(lambda m: jax.jvp(combinations, (m,), (jnp.ones_like(m),)))(jnp.asarray(parameters))
    expected_jvp = np.asarray([decimal_combinations(value, derivative=True) for value in parameters])
    _, pullback = jax.vjp(combinations, jnp.asarray(parameters))
    reverse = pullback(jnp.ones_like(jax_value))[0]
    emit(
        'elliptic_combinations', parameter_count=len(parameters),
        host_gap=component_gap(host.T, expected.T).tolist(),
        jax_gap=component_gap(np.asarray(jax_value).T, expected.T).tolist(),
        jvp_gap=component_gap(np.asarray(jax_jvp).T, expected_jvp.T).tolist(),
        vjp_finite=bool(np.all(np.isfinite(np.asarray(reverse)))),
        least_subnormal_host=host[:4].tolist(), least_subnormal_expected=expected[:4].tolist(),
        leading_units=(host[:4, 0] / smallest).tolist(),
        boundary_values=host[-4:].tolist(), boundary_jvp=np.asarray(jax_jvp)[-4:].tolist(),
    )


def elliptic_targeted_audit():
    smallest = np.nextafter(0.0, 1.0)
    parameters = np.asarray([
        smallest, 2 * smallest, 3 * smallest, 4 * smallest,
        1e-300, 1e-200, 1e-50, 1e-16, 1e-12, 0.5,
        np.nextafter(_ELLIPTIC_CONDITIONED_LIMIT, 0.0),
        _ELLIPTIC_CONDITIONED_LIMIT,
        np.nextafter(_ELLIPTIC_CONDITIONED_LIMIT, np.inf),
    ])
    complement = 1.0 - parameters
    first = scipy.special.ellipk(parameters)
    second = scipy.special.ellipe(parameters)
    host = np.asarray(_filament_elliptic_combinations(np, parameters, complement, first, second)).T
    expected = np.asarray([decimal_combinations(value) for value in parameters])
    def combinations(m):
        first_jax, second_jax = complete_kind(1.0 - m, xp=jnp)
        return jnp.stack(_filament_elliptic_combinations(jnp, m, 1.0 - m, first_jax, second_jax), axis=-1)
    jax_value, jax_jvp = jax.jit(lambda m: jax.jvp(combinations, (m,), (jnp.ones_like(m),)))(jnp.asarray(parameters))
    expected_jvp = np.asarray([decimal_combinations(value, derivative=True) for value in parameters])
    for index, parameter in enumerate(parameters):
        value_scale = np.maximum(np.abs(expected[index]), np.finfo(float).tiny)
        jvp_scale = np.maximum(np.abs(expected_jvp[index]), np.finfo(float).tiny)
        emit('elliptic_point', parameter=float(parameter), host=host[index].tolist(), expected=expected[index].tolist(),
             jax=np.asarray(jax_value)[index].tolist(), jvp=np.asarray(jax_jvp)[index].tolist(), expected_jvp=expected_jvp[index].tolist(),
             host_relative=(np.abs(host[index]-expected[index])/value_scale).tolist(),
             jax_relative=(np.abs(np.asarray(jax_value)[index]-expected[index])/value_scale).tolist(),
             jvp_relative=(np.abs(np.asarray(jax_jvp)[index]-expected_jvp[index])/jvp_scale).tolist())


def cpu_and_shape_audit():
    parameters_corner = (np.linspace(6.0, 6.4, 256), np.linspace(-0.1, 0.1, 256), 6.2, 0.0, 0.1, 0.08)
    parameters_direct = (np.linspace(7.0, 8.0, 256), np.linspace(0.8, 1.2, 256), 6.2, 0.0, 0.1, 0.08)
    def best_time(function, arguments, repeat=5):
        function(*arguments)
        samples=[]
        for _ in range(repeat):
            start=time.perf_counter(); function(*arguments); samples.append(time.perf_counter()-start)
        return min(samples), samples
    corner_t, corner_samples = best_time(lambda *p: corner_only(np, p), parameters_corner)
    direct_t, direct_samples = best_time(lambda *p: direct_only(np, p), parameters_direct)
    public_corner_t, public_corner_samples = best_time(lambda *p: public(np, p), parameters_corner)
    public_direct_t, public_direct_samples = best_time(lambda *p: public(np, p), parameters_direct)
    def scalar(rr, zz):
        return public(jnp, (rr, zz, 6.2, 0.0, 0.1, 0.08))
    def batch(rr, zz):
        return public(jnp, (rr, zz, 6.2, 0.0, 0.1, 0.08))
    start=time.perf_counter(); scalar_compiled=jax.jit(scalar).lower(jnp.asarray(7.0), jnp.asarray(0.9)).compile(); scalar_compile=time.perf_counter()-start
    start=time.perf_counter(); batch_compiled=jax.jit(batch).lower(jnp.ones(16)*7.0, jnp.ones(16)*0.9).compile(); batch_compile=time.perf_counter()-start
    scalar_result=np.asarray(scalar_compiled(jnp.asarray(7.0), jnp.asarray(0.9)))
    batch_result=np.asarray(batch_compiled(jnp.ones(16)*7.0, jnp.ones(16)*0.9))
    scalar_jaxpr=str(jax.make_jaxpr(scalar)(jnp.asarray(7.0), jnp.asarray(0.9)))
    batch_jaxpr=str(jax.make_jaxpr(batch)(jnp.ones(16)*7.0, jnp.ones(16)*0.9))
    emit(
        'cpu_cost_and_static_shape', backend=jax.default_backend(), target_count=256,
        corner_best_s=corner_t, direct_best_s=direct_t,
        direct_over_corner=direct_t/corner_t,
        public_corner_best_s=public_corner_t, public_direct_best_s=public_direct_t,
        corner_samples_s=corner_samples, direct_samples_s=direct_samples,
        public_corner_samples_s=public_corner_samples, public_direct_samples_s=public_direct_samples,
        scalar_compile_s=scalar_compile, batch16_compile_s=batch_compile,
        scalar_shape=list(scalar_result.shape), batch16_shape=list(batch_result.shape),
        scalar_jaxpr_sha256=hashlib.sha256(scalar_jaxpr.encode()).hexdigest(),
        batch16_jaxpr_sha256=hashlib.sha256(batch_jaxpr.encode()).hexdigest(),
        scalar_jaxpr_chars=len(scalar_jaxpr), batch16_jaxpr_chars=len(batch_jaxpr),
        gl6_order=_SECTION_QUADRATURE_ORDER,
    )


def main():
    emit('environment', python=os.sys.version, numpy=np.__version__, scipy=scipy.__version__, jax=jax.__version__, backend=jax.default_backend())
    zeta_audit()
    coherent_zeta_and_cylinder_audit()
    direct_reference_and_derivative_audit()
    route_seam_audit()
    held_arm_and_contract_audit()
    elliptic_audit()
    cpu_and_shape_audit()
    emit('audit_complete', status='completed')


if __name__ == '__main__':
    if os.environ.get('AUDIT_RESUME') == '3':
        elliptic_targeted_audit()
        emit('audit_complete', status='targeted_elliptic_completed')
    elif os.environ.get('AUDIT_RESUME') == '2':
        held_arm_and_contract_audit()
        elliptic_audit()
        cpu_and_shape_audit()
        emit('audit_complete', status='completed_after_contract_resume')
    elif os.environ.get('AUDIT_RESUME') == '1':
        direct_reference_and_derivative_audit(start_case=2)
        route_seam_audit()
        held_arm_and_contract_audit()
        elliptic_audit()
        cpu_and_shape_audit()
        emit('audit_complete', status='completed_after_harness_resume')
    else:
        main()
