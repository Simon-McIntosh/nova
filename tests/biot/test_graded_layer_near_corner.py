"""Resolve the logarithmic end layer of a ring corner integral."""

import jax
import jax.numpy as jnp
import mpmath as mp
import numpy as np
import os
import pytest

from nova.jax.config import configure_dtypes

configure_dtypes()
assert jax.config.jax_enable_x64 is True

import nova.biot.gradedresidual as gradedresidual  # noqa: E402
from tangent_identity import load_base_module  # noqa: E402

BASE_GRADED = load_base_module(
    "nova/biot/gradedresidual.py", "646b68b8e9481178e41cc2059c8a9a534fa9b9b3"
)


@pytest.fixture(autouse=True)
def restore_floor_when_requested(monkeypatch):
    """Restore the clipped layer for the declared negative control."""
    if os.environ.get("NOVA_RESTORE_LAYER_FLOOR") != "1":
        return

    def clipped_width(offset, end, scale, xp):
        reach = xp.where(offset > 0.0, offset, xp.abs(end))
        return gradedresidual._fixed_nodes(
            xp.where(
                reach > 0.0,
                xp.clip(reach / scale, gradedresidual.LAYER_FLOOR, 1.0),
                1.0,
            ),
            xp,
        )

    monkeypatch.setattr(gradedresidual, "_grading_width", clipped_width)


def _residual(radius, level, radial_offset, module=gradedresidual):
    span = 2.0 * radius
    zero = jnp.zeros_like(radial_offset)
    limit = jnp.full_like(radial_offset, gradedresidual.QUARTER)
    panels = (
        (jnp.abs(level), radial_offset + span, span, zero, limit),
        (jnp.abs(level), radial_offset, span, zero, limit),
    )

    def pieces(x, y):
        numerator = radial_offset[:, None] + span[:, None] * y
        denominator = jnp.sqrt(level[:, None] ** 2 + span[:, None] ** 2 * x * y)
        return numerator, denominator

    return module.graded_residual(panels, pieces, 128, jnp)


def _reference(radius, level, radial_offset):
    """Integrate the exact integral and its radial-offset derivative at 50 digits."""
    with mp.workdps(50):
        r, u, o = (mp.mpf(value) for value in (radius, level, radial_offset))
        limit = mp.pi / 2
        near = [mp.mpf(10) ** exponent for exponent in range(-15, 0)]
        points = [mp.mpf(0), *near, limit / 2]
        points += [limit - point for point in reversed(near)]
        points += [limit]

        def terms(angle):
            sine, cosine = mp.sin(angle), mp.cos(angle)
            numerator = o + 2 * r * cosine**2
            denominator_squared = u**2 + 4 * r**2 * sine**2 * cosine**2
            return numerator, denominator_squared

        value = mp.quad(
            lambda angle: mp.asinh(terms(angle)[0] / mp.sqrt(terms(angle)[1])),
            points,
        )
        derivative = mp.quad(
            lambda angle: 1 / mp.sqrt(terms(angle)[0] ** 2 + terms(angle)[1]),
            points,
        )
    return mp.nstr(value, 50), mp.nstr(derivative, 50)


def test_near_corner_layer_value_and_derivative():
    radius, level = 1.2, 1e-12
    offsets = [0.0]
    offsets.extend(
        sign * 10.0**exponent for exponent in range(-12, -2) for sign in (-1.0, 1.0)
    )
    offsets.extend((-1e-2, 1e-2))
    r = jnp.full(len(offsets), radius)
    u = jnp.full(len(offsets), level)
    o = jnp.asarray(offsets)
    value, derivative = jax.jvp(
        lambda offset: _residual(r, u, offset), (o,), (jnp.ones_like(o),)
    )
    old = np.asarray(_residual(r, u, o, BASE_GRADED))
    value, derivative = np.asarray(value), np.asarray(derivative)
    by_decade = {}
    less_accurate = []
    outside = []
    for index, offset in enumerate(offsets):
        with mp.workdps(50):
            exact_value, exact_derivative = (
                mp.mpf(number) for number in _reference(radius, level, offset)
            )
            value_error = float(
                abs(mp.mpf(float(value[index])) - exact_value) / abs(exact_value)
            )
            derivative_error = float(
                abs(mp.mpf(float(derivative[index])) - exact_derivative)
                / abs(exact_derivative)
            )
            base_error = float(
                abs(mp.mpf(float(old[index])) - exact_value) / abs(exact_value)
            )
        decade = "exact" if offset == 0.0 else f"{abs(offset):.0e}"
        by_decade.setdefault(decade, []).append(
            (value_error, derivative_error, base_error)
        )
        if abs(offset) > 1e-3:
            outside_error = abs(value[index] - old[index]) / abs(old[index])
            outside.append((offset, outside_error))
        elif value_error > base_error:
            less_accurate.append((offset, value_error, base_error))
    maxima = {
        decade: np.max(np.asarray(errors), axis=0)
        for decade, errors in by_decade.items()
    }
    for decade, (value_error, derivative_error, base_error) in maxima.items():
        print(
            f"CORNER decade={decade} value_relative_max={value_error:.3e} "
            f"derivative_relative_max={derivative_error:.3e} "
            f"base_value_relative_max={base_error:.3e}"
        )
    print(f"CORNER less_accurate={less_accurate} outside={outside}")
    assert all(
        np.isfinite(value_error) and value_error <= 1e-9
        for value_error, _, _ in maxima.values()
    )
    assert all(
        np.isfinite(derivative_error) and derivative_error <= 1e-9
        for _, derivative_error, _ in maxima.values()
    )
    assert not less_accurate
    assert all(error <= 1e-13 for _, error in outside)


def test_moving_panel_bounds_follow_endpoint_integrand():
    radius, level, offset = 1.2, 1e-12, 1e-12

    def integral(bounds):
        def array(value):
            return jnp.asarray([value])

        panel = (
            array(level),
            array(offset + 2 * radius),
            array(2 * radius),
            bounds[:1],
            bounds[1:],
        )
        return gradedresidual.graded_residual(
            (panel,),
            lambda x, y: (
                offset + 2 * radius * y,
                jnp.sqrt(level**2 + 4 * radius**2 * x * y),
            ),
            128,
            jnp,
        )[0]

    bounds = jnp.asarray([0.1, 0.6])
    direction = jnp.asarray([0.3, -0.7])
    _, derivative = jax.jvp(integral, (bounds,), (direction,))
    sine, cosine = np.sin(np.asarray(bounds)), np.cos(np.asarray(bounds))
    integrand = np.arcsinh(
        (offset + 2 * radius * cosine**2)
        / np.sqrt(level**2 + 4 * radius**2 * sine**2 * cosine**2)
    )
    expected = integrand[1] * float(direction[1]) - integrand[0] * float(direction[0])
    np.testing.assert_allclose(float(derivative), expected, rtol=1e-13, atol=0)


def _program_floor(function, primals, tangents, select=lambda value: value):
    """Measure base roundoff between standalone and primal-plus-JVP programs."""
    standalone = jax.jit(function).lower(*primals).compile()
    differentiated = (
        jax.jit(lambda p, t: jax.jvp(function, p, t)).lower(primals, tangents).compile()
    )
    alone = np.asarray(select(standalone(*primals)))
    together = np.asarray(select(differentiated(primals, tangents)[0]))
    assert np.isfinite(alone).all() and np.isfinite(together).all()
    scale = np.where(alone == 0.0, 1.0, np.abs(alone))
    difference = np.abs(alone - together) / scale
    # The same comparison must detect a known present one-ULP difference.
    control = np.abs(alone - np.nextafter(alone, np.inf)) / scale
    assert np.all(control > 0.0)
    print(f"PROGRAM_FLOOR positive_control_min={control.min():.17g}")
    return alone, together, difference


def _decade_floor(offsets, differences):
    """Use each offset decade's measured maximum, without a fitted tolerance."""
    decades = np.floor(np.log10(np.abs(offsets))).astype(int)
    floors = np.empty_like(differences)
    for decade in np.unique(decades):
        mask = decades == decade
        floors[mask] = differences[mask].max()
    return decades, floors


def test_base_program_roundoff_floor():
    """Rejudge retained exact-reference errors with an independently measured floor."""
    import json
    from pathlib import Path
    import test_vertex_term_tangents as cohort

    receipt_directory = os.environ.get("NOVA_CORNER_FLOOR_DIRECTORY")
    if receipt_directory is None:
        pytest.skip("requires the retained near-corner reference receipts")
    assert jax.default_backend() == "gpu"
    assert "H200" in jax.devices()[0].device_kind
    directory = Path(receipt_directory)
    isolated = json.loads((directory / "graded-cohort-measurement.json").read_text())
    composed = json.loads((directory / "original-measurement.json").read_text())
    _, _, primals, tangents = cohort.CASES["arsinh_terms"]
    r, z, cr, cz = map(np.asarray, primals[:4])
    index = np.flatnonzero(cohort._near_corner(primals) & (cz != z) & (cr != r))
    assert index.tolist() == composed["indices"] == [row["index"] for row in isolated]
    assert index.size == 1189
    inputs = np.asarray([row["inputs"] for row in isolated])
    np.testing.assert_array_equal(
        inputs[:, :3].T, [r[index], (cz - z)[index], (cr - r)[index]]
    )
    args = tuple(jnp.asarray(value) for value in inputs[:, :3].T)
    directions = tuple(jnp.asarray(value) for value in inputs[:, 3:].T)
    print(
        f"FLOOR_PROVENANCE cwd={Path.cwd().resolve()} "
        f"graded_module={BASE_GRADED.__file__} "
        f"composed_module={cohort.BASE.__file__} device={jax.devices()[0].device_kind}"
    )
    programs = {
        "isolated": _program_floor(
            lambda *p: _residual(*p, module=BASE_GRADED), args, directions
        ),
        "composed": _program_floor(
            lambda *p: cohort.BASE._Vertex(
                *p[:4], 128, residual=True, xp=jnp
            ).arsinh_terms(),
            primals,
            tangents,
            select=lambda value: value[1][index],
        ),
    }
    output = {
        "base_revision": "646b68b8e9481178e41cc2059c8a9a534fa9b9b3",
        "device": jax.devices()[0].device_kind,
        "indices": index.tolist(),
        "arms": {},
    }
    for name, (alone, together, difference) in programs.items():
        errors = np.asarray(
            [row["value_error"] for row in isolated]
            if name == "isolated"
            else composed["value_error"]
        )
        base_errors = np.asarray(
            [row["base_error"] for row in isolated]
            if name == "isolated"
            else composed["base_error"]
        )
        decades, floor = _decade_floor(cr[index] - r[index], difference)
        strict = errors > base_errors
        degraded = errors > base_errors + floor
        assert int(strict.sum()) == (169 if name == "isolated" else 177)
        summary = []
        for decade in np.unique(decades):
            selected = decades == decade
            row = {
                "decade": int(decade),
                "samples": int(selected.sum()),
                "floor": float(floor[selected].max()),
                "strict_degraded": int(strict[selected].sum()),
                "degraded": int(degraded[selected].sum()),
            }
            summary.append(row)
            print(
                f"ROUNDING_FLOOR arm={name} "
                + " ".join(f"{k}={v}" for k, v in row.items())
            )
        print(
            f"ROUNDING_SUMMARY arm={name} samples={index.size} "
            f"strict_degraded={strict.sum()} degraded={degraded.sum()} "
            f"floor_max={floor.max():.17g}"
        )
        output["arms"][name] = {
            "standalone": alone.tolist(),
            "jvp_primal": together.tolist(),
            "differences": difference.tolist(),
            "decade_floor": floor.tolist(),
            "summary": summary,
            "degraded": int(degraded.sum()),
        }
    (directory / "program-floor-h200.json").write_text(json.dumps(output))
