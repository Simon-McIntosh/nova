"""Analytic checks for the independent quadratic-cell integral."""

from scripts.prototypes.xpoint_cell_owner.conic_integral import integrate_cell


POLYGON = ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0))
CENTRE = (0.0, 0.0)
SADDLE = (0.0, 0.0)
AXIS = (1.0, 0.0)
CONSTANT_DENSITY = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def _integral(offset, **options):
    return integrate_cell(
        POLYGON,
        CENTRE,
        1.0,
        (offset, 0.0, 0.0, 1.0, 0.0, -1.0),
        0.0,
        1.0,
        SADDLE,
        AXIS,
        CONSTANT_DENSITY,
        **options,
    )


def test_axis_facing_hyperbola_has_analytic_area_and_first_moment():
    value = _integral(0.0)
    assert abs(value.area - 1.0) < 1e-12
    assert abs(value.current - 1.0) < 1e-12
    assert abs(value.radial_moment - 2 / 3) < 1e-12
    assert abs(value.vertical_moment) < 1e-12


def test_positive_critical_value_joins_the_quadratic_lobes():
    half = _integral(0.1)
    connected = _integral(0.1, axis_component="bell")
    assert abs(connected.area - 2 * half.area) < 1e-12
    assert abs(connected.radial_moment) < 1e-12


def test_moving_conic_root_breaks_independent_quadrature_agreement():
    reference = _integral(-0.1, outer_rule="dense")
    measured = _integral(-0.1)
    perturbed = _integral(-0.1, branch_shift=1e-4)
    assert abs(measured.area - reference.area) / reference.area < 1e-12
    assert abs(perturbed.vertical_moment - reference.vertical_moment) > 1e-6


def test_traced_connected_component_matches_independent_integral():
    from nova.jax.config import configure_dtypes

    configure_dtypes()
    import jax
    import jax.numpy as jnp

    from scripts.prototypes.xpoint_cell_owner.device_owner import (
        integrate_cell as traced_integral,
    )

    reference = _integral(0.1, axis_component="bell")
    moments, valid = jax.jit(
        lambda offset: traced_integral(
            jnp.asarray(POLYGON),
            jnp.asarray(len(POLYGON)),
            jnp.asarray(CENTRE),
            jnp.asarray(1.0),
            jnp.asarray((offset, 0.0, 0.0, 1.0, 0.0, -1.0)),
            jnp.asarray(0.0),
            jnp.asarray(1.0),
            jnp.asarray(SADDLE),
            jnp.asarray(AXIS),
            jnp.asarray(CONSTANT_DENSITY),
            axis_component="bell",
        )
    )(0.1)
    assert bool(valid)
    assert abs(float(moments[0]) - reference.area) < 1e-12
    assert abs(float(moments[1]) - reference.current) < 1e-12
    assert abs(float(moments[2]) - reference.radial_moment) < 1e-12
