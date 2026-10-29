"""Near-horizontal polygon edges have stable field Hessians."""

from nova.jax.config import configure_dtypes

configure_dtypes()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from nova.biot import polygonanalytic as kernel  # noqa: E402
from nova.equilibrium.topology import BiotMomentCoupling  # noqa: E402


SOURCE = {
    ("split", 212): (
        (
            (4.075246269680356, -0.11739082463124054),
            (4.007470645479117, -0.23478164926248135),
            (3.871919397076639, -0.2347816492624813),
            (3.8041437728754, -0.1173908246312405),
            (3.8719193970766392, 3.0531133177191805e-16),
            (4.007470645479117, 2.914335439641036e-16),
        ),
        (-36.202074306735994, -2.3379459363543447, -0.01336038977460774),
    ),
    ("split", 234): (
        (
            (4.075246269680356, 0.11739082463124093),
            (4.007470645479117, 9.71445146547012e-17),
            (3.871919397076639, 1.8041124150158794e-16),
            (3.8041437728754, 0.11739082463124097),
            (3.8719193970766392, 0.23478164926248177),
            (4.007470645479117, 0.23478164926248174),
        ),
        (-36.20207430602841, -2.3379459363104047, 0.013360389700636688),
    ),
    ("integration", 212): (
        (
            (4.075246269680355, -0.11739082463124098),
            (4.007470645479117, -0.23478164926248185),
            (3.8719193970766383, -0.23478164926248174),
            (3.8041437728753995, -0.11739082463124094),
            (3.871919397076639, -1.1102230246251565e-16),
            (4.007470645479117, -1.249000902703301e-16),
        ),
        (-36.20207430967275, -2.3379459365446564, -0.01336038972254272),
    ),
    ("integration", 234): (
        (
            (4.075246269680355, 0.11739082463124048),
            (4.007470645479117, -3.7470027081099033e-16),
            (3.8719193970766383, -2.914335439641036e-16),
            (3.8041437728753995, 0.11739082463124052),
            (3.871919397076639, 0.23478164926248135),
            (4.007470645479117, 0.23478164926248135),
        ),
        (-36.20207431027029, -2.3379459365809794, 0.01336038978792331),
    ),
}


def _rows(point, coupling, *, primal=False):
    evaluate = (
        kernel._packed_analytic_moments_primal
        if primal
        else kernel.packed_analytic_moments
    )
    return evaluate(
        jnp,
        jnp.asarray([point[0]]),
        jnp.asarray([point[1]]),
        coupling.edge,
        coupling.weight,
        coupling.norm,
        coupling.centre.T,
        coupling.centre.T,
        coupling.reflection_axis,
        coupling.reflection_partner,
    )


def _field(point, coupling, coefficient):
    rows = _rows(point, coupling)
    radial = jnp.stack(rows[6:9], axis=-1)[0] @ coefficient
    vertical = -jnp.stack(rows[3:6], axis=-1)[0] @ coefficient
    return 2.0 * jnp.pi * point[0] * jnp.stack((radial, vertical))


def _check(source, cell, height):
    vertices, moments = SOURCE[source, cell]
    coupling = BiotMomentCoupling.from_polygons((np.asarray(vertices),))
    coefficient = jnp.asarray(
        np.concatenate(
            (moments[:1], np.asarray(coupling.inverse_second_moment)[0] @ moments[1:])
        )
    )
    point = jnp.asarray((6.176290619919, height))
    actual = np.asarray(
        jax.jacfwd(lambda target: _field(target, coupling, coefficient))(point)
    )
    differences = []
    for step in (1e-5, 5e-6):
        columns = []
        for direction in np.eye(2):
            offset = jnp.asarray(direction * step)
            columns.append(
                np.asarray(
                    (
                        _field(point + offset, coupling, coefficient)
                        - _field(point - offset, coupling, coefficient)
                    )
                    / (2 * step)
                )
            )
        differences.append(np.stack(columns, axis=-1))
    reference = (4.0 * differences[1] - differences[0]) / 3.0
    np.testing.assert_allclose(actual, reference, atol=1e-6, rtol=1e-6)
    for observed, original in zip(
        _rows(point, coupling), _rows(point, coupling, primal=True), strict=True
    ):
        assert np.asarray(observed).tobytes() == np.asarray(original).tobytes()


@pytest.mark.parametrize("source,cell", SOURCE)
def test_named_source_cell_hessians(source, cell):
    _check(source, cell, 0.0)


@pytest.mark.parametrize("offset", (0.0, 1e-16, 1e-12, 1e-8, 1e-4))
def test_height_sweep(offset):
    corner = SOURCE["integration", 212][0][4][1]
    _check("integration", 212, corner + offset)
