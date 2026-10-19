"""The census retains every polarity axis, not only the one nearest the seed.

The gather hands the read its axis candidates: slot zero is the axis nearest
the previous root, the saddle slots follow, and the remaining axes fill what is
left. A read ranks those candidates by signed level, so an axis dropped from
the gather can never be chosen however strong it is.
"""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nova.equilibrium import topology
from nova.equilibrium.solve_request import TopologyPolicy
from nova.linalg.tensor_spline import fit_tensor_spline
from scripts.prototypes.efit_field_swap.spline_field import TraceSplineField
from scripts.prototypes.read_robustness.census import census

assert jax.config.jax_enable_x64 is True

SHALLOW = np.array([1.0, 0.0])
DEEP = np.array([3.0, 0.0])
SADDLE = np.array([2.0, 0.0])


def _unequal_wells(radial, vertical, tilt=0.2):
    """Two flux maxima of different level, separated by a saddle.

    ``-((t^2 - 1)^2) - y^2 + tilt * t`` with ``t = x - 2`` places maxima at
    ``t = +-1`` and a saddle at ``t = 0``; the tilt raises the ``t = +1`` well
    above the ``t = -1`` one, so the strongest axis is not the one nearest a
    seed placed on the shallow well.
    """
    r = np.asarray(radial)[None, :]
    z = np.asarray(vertical)[:, None]
    t = r - 2.0
    return -((t * t - 1.0) ** 2) - z * z + tilt * t


def well_field(geometry, radial, vertical):
    values = _unequal_wells(radial, vertical)
    field = TraceSplineField(
        spline=fit_tensor_spline(
            jnp.asarray(radial), jnp.asarray(vertical), jnp.asarray(values)
        )
    )
    return field.sampled_for(geometry)


@pytest.fixture(scope="module")
def geometry():
    cells = []
    samples = []
    for r in range(4):
        for z in range(4):
            low = np.array([float(r), float(z) - 2.0])
            cells.append(low + np.array([[0, 0], [1, 0], [1, 1], [0, 1]]))
            samples.append(
                low
                + np.array(
                    [
                        [0, 0],
                        [1, 0],
                        [1, 1],
                        [0, 1],
                        [0.5, 0],
                        [1, 0.5],
                        [0.5, 1],
                        [0, 0.5],
                    ]
                )
            )
    wall = np.array([[0, -2], [4, -2], [4, 2], [0, 2]], dtype=float)
    return topology.TopologyGeometry.from_cells(cells, np.asarray(samples), (wall,))


def previous_root(field, position):
    """The prior root positions the census continues identities from."""
    return jnp.asarray([position])


def retained_positions(found):
    return np.asarray(found.roots.position)[np.asarray(found.roots.valid)]


def test_census_retains_every_axis_not_only_the_seeded_one(geometry):
    field = well_field(geometry, np.linspace(0.0, 4.0, 33), np.linspace(-2.0, 2.0, 33))
    convention = topology.TopologyConvention(jnp.asarray(-1.0), jnp.asarray(1.0))
    found = census(
        field, geometry, convention, TopologyPolicy(), previous_root(field, SHALLOW)
    )
    assert int(found.axes) == 2, "both wells are polarity axes"
    retained = retained_positions(found)
    # Slot zero carries the axis nearest the prior root; the gather must also
    # carry the other one, which the read ranks by signed level and prefers.
    np.testing.assert_allclose(
        retained[0], SHALLOW, atol=0.05, err_msg="slot zero is the seeded axis"
    )
    near_deep = np.linalg.norm(retained - DEEP, axis=1) < 0.05
    assert near_deep.any(), (
        "the census dropped the axis it did not seed, so the read cannot rank it"
    )


def test_census_marks_the_retained_axes_valid_over_the_saddle(geometry):
    field = well_field(geometry, np.linspace(0.0, 4.0, 33), np.linspace(-2.0, 2.0, 33))
    convention = topology.TopologyConvention(jnp.asarray(-1.0), jnp.asarray(1.0))
    found = census(
        field, geometry, convention, TopologyPolicy(), previous_root(field, SHALLOW)
    )
    valid = np.asarray(found.roots.valid)
    eigen = np.linalg.eigvalsh(convention.sigma * np.asarray(found.roots.jet.hessian))
    positions = np.asarray(found.roots.position)
    deep_slot = int(np.argmin(np.linalg.norm(positions - DEEP, axis=1)))
    assert valid[deep_slot], "the unseeded axis is retained as a live slot"
    assert np.all(eigen[deep_slot] < 0), "and it carries the axis Hessian type"
