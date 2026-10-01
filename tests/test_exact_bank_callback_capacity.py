"""The flux-selected callback sizes its cut-cell bank to the mesh.

The whole-cell branch of the traced current moment path integrates every cell
at an inherited cut-cell capacity of one, then a flux-selected profile
re-derives a per-cell quadratic level boundary and calls the same integrator
back in. That re-derived boundary is a mesh object: a level crossing more than
one cell marks more cut cells than the inherited capacity can hold, the
integrator fails closed, and every returned moment becomes non-finite. The
callback must therefore carry at least the number of cells in its own call.

The fixture is a two-cell support whose quadratic level crosses both cells in
their interior, so it exercises exactly the case the inherited capacity one
cannot hold.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nova.equilibrium.clip_quadrature import (
    _quadratic_support,
    clipped_support_current_moments,
)
from nova.equilibrium.source import (
    DomainProfile,
    PolynomialFluxFunction,
    _FluxSelectedProfile,
)
from nova.equilibrium.stencil_mesh import FluxFieldPolynomial
from nova.jax.config import configure_dtypes


class _Support(NamedTuple):
    support_vertices: jax.Array
    vertex_count: jax.Array
    centroids: jax.Array
    included: jax.Array
    boundary: jax.Array


#: Cell centres 4 m apart, each unit square. A quadratic level whose inside
#: region is ``local_radial < 0.5`` crosses every cell through its interior, so
#: two cells of the support become cut cells at the same time. Both centres sit
#: at positive major radius: the toroidal current density carries a ``1/R``
#: term, so a cell spanning the axis would pole for a reason unrelated to the
#: bank capacity under test.
_CELL_CENTRES = ((2.0, 0.0), (6.0, 0.0))
_SQUARE = np.asarray([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
_PADDED_CAPACITY = 8

#: ``inside_coefficient = -field.coefficient`` with the constant raised by one.
#: This field therefore gives the level ``0.5 - local_radial``, which is
#: positive inside and crosses each cell's unit square once.
_FIELD_COEFFICIENT = (0.5, 1.0, 0.0, 0.0, 0.0, 0.0)


@pytest.fixture(scope="module", autouse=True)
def _binary64_cpu() -> None:
    configure_dtypes()
    assert jax.config.jax_enable_x64
    assert jax.default_backend() == "cpu"


def _two_cell_support() -> _Support:
    vertices = np.zeros((len(_CELL_CENTRES), _PADDED_CAPACITY, 2), dtype=np.float64)
    for cell, centre in enumerate(_CELL_CENTRES):
        vertices[cell, : len(_SQUARE)] = _SQUARE + np.asarray(centre)
    return _Support(
        support_vertices=jnp.asarray(vertices),
        vertex_count=jnp.full(len(_CELL_CENTRES), len(_SQUARE), dtype=jnp.int32),
        centroids=jnp.asarray(_CELL_CENTRES, dtype=jnp.float64),
        included=jnp.ones(len(_CELL_CENTRES), dtype=bool),
        boundary=jnp.ones(len(_CELL_CENTRES), dtype=bool),
    )


def _crossing_field() -> FluxFieldPolynomial:
    cell_count = len(_CELL_CENTRES)
    return FluxFieldPolynomial(
        coefficient=jnp.tile(jnp.asarray([_FIELD_COEFFICIENT]), (cell_count, 1)),
        centre=jnp.asarray(_CELL_CENTRES, dtype=jnp.float64),
        scale=jnp.ones((cell_count, 2)),
        active=jnp.ones(cell_count, dtype=bool),
    )


def _confined_profile() -> _FluxSelectedProfile:
    return _FluxSelectedProfile(
        confined=DomainProfile(
            p_prime=PolynomialFluxFunction(
                coefficients=jnp.asarray([0.0]), normalisation=jnp.asarray(1.0)
            ),
            ff_prime=PolynomialFluxFunction(
                coefficients=jnp.asarray([1.0]), normalisation=jnp.asarray(1.0)
            ),
        ),
        open_field_line=None,
    )


def test_quadratic_level_crosses_both_cells():
    """Premise: the fixture level really cuts two cells, not zero or one."""
    support = _two_cell_support()
    field = _crossing_field()
    coefficient = -jnp.asarray(field.coefficient)
    coefficient = coefficient.at[:, 0].add(1.0)
    clipped = _quadratic_support(
        jnp.asarray(support.support_vertices),
        jnp.asarray(support.vertex_count),
        jnp.asarray(support.centroids),
        coefficient,
        field.centre,
        field.scale,
        support.included,
    )
    boundary = np.asarray(clipped.boundary, dtype=bool)
    assert int(np.count_nonzero(boundary)) == len(_CELL_CENTRES)


def test_inherited_capacity_one_would_overflow_the_rederived_bank():
    """Premise: the re-derived cut count exceeds the inherited capacity one."""
    support = _two_cell_support()
    field = _crossing_field()
    coefficient = -jnp.asarray(field.coefficient)
    coefficient = coefficient.at[:, 0].add(1.0)
    clipped = _quadratic_support(
        jnp.asarray(support.support_vertices),
        jnp.asarray(support.vertex_count),
        jnp.asarray(support.centroids),
        coefficient,
        field.centre,
        field.scale,
        support.included,
    )
    cut_count = int(np.count_nonzero(np.asarray(clipped.boundary, dtype=bool)))
    assert cut_count > 1


def test_flux_selected_moments_size_their_own_bank_to_the_moments():
    """Every returned moment is finite and the fixture carries a known current."""
    support = _two_cell_support()
    moments = clipped_support_current_moments(
        support,
        support.included,
        _crossing_field(),
        _confined_profile(),
        cut_cell_capacity=1,
        boundary_reduction=True,
    )
    values = np.asarray(moments)
    assert np.all(np.isfinite(values))
    # A finite all-zero result would also pass a finiteness check, so the
    # fixture must be shown to carry current before finiteness means anything.
    current = np.asarray(moments.cell_current)
    assert np.all(np.abs(current) > 1.0)
