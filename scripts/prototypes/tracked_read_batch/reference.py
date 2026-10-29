"""The analytic boundary reference for a certificate state, kept out of tests.

The reference is a direct evaluation of the analytic flux alone: a diverted
state's boundary is the saddle returned by the certificate, and a limited
state's is the wall point where that flux is extremal, selected and refined by
:func:`nova.geometry.select.host_wall_flux` over the first-wall units the
geometry already carries. Nothing here consults a read, and nothing here
imports a test module, so a prototype and a test can share one reference.
"""

from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp

from nova.geometry import select


def psi(analytic, point):
    """The analytic flux at one point, as a float."""
    return float(np.asarray(analytic.evaluate(jnp.asarray(point)).value))


def wall_flux(analytic, wall):
    """The analytic flux at every wall vertex, in the wall's own order."""
    wall = np.asarray(wall, dtype=float).reshape(-1, 2)
    return np.asarray(
        jax.vmap(lambda point: analytic.evaluate(point).value)(jnp.asarray(wall))
    )


def analytic_boundary(case, references):
    """The boundary point and its flux, from the analytic field alone.

    A diverted state's boundary is the saddle of the certificate. A limited
    state's is the extremal flux on the wall: the outermost closed surface is
    the one that touches the wall at the largest flux, so the wall polygon's
    analytic flux is handed to ``host_wall_flux`` with the polarity that
    selects that maximum, and the returned point and flux are read back.
    """
    analytic = case["analytic"]
    if len(references) > 1:
        return references[1], psi(analytic, references[1])

    wall = np.asarray(case["geometry"].wall_points, dtype=float).reshape(-1, 2)
    values = wall_flux(analytic, wall)
    x_coordinate, z_coordinate, flux, _ = select.host_wall_flux(
        wall[:, 0], wall[:, 1], values, 1
    )
    point = np.asarray((x_coordinate, z_coordinate))
    return point, psi(analytic, point)
