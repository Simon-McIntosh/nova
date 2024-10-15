"""Manage selection algorithms."""

import jax
import jax.numpy as jnp


@jax.jit
def bisect(vector, value):
    """Return the bisect left index, assuming vector is sorted.

    The return index i is such that all e in vector[:i] have e < value,
    and all e in vector[i:] have e >= value.

    Addapted from bisect.bisect_left to enable jit compilation.
    """

    def cond(val):
        low, high = val
        return low < high

    def body(val):
        low, high = val
        mid = (low + high) // 2
        low = jnp.where(vector[mid] < value, mid + 1, low)
        high = jnp.where(vector[mid] >= value, mid, high)
        return [low, high]

    low, high = jax.lax.while_loop(cond, body, [0, len(vector)])
    return low


def bisect_right(vector, value):
    """Return the bisect right index, assuming vector is sorted.

    The return value i is such that all e in vector[:i] have e <= value,
    and all e in vector[i:] have e > value.

    Addapted from bisect.bisect_right to enable jit compilation.
    """
    low, high = 0, len(vector)
    while low < high:
        mid = (low + high) // 2
        if value < vector[mid]:
            high = mid
        else:
            low = mid + 1
    return low


def bisect_2d(vector, value):
    """Return vector of bisection values."""
    number = len(value)
    index = jnp.zeros(number, dtype=jnp.int16)
    for i in jnp.arange(number):
        index[i] = bisect_right(vector, value[i])
    return index


@jax.jit
def length_2d(x_coordinate, z_coordinate):
    """Return the cumalative length of a 2d polyline."""
    points = jnp.column_stack((x_coordinate, z_coordinate))
    delta = jnp.sqrt(jnp.sum((points[1:] - points[:-1]) ** 2, axis=1))
    return jnp.append(0, delta.cumsum())


@jax.jit
def quadratic_wall(w_cluster, psi_cluster):
    """Return psi quatratic coefficients."""
    coefficient_matrix = jnp.column_stack(
        (w_cluster**2, w_cluster, jnp.ones_like(w_cluster))
    )
    coefficients = jnp.linalg.lstsq(coefficient_matrix, psi_cluster)[0]
    return coefficients


@jax.jit
def wall_length(coef):
    """Return location of wall null."""
    return -coef[1] / (2 * coef[0])


@jax.jit
def wall_coordinate(w_coordinate, x_cluster, z_cluster, w_cluster):
    """Return wall coordinates."""
    x_coordinate = jnp.interp(w_coordinate, w_cluster, x_cluster)
    z_coordinate = jnp.interp(w_coordinate, w_cluster, z_cluster)
    return x_coordinate, z_coordinate


@jax.jit
def wall_index(psi_wall):
    """Return cluster index and roll."""
    index = jnp.nanargmax(psi_wall)
    offset = jnp.piecewise(index, [index == 0, index == len(psi_wall) - 1], [1, -1, 0])
    return index + offset, offset


@jax.jit
def wall_cluster(index, roll, value):
    """Return cluster of 3 points bounding wall index."""
    value = jnp.where(roll != 0, jnp.roll(value, roll), value)
    return jax.lax.dynamic_slice(value, [index - 1], 3)


@jax.jit
def wall_flux(x_wall, z_wall, psi_wall, polarity):
    """Return sub-panel wall flux coordinates and value."""
    index, roll = wall_index(polarity * psi_wall)
    x_cluster = wall_cluster(index, roll, x_wall)
    z_cluster = wall_cluster(index, roll, z_wall)
    psi_cluster = wall_cluster(index, roll, psi_wall)
    w_cluster = length_2d(x_cluster, z_cluster)
    coef = quadratic_wall(w_cluster, psi_cluster)
    w_coordinate = wall_length(coef)
    psi = coef[0] * w_coordinate**2 + coef[1] * w_coordinate + coef[2]
    x_coordinate, z_coordinate = wall_coordinate(
        w_coordinate, x_cluster, z_cluster, w_cluster
    )
    condlist = [
        coef[0] > 0,
        coef[0] < 0,
    ]
    choicelist = [-1, 1]
    null_type = jax.numpy.select(condlist, choicelist, default=jnp.nan)
    return jnp.where(
        polarity != 0,
        jnp.r_[x_coordinate, z_coordinate, psi, null_type],
        jnp.nan * jnp.ones(4),
    )


@jax.jit
def quadratic_surface(x_cluster, z_cluster, psi_cluster):
    """Return psi quatratic surface coefficients."""
    coefficient_matrix = jnp.column_stack(
        (
            x_cluster**2,
            z_cluster**2,
            x_cluster,
            z_cluster,
            x_cluster * z_cluster,
            jnp.ones_like(x_cluster),
        )
    )
    coefficients = jnp.linalg.lstsq(coefficient_matrix, psi_cluster)[0]
    return coefficients


def _check_ntype(ntype):
    """Raise null type errors."""
    if ntype == -10:
        raise ValueError("Plane surface")
    if ntype == -11:
        raise ValueError("Coefficients form a degenerate surface.")


@jax.jit
def null_type(coefficients, atol=1e-12):
    """Return null type.

        - 0: saddle
            :math:`4AB - E^2 < 0`
        - -1: minimum
            :math:`A>0` and :math:`B>0`
        - 1: maximum
            :math:`A<0` and :math:`B<0`

    Raises
    ------
    ValueError
        degenerate surface
    """
    root = 4 * coefficients[0] * coefficients[1] - coefficients[4] ** 2
    condlist = [
        abs(root) < atol,
        root < 0,
        (coefficients[0] > 0) & (coefficients[1] > 0),
        (coefficients[0] < 0) & (coefficients[1] < 0),
    ]
    choicelist = [jnp.nan, 0, -1, 1]
    return jax.numpy.select(condlist, choicelist, default=-11)


@jax.jit
def null_coordinate(coefficients, cluster=None):
    """
    Return null coodinates in 2D plane.

    Returns
    -------
    x_coordinate: float
        subgrid field null x_coordinate
    z_coordinate: float
        subgrid field null z_coordinate

    Raises
    ------
    ValueError
        subgrid coordinate outside cluster
    """
    root = 4 * coefficients[0] * coefficients[1] - coefficients[4] ** 2
    x_coordinate = (
        coefficients[4] * coefficients[3] - 2 * coefficients[1] * coefficients[2]
    ) / root
    z_coordinate = (
        coefficients[4] * coefficients[2] - 2 * coefficients[0] * coefficients[3]
    ) / root
    """    # TODO reimplement error checking with jax callbacks
    if cluster is not None:
        for i, coord in enumerate([x_coordinate, z_coordinate]):
            maximum, minimum = jnp.max(cluster[i]), jnp.min(cluster[i])
            delta = maximum - minimum
            # assert coord >= minimum - 2 * delta
            # assert coord <= maximum + 2 * delta
    """
    return x_coordinate, z_coordinate


@jax.jit
def null(coef, coords):
    """Return null poloidal flux."""
    return (
        jnp.array(
            [
                coords[0] ** 2,
                coords[1] ** 2,
                coords[0],
                coords[1],
                coords[0] * coords[1],
                1,
            ]
        )
        @ coef
    )


@jax.jit
def subnull(cluster):
    """Return subgrid null coordinates, value, and type.

    Parameters
    ----------
    cluster: jnp.ndarray (3, N)
        Cluster coordinates and flux values [x, z, psi].
    """
    coef = quadratic_surface(*cluster)
    ntype = null_type(coef)
    coords = null_coordinate(coef, cluster[:2])
    psi = null(coef, coords)
    return jnp.r_[coords, psi, ntype]
