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
    """Return sub-panel wall flux coordinates, value and null type.

    The serial peer :func:`nova.geometry.select.wall_flux` computes the same
    sub-panel fit; the two are separate backends of one algorithm, not copies,
    and their return conventions DIFFER -- do not swap one for the other
    without adapting the call:

    * this one returns a 4-element ARRAY ``[x, z, psi, null_type]`` and takes
      ``polarity`` positionally; the serial one returns a 3-TUPLE ``(x, z,
      psi)`` with ``polarity=1`` defaulted and no type;
    * a zero polarity -- no plasma current, so no wall-limit point exists --
      returns all-NaN here and ``(0, 0, 0)`` there. NaN is the honest answer:
      ``(0, 0, 0)`` is a well-formed coordinate and flux, so a caller cannot
      tell it from a real result.

    The split is a backend difference the shapes force: this path is a
    fixed-shape reduction with no data-dependent branch, so it cannot return a
    different arity for the zero-polarity case and folds it into a NaN select
    instead.
    """
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


#: floor on the Hessian determinant of the fitted surface. A degenerate (planar)
#: cluster drives it to zero and the stationary point is the ratio it divides,
#: so an unguarded fit returns +-inf there -- which a vmapped reduction then
#: carries into every gradient that touches the batch. The null is reported as
#: degenerate through the type either way, so the coordinate is discarded; the
#: floor only keeps it finite on the way out.
_ROOT_FLOOR = 1e-30


@jax.jit
def null_type(coefficients, atol=1e-12):
    """Return null type.

        - 0: saddle
            :math:`4AB - E^2 < 0`
        - -1: minimum
            :math:`A>0` and :math:`B>0`
        - 1: maximum
            :math:`A<0` and :math:`B<0`
        - NaN: degenerate, :math:`|4AB - E^2| < atol` (a planar cluster)

    The four conditions are exhaustive, so the default is never selected:
    reaching it needs :math:`4AB - E^2 > atol > 0`, which forces
    :math:`AB > E^2/4 \\ge 0`, so ``A`` and ``B`` share a sign and one of the
    last two conditions has already fired.
    """
    root = 4 * coefficients[0] * coefficients[1] - coefficients[4] ** 2
    condlist = [
        abs(root) < atol,
        root < 0,
        (coefficients[0] > 0) & (coefficients[1] > 0),
        (coefficients[0] < 0) & (coefficients[1] < 0),
    ]
    choicelist = [jnp.nan, 0.0, -1.0, 1.0]
    return jax.numpy.select(condlist, choicelist, default=jnp.nan)


@jax.jit
def null_coordinate(coefficients):
    """
    Return null coodinates in 2D plane.

    Returns
    -------
    x_coordinate: float
        subgrid field null x_coordinate
    z_coordinate: float
        subgrid field null z_coordinate

    A degenerate cluster is reported through :func:`null_type` rather than here;
    the determinant is floored at :data:`_ROOT_FLOOR` so the coordinate stays
    finite and the batch it rides in keeps differentiable.
    """
    root = 4 * coefficients[0] * coefficients[1] - coefficients[4] ** 2
    root = jnp.where(
        jnp.abs(root) < _ROOT_FLOOR, jnp.sign(root) * _ROOT_FLOOR + _ROOT_FLOOR, root
    )
    x_coordinate = (
        coefficients[4] * coefficients[3] - 2 * coefficients[1] * coefficients[2]
    ) / root
    z_coordinate = (
        coefficients[4] * coefficients[2] - 2 * coefficients[0] * coefficients[3]
    ) / root
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
    coords = null_coordinate(coef)
    psi = null(coef, coords)
    return jnp.r_[coords, psi, ntype]
