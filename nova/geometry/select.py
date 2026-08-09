"""Host and optionally traced selection algorithms."""

import numpy as np

from nova import njit


_ROOT_FLOOR = 1e-30

__all__ = [
    "bisect",
    "bisect_2d",
    "bisect_right",
    "host_quadratic_surface",
    "host_quadratic_wall",
    "host_subnull",
    "host_wall_flux",
    "host_wall_index",
    "length_2d",
    "null_coordinate",
    "null_flux",
    "null_type",
    "wall_coordinate",
    "wall_length",
]


@njit(cache=True)
def bisect(vector, value):
    """Return the left insertion index in a sorted host vector."""
    low, high = 0, len(vector)
    while low < high:
        middle = (low + high) // 2
        if vector[middle] < value:
            low = middle + 1
        else:
            high = middle
    return low


@njit(cache=True)
def bisect_right(vector, value):
    """Return the right insertion index in a sorted host vector."""
    low, high = 0, len(vector)
    while low < high:
        middle = (low + high) // 2
        if value < vector[middle]:
            high = middle
        else:
            low = middle + 1
    return low


@njit(cache=True)
def bisect_2d(vector, value):
    """Return right insertion indices for a host vector of values."""
    index = np.zeros(len(value), dtype=np.int16)
    for position in np.arange(len(value)):
        index[position] = bisect_right(vector, value[position])
    return index


def length_2d(x_coordinate, z_coordinate, *, array_namespace=np):
    """Return cumulative polyline length in the supplied array namespace."""
    points = array_namespace.stack((x_coordinate, z_coordinate), axis=-1)
    delta = array_namespace.sqrt(
        array_namespace.sum((points[1:] - points[:-1]) ** 2, axis=1)
    )
    return array_namespace.concatenate(
        (array_namespace.zeros(1, dtype=delta.dtype), array_namespace.cumsum(delta))
    )


@njit(cache=True)
def host_quadratic_wall(w_cluster, psi_cluster):
    """Fit host float64 wall-quadratic coefficients with GELSD."""
    coefficient_matrix = np.column_stack(
        (w_cluster**2, w_cluster, np.ones_like(w_cluster))
    )
    return np.linalg.lstsq(
        coefficient_matrix, psi_cluster.astype(np.float64), rcond=-1.0
    )[0]


def wall_length(coefficients, *, array_namespace=np):
    """Return the stationary wall length using backend-native division."""
    del array_namespace
    return -coefficients[1] / (2 * coefficients[0])


def wall_coordinate(
    coordinate,
    x_cluster,
    z_cluster,
    length_cluster,
    *,
    array_namespace=np,
):
    """Interpolate a wall coordinate in the supplied array namespace."""
    return (
        array_namespace.interp(coordinate, length_cluster, x_cluster),
        array_namespace.interp(coordinate, length_cluster, z_cluster),
    )


@njit(cache=True)
def host_wall_index(psi_wall):
    """Return the eager wall-extremum cluster index and any required roll."""
    index = np.argmax(psi_wall)
    if index == 0:
        return index + 1, 1
    if index == len(psi_wall) - 1:
        return index - 1, -1
    return index, 0


def host_wall_flux(x_wall, z_wall, psi_wall, polarity=1):
    """Return eager wall extremum ``[x, z, psi, type]``.

    Zero polarity returns four NaNs.  The host route retains eager mutation,
    NumPy NaN selection, float64/GELSD fitting, and its default polarity.
    """
    if polarity == 0:
        return np.full(4, np.nan)
    index, roll = host_wall_index(polarity * psi_wall)
    if roll != 0:
        x_wall = np.roll(x_wall, roll)
        z_wall = np.roll(z_wall, roll)
        psi_wall = np.roll(psi_wall, roll)
    x_cluster = x_wall[index - 1 : index + 2]
    z_cluster = z_wall[index - 1 : index + 2]
    psi_cluster = psi_wall[index - 1 : index + 2]
    length_cluster = length_2d(x_cluster, z_cluster)
    coefficients = host_quadratic_wall(length_cluster, psi_cluster)
    coordinate = wall_length(coefficients)
    psi = coefficients[0] * coordinate**2 + coefficients[1] * coordinate
    psi += coefficients[2]
    x_coordinate, z_coordinate = wall_coordinate(
        coordinate, x_cluster, z_cluster, length_cluster
    )
    if coefficients[0] > 0:
        kind = -1.0
    elif coefficients[0] < 0:
        kind = 1.0
    else:
        kind = np.nan
    return np.array([x_coordinate, z_coordinate, psi, kind])


@njit(cache=True)
def host_quadratic_surface(x_cluster, z_cluster, psi_cluster):
    """Fit host float64 surface-quadratic coefficients with GELSD."""
    coefficient_matrix = np.column_stack(
        (
            x_cluster**2,
            z_cluster**2,
            x_cluster,
            z_cluster,
            x_cluster * z_cluster,
            np.ones_like(x_cluster),
        )
    )
    return np.linalg.lstsq(
        coefficient_matrix, psi_cluster.astype(np.float64), rcond=-1.0
    )[0]


def null_type(coefficients, atol=1e-12, *, array_namespace=np):
    """Classify quadratic coefficients, returning NaN when degenerate."""
    determinant = 4 * coefficients[0] * coefficients[1] - coefficients[4] ** 2
    return array_namespace.where(
        array_namespace.abs(determinant) < atol,
        array_namespace.nan,
        array_namespace.where(
            determinant < 0,
            0.0,
            array_namespace.where(
                (coefficients[0] > 0) & (coefficients[1] > 0),
                -1.0,
                array_namespace.where(
                    (coefficients[0] < 0) & (coefficients[1] < 0),
                    1.0,
                    array_namespace.nan,
                ),
            ),
        ),
    )


def null_coordinate(
    coefficients,
    cluster=None,
    *,
    array_namespace=np,
):
    """Return the stationary coordinate of a fitted quadratic surface.

    The determinant floor preserves its sign.  Host composites additionally
    retain the loose two-cell support assertion; traced composites extrapolate.
    """
    determinant = 4 * coefficients[0] * coefficients[1] - coefficients[4] ** 2
    determinant = array_namespace.where(
        array_namespace.abs(determinant) < _ROOT_FLOOR,
        array_namespace.where(determinant < 0, -_ROOT_FLOOR, _ROOT_FLOOR),
        determinant,
    )
    x_coordinate = (
        coefficients[4] * coefficients[3] - 2 * coefficients[1] * coefficients[2]
    ) / determinant
    z_coordinate = (
        coefficients[4] * coefficients[2] - 2 * coefficients[0] * coefficients[3]
    ) / determinant
    if cluster is not None:
        for axis, coordinate in enumerate((x_coordinate, z_coordinate)):
            maximum = np.max(cluster[axis])
            minimum = np.min(cluster[axis])
            delta = maximum - minimum
            assert coordinate >= minimum - 2 * delta
            assert coordinate <= maximum + 2 * delta
    return x_coordinate, z_coordinate


def null_flux(coefficients, coordinates, *, array_namespace=np):
    """Evaluate fitted poloidal flux at a stationary coordinate."""
    basis = array_namespace.stack(
        (
            coordinates[0] ** 2,
            coordinates[1] ** 2,
            coordinates[0],
            coordinates[1],
            coordinates[0] * coordinates[1],
            array_namespace.ones_like(coordinates[0]),
        )
    )
    return basis @ coefficients


def host_subnull(x_cluster, z_cluster, psi_cluster):
    """Return eager subgrid null ``[x, z, psi, type]``."""
    coefficients = host_quadratic_surface(x_cluster, z_cluster, psi_cluster)
    kind = null_type(coefficients)
    coordinates = null_coordinate(coefficients, (x_cluster, z_cluster))
    psi = null_flux(coefficients, coordinates)
    return np.array([coordinates[0], coordinates[1], psi, kind])


try:
    import jax
    import jax.numpy as jnp
except ModuleNotFoundError as error:
    if error.name != "jax":
        raise
else:

    @jax.jit
    def _traced_argmax(values):
        """Return the first maximum index with a dtype-exact reduction seed."""
        values = jax.lax.stop_gradient(values)
        indices = jax.lax.broadcasted_iota(jnp.int32, values.shape, 0)
        initial = (
            jnp.asarray(-jnp.inf, dtype=values.dtype),
            jnp.asarray(values.size, dtype=jnp.int32),
        )

        def choose(left, right):
            left_value, left_index = left
            right_value, right_index = right
            take_right = (right_value > left_value) | (
                (right_value == left_value) & (right_index < left_index)
            )
            return (
                jnp.where(take_right, right_value, left_value),
                jnp.where(take_right, right_index, left_index),
            )

        return jax.lax.reduce((values, indices), initial, choose, dimensions=(0,))[1]

    @jax.jit
    def traced_quadratic_wall(w_cluster, psi_cluster):
        """Fit wall-quadratic coefficients with traced device semantics."""
        coefficient_matrix = jnp.column_stack(
            (w_cluster**2, w_cluster, jnp.ones_like(w_cluster))
        )
        return jnp.linalg.lstsq(coefficient_matrix, psi_cluster)[0]

    @jax.jit
    def traced_wall_index(psi_wall):
        """Return fixed-shape traced wall cluster index and roll."""
        valid = ~jnp.isnan(psi_wall)
        score = jnp.where(valid, psi_wall, jnp.asarray(-jnp.inf, psi_wall.dtype))
        index = jnp.where(jnp.any(valid), _traced_argmax(score), -1)
        offset = jnp.piecewise(
            index, [index == 0, index == len(psi_wall) - 1], [1, -1, 0]
        )
        return index + offset, offset

    @jax.jit
    def _traced_wall_cluster(index, roll, value):
        """Return three traced wall values around a selected index."""
        value = jnp.where(roll != 0, jnp.roll(value, roll), value)
        return jax.lax.dynamic_slice(value, [index - 1], 3)

    @jax.jit
    def traced_wall_flux(x_wall, z_wall, psi_wall, polarity):
        """Return traced wall extremum ``[x, z, psi, type]``."""
        index, roll = traced_wall_index(polarity * psi_wall)
        x_cluster = _traced_wall_cluster(index, roll, x_wall)
        z_cluster = _traced_wall_cluster(index, roll, z_wall)
        psi_cluster = _traced_wall_cluster(index, roll, psi_wall)
        length_cluster = length_2d(x_cluster, z_cluster, array_namespace=jnp)
        coefficients = traced_quadratic_wall(length_cluster, psi_cluster)
        coordinate = wall_length(coefficients, array_namespace=jnp)
        psi = coefficients[0] * coordinate**2 + coefficients[1] * coordinate
        psi += coefficients[2]
        x_coordinate, z_coordinate = wall_coordinate(
            coordinate,
            x_cluster,
            z_cluster,
            length_cluster,
            array_namespace=jnp,
        )
        kind = jnp.where(
            coefficients[0] > 0,
            -1.0,
            jnp.where(coefficients[0] < 0, 1.0, jnp.nan),
        )
        result = jnp.stack((x_coordinate, z_coordinate, psi, kind))
        return jnp.where(polarity != 0, result, jnp.full(4, jnp.nan))

    @jax.jit
    def traced_quadratic_surface(x_cluster, z_cluster, psi_cluster):
        """Fit surface-quadratic coefficients with traced device semantics."""
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
        return jnp.linalg.lstsq(coefficient_matrix, psi_cluster)[0]

    @jax.jit
    def traced_subnull(x_cluster, z_cluster, psi_cluster):
        """Return traced subgrid null ``[x, z, psi, type]``.

        Single-precision coordinate inputs must already be dimensionless local
        coordinates.  Physical-coordinate fits require explicit float64 inputs;
        :class:`nova.biot.null.Null2D` owns normalization and reconstruction.
        """
        coefficients = traced_quadratic_surface(x_cluster, z_cluster, psi_cluster)
        kind = null_type(coefficients, array_namespace=jnp)
        coordinates = null_coordinate(coefficients, array_namespace=jnp)
        psi = null_flux(coefficients, coordinates, array_namespace=jnp)
        return jnp.stack((coordinates[0], coordinates[1], psi, kind))

    __all__ += [
        "traced_quadratic_surface",
        "traced_quadratic_wall",
        "traced_subnull",
        "traced_wall_flux",
        "traced_wall_index",
    ]
