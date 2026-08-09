"""Device-resident critical points of a scalar field on a rectangular grid.

Candidate existence is the signed degree of the native finite-difference
gradient around each rectangular cell.  Confidence evidence can therefore
defer an uncertain nonzero-degree cell, but cannot make it disappear.  Every
candidate is fitted in exact dimensionless stencil coordinates before its
sub-cell offset is combined with the physical grid metadata.

The batch API is the production kernel.  Scalar axis and X-point helpers are
thin fixed-shape adapters for the reconstruction and connectivity consumers.
The older scalar-ring diagnostic remains available for audit comparison, but
does not participate in candidate generation.
"""

from __future__ import annotations

from functools import lru_cache, partial

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np

from nova.equilibrium.morphology import _dilate4

STATE_ABSENT = 0
STATE_UNRESOLVED = 1
STATE_RESOLVED = 2

BOUNDARY_SNR_THRESHOLD = 5.0
CLASS_MARGIN_THRESHOLD = 16.0
ROOT_SUPPORT_LIMIT = 1.0
LOCAL_DESIGN_CONDITION = 4.242640687119286

__all__ = [
    "STATE_ABSENT",
    "STATE_UNRESOLVED",
    "STATE_RESOLVED",
    "critical_point_candidates_batch",
    "gradient_cell_degree",
    "magnetic_axis_subgrid",
    "ring_sign_changes",
    "xpoint_candidates",
]


_RING = (
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
    (-1, 0),
    (-1, 1),
)


def _explicit_float_array(value):
    """Preserve an explicitly supplied NumPy floating dtype at the JAX boundary."""
    if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.floating):
        return jnp.asarray(value, dtype=value.dtype)
    return jnp.asarray(value)


def _arg_extreme(values, axis, *, maximize):
    """Return first extreme indices with an initial matching the value dtype."""
    values = jax.lax.stop_gradient(values)
    axis %= values.ndim
    indices = jax.lax.broadcasted_iota(jnp.int32, values.shape, axis)
    initial_value = -jnp.inf if maximize else jnp.inf
    initial = (
        jnp.asarray(initial_value, dtype=values.dtype),
        jnp.asarray(values.shape[axis], dtype=jnp.int32),
    )

    def choose(left, right):
        left_value, left_index = left
        right_value, right_index = right
        better = right_value > left_value if maximize else right_value < left_value
        take_right = better | ((right_value == left_value) & (right_index < left_index))
        return (
            jnp.where(take_right, right_value, left_value),
            jnp.where(take_right, right_index, left_index),
        )

    return jax.lax.reduce((values, indices), initial, choose, dimensions=(axis,))[1]


def _argmin_exact(values, axis=-1):
    """Return first minimum indices without a default-dtype reduction seed."""
    return _arg_extreme(values, axis, maximize=False)


def _argmax_exact(values, axis=-1):
    """Return first maximum indices without a default-dtype reduction seed."""
    return _arg_extreme(values, axis, maximize=True)


def _normal_cdf(value):
    """Evaluate the standard normal CDF with constants matching value dtype."""
    one = jnp.asarray(1.0, dtype=value.dtype)
    two = jnp.asarray(2.0, dtype=value.dtype)
    half = jnp.asarray(0.5, dtype=value.dtype)
    return half * (one + jax.lax.erf(value / jnp.sqrt(two)))


def _shift(field, dz, dr):
    """Return a shifted view; callers exclude the wrapped border."""
    return jnp.roll(field, shift=(-dz, -dr), axis=(-2, -1))


@jax.jit
def ring_sign_changes(psi):
    """Return the legacy scalar-ring sign-change diagnostic at each vertex."""
    nz, nr = psi.shape
    ring = jnp.stack([_shift(psi, dz, dr) > psi for dz, dr in _RING], axis=0)
    changes = jnp.zeros((nz, nr), dtype=jnp.int32)
    for index in range(8):
        changes += (ring[index] != ring[(index + 1) % 8]).astype(jnp.int32)
    interior = jnp.zeros((nz, nr), dtype=bool).at[1:-1, 1:-1].set(True)
    return jnp.where(interior, changes, -1)


def _native_gradient(fields, rg, zg):
    """Finite-difference gradient at native vertices on a shared grid."""
    dr_mid = (rg[2:] - rg[:-2]).astype(fields.dtype)
    dz_mid = (zg[2:] - zg[:-2]).astype(fields.dtype)
    radial_middle = (fields[..., 2:] - fields[..., :-2]) / dr_mid
    radial = jnp.concatenate(
        [
            ((fields[..., 1] - fields[..., 0]) / (rg[1] - rg[0]).astype(fields.dtype))[
                ..., None
            ],
            radial_middle,
            (
                (fields[..., -1] - fields[..., -2])
                / (rg[-1] - rg[-2]).astype(fields.dtype)
            )[..., None],
        ],
        axis=-1,
    )
    vertical_middle = (fields[..., 2:, :] - fields[..., :-2, :]) / dz_mid[:, None]
    vertical = jnp.concatenate(
        [
            (
                (fields[..., 1, :] - fields[..., 0, :])
                / (zg[1] - zg[0]).astype(fields.dtype)
            )[..., None, :],
            vertical_middle,
            (
                (fields[..., -1, :] - fields[..., -2, :])
                / (zg[-1] - zg[-2]).astype(fields.dtype)
            )[..., None, :],
        ],
        axis=-2,
    )
    return radial, vertical


def gradient_cell_degree(fields, rg, zg):
    """Return signed native gradient degree and boundary-vector margin per cell.

    ``fields`` is ``(batch, nz, nr)``.  The cell traversal is counter-clockwise
    in physical ``(R, Z)`` coordinates, so a saddle has index ``-1`` and an
    extremum index ``+1``.
    """
    radial, vertical = _native_gradient(fields, rg, zg)
    corners = (
        (slice(None, -1), slice(None, -1)),
        (slice(None, -1), slice(1, None)),
        (slice(1, None), slice(1, None)),
        (slice(1, None), slice(None, -1)),
    )
    vectors = jnp.stack(
        [
            jnp.stack([radial[..., rows, cols], vertical[..., rows, cols]], axis=-1)
            for rows, cols in corners
        ],
        axis=-2,
    )
    following = jnp.roll(vectors, -1, axis=-2)
    cross = vectors[..., 0] * following[..., 1] - vectors[..., 1] * following[..., 0]
    dot = jnp.sum(vectors * following, axis=-1)
    winding = jnp.sum(jnp.arctan2(cross, dot), axis=-1) / (2.0 * jnp.pi)
    degree = jnp.rint(winding).astype(jnp.int32)
    margin = jnp.min(jnp.linalg.norm(vectors, axis=-1), axis=-1)
    return degree, winding, margin, radial, vertical


def _automatic_noise_sigma(fields):
    """Estimate unresolved sample noise with quadratic-annihilating differences."""
    third_r = (
        fields[..., 3:]
        - 3.0 * fields[..., 2:-1]
        + 3.0 * fields[..., 1:-2]
        - fields[..., :-3]
    )
    third_z = (
        fields[..., 3:, :]
        - 3.0 * fields[..., 2:-1, :]
        + 3.0 * fields[..., 1:-2, :]
        - fields[..., :-3, :]
    )
    samples = jnp.concatenate(
        [third_r.reshape(fields.shape[0], -1), third_z.reshape(fields.shape[0], -1)],
        axis=1,
    )
    return jnp.median(jnp.abs(samples), axis=1) / 3.016


def _cell_gate(vertex_gate):
    """A cell is geometrically eligible when any of its vertices is eligible."""
    return (
        vertex_gate[..., :-1, :-1]
        | vertex_gate[..., :-1, 1:]
        | vertex_gate[..., 1:, 1:]
        | vertex_gate[..., 1:, :-1]
    )


def _fit_selected_centres(fields, rg, zg, centre_rows, centre_columns):
    """Fit exact-offset total quadratics at supplied native vertices."""
    batch, nz, nr = fields.shape
    centre_rows = jnp.clip(centre_rows, 1, nz - 2)
    centre_columns = jnp.clip(centre_columns, 1, nr - 2)

    offset_z = jnp.asarray([-1, -1, -1, 0, 0, 0, 1, 1, 1])
    offset_r = jnp.asarray([-1, 0, 1, -1, 0, 1, -1, 0, 1])
    sample_rows = centre_rows[..., None] + offset_z
    sample_columns = centre_columns[..., None] + offset_r
    batch_index = jnp.arange(batch)[:, None, None]
    clusters = fields[batch_index, sample_rows, sample_columns]
    flux_offset = jnp.mean(clusters, axis=-1)
    centred = (clusters - flux_offset[..., None]).astype(jnp.float32)

    design = jnp.asarray(
        [
            [1, 1, -1, -1, 1, 1],
            [0, 1, 0, -1, 0, 1],
            [1, 1, 1, -1, -1, 1],
            [1, 0, -1, 0, 0, 1],
            [0, 0, 0, 0, 0, 1],
            [1, 0, 1, 0, 0, 1],
            [1, 1, -1, 1, -1, 1],
            [0, 1, 0, 1, 0, 1],
            [1, 1, 1, 1, 1, 1],
        ],
        dtype=jnp.float32,
    )
    inverse = jnp.asarray(
        [
            [1 / 6, -1 / 3, 1 / 6, 1 / 6, -1 / 3, 1 / 6, 1 / 6, -1 / 3, 1 / 6],
            [1 / 6, 1 / 6, 1 / 6, -1 / 3, -1 / 3, -1 / 3, 1 / 6, 1 / 6, 1 / 6],
            [-1 / 6, 0, 1 / 6, -1 / 6, 0, 1 / 6, -1 / 6, 0, 1 / 6],
            [-1 / 6, -1 / 6, -1 / 6, 0, 0, 0, 1 / 6, 1 / 6, 1 / 6],
            [1 / 4, 0, -1 / 4, 0, 0, 0, -1 / 4, 0, 1 / 4],
            [-1 / 9, 2 / 9, -1 / 9, 2 / 9, 5 / 9, 2 / 9, -1 / 9, 2 / 9, -1 / 9],
        ],
        dtype=jnp.float32,
    )
    coefficients = jnp.einsum("ij,bnj->bni", inverse, centred)
    fitted = jnp.einsum("ji,bni->bnj", design, coefficients)
    residual = centred - fitted
    residual_sigma = jnp.sqrt(jnp.sum(residual**2, axis=-1) / 3.0)
    residual_rms = jnp.sqrt(jnp.mean(residual**2, axis=-1))

    a, b, c, d, e, _constant = jnp.moveaxis(coefficients, -1, 0)
    h00 = 2.0 * a
    h11 = 2.0 * b
    h01 = e
    determinant = h00 * h11 - h01**2
    scale = jnp.maximum(jnp.maximum(jnp.abs(h00 * h11), h01**2), 1.0e-30)
    nonsingular = jnp.abs(determinant) > 128.0 * jnp.finfo(jnp.float32).eps * scale
    safe_determinant = jnp.where(nonsingular, determinant, 1.0)
    local_r = (h01 * d - h11 * c) / safe_determinant
    local_z = (h01 * c - h00 * d) / safe_determinant
    local_r = jnp.clip(jnp.nan_to_num(local_r), -4.0, 4.0)
    local_z = jnp.clip(jnp.nan_to_num(local_z), -4.0, 4.0)
    root_support = jnp.maximum(jnp.abs(local_r), jnp.abs(local_z))

    stationary_basis = jnp.stack(
        [
            local_r**2,
            local_z**2,
            local_r,
            local_z,
            local_r * local_z,
            jnp.ones_like(local_r),
        ],
        axis=-1,
    )
    stationary_flux = (
        jnp.sum(stationary_basis * coefficients, axis=-1).astype(fields.dtype)
        + flux_offset
    )
    radial_scale = 0.5 * (rg[centre_columns + 1] - rg[centre_columns - 1])
    vertical_scale = 0.5 * (zg[centre_rows + 1] - zg[centre_rows - 1])
    radius = rg[centre_columns] + local_r.astype(rg.dtype) * radial_scale
    height = zg[centre_rows] + local_z.astype(zg.dtype) * vertical_scale

    trace_half = 0.5 * (h00 + h11)
    eigen_delta = jnp.sqrt((0.5 * (h00 - h11)) ** 2 + h01**2)
    eigenvalues = jnp.stack(
        [trace_half - eigen_delta, trace_half + eigen_delta], axis=-1
    )
    minimum_absolute_eigenvalue = jnp.min(jnp.abs(eigenvalues), axis=-1)
    maximum_absolute_eigenvalue = jnp.max(jnp.abs(eigenvalues), axis=-1)
    fitted_index = jnp.where(determinant < 0, -1, 1).astype(jnp.int32)
    ntype = jnp.where(
        fitted_index < 0,
        0.0,
        jnp.where(trace_half > 0, -1.0, 1.0),
    )
    solve_r = h00 * local_r + h01 * local_z + c
    solve_z = h01 * local_r + h11 * local_z + d
    solve_residual = jnp.sqrt(solve_r**2 + solve_z**2)
    inverse_hessian = (
        jnp.stack(
            [
                jnp.stack([h11, -h01], axis=-1),
                jnp.stack([-h01, h00], axis=-1),
            ],
            axis=-2,
        )
        / safe_determinant[..., None, None]
    )
    return {
        "r": radius,
        "z": height,
        "psi": stationary_flux,
        "ntype": ntype,
        "fitted_index": fitted_index,
        "nonsingular": nonsingular,
        "root_support": root_support,
        "residual_sigma": residual_sigma,
        "residual_rms": residual_rms,
        "solve_residual": solve_residual,
        "minimum_curvature": minimum_absolute_eigenvalue,
        "maximum_curvature": maximum_absolute_eigenvalue,
        "eigenvalues": eigenvalues,
        "inverse_hessian": inverse_hessian,
        "coefficients": coefficients,
        "centre_rows": centre_rows,
        "centre_columns": centre_columns,
        "local_r": local_r,
        "local_z": local_z,
    }


def _fit_selected_cells(
    fields,
    rg,
    zg,
    radial_gradient,
    vertical_gradient,
    source_cells,
):
    """Choose one native vertex per gathered cell and fit its compact stencil."""
    batch, _nz, nr = fields.shape
    rows = source_cells // (nr - 1)
    columns = source_cells % (nr - 1)
    corner_rows = jnp.stack([rows, rows, rows + 1, rows + 1], axis=-1)
    corner_columns = jnp.stack([columns, columns + 1, columns + 1, columns], axis=-1)
    flat_corners = (corner_rows * nr + corner_columns).reshape(batch, -1)
    gradient_norm = (
        jnp.take_along_axis(
            radial_gradient.reshape(batch, -1), flat_corners, axis=1
        ).reshape(*corner_rows.shape)
        ** 2
        + jnp.take_along_axis(
            vertical_gradient.reshape(batch, -1), flat_corners, axis=1
        ).reshape(*corner_rows.shape)
        ** 2
    )
    nearest = _argmin_exact(gradient_norm, axis=-1)
    centre_rows = jnp.take_along_axis(corner_rows, nearest[..., None], axis=-1)[..., 0]
    centre_columns = jnp.take_along_axis(corner_columns, nearest[..., None], axis=-1)[
        ..., 0
    ]
    return _fit_selected_centres(fields, rg, zg, centre_rows, centre_columns)


@lru_cache(maxsize=None)
def _quadratic_fit_matrices(radius, exclude_inner):
    """Return host-precomputed fixed-stencil least-squares matrices."""
    offsets = [
        (offset_z, offset_r)
        for offset_z in range(-radius, radius + 1)
        for offset_r in range(-radius, radius + 1)
        if not exclude_inner or max(abs(offset_z), abs(offset_r)) > 1
    ]
    first = np.asarray([item[1] for item in offsets], dtype=np.float32)
    second = np.asarray([item[0] for item in offsets], dtype=np.float32)
    design = np.stack(
        [
            first**2,
            second**2,
            first,
            second,
            first * second,
            np.ones_like(first),
        ],
        axis=-1,
    )
    gram_inverse = np.linalg.inv(design.T @ design).astype(np.float32)
    inverse = (gram_inverse @ design.T).astype(np.float32)
    return offsets, design, inverse, gram_inverse


def _scale_fit_selected(
    fields,
    rg,
    zg,
    centre_rows,
    centre_columns,
    *,
    radius,
    exclude_inner=False,
):
    """Fit gathered fixed-radius evidence in exact index coordinates."""
    batch, nz, nr = fields.shape
    offsets, design_host, inverse_host, gram_inverse_host = _quadratic_fit_matrices(
        radius, exclude_inner
    )
    sample_count = len(offsets)
    degrees_of_freedom = sample_count - 6
    safe_rows = jnp.clip(centre_rows, radius, nz - radius - 1)
    safe_columns = jnp.clip(centre_columns, radius, nr - radius - 1)
    offset_z = jnp.asarray([item[0] for item in offsets])
    offset_r = jnp.asarray([item[1] for item in offsets])
    sample_rows = safe_rows[..., None] + offset_z
    sample_columns = safe_columns[..., None] + offset_r
    batch_index = jnp.arange(batch)[:, None, None]
    clusters = fields[batch_index, sample_rows, sample_columns]
    flux_offset = jnp.mean(clusters, axis=-1)
    centred = (clusters - flux_offset[..., None]).astype(jnp.float32)
    design = jnp.asarray(design_host)
    inverse = jnp.asarray(inverse_host)
    gram_inverse = jnp.asarray(gram_inverse_host)
    coefficients = jnp.einsum("ij,bnj->bni", inverse, centred)
    residual = centred - jnp.einsum("ji,bni->bnj", design, coefficients)
    residual_sum_squares = jnp.sum(residual**2, axis=-1)
    residual_rms = jnp.sqrt(jnp.mean(residual**2, axis=-1))
    residual_sigma = jnp.sqrt(residual_sum_squares / degrees_of_freedom)
    a, b, c, d, e, _constant = jnp.moveaxis(coefficients, -1, 0)
    h00 = 2.0 * a
    h11 = 2.0 * b
    hessian = jnp.stack(
        [jnp.stack([h00, e], axis=-1), jnp.stack([e, h11], axis=-1)],
        axis=-2,
    )
    trace_half = 0.5 * (h00 + h11)
    half_difference = 0.5 * (h00 - h11)
    eigen_delta = jnp.sqrt(half_difference**2 + e**2)
    eigenvalues = jnp.stack(
        [trace_half - eigen_delta, trace_half + eigen_delta], axis=-1
    )
    safe_eigen_delta = jnp.maximum(eigen_delta, jnp.finfo(jnp.float32).tiny)
    normalized_difference = half_difference / safe_eigen_delta
    normalized_cross = e / safe_eigen_delta
    eigen_jacobian = jnp.stack(
        [
            jnp.stack(
                [
                    1.0 - normalized_difference,
                    1.0 + normalized_difference,
                    -normalized_cross,
                ],
                axis=-1,
            ),
            jnp.stack(
                [
                    1.0 + normalized_difference,
                    1.0 - normalized_difference,
                    normalized_cross,
                ],
                axis=-1,
            ),
        ],
        axis=-2,
    )
    determinant = 4.0 * a * b - e**2
    nonsingular = jnp.abs(determinant) > (
        128.0
        * jnp.finfo(jnp.float32).eps
        * jnp.maximum(
            jnp.max(jnp.abs(eigenvalues), axis=-1) ** 2,
            jnp.finfo(jnp.float32).tiny,
        )
    )
    safe_determinant = jnp.where(nonsingular, determinant, 1.0)
    inverse_hessian = (
        jnp.stack(
            [
                jnp.stack([2.0 * b, -e], axis=-1),
                jnp.stack([-e, 2.0 * a], axis=-1),
            ],
            axis=-2,
        )
        / safe_determinant[..., None, None]
    )
    gradient = jnp.stack([c, d], axis=-1)
    stationary = -jnp.einsum("bnij,bnj->bni", inverse_hessian, gradient)
    stationary = jnp.where(
        nonsingular[..., None], stationary, jnp.zeros_like(stationary)
    )
    fitted_index = jnp.where(determinant < 0, -1, 1).astype(jnp.int32)
    radial_scale = 0.5 * (rg[safe_columns + 1] - rg[safe_columns - 1])
    vertical_scale = 0.5 * (zg[safe_rows + 1] - zg[safe_rows - 1])
    physical_r = rg[safe_columns] + stationary[..., 0].astype(rg.dtype) * radial_scale
    height = zg[safe_rows] + stationary[..., 1].astype(zg.dtype) * vertical_scale
    stationary_basis = jnp.stack(
        [
            stationary[..., 0] ** 2,
            stationary[..., 1] ** 2,
            stationary[..., 0],
            stationary[..., 1],
            stationary[..., 0] * stationary[..., 1],
            jnp.ones_like(stationary[..., 0]),
        ],
        axis=-1,
    )
    stationary_flux = (
        jnp.sum(stationary_basis * coefficients, axis=-1).astype(fields.dtype)
        + flux_offset
    )
    ntype = jnp.where(
        fitted_index < 0,
        0.0,
        jnp.where(jnp.trace(hessian, axis1=-2, axis2=-1) > 0, -1.0, 1.0),
    )
    geometry_supported = (
        (centre_rows >= radius)
        & (centre_rows < nz - radius)
        & (centre_columns >= radius)
        & (centre_columns < nr - radius)
    )
    return {
        "coefficients": coefficients,
        "r": physical_r,
        "z": height,
        "psi": stationary_flux,
        "ntype": ntype,
        "eigenvalues": eigenvalues,
        "eigen_jacobian": eigen_jacobian,
        "hessian": hessian,
        "inverse_hessian": inverse_hessian,
        "fitted_index": fitted_index,
        "nonsingular": nonsingular,
        "root_support": jnp.max(jnp.abs(stationary), axis=-1),
        "residual_rms": residual_rms,
        "residual_sigma": residual_sigma,
        "residual_sum_squares": residual_sum_squares,
        "degrees_of_freedom": degrees_of_freedom,
        "gram_inverse": gram_inverse,
        "geometry_supported": geometry_supported,
        "centre_rows": safe_rows,
        "centre_columns": safe_columns,
        "stationary": stationary,
    }


def _compact_refit_near_scale(
    fields,
    rg,
    zg,
    scale_fit,
    source_cells,
    rescue_scale,
    target_index,
):
    """Choose one supported native 3x3 refit from four topology-cell vertices."""
    batch, nz, nr = fields.shape
    count = scale_fit["r"].shape[1]
    source_rows = source_cells // (nr - 1)
    source_columns = source_cells % (nr - 1)
    scale_rows = jnp.clip(
        jnp.searchsorted(zg, scale_fit["z"], side="right") - 1, 0, nz - 2
    )
    scale_columns = jnp.clip(
        jnp.searchsorted(rg, scale_fit["r"], side="right") - 1, 0, nr - 2
    )
    base_rows = jnp.where(rescue_scale == 0, source_rows, scale_rows)
    base_columns = jnp.where(rescue_scale == 0, source_columns, scale_columns)
    centre_rows = jnp.clip(
        jnp.stack([base_rows, base_rows, base_rows + 1, base_rows + 1], axis=-1),
        1,
        nz - 2,
    )
    centre_columns = jnp.clip(
        jnp.stack(
            [base_columns, base_columns + 1, base_columns + 1, base_columns],
            axis=-1,
        ),
        1,
        nr - 2,
    )
    candidates = _fit_selected_centres(
        fields,
        rg,
        zg,
        centre_rows.reshape(batch, -1),
        centre_columns.reshape(batch, -1),
    )
    shaped = {
        key: value.reshape(batch, count, 4, *value.shape[2:])
        for key, value in candidates.items()
    }
    distance = jnp.asarray(
        jnp.sqrt(
            ((shaped["r"] - scale_fit["r"][..., None]) / jnp.min(jnp.diff(rg))) ** 2
            + ((shaped["z"] - scale_fit["z"][..., None]) / jnp.min(jnp.diff(zg))) ** 2
        ),
        dtype=fields.dtype,
    )
    supported = (
        shaped["nonsingular"]
        & (shaped["fitted_index"] == target_index)
        & (shaped["root_support"] <= ROOT_SUPPORT_LIMIT)
    )
    selection_cost = jnp.where(
        supported, distance, jnp.asarray(jnp.inf, dtype=fields.dtype)
    )
    selected = _argmin_exact(selection_cost, axis=2)

    def take(value):
        index = selected.reshape(batch, count, 1, *([1] * (value.ndim - 3)))
        return jnp.take_along_axis(value, index, axis=2)[:, :, 0]

    return {key: take(value) for key, value in shaped.items()}


def _square_loop_offsets(radius):
    """Return a counter-clockwise native-vertex square loop."""
    loop = []
    loop.extend((-radius, offset) for offset in range(-radius, radius + 1))
    loop.extend((offset, radius) for offset in range(-radius + 1, radius + 1))
    loop.extend((radius, offset) for offset in range(radius - 1, -radius - 1, -1))
    loop.extend((offset, -radius) for offset in range(radius - 1, -radius, -1))
    return loop


def _native_loop_probe(radial_gradient, vertical_gradient, radius):
    """Return unsmoothed degree and margin around every native vertex."""
    loop = _square_loop_offsets(radius)
    radial = jnp.stack(
        [_shift(radial_gradient, offset_z, offset_r) for offset_z, offset_r in loop],
        axis=-1,
    )
    vertical = jnp.stack(
        [_shift(vertical_gradient, offset_z, offset_r) for offset_z, offset_r in loop],
        axis=-1,
    )
    following_radial = jnp.roll(radial, -1, axis=-1)
    following_vertical = jnp.roll(vertical, -1, axis=-1)
    cross = radial * following_vertical - vertical * following_radial
    dot = radial * following_radial + vertical * following_vertical
    winding = jnp.sum(jnp.arctan2(cross, dot), axis=-1) / (2.0 * jnp.pi)
    degree = jnp.rint(winding).astype(jnp.int32)
    margin = jnp.min(jnp.sqrt(radial**2 + vertical**2), axis=-1)
    nz, nr = radial_gradient.shape[-2:]
    interior = (
        jnp.zeros((nz, nr), dtype=bool)
        .at[radius : nz - radius, radius : nr - radius]
        .set(True)
    )
    return (
        jnp.where(interior, degree, 0)[..., :-1, :-1],
        jnp.where(interior, margin, 0.0)[..., :-1, :-1],
    )


def _window_max(array, size, initial):
    """Return a same-shaped square-window maximum."""
    return jax.lax.reduce_window(
        array,
        jnp.asarray(initial, dtype=array.dtype),
        jax.lax.max,
        (1, size, size),
        (1, 1, 1),
        "SAME",
    )


def _rescue_representatives(probe, margin, occupied, target_index, radius):
    """Collapse overlapping wider-loop probes to deterministic rescue seeds."""
    window = 2 * radius + 1
    unsupported = ~(_window_max(occupied.astype(jnp.int8), window, 0) > 0)
    candidates = (probe == target_index) & unsupported
    weighted = jnp.where(candidates, margin, -jnp.inf)
    local_maximum = _window_max(weighted, window, -jnp.inf)
    maxima = candidates & (margin >= local_maximum)
    source = jnp.arange(probe.shape[-2] * probe.shape[-1], dtype=jnp.int32).reshape(
        probe.shape[-2:]
    )
    source = jnp.broadcast_to(source, probe.shape)
    sentinel = jnp.asarray(source.size, dtype=jnp.int32)
    local_minimum_source = -_window_max(
        -jnp.where(maxima, source, sentinel), window, -sentinel
    )
    return maxima & (source == local_minimum_source)


def _gather_loop_evidence(
    radial_gradient,
    vertical_gradient,
    rows,
    columns,
    radius,
):
    """Return degree, gradient margin, and geometry support for gathered loops."""
    loop = _square_loop_offsets(radius)
    offset_z = jnp.asarray([item[0] for item in loop])
    offset_r = jnp.asarray([item[1] for item in loop])
    batch, nz, nr = radial_gradient.shape
    safe_rows = jnp.clip(rows, radius, nz - radius - 1)
    safe_columns = jnp.clip(columns, radius, nr - radius - 1)
    batch_index = jnp.arange(batch)[:, None, None]
    sample_rows = safe_rows[..., None] + offset_z
    sample_columns = safe_columns[..., None] + offset_r
    radial = radial_gradient[batch_index, sample_rows, sample_columns]
    vertical = vertical_gradient[batch_index, sample_rows, sample_columns]
    following_radial = jnp.roll(radial, -1, axis=-1)
    following_vertical = jnp.roll(vertical, -1, axis=-1)
    cross = radial * following_vertical - vertical * following_radial
    dot = radial * following_radial + vertical * following_vertical
    winding = jnp.sum(jnp.arctan2(cross, dot), axis=-1) / (2.0 * jnp.pi)
    degree = jnp.rint(winding).astype(jnp.int32)
    margin = jnp.min(jnp.sqrt(radial**2 + vertical**2), axis=-1)
    geometry_supported = (
        (rows >= radius)
        & (rows < nz - radius)
        & (columns >= radius)
        & (columns < nr - radius)
    )
    return degree, margin, geometry_supported


def _domain_gradient_degree(radial_gradient, vertical_gradient):
    """Return the unsmoothed gradient degree on the outer grid boundary."""
    nz, nr = radial_gradient.shape[-2:]
    points = []
    points.extend((0, column) for column in range(nr))
    points.extend((row, nr - 1) for row in range(1, nz))
    points.extend((nz - 1, column) for column in range(nr - 2, -1, -1))
    points.extend((row, 0) for row in range(nz - 2, 0, -1))
    rows = jnp.asarray([point[0] for point in points])
    columns = jnp.asarray([point[1] for point in points])
    radial = radial_gradient[:, rows, columns]
    vertical = vertical_gradient[:, rows, columns]
    following_radial = jnp.roll(radial, -1, axis=-1)
    following_vertical = jnp.roll(vertical, -1, axis=-1)
    cross = radial * following_vertical - vertical * following_radial
    dot = radial * following_radial + vertical * following_vertical
    winding = jnp.sum(jnp.arctan2(cross, dot), axis=-1) / (2.0 * jnp.pi)
    return jnp.rint(winding).astype(jnp.int32)


def _same_root_clusters(
    present,
    score,
    radius,
    height,
    position_sigma,
    native_index,
    fit_rows,
    fit_columns,
    fit_local_r,
    fit_local_z,
    rescue_scale,
    source_cells,
    minimum_spacing,
):
    """Cluster uncertainty-overlapping same-sign evidence on device."""
    count = present.shape[1]
    radial_distance = (radius[:, :, None] - radius[:, None, :]) / minimum_spacing
    vertical_distance = (height[:, :, None] - height[:, None, :]) / minimum_spacing
    distance = jnp.sqrt(radial_distance**2 + vertical_distance**2)
    uncertainty_limit = 2.0 * jnp.sqrt(
        position_sigma[:, :, None] ** 2 + position_sigma[:, None, :] ** 2
    )
    shared_fit_center = (fit_rows[:, :, None] == fit_rows[:, None, :]) & (
        fit_columns[:, :, None] == fit_columns[:, None, :]
    )
    same_root = (
        present[:, :, None]
        & present[:, None, :]
        & (native_index[:, :, None] == native_index[:, None, :])
        & (shared_fit_center | (distance <= uncertainty_limit))
    )
    tie = source_cells.astype(score.dtype) * (
        jnp.finfo(score.dtype).eps / max(count, 1)
    )
    adjusted_score = score - tie
    parent = _argmax_exact(
        jnp.where(same_root, adjusted_score[:, None, :], -jnp.inf), axis=-1
    ).astype(jnp.int32)
    maximum_hops = max(1, (count - 1).bit_length())

    def compress_parent(_iteration, current_parent):
        return jnp.take_along_axis(current_parent, current_parent, axis=1)

    parent = jax.lax.fori_loop(0, maximum_hops, compress_parent, parent)

    batch = present.shape[0]
    local_source = jnp.arange(count, dtype=jnp.int32)[None, :]
    representative = present & (parent == local_source)
    base = jnp.arange(batch, dtype=jnp.int32)[:, None] * count
    flat_parent = (parent + base).reshape(-1)
    flat_present = present.reshape(-1).astype(jnp.int32)
    cluster_size = (
        jnp.zeros(batch * count, dtype=jnp.int32)
        .at[flat_parent]
        .add(flat_present)
        .reshape(batch, count)
    )
    unit_member = present & (rescue_scale == 0)
    unit_count = (
        jnp.zeros(batch * count, dtype=jnp.int32)
        .at[flat_parent]
        .add(unit_member.reshape(-1).astype(jnp.int32))
        .reshape(batch, count)
    )
    unit_index_sum = (
        jnp.zeros(batch * count, dtype=jnp.int32)
        .at[flat_parent]
        .add(jnp.where(unit_member, native_index, 0).reshape(-1))
        .reshape(batch, count)
    )
    member_index_sum = jnp.where(unit_count > 0, unit_index_sum, native_index)
    parent_fit_rows = jnp.take_along_axis(fit_rows, parent, axis=1)
    parent_fit_columns = jnp.take_along_axis(fit_columns, parent, axis=1)
    root_extent = jnp.ceil(
        jnp.maximum(
            jnp.abs(fit_rows + fit_local_z - parent_fit_rows),
            jnp.abs(fit_columns + fit_local_r - parent_fit_columns),
        )
        + position_sigma
    ).astype(jnp.int32)
    member_extent = jnp.maximum(root_extent, 1)
    containment_radius = (
        jnp.zeros(batch * count, dtype=jnp.int32)
        .at[flat_parent]
        .max(jnp.where(present, member_extent, 0).reshape(-1))
        .reshape(batch, count)
    )
    duplicate_source = jnp.take_along_axis(source_cells, parent, axis=1)
    duplicate_source = jnp.where(
        present & ~representative, duplicate_source, jnp.asarray(-1, jnp.int32)
    )
    return {
        "representative": representative,
        "parent": parent,
        "duplicate_of_source": duplicate_source,
        "cluster_size": cluster_size,
        "member_index_sum": member_index_sum,
        "containment_radius": containment_radius,
    }


@partial(jax.jit, static_argnames=("estimate_noise",), inline=False)
def _sample_noise_sigma(fields, supplied_noise_sigma, *, estimate_noise):
    """Return supplied covariance scale or estimate it in a lightweight stage."""
    if estimate_noise:
        return _automatic_noise_sigma(fields)
    return supplied_noise_sigma


@partial(
    jax.jit,
    static_argnames=("work_slots", "material_dilate"),
    inline=False,
)
def _native_candidate_stage(
    fields,
    rg,
    zg,
    inside_limiter,
    extra_masks,
    sample_sigma,
    *,
    work_slots,
    material_dilate,
    target_index,
):
    """Generate and rank native unsmoothed degree evidence."""
    batch, nz, nr = fields.shape
    degree, winding, cell_margin, radial_gradient, vertical_gradient = (
        gradient_cell_degree(fields, rg, zg)
    )
    gate = inside_limiter
    for _ in range(material_dilate):
        gate = _dilate4(gate)
    eligible = _cell_gate(gate) & _cell_gate(extra_masks)
    unit_candidates = (degree == target_index) & eligible
    absolute_scale = jnp.max(jnp.abs(fields), axis=(-2, -1))
    numeric_floor = 32.0 * jnp.finfo(jnp.float32).eps * absolute_scale
    minimum_spacing = jnp.minimum(jnp.min(jnp.diff(rg)), jnp.min(jnp.diff(zg))).astype(
        fields.dtype
    )
    gradient_noise_sigma = jnp.maximum(
        sample_sigma[:, None, None] / (jnp.sqrt(2.0) * minimum_spacing),
        numeric_floor[:, None, None] / minimum_spacing,
    )

    radius_two_degree, radius_two_margin = _native_loop_probe(
        radial_gradient, vertical_gradient, 2
    )
    radius_two_rescue = (
        _rescue_representatives(
            radius_two_degree,
            radius_two_margin,
            unit_candidates,
            target_index,
            2,
        )
        & eligible
    )
    radius_four_degree, radius_four_margin = _native_loop_probe(
        radial_gradient, vertical_gradient, 4
    )
    radius_four_rescue = (
        _rescue_representatives(
            radius_four_degree,
            radius_four_margin,
            unit_candidates | radius_two_rescue,
            target_index,
            4,
        )
        & eligible
    )
    candidate_mask = unit_candidates | radius_two_rescue | radius_four_rescue
    rescue_scale_grid = jnp.where(
        unit_candidates,
        0,
        jnp.where(radius_two_rescue, 2, jnp.where(radius_four_rescue, 4, -1)),
    ).astype(jnp.int8)
    seed_margin = jnp.where(
        unit_candidates,
        cell_margin,
        jnp.where(radius_two_rescue, radius_two_margin, radius_four_margin),
    )
    seed_boundary_snr = seed_margin / gradient_noise_sigma
    cheap_score = (
        2.0 * unit_candidates.astype(fields.dtype)
        + jnp.log1p(jnp.maximum(seed_boundary_snr, 0.0))
        + 0.05 * radius_two_rescue.astype(fields.dtype)
        + 0.025 * radius_four_rescue.astype(fields.dtype)
    )

    count = (nz - 1) * (nr - 1)
    work_rank_slots = min(count, work_slots + 1)
    source = jnp.arange(count, dtype=cheap_score.dtype)
    cheap_score = cheap_score.reshape(batch, count) - source[None, :] * (
        jnp.finfo(cheap_score.dtype).eps / count
    )
    _ranked_seed_score, ranked_source = jax.lax.top_k(
        jnp.where(candidate_mask.reshape(batch, count), cheap_score, -jnp.inf),
        work_rank_slots,
    )
    work_source = ranked_source[:, :work_slots]

    def gather_grid(array):
        return jnp.take_along_axis(array.reshape(batch, count), work_source, axis=1)

    work_present = gather_grid(candidate_mask)
    work_rescue_scale = gather_grid(rescue_scale_grid)
    work_native_index = jnp.where(work_present, target_index, 0).astype(jnp.int32)
    candidate_count = jnp.sum(candidate_mask, axis=(-2, -1), dtype=jnp.int32)
    work_overflow = candidate_count > work_slots
    prework_discarded_bound = jnp.where(
        work_overflow, jnp.asarray(1.012, fields.dtype), jnp.nan
    )
    return {
        "radial_gradient": radial_gradient,
        "vertical_gradient": vertical_gradient,
        "work_source": work_source,
        "work_present": work_present,
        "work_rescue_scale": work_rescue_scale,
        "work_native_index": work_native_index,
        "work_winding": gather_grid(winding),
        "sample_sigma": sample_sigma,
        "numeric_floor": numeric_floor,
        "minimum_spacing": minimum_spacing,
        "gradient_noise_sigma": gradient_noise_sigma[:, 0, 0],
        "candidate_count": candidate_count,
        "work_overflow": work_overflow,
        "prework_discarded_bound": prework_discarded_bound,
        "candidate_index_sum": jnp.sum(
            jnp.where(unit_candidates, degree, 0),
            axis=(-2, -1),
            dtype=jnp.int32,
        ),
        "eligible_cell_index_sum": jnp.sum(
            jnp.where(eligible, degree, 0), axis=(-2, -1), dtype=jnp.int32
        ),
        "domain_signed_index": _domain_gradient_degree(
            radial_gradient, vertical_gradient
        ),
        "unit_candidate_count": jnp.sum(
            unit_candidates, axis=(-2, -1), dtype=jnp.int32
        ),
        "rescue_candidate_count": jnp.sum(
            radius_two_rescue | radius_four_rescue,
            axis=(-2, -1),
            dtype=jnp.int32,
        ),
    }


@partial(jax.jit, inline=False)
def _gathered_confidence_stage(
    fields,
    rg,
    zg,
    supplied_noise_sigma,
    radial_gradient,
    vertical_gradient,
    work_source,
    work_present,
    work_rescue_scale,
    work_native_index,
    sample_sigma,
    numeric_floor,
    minimum_spacing,
    gradient_noise_sigma,
    *,
    target_index,
):
    """Fit only gathered candidates and return compact confidence evidence."""
    fit = _fit_selected_cells(
        fields,
        rg,
        zg,
        radial_gradient,
        vertical_gradient,
        work_source,
    )
    middle_fit = _scale_fit_selected(
        fields,
        rg,
        zg,
        fit["centre_rows"],
        fit["centre_columns"],
        radius=2,
    )
    outer_fit = _scale_fit_selected(
        fields,
        rg,
        zg,
        fit["centre_rows"],
        fit["centre_columns"],
        radius=3,
        exclude_inner=True,
    )
    confidence_fit = _scale_fit_selected(
        fields,
        rg,
        zg,
        fit["centre_rows"],
        fit["centre_columns"],
        radius=3,
    )
    fit = _compact_refit_near_scale(
        fields,
        rg,
        zg,
        confidence_fit,
        work_source,
        work_rescue_scale,
        target_index,
    )
    flux_sigma = jnp.where(
        supplied_noise_sigma[:, None] >= 0,
        jnp.maximum(sample_sigma[:, None], numeric_floor[:, None]),
        jnp.maximum(
            jnp.maximum(sample_sigma[:, None], outer_fit["residual_sigma"]),
            jnp.maximum(fit["residual_sigma"], numeric_floor[:, None]),
        ),
    )
    loop_degree_values = []
    loop_margin_values = []
    loop_geometry_values = []
    loop_support_values = []
    for radius in (1, 2, 4):
        loop_degree, loop_margin, loop_geometry = _gather_loop_evidence(
            radial_gradient,
            vertical_gradient,
            fit["centre_rows"],
            fit["centre_columns"],
            radius,
        )
        loop_supported = loop_geometry & (loop_degree == work_native_index)
        loop_degree_values.append(loop_degree)
        loop_margin_values.append(loop_margin)
        loop_geometry_values.append(loop_geometry)
        loop_support_values.append(loop_supported)
    native_loop_support = sum(item.astype(jnp.int8) for item in loop_support_values)

    covariance_indices = jnp.asarray([0, 1, 4])
    hessian_covariance = confidence_fit["gram_inverse"][
        covariance_indices[:, None], covariance_indices
    ]
    eigen_jacobian = confidence_fit["eigen_jacobian"]
    eigen_variance_unit = jnp.einsum(
        "bnki,ij,bnkj->bnk",
        eigen_jacobian,
        hessian_covariance,
        eigen_jacobian,
    )
    eigen_sigma = flux_sigma[..., None] * jnp.sqrt(
        jnp.maximum(eigen_variance_unit, 0.0)
    )
    class_margin = jnp.min(
        jnp.abs(confidence_fit["eigenvalues"])
        / jnp.maximum(eigen_sigma, jnp.finfo(jnp.float32).tiny),
        axis=-1,
    )
    class_probability = _normal_cdf(class_margin)
    scale_drift = jnp.sqrt(
        ((fit["r"] - confidence_fit["r"]) / minimum_spacing) ** 2
        + ((fit["z"] - confidence_fit["z"]) / minimum_spacing) ** 2
    )
    gradient_covariance = jnp.asarray(
        [[1.0 / 6.0, 0.0], [0.0, 1.0 / 6.0]], dtype=jnp.float32
    )
    position_covariance_unit = jnp.einsum(
        "bnij,jk,bnlk->bnil",
        fit["inverse_hessian"],
        gradient_covariance,
        fit["inverse_hessian"],
    )
    compact_position_sigma = flux_sigma * jnp.sqrt(
        jnp.maximum(
            position_covariance_unit[..., 0, 0], position_covariance_unit[..., 1, 1]
        )
    )
    position_sigma = jnp.sqrt(compact_position_sigma**2 + (scale_drift / 1.96) ** 2)
    middle_drift = jnp.sqrt(
        ((fit["r"] - middle_fit["r"]) / minimum_spacing) ** 2
        + ((fit["z"] - middle_fit["z"]) / minimum_spacing) ** 2
    )
    middle_supported = (
        middle_fit["geometry_supported"]
        & middle_fit["nonsingular"]
        & (middle_fit["fitted_index"] == work_native_index)
        & (middle_drift <= 2.0)
    )
    outer_supported = (
        confidence_fit["geometry_supported"]
        & confidence_fit["nonsingular"]
        & (confidence_fit["fitted_index"] == work_native_index)
        & (scale_drift <= 3.0)
    )
    scale_support = (
        1 + middle_supported.astype(jnp.int8) + outer_supported.astype(jnp.int8)
    )

    native_offset = jnp.stack(
        [
            fit["centre_columns"].astype(jnp.float32)
            + fit["local_r"]
            - outer_fit["centre_columns"].astype(jnp.float32),
            fit["centre_rows"].astype(jnp.float32)
            + fit["local_z"]
            - outer_fit["centre_rows"].astype(jnp.float32),
        ],
        axis=-1,
    )
    outer_gradient = jnp.stack(
        [outer_fit["coefficients"][..., 2], outer_fit["coefficients"][..., 3]],
        axis=-1,
    )
    independent_gradient = (
        jnp.einsum("bnij,bnj->bni", outer_fit["hessian"], native_offset)
        + outer_gradient
    )
    first, second = native_offset[..., 0], native_offset[..., 1]
    gradient_design = jnp.stack(
        [
            jnp.stack(
                [
                    2.0 * first,
                    jnp.zeros_like(first),
                    jnp.ones_like(first),
                    jnp.zeros_like(first),
                    second,
                    jnp.zeros_like(first),
                ],
                axis=-1,
            ),
            jnp.stack(
                [
                    jnp.zeros_like(second),
                    2.0 * second,
                    jnp.zeros_like(second),
                    jnp.ones_like(second),
                    first,
                    jnp.zeros_like(second),
                ],
                axis=-1,
            ),
        ],
        axis=-2,
    )
    independent_covariance_unit = jnp.einsum(
        "bnki,ij,bnlj->bnkl",
        gradient_design,
        outer_fit["gram_inverse"],
        gradient_design,
    )
    native_gradient_covariance_unit = jnp.asarray(
        [[1.0 / 6.0, 0.0], [0.0, 1.0 / 6.0]], dtype=jnp.float32
    )
    native_position_covariance_unit = jnp.einsum(
        "bnij,jk,bnlk->bnil",
        fit["inverse_hessian"],
        native_gradient_covariance_unit,
        fit["inverse_hessian"],
    )
    propagated_native_covariance_unit = jnp.einsum(
        "bnij,bnjk,bnlk->bnil",
        outer_fit["hessian"],
        native_position_covariance_unit,
        outer_fit["hessian"],
    )
    independent_covariance = (
        independent_covariance_unit + propagated_native_covariance_unit
    ) * flux_sigma[..., None, None] ** 2
    covariance_a = independent_covariance[..., 0, 0]
    covariance_b = independent_covariance[..., 1, 1]
    covariance_c = independent_covariance[..., 0, 1]
    covariance_determinant = covariance_a * covariance_b - covariance_c**2
    safe_covariance_determinant = jnp.maximum(
        covariance_determinant, jnp.finfo(jnp.float32).tiny
    )
    residual_quadratic = (
        covariance_b * independent_gradient[..., 0] ** 2
        + covariance_a * independent_gradient[..., 1] ** 2
        - 2.0
        * covariance_c
        * independent_gradient[..., 0]
        * independent_gradient[..., 1]
    ) / safe_covariance_determinant
    independent_residual = jnp.sqrt(jnp.maximum(residual_quadratic, 0.0) / 2.0)
    probability_shape = jnp.asarray(
        0.5 * outer_fit["degrees_of_freedom"], dtype=jnp.float32
    )
    probability_argument = (
        outer_fit["residual_sum_squares"]
        / jnp.maximum(2.0 * flux_sigma**2, jnp.finfo(jnp.float32).tiny)
    ).astype(jnp.float32)
    fit_probability = jsp.special.gammaincc(
        probability_shape, probability_argument
    ).astype(fields.dtype)

    native_index_consistent = fit["fitted_index"] == work_native_index
    confidence_index_consistent = confidence_fit["fitted_index"] == work_native_index
    polarity_consistent = (target_index < 0) | (fit["ntype"] == confidence_fit["ntype"])
    structural = (
        work_present
        & fit["nonsingular"]
        & confidence_fit["nonsingular"]
        & native_index_consistent
        & confidence_index_consistent
        & polarity_consistent
        & (fit["root_support"] <= ROOT_SUPPORT_LIMIT)
    )
    class_score = class_margin / CLASS_MARGIN_THRESHOLD
    evidence_score = jnp.where(structural, jnp.clip(class_score, 0.0, 1.0), 0.0)
    ranking_score = (
        evidence_score
        + 0.01 * (scale_support.astype(fields.dtype) / 3.0)
        + 0.001 * jnp.clip(fit_probability, 0.0, 1.0)
        + 0.001 / (1.0 + independent_residual)
    )
    return {
        "r": fit["r"],
        "z": fit["z"],
        "psi": fit["psi"],
        "ntype": fit["ntype"],
        "fit_nonsingular": fit["nonsingular"],
        "root_support": fit["root_support"],
        "fitted_index": fit["fitted_index"],
        "fit_center_rows": fit["centre_rows"],
        "fit_center_columns": fit["centre_columns"],
        "fit_local_r": fit["local_r"],
        "fit_local_z": fit["local_z"],
        "scale_r": confidence_fit["r"],
        "scale_z": confidence_fit["z"],
        "confidence_fitted_index": confidence_fit["fitted_index"],
        "position_sigma": position_sigma,
        "structural": structural,
        "evidence_score": evidence_score,
        "ranking_score": ranking_score,
        "loop_degrees": jnp.stack(loop_degree_values, axis=-1),
        "loop_margins": jnp.stack(loop_margin_values, axis=-1),
        "loop_geometry": jnp.stack(loop_geometry_values, axis=-1),
        "class_margin": class_margin,
        "class_probability": class_probability,
        "independent_residual": independent_residual,
        "fit_probability": fit_probability,
        "scale_support": scale_support,
        "scale_drift": scale_drift,
        "native_loop_support": native_loop_support,
    }


@partial(
    jax.jit,
    static_argnames=("k_slots",),
    inline=False,
)
def _cluster_compaction_stage(
    native,
    evidence,
    *,
    k_slots,
    target_index,
):
    """Certify topology, deduplicate roots, and compact ranked output."""
    work_present = native["work_present"]
    work_source = native["work_source"]
    work_rescue_scale = native["work_rescue_scale"]
    work_native_index = native["work_native_index"]
    batch, work_slots = work_source.shape
    output_dtype = evidence["r"].dtype
    clusters = _same_root_clusters(
        work_present,
        evidence["ranking_score"],
        evidence["r"],
        evidence["z"],
        evidence["position_sigma"],
        work_native_index,
        evidence["fit_center_rows"],
        evidence["fit_center_columns"],
        evidence["fit_local_r"],
        evidence["fit_local_z"],
        work_rescue_scale,
        work_source,
        native["minimum_spacing"],
    )
    representative = clusters["representative"]
    use_radius_one = clusters["containment_radius"] <= 1
    use_radius_two = (clusters["containment_radius"] > 1) & (
        clusters["containment_radius"] <= 2
    )
    degree_one = evidence["loop_degrees"][..., 0]
    degree_two = evidence["loop_degrees"][..., 1]
    degree_four = evidence["loop_degrees"][..., 2]
    margin_one = evidence["loop_margins"][..., 0]
    margin_two = evidence["loop_margins"][..., 1]
    margin_four = evidence["loop_margins"][..., 2]
    geometry_one = evidence["loop_geometry"][..., 0]
    geometry_two = evidence["loop_geometry"][..., 1]
    geometry_four = evidence["loop_geometry"][..., 2]
    noise_floor = jnp.maximum(native["gradient_noise_sigma"][:, None], 1.0e-30)
    pair_one_two = (
        use_radius_one
        & geometry_one
        & geometry_two
        & (degree_one == degree_two)
        & (degree_one != 0)
    )
    weak_intermediate_loop = margin_two / noise_floor < BOUNDARY_SNR_THRESHOLD
    pair_one_four = (
        use_radius_one
        & ~pair_one_two
        & weak_intermediate_loop
        & geometry_one
        & geometry_four
        & (degree_one == degree_four)
        & (degree_one != 0)
    )
    pair_two_four = (
        use_radius_two
        & geometry_two
        & geometry_four
        & (degree_two == degree_four)
        & (degree_two != 0)
    )
    enclosing_degree = jnp.where(pair_one_two | pair_one_four, degree_one, degree_two)
    confirming_degree = jnp.where(pair_one_two, degree_two, degree_four)
    enclosing_margin = jnp.where(pair_one_two | pair_one_four, margin_one, margin_two)
    confirming_margin = jnp.where(pair_one_two, margin_two, margin_four)
    cluster_boundary_snr = (
        jnp.minimum(enclosing_margin, confirming_margin) / noise_floor
    )
    index_certified = representative & (pair_one_two | pair_one_four | pair_two_four)
    cluster_signed_index = jnp.where(index_certified, enclosing_degree, 0)
    simple_index = cluster_signed_index == target_index
    resolved_work = (
        representative
        & evidence["structural"]
        & simple_index
        & (evidence["evidence_score"] >= 1.0)
    )
    ranked_score, ranked_work_source = jax.lax.top_k(
        jnp.where(representative, evidence["ranking_score"], -jnp.inf), k_slots + 1
    )
    selected_work = ranked_work_source[:, :k_slots]

    member_mask = work_present[:, None, :] & (
        clusters["parent"][:, None, :] == selected_work[:, :, None]
    )
    _member_score, member_work_source = jax.lax.top_k(
        jnp.where(
            member_mask,
            -work_source[:, None, :].astype(output_dtype),
            -jnp.inf,
        ),
        4,
    )
    member_present = jnp.take_along_axis(member_mask, member_work_source, axis=2)

    def select_members(array):
        broadcast = jnp.broadcast_to(array[:, None, :], (batch, k_slots, work_slots))
        return jnp.take_along_axis(broadcast, member_work_source, axis=2)

    def select(array):
        return jnp.take_along_axis(array, selected_work, axis=1)

    present = select(representative)
    resolved = select(resolved_work)
    state = jnp.where(
        resolved,
        STATE_RESOLVED,
        jnp.where(present, STATE_UNRESOLVED, STATE_ABSENT),
    ).astype(jnp.int8)
    candidate_count = native["candidate_count"]
    cluster_count = jnp.sum(representative, axis=1, dtype=jnp.int32)
    work_overflow = native["work_overflow"]
    overflow = candidate_count > k_slots
    nan = jnp.full((batch, k_slots), jnp.nan, dtype=output_dtype)
    selected_source = select(work_source).astype(jnp.int32)
    coordinates_present = present & select(evidence["fit_nonsingular"])
    final_discarded_bound = jnp.where(
        cluster_count > k_slots, ranked_score[:, k_slots], jnp.nan
    )
    discarded_score_upper_bound = jnp.where(
        work_overflow,
        native["prework_discarded_bound"],
        final_discarded_bound,
    )

    return {
        "r": jnp.where(coordinates_present, select(evidence["r"]), nan),
        "z": jnp.where(coordinates_present, select(evidence["z"]), nan),
        "psi": jnp.where(coordinates_present, select(evidence["psi"]), nan),
        "ntype": jnp.where(coordinates_present, select(evidence["ntype"]), nan),
        "scale_r": jnp.where(present, select(evidence["scale_r"]), nan),
        "scale_z": jnp.where(present, select(evidence["scale_z"]), nan),
        "present": present,
        "valid": resolved,
        "resolved": resolved,
        "state": state,
        "confidence": select(evidence["evidence_score"]).astype(output_dtype),
        "score": select(evidence["ranking_score"]).astype(output_dtype),
        "source_cell": selected_source,
        "fit_center_row": select(evidence["fit_center_rows"]).astype(jnp.int32),
        "fit_center_column": select(evidence["fit_center_columns"]).astype(jnp.int32),
        "native_signed_index": select(work_native_index),
        "cluster_index_sum": select(cluster_signed_index),
        "member_index_sum": select(clusters["member_index_sum"]),
        "cluster_index_certified": select(index_certified),
        "cluster_containment_radius": select(clusters["containment_radius"]),
        "cluster_size": select(clusters["cluster_size"]),
        "duplicate_of_source": select(clusters["duplicate_of_source"]),
        "member_present": member_present,
        "member_source_cell": jnp.where(
            member_present, select_members(work_source), -1
        ).astype(jnp.int32),
        "member_r": jnp.where(
            member_present, select_members(evidence["r"]), jnp.nan
        ).astype(output_dtype),
        "member_z": jnp.where(
            member_present, select_members(evidence["z"]), jnp.nan
        ).astype(output_dtype),
        "member_rescue_scale": jnp.where(
            member_present, select_members(work_rescue_scale), -1
        ).astype(jnp.int8),
        "native_winding": select(native["work_winding"]).astype(output_dtype),
        "rescue_scale": select(work_rescue_scale),
        "boundary_snr": select(cluster_boundary_snr).astype(output_dtype),
        "enclosing_loop_degree": select(enclosing_degree),
        "confirming_loop_degree": select(confirming_degree),
        "enclosing_loop_margin": select(enclosing_margin).astype(output_dtype),
        "confirming_loop_margin": select(confirming_margin).astype(output_dtype),
        "loop_degree_radius_1": select(evidence["loop_degrees"][..., 0]),
        "loop_degree_radius_2": select(evidence["loop_degrees"][..., 1]),
        "loop_degree_radius_4": select(evidence["loop_degrees"][..., 2]),
        "loop_margin_radius_1": select(evidence["loop_margins"][..., 0]).astype(
            output_dtype
        ),
        "loop_margin_radius_2": select(evidence["loop_margins"][..., 1]).astype(
            output_dtype
        ),
        "loop_margin_radius_4": select(evidence["loop_margins"][..., 2]).astype(
            output_dtype
        ),
        "class_margin": select(evidence["class_margin"]).astype(output_dtype),
        "class_probability": select(evidence["class_probability"]).astype(output_dtype),
        "position_sigma_cell": select(evidence["position_sigma"]).astype(output_dtype),
        "root_support_cell": select(evidence["root_support"]).astype(output_dtype),
        "normalized_residual": select(evidence["independent_residual"]).astype(
            output_dtype
        ),
        "root_residual_snr": select(evidence["independent_residual"]).astype(
            output_dtype
        ),
        "fit_probability": select(evidence["fit_probability"]).astype(output_dtype),
        "scale_support": select(evidence["scale_support"]),
        "scale_drift_cell": select(evidence["scale_drift"]).astype(output_dtype),
        "native_loop_support": select(evidence["native_loop_support"]),
        "fit_condition": jnp.full(
            (batch, k_slots), LOCAL_DESIGN_CONDITION, dtype=output_dtype
        ),
        "confidence_fit_condition": jnp.full(
            (batch, k_slots), 12.912939354880008, dtype=output_dtype
        ),
        "fitted_signed_index": select(evidence["fitted_index"]),
        "confidence_fitted_signed_index": select(evidence["confidence_fitted_index"]),
        "candidate_count": candidate_count,
        "cluster_count": cluster_count,
        "work_capacity": jnp.full((batch,), work_slots, dtype=jnp.int32),
        "work_overflow": work_overflow,
        "overflow": overflow,
        "discarded_score_upper_bound": discarded_score_upper_bound.astype(output_dtype),
        "candidate_index_sum": native["candidate_index_sum"],
        "raw_candidate_index_sum": native["candidate_index_sum"],
        "eligible_cell_index_sum": native["eligible_cell_index_sum"],
        "domain_signed_index": native["domain_signed_index"],
        "unit_candidate_count": native["unit_candidate_count"],
        "rescue_candidate_count": native["rescue_candidate_count"],
        "sample_noise_sigma": native["sample_sigma"].astype(output_dtype),
    }


def _critical_point_candidates_batch(
    fields,
    rg,
    zg,
    inside_limiter,
    extra_masks,
    supplied_noise_sigma,
    *,
    k_slots,
    material_dilate,
    target_index,
    estimate_noise,
):
    """Run the device-resident detector as three bounded compiled stages."""
    sample_sigma = _sample_noise_sigma(
        fields, supplied_noise_sigma, estimate_noise=estimate_noise
    )
    cell_count = (fields.shape[-2] - 1) * (fields.shape[-1] - 1)
    if k_slots <= 8:
        requested_work_slots = 32
    elif k_slots <= 64:
        requested_work_slots = 128
    else:
        requested_work_slots = 2 * k_slots
    work_slots = min(cell_count, requested_work_slots)
    native = _native_candidate_stage(
        fields,
        rg,
        zg,
        inside_limiter,
        extra_masks,
        sample_sigma,
        work_slots=work_slots,
        material_dilate=material_dilate,
        target_index=target_index,
    )
    evidence = _gathered_confidence_stage(
        fields,
        rg,
        zg,
        supplied_noise_sigma,
        native["radial_gradient"],
        native["vertical_gradient"],
        native["work_source"],
        native["work_present"],
        native["work_rescue_scale"],
        native["work_native_index"],
        native["sample_sigma"],
        native["numeric_floor"],
        native["minimum_spacing"],
        native["gradient_noise_sigma"],
        target_index=target_index,
    )
    compact_native = {
        key: value
        for key, value in native.items()
        if key not in {"radial_gradient", "vertical_gradient", "numeric_floor"}
    }
    return _cluster_compaction_stage(
        compact_native,
        evidence,
        k_slots=k_slots,
        target_index=target_index,
    )


def critical_point_candidates_batch(
    fields,
    rg,
    zg,
    inside_limiter,
    *,
    k_slots=8,
    extra_mask=None,
    material_dilate=1,
    target_index=-1,
    noise_sigma=None,
):
    """Return ranked fixed-capacity critical points for ``(batch, nz, nr)``.

    Candidate count and signed-index totals describe the complete pre-capacity
    set.  ``present`` distinguishes a retained native candidate from padding;
    ``state`` then distinguishes resolved and unresolved evidence.  Supplying a
    scalar or per-field sample-noise standard deviation is preferred when the
    acquisition covariance is known; otherwise a quadratic-annihilating robust
    estimate is used.
    """
    fields = _explicit_float_array(fields)
    if fields.ndim != 3:
        raise ValueError("fields must have shape (batch, nz, nr)")
    inside_limiter = jnp.asarray(inside_limiter, dtype=bool)
    if extra_mask is None:
        extra_masks = jnp.ones_like(fields, dtype=bool)
    else:
        extra_masks = jnp.asarray(extra_mask, dtype=bool)
        if extra_masks.ndim == 2:
            extra_masks = jnp.broadcast_to(extra_masks, fields.shape)
    if noise_sigma is None:
        supplied_noise_sigma = jnp.full((fields.shape[0],), -1.0, dtype=fields.dtype)
    else:
        supplied_noise_sigma = jnp.broadcast_to(
            jnp.asarray(noise_sigma, dtype=fields.dtype), (fields.shape[0],)
        )
    return _critical_point_candidates_batch(
        fields,
        _explicit_float_array(rg),
        _explicit_float_array(zg),
        inside_limiter,
        extra_masks,
        supplied_noise_sigma,
        k_slots=int(k_slots),
        material_dilate=int(material_dilate),
        target_index=int(target_index),
        estimate_noise=noise_sigma is None,
    )


def _scalar_result(result):
    """Remove the singleton batch axis from a result tree."""
    return jax.tree.map(lambda value: value[0], result)


def _refine_selected_vertices(psi, rg, zg, rows, columns):
    """Differentiably fit only the already selected fixed-capacity vertices."""
    offset_z = jnp.asarray([-1, -1, -1, 0, 0, 0, 1, 1, 1])
    offset_r = jnp.asarray([-1, 0, 1, -1, 0, 1, -1, 0, 1])
    clusters = psi[rows[:, None] + offset_z, columns[:, None] + offset_r]
    flux_offset = jnp.mean(clusters, axis=-1)
    centred = (clusters - flux_offset[:, None]).astype(psi.dtype)
    inverse = jnp.asarray(
        [
            [1 / 6, -1 / 3, 1 / 6, 1 / 6, -1 / 3, 1 / 6, 1 / 6, -1 / 3, 1 / 6],
            [1 / 6, 1 / 6, 1 / 6, -1 / 3, -1 / 3, -1 / 3, 1 / 6, 1 / 6, 1 / 6],
            [-1 / 6, 0, 1 / 6, -1 / 6, 0, 1 / 6, -1 / 6, 0, 1 / 6],
            [-1 / 6, -1 / 6, -1 / 6, 0, 0, 0, 1 / 6, 1 / 6, 1 / 6],
            [1 / 4, 0, -1 / 4, 0, 0, 0, -1 / 4, 0, 1 / 4],
            [-1 / 9, 2 / 9, -1 / 9, 2 / 9, 5 / 9, 2 / 9, -1 / 9, 2 / 9, -1 / 9],
        ],
        dtype=psi.dtype,
    )
    coefficients = jnp.einsum("ij,nj->ni", inverse, centred)
    a, b, c, d, e, constant = jnp.moveaxis(coefficients, -1, 0)
    h00 = 2.0 * a
    h11 = 2.0 * b
    determinant = h00 * h11 - e**2
    safe = jnp.where(jnp.abs(determinant) > 1.0e-30, determinant, 1.0)
    local_r = jnp.clip((e * d - h11 * c) / safe, -4.0, 4.0)
    local_z = jnp.clip((e * c - h00 * d) / safe, -4.0, 4.0)
    radial_scale = 0.5 * (rg[columns + 1] - rg[columns - 1])
    vertical_scale = 0.5 * (zg[rows + 1] - zg[rows - 1])
    radius = rg[columns] + local_r.astype(rg.dtype) * radial_scale
    height = zg[rows] + local_z.astype(zg.dtype) * vertical_scale
    stationary_flux = (
        a * local_r**2
        + b * local_z**2
        + c * local_r
        + d * local_z
        + e * local_r * local_z
        + constant
    ).astype(psi.dtype) + flux_offset
    ntype = jnp.where(
        determinant < 0,
        0.0,
        jnp.where(0.5 * (h00 + h11) > 0, -1.0, 1.0),
    )
    return radius, height, stationary_flux, ntype


def xpoint_candidates(
    psi,
    rg,
    zg,
    inside_limiter,
    k_slots=6,
    extra_mask=None,
    material_dilate=1,
    noise_sigma=None,
):
    """Return score-ranked rectangular X-point evidence in fixed slots."""
    psi = _explicit_float_array(psi)
    rg = _explicit_float_array(rg)
    zg = _explicit_float_array(zg)
    mask = None if extra_mask is None else jnp.asarray(extra_mask)[None]
    selected = _scalar_result(
        critical_point_candidates_batch(
            jax.lax.stop_gradient(psi)[None],
            rg,
            zg,
            inside_limiter,
            k_slots=k_slots,
            extra_mask=mask,
            material_dilate=material_dilate,
            target_index=-1,
            noise_sigma=noise_sigma,
        )
    )
    refined = _refine_selected_vertices(
        psi,
        rg,
        zg,
        selected["fit_center_row"],
        selected["fit_center_column"],
    )
    selected = dict(selected)
    nan = jnp.full(selected["present"].shape, jnp.nan, dtype=psi.dtype)
    for key, value in zip(("r", "z", "psi", "ntype"), refined, strict=True):
        selected[key] = jnp.where(selected["present"], value, nan)
    return selected


def magnetic_axis_subgrid(psi, rg, zg, inside_limiter, region=None, noise_sigma=None):
    """Return the deepest resolved rectangular extremum as the magnetic axis."""
    psi = _explicit_float_array(psi)
    rg = _explicit_float_array(rg)
    zg = _explicit_float_array(zg)
    mask = jnp.asarray(inside_limiter, dtype=bool)
    if region is not None:
        mask &= jnp.asarray(region) > 0.5
    result = _scalar_result(
        critical_point_candidates_batch(
            jax.lax.stop_gradient(psi)[None],
            rg,
            zg,
            inside_limiter,
            k_slots=1,
            extra_mask=mask[None],
            material_dilate=0,
            target_index=1,
            noise_sigma=noise_sigma,
        )
    )
    refined = _refine_selected_vertices(
        psi,
        rg,
        zg,
        result["fit_center_row"],
        result["fit_center_column"],
    )
    found = result["resolved"][0]
    return {
        "r": jnp.where(found, refined[0][0], jnp.nan),
        "z": jnp.where(found, refined[1][0], jnp.nan),
        "psi": jnp.where(found, refined[2][0], jnp.nan),
        "ntype": jnp.where(found, refined[3][0], jnp.nan),
        "found": found,
        "present": result["present"][0],
        "state": result["state"][0],
        "confidence": result["confidence"][0],
        "candidate_count": result["candidate_count"],
        "overflow": result["overflow"],
        "discarded_score_upper_bound": result["discarded_score_upper_bound"],
        "native_signed_index": result["native_signed_index"][0],
        "source_cell": result["source_cell"][0],
        "position_sigma_cell": result["position_sigma_cell"][0],
    }
