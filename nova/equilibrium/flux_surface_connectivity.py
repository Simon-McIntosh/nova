"""Accelerator-native flux-surface connectivity and contour geometry (JAX).

Fixed-shape, ``jit`` / ``vmap`` / ``grad``-safe flux-surface metrics from a
solved poloidal-flux field ψ(R, Z) — the device-native replacement for the
host-side coarea binning (``argsort`` + cumulative-sum + ``scipy.ndimage.label``)
that cannot batch on an accelerator.  The averaging path is a fixed-shape
reduction over the full grid: no data-dependent shapes, no sort, and no contour
or marching-squares extraction.

The flux-surface averaging path remains contour-free.  A separate contour
primitive at the end of this module extracts fixed-capacity cubic Hermite arcs
from the global tensor spline.  It does not participate in clipped-support
integration, whose polygon-exact path remains independent.

Three device primitives:

1. **connectivity core weight** — a dilate-by-doubling flood-fill from the axis
   cell over the confined level set {ψ_N < 1 ∧ inside-limiter}.  Associative
   scans double the reachable distance along uninterrupted row and column
   segments, and the fill stops at its fixed point.  It selects the
   axis-connected core and rejects any disconnected private pocket at comparable
   flux by CONNECTIVITY, never by ψ height or sign-of-Z — the same rule as
   :mod:`nova.equilibrium.connectivity_boundary`, but as a fixed-shape
   device kernel rather than ``scipy.ndimage.label``.

2. **smooth-CDF coarea average** — the flux-surface average
   ⟨X⟩(u) = (d/du ∫_{ψ_N<u} X dV) / (d/du ∫_{ψ_N<u} dV) evaluated as the coarea
   path does — a difference of cumulative volume integrals — but with the hard
   ``interp(cumulative-sum-over-sorted-cells)`` step replaced by a SMOOTH
   Gaussian-CDF cumulative on a FIXED ψ_N level grid:

       C_X(ℓ) = Σ_c m_c dV_c X_c Φ((ℓ − ψ_N,c)/h),   ⟨X⟩_j = ΔC_X / ΔC_V,

   with m_c the core weight, Φ the standard-normal CDF, and h the bandwidth.  As
   h → 0, Φ → the step function and this reduces EXACTLY to the coarea estimator;
   at finite h it is the same estimator with the cell-binning noise smoothed out.
   Being a difference of cumulatives it is edge-unbiased (unlike a point-KDE
   ratio), and it uses no sort and no data-dependent shapes.

The averaged quantities returned are dV/dψ_N (→ V′), ⟨1/R²⟩ (→ g3), ⟨1/R⟩, and
⟨|∇ψ|²/R²⟩ (→ g2 via dV/dΦ), sampled at the same ψ_N mid-levels the coarea path
reports.  The profile / F / ρ̂ / ψ assembly downstream is unchanged and validated
(:func:`the current-diffusion assembly.flux_surface_geometry`); this module
supplies ONLY the geometric flux-surface averages, so the two paths differ purely
in how the surfaces are averaged — a clean apples-to-apples comparison.

Design intent: this is the GPU inner-loop kernel for the batched corpus labeller
(a ``jax.vmap`` over slices sharing one campaign grid).  It runs on CPU today (no
CUDA jaxlib) with identical semantics; the device is chosen by the installed
jaxlib, not by this code.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from nova.equilibrium.morphology import _dilate4 as _dilate4
from nova.linalg.tensor_spline import fit_tensor_spline


_SQRT2 = 2.0**0.5

__all__ = [
    "flood_fill_core",
    "flood_fill_core_with_steps",
    "label_connected_components",
    "label_connected_components_with_steps",
    "private_flux_mask",
    "traced_spline_contour",
    "traced_flux_surface_bins",
]


# ---------------------------------------------------------------------------
# connectivity core weight — dilate-by-doubling flood-fill (device kernel)
# ---------------------------------------------------------------------------


def _compose_segment_reach(left, right):
    """Compose boolean reach maps for adjacent segments.

    Each pair ``(reached, open_)`` represents ``reached | (open_ & incoming)``.
    Composition is associative, so :func:`jax.lax.associative_scan` evaluates a
    complete row or column with logarithmic propagation depth.
    """
    reached_left, open_left = left
    reached_right, open_right = right
    return (
        reached_right | (open_right & reached_left),
        open_right & open_left,
    )


def _fill_segments(core: jnp.ndarray, confined: jnp.ndarray, axis: int) -> jnp.ndarray:
    """Fill every confined segment on ``axis`` touched by ``core``."""
    elements = (core & confined, confined)
    forward = jax.lax.associative_scan(_compose_segment_reach, elements, axis=axis)[0]
    backward = jax.lax.associative_scan(
        _compose_segment_reach, elements, axis=axis, reverse=True
    )[0]
    return (forward | backward) & confined


def _compose_segment_minimum(left, right):
    """Compose minimum-label maps for adjacent confined segments."""
    label_left, open_left = left
    label_right, open_right = right
    return (
        jnp.where(open_right, jnp.minimum(label_right, label_left), label_right),
        open_right & open_left,
    )


def _fill_label_segments(
    labels: jnp.ndarray, confined: jnp.ndarray, axis: int
) -> jnp.ndarray:
    """Propagate each segment's minimum positive label along ``axis``."""
    sentinel = jnp.asarray(jnp.iinfo(labels.dtype).max, dtype=labels.dtype)
    elements = (jnp.where(confined, labels, sentinel), confined)
    forward = jax.lax.associative_scan(_compose_segment_minimum, elements, axis=axis)[0]
    backward = jax.lax.associative_scan(
        _compose_segment_minimum, elements, axis=axis, reverse=True
    )[0]
    return jnp.where(confined, jnp.minimum(forward, backward), 0)


@partial(jax.jit, static_argnums=(2,))
def flood_fill_core_with_steps(
    confined: jnp.ndarray, seed: jnp.ndarray, n_iter: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return the axis-connected component and doubling-step count.

    ``confined`` and ``seed`` are ``(nz, nr)`` boolean fields. ``n_iter`` remains
    a static safety cap and must be at least the component's grid diameter
    (``nr + nz`` is sufficient).  One step closes every reached row segment and
    then every reached column segment.  The associative scans double their reach
    internally, while the outer loop stops as soon as the component is unchanged.

    The returned core is the float 0/1 component containing the seed, never a
    disconnected pocket.  The scalar step count is an execution diagnostic.
    """

    start = seed & confined

    def condition(state):
        step, core, previous = state
        return (step < n_iter) & jnp.any(core != previous)

    def body(state):
        step, core, _previous = state
        row_filled = _fill_segments(core, confined, axis=1)
        next_core = _fill_segments(row_filled, confined, axis=0)
        return step + 1, next_core, core

    step, core, _previous = jax.lax.while_loop(
        condition,
        body,
        (jnp.asarray(0, dtype=jnp.int32), start, jnp.zeros_like(start)),
    )
    # Exact 0/1 connectivity weights need no double-precision storage.  Their
    # products promote to the solved field's selected compute dtype.
    return core.astype(jnp.float32), step


@partial(jax.jit, static_argnums=(2,))
def flood_fill_core(
    confined: jnp.ndarray, seed: jnp.ndarray, n_iter: int
) -> jnp.ndarray:
    """Return the seed-connected component of ``confined`` as float 0/1 weights."""
    core, _steps = flood_fill_core_with_steps(confined, seed, n_iter)
    return core


@partial(jax.jit, static_argnums=(1,))
def label_connected_components_with_steps(
    confined: jnp.ndarray, n_iter: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return canonical labels for every 4-connected confined component.

    Every confined cell starts with its one-based flat-grid index. Row and
    column scans then propagate the minimum index within each uninterrupted
    segment. Alternating the two segment fills to a fixed point leaves every
    component carrying its minimum member index, with zero reserved for cells
    outside ``confined``. ``n_iter`` is a static safety cap; ``nr + nz`` is
    sufficient for a grid whose components use 4-connectivity.

    Labels are stable across execution order and need not be consecutive. The
    scalar step count is an execution diagnostic.
    """
    initial = jnp.arange(confined.size, dtype=jnp.int32).reshape(confined.shape) + 1
    initial = jnp.where(confined, initial, 0)

    def condition(state):
        step, labels, previous = state
        return (step < n_iter) & jnp.any(labels != previous)

    def body(state):
        step, labels, _previous = state
        row_filled = _fill_label_segments(labels, confined, axis=1)
        next_labels = _fill_label_segments(row_filled, confined, axis=0)
        return step + 1, next_labels, labels

    step, labels, _previous = jax.lax.while_loop(
        condition,
        body,
        (jnp.asarray(0, dtype=jnp.int32), initial, jnp.zeros_like(initial)),
    )
    return labels, step


@partial(jax.jit, static_argnums=(1,))
def label_connected_components(confined: jnp.ndarray, n_iter: int) -> jnp.ndarray:
    """Return canonical labels for every 4-connected confined component."""
    labels, _steps = label_connected_components_with_steps(confined, n_iter)
    return labels


@jax.jit
def private_flux_mask(
    component_labels: jnp.ndarray, axis_seed: jnp.ndarray
) -> jnp.ndarray:
    """Return labelled confined cells disconnected from the magnetic axis.

    ``component_labels`` comes from :func:`label_connected_components` and
    uses zero for unconfined cells. ``axis_seed`` is a boolean field containing
    the binding-level magnetic-axis cell. A confined cell is private precisely
    when its positive component label differs from the axis component label.
    """
    sentinel = jnp.asarray(jnp.iinfo(component_labels.dtype).max)
    axis_label = jnp.min(
        jnp.where(axis_seed & (component_labels > 0), component_labels, sentinel)
    )
    return (component_labels > 0) & (component_labels != axis_label)


# ---------------------------------------------------------------------------
# kernel coarea flux-surface averages (device kernel)
# ---------------------------------------------------------------------------


def _gaussian_cdf(z: jnp.ndarray) -> jnp.ndarray:
    """Standard-normal CDF Φ(z) — the smooth cumulative-volume weight."""
    return 0.5 * (1.0 + jax.scipy.special.erf(z / _SQRT2))


@partial(jax.jit, static_argnums=(8,))
def traced_flux_surface_bins(
    psi2d: jnp.ndarray,
    rg: jnp.ndarray,
    zg: jnp.ndarray,
    inside_limiter: jnp.ndarray,
    axis_psi: jnp.ndarray,
    boundary_psi: jnp.ndarray,
    psin_min: jnp.ndarray,
    psin_max: jnp.ndarray,
    n_psin: int,
    h_factor: jnp.ndarray = 1.25,
) -> dict:
    """Contour-free flux-surface averages on a fixed ψ_N mid-level grid.

    ``psi2d`` is the total poloidal flux ``(nz, nr)``; ``rg``/``zg`` the axis-
    ordered grid coordinates; ``inside_limiter`` the ``(nz, nr)`` limiter mask.
    ``axis_psi``/``boundary_psi`` set the ψ_N normalisation (the axis cell is the
    flood-fill seed, located as the extreme of ψ_N on the confined side).

    Returns a dict of arrays at the ``n_psin`` mid-levels ``pn_s``:
    ``dv_dpn`` (dV/dψ_N), ``inv_r2`` (⟨1/R²⟩), ``inv_r`` (⟨1/R⟩), ``grad2_r2``
    (⟨|∇ψ|²/R²⟩), ``v_cum`` (∫_{ψ_N<u} dV), plus the ``core_fraction`` diagnostic.
    All fixed-shape; ``jit``/``vmap``/``grad``-safe.
    """
    nz = zg.shape[0]
    nr = rg.shape[0]
    dr = rg[1] - rg[0]
    dz = zg[1] - zg[0]
    mesh_r = jnp.broadcast_to(rg[jnp.newaxis, :], (nz, nr))

    span = boundary_psi - axis_psi
    span = jnp.where(jnp.abs(span) < 1e-12, 1e-12, span)
    psi_n = (psi2d - axis_psi) / span

    confined = (psi_n < 1.0) & inside_limiter

    # axis cell = the confined cell whose ψ_N is smallest (deepest core); seed the
    # flood-fill there.  argmin over a masked field is fixed-shape (no gather).
    pn_seed = jnp.where(confined, psi_n, jnp.inf)
    seed_flat = jnp.argmin(pn_seed.reshape(-1))
    seed = jnp.zeros((nz, nr), dtype=bool).reshape(-1).at[seed_flat].set(True)
    seed = seed.reshape(nz, nr)
    core = flood_fill_core(confined, seed, nr + nz)  # (nz, nr) float 0/1

    # per-cell geometric weights (all cells; the core weight zeros out the rest)
    dvol = 2.0 * jnp.pi * mesh_r * dr * dz  # 2πR dA
    gz, gr = jnp.gradient(psi2d, zg, rg)  # axis 0 = Z, axis 1 = R
    grad2 = gr**2 + gz**2
    inv_r2_cell = 1.0 / mesh_r**2
    inv_r_cell = 1.0 / mesh_r

    w_cell = (core * dvol).reshape(-1)  # (ncell,) volume-in-core weight m_c·dV_c
    pn_flat = psi_n.reshape(-1)
    ir2_flat = inv_r2_cell.reshape(-1)
    ir_flat = inv_r_cell.reshape(-1)
    g2r2_flat = (grad2 * inv_r2_cell).reshape(-1)

    # fixed ψ_N level grid (data-independent → fixed shape, vmap-safe)
    levels = jnp.linspace(
        psin_min, psin_max, n_psin + 1, dtype=psi2d.dtype
    )  # (n_psin+1,)
    pn_s = 0.5 * (levels[:-1] + levels[1:])  # metric samples at mid-levels
    dlevel = (psin_max - psin_min) / n_psin
    h = h_factor * dlevel

    # smooth cumulative volume integrals C_Q(ℓ) = Σ_c w_c q_c Φ((ℓ − ψ_N,c)/h):
    # the coarea path's interp(cumsum-over-sorted-cells) with the hard step
    # replaced by the Gaussian CDF — same estimator, differentiable + fixed-shape.
    zc = (levels[:, jnp.newaxis] - pn_flat[jnp.newaxis, :]) / h  # (n_lvl, ncell)
    wcdf = w_cell[jnp.newaxis, :] * _gaussian_cdf(zc)  # (n_lvl, ncell)
    c_v = jnp.sum(wcdf, axis=1)
    c_r2 = jnp.sum(wcdf * ir2_flat[jnp.newaxis, :], axis=1)
    c_ir = jnp.sum(wcdf * ir_flat[jnp.newaxis, :], axis=1)
    c_g2 = jnp.sum(wcdf * g2r2_flat[jnp.newaxis, :], axis=1)

    dv_lvl = jnp.diff(c_v)  # ΔV over each level interval
    dv_lvl_safe = jnp.where(dv_lvl > 0, dv_lvl, 1.0)
    dv_dpn = dv_lvl / dlevel  # dV/dψ_N at mid-levels
    inv_r2 = jnp.diff(c_r2) / dv_lvl_safe  # ⟨1/R²⟩
    inv_r = jnp.diff(c_ir) / dv_lvl_safe  # ⟨1/R⟩
    grad2_r2 = jnp.diff(c_g2) / dv_lvl_safe  # ⟨|∇ψ|²/R²⟩
    v_cum = 0.5 * (c_v[:-1] + c_v[1:])  # cumulative V at mid-levels

    v_total = jnp.sum(w_cell)  # total core volume Σ m_c dV_c
    ncore = jnp.sum(core)
    return {
        "pn_s": pn_s,
        "dv_dpn": dv_dpn,
        "inv_r2": inv_r2,
        "inv_r": inv_r,
        "grad2_r2": grad2_r2,
        "v_cum": v_cum,
        "v_total": v_total,
        "span": span,
        "core_fraction": ncore / (nz * nr),
        "n_core_cells": ncore,
        "well_posed": (dv_lvl > 0).all() & (ncore >= 200),
    }


# ---------------------------------------------------------------------------
# global-spline contour geometry (separate from polygon clip integration)
# ---------------------------------------------------------------------------


def _structured_cell_edges(
    radial: jnp.ndarray, vertical: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return counter-clockwise physical edge endpoints for every grid cell."""
    cell_shape = (vertical.size - 1, radial.size - 1)
    radial_low = jnp.broadcast_to(radial[:-1][None, :], cell_shape)
    radial_high = jnp.broadcast_to(radial[1:][None, :], cell_shape)
    vertical_low = jnp.broadcast_to(vertical[:-1, None], cell_shape)
    vertical_high = jnp.broadcast_to(vertical[1:, None], cell_shape)
    start = jnp.stack(
        (
            jnp.stack((radial_low, vertical_low), axis=-1),
            jnp.stack((radial_high, vertical_low), axis=-1),
            jnp.stack((radial_high, vertical_high), axis=-1),
            jnp.stack((radial_low, vertical_high), axis=-1),
        ),
        axis=-2,
    )
    return start, jnp.roll(start, shift=-1, axis=-2)


def _ordered_crossing_indices(edge_crossing: jnp.ndarray) -> jnp.ndarray:
    """Pack four edge indices per cell without a data-dependent output shape."""
    rank = jnp.cumsum(edge_crossing.astype(jnp.int32), axis=-1) - 1
    edge_index = jnp.arange(4, dtype=jnp.int32)
    return jnp.stack(
        [
            jnp.sum(jnp.where(edge_crossing & (rank == slot), edge_index, 0), axis=-1)
            for slot in range(4)
        ],
        axis=-1,
    )


def _paired_edge_values(values: jnp.ndarray, indices: jnp.ndarray) -> jnp.ndarray:
    """Gather two edge endpoints for each of two fixed-capacity segments."""
    return jnp.take_along_axis(values[..., None, :, :], indices[..., None], axis=-2)


@partial(jax.jit, static_argnums=(4, 5))
def traced_spline_contour(
    values: jnp.ndarray,
    radial: jnp.ndarray,
    vertical: jnp.ndarray,
    level: jnp.ndarray,
    bisection_steps: int = 40,
    saddle_steps: int = 8,
) -> dict[str, jnp.ndarray]:
    """Extract fixed-capacity cubic contour arcs from a global tensor spline.

    ``values`` has shape ``(vertical.size, radial.size)``.  Every cell owns four
    edge slots and at most two cubic Bezier segments, so output shapes depend
    only on the input lattice.  Every inactive padded edge, segment, and saddle
    slot is canonical exact zero.  Edge crossings are fixed-iteration roots of
    the global spline restriction.  Segment endpoint tangents are perpendicular
    to the same spline's gradient, giving geometrically continuous tangents
    where adjacent cells share a crossing.

    Diagonal corner configurations use the global spline's interior stationary
    value when a saddle is found, with a centre-value fallback.  Only an
    interior value indistinguishable from the requested level uses the declared
    deterministic pairing.  The returned masks distinguish resolved cells from
    those tie-broken cells.

    This function returns contour geometry only.  It neither imports nor calls
    the polygonal clipped-support integration path.
    """
    values = jnp.asarray(values)
    radial = jnp.asarray(radial, dtype=values.dtype)
    vertical = jnp.asarray(vertical, dtype=values.dtype)
    level = jnp.asarray(level, dtype=values.dtype)
    spline = fit_tensor_spline(radial, vertical, values)

    corners = jnp.stack(
        (
            values[:-1, :-1],
            values[:-1, 1:],
            values[1:, 1:],
            values[1:, :-1],
        ),
        axis=-1,
    )
    corner_delta = corners - level
    edge_start_delta = corner_delta
    edge_end_delta = jnp.roll(corner_delta, shift=-1, axis=-1)
    edge_crossing = (edge_start_delta >= 0.0) != (edge_end_delta >= 0.0)
    crossing_count = jnp.sum(edge_crossing, axis=-1, dtype=jnp.int32)

    edge_start, edge_end = _structured_cell_edges(radial, vertical)
    edge_vector = edge_end - edge_start
    lower = jnp.zeros(edge_crossing.shape, dtype=values.dtype)
    upper = jnp.ones(edge_crossing.shape, dtype=values.dtype)
    lower_value = edge_start_delta

    def bisect(_iteration, state):
        low, high, low_value = state
        middle = 0.5 * (low + high)
        point = edge_start + middle[..., None] * edge_vector
        middle_value = spline(point[..., 0], point[..., 1]) - level
        same_side = (low_value >= 0.0) == (middle_value >= 0.0)
        next_low = jnp.where(same_side, middle, low)
        next_high = jnp.where(same_side, high, middle)
        next_value = jnp.where(same_side, middle_value, low_value)
        return next_low, next_high, next_value

    lower, upper, _ = jax.lax.fori_loop(
        0, bisection_steps, bisect, (lower, upper, lower_value)
    )
    edge_parameter = 0.5 * (lower + upper)
    edge_point = edge_start + edge_parameter[..., None] * edge_vector
    edge_evaluation = spline.evaluate(edge_point[..., 0], edge_point[..., 1])
    gradient_norm = jnp.hypot(
        edge_evaluation.radial_derivative,
        edge_evaluation.vertical_derivative,
    )
    safe_gradient_norm = jnp.where(gradient_norm > 0.0, gradient_norm, 1.0)
    edge_tangent = jnp.stack(
        (
            -edge_evaluation.vertical_derivative / safe_gradient_norm,
            edge_evaluation.radial_derivative / safe_gradient_norm,
        ),
        axis=-1,
    )

    radial_low = jnp.broadcast_to(radial[:-1][None, :], crossing_count.shape)
    radial_high = jnp.broadcast_to(radial[1:][None, :], crossing_count.shape)
    vertical_low = jnp.broadcast_to(vertical[:-1, None], crossing_count.shape)
    vertical_high = jnp.broadcast_to(vertical[1:, None], crossing_count.shape)
    centre_radial = 0.5 * (radial_low + radial_high)
    centre_vertical = 0.5 * (vertical_low + vertical_high)

    def locate_saddle(_iteration, point):
        evaluation = spline.evaluate(point[..., 0], point[..., 1])
        determinant = (
            evaluation.radial_second_derivative * evaluation.vertical_second_derivative
            - evaluation.mixed_derivative**2
        )
        safe_determinant = jnp.where(
            jnp.abs(determinant) > jnp.finfo(values.dtype).tiny,
            determinant,
            1.0,
        )
        radial_step = (
            evaluation.vertical_second_derivative * evaluation.radial_derivative
            - evaluation.mixed_derivative * evaluation.vertical_derivative
        ) / safe_determinant
        vertical_step = (
            evaluation.radial_second_derivative * evaluation.vertical_derivative
            - evaluation.mixed_derivative * evaluation.radial_derivative
        ) / safe_determinant
        return jnp.stack(
            (
                jnp.clip(point[..., 0] - radial_step, radial_low, radial_high),
                jnp.clip(point[..., 1] - vertical_step, vertical_low, vertical_high),
            ),
            axis=-1,
        )

    saddle_point = jax.lax.fori_loop(
        0,
        saddle_steps,
        locate_saddle,
        jnp.stack((centre_radial, centre_vertical), axis=-1),
    )
    saddle_evaluation = spline.evaluate(saddle_point[..., 0], saddle_point[..., 1])
    saddle_hessian_determinant = (
        saddle_evaluation.radial_second_derivative
        * saddle_evaluation.vertical_second_derivative
        - saddle_evaluation.mixed_derivative**2
    )
    minimum_spacing = jnp.minimum(
        radial_high - radial_low, vertical_high - vertical_low
    )
    field_scale = jnp.maximum(jnp.max(jnp.abs(corner_delta), axis=-1), 1.0)
    gradient_tolerance = (
        jnp.sqrt(jnp.finfo(values.dtype).eps) * field_scale / minimum_spacing
    )
    saddle_gradient = jnp.hypot(
        saddle_evaluation.radial_derivative,
        saddle_evaluation.vertical_derivative,
    )
    saddle_stationary = (
        (saddle_hessian_determinant < 0.0)
        & (saddle_gradient <= gradient_tolerance)
        & jnp.isfinite(saddle_evaluation.value)
    )
    centre_value = spline(centre_radial, centre_vertical)
    decision_delta = jnp.where(
        saddle_stationary, saddle_evaluation.value - level, centre_value - level
    )
    decision_tolerance = 128.0 * jnp.finfo(values.dtype).eps * field_scale
    decision_tie = jnp.abs(decision_delta) <= decision_tolerance
    same_side_as_first_corner = jnp.where(
        decision_tie,
        True,
        (decision_delta >= 0.0) == (corner_delta[..., 0] >= 0.0),
    )

    packed = _ordered_crossing_indices(edge_crossing)
    regular_pairs = jnp.stack(
        (
            jnp.stack((packed[..., 0], packed[..., 1]), axis=-1),
            jnp.zeros(packed.shape[:-1] + (2,), dtype=jnp.int32),
        ),
        axis=-2,
    )
    paired_around_second_and_fourth = jnp.asarray(((0, 1), (2, 3)), jnp.int32)
    paired_around_first_and_third = jnp.asarray(((3, 0), (1, 2)), jnp.int32)
    saddle_pairs = jnp.where(
        same_side_as_first_corner[..., None, None],
        paired_around_second_and_fourth,
        paired_around_first_and_third,
    )
    ambiguous = crossing_count == 4
    pair_indices = jnp.where(ambiguous[..., None, None], saddle_pairs, regular_pairs)
    segment_valid = jnp.stack((crossing_count == 2, jnp.zeros_like(ambiguous)), axis=-1)
    segment_valid = jnp.where(
        ambiguous[..., None], jnp.ones_like(segment_valid), segment_valid
    )

    segment_point = _paired_edge_values(edge_point, pair_indices)
    segment_tangent = _paired_edge_values(edge_tangent, pair_indices)
    chord = segment_point[..., 1, :] - segment_point[..., 0, :]
    reverse = (
        jnp.sum(
            chord * (segment_tangent[..., 0, :] + segment_tangent[..., 1, :]), axis=-1
        )
        < 0.0
    )
    endpoint_order = jnp.where(
        reverse[..., None], jnp.asarray((1, 0)), jnp.asarray((0, 1))
    )
    segment_point = jnp.take_along_axis(
        segment_point, endpoint_order[..., None], axis=-2
    )
    segment_tangent = jnp.take_along_axis(
        segment_tangent, endpoint_order[..., None], axis=-2
    )
    chord = segment_point[..., 1, :] - segment_point[..., 0, :]
    chord_length = jnp.linalg.norm(chord, axis=-1)
    minimum_scale = 0.25 * chord_length
    start_scale = jnp.maximum(
        jnp.sum(chord * segment_tangent[..., 0, :], axis=-1), minimum_scale
    )
    end_scale = jnp.maximum(
        jnp.sum(chord * segment_tangent[..., 1, :], axis=-1), minimum_scale
    )
    start_control = (
        segment_point[..., 0, :]
        + start_scale[..., None] * segment_tangent[..., 0, :] / 3.0
    )
    end_control = (
        segment_point[..., 1, :]
        - end_scale[..., None] * segment_tangent[..., 1, :] / 3.0
    )
    controls = jnp.stack(
        (
            segment_point[..., 0, :],
            start_control,
            end_control,
            segment_point[..., 1, :],
        ),
        axis=-2,
    )

    canonical_edge_parameter = jnp.where(edge_crossing, edge_parameter, 0.0)
    canonical_edge_point = jnp.where(edge_crossing[..., None], edge_point, 0.0)
    canonical_edge_tangent = jnp.where(edge_crossing[..., None], edge_tangent, 0.0)
    canonical_pair_indices = jnp.where(segment_valid[..., None], pair_indices, 0)
    canonical_segment_point = jnp.where(
        segment_valid[..., None, None], segment_point, 0.0
    )
    canonical_segment_tangent = jnp.where(
        segment_valid[..., None, None], segment_tangent, 0.0
    )
    canonical_controls = jnp.where(segment_valid[..., None, None], controls, 0.0)
    canonical_saddle_point = jnp.where(ambiguous[..., None], saddle_point, 0.0)
    canonical_saddle_value = jnp.where(ambiguous, saddle_evaluation.value, 0.0)

    return {
        "edge_crossing": edge_crossing,
        "edge_parameter": canonical_edge_parameter,
        "edge_crossing_rz": canonical_edge_point,
        "edge_tangent_rz": canonical_edge_tangent,
        "segment_valid": segment_valid,
        "segment_edge_indices": canonical_pair_indices,
        "segment_endpoints_rz": canonical_segment_point,
        "segment_endpoint_tangents_rz": canonical_segment_tangent,
        "segment_controls_rz": canonical_controls,
        "cell_crossing_count": crossing_count,
        "ambiguous_saddle": ambiguous,
        "ambiguous_resolved": ambiguous & ~decision_tie,
        "ambiguous_tie_broken": ambiguous & decision_tie,
        "saddle_stationary": ambiguous & saddle_stationary,
        "saddle_rz": canonical_saddle_point,
        "saddle_value": canonical_saddle_value,
        "well_formed": jnp.all(
            (crossing_count == 0) | (crossing_count == 2) | (crossing_count == 4)
        ),
    }
