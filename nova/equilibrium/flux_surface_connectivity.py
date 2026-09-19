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

Component labelling exposes the topology distinction explicitly. The scan path
labels four-connected cells on a rectangular tensor lattice. The plasma mesh
shipped by the package is instead hexagonal and half-offset, so its component
path consumes the centre-first six-neighbour rings from
:func:`nova.geometry.hexstencil.hex_stencil` or the identically packed trimmed
mesh tessellation. Six-neighbour adjacency is the bulk topology, but a graph edge
is not admissible merely because both cell centroids lie inside a binding level:
the shared physical edge must contain a portion strictly on the magnetic-axis
side of that level. The admissibility test evaluates the global tensor spline,
so a contour that closes a saddle neck between centroids cannot bridge public and
private flux through a cut cell.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from nova.equilibrium.morphology import _dilate4 as _dilate4
from nova.equilibrium.parallel_components import (
    label_parallel_connected_components_with_steps,
    label_parallel_graph_components_with_steps,
)
from nova.linalg.split_spline import fit_split_spline
from nova.linalg.tensor_spline import TensorBSpline, fit_tensor_spline


_SQRT2 = 2.0**0.5

__all__ = [
    "flood_fill_core",
    "flood_fill_core_with_steps",
    "hex_edge_admissibility",
    "label_hex_connected_components",
    "label_hex_connected_components_with_steps",
    "label_saddle_aware_hex_connected_components",
    "label_saddle_aware_hex_connected_components_with_steps",
    "label_connected_components",
    "label_connected_components_with_steps",
    "label_connected_components_fixed_point",
    "census_stationary_receipt",
    "polish_census_stationary_points",
    "polish_stationary_points",
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


def _segment_propagation(confined: jnp.ndarray):
    """Return the minimum-label propagation step along rows then columns."""

    def propagate(labels: jnp.ndarray) -> jnp.ndarray:
        row_filled = _fill_label_segments(labels, confined, axis=1)
        return _fill_label_segments(row_filled, confined, axis=0)

    return propagate


def _iterate_component_labels(
    confined: jnp.ndarray, n_iter: int, propagate
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Propagate canonical minimum labels with fixed-cap masked execution."""
    initial = jnp.arange(confined.size, dtype=jnp.int32).reshape(confined.shape) + 1
    initial = jnp.where(confined, initial, 0)

    def body(_iteration, state):
        labels, previous, step = state
        active = jnp.any(labels != previous)
        propagated = propagate(labels)
        return (
            jnp.where(active, propagated, labels),
            jnp.where(active, labels, previous),
            step + active.astype(jnp.int32),
        )

    labels, _previous, step = jax.lax.fori_loop(
        0,
        n_iter,
        body,
        (initial, jnp.zeros_like(initial), jnp.asarray(0, dtype=jnp.int32)),
    )
    return labels, step


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

    The exact rectangular hook-and-compress primitive leaves every component
    carrying its minimum member index, with zero reserved for cells outside
    ``confined``. ``n_iter`` remains the caller's static safety cap. The pass
    itself reaches and confirms its fixed point within
    ``ceil(log2(cell_count)) + 2`` trips, so a larger caller cap does not enlarge
    the compiled schedule.

    Labels are stable across execution order and need not be consecutive. The
    scalar step count is an execution diagnostic.
    """

    schedule_limit = (max(confined.size, 1) - 1).bit_length() + 2
    if n_iter < schedule_limit:
        return _iterate_component_labels(
            confined, n_iter, _segment_propagation(confined)
        )
    labels, steps, _settled = label_parallel_connected_components_with_steps(confined)
    return labels, steps


@partial(jax.jit, static_argnums=(1,))
def label_connected_components_fixed_point(
    confined: jnp.ndarray, n_iter: int
) -> jnp.ndarray:
    """Return canonical labels from masked segment propagation alone.

    ``label_connected_components`` hands a sufficient caller cap to the
    fixed-shape parallel pass, so its labels are not an independent authority
    on their own. This entry point never routes: it runs the masked segment
    propagation for ``n_iter`` iterations, which is the caller-capped fixed
    point an identity gate compares the parallel pass against.
    """
    labels, _steps = _iterate_component_labels(
        confined, n_iter, _segment_propagation(confined)
    )
    return labels


@partial(jax.jit, static_argnums=(1,))
def label_connected_components(confined: jnp.ndarray, n_iter: int) -> jnp.ndarray:
    """Return canonical labels for every 4-connected confined component."""
    labels, _steps = label_connected_components_with_steps(confined, n_iter)
    return labels


def _propagate_hex_ring_minima(
    labels: jnp.ndarray, confined: jnp.ndarray, rings: jnp.ndarray
) -> jnp.ndarray:
    """Propagate labels through centre-first six-neighbour rings once."""
    flat_labels = labels.reshape(-1)
    flat_confined = confined.reshape(-1)
    ring_is_open = flat_confined[rings] & flat_confined[rings[:, :1]]
    sentinel = jnp.asarray(jnp.iinfo(labels.dtype).max, dtype=labels.dtype)
    ring_labels = jnp.where(ring_is_open, flat_labels[rings], sentinel)
    ring_minimum = jnp.min(ring_labels, axis=1, keepdims=True)
    propagated = jnp.where(ring_is_open, ring_minimum, sentinel)
    return flat_labels.at[rings].min(propagated).reshape(labels.shape)


def _propagate_admissible_hex_minima(
    labels: jnp.ndarray,
    confined: jnp.ndarray,
    rings: jnp.ndarray,
    link_admissible: jnp.ndarray,
) -> jnp.ndarray:
    """Propagate labels once through admissible centre-to-neighbour links."""
    flat_labels = labels.reshape(-1)
    flat_confined = confined.reshape(-1)
    centre_is_open = flat_confined[rings[:, :1]]
    ring_is_open = jnp.concatenate(
        (
            centre_is_open,
            link_admissible[:, 1:] & centre_is_open & flat_confined[rings[:, 1:]],
        ),
        axis=1,
    )
    sentinel = jnp.asarray(jnp.iinfo(labels.dtype).max, dtype=labels.dtype)
    ring_labels = jnp.where(ring_is_open, flat_labels[rings], sentinel)
    ring_minimum = jnp.min(ring_labels, axis=1, keepdims=True)
    propagated = jnp.where(ring_is_open, ring_minimum, sentinel)
    return flat_labels.at[rings].min(propagated).reshape(labels.shape)


@partial(jax.jit, static_argnums=(6,))
def hex_edge_admissibility(
    values: jnp.ndarray,
    radial: jnp.ndarray,
    vertical: jnp.ndarray,
    level: jnp.ndarray,
    axis_value: jnp.ndarray,
    shared_edge_rz: jnp.ndarray,
    stationary_steps: int = 8,
    edge_values: jnp.ndarray | None = None,
    surface: TensorBSpline | None = None,
) -> jnp.ndarray:
    """Return a fixed-shape mask for hex links open at ``level``.

    ``shared_edge_rz`` has shape ``(n_ring, 7, 2, 2)``: ring, centre-first
    slot, endpoint, and ``(R, Z)`` coordinate. Slot zero is padding and is
    returned open. Every other slot describes the physical edge shared by the
    ring centre and that neighbour.

    A link is open only if some portion of its shared edge is strictly on the
    same side of ``level`` as ``axis_value``. Endpoints and fixed-iteration
    stationary searches from seven edge parameters evaluate the global tensor
    B-spline restriction; thus a tangent contact at the binding level has zero
    measure on the axis side and closes the link. No per-cell reconstruction or
    compacted gather is used.
    """
    values = jnp.asarray(values)
    radial = jnp.asarray(radial, dtype=values.dtype)
    vertical = jnp.asarray(vertical, dtype=values.dtype)
    level = jnp.asarray(level, dtype=values.dtype)
    axis_value = jnp.asarray(axis_value, dtype=values.dtype)
    shared_edge_rz = jnp.asarray(shared_edge_rz, dtype=values.dtype)
    if edge_values is not None:
        samples = jnp.asarray(edge_values, dtype=values.dtype)
        if samples.shape[-1] != 3:
            raise ValueError("carrier edge values must contain start, midpoint, end")
        start, midpoint, end = samples[..., 0], samples[..., 1], samples[..., 2]
        curvature = 2.0 * (start + end - 2.0 * midpoint)
        slope = end - start - curvature
        safe_curvature = jnp.where(
            jnp.abs(curvature) > jnp.finfo(values.dtype).tiny, curvature, 1.0
        )
        stationary = jnp.clip(-slope / (2.0 * safe_curvature), 0.0, 1.0)
        stationary_value = curvature * stationary**2 + slope * stationary + start
        samples = jnp.concatenate((samples, stationary_value[..., None]), axis=-1)
        side = jnp.where(axis_value >= level, 1.0, -1.0)
        field_scale = jnp.maximum(jnp.max(jnp.abs(values - level)), 1.0)
        strict_tolerance = 128.0 * jnp.finfo(values.dtype).eps * field_scale
        open_link = jnp.max(side * (samples - level), axis=-1) > strict_tolerance
        return open_link.at[:, 0].set(True)

    spline = fit_tensor_spline(radial, vertical, values) if surface is None else surface

    edge_start = shared_edge_rz[..., 0, :]
    edge_end = shared_edge_rz[..., 1, :]
    padding_point = jnp.stack((radial[0], vertical[0]))
    edge_start = edge_start.at[:, 0].set(padding_point)
    edge_end = edge_end.at[:, 0].set(padding_point)
    edge_vector = edge_end - edge_start

    def locate_stationary(_iteration, parameter):
        point = (
            edge_start[..., None, :] + parameter[..., None] * edge_vector[..., None, :]
        )
        evaluation = spline.evaluate(point[..., 0], point[..., 1])
        first = (
            evaluation.radial_derivative * edge_vector[..., None, 0]
            + evaluation.vertical_derivative * edge_vector[..., None, 1]
        )
        second = (
            evaluation.radial_second_derivative * edge_vector[..., None, 0] ** 2
            + 2.0
            * evaluation.mixed_derivative
            * edge_vector[..., None, 0]
            * edge_vector[..., None, 1]
            + evaluation.vertical_second_derivative * edge_vector[..., None, 1] ** 2
        )
        safe_second = jnp.where(
            jnp.abs(second) > jnp.finfo(values.dtype).tiny, second, 1.0
        )
        candidate = jnp.clip(parameter - first / safe_second, 0.0, 1.0)
        return jnp.where(
            jnp.abs(second) > jnp.finfo(values.dtype).tiny, candidate, parameter
        )

    seeds = jnp.linspace(0.0, 1.0, 7, dtype=values.dtype)
    initial = jnp.broadcast_to(seeds, edge_start.shape[:-1] + seeds.shape)
    stationary = jax.lax.fori_loop(0, stationary_steps, locate_stationary, initial)
    stationary_point = (
        edge_start[..., None, :] + stationary[..., None] * edge_vector[..., None, :]
    )
    samples = jnp.concatenate(
        (
            spline(edge_start[..., 0], edge_start[..., 1])[..., None],
            spline(edge_end[..., 0], edge_end[..., 1])[..., None],
            spline(stationary_point[..., 0], stationary_point[..., 1]),
        ),
        axis=-1,
    )
    side = jnp.where(axis_value >= level, 1.0, -1.0)
    field_scale = jnp.maximum(jnp.max(jnp.abs(values - level)), 1.0)
    strict_tolerance = 128.0 * jnp.finfo(values.dtype).eps * field_scale
    open_link = jnp.max(side * (samples - level), axis=-1) > strict_tolerance
    return open_link.at[:, 0].set(True)


@partial(jax.jit, static_argnums=(2,))
def label_hex_connected_components_with_steps(
    confined: jnp.ndarray, rings: jnp.ndarray, n_iter: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return canonical labels for components on a hexagonal plasma mesh.

    ``rings`` follows :func:`nova.geometry.hexstencil.hex_stencil`: every row
    contains one centre flat index followed by its six touching neighbours.
    The same centre-first rows are produced by the trimmed plasma mesh's
    tessellation, so component labelling consumes the topology used by the null
    search and quadratic derivative operator instead of interpreting the array
    carrying the cells as a rectangular raster.

    Every confined cell starts with its one-based flat index. The explicit-graph
    hook-and-compress pass consumes the centre-first six-neighbour rows directly.
    A ring whose centre is not confined is closed: its neighbours cannot connect
    through a missing cell. Zero remains reserved for unconfined cells.
    ``n_iter`` remains the caller's static cap; the pass needs at most
    ``ceil(log2(cell_count)) + 2`` trips.
    """
    labels, steps, _settled = label_parallel_graph_components_with_steps(
        confined,
        rings,
        jnp.ones(rings.shape, dtype=bool),
        n_iter,
    )
    return labels, steps


@partial(jax.jit, static_argnums=(2,))
def label_hex_connected_components(
    confined: jnp.ndarray, rings: jnp.ndarray, n_iter: int
) -> jnp.ndarray:
    """Return canonical labels for every six-neighbour confined component."""
    labels, _steps = label_hex_connected_components_with_steps(confined, rings, n_iter)
    return labels


@partial(jax.jit, static_argnums=(3,))
def label_saddle_aware_hex_connected_components_with_steps(
    confined: jnp.ndarray,
    rings: jnp.ndarray,
    link_admissible: jnp.ndarray,
    n_iter: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Label six-neighbour components without crossing closed level edges.

    The graph remains the supplied hex ring graph in the bulk. The fixed-shape
    ``link_admissible`` mask has the same cells-by-seven packing as ``rings``;
    only its six centre-to-neighbour entries control propagation.
    """
    labels, steps, _settled = label_parallel_graph_components_with_steps(
        confined,
        rings,
        link_admissible,
        n_iter,
    )
    return labels, steps


@partial(jax.jit, static_argnums=(3,))
def label_saddle_aware_hex_connected_components(
    confined: jnp.ndarray,
    rings: jnp.ndarray,
    link_admissible: jnp.ndarray,
    n_iter: int,
) -> jnp.ndarray:
    """Return saddle-aware labels on a six-neighbour bulk graph."""
    labels, _steps = label_saddle_aware_hex_connected_components_with_steps(
        confined, rings, link_admissible, n_iter
    )
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


# A cell boundary can meet one edge of the lattice more than once, so a node id
# names a crossing on an edge and not the edge alone.  The node of a crossing is
# ``EDGE_CROSSING_CAPACITY * edge_id + rank_along_edge``, which leaves the four
# crossing slots of one cell a distinct id each.
EDGE_CROSSING_CAPACITY = 4


def _structured_edge_nodes(radial_size: int, vertical_size: int) -> jnp.ndarray:
    """Return canonical global node ids for each cell's four physical edges."""
    cell_radial = radial_size - 1
    cell_vertical = vertical_size - 1
    row = jnp.arange(cell_vertical, dtype=jnp.int32)[:, None]
    column = jnp.arange(cell_radial, dtype=jnp.int32)[None, :]
    horizontal_count = vertical_size * cell_radial
    bottom = row * cell_radial + column
    right = horizontal_count + row * radial_size + column + 1
    top = (row + 1) * cell_radial + column
    left = horizontal_count + row * radial_size + column
    return jnp.stack((bottom, right, top, left), axis=-1)


@partial(jax.jit, static_argnums=(5,))
def _polish_stationary_points_in_bounds(
    spline,
    seed_rz: jnp.ndarray,
    valid: jnp.ndarray,
    lower_rz: jnp.ndarray,
    upper_rz: jnp.ndarray,
    stationary_steps: int,
) -> dict[str, jnp.ndarray]:
    """Polish fixed-slot seeds within caller-supplied coordinate bounds.

    The nonlinear solve executes its literal shared safety cap while masking
    lanes that have reached the gradient tolerance. ``custom_root``
    differentiates the stationary equation rather than the iteration history,
    so reverse-mode derivatives remain valid independently of the primal trips.
    """
    seed_rz = jnp.asarray(seed_rz, dtype=spline.coefficients.dtype)
    valid = jnp.asarray(valid, dtype=bool)
    lower_rz = jnp.asarray(lower_rz, dtype=seed_rz.dtype)
    upper_rz = jnp.asarray(upper_rz, dtype=seed_rz.dtype)
    radial_seed = seed_rz[..., 0]
    vertical_seed = seed_rz[..., 1]
    seed_in_domain = (
        valid
        & jnp.isfinite(radial_seed)
        & jnp.isfinite(vertical_seed)
        & (radial_seed >= spline.radial[0])
        & (radial_seed <= spline.radial[-1])
        & (vertical_seed >= spline.vertical[0])
        & (vertical_seed <= spline.vertical[-1])
    )
    radial_low = lower_rz[..., 0]
    radial_high = upper_rz[..., 0]
    vertical_low = lower_rz[..., 1]
    vertical_high = upper_rz[..., 1]
    radial_seed = jnp.where(
        jnp.isfinite(radial_seed),
        jnp.clip(radial_seed, radial_low, radial_high),
        radial_low,
    )
    vertical_seed = jnp.where(
        jnp.isfinite(vertical_seed),
        jnp.clip(vertical_seed, vertical_low, vertical_high),
        vertical_low,
    )

    field_scale = jnp.maximum(
        jnp.max(spline.coefficients) - jnp.min(spline.coefficients),
        jnp.finfo(seed_rz.dtype).tiny,
    )
    minimum_spacing = jnp.minimum(
        radial_high - radial_low, vertical_high - vertical_low
    )
    gradient_tolerance = (
        128.0 * jnp.finfo(seed_rz.dtype).eps * field_scale / minimum_spacing
    )

    initial = jnp.stack((radial_seed, vertical_seed), axis=-1)

    def stationary_equation(point):
        evaluation = spline.evaluate(point[..., 0], point[..., 1])
        gradient = jnp.stack(
            (evaluation.radial_derivative, evaluation.vertical_derivative), axis=-1
        )
        return jnp.where(seed_in_domain[..., None], gradient, point - initial)

    def solve(_equation, starting_point):
        initial_count = jnp.zeros(valid.shape, dtype=jnp.int32)

        def needs_step(point):
            evaluation = spline.evaluate(point[..., 0], point[..., 1])
            gradient_norm = jnp.hypot(
                evaluation.radial_derivative, evaluation.vertical_derivative
            )
            return (
                seed_in_domain
                & jnp.isfinite(gradient_norm)
                & (gradient_norm > gradient_tolerance)
            )

        def newton_step(_iteration, state):
            point, count = state
            active = needs_step(point)
            evaluation = spline.evaluate(point[..., 0], point[..., 1])
            determinant = (
                evaluation.radial_second_derivative
                * evaluation.vertical_second_derivative
                - evaluation.mixed_derivative**2
            )
            safe_determinant = jnp.where(
                jnp.abs(determinant) > jnp.finfo(point.dtype).tiny,
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
            candidate = jnp.stack(
                (
                    jnp.clip(point[..., 0] - radial_step, radial_low, radial_high),
                    jnp.clip(
                        point[..., 1] - vertical_step, vertical_low, vertical_high
                    ),
                ),
                axis=-1,
            )
            next_point = jnp.where(active[..., None], candidate, point)
            return next_point, count + active.astype(jnp.int32)

        position, count = jax.lax.fori_loop(
            0,
            stationary_steps,
            newton_step,
            (starting_point, initial_count),
        )
        return position, count.astype(starting_point.dtype)

    def tangent_solve(linear_equation, right_hand_side):
        radial_basis = jnp.zeros_like(right_hand_side).at[..., 0].set(1.0)
        vertical_basis = jnp.zeros_like(right_hand_side).at[..., 1].set(1.0)
        radial_column = linear_equation(radial_basis)
        vertical_column = linear_equation(vertical_basis)
        hessian = jnp.stack((radial_column, vertical_column), axis=-1)
        determinant = (
            hessian[..., 0, 0] * hessian[..., 1, 1]
            - hessian[..., 0, 1] * hessian[..., 1, 0]
        )
        safe_determinant = jnp.where(
            jnp.abs(determinant) > jnp.finfo(right_hand_side.dtype).tiny,
            determinant,
            1.0,
        )
        return jnp.stack(
            (
                (
                    hessian[..., 1, 1] * right_hand_side[..., 0]
                    - hessian[..., 0, 1] * right_hand_side[..., 1]
                )
                / safe_determinant,
                (
                    hessian[..., 0, 0] * right_hand_side[..., 1]
                    - hessian[..., 1, 0] * right_hand_side[..., 0]
                )
                / safe_determinant,
            ),
            axis=-1,
        )

    position, iteration_count = jax.lax.custom_root(
        stationary_equation,
        initial,
        solve,
        tangent_solve,
        has_aux=True,
    )
    evaluation = spline.evaluate(position[..., 0], position[..., 1])
    determinant = (
        evaluation.radial_second_derivative * evaluation.vertical_second_derivative
        - evaluation.mixed_derivative**2
    )
    gradient_norm = jnp.hypot(
        evaluation.radial_derivative,
        evaluation.vertical_derivative,
    )
    finite_result = (
        jnp.isfinite(evaluation.value)
        & jnp.isfinite(gradient_norm)
        & jnp.isfinite(determinant)
    )
    in_domain = (
        seed_in_domain
        & (position[..., 0] >= spline.radial[0])
        & (position[..., 0] <= spline.radial[-1])
        & (position[..., 1] >= spline.vertical[0])
        & (position[..., 1] <= spline.vertical[-1])
    )
    converged = in_domain & finite_result & (gradient_norm <= gradient_tolerance)
    hessian_type = jnp.sign(determinant).astype(jnp.int8)
    gradient = jnp.stack(
        (evaluation.radial_derivative, evaluation.vertical_derivative), axis=-1
    )
    hessian = jnp.stack(
        (
            jnp.stack(
                (evaluation.radial_second_derivative, evaluation.mixed_derivative),
                axis=-1,
            ),
            jnp.stack(
                (evaluation.mixed_derivative, evaluation.vertical_second_derivative),
                axis=-1,
            ),
        ),
        axis=-2,
    )

    return {
        "position_rz": jnp.where(valid[..., None], position, 0.0),
        "value": jnp.where(valid, evaluation.value, 0.0),
        "gradient_norm": jnp.where(valid, gradient_norm, 0.0),
        "gradient": jnp.where(valid[..., None], gradient, 0.0),
        "hessian": jnp.where(valid[..., None, None], hessian, 0.0),
        "hessian_type": jnp.where(valid, hessian_type, 0),
        "converged": valid & converged,
        "in_domain": valid & in_domain,
        "iteration_count": jnp.where(valid, iteration_count, 0).astype(jnp.int32),
    }


@partial(jax.jit, static_argnums=(3,))
def polish_stationary_points(
    spline,
    seed_rz: jnp.ndarray,
    valid: jnp.ndarray,
    stationary_steps: int = 8,
) -> dict[str, jnp.ndarray]:
    """Polish fixed-slot stationary-point seeds on a global tensor spline.

    Every slot is updated independently within the spline domain, so polishing
    one candidate never perturbs another candidate's primal coordinates.
    ``valid`` has the fixed leading shape of ``seed_rz`` and inactive slots are
    returned as canonical exact-zero padding. Hessian type is ``-1`` for a
    saddle, ``0`` for a degenerate Hessian, and ``1`` for an extremum.  The
    supplied seeds are also the warm-tracking state: a converged seed takes zero
    active updates while the kernel still executes the shared safety cap.
    """
    lower_rz = jnp.stack((spline.radial[0], spline.vertical[0]))
    upper_rz = jnp.stack((spline.radial[-1], spline.vertical[-1]))
    return _polish_stationary_points_in_bounds(
        spline, seed_rz, valid, lower_rz, upper_rz, stationary_steps
    )


def _coordinate_grids(
    radial: jnp.ndarray, vertical: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return paired sample coordinates for axes or half-offset arrays."""
    if radial.ndim == 1 and vertical.ndim == 1:
        return jnp.meshgrid(radial, vertical)
    return jnp.broadcast_arrays(radial, vertical)


def _minimum_cell_pitch(
    radial: jnp.ndarray, vertical: jnp.ndarray, valid: jnp.ndarray
) -> jnp.ndarray:
    """Return the shortest adjacent sample spacing on a structured lattice."""
    radial_grid, vertical_grid = _coordinate_grids(radial, vertical)
    horizontal = jnp.hypot(
        jnp.diff(radial_grid, axis=1), jnp.diff(vertical_grid, axis=1)
    )
    horizontal = jnp.where(valid[:, 1:] & valid[:, :-1], horizontal, jnp.inf)
    vertical_step = jnp.hypot(
        jnp.diff(radial_grid, axis=0), jnp.diff(vertical_grid, axis=0)
    )
    vertical_step = jnp.where(valid[1:] & valid[:-1], vertical_step, jnp.inf)
    pitch = jnp.minimum(jnp.min(horizontal), jnp.min(vertical_step))
    return jnp.where(jnp.isfinite(pitch), pitch, 1.0)


def _census_local_values(
    values: jnp.ndarray,
    radial: jnp.ndarray,
    vertical: jnp.ndarray,
    sample_valid: jnp.ndarray,
    census_position: jnp.ndarray,
    evaluation_position: jnp.ndarray,
) -> jnp.ndarray:
    """Evaluate local census quadratics as confidence evidence.

    ``Null2D`` assigns each candidate to a centre-first, six-neighbour cluster
    and fits a quadratic in locally normalised coordinates. Recovering the
    same six axial offsets around that centre reproduce the interpolation on
    the raster and on the carrier's half-offset embedding. These values measure
    agreement with the admitting census; they do not author a published
    stationary-point position or field value.
    """
    radial_grid, vertical_grid = _coordinate_grids(radial, vertical)
    coordinates = jnp.stack((radial_grid, vertical_grid), axis=-1).reshape((-1, 2))
    flattened_values = values.reshape(-1)
    flattened_valid = sample_valid.reshape(-1)
    row_offset = jnp.asarray((0, 0, -1, -1, 0, 1, 1), dtype=jnp.int32)
    column_offset = jnp.asarray((0, -1, 0, 1, 1, 0, -1), dtype=jnp.int32)

    def evaluate(census_rz, evaluation_rz):
        centre_distance = jnp.sum((coordinates - census_rz) ** 2, axis=-1)
        centre_index = jnp.argmin(jnp.where(flattened_valid, centre_distance, jnp.inf))
        centre = coordinates[centre_index]
        centre_row = centre_index // values.shape[1]
        centre_column = centre_index % values.shape[1]
        row = jnp.clip(centre_row + row_offset, 0, values.shape[0] - 1)
        column = jnp.clip(centre_column + column_offset, 0, values.shape[1] - 1)
        indices = row * values.shape[1] + column
        indices = jnp.where(flattened_valid[indices], indices, centre_index)
        cluster = coordinates[indices]
        offset = cluster - centre
        scale = jnp.max(jnp.abs(offset), axis=0)
        scale = jnp.where(scale > 0.0, scale, 1.0)
        local = offset / scale
        design = jnp.column_stack(
            (
                local[:, 0] ** 2,
                local[:, 1] ** 2,
                local[:, 0],
                local[:, 1],
                local[:, 0] * local[:, 1],
                jnp.ones(7, dtype=values.dtype),
            )
        )
        coefficient = jnp.linalg.lstsq(design, flattened_values[indices])[0]
        query = (evaluation_rz - centre) / scale
        basis = jnp.stack(
            (
                query[0] ** 2,
                query[1] ** 2,
                query[0],
                query[1],
                query[0] * query[1],
                jnp.asarray(1.0, dtype=values.dtype),
            )
        )
        return basis @ coefficient

    return jax.vmap(evaluate)(census_position, evaluation_position)


def census_stationary_receipt(
    values: jnp.ndarray,
    radial: jnp.ndarray,
    vertical: jnp.ndarray,
    census: dict[str, jnp.ndarray],
    selected_index: jnp.ndarray,
    selected_stationary: jnp.ndarray,
    selected_valid: jnp.ndarray,
) -> dict[str, jnp.ndarray]:
    """Project selected authoritative census rows into a read receipt.

    A structured fixed-design census has already polished every retained row
    on the complete-map tensor spline and banked the gradient, Hessian
    determinant, displacement, uncertainty, origin and multiplicity used to
    retain it.  Selecting two rows must not fit or polish the same map again.
    The local quadratic value remains an inexpensive independent interpolation
    check for the boundary deadband; split-spline conditioning belongs to a
    separately requested diagnostic rather than this production read.
    """
    values = jnp.asarray(values)
    radial = jnp.asarray(radial, dtype=values.dtype)
    vertical = jnp.asarray(vertical, dtype=values.dtype)
    selected_index = jnp.asarray(selected_index, dtype=jnp.int32)
    selected_stationary = jnp.asarray(selected_stationary, dtype=values.dtype)
    selected_valid = jnp.asarray(selected_valid, dtype=bool)
    kind = jnp.arange(2, dtype=jnp.int32)
    safe_index = jnp.maximum(selected_index, 0)
    retained_valid = census["retained_valid"][kind, safe_index]
    valid = selected_valid & (selected_index >= 0) & retained_valid

    def gather(name):
        return census[name][kind, safe_index]

    position = selected_stationary[:, :2]
    value = selected_stationary[:, 2]
    local_value = _census_local_values(
        values,
        radial,
        vertical,
        jnp.ones(values.shape, dtype=bool),
        position,
        position,
    )
    local_value = jnp.where(valid, local_value, jnp.nan)
    hessian_determinant = gather("retained_spline_hessian_determinant")
    spline_authored = valid & jnp.asarray(census["spline_authored"], dtype=bool)
    return {
        "converged": valid,
        "position_rz": position,
        "value": value,
        "gradient": gather("retained_spline_gradient"),
        "gradient_norm": gather("retained_spline_gradient_norm"),
        "hessian_determinant": hessian_determinant,
        "hessian_type": jnp.sign(hessian_determinant).astype(jnp.int8),
        "in_domain": valid,
        "local_value_evidence": local_value,
        "spline_value": value,
        "spline_authored": spline_authored,
        "complete_map": jnp.broadcast_to(
            jnp.asarray(census["spline_authored"], dtype=bool), valid.shape
        ),
        "value_replaced": jnp.zeros(valid.shape, dtype=bool),
        "census_position_rz": position,
        "selected_position_rz": position,
        "selected_value": value,
        "representative_origin_index": gather("retained_representative_origin_index"),
        "representative_origin_rz": gather("retained_representative_origin_rz"),
        "multiplicity": gather("retained_multiplicity"),
        "requested_displacement": gather("retained_requested_displacement"),
        "root_uncertainty": gather("retained_root_uncertainty"),
    }


@jax.jit
def polish_census_stationary_points(
    values: jnp.ndarray,
    radial: jnp.ndarray,
    vertical: jnp.ndarray,
    interface_value: jnp.ndarray,
    polarity: jnp.ndarray,
    selected_extremum: jnp.ndarray,
    selected_saddle: jnp.ndarray,
    sample_valid: jnp.ndarray | None = None,
    execute: jnp.ndarray = True,
    surface: TensorBSpline | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, dict[str, jnp.ndarray]]:
    """Apply fit-qualified tensor-spline polish to census-selected nulls.

    Candidate production, ranking, and count are complete before this function
    receives its two fixed slots.  Stationarity is measured by the fitted
    gradient made dimensionless with the coordinate span and field range.  Its
    tolerance is the larger of two independently derived floors.  The
    deterministic coefficient forward-error scale in the solved coordinates is
    ``n_active * condition_number * eps * field_range``.  The fit retains its
    column scales ``D``; multiplying that error by ``B' D^-1`` and dividing by
    ``field_range / span`` gives the dimensionless roundoff gradient floor.  The
    exterior level-squared basis is normalized by the field range before the
    solve, so its columns have the same units as the interior columns.  The
    representation floor
    treats the fit's sample RMS residual as unresolved field variation across
    one cell: ``(rms / field_range) / (cell_pitch / span)``. The representation
    floor is separately invariant to coordinate and field units. The complete
    scaled-coordinate construction makes the roundoff floor invariant as well,
    and the active coefficient count is evaluated separately at each slot.

    A seed already below that fitted-gradient floor takes zero active updates.
    Every other valid seed receives the shared fixed-cap Newton polish. Fit-value
    consistency has its own dimensionless floor: the larger of the
    normalized sample RMS and ``n_active * condition_number * eps``.  It is not
    divided by cell pitch because it bounds a value rather than a gradient.
    The difference from the census interpolation remains explicit confidence
    evidence but cannot override the complete map. The split fit and the
    seven-point census quadratic remain evidence only. For a complete structured map,
    an accepted result takes both its position and value from the map's global
    tensor spline. Unsupported sparse or half-offset carriers retain their
    census rows without claiming a polish. A fit or result outside the finite,
    Hessian-type, and stationarity gates retains the complete census row while
    preserving the attempted receipt.
    """
    values = jnp.asarray(values)
    radial = jnp.asarray(radial, dtype=values.dtype)
    vertical = jnp.asarray(vertical, dtype=values.dtype)
    interface_value = jnp.asarray(interface_value, dtype=values.dtype)
    if sample_valid is None:
        sample_valid = jnp.ones(values.shape, dtype=bool)
    else:
        sample_valid = jnp.asarray(sample_valid, dtype=bool)
        if sample_valid.shape != values.shape:
            raise ValueError("sample_valid must have the values shape")
    level_set = jnp.asarray(polarity, dtype=values.dtype) * (interface_value - values)
    finite_fit = (
        jnp.asarray(execute, dtype=bool)
        & jnp.isfinite(interface_value)
        & jnp.all(jnp.where(sample_valid, jnp.isfinite(values), True))
    )
    evidence_spline = fit_split_spline(
        radial,
        vertical,
        values,
        level_set,
        valid=sample_valid,
        execute=finite_fit,
    )
    tensor_supported = (
        radial.ndim == 1
        and vertical.ndim == 1
        and radial.size >= 4
        and vertical.size >= 4
    )
    if tensor_supported:
        authority_spline = (
            fit_tensor_spline(
                radial,
                vertical,
                jnp.where(sample_valid, values, 0.0),
            )
            if surface is None
            else surface
        )
        complete_map = jnp.all(sample_valid) & jnp.all(jnp.isfinite(values))
    else:
        inactive_axis = jnp.arange(4, dtype=values.dtype)
        authority_spline = fit_tensor_spline(
            inactive_axis,
            inactive_axis,
            jnp.zeros((4, 4), dtype=values.dtype),
        )
        complete_map = jnp.asarray(False)
    selected = jnp.stack((selected_extremum, selected_saddle))
    valid = (
        evidence_spline.fit_executed
        & complete_map
        & jnp.all(jnp.isfinite(selected[:, :3]), axis=-1)
    )
    radial_grid, vertical_grid = _coordinate_grids(radial, vertical)
    radial_min = jnp.min(jnp.where(sample_valid, radial_grid, jnp.inf))
    radial_max = jnp.max(jnp.where(sample_valid, radial_grid, -jnp.inf))
    vertical_min = jnp.min(jnp.where(sample_valid, vertical_grid, jnp.inf))
    vertical_max = jnp.max(jnp.where(sample_valid, vertical_grid, -jnp.inf))
    coordinate_span = jnp.maximum(
        radial_max - radial_min,
        vertical_max - vertical_min,
    )
    coordinate_span = jnp.where(evidence_spline.fit_executed, coordinate_span, 1.0)
    field_min = jnp.min(jnp.where(sample_valid, values, jnp.inf))
    field_max = jnp.max(jnp.where(sample_valid, values, -jnp.inf))
    field_scale = jnp.maximum(field_max - field_min, jnp.finfo(values.dtype).tiny)
    field_scale = jnp.where(evidence_spline.fit_executed, field_scale, 1.0)
    seed_evaluation = authority_spline.evaluate(selected[:, 0], selected[:, 1])
    (
        seed_active_count,
        _seed_value_basis_norm,
        seed_derivative_basis_norm,
    ) = evidence_spline.scaled_basis_receipt(selected[:, 0], selected[:, 1])
    normalized_cell_pitch = (
        _minimum_cell_pitch(radial, vertical, sample_valid) / coordinate_span
    )
    representation_floor = (
        evidence_spline.sample_rms_residual / field_scale / normalized_cell_pitch
    )
    seed_roundoff_floor = (
        seed_active_count.astype(values.dtype)
        * evidence_spline.condition_number
        * jnp.finfo(values.dtype).eps
        * seed_derivative_basis_norm
        * coordinate_span
    )
    seed_stationarity_tolerance = jnp.maximum(seed_roundoff_floor, representation_floor)
    seed_gradient = jnp.stack(
        (seed_evaluation.radial_derivative, seed_evaluation.vertical_derivative),
        axis=-1,
    )
    seed_gradient_norm = jnp.linalg.norm(seed_gradient, axis=-1)
    seed_normalized_gradient = seed_gradient_norm * coordinate_span / field_scale
    seed_stationary = (
        valid
        & jnp.isfinite(seed_normalized_gradient)
        & (seed_normalized_gradient <= seed_stationarity_tolerance)
    )
    seed_hessian = jnp.stack(
        (
            jnp.stack(
                (
                    seed_evaluation.radial_second_derivative,
                    seed_evaluation.mixed_derivative,
                ),
                axis=-1,
            ),
            jnp.stack(
                (
                    seed_evaluation.mixed_derivative,
                    seed_evaluation.vertical_second_derivative,
                ),
                axis=-1,
            ),
        ),
        axis=-2,
    )
    seed_determinant = (
        seed_hessian[..., 0, 0] * seed_hessian[..., 1, 1]
        - seed_hessian[..., 0, 1] * seed_hessian[..., 1, 0]
    )
    active_polish = valid & ~seed_stationary

    def refine_selected(_):
        return polish_stationary_points(
            authority_spline,
            selected[:, :2],
            active_polish,
            stationary_steps=8,
        )

    def retain_stationary(_):
        seed_hessian_type = jnp.sign(seed_determinant).astype(jnp.int8)
        return {
            "position_rz": jnp.where(valid[:, None], selected[:, :2], 0.0),
            "value": jnp.where(valid, seed_evaluation.value, 0.0),
            "gradient_norm": jnp.where(valid, seed_gradient_norm, 0.0),
            "gradient": jnp.where(valid[:, None], seed_gradient, 0.0),
            "hessian": jnp.where(valid[:, None, None], seed_hessian, 0.0),
            "hessian_type": jnp.where(valid, seed_hessian_type, 0),
            # The refine lane masks every slot it does not polish; this lane
            # polishes none, so its raw convergence slot must carry the same
            # masked value. The acceptance qualification is recomputed below
            # from the merged attempt, which is what the receipt publishes.
            "converged": jnp.zeros(valid.shape, dtype=bool),
            "in_domain": valid,
            "iteration_count": jnp.zeros(valid.shape, dtype=jnp.int32),
        }

    attempted = jax.lax.cond(
        jnp.any(active_polish), refine_selected, retain_stationary, operand=None
    )
    attempted = attempted | {
        "position_rz": jnp.where(
            seed_stationary[:, None], selected[:, :2], attempted["position_rz"]
        ),
        "value": jnp.where(seed_stationary, seed_evaluation.value, attempted["value"]),
        "gradient": jnp.where(
            seed_stationary[:, None], seed_gradient, attempted["gradient"]
        ),
        "gradient_norm": jnp.where(
            seed_stationary, seed_gradient_norm, attempted["gradient_norm"]
        ),
        "hessian": jnp.where(
            seed_stationary[:, None, None], seed_hessian, attempted["hessian"]
        ),
        "hessian_type": jnp.where(
            seed_stationary,
            jnp.sign(seed_determinant).astype(jnp.int8),
            attempted["hessian_type"],
        ),
        "in_domain": jnp.where(seed_stationary, valid, attempted["in_domain"]),
    }
    normalized_gradient = attempted["gradient_norm"] * coordinate_span / field_scale
    active_count, value_basis_norm, derivative_basis_norm = (
        evidence_spline.scaled_basis_receipt(
            attempted["position_rz"][:, 0], attempted["position_rz"][:, 1]
        )
    )
    roundoff_floor = (
        active_count.astype(values.dtype)
        * evidence_spline.condition_number
        * jnp.finfo(values.dtype).eps
        * derivative_basis_norm
        * coordinate_span
    )
    stationarity_tolerance = jnp.maximum(roundoff_floor, representation_floor)
    normalized_value_change = jnp.abs(attempted["value"] - selected[:, 2]) / field_scale
    normalized_sample_rms = evidence_spline.sample_rms_residual / field_scale
    representation_adequate = normalized_sample_rms <= normalized_value_change
    value_consistency_tolerance = jnp.maximum(
        evidence_spline.sample_rms_residual / field_scale,
        active_count.astype(values.dtype)
        * evidence_spline.condition_number
        * jnp.finfo(values.dtype).eps
        * value_basis_norm,
    )
    expected_hessian_type = jnp.asarray((1, -1), dtype=jnp.int8)
    converged = (
        evidence_spline.solve_converged
        & valid
        & attempted["in_domain"]
        & jnp.isfinite(attempted["value"])
        & jnp.all(jnp.isfinite(attempted["gradient"]), axis=-1)
        & jnp.all(jnp.isfinite(attempted["hessian"]), axis=(-2, -1))
        & (attempted["hessian_type"] == expected_hessian_type)
        & (normalized_gradient <= stationarity_tolerance)
    )
    local_value_evidence = _census_local_values(
        values,
        radial,
        vertical,
        sample_valid,
        selected[:, :2],
        attempted["position_rz"],
    )
    polished = selected.at[:, :2].set(attempted["position_rz"])
    polished = polished.at[:, 2].set(attempted["value"])
    retained = jnp.where(converged[:, None], polished, selected)
    evidence_evaluation = evidence_spline.evaluate(
        attempted["position_rz"][:, 0], attempted["position_rz"][:, 1]
    )
    receipt = attempted | {
        "converged": converged,
        "fit_attempted": jnp.broadcast_to(evidence_spline.fit_executed, valid.shape),
        "fit_converged": jnp.broadcast_to(evidence_spline.solve_converged, valid.shape),
        "fit_iterations": jnp.broadcast_to(
            evidence_spline.solve_iterations, valid.shape
        ),
        "fit_residual": jnp.broadcast_to(evidence_spline.solve_residual, valid.shape),
        "interface_value": jnp.broadcast_to(interface_value, valid.shape),
        "stationarity_tolerance": jnp.broadcast_to(stationarity_tolerance, valid.shape),
        "roundoff_floor": roundoff_floor,
        "representation_floor": jnp.broadcast_to(representation_floor, valid.shape),
        "active_derivative_basis_count": active_count,
        "derivative_basis_norm": derivative_basis_norm,
        "value_basis_norm": value_basis_norm,
        "sample_rms_residual": jnp.broadcast_to(
            evidence_spline.sample_rms_residual, valid.shape
        ),
        "seed_normalized_gradient": seed_normalized_gradient,
        "normalized_gradient": normalized_gradient,
        "normalized_value_change": normalized_value_change,
        "value_consistency_tolerance": value_consistency_tolerance,
        "local_value_consistent": (
            normalized_value_change <= value_consistency_tolerance
        ),
        "seed_stationary": seed_stationary,
        "fit_value": evidence_evaluation.value,
        "local_value_evidence": local_value_evidence,
        "spline_value": attempted["value"],
        "spline_authored": converged,
        "complete_map": jnp.broadcast_to(complete_map, valid.shape),
        "representation_adequate": representation_adequate,
        "value_replaced": jnp.zeros(valid.shape, dtype=bool),
        "census_position_rz": selected[:, :2],
        "selected_position_rz": retained[:, :2],
        "value": retained[:, 2],
        "selected_value": retained[:, 2],
    }
    return retained[0], retained[1], receipt


@partial(jax.jit, static_argnums=(4, 5))
def traced_spline_contour(
    values: jnp.ndarray,
    radial: jnp.ndarray,
    vertical: jnp.ndarray,
    level: jnp.ndarray,
    bisection_steps: int = 40,
    saddle_steps: int = 8,
    surface: TensorBSpline | None = None,
    axis_rz: jnp.ndarray | None = None,
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

    A tie is the boundary level itself passing through the saddle, where the
    signed delta carries no information the pairing can follow.  When
    ``axis_rz`` is supplied, the pairing of such a cell is instead taken from
    the separatrix asymptotes at the saddle, the Hessian eigenvector
    directions.  Opposite sectors of the indefinite quadratic form share a
    sign, so the two crossings joined by one hyperbola branch agree in it.  The
    axis lies in the lobe sector, which is what separates that sector from the
    leg sectors by a scalar test on the crossings.

    This function returns contour geometry only.  It neither imports nor calls
    the polygonal clipped-support integration path.
    """
    values = jnp.asarray(values)
    radial = jnp.asarray(radial, dtype=values.dtype)
    vertical = jnp.asarray(vertical, dtype=values.dtype)
    level = jnp.asarray(level, dtype=values.dtype)
    spline = fit_tensor_spline(radial, vertical, values) if surface is None else surface

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

    edge_start, edge_end = _structured_cell_edges(radial, vertical)
    edge_vector = edge_end - edge_start

    # Split each cell edge at the stationary points of the cubic the spline
    # restricts to it, and bracket one root on every monotone piece whose ends
    # straddle the level.  A cubic restricted to one edge can leave both
    # endpoints on the same side of the level and still cross twice between
    # them, and at the boundary level the field's saddle produces exactly that
    # pair, so a test on the endpoints of a fixed subdivision loses both
    # crossings whenever the pair happens to fall inside one sub-interval, and
    # the lobe the saddle pinches stops closing.
    #
    # The edge derivative is the chain rule of the spline's own partials, so
    # delta'(t) = dr/dt . dp/dr + dz/dt . dp/dz is quadratic in the edge
    # parameter; three evaluations fix its coefficients, and its roots are the
    # at most two interior parameters where a single piece can hold two level
    # crossings.  A negative discriminant is clamped rather than masked, which
    # leaves a spurious split point on a monotone edge: splitting a monotone
    # interval anywhere keeps every piece monotone, so the extra piece costs
    # one evaluation and cannot hide a crossing.
    derivative_nodes = jnp.asarray((0.0, 0.5, 1.0), dtype=values.dtype)
    derivative_point = (
        edge_start[..., None, :]
        + derivative_nodes[..., None] * edge_vector[..., None, :]
    )
    derivative_evaluation = spline.evaluate(
        derivative_point[..., 0], derivative_point[..., 1]
    )
    edge_slope = (
        derivative_evaluation.radial_derivative * edge_vector[..., None, 0]
        + derivative_evaluation.vertical_derivative * edge_vector[..., None, 1]
    )
    slope_start, slope_middle, slope_end = (
        edge_slope[..., 0],
        edge_slope[..., 1],
        edge_slope[..., 2],
    )
    quadratic_term = 2.0 * (slope_end + slope_start - 2.0 * slope_middle)
    linear_term = (slope_end - slope_start) - quadratic_term
    constant_term = slope_start
    discriminant = linear_term**2 - 4.0 * quadratic_term * constant_term
    root_offset = jnp.sqrt(jnp.maximum(discriminant, 0.0))
    # Stable quadratic roots: a vanishing leading coefficient sends one to
    # infinity, where the interior test drops it, and leaves the other on the
    # linear root.
    far_parameter = jnp.asarray(2.0, dtype=values.dtype)
    plus_interior = -2.0 * constant_term / (linear_term + root_offset)
    minus_interior = -2.0 * constant_term / (linear_term - root_offset)
    plus_valid = (plus_interior > 0.0) & (plus_interior < 1.0)
    minus_valid = (minus_interior > 0.0) & (minus_interior < 1.0)
    plus_parameter = jnp.where(plus_valid, plus_interior, far_parameter)
    minus_parameter = jnp.where(minus_valid, minus_interior, far_parameter)
    stationary_low = jnp.minimum(plus_parameter, minus_parameter)
    stationary_high = jnp.maximum(plus_parameter, minus_parameter)
    stationary_count = plus_valid.astype(jnp.int32) + minus_valid.astype(jnp.int32)
    # Three pieces per edge is the fixed shape: no stationary point collapses
    # the middle piece onto an endpoint, one opens the two pieces it bounds,
    # and the padded pieces carry an exact-zero parameter through the mask.
    break_low = jnp.where(stationary_count >= 1, stationary_low, 0.0)
    break_high = jnp.where(stationary_count == 2, stationary_high, 1.0)
    piece_low = jnp.stack((jnp.zeros_like(break_low), break_low, break_high), axis=-1)
    piece_high = jnp.stack((break_low, break_high, jnp.ones_like(break_low)), axis=-1)
    piece_low_point = (
        edge_start[..., None, :] + piece_low[..., None] * edge_vector[..., None, :]
    )
    piece_high_point = (
        edge_start[..., None, :] + piece_high[..., None] * edge_vector[..., None, :]
    )
    sub_low_delta = spline(piece_low_point[..., 0], piece_low_point[..., 1]) - level
    sub_high_delta = spline(piece_high_point[..., 0], piece_high_point[..., 1]) - level
    sub_crossing = (sub_low_delta >= 0.0) != (sub_high_delta >= 0.0)
    edge_count = jnp.sum(sub_crossing.astype(jnp.int32), axis=-1)

    def bisect_sub(low_parameter, high_parameter, low_value):
        """Locate one root on each sub-interval whose ends straddle the level."""
        lower = jnp.broadcast_to(low_parameter, low_value.shape)
        upper = jnp.broadcast_to(high_parameter, low_value.shape)

        def bisect(_iteration, state):
            low, high, value_at_low = state
            middle = 0.5 * (low + high)
            point = (
                edge_start[..., None, :] + middle[..., None] * edge_vector[..., None, :]
            )
            middle_value = spline(point[..., 0], point[..., 1]) - level
            same_side = (value_at_low >= 0.0) == (middle_value >= 0.0)
            return (
                jnp.where(same_side, middle, low),
                jnp.where(same_side, high, middle),
                jnp.where(same_side, middle_value, value_at_low),
            )

        low, high, _ = jax.lax.fori_loop(
            0, bisection_steps, bisect, (lower, upper, low_value)
        )
        return 0.5 * (low + high)

    sub_parameter = bisect_sub(piece_low, piece_high, sub_low_delta)
    sub_edge = jnp.broadcast_to(
        jnp.arange(4, dtype=jnp.int32).reshape((1, 1, 4, 1)), sub_crossing.shape
    )

    # Pack the crossings into the fixed four-slot ring in edge order and, on an
    # edge carrying two, ascending parameter order.  A cell whose boundary the
    # level crosses more than four times exceeds the ring and is reported
    # through the well-formed mask rather than silently truncated.
    candidate_index = (
        (jnp.cumsum(edge_count, axis=-1) - edge_count)[..., :, None]
        + jnp.cumsum(sub_crossing.astype(jnp.int32), axis=-1)
        - sub_crossing.astype(jnp.int32)
    )
    ring_shape = candidate_index.shape[:-2]
    slot_count = candidate_index.shape[-1] * candidate_index.shape[-2]
    index_flat = candidate_index.reshape(ring_shape + (slot_count,))
    valid_flat = sub_crossing.reshape(ring_shape + (slot_count,))
    parameter_flat = sub_parameter.reshape(ring_shape + (slot_count,))
    edge_flat = sub_edge.reshape(ring_shape + (slot_count,))
    slot_parameters = []
    slot_edges = []
    slot_valid = []
    for slot in range(4):
        occupied = (index_flat == slot) & valid_flat
        slot_parameters.append(jnp.where(occupied, parameter_flat, 0.0).sum(axis=-1))
        slot_edges.append(jnp.where(occupied, edge_flat, 0).sum(axis=-1))
        slot_valid.append(jnp.any(occupied, axis=-1))
    edge_parameter = jnp.stack(slot_parameters, axis=-1)
    edge_source_index = jnp.stack(slot_edges, axis=-1)
    edge_crossing = jnp.stack(slot_valid, axis=-1)
    crossing_count = jnp.sum(edge_count, axis=-1, dtype=jnp.int32)
    crossing_overflow = crossing_count > 4

    slot_edge = jnp.clip(edge_source_index, 0, 3)
    slot_start = jnp.take_along_axis(edge_start, slot_edge[..., None], axis=-2)
    slot_vector = jnp.take_along_axis(edge_vector, slot_edge[..., None], axis=-2)
    edge_point = slot_start + edge_parameter[..., None] * slot_vector
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

    saddle_polish = _polish_stationary_points_in_bounds(
        spline,
        jnp.stack((centre_radial, centre_vertical), axis=-1),
        jnp.ones_like(crossing_count, dtype=bool),
        jnp.stack((radial_low, vertical_low), axis=-1),
        jnp.stack((radial_high, vertical_high), axis=-1),
        saddle_steps,
    )
    saddle_point = saddle_polish["position_rz"]
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
    saddle_gradient = saddle_polish["gradient_norm"]
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
    # The tie value is the orientation-only fallback: it is what remains when
    # no axis is supplied to select the lobe sector from.
    same_side_as_first_corner = jnp.where(
        decision_tie,
        True,
        (decision_delta >= 0.0) == (corner_delta[..., 0] >= 0.0),
    )

    packed = _ordered_crossing_indices(edge_crossing)
    edge_node = _structured_edge_nodes(radial.size, vertical.size)
    # A ring slot is not an edge once an edge carries two crossings, so the
    # node and the reported edge of each slot are gathered through the physical
    # edge the slot's crossing lies on.  Sharing a node with the neighbouring
    # cell requires that physical edge, not the slot ordinal.
    slot_edge = jnp.clip(edge_source_index, 0, 3)
    edge_node = jnp.take_along_axis(edge_node, slot_edge, axis=-1)
    # An edge can carry two crossings, and they are distinct points that must
    # stay distinct nodes.  Each slot therefore numbers itself among the
    # crossings of its own edge in a direction-independent order -- ascending
    # vertical for a radial edge, ascending radius for a vertical one -- so the
    # two cells sharing the edge agree on the number and a crossing keeps one
    # node identity across the cells it separates.
    edge_spans_radius = (slot_edge % 2) == 0
    crossing_key = jnp.where(edge_spans_radius, edge_point[..., 1], edge_point[..., 0])
    slot_ordinal = jnp.arange(4, dtype=jnp.int32)
    same_edge = slot_edge[..., :, None] == slot_edge[..., None, :]
    lower_key = crossing_key[..., :, None] < crossing_key[..., None, :]
    tied_key = (crossing_key[..., :, None] == crossing_key[..., None, :]) & (
        slot_ordinal[..., :, None] > slot_ordinal[..., None, :]
    )
    edge_rank = jnp.sum(
        (same_edge & edge_crossing[..., None, :] & (lower_key | tied_key)).astype(
            jnp.int32
        ),
        axis=-1,
    )
    slot_node = EDGE_CROSSING_CAPACITY * edge_node + edge_rank
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
    # A tie is the level passing through the saddle, where the signed delta
    # that decides every other diagonal cell carries no information.  The four
    # crossings are then grouped by the sign of the quadratic form at the
    # saddle, which is constant on each separatrix sector and opposite between
    # opposite sectors.  The lobe sector holds the axis, so it is the sector
    # whose sign agrees with the axis's, and pairing its two crossings closes
    # the lobe instead of running it out along a leg.  Around the cell the sign
    # runs in two blocks of two, so exactly one of the two pairings groups
    # equal signs.
    asymptote_pairing_applied = jnp.zeros_like(ambiguous)
    if axis_rz is not None:
        hessian_radial = saddle_evaluation.radial_second_derivative
        hessian_vertical = saddle_evaluation.vertical_second_derivative
        hessian_mixed = saddle_evaluation.mixed_derivative

        def _quadratic_form(offset):
            r = hessian_radial[..., None]
            m = hessian_mixed[..., None]
            z = hessian_vertical[..., None]
            u = offset[..., 0]
            v = offset[..., 1]
            return r * u * u + 2.0 * m * u * v + z * v * v

        axis = jnp.asarray(axis_rz, dtype=values.dtype)
        crossing_lobe_side = _quadratic_form(edge_point - saddle_point[..., None, :])
        axis_lobe_side = _quadratic_form(axis - saddle_point[..., None, :])
        # A crossing lies on the level set, where the form's leading term
        # cancels, so its magnitude carries no sector information and only its
        # sign does.  Comparing the signs is what labels the sector each
        # crossing lies in.
        crossing_on_axis_side = (crossing_lobe_side >= 0.0) == (axis_lobe_side >= 0.0)
        # The two crossings bounding the lobe sector are the pair that closes
        # it, so they are joined to each other and the two leg-side crossings
        # are left to join each other.  The pairs are read from the sector
        # labels rather than chosen between two fixed ring arrangements, so a
        # cell whose sector pair is not adjacent in slot order is grouped
        # correctly as well.
        axis_rank = jnp.cumsum(
            crossing_on_axis_side.astype(jnp.int32), axis=-1
        ) - crossing_on_axis_side.astype(jnp.int32)
        leg_rank = jnp.cumsum((~crossing_on_axis_side).astype(jnp.int32), axis=-1) - (
            ~crossing_on_axis_side
        ).astype(jnp.int32)
        slot_index = jnp.arange(4, dtype=jnp.int32)

        def _sector_partner(on_axis_side, rank):
            """Return the slot of the numbered crossing on one side of the saddle."""
            within_sector_rank = jnp.where(on_axis_side, axis_rank, leg_rank)
            match = (crossing_on_axis_side == on_axis_side) & (
                within_sector_rank == rank
            )
            return jnp.sum(jnp.where(match, slot_index, 0), axis=-1)

        tie_pairs = jnp.stack(
            (
                jnp.stack(
                    (_sector_partner(True, 0), _sector_partner(True, 1)), axis=-1
                ),
                jnp.stack(
                    (_sector_partner(False, 0), _sector_partner(False, 1)), axis=-1
                ),
            ),
            axis=-2,
        )
        # Only a cell whose four crossings split two and two across the saddle
        # has two sectors to group; anything else is left to the corner-sign
        # pairing rather than joined to a defaulted slot.
        sector_split = jnp.sum(crossing_on_axis_side.astype(jnp.int32), axis=-1) == 2
        asymptote_pairing_applied = (
            ambiguous & decision_tie & saddle_stationary & sector_split
        )
        saddle_pairs = jnp.where(
            asymptote_pairing_applied[..., None, None], tie_pairs, saddle_pairs
        )
    pair_indices = jnp.where(ambiguous[..., None, None], saddle_pairs, regular_pairs)
    segment_valid = jnp.stack((crossing_count == 2, jnp.zeros_like(ambiguous)), axis=-1)
    segment_valid = jnp.where(
        ambiguous[..., None], jnp.ones_like(segment_valid), segment_valid
    )

    segment_point = _paired_edge_values(edge_point, pair_indices)
    segment_node = jnp.take_along_axis(slot_node[..., None, :], pair_indices, axis=-1)
    segment_source = jnp.take_along_axis(
        jnp.clip(edge_source_index, 0, 3)[..., None, :], pair_indices, axis=-1
    )
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
    canonical_segment_source = jnp.where(segment_valid[..., None], segment_source, 0)
    canonical_segment_point = jnp.where(
        segment_valid[..., None, None], segment_point, 0.0
    )
    canonical_segment_node = jnp.where(segment_valid[..., None], segment_node, 0)
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
        "segment_edge_indices": canonical_segment_source,
        "segment_endpoints_rz": canonical_segment_point,
        "segment_node_indices": canonical_segment_node,
        "segment_endpoint_tangents_rz": canonical_segment_tangent,
        "segment_controls_rz": canonical_controls,
        "cell_crossing_count": crossing_count,
        "ambiguous_saddle": ambiguous,
        "ambiguous_resolved": ambiguous & ~decision_tie,
        "ambiguous_tie_broken": ambiguous & decision_tie,
        "asymptote_pairing_applied": asymptote_pairing_applied,
        "saddle_stationary": ambiguous & saddle_stationary,
        "saddle_rz": canonical_saddle_point,
        "saddle_value": canonical_saddle_value,
        "segment_saddle_rz": jnp.where(
            (segment_valid & (ambiguous & decision_tie & saddle_stationary)[..., None])[
                ..., None
            ],
            saddle_point[..., None, :],
            0.0,
        ),
        "segment_at_saddle": segment_valid
        & (ambiguous & decision_tie & saddle_stationary)[..., None],
        "edge_node_capacity": jnp.asarray(
            EDGE_CROSSING_CAPACITY
            * (vertical.size * (radial.size - 1) + (vertical.size - 1) * radial.size),
            dtype=jnp.int32,
        ),
        "crossing_overflow": crossing_overflow,
        "well_formed": jnp.all(
            ((crossing_count == 0) | (crossing_count == 2) | (crossing_count == 4))
            & ~crossing_overflow
        ),
    }
