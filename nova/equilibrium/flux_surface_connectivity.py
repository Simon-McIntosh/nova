"""Contour-free, accelerator-native flux-surface averaging (JAX).

Fixed-shape, ``jit`` / ``vmap`` / ``grad``-safe flux-surface metrics from a
solved poloidal-flux field ψ(R, Z) — the device-native replacement for the
host-side coarea binning (``argsort`` + cumulative-sum + ``scipy.ndimage.label``)
that cannot batch on an accelerator.  Everything here is a fixed-shape reduction
over the full grid: no data-dependent shapes, no sort, and no contour / marching-
squares / level-set extraction at all.

Two device primitives:

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


_SQRT2 = 2.0**0.5

__all__ = [
    "flood_fill_core",
    "flood_fill_core_with_steps",
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
