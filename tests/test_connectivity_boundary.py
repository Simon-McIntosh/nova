"""Tests for the accelerator-native connectivity boundary read (JAX).

The device flood-fill boundary read resolves the last-closed-flux-surface with
fixed-shape, ``jit`` / ``vmap`` / ``grad``-safe primitives. Pinned here three
ways that stand alone on the nova spine (no data):

* **accelerator compliance** - compiles + runs under ``jit`` / ``vmap`` (a batch
  of psi fields on one grid) / ``grad``, output fixed-shape regardless of core
  size;
* **contour-free** - imports no contourpy / marching-squares / ``scipy.ndimage``
  and calls no ``argwhere`` / ``label`` (AST-checked);
* **emergent X-set** - a double-null field fills both null slots with distinct
  saddles, never two copies of one stencil-degenerate saddle;
* **termination** - the push binds at the exact interpolated wall tangency of a
  limited field and at the Newton-refined saddle flux of a diverted field
  (synthetic truth in place of a host contour reader);
* **marginal continuity** - the binding flux moves smoothly through the
  limited->diverted transition where a classify-first read steps;
* **Lipschitz smooth weight** - a sub-temperature flux ripple moves the smooth
  core weight by a small, bounded amount (no discrete mask flips);
* **no-flip X-set** - across a binding switch between two nulls the emitted
  X-slots hold real null positions, never a mid-band blend.
"""

from __future__ import annotations

import ast
import inspect

import numpy as np
import pytest

from nova.utilities.importmanager import skip_import

with skip_import("jax"):
    import jax
    import jax.numpy as jnp

    from nova.jax import connectivity_boundary as cb
    from nova.jax.stencil_nulls import xpoint_candidates

from nova.jax.equilibrium_labels import LCFS_ANGLES
from nova.jax.wall_mask import inside_polygon as _inside_polygon


def _limited_field(nr=81, nz=101):
    """A single O-point Gaussian - a wall-limited plasma (no separatrix X)."""
    rg = np.linspace(0.2, 1.8, nr)
    zg = np.linspace(-1.0, 1.0, nz)
    rr, zz = np.meshgrid(rg, zg)
    psi = np.exp(-(((rr - 1.0) ** 2 + zz**2) / 0.3**2))
    lr = np.array([0.55, 1.45, 1.45, 0.55, 0.55])
    lz = np.array([-0.55, -0.55, 0.55, 0.55, -0.55])
    inside = _inside_polygon(rr.ravel(), zz.ravel(), lr, lz).reshape(nz, nr)
    return psi, rg, zg, (1.0, 0.0), lr, lz, inside


def _diverted_field(nr=101, nz=141):
    """Two same-sign Gaussians with a saddle between them - a diverted separatrix."""
    rg = np.linspace(0.2, 1.8, nr)
    zg = np.linspace(-1.2, 1.2, nz)
    rr, zz = np.meshgrid(rg, zg)
    s = 0.28

    def blob(r0, z0):
        return np.exp(-(((rr - r0) ** 2 + (zz - z0) ** 2) / s**2))

    psi = blob(1.0, 0.25) + 0.9 * blob(1.0, -0.75)
    lr = np.array([0.25, 1.75, 1.75, 0.25, 0.25])
    lz = np.array([-1.1, -1.1, 1.1, 1.1, -1.1])
    inside = _inside_polygon(rr.ravel(), zz.ravel(), lr, lz).reshape(nz, nr)
    return psi, rg, zg, (1.0, 0.25), lr, lz, inside


class _Grid:
    """Minimal grid duck-type for the host adapters."""

    def __init__(self, rg, zg, inside, lr=None, lz=None):
        self.rg, self.zg, self.inside_limiter = rg, zg, inside
        self.limiter_r, self.limiter_z = lr, lz


# --- accelerator compliance -------------------------------------------------


def test_jit_vmap_grad_safe_and_fixed_shape():
    ang = jnp.asarray(np.asarray(LCFS_ANGLES))
    small = _limited_field(nr=61, nz=61)
    big = _diverted_field(nr=61, nz=61)

    def read(psi, rg, zg, inside, ar, az):
        return cb.boundary_read_jax(
            jnp.asarray(psi),
            jnp.asarray(rg),
            jnp.asarray(zg),
            jnp.asarray(inside),
            jnp.asarray(float(ar)),
            jnp.asarray(float(az)),
            64,
            14,
            256,
            ang,
            jnp.asarray(0.999),
        )

    o_s = read(small[0], small[1], small[2], small[6], *small[3])
    o_b = read(big[0], big[1], big[2], big[6], *big[3])
    assert o_s["radii"].dtype == jnp.float64
    assert int(o_s["n_core_cells"]) != int(o_b["n_core_cells"])
    assert np.asarray(o_s["radii"]).shape == (len(LCFS_ANGLES),)
    assert bool(o_s["found"]) and bool(o_b["found"])

    psi, rg, zg, axis, _lr, _lz, inside = _limited_field(nr=61, nz=61)
    batch = jnp.stack(
        [jnp.asarray(psi), jnp.asarray(psi * 1.03), jnp.asarray(psi * 0.97)]
    )
    vfun = jax.vmap(
        lambda p: cb.boundary_read_jax(
            p,
            jnp.asarray(rg),
            jnp.asarray(zg),
            jnp.asarray(inside),
            jnp.asarray(1.0),
            jnp.asarray(0.0),
            64,
            14,
            256,
            ang,
            jnp.asarray(0.999),
        )["psi_bnd"]
    )
    vb = vfun(batch)
    assert vb.shape == (3,)
    assert np.all(np.isfinite(np.asarray(vb)))

    def loss(az):
        return cb.boundary_read_jax(
            jnp.asarray(psi),
            jnp.asarray(rg),
            jnp.asarray(zg),
            jnp.asarray(inside),
            jnp.asarray(1.0),
            az,
            64,
            14,
            256,
            ang,
            jnp.asarray(0.999),
        )["psi_bnd"]

    g = jax.grad(loss)(jnp.asarray(0.0))
    assert np.isfinite(float(g))


def test_module_is_contour_free():
    """Imports no contour / ndimage machinery and calls no argwhere / label."""
    tree = ast.parse(inspect.getsource(cb))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            imported += [base] + [f"{base}.{a.name}" for a in node.names]
    banned = ("contourpy", "matplotlib", "skimage", "scipy.ndimage", "ndimage")
    for imp in imported:
        assert not any(b in imp for b in banned), f"boundary read imports {imp!r}"
    calls = {
        n.func.attr
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    }
    assert "argwhere" not in calls and "label" not in calls


def test_batched_matches_per_slice():
    """vmap over a batch of psi fields equals the per-slice reads, element-wise."""
    ang = jnp.asarray(np.asarray(LCFS_ANGLES))
    psi, rg, zg, _axis, _lr, _lz, inside = _limited_field(nr=61, nz=61)
    slices = [jnp.asarray(psi), jnp.asarray(psi * 1.03), jnp.asarray(psi * 0.97)]

    def read(p):
        return cb.boundary_read_jax(
            p,
            jnp.asarray(rg),
            jnp.asarray(zg),
            jnp.asarray(inside),
            jnp.asarray(1.0),
            jnp.asarray(0.0),
            64,
            14,
            256,
            ang,
            jnp.asarray(0.999),
        )

    per_slice_bnd = np.array([float(read(p)["psi_bnd"]) for p in slices])
    per_slice_radii = np.stack([np.asarray(read(p)["radii"]) for p in slices])

    batch = jnp.stack(slices)
    batched = jax.vmap(read)(batch)
    assert np.allclose(np.asarray(batched["psi_bnd"]), per_slice_bnd, atol=1e-10)
    assert np.allclose(np.asarray(batched["radii"]), per_slice_radii, atol=1e-10)


# --- emergent X-set: distinct nulls, not stencil duplicates ------------------


def _double_null_field(nr=45, nz=61):
    """A near-balanced double-null whose saddles each fire TWO stencil vertices."""
    rg = np.linspace(0.2, 1.8, nr)
    zg = np.linspace(-1.4, 1.4, nz)
    r0 = 1.0 + 0.5 * float(rg[1] - rg[0])
    rr, zz = np.meshgrid(rg, zg)

    def blob(z0, a):
        return a * np.exp(-((rr - r0) ** 2 / 0.45**2 + (zz - z0) ** 2 / 0.28**2))

    psi = blob(0.0, 1.0) + blob(-0.9, 0.9) + blob(0.9, 0.88)
    lr = np.array([0.25, 1.75, 1.75, 0.25, 0.25])
    lz = np.array([-1.3, -1.3, 1.3, 1.3, -1.3])
    inside = _inside_polygon(rr.ravel(), zz.ravel(), lr, lz).reshape(nz, nr)
    return psi, rg, zg, (r0, 0.0), lr, lz, inside


def test_emergent_xset_holds_both_nulls_of_a_double_null():
    psi, rg, zg, axis, lr, lz, inside = _double_null_field()
    gpu = cb.boundary_read(psi, _Grid(rg, zg, inside, lr, lz), axis, lcfs_norm=1.0)
    assert gpu.found and gpu.is_diverted
    xset = np.asarray(gpu.xset, dtype=np.float64)
    finite = xset[np.isfinite(xset).all(axis=1)]
    assert finite.shape[0] == 2
    assert np.sign(finite[0, 1]) != np.sign(finite[1, 1])
    assert np.all(np.abs(np.abs(finite[:, 1]) - 0.46) < 0.15)


# --- termination: exact wall tangency / synthetic saddle truth ---------------


def _psi_limited(r, z):
    """The limited field as an analytic function (for exact wall flux)."""
    return np.exp(-(((r - 1.0) ** 2 + z**2) / 0.3**2))


def _psi_diverted(r, z):
    """The diverted field as an analytic function (for the saddle truth)."""
    s = 0.28
    return np.exp(-(((r - 1.0) ** 2 + (z - 0.25) ** 2) / s**2)) + 0.9 * np.exp(
        -(((r - 1.0) ** 2 + (z + 0.75) ** 2) / s**2)
    )


def _dense_wall(lr, lz, m=720):
    """Uniform arc-length resample of a closed limiter loop."""
    rr = np.append(lr, lr[0])
    zz = np.append(lz, lz[0])
    seg = np.hypot(np.diff(rr), np.diff(zz))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    q = np.linspace(0.0, s[-1], m, endpoint=False)
    return np.interp(q, s, rr), np.interp(q, s, zz)


def _newton_saddle(fn, r0, z0, h=1e-6, iters=60):
    """Newton-refined stationary point of an analytic field (the truth)."""
    x = np.array([r0, z0], float)
    for _ in range(iters):
        r, z = x
        grad = np.array(
            [
                (fn(r + h, z) - fn(r - h, z)) / (2 * h),
                (fn(r, z + h) - fn(r, z - h)) / (2 * h),
            ]
        )
        hess = np.array(
            [
                [
                    (fn(r + h, z) - 2 * fn(r, z) + fn(r - h, z)) / h**2,
                    (
                        fn(r + h, z + h)
                        - fn(r + h, z - h)
                        - fn(r - h, z + h)
                        + fn(r - h, z - h)
                    )
                    / (4 * h**2),
                ],
                [
                    (
                        fn(r + h, z + h)
                        - fn(r + h, z - h)
                        - fn(r - h, z + h)
                        + fn(r - h, z - h)
                    )
                    / (4 * h**2),
                    (fn(r, z + h) - 2 * fn(r, z) + fn(r, z - h)) / h**2,
                ],
            ]
        )
        x = x - np.linalg.solve(hess, grad)
    return x


def test_wall_termination_binds_at_exact_tangency():
    """A limited push terminates at the interpolated wall tangency.

    With the exact node flux supplied (the campaign ``g_wall`` GEMM path) the
    binding flux reproduces the confined-most wall value to round-off.
    """
    psi, rg, zg, axis, lr, lz, inside = _limited_field()
    wall_r, wall_z = _dense_wall(lr, lz)
    wall_psi = _psi_limited(wall_r, wall_z)
    grid = _Grid(rg, zg, inside, lr, lz)
    grid.wall_r, grid.wall_z = wall_r, wall_z
    out = cb.boundary_read(psi, grid, axis, wall_psi=wall_psi)
    truth = wall_psi.max()
    span = abs(out.psi_axis - truth)
    assert out.found and not out.is_diverted
    assert abs(out.psi_bnd - truth) / span < 1e-12
    # the reported ring flux sits the lcfs_norm fraction inside the binding
    assert out.psi_lcfs == pytest.approx(
        out.psi_axis + 0.999 * (out.psi_bnd - out.psi_axis), rel=1e-12
    )


def test_x_point_termination_binds_at_synthetic_saddle():
    """A diverted push terminates at the separatrix saddle flux and position."""
    psi, rg, zg, axis, lr, lz, inside = _diverted_field()
    saddle = _newton_saddle(_psi_diverted, 1.0, -0.25)
    psi_x = _psi_diverted(*saddle)
    out = cb.boundary_read(psi, _Grid(rg, zg, inside, lr, lz), axis)
    span = abs(out.psi_axis - psi_x)
    assert out.found and out.is_diverted
    # biquadratic sub-null refine: measured 2.9e-6 of span on this grid
    assert abs(out.psi_bnd - psi_x) / span < 1e-4
    xset = np.asarray(out.xset, dtype=np.float64)
    finite = xset[np.isfinite(xset).all(axis=1)]
    assert finite.shape[0] >= 1
    # measured 1.2e-5 m position error on this grid
    assert np.hypot(*(finite[0] - saddle)) < 1e-3


# --- continuity through the marginal transition ------------------------------


def _sweep_field(amp, nr=121, nz=161):
    """Plasma O-point plus a growing lower blob that pulls a saddle in-vessel.

    Small ``amp``: single O-point, wall-limited.  Large ``amp``: an in-vessel
    saddle binds and the plasma diverts.  A classify-first read (innermost
    in-vessel X flux, else wall tangency) steps when the saddle first
    classifies; the connectivity read never decides which surface bounds, so
    its binding flux stays smooth.
    """
    rg = np.linspace(0.25, 1.75, nr)
    zg = np.linspace(-1.15, 1.15, nz)
    rr, zz = np.meshgrid(rg, zg)
    psi = _psi_sweep(rr, zz, amp)
    lr = np.array([0.3, 1.7, 1.7, 0.3, 0.3])
    lz = np.array([-1.05, -1.05, 1.05, 1.05, -1.05])
    inside = _inside_polygon(rr.ravel(), zz.ravel(), lr, lz).reshape(nz, nr)
    return psi, rg, zg, (1.0, 0.1), lr, lz, inside


def _psi_sweep(r, z, amp):
    return np.exp(-(((r - 1.0) ** 2 + (z - 0.1) ** 2) / 0.34**2)) + amp * np.exp(
        -(((r - 1.0) ** 2 + (z + 0.7) ** 2) / 0.26**2)
    )


def _classify_first_psi_bnd(psi, rg, zg, inside, psi_axis, lr, lz, amp):
    """The classify-first binding: innermost in-vessel X flux, else tangency."""
    cands = xpoint_candidates(
        jnp.asarray(psi), jnp.asarray(rg), jnp.asarray(zg), jnp.asarray(inside)
    )
    x_psi = np.asarray(cands["psi"])[np.asarray(cands["valid"])]
    if x_psi.size:
        return float(x_psi.max()) - psi_axis, int(x_psi.size)
    wall_r, wall_z = _dense_wall(lr, lz)
    return float(_psi_sweep(wall_r, wall_z, amp).max()) - psi_axis, 0


def test_continuous_through_marginal_transition():
    """The binding flux is smooth across limited->diverted; classify-first steps."""
    amps = np.linspace(0.0, 0.9, 31)
    conn, clf, n_x = [], [], []
    for amp in amps:
        psi, rg, zg, axis, lr, lz, inside = _sweep_field(amp)
        read = cb.boundary_read(psi, _Grid(rg, zg, inside, lr, lz), axis)
        conn.append(read.psi_bnd - read.psi_axis)
        step, count = _classify_first_psi_bnd(
            psi, rg, zg, inside, read.psi_axis, lr, lz, amp
        )
        clf.append(step)
        n_x.append(count)
    conn = np.asarray(conn)
    clf = np.asarray(clf)
    span = float(np.nanmax(np.abs(conn)))
    conn_step = float(np.max(np.abs(np.diff(conn)))) / span
    clf_step = float(np.max(np.abs(np.diff(clf)))) / span
    assert int(np.sum(np.diff(n_x) != 0)) >= 1  # the sweep really transitions
    assert conn_step < 0.05
    assert clf_step > 3.0 * conn_step


# --- Lipschitz smooth-weight bound --------------------------------------------


def test_smooth_weight_is_lipschitz_in_psi():
    """A sub-temperature flux ripple moves the smooth weight boundedly.

    The sigmoid body moves by a bounded fraction (a hard mask flip is 1.0);
    the retracted flood gate is a boolean selection whose O(tau) shell caps
    any residual flip well below one.
    """
    psi, rg, zg, axis, lr, lz, inside = _limited_field()
    grid = _Grid(rg, zg, inside, lr, lz)
    base = cb.boundary_read_smooth(psi, grid, axis, temperature=1e-3)
    span = float(base["psi_bnd"] - base["psi_axis"])
    rng = np.random.default_rng(7)
    ripple = 1e-4 * abs(span) * rng.standard_normal(psi.shape)
    moved = cb.boundary_read_smooth(psi + ripple, grid, axis, temperature=1e-3)
    delta = np.abs(moved["core_weight"] - base["core_weight"])
    assert delta.max() < 0.8
    assert delta.mean() < 0.01


# --- no-flip across a topology switch -----------------------------------------


def _imbalanced_double_null(upper_amp, nr=45, nz=61):
    """The double-null field with a variable upper saddle depth."""
    rg = np.linspace(0.2, 1.8, nr)
    zg = np.linspace(-1.4, 1.4, nz)
    r0 = 1.0 + 0.5 * float(rg[1] - rg[0])
    rr, zz = np.meshgrid(rg, zg)

    def blob(z0, amp):
        return amp * np.exp(-((rr - r0) ** 2 / 0.45**2 + (zz - z0) ** 2 / 0.28**2))

    psi = blob(0.0, 1.0) + blob(-0.9, 0.9) + blob(0.9, upper_amp)
    lr = np.array([0.25, 1.75, 1.75, 0.25, 0.25])
    lz = np.array([-1.3, -1.3, 1.3, 1.3, -1.3])
    inside = _inside_polygon(rr.ravel(), zz.ravel(), lr, lz).reshape(nz, nr)
    return psi, rg, zg, (r0, 0.0), lr, lz, inside


def test_no_flip_on_topology_switch():
    """Sweeping the null imbalance never emits a mid-band X-slot blend.

    As the upper saddle deepens through the lower one's depth the binding
    null switches; every emitted slot must hold a REAL null position (|Z|
    near the saddle height), with the R/Z finite-masks coupled — a value
    forced between the nulls is the flip artifact this pins out.
    """
    heights = []
    for upper_amp in np.linspace(0.82, 0.98, 17):
        psi, rg, zg, axis, lr, lz, inside = _imbalanced_double_null(upper_amp)
        read = cb.boundary_read(psi, _Grid(rg, zg, inside, lr, lz), axis, lcfs_norm=1.0)
        xset = np.asarray(read.xset, dtype=np.float64)
        for row in xset:
            assert np.isfinite(row[0]) == np.isfinite(row[1])
            if np.isfinite(row[1]):
                heights.append(row[1])
    heights = np.abs(np.asarray(heights))
    assert heights.size >= 17  # at least the binding null every step
    assert np.all(np.abs(heights - 0.46) < 0.15)  # real null heights only


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
