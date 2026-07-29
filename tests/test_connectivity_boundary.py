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
  saddles, never two copies of one stencil-degenerate saddle.

Host-reproduction (device vs the scipy ``lcfs_contour``) and the limited->diverted
continuity gate are re-banked: they require the host topology labeller and the
gate-eval script, which are not part of this increment.
"""

from __future__ import annotations

import ast
import inspect

import numpy as np

from nova.utilities.importmanager import skip_import

with skip_import("jax"):
    import jax
    import jax.numpy as jnp

    from nova.jax import connectivity_boundary as cb

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


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
