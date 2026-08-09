"""Tests for the contour-free, accelerator-native (JAX) flux-surface averaging.

The connectivity FSA is the device-native replacement for the host coarea
binning. Pinned three ways, none of which touch data:

* **accelerator compliance** - the kernel compiles and runs under ``jax.jit``,
  ``jax.vmap`` (a batch of psi fields on one grid), and ``jax.grad``; its output
  is FIXED-SHAPE regardless of how many cells fall in the core;
* **contour-free** - the module imports no contour / marching-squares / level-
  set / ``scipy.ndimage`` machinery, and its flood-fill core matches
  ``scipy.ndimage.label`` on a synthetic confined set;
* **fixed output shape** - the metric arrays are ``(n_psin,)`` whatever the core
  size, the property a device batch requires.

The engine-level physics-agreement cases (Ampere closure, coarea tracking) stay
with the free-boundary solver: they need the current-diffusion assembly.
"""

from __future__ import annotations

import importlib.util
import inspect

import numpy as np

from nova.utilities.importmanager import skip_import

with skip_import("jax"):
    import jax
    import jax.numpy as jnp

    from nova.equilibrium import flux_surface_connectivity as fsc


def _solovev_psi(*, nr=65, nz=97, rax=0.9, a=0.55, elong=1.6):
    """A Solov'ev-like psi with psi_axis > psi_bdry, + its grid."""
    rg = np.linspace(0.2, 1.6, nr)
    zg = np.linspace(-1.1, 1.1, nz)
    R, Z = np.meshgrid(rg, zg)
    psi_n = ((R - rax) / a) ** 2 + (Z / (a * elong)) ** 2
    psi = -psi_n  # axis at 0 (high), decreasing outward
    inside = np.ones((nz, nr), dtype=bool)
    inside[R < 0.25] = False
    return psi, rg, zg, inside


def _bins(psi, rg, zg, inside, psi_axis=0.0, psi_bnd=-1.0, n_psin=28):
    return fsc.traced_flux_surface_bins(
        jnp.asarray(psi),
        jnp.asarray(rg),
        jnp.asarray(zg),
        jnp.asarray(inside),
        jnp.asarray(float(psi_axis)),
        jnp.asarray(float(psi_bnd)),
        jnp.asarray(0.04),
        jnp.asarray(0.985),
        int(n_psin),
        jnp.asarray(1.25),
    )


def test_jax_fsa_is_fp64_jit_vmap_grad_safe():
    """The kernel runs fp64 and compiles under jit / vmap / grad."""
    psi, rg, zg, inside = _solovev_psi()
    out = _bins(psi, rg, zg, inside)
    assert out["inv_r2"].dtype == jnp.float64
    assert bool(out["well_posed"])
    assert np.all(np.isfinite(np.asarray(out["inv_r2"])))
    assert abs(float(out["inv_r2"][0]) - 1.0 / 0.9**2) < 0.1

    batch = jnp.stack(
        [jnp.asarray(psi), jnp.asarray(psi * 1.01), jnp.asarray(psi * 0.99)]
    )
    vfun = jax.vmap(
        lambda p: fsc.traced_flux_surface_bins(
            p,
            jnp.asarray(rg),
            jnp.asarray(zg),
            jnp.asarray(inside),
            jnp.asarray(0.0),
            jnp.asarray(-1.0),
            jnp.asarray(0.04),
            jnp.asarray(0.985),
            28,
            jnp.asarray(1.25),
        )["inv_r2"]
    )
    vb = vfun(batch)
    assert vb.shape == (3, 28)

    def loss(pb):
        o = fsc.traced_flux_surface_bins(
            jnp.asarray(psi),
            jnp.asarray(rg),
            jnp.asarray(zg),
            jnp.asarray(inside),
            jnp.asarray(0.0),
            pb,
            jnp.asarray(0.04),
            jnp.asarray(0.985),
            28,
            jnp.asarray(1.25),
        )
        return jnp.mean(o["inv_r2"])

    g = jax.grad(loss)(jnp.asarray(-1.0))
    assert np.isfinite(float(g))


def test_output_shape_is_fixed_independent_of_core_size():
    """The metric arrays are (n_psin,) whatever the core size."""
    n_psin = 28
    small = _solovev_psi(a=0.35)
    big = _solovev_psi(a=0.7)
    o_s = _bins(*small, n_psin=n_psin)
    o_b = _bins(*big, n_psin=n_psin)
    assert int(o_s["n_core_cells"]) != int(o_b["n_core_cells"])
    for k in ("pn_s", "dv_dpn", "inv_r2", "inv_r", "grad2_r2", "v_cum"):
        assert np.asarray(o_s[k]).shape == (n_psin,)
        assert np.asarray(o_b[k]).shape == (n_psin,)


def test_batched_matches_per_slice():
    """vmap over a batch of psi fields equals the per-slice metric reads."""
    psi, rg, zg, inside = _solovev_psi()
    slices = [jnp.asarray(psi), jnp.asarray(psi * 1.01), jnp.asarray(psi * 0.99)]

    def read(p):
        return fsc.traced_flux_surface_bins(
            p,
            jnp.asarray(rg),
            jnp.asarray(zg),
            jnp.asarray(inside),
            jnp.asarray(0.0),
            jnp.asarray(-1.0),
            jnp.asarray(0.04),
            jnp.asarray(0.985),
            28,
            jnp.asarray(1.25),
        )["inv_r2"]

    per_slice = np.stack([np.asarray(read(p)) for p in slices])
    batched = np.asarray(jax.vmap(read)(jnp.stack(slices)))
    assert np.allclose(batched, per_slice, atol=1e-10)


def test_module_is_contour_free():
    """The FSA path imports no contour / ndimage machinery (AST-checked)."""
    import ast

    tree = ast.parse(inspect.getsource(fsc))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            imported += [base] + [f"{base}.{a.name}" for a in node.names]
    banned = ("contourpy", "matplotlib", "skimage", "scipy.ndimage", "ndimage")
    for imp in imported:
        assert not any(b in imp for b in banned), f"contour-free FSA imports {imp!r}"
    calls = {
        n.func.attr
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    }
    assert "argsort" not in calls and "sort" not in calls


def test_traced_kernel_has_no_host_adapter_or_dependency_named_alias():
    """Only the traced fixed-shape implementation remains executable."""
    assert callable(fsc.traced_flux_surface_bins)
    assert not hasattr(fsc, "flux_surface_bins_jax")
    assert not hasattr(fsc, "flux_surface_bins")
    assert importlib.util.find_spec("nova.jax.flux_surface_connectivity") is None


def test_flood_fill_core_matches_ndimage_label():
    """The device flood-fill selects exactly the axis-connected component that
    scipy.ndimage.label would - connectivity correct, computed fixed-shape."""
    from scipy import ndimage

    psi, rg, zg, inside = _solovev_psi()
    R, _ = np.meshgrid(rg, zg)
    psi_n = (psi - 0.0) / (-1.0)
    confined = (psi_n < 1.0) & inside
    confined[5:10, 3:6] = True  # a disconnected private-like pocket
    ia = int(np.argmin(np.abs(zg - 0.0)))
    ja = int(np.argmin(np.abs(rg - 0.9)))
    seed = np.zeros_like(confined)
    seed[ia, ja] = True

    core = np.asarray(
        fsc.flood_fill_core(jnp.asarray(confined), jnp.asarray(seed), rg.size + zg.size)
    ).astype(bool)
    labels, _ = ndimage.label(confined)
    ref = labels == labels[ia, ja]
    assert np.array_equal(core, ref)
    assert not core[7, 4]  # the disconnected pocket is correctly excluded


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
