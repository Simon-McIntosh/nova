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
import importlib.util
import inspect
import json
from types import SimpleNamespace

import numpy as np
import pytest

from nova.utilities.importmanager import skip_import

with skip_import("jax"):
    import jax
    import jax.numpy as jnp

    from nova.biot.null import Null1D, Null2D
    from nova.biot.target import FluxTarget
    from nova.equilibrium import connectivity_boundary as cb
    from nova.equilibrium.conservation import FluxLattice
    from nova.equilibrium.forward import ForwardProfile
    from nova.equilibrium.forward_operator import ForwardFluxOperator
    from nova.equilibrium.source import DomainProfile, ForwardSource
    from nova.equilibrium.stencil_nulls import xpoint_candidates
    from nova.equilibrium.topology import TopologyClass
    from nova.geometry.hexstencil import hex_stencil
    from nova.jax.config import configure_dtypes

from nova.equilibrium.labels import LCFS_ANGLES
from nova.equilibrium.wall_mask import inside_polygon as _inside_polygon


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


def _forward_operator(rg, zg, inside, wall_r, wall_z, *, polarity=1):
    """Build the topology seam without materialising Green matrices."""
    lattice = FluxLattice(rg, zg)
    node_count = lattice.node_count
    wall_count = len(wall_r)
    grid = FluxTarget(
        source_target=jnp.zeros((node_count, 1)),
        plasma_target=jnp.zeros((node_count, 1)),
        null=Null2D.from_coordinates(
            lattice.coordinate,
            hex_stencil(lattice.shape),
        ),
    )
    wall = FluxTarget(
        source_target=jnp.zeros((wall_count, 1)),
        plasma_target=jnp.zeros((wall_count, 1)),
        null=Null1D(jnp.asarray(np.c_[wall_r, wall_z], dtype=jnp.float64)),
    )

    def zero(psi_norm):
        return jnp.zeros_like(psi_norm)

    operator = ForwardFluxOperator(
        grid=grid,
        wall=wall,
        source=ForwardSource(core=DomainProfile(p_prime=zero, ff_prime=zero)),
        external_current=jnp.zeros(1),
        area=jnp.asarray(lattice.cell_area),
        inside_material=jnp.asarray(inside.T.ravel()),
        polarity=polarity,
        use_linear_moments=False,
    )
    return operator, lattice


def _forward_state(psi, wall_psi):
    """Pack a conventional height-by-radius field for the forward operator."""
    return jnp.asarray(np.r_[psi.T.ravel(), wall_psi])


def _class_disagreement_count(reads):
    """Count Boolean classifications that disagree with the margin sign."""
    classifications = np.asarray([bool(row[2].diverted) for row in reads])
    margins = np.asarray([float(row[2].class_margin) for row in reads])
    return int(np.count_nonzero(classifications != (margins >= 0.0)))


# --- accelerator compliance -------------------------------------------------


def test_explicit_double_extreme_reductions_keep_first_tie():
    """False-plus-allow keeps exact fp64 selection and first-index ties."""
    configure_dtypes()
    values = jnp.asarray([2.0, -3.0, 7.0, 7.0], dtype=jnp.float64)

    assert values.dtype == jnp.float64
    assert int(cb._argmax_exact(values)) == 2
    assert int(cb._argmin_exact(values)) == 1


def test_jit_vmap_grad_safe_and_fixed_shape():
    configure_dtypes()
    ang = jnp.asarray(np.asarray(LCFS_ANGLES))
    small = _limited_field(nr=61, nz=61)
    big = _diverted_field(nr=61, nz=61)

    def read(psi, rg, zg, inside, ar, az):
        return cb.traced_boundary_read(
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
    assert int(o_s["axis_state"]) in (1, 2)
    assert int(o_s["axis_candidate_count"]) >= 1
    assert int(o_s["x_candidate_count"]) >= 0
    assert int(o_s["x_unresolved_count"]) >= 0
    assert np.asarray(o_s["x_overflow"]).shape == ()

    psi, rg, zg, axis, _lr, _lz, inside = _limited_field(nr=61, nz=61)
    batch = jnp.stack(
        [jnp.asarray(psi), jnp.asarray(psi * 1.03), jnp.asarray(psi * 0.97)]
    )
    vfun = jax.vmap(
        lambda p: cb.traced_boundary_read(
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
        return cb.traced_boundary_read(
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


def test_typed_saddle_selector_masks_points_outside_wall_polygon():
    """A lower-level saddle inside the wall bounds but outside its polygon loses."""
    configure_dtypes()
    psi, rg, zg, axis, _lr, _lz, _inside = _limited_field(nr=41, nz=41)
    wall_r = np.asarray([0.5, 1.0, 1.5, 1.0, 0.5])
    wall_z = np.asarray([0.0, -0.8, 0.0, 0.8, 0.0])
    rr, zz = np.meshgrid(rg, zg)
    inside = _inside_polygon(rr.ravel(), zz.ravel(), wall_r, wall_z).reshape(zz.shape)
    psi_axis = float(psi[np.argmin(abs(zg)), np.argmin(abs(rg - axis[0]))])
    edge = np.concatenate([psi[0], psi[-1], psi[:, 0], psi[:, -1]])
    psi_out = edge[np.argmax(abs(edge - psi_axis))]
    span = psi_out - psi_axis

    candidates = jnp.asarray(
        [
            [0.55, 0.60, psi_axis + 0.20 * span, 0.0],
            [1.00, -0.50, psi_axis + 1.20 * span, 0.0],
        ]
    )
    wall_candidate = jnp.asarray([1.50, 0.0, psi_axis + 0.60 * span])
    wall_flux = jax.vmap(
        lambda r, z: cb._bilerp(
            jnp.asarray(psi), jnp.asarray(rg), jnp.asarray(zg), r, z
        )
    )(jnp.asarray(wall_r), jnp.asarray(wall_z))

    def diagnose(typed_candidates):
        return cb.traced_margin_candidate_diagnostics(
            jnp.asarray(psi),
            jnp.asarray(rg),
            jnp.asarray(zg),
            jnp.asarray(inside),
            jnp.asarray(axis[0]),
            jnp.asarray(axis[1]),
            32,
            10,
            jnp.asarray(wall_r),
            jnp.asarray(wall_z),
            wall_flux,
            typed_candidates,
            wall_candidate,
        )

    diagnostic = diagnose(candidates)
    assert np.asarray(diagnostic["typed_candidate_inside_wall"]).tolist() == [
        False,
        True,
    ]
    assert int(diagnostic["selected_typed_candidate_index"]) == 1
    np.testing.assert_allclose(
        np.asarray(diagnostic["selected_typed_candidate"][:2]),
        [1.0, -0.5],
        rtol=0.0,
        atol=0.0,
    )
    assert float(diagnostic["selected_x_normalized_flux_operand"]) == pytest.approx(1.2)
    assert int(diagnostic["reachable_wall_node_count"]) > 0
    assert np.isfinite(float(diagnostic["limiter_flux"]))
    assert float(diagnostic["wall_normalized_flux_operand"]) == pytest.approx(
        (float(diagnostic["limiter_flux"]) - float(diagnostic["axis_flux"]))
        / float(diagnostic["outward_flux_span"])
    )

    batched = jax.vmap(diagnose)(jnp.stack((candidates, candidates)))
    assert np.asarray(batched["selected_typed_candidate_index"]).tolist() == [1, 1]
    assert np.asarray(batched["typed_candidate_eligible"]).shape == (2, 2)


def test_reachable_wall_minimum_is_refined_along_polyline():
    """The reachable ring minimum moves off-node while a lower shadowed node loses."""
    configure_dtypes()
    rg = jnp.linspace(0.0, 4.0, 9)
    zg = jnp.linspace(-1.0, 1.0, 7)
    rr, zz = jnp.meshgrid(rg, zg)
    psi = (rr - 2.25) ** 2 + zz**2 + 1.0
    wall_r = jnp.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 0.0])
    wall_z = jnp.zeros_like(wall_r)
    wall_psi = (wall_r - 2.25) ** 2 + 1.0
    wall_psi = wall_psi.at[4].set(0.1)
    reachable = jnp.asarray([False, True, True, True, False, False])

    refined = cb._reachable_wall_limiter_point(
        psi,
        rg,
        zg,
        wall_r,
        wall_z,
        wall_psi,
        reachable,
        jnp.asarray(0.0),
        exact_nodes=True,
    )

    assert int(refined["node_index"]) == 2
    assert float(refined["node_arc"]) == pytest.approx(2.0)
    assert float(refined["shift"]) == pytest.approx(0.25)
    assert float(refined["arc"]) == pytest.approx(2.25)
    assert float(refined["r"]) == pytest.approx(2.25)
    assert float(refined["z"]) == pytest.approx(0.0)
    assert float(refined["psi"]) == pytest.approx(1.0)
    assert float(refined["distance"]) == pytest.approx(1.0)
    assert bool(refined["flux_from_global_surface"])

    node_only = cb._reachable_wall_limiter_point(
        psi,
        rg,
        zg,
        wall_r,
        wall_z,
        wall_psi,
        reachable.at[1].set(False).at[3].set(False),
        jnp.asarray(0.0),
        exact_nodes=True,
    )
    assert float(node_only["shift"]) == 0.0
    assert float(node_only["psi"]) == pytest.approx(1.0625)
    assert not bool(node_only["flux_from_global_surface"])

    compiled = jax.jit(
        cb._reachable_wall_limiter_point, static_argnames=("exact_nodes",)
    )(
        psi,
        rg,
        zg,
        wall_r,
        wall_z,
        wall_psi,
        reachable,
        jnp.asarray(0.0),
        exact_nodes=True,
    )
    np.testing.assert_allclose(compiled["r"], refined["r"], rtol=0.0, atol=0.0)
    batched = jax.vmap(
        lambda field, exact_flux, mask, axis_flux: cb._reachable_wall_limiter_point(
            field,
            rg,
            zg,
            wall_r,
            wall_z,
            exact_flux,
            mask,
            axis_flux,
            exact_nodes=True,
        )
    )(
        jnp.stack((psi, psi + 1.0)),
        jnp.stack((wall_psi, wall_psi + 1.0)),
        jnp.stack((reachable, reachable)),
        jnp.asarray([0.0, 1.0]),
    )
    np.testing.assert_allclose(
        batched["r"], [2.25, 2.25], rtol=0.0, atol=np.spacing(2.25)
    )


def test_pre_saddle_axis_component_uses_admissible_hex_links():
    """The production partition keeps the lobe behind a saddle unreachable."""
    configure_dtypes()
    inside = np.zeros((11, 13), dtype=bool)
    inside[2:5, 5:8] = True
    inside[5:8, 3:5] = True
    radial = np.arange(inside.shape[1], dtype=float)
    vertical = np.arange(inside.shape[0], dtype=float)
    radius, height = np.meshgrid(radial, vertical)
    saddle = -(radius - 4.5) * (height - 4.5)
    axis_cell = (3, 6)
    normalized_flux = saddle[axis_cell] - saddle

    region = cb._axis_component_before_level(
        jnp.asarray(normalized_flux),
        jnp.asarray(inside),
        jnp.asarray(radial),
        jnp.asarray(vertical),
        jnp.asarray(radial[axis_cell[1]]),
        jnp.asarray(vertical[axis_cell[0]]),
        jnp.asarray(saddle[axis_cell]),
    )

    assert bool(region[axis_cell])
    assert not bool(region[6, 3])
    assert int(jnp.count_nonzero(region)) == 9


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


def test_mechanism_names_replace_dependency_names_without_aliases():
    """The public boundary API distinguishes traced, smooth, and host reads."""
    assert cb.traced_boundary_read is not cb.traced_smooth_boundary_read
    assert callable(cb.host_boundary_read)
    assert callable(cb.host_boundary_read_smooth)
    assert callable(cb.host_boundary_read_batch)
    assert not hasattr(cb, "boundary_read_jax")
    assert not hasattr(cb, "boundary_read_smooth_jax")
    assert not hasattr(cb, "boundary_read")
    assert not hasattr(cb, "boundary_read_smooth")
    assert not hasattr(cb, "boundary_read_batch")
    assert importlib.util.find_spec("nova.jax.connectivity_boundary") is None


def test_batched_matches_per_slice():
    """vmap over a batch of psi fields equals the per-slice reads, element-wise."""
    ang = jnp.asarray(np.asarray(LCFS_ANGLES))
    psi, rg, zg, _axis, _lr, _lz, inside = _limited_field(nr=61, nz=61)
    slices = [jnp.asarray(psi), jnp.asarray(psi * 1.03), jnp.asarray(psi * 0.97)]

    def read(p):
        return cb.traced_boundary_read(
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
    gpu = cb.host_boundary_read(psi, _Grid(rg, zg, inside, lr, lz), axis, lcfs_norm=1.0)
    assert gpu.found and gpu.is_diverted
    assert gpu.axis_state == 2
    assert gpu.axis_candidate_count >= 1
    assert gpu.x_candidate_count >= 2
    assert not gpu.x_overflow
    assert gpu.x_binding_state in (1, 2)
    assert gpu.boundary_resolved == (gpu.x_binding_state == 2)
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
    out = cb.host_boundary_read(psi, grid, axis, wall_psi=wall_psi)
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
    out = cb.host_boundary_read(psi, _Grid(rg, zg, inside, lr, lz), axis)
    span = abs(out.psi_axis - psi_x)
    assert out.found and out.is_diverted
    # biquadratic sub-null refine: measured 2.9e-6 of span on this grid
    assert abs(out.psi_bnd - psi_x) / span < 1e-4
    xset = np.asarray(out.xset, dtype=np.float64)
    finite = xset[np.isfinite(xset).all(axis=1)]
    assert finite.shape[0] >= 1
    spline = cb.fit_tensor_spline(jnp.asarray(rg), jnp.asarray(zg), jnp.asarray(psi))
    polished = spline.evaluate(
        jnp.asarray(finite[0, 0], dtype=spline.coefficients.dtype),
        jnp.asarray(finite[0, 1], dtype=spline.coefficients.dtype),
    )
    assert (
        float(jnp.hypot(polished.radial_derivative, polished.vertical_derivative))
        < 1.0e-12
    )
    polished_axis = spline.evaluate(
        jnp.asarray(out.axis[0], dtype=spline.coefficients.dtype),
        jnp.asarray(out.axis[1], dtype=spline.coefficients.dtype),
    )
    assert (
        float(
            jnp.hypot(
                polished_axis.radial_derivative, polished_axis.vertical_derivative
            )
        )
        < 1.0e-12
    )
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


def _persistent_saddle_field(nr=81, nz=81):
    """A resolved axis/saddle pair whose wall level can cross the saddle."""
    rg = np.linspace(0.5, 1.5, nr)
    zg = np.linspace(-0.5, 0.5, nz)
    rr, zz = np.meshgrid(rg, zg)
    radial = rr - 1.0
    psi = -(radial**3 / 3.0 - 0.2**2 * radial + zz**2)
    lr = np.array([0.65, 1.45, 1.45, 0.65, 0.65])
    lz = np.array([-0.45, -0.45, 0.45, 0.45, -0.45])
    inside = _inside_polygon(rr.ravel(), zz.ravel(), lr, lz).reshape(nz, nr)
    return psi, rg, zg, (1.2, 0.0), lr, lz, inside


def _persistent_saddle_psi(r, z):
    """Evaluate the resolved axis/saddle field at arbitrary coordinates."""
    radial = np.asarray(r) - 1.0
    return -(radial**3 / 3.0 - 0.2**2 * radial + np.asarray(z) ** 2)


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
        read = cb.host_boundary_read(psi, _Grid(rg, zg, inside, lr, lz), axis)
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


@pytest.mark.slow
def test_forward_topology_margin_tracks_reachable_wall_and_terminal_gate():
    """The margin excludes wall flux that the pre-saddle core cannot reach."""
    configure_dtypes()
    amplitudes = np.linspace(0.0, 0.9, 19)
    growing_psi, growing_rg, growing_zg, _axis, lr, lz, growing_inside = _sweep_field(
        float(amplitudes[0]), 61, 61
    )
    growing_wall_r, growing_wall_z = _dense_wall(lr, lz, m=160)
    growing_operator, _growing_lattice = _forward_operator(
        growing_rg,
        growing_zg,
        growing_inside,
        growing_wall_r,
        growing_wall_z,
    )
    growing_reads = []
    for amplitude in amplitudes:
        growing_psi = _psi_sweep(*np.meshgrid(growing_rg, growing_zg), float(amplitude))
        growing_wall_psi = _psi_sweep(growing_wall_r, growing_wall_z, float(amplitude))
        growing_state = _forward_state(growing_psi, growing_wall_psi)
        _growing_masks, growing_topology = growing_operator.read(growing_state)
        growing_reads.append((float(amplitude), growing_state, growing_topology))
    growing_disagreement_count = _class_disagreement_count(growing_reads)
    assert growing_disagreement_count == 0

    psi, rg, zg, _axis, lr, lz, inside = _persistent_saddle_field()
    wall_r, wall_z = _dense_wall(lr, lz, m=160)
    operator, lattice = _forward_operator(rg, zg, inside, wall_r, wall_z)
    wall_psi = _persistent_saddle_psi(wall_r, wall_z)
    _base_masks, base = operator.read(_forward_state(psi, wall_psi))
    crossing_shift = float(base.x_point_flux) - float(np.max(wall_psi))
    scale = abs(float(base.axis_flux) - float(base.x_point_flux))
    shifts = crossing_shift + scale * np.linspace(-0.75, 0.75, 17)

    reads = []
    for shift in shifts:
        state = _forward_state(psi, wall_psi + float(shift))
        _masks, topology = operator.read(state)
        reads.append((float(shift), state, topology))

    classifications = np.asarray([bool(row[2].diverted) for row in reads])
    margins = np.asarray([float(row[2].class_margin) for row in reads])
    assert np.all(np.isposinf(margins))

    negative_operator, _negative_lattice = _forward_operator(
        rg,
        zg,
        inside,
        wall_r,
        wall_z,
        polarity=-1,
    )
    negative_reads = []
    for shift, state, _topology in reads:
        negative_state = -state
        _negative_masks, negative_topology = negative_operator.read(negative_state)
        negative_reads.append((shift, negative_state, negative_topology))
    negative_classifications = np.asarray(
        [bool(row[2].diverted) for row in negative_reads]
    )
    negative_margins = np.asarray(
        [float(row[2].class_margin) for row in negative_reads]
    )
    assert np.all(np.isposinf(negative_margins))
    assert np.array_equal(negative_classifications, classifications)

    # The whole-polygon landmark read crosses when unreachable wall flux is
    # translated through the saddle. Public class authority stays with the
    # connectivity comparator and therefore remains diverted at +infinity.
    assert np.all(classifications)
    assert _class_disagreement_count(reads) == 0

    diverted_state = reads[0][1]
    _diverted_masks, diverted = operator.read(diverted_state)
    assert bool(diverted.diverted)
    unreachable = int(np.argmin(wall_z))
    shadow_flux = np.asarray(wall_psi + reads[0][0]).copy()
    shadow_flux[unreachable] = 0.5 * (
        float(diverted.axis_flux) + float(diverted.x_point_flux)
    )
    assert shadow_flux[unreachable] > float(diverted.x_point_flux)
    _shadow_masks, shadowed = operator.read(_forward_state(psi, shadow_flux))
    assert bool(shadowed.diverted)
    assert np.isposinf(float(shadowed.class_margin))

    limited = reads[-1]
    profile = ForwardProfile(operator=operator, lattice=lattice)
    _limited_masks, limited_topology = operator.read(limited[1])
    from benchmarks.diiid_forward_gs_match import (  # noqa: PLC0415
        _terminal_xpoint_diagnostics,
    )

    xpoint_diagnostics = _terminal_xpoint_diagnostics(
        profile, limited[1], limited_topology
    )
    assert xpoint_diagnostics["typed_saddle_candidate_count"] >= 1
    np.testing.assert_allclose(
        xpoint_diagnostics["selected_x_coordinate_m"],
        np.asarray(limited_topology.x_point),
        rtol=0.0,
        atol=0.0,
    )
    assert xpoint_diagnostics["selected_x_flux_wb"] == float(
        limited_topology.x_point_flux
    )
    assert xpoint_diagnostics["class_margin_from_operands"] == float(
        limited_topology.class_margin
    )
    assert xpoint_diagnostics["wall_operand"]["normalized_flux"] is None
    assert (
        xpoint_diagnostics["wall_operand"]["normalized_flux_nonfinite"]
        == "positive_infinity"
    )
    assert xpoint_diagnostics["connectivity_admission"]["candidates"]
    serializable_diagnostics = {
        **xpoint_diagnostics,
        "class_margin_from_operands": None,
        "class_margin_from_operands_nonfinite": "positive_infinity",
    }
    json.dumps(serializable_diagnostics, allow_nan=False)
    equilibrium = SimpleNamespace(
        flux=limited[1],
        fixed_point=SimpleNamespace(residual=jnp.asarray(0.0)),
        finite=SimpleNamespace(passed=jnp.asarray(True)),
    )
    profile._solve_accelerated = lambda *_args, **_kwargs: equilibrium
    receipt = profile._branch_receipt(
        limited[1],
        TopologyClass.DIVERTED,
        None,
        route="fixed_point",
        tolerance=1.0e-12,
        iterations=0,
    )
    assert bool(receipt.achieved_class)
    assert bool(receipt.requested_class)
    assert bool(receipt.topology_consistent)
    assert bool(receipt.converged)
    assert float(receipt.residual) == 0.0

    print(
        "topology-margin evidence: "
        f"growing_classified={len(growing_reads)}, "
        f"growing_disagreement_count={growing_disagreement_count}, "
        f"persistent_classified={len(reads)}, "
        "persistent_reachable_wall_operand=false, "
        f"negative_classified={len(negative_reads)}, "
        "negative_reachable_wall_operand=false, "
        "unreachable_wall_vertex_ignored=true, terminal_wrong_class_rejected=true"
    )


def test_forward_topology_nan_is_explicitly_indeterminate(monkeypatch):
    """An unresolved comparator cannot silently qualify either topology class."""
    configure_dtypes()
    psi, rg, zg, _axis, lr, lz, inside = _persistent_saddle_field()
    wall_r, wall_z = _dense_wall(lr, lz, m=160)
    operator, _lattice = _forward_operator(rg, zg, inside, wall_r, wall_z)
    state = _forward_state(psi, _persistent_saddle_psi(wall_r, wall_z))
    calls = 0

    def indeterminate_read(*args, **kwargs):
        del args, kwargs
        nonlocal calls
        calls += 1
        return {"class_margin": jnp.asarray(jnp.nan)}

    monkeypatch.setattr(
        "nova.equilibrium.forward_operator.traced_boundary_read",
        indeterminate_read,
    )
    _masks, topology = operator.read(state, TopologyClass.DIVERTED)

    assert np.isnan(float(topology.class_margin))
    assert not bool(topology.class_determinate)
    assert not bool(topology.diverted)
    assert calls == 1


# --- Lipschitz smooth-weight bound --------------------------------------------


def test_smooth_weight_is_lipschitz_in_psi():
    """A sub-temperature flux ripple moves the smooth weight boundedly.

    The sigmoid body moves by a bounded fraction (a hard mask flip is 1.0);
    the retracted flood gate is a boolean selection whose O(tau) shell caps
    any residual flip well below one.
    """
    psi, rg, zg, axis, lr, lz, inside = _limited_field()
    grid = _Grid(rg, zg, inside, lr, lz)
    base = cb.host_boundary_read_smooth(psi, grid, axis, temperature=1e-3)
    span = float(base["psi_bnd"] - base["psi_axis"])
    rng = np.random.default_rng(7)
    ripple = 1e-4 * abs(span) * rng.standard_normal(psi.shape)
    moved = cb.host_boundary_read_smooth(psi + ripple, grid, axis, temperature=1e-3)
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
        read = cb.host_boundary_read(
            psi, _Grid(rg, zg, inside, lr, lz), axis, lcfs_norm=1.0
        )
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
