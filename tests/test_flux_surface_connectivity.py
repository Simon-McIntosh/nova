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
import math

import numpy as np
import pytest

from nova.geometry.hexstencil import HEX_RING, hex_stencil
from nova.jax.config import configure_dtypes
from nova.utilities.importmanager import skip_import

with skip_import("jax"):
    import jax
    import jax.numpy as jnp

    from nova.equilibrium import flux_surface_connectivity as fsc


def _f64(value):
    """Construct an explicitly selected double-precision JAX value."""
    configure_dtypes()
    return jnp.asarray(value, dtype=jnp.float64)


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
        _f64(psi),
        _f64(rg),
        _f64(zg),
        jnp.asarray(inside),
        _f64(psi_axis),
        _f64(psi_bnd),
        _f64(0.04),
        _f64(0.985),
        int(n_psin),
        _f64(1.25),
    )


def test_jax_fsa_is_fp64_jit_vmap_grad_safe():
    """The kernel runs fp64 and compiles under jit / vmap / grad."""
    psi, rg, zg, inside = _solovev_psi()
    out = _bins(psi, rg, zg, inside)
    assert out["inv_r2"].dtype == jnp.float64
    assert bool(out["well_posed"])
    assert np.all(np.isfinite(np.asarray(out["inv_r2"])))
    assert abs(float(out["inv_r2"][0]) - 1.0 / 0.9**2) < 0.1

    batch = jnp.stack([_f64(psi), _f64(psi * 1.01), _f64(psi * 0.99)])
    vfun = jax.vmap(
        lambda p: fsc.traced_flux_surface_bins(
            p,
            _f64(rg),
            _f64(zg),
            jnp.asarray(inside),
            _f64(0.0),
            _f64(-1.0),
            _f64(0.04),
            _f64(0.985),
            28,
            _f64(1.25),
        )["inv_r2"]
    )
    vb = vfun(batch)
    assert vb.shape == (3, 28)

    def loss(pb):
        o = fsc.traced_flux_surface_bins(
            _f64(psi),
            _f64(rg),
            _f64(zg),
            jnp.asarray(inside),
            _f64(0.0),
            pb,
            _f64(0.04),
            _f64(0.985),
            28,
            _f64(1.25),
        )
        return jnp.mean(o["inv_r2"])

    g = jax.grad(loss)(_f64(-1.0))
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
    slices = [_f64(psi), _f64(psi * 1.01), _f64(psi * 0.99)]

    def read(p):
        return fsc.traced_flux_surface_bins(
            p,
            _f64(rg),
            _f64(zg),
            jnp.asarray(inside),
            _f64(0.0),
            _f64(-1.0),
            _f64(0.04),
            _f64(0.985),
            28,
            _f64(1.25),
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

    core, steps = fsc.flood_fill_core_with_steps(
        jnp.asarray(confined), jnp.asarray(seed), rg.size + zg.size
    )
    core = np.asarray(core).astype(bool)
    labels, _ = ndimage.label(confined)
    ref = labels == labels[ia, ja]
    assert np.array_equal(core, ref)
    assert not core[7, 4]  # the disconnected pocket is correctly excluded
    assert int(steps) <= math.ceil(math.log2(rg.size + zg.size))


def test_component_labels_and_private_mask_match_ndimage():
    """Every component is labelled and only non-axis components are private."""
    from scipy import ndimage

    confined = np.zeros((19, 27), dtype=bool)
    confined[2:15, 2:9] = True
    confined[7:12, 9:17] = True
    confined[1:5, 20:25] = True
    confined[14:18, 18:22] = True
    seed = np.zeros_like(confined)
    seed[8, 4] = True

    labels, steps = fsc.label_connected_components_with_steps(
        jnp.asarray(confined), sum(confined.shape)
    )
    labels = np.asarray(labels)
    reference, component_count = ndimage.label(confined)
    private = np.asarray(fsc.private_flux_mask(labels, jnp.asarray(seed)))
    reference_private = (reference != 0) & (reference != reference[8, 4])

    assert np.array_equal(labels == 0, reference == 0)
    assert len(np.unique(labels[labels > 0])) == component_count == 3
    for reference_label in range(1, component_count + 1):
        component = reference == reference_label
        assert np.unique(labels[component]).size == 1
    assert np.array_equal(private, reference_private)
    assert int(steps) <= sum(confined.shape)

    batch = jnp.stack((jnp.asarray(confined), jnp.asarray(np.fliplr(confined))))
    batched = jax.vmap(
        lambda mask: fsc.label_connected_components(mask, sum(confined.shape))
    )(batch)
    per_slice = jnp.stack(
        [fsc.label_connected_components(mask, sum(confined.shape)) for mask in batch]
    )
    assert np.array_equal(np.asarray(batched), np.asarray(per_slice))


def test_connectivity_can_reject_inside_height_band_wall_cell():
    """A private cell inside the saddle-height band defeats the height proxy."""
    confined = np.zeros((13, 17), dtype=bool)
    confined[3:10, 2:9] = True
    confined[5:8, 12:16] = True
    seed = np.zeros_like(confined)
    seed[6, 4] = True

    labels = fsc.label_connected_components(jnp.asarray(confined), sum(confined.shape))
    private = np.asarray(fsc.private_flux_mask(labels, jnp.asarray(seed)))
    labels = np.asarray(labels)

    saddle_height_min = 2
    saddle_height_max = 10
    wall_cell = (6, 14)
    vertical_rule_shadowed = not (
        saddle_height_min <= wall_cell[0] <= saddle_height_max
    )
    connectivity_rule_shadowed = bool(private[wall_cell])

    assert not vertical_rule_shadowed
    assert connectivity_rule_shadowed
    assert int(labels[seed][0]) == 54
    assert int(labels[wall_cell]) == 98


def test_hex_component_labels_match_six_neighbour_reference():
    """Ring labels match a six-neighbour host reference and remain batchable."""
    from scipy import ndimage

    confined = np.zeros((19, 27), dtype=bool)
    confined[2:15, 2:9] = True
    confined[7:12, 9:17] = True
    confined[1:5, 20:25] = True
    confined[14:18, 18:22] = True
    rings = hex_stencil(confined.shape)
    structure = np.zeros((3, 3), dtype=bool)
    structure[1, 1] = True
    structure[HEX_RING[:, 0] + 1, HEX_RING[:, 1] + 1] = True

    labels, steps = fsc.label_hex_connected_components_with_steps(
        jnp.asarray(confined), jnp.asarray(rings), confined.size
    )
    labels = np.asarray(labels)
    reference, component_count = ndimage.label(confined, structure=structure)

    assert np.array_equal(labels == 0, reference == 0)
    assert len(np.unique(labels[labels > 0])) == component_count == 3
    for reference_label in range(1, component_count + 1):
        component = reference == reference_label
        assert np.unique(labels[component]).size == 1
    assert int(steps) <= confined.size

    batch = jnp.stack((jnp.asarray(confined), jnp.asarray(np.flipud(confined))))
    batched = jax.vmap(
        lambda mask: fsc.label_hex_connected_components(
            mask, jnp.asarray(rings), confined.size
        )
    )(batch)
    per_slice = jnp.stack(
        [
            fsc.label_hex_connected_components(mask, jnp.asarray(rings), confined.size)
            for mask in batch
        ]
    )
    assert np.array_equal(np.asarray(batched), np.asarray(per_slice))


def _ring_shared_edges(shape, rings, half_length=0.45):
    """Return square-index Voronoi edges for a centre-first ring gather."""
    row, column = np.indices(shape)
    centres = np.stack((column, row), axis=-1).reshape(-1, 2).astype(float)
    centre = centres[rings[:, :1]]
    neighbour = centres[rings]
    midpoint = 0.5 * (centre + neighbour)
    separation = neighbour - centre
    norm = np.linalg.norm(separation, axis=-1, keepdims=True)
    norm[:, 0] = 1.0
    tangent = np.stack((-separation[..., 1], separation[..., 0]), axis=-1) / norm
    endpoints = np.stack(
        (midpoint - half_length * tangent, midpoint + half_length * tangent), axis=-2
    )
    endpoints[:, 0] = centre[:, 0, None, :]
    return endpoints


def _saddle_pinch_fixture():
    """Return the diagonal pinch and its global smooth saddle field."""
    confined = np.zeros((11, 13), dtype=bool)
    confined[2:5, 5:8] = True
    confined[5:8, 3:5] = True
    seed = np.zeros_like(confined)
    seed[3, 6] = True
    vertical = np.arange(confined.shape[0], dtype=float)
    radial = np.arange(confined.shape[1], dtype=float)
    radius, height = np.meshgrid(radial, vertical)
    values = -(radius - 4.5) * (height - 4.5)
    rings = hex_stencil(confined.shape)
    shared_edges = _ring_shared_edges(confined.shape, rings)
    return confined, seed, radial, vertical, values, rings, shared_edges


def test_saddle_level_closes_hex_neck_without_changing_bulk_adjacency():
    """A strict shared-edge test preserves the private lobe behind a saddle."""
    confined, seed, radial, vertical, values, rings, shared_edges = (
        _saddle_pinch_fixture()
    )
    wall_cell = (6, 3)

    square_labels = fsc.label_connected_components(
        jnp.asarray(confined), sum(confined.shape)
    )
    hex_labels = fsc.label_hex_connected_components(
        jnp.asarray(confined),
        jnp.asarray(rings),
        confined.size,
    )
    link_admissible = fsc.hex_edge_admissibility(
        _f64(values),
        _f64(radial),
        _f64(vertical),
        _f64(0.0),
        _f64(values[seed][0]),
        _f64(shared_edges),
    )
    corrected_labels = fsc.label_saddle_aware_hex_connected_components(
        jnp.asarray(confined),
        jnp.asarray(rings),
        link_admissible,
        confined.size,
    )
    square_labels = np.asarray(square_labels)
    hex_labels = np.asarray(hex_labels)
    corrected_labels = np.asarray(corrected_labels)
    square_private = np.asarray(fsc.private_flux_mask(square_labels, jnp.asarray(seed)))
    hex_private = np.asarray(fsc.private_flux_mask(hex_labels, jnp.asarray(seed)))
    corrected_private = np.asarray(
        fsc.private_flux_mask(corrected_labels, jnp.asarray(seed))
    )

    assert int(np.count_nonzero(confined & ~square_private)) == 9
    assert int(np.count_nonzero(square_private)) == 6
    assert int(np.count_nonzero(confined & ~hex_private)) == 15
    assert int(np.count_nonzero(hex_private)) == 0
    assert int(np.count_nonzero(confined & ~corrected_private)) == 9
    assert int(np.count_nonzero(corrected_private)) == 6
    assert int(square_labels[seed][0]) == 32
    assert int(square_labels[wall_cell]) == 69
    assert int(hex_labels[seed][0]) == int(hex_labels[wall_cell]) == 32
    assert int(corrected_labels[seed][0]) == 32
    assert int(corrected_labels[wall_cell]) == 69
    assert square_private[wall_cell]
    assert not hex_private[wall_cell]
    assert corrected_private[wall_cell]

    bridge_flat = np.ravel_multi_index((4, 5), confined.shape)
    bridge_ring = np.flatnonzero(rings[:, 0] == bridge_flat)[0]
    bridge_slot = np.flatnonzero(
        rings[bridge_ring] == np.ravel_multi_index((5, 4), confined.shape)
    )[0]
    assert not bool(np.asarray(link_admissible)[bridge_ring, bridge_slot])

    cell_area = np.linspace(0.7, 1.3, confined.size).reshape(confined.shape)
    public_area = np.where(confined & ~corrected_private, cell_area, 0.0)
    private_area = np.where(corrected_private, cell_area, 0.0)
    scrape_off_area = np.where(~confined, cell_area, 0.0)
    np.testing.assert_allclose(public_area + private_area + scrape_off_area, cell_area)


def test_saddle_aware_hex_labels_are_fixed_shape_jit_and_vmap_safe():
    """Cells-by-seven link masks stay fixed under compilation and batching."""
    confined, _seed, radial, vertical, values, rings, shared_edges = (
        _saddle_pinch_fixture()
    )
    admissible = fsc.hex_edge_admissibility(
        _f64(values),
        _f64(radial),
        _f64(vertical),
        _f64(0.0),
        _f64(values[3, 6]),
        _f64(shared_edges),
    )
    assert admissible.shape == rings.shape
    assert bool(np.all(np.asarray(admissible)[:, 0]))

    batch = jnp.stack((jnp.asarray(confined), jnp.asarray(np.flipud(confined))))
    batched = jax.jit(
        jax.vmap(
            lambda mask: fsc.label_saddle_aware_hex_connected_components(
                mask, jnp.asarray(rings), admissible, confined.size
            )
        )
    )(batch)
    per_slice = jnp.stack(
        [
            fsc.label_saddle_aware_hex_connected_components(
                mask, jnp.asarray(rings), admissible, confined.size
            )
            for mask in batch
        ]
    )
    assert np.array_equal(np.asarray(batched), np.asarray(per_slice))
    assert "pure_callback" not in inspect.getsource(fsc)


def test_doubling_fill_matches_fixed_iteration_fixtures_exactly():
    """Doubling and one-cell dilation reach exactly the same fixed point."""

    def fixed_iteration_fill(confined, seed):
        core = seed & confined
        for _ in range(sum(confined.shape)):
            up = np.zeros_like(core)
            down = np.zeros_like(core)
            left = np.zeros_like(core)
            right = np.zeros_like(core)
            up[1:, :] = core[:-1, :]
            down[:-1, :] = core[1:, :]
            left[:, 1:] = core[:, :-1]
            right[:, :-1] = core[:, 1:]
            core = (core | up | down | left | right) & confined
        return core

    fixtures = []
    psi, rg, zg, inside = _solovev_psi()
    psi_n = psi / -1.0
    ellipse = (psi_n < 1.0) & inside
    ellipse[5:10, 3:6] = True
    fixtures.append((ellipse, (np.abs(zg).argmin(), np.abs(rg - 0.9).argmin())))

    corridor = np.zeros((49, 73), dtype=bool)
    corridor[8:41, 6:31] = True
    corridor[22:27, 31:58] = True
    corridor[13:36, 58:67] = True
    corridor[3:9, 48:55] = True  # disconnected pocket above the corridor
    fixtures.append((corridor, (24, 12)))

    for confined, seed_index in fixtures:
        seed = np.zeros_like(confined)
        seed[seed_index] = True
        fixed = fixed_iteration_fill(confined, seed)
        doubled, steps = fsc.flood_fill_core_with_steps(
            jnp.asarray(confined), jnp.asarray(seed), sum(confined.shape)
        )
        assert np.array_equal(np.asarray(doubled).astype(bool), fixed)
        assert int(steps) <= math.ceil(math.log2(sum(confined.shape)))


def test_stationary_point_polish_converges_for_saddles_and_extrema():
    """Fixed-slot spline Newton polish reaches machine-precision stationarity."""
    radial = _f64(np.linspace(0.2, 2.0, 9))
    vertical = _f64(np.linspace(-1.4, 1.3, 11))
    mesh_r, mesh_z = jnp.meshgrid(radial, vertical)
    stationary_rz = np.asarray((1.13, -0.27))
    seed_rz = _f64(
        np.asarray(
            (
                (1.06, -0.19),
                (1.21, -0.36),
                (0.0, 0.0),
            )
        )
    )
    valid = jnp.asarray((True, True, False))

    fields_and_types = (
        ((mesh_r - stationary_rz[0]) ** 2 - (mesh_z - stationary_rz[1]) ** 2, -1),
        ((mesh_r - stationary_rz[0]) ** 2 + (mesh_z - stationary_rz[1]) ** 2, 1),
        (-((mesh_r - stationary_rz[0]) ** 2) - (mesh_z - stationary_rz[1]) ** 2, 1),
    )
    for values, expected_type in fields_and_types:
        spline = fsc.fit_tensor_spline(radial, vertical, values)
        result = fsc.polish_stationary_points(spline, seed_rz, valid)
        np.testing.assert_allclose(
            np.asarray(result["position_rz"][:2]),
            np.broadcast_to(stationary_rz, (2, 2)),
            atol=2e-15,
            rtol=0.0,
        )
        assert float(jnp.max(result["gradient_norm"][:2])) < 1e-14
        assert np.array_equal(
            np.asarray(result["hessian_type"]), [expected_type] * 2 + [0]
        )
        assert np.array_equal(np.asarray(result["converged"]), [True, True, False])
        assert np.array_equal(np.asarray(result["in_domain"]), [True, True, False])
        assert np.array_equal(np.asarray(result["position_rz"][-1]), [0.0, 0.0])
        assert float(result["value"][-1]) == 0.0
        assert float(result["gradient_norm"][-1]) == 0.0


def test_stationary_point_polish_has_eager_jit_and_vmap_parity():
    """The fixed-slot result is identical across JAX execution transforms."""
    radial = _f64(np.linspace(0.1, 2.1, 10))
    vertical = _f64(np.linspace(-1.5, 1.5, 12))
    mesh_r, mesh_z = jnp.meshgrid(radial, vertical)
    values = (mesh_r - 1.17) ** 2 - 1.3 * (mesh_z + 0.31) ** 2
    spline = fsc.fit_tensor_spline(radial, vertical, values)
    seeds = _f64(np.asarray(((1.08, -0.23), (1.24, -0.38), (9.0, 9.0))))
    valid = jnp.asarray((True, True, True))

    eager = fsc.polish_stationary_points(spline, seeds, valid)
    compiled = jax.jit(fsc.polish_stationary_points)(spline, seeds, valid)
    batch_seeds = jnp.stack((seeds, seeds.at[:, 0].add(0.01)))
    batched = jax.vmap(lambda item: fsc.polish_stationary_points(spline, item, valid))(
        batch_seeds
    )
    per_slice = jax.tree.map(
        lambda *items: jnp.stack(items),
        *[fsc.polish_stationary_points(spline, item, valid) for item in batch_seeds],
    )

    for key in eager:
        np.testing.assert_array_equal(np.asarray(compiled[key]), np.asarray(eager[key]))
        np.testing.assert_array_equal(
            np.asarray(batched[key]), np.asarray(per_slice[key])
        )
    assert np.array_equal(np.asarray(eager["in_domain"]), [True, True, False])
    assert not bool(eager["converged"][-1])


def test_stationary_point_polish_warm_exit_and_implicit_gradient():
    """Warm roots cost zero trips and expose the stationary solution derivative."""
    radial = _f64(np.linspace(0.2, 2.0, 9))
    vertical = _f64(np.linspace(-1.4, 1.3, 11))
    mesh_r, mesh_z = jnp.meshgrid(radial, vertical)
    seed = _f64(np.asarray(((1.13, -0.27),)))
    valid = jnp.asarray((True,))

    def polished_radius(radial_shift):
        values = (mesh_r - (1.13 + radial_shift)) ** 2 - (mesh_z + 0.27) ** 2
        spline = fsc.fit_tensor_spline(radial, vertical, values)
        return fsc.polish_stationary_points(spline, seed, valid)["position_rz"][0, 0]

    result = fsc.polish_stationary_points(
        fsc.fit_tensor_spline(
            radial, vertical, (mesh_r - 1.13) ** 2 - (mesh_z + 0.27) ** 2
        ),
        seed,
        valid,
    )
    assert int(result["iteration_count"][0]) == 0
    assert float(jax.grad(polished_radius)(_f64(0.0))) == pytest.approx(
        1.0, abs=2.0e-13
    )


def test_traced_contour_retains_cell_confined_saddle_polish():
    """Contour ambiguity resolution still polishes inside the owning cell."""
    radial = _f64(np.linspace(0.2, 2.0, 9))
    vertical = _f64(np.linspace(-1.4, 1.3, 11))
    stationary_rz = np.asarray(
        (
            0.5 * float(radial[4] + radial[5]),
            0.5 * float(vertical[4] + vertical[5]),
        )
    )
    mesh_r, mesh_z = jnp.meshgrid(radial, vertical)
    values = (mesh_r - stationary_rz[0]) * (mesh_z - stationary_rz[1])

    result = fsc.traced_spline_contour(values, radial, vertical, _f64(0.0))
    ambiguous = np.asarray(result["ambiguous_saddle"])

    assert np.count_nonzero(ambiguous) == 1
    np.testing.assert_allclose(
        np.asarray(result["saddle_rz"])[ambiguous],
        stationary_rz[None, :],
        atol=2e-15,
        rtol=0.0,
    )
    assert bool(np.asarray(result["saddle_stationary"])[ambiguous][0])
    assert bool(np.asarray(result["ambiguous_tie_broken"])[ambiguous][0])


def test_traced_contour_assigns_shared_edge_nodes_and_saddle_segments():
    """Contour arcs expose canonical graph nodes and exact-level saddle slots."""
    radial = _f64(np.linspace(-2.0, 2.0, 34))
    vertical = radial
    mesh_r, mesh_z = jnp.meshgrid(radial, vertical)
    rotated_radius = (mesh_r + mesh_z) / np.sqrt(2.0)
    rotated_height = (mesh_z - mesh_r) / np.sqrt(2.0)
    values = rotated_height**2 - rotated_radius**2 + rotated_radius**3

    result = fsc.traced_spline_contour(values, radial, vertical, _f64(0.0))
    valid = np.asarray(result["segment_valid"]).reshape(-1)
    nodes = np.asarray(result["segment_node_indices"]).reshape(-1, 2)[valid]
    saddle_segment = np.asarray(result["segment_at_saddle"])

    unique, counts = np.unique(nodes.reshape(-1), return_counts=True)
    assert np.count_nonzero(counts == 1) == 2
    assert np.count_nonzero(counts == 2) == unique.size - 2
    assert np.count_nonzero(saddle_segment) == 2
    np.testing.assert_array_equal(
        np.asarray(result["segment_saddle_rz"])[~saddle_segment], 0.0
    )


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
