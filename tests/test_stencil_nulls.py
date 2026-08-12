"""Sub-grid stencil null locator — synthetic-field unit tests.

Fast, self-contained (no IMAS data): verifies the rectangular and hex-ring
classifiers (0 → O, 4 → X), the biquadratic sub-grid refinement against a
symmetry-known X-point, and that the axis position carries a finite ``jax.grad``.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys

import numpy as np
import pytest

from nova.utilities.importmanager import skip_import

HEX_QUADRATIC_CONDITION = 5.47318837498164

with skip_import("jax"):
    import jax
    import jax.numpy as jnp

    from nova.equilibrium.stencil_nulls import (
        STATE_RESOLVED,
        STATE_UNRESOLVED,
        critical_point_candidates_batch,
        magnetic_axis_subgrid,
        ring_sign_changes,
        xpoint_candidates,
    )
    from nova.jax.config import configure_dtypes


def test_equilibrium_module_is_the_only_stencil_implementation():
    assert importlib.util.find_spec("nova.jax.stencil_nulls") is None


def test_fieldnull_result_is_independent_of_connectivity_import_order():
    """Explicit dtype setup removes the connectivity x64 import-order effect."""
    script = """
import importlib
import json
import sys

import numpy as np

for name in sys.argv[1:]:
    importlib.import_module(name)

from nova.jax.config import configure_dtypes
configure_dtypes()

import jax
import jax.numpy as jnp
from nova.equilibrium.stencil_nulls import magnetic_axis_subgrid

rg = np.linspace(6.18, 6.22, 21, dtype=np.float64)
zg = np.linspace(-0.03, 0.03, 25, dtype=np.float64)
rr, zz = np.meshgrid(rg, zg)
truth = (6.2031, -0.0047)
field = -((rr - truth[0]) ** 2 + 1.3 * (zz - truth[1]) ** 2)
result = magnetic_axis_subgrid(
    jnp.asarray(field, dtype=jnp.float32),
    jnp.asarray(rg, dtype=jnp.float64),
    jnp.asarray(zg, dtype=jnp.float64),
    jnp.ones(field.shape, dtype=bool),
)
print(json.dumps({
    'r': float(result['r']),
    'z': float(result['z']),
    'kind': float(result['ntype']),
    'found': bool(result['found']),
    'x64': bool(jax.config.x64_enabled),
}))
"""
    modules = (
        "nova.equilibrium.stencil_nulls",
        "nova.equilibrium.flux_surface_connectivity",
    )
    rows = []
    for order in (modules, modules[::-1]):
        result = subprocess.run(
            [sys.executable, "-c", script, *order],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        rows.append(json.loads(result.stdout.splitlines()[-1]))

    assert rows[0] == rows[1]
    assert rows[0]["found"]
    assert rows[0]["x64"]
    assert rows[0]["kind"] == 1.0
    assert abs(rows[0]["r"] - 6.2031) < 2e-5
    assert abs(rows[0]["z"] + 0.0047) < 2e-5


def _two_peak_field(nr=81, nz=101, rc=1.007, z1=-0.30, z2=0.30, w=0.15):
    """Two positive Gaussians stacked in Z: two O-points, one X-point between.

    By the up-down symmetry the saddle sits exactly at ``(rc, (z1+z2)/2)``, giving
    a ground-truth sub-grid target the biquadratic refinement must recover.
    """
    rg = np.linspace(0.5, 1.5, nr)
    zg = np.linspace(-0.8, 0.8, nz)
    rr, zz = np.meshgrid(rg, zg)  # (nz, nr)
    psi = np.exp(-(((rr - rc) ** 2 + (zz - z1) ** 2) / w**2)) + np.exp(
        -(((rr - rc) ** 2 + (zz - z2) ** 2) / w**2)
    )
    return jnp.asarray(psi), jnp.asarray(rg), jnp.asarray(zg), rc, 0.5 * (z1 + z2)


def test_classifier_finds_o_and_x():
    psi, rg, zg, _rc, _xz = _two_peak_field()
    counts = np.asarray(ring_sign_changes(psi))
    assert (counts == 0).any(), "no O-point (0 sign changes) classified"
    assert (counts == 4).any(), "no X-point (4 sign changes) classified"
    # the border carries no full ring
    assert (counts[0, :] == -1).all() and (counts[:, 0] == -1).all()


def test_hex_plasma_stencil_classifies_saddle_and_has_full_rank_fit():
    from scipy.spatial import Delaunay

    from nova.biot.plasmagrid import PlasmaGrid

    angles = np.arange(6) * np.pi / 3.0
    points = np.vstack(([0.0, 0.0], np.column_stack((np.cos(angles), np.sin(angles)))))
    triangulation = Delaunay(points)
    boundary = np.unique(triangulation.convex_hull)
    stencil, centre_index = PlasmaGrid.loop_neighbour_vertices(
        points, triangulation.vertex_neighbor_vertices, boundary
    )

    assert stencil.shape == (1, 7)
    np.testing.assert_array_equal(centre_index, [0])
    saddle = points[:, 0] ** 2 - points[:, 1] ** 2
    np.testing.assert_array_equal(
        np.asarray(ring_sign_changes(jnp.asarray(saddle), jnp.asarray(stencil))), [4]
    )

    cluster = points[stencil[0]]
    offsets = cluster - cluster[0]
    local = offsets / np.max(np.abs(offsets), axis=0)
    x_coordinate, z_coordinate = local.T
    design = np.column_stack(
        (
            x_coordinate**2,
            z_coordinate**2,
            x_coordinate,
            z_coordinate,
            x_coordinate * z_coordinate,
            np.ones_like(x_coordinate),
        )
    )
    assert np.linalg.matrix_rank(design) == 6
    assert np.linalg.cond(design) == pytest.approx(HEX_QUADRATIC_CONDITION, rel=1.0e-12)


def test_axis_is_a_gaussian_peak():
    psi, rg, zg, rc, _xz = _two_peak_field()
    inside = jnp.ones(psi.shape, dtype=bool)
    ax = magnetic_axis_subgrid(psi, rg, zg, inside)
    assert bool(ax["found"])
    dr = float(rg[1] - rg[0])
    # the axis is one of the two peaks: near rc in R, near ±0.30 in Z
    assert abs(float(ax["r"]) - rc) < 3 * dr
    assert min(abs(float(ax["z"]) - 0.30), abs(float(ax["z"]) + 0.30)) < 3 * float(
        zg[1] - zg[0]
    )
    assert float(ax["ntype"]) > 0  # a maximum (both curvatures negative → +1)


def test_xpoint_subgrid_matches_symmetry():
    psi, rg, zg, rc, xz = _two_peak_field()
    inside = jnp.ones(psi.shape, dtype=bool)
    xc = xpoint_candidates(psi, rg, zg, inside, k_slots=6)
    valid = np.asarray(xc["valid"])
    assert valid.any(), "no valid X-point found"
    rr = np.asarray(xc["r"])[valid]
    zz = np.asarray(xc["z"])[valid]
    # the symmetry saddle nearest the known midpoint
    d = np.hypot(rr - rc, zz - xz)
    k = int(np.argmin(d))
    dr = float(rg[1] - rg[0])
    dz = float(zg[1] - zg[0])
    assert abs(rr[k] - rc) < dr, f"X R off by {abs(rr[k] - rc):.4f} (> {dr:.4f})"
    assert abs(zz[k] - xz) < dz, f"X Z off by {abs(zz[k] - xz):.4f} (> {dz:.4f})"
    types = np.asarray(xc["ntype"])[valid]
    assert abs(types[k]) < 0.5, "midpoint null is not typed as a saddle"


def test_axis_gradient_is_finite():
    psi, rg, zg, _rc, _xz = _two_peak_field()
    inside = jnp.ones(psi.shape, dtype=bool)

    def axis_r(p):
        return magnetic_axis_subgrid(p, rg, zg, inside)["r"]

    g = jax.grad(axis_r)(psi)
    g = np.asarray(g)
    assert np.all(np.isfinite(g)), "axis-R gradient has non-finite entries"
    assert np.any(g != 0.0), "axis-R gradient is identically zero (no signal)"


def test_extra_mask_restricts_candidates():
    psi, rg, zg, rc, xz = _two_peak_field()
    inside = jnp.ones(psi.shape, dtype=bool)
    # a flux-proximity band far from the saddle flux removes the X entirely
    xc_all = xpoint_candidates(psi, rg, zg, inside, k_slots=6)
    assert np.asarray(xc_all["valid"]).any()
    empty_mask = jnp.zeros(psi.shape, dtype=bool)
    xc_none = xpoint_candidates(psi, rg, zg, inside, k_slots=6, extra_mask=empty_mask)
    assert not np.asarray(xc_none["valid"]).any()


def _periodic_field(resolution=61):
    radial = np.linspace(0.5, 1.5, resolution)
    vertical = np.linspace(-0.8, 0.8, resolution)
    rr, zz = np.meshgrid(radial, vertical)
    field = np.sin(8.0 * np.pi * (rr - 0.5)) * np.sin(8.0 * np.pi * (zz + 0.8) / 1.6)
    return field, radial, vertical


def test_native_degree_collapses_legacy_duplicates_and_reports_overflow():
    configure_dtypes()
    field, radial, vertical = _periodic_field()
    inside = jnp.ones(field.shape, dtype=bool)
    legacy_raw = (
        np.asarray(ring_sign_changes(jnp.asarray(field, dtype=jnp.float64))) == 4
    )
    assert int(np.sum(legacy_raw)) == 99

    result = xpoint_candidates(
        jnp.asarray(field),
        jnp.asarray(radial),
        jnp.asarray(vertical),
        inside,
        k_slots=8,
        material_dilate=0,
        noise_sigma=0.0,
    )
    assert int(result["candidate_count"]) == 49
    assert int(result["candidate_index_sum"]) == -49
    assert int(result["domain_signed_index"]) == 15
    assert bool(result["overflow"])
    assert np.isfinite(float(result["discarded_score_upper_bound"]))
    assert np.all(np.asarray(result["cluster_size"])[result["present"]] == 1)


def test_plateau_is_absent_and_noise_candidates_remain_unresolved():
    grid = jnp.linspace(-1.0, 1.0, 41)
    inside = jnp.ones((41, 41), dtype=bool)
    flat = magnetic_axis_subgrid(jnp.zeros((41, 41)), grid, grid, inside)
    assert not bool(flat["found"])
    assert not bool(flat["present"])

    noise = np.random.default_rng(19).standard_normal((41, 41))
    candidates = xpoint_candidates(
        jnp.asarray(noise), grid, grid, inside, k_slots=32, noise_sigma=1.0
    )
    assert int(candidates["candidate_count"]) > 0
    assert not np.asarray(candidates["resolved"]).any()
    assert np.all(
        np.asarray(candidates["state"])[np.asarray(candidates["present"])]
        == STATE_UNRESOLVED
    )


def test_host_float64_field_keeps_the_device_axis_contract():
    """A host round trip must not silently demote an explicit fp64 flux map."""
    configure_dtypes()
    radial = np.linspace(0.65, 1.35, 17, dtype=np.float64)
    vertical = np.linspace(-0.5, 0.5, 17, dtype=np.float64)
    rr, zz = np.meshgrid(radial, vertical)
    field = -((rr - 1.021) ** 2 + 1.3 * (zz - 0.013) ** 2)
    inside = np.ones(field.shape, dtype=bool)
    host = magnetic_axis_subgrid(field, radial, vertical, inside)
    device = magnetic_axis_subgrid(
        jnp.asarray(field, dtype=jnp.float64),
        jnp.asarray(radial, dtype=jnp.float64),
        jnp.asarray(vertical, dtype=jnp.float64),
        jnp.asarray(inside),
    )
    np.testing.assert_array_equal(
        [float(host["r"]), float(host["z"])],
        [float(device["r"]), float(device["z"])],
    )


def test_weak_native_candidate_is_retained_before_confidence_resolution():
    coordinate = jnp.arange(-10.0, 11.0)
    rr, zz = jnp.meshgrid(coordinate, coordinate)
    inside = jnp.ones(rr.shape, dtype=bool)
    weak = 0.005 * ((rr - 0.3) ** 2 - 0.8 * (zz + 0.2) ** 2)
    result = xpoint_candidates(
        weak,
        coordinate,
        coordinate,
        inside,
        k_slots=4,
        material_dilate=0,
        noise_sigma=0.01,
    )
    assert int(result["candidate_count"]) == 1
    assert bool(result["present"][0])
    assert int(result["state"][0]) == STATE_UNRESOLVED
    assert int(result["native_signed_index"][0]) == -1


def test_normalized_fp32_fit_is_stable_at_iter_scale_coordinates():
    radial = np.linspace(6.18, 6.26, 41)
    vertical = np.linspace(-0.04, 0.04, 41)
    spacing = radial[1] - radial[0]
    truth = np.array([6.2187, -0.0113])
    rr, zz = np.meshgrid(radial, vertical)
    field = ((rr - truth[0]) / spacing) ** 2 - 0.7 * ((zz - truth[1]) / spacing) ** 2
    result = xpoint_candidates(
        jnp.asarray(field, dtype=jnp.float32),
        jnp.asarray(radial),
        jnp.asarray(vertical),
        jnp.ones(field.shape, dtype=bool),
        k_slots=4,
        material_dilate=0,
        noise_sigma=0.0,
    )
    assert bool(result["resolved"][0])
    error_cells = (
        np.hypot(float(result["r"][0]) - truth[0], float(result["z"][0]) - truth[1])
        / spacing
    )
    assert error_cells <= 0.02


def test_batch_order_and_scalar_adapter_agree():
    field, radial, vertical = _periodic_field(41)
    fields = jnp.asarray(np.stack([field, 0.7 * field]))
    inside = jnp.ones(field.shape, dtype=bool)
    batched = critical_point_candidates_batch(
        fields,
        jnp.asarray(radial),
        jnp.asarray(vertical),
        inside,
        k_slots=8,
        material_dilate=0,
        target_index=-1,
        noise_sigma=0.0,
    )
    reversed_batch = critical_point_candidates_batch(
        fields[::-1],
        jnp.asarray(radial),
        jnp.asarray(vertical),
        inside,
        k_slots=8,
        material_dilate=0,
        target_index=-1,
        noise_sigma=0.0,
    )
    scalar = xpoint_candidates(
        fields[0],
        jnp.asarray(radial),
        jnp.asarray(vertical),
        inside,
        k_slots=8,
        material_dilate=0,
        noise_sigma=0.0,
    )
    np.testing.assert_array_equal(
        np.asarray(batched["source_cell"]),
        np.asarray(reversed_batch["source_cell"])[::-1],
    )
    np.testing.assert_array_equal(
        np.asarray(batched["source_cell"])[0], np.asarray(scalar["source_cell"])
    )
    np.testing.assert_allclose(
        np.asarray(batched["r"])[0], np.asarray(scalar["r"]), equal_nan=True
    )
    assert np.all(
        np.asarray(batched["state"])[np.asarray(batched["resolved"])] == STATE_RESOLVED
    )


def test_cpu_gpu_candidate_metadata_and_coordinates_match():
    try:
        gpu = jax.devices("gpu")[0]
    except RuntimeError:
        pytest.skip("no GPU device")
    cpu = jax.devices("cpu")[0]
    field, radial, vertical = _periodic_field(41)
    inside = np.ones(field.shape, dtype=bool)

    def run(device):
        return xpoint_candidates(
            jax.device_put(field, device),
            jax.device_put(radial, device),
            jax.device_put(vertical, device),
            jax.device_put(inside, device),
            k_slots=16,
            material_dilate=0,
            noise_sigma=0.0,
        )

    cpu_result = run(cpu)
    gpu_result = run(gpu)
    jax.block_until_ready((cpu_result, gpu_result))
    for key in ("candidate_count", "overflow", "source_cell", "state"):
        np.testing.assert_array_equal(
            np.asarray(cpu_result[key]), np.asarray(gpu_result[key])
        )
    for key in ("r", "z"):
        np.testing.assert_allclose(
            np.asarray(cpu_result[key]),
            np.asarray(gpu_result[key]),
            atol=1.0e-10,
            rtol=0.0,
            equal_nan=True,
        )


if __name__ == "__main__":
    pytest.main([__file__])
