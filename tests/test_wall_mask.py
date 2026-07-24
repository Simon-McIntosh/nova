"""Tests for the machine-agnostic wall (supercover raster, tiles-as-holes, nodes).

Pins the wall-as-DATA abstraction the connectivity read consumes:

* **single-vessel parity** - one closed vessel unit reproduces the plain
  point-in-polygon limiter mask (byte-identical), so the single-loop path is
  unchanged;
* **tiles-as-holes** - a material unit excises its cells from the occupiable
  region (contact = poke-through), and a thin blade (t < d) still leaves a
  >=1-cell supercover obstacle;
* **diagnostics** - a sub-grid unit fires the thin-unit warning; two disjoint
  units whose rasters fuse fire the gap-merge warning; neither raises;
* **nodes** - every unit is resampled at ~d/2 and tagged by unit.

The g_wall exactness tests live with the free-boundary solver (they need the
equilibrium grid); this file covers the standalone host geometry.
"""

from __future__ import annotations

import numpy as np

from nova.jax import wall_mask as wm
from nova.jax.wall_mask import inside_polygon as _inside_polygon


def _grid(nr=81, nz=101, r=(0.2, 1.8), z=(-1.1, 1.1)):
    rg = np.linspace(*r, nr)
    zg = np.linspace(*z, nz)
    return rg, zg


# --- single-vessel parity ---------------------------------------------------


def test_single_vessel_reproduces_inside_polygon():
    """One closed vessel unit == the plain ray-cast limiter mask."""
    rg, zg = _grid()
    lr = np.array([0.3, 1.7, 1.7, 0.3, 0.3])
    lz = np.array([-1.0, -1.0, 1.0, 1.0, -1.0])
    mesh_r, mesh_z = np.meshgrid(rg, zg)
    ref = _inside_polygon(mesh_r.ravel(), mesh_z.ravel(), lr, lz).reshape(
        zg.size, rg.size
    )
    mask, diags = wm.build_wall_mask(rg, zg, [wm.vessel_unit(lr, lz)])
    assert np.array_equal(mask, ref)
    assert diags == []


# --- tiles as holes ---------------------------------------------------------


def test_material_tile_excises_cells():
    """A material tile is a HOLE: its cells are removed from the occupiable set."""
    rg, zg = _grid()
    lr = np.array([0.3, 1.7, 1.7, 0.3, 0.3])
    lz = np.array([-1.0, -1.0, 1.0, 1.0, -1.0])
    tr = np.array([0.9, 1.1, 1.1, 0.9, 0.9])
    tz = np.array([-0.1, -0.1, 0.1, 0.1, -0.1])
    vessel_only, _ = wm.build_wall_mask(rg, zg, [wm.vessel_unit(lr, lz)])
    with_tile, _ = wm.build_wall_mask(
        rg, zg, [wm.vessel_unit(lr, lz), wm.material_unit(tr, tz)]
    )
    removed = vessel_only & ~with_tile
    assert removed.any()
    assert np.array_equal(removed, vessel_only & removed)
    jc = np.argmin(np.abs(rg - 1.0))
    ic = np.argmin(np.abs(zg - 0.0))
    assert not with_tile[ic, jc]


def test_thin_blade_leaves_at_least_one_cell():
    """A blade thinner than d still leaves a contiguous >=1-cell barrier."""
    rg, zg = _grid()
    dr = float(rg[1] - rg[0])
    r0 = 1.0
    thin = dr / 4.0
    br = np.array([r0 - thin, r0 + thin, r0 + thin, r0 - thin, r0 - thin])
    bz = np.array([-0.3, -0.3, 0.3, 0.3, -0.3])
    raster = wm.supercover_raster(rg, zg, wm.material_unit(br, bz))
    i0 = np.argmin(np.abs(zg - (-0.3)))
    i1 = np.argmin(np.abs(zg - 0.3))
    for i in range(min(i0, i1), max(i0, i1) + 1):
        assert raster[i, :].any(), f"blade left no obstacle in row {i}"


def test_open_line_primitive_marks_crossed_cells():
    """An open polyline (no fill) marks exactly the cells its segments cross."""
    rg, zg = _grid()
    lr = np.array([0.6, 1.4])
    lz = np.array([-0.4, 0.4])
    raster = wm.supercover_raster(rg, zg, wm.material_unit(lr, lz, closed=False))
    assert raster.any()
    assert raster[np.argmin(np.abs(zg + 0.4)), np.argmin(np.abs(rg - 0.6))]
    assert raster[np.argmin(np.abs(zg - 0.4)), np.argmin(np.abs(rg - 1.4))]


# --- diagnostics (warnings, never errors) -----------------------------------


def test_thin_unit_diagnostic_fires():
    """A sub-grid closed tile reports the thin-unit warning with a thickness proxy."""
    rg, zg = _grid()
    dr = float(rg[1] - rg[0])
    lr = np.array([0.3, 1.7, 1.7, 0.3, 0.3])
    lz = np.array([-1.0, -1.0, 1.0, 1.0, -1.0])
    thin = dr / 3.0
    tr = np.array([1.0 - thin, 1.0 + thin, 1.0 + thin, 1.0 - thin, 1.0 - thin])
    tz = np.array([-0.3, -0.3, 0.3, 0.3, -0.3])
    _mask, diags = wm.build_wall_mask(
        rg, zg, [wm.vessel_unit(lr, lz), wm.material_unit(tr, tz, name="blade")]
    )
    thin_diags = [d for d in diags if d.kind == "thin_unit"]
    assert len(thin_diags) == 1
    assert thin_diags[0].detail["thickness_proxy_m"] < dr


def test_fat_tile_no_thin_diagnostic():
    rg, zg = _grid()
    lr = np.array([0.3, 1.7, 1.7, 0.3, 0.3])
    lz = np.array([-1.0, -1.0, 1.0, 1.0, -1.0])
    tr = np.array([0.85, 1.15, 1.15, 0.85, 0.85])
    tz = np.array([-0.2, -0.2, 0.2, 0.2, -0.2])
    _mask, diags = wm.build_wall_mask(
        rg, zg, [wm.vessel_unit(lr, lz), wm.material_unit(tr, tz)]
    )
    assert [d for d in diags if d.kind == "thin_unit"] == []


def test_gap_merge_diagnostic_fires():
    """Two disjoint tiles a sub-cell gap apart report the gap-merge warning."""
    rg, zg = _grid(nr=61, nz=61)
    dr = float(rg[1] - rg[0])
    lr = np.array([0.3, 1.7, 1.7, 0.3, 0.3])
    lz = np.array([-1.0, -1.0, 1.0, 1.0, -1.0])
    gap = 0.6 * dr
    ta = np.array([0.9, 1.0, 1.0, 0.9, 0.9])
    tz = np.array([-0.15, -0.15, 0.15, 0.15, -0.15])
    tb = ta + (0.1 + gap)
    _mask, diags = wm.build_wall_mask(
        rg,
        zg,
        [wm.vessel_unit(lr, lz), wm.material_unit(ta, tz), wm.material_unit(tb, tz)],
    )
    gm = [d for d in diags if d.kind == "gap_merge"]
    assert len(gm) == 1
    assert set(gm[0].units) == {1, 2}


# --- nodes ------------------------------------------------------------------


def test_densify_units_spacing_and_tags():
    rg, zg = _grid()
    delta = min(float(rg[1] - rg[0]), float(zg[1] - zg[0]))
    lr = np.array([0.3, 1.7, 1.7, 0.3, 0.3])
    lz = np.array([-1.0, -1.0, 1.0, 1.0, -1.0])
    tr = np.array([0.9, 1.1, 1.1, 0.9, 0.9])
    tz = np.array([-0.1, -0.1, 0.1, 0.1, -0.1])
    units = [wm.vessel_unit(lr, lz), wm.material_unit(tr, tz)]
    wr, wz, uid = wm.densify_units(units, spacing=0.5 * delta)
    assert wr.shape == wz.shape == uid.shape
    assert set(np.unique(uid)) == {0, 1}
    for k in (0, 1):
        sel = uid == k
        pr, pz = wr[sel], wz[sel]
        d = np.hypot(np.diff(pr), np.diff(pz))
        assert np.max(d) <= 0.75 * delta + 1e-9


def test_no_units_yields_no_wall_sentinel():
    wr, wz, uid = wm.densify_units([], spacing=0.01)
    assert wr[0] > 1e29 and wz[0] > 1e29


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
