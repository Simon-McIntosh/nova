"""Behaviour pins for the x-point-versus-wall boundary selector.

``Topology.boundary`` decides which surface bounds the plasma — the primary
x-point saddle or the wall tangency — with a vertical shadow rule: a wall
contact lying vertically beyond the x-point band sits in the private-flux
shadow of a null and cannot bind; inside the band the confined-most flux wins.
These pins drive the selector, the x-point ionization mask and the LCFS flux
directly with synthetic null data where the correct selection is known, and
close with an end-to-end ``update`` on a raster grid.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from nova.utilities.importmanager import skip_import

with skip_import("jax"):
    import jax.numpy as jnp

    from nova.biot.null import Null1D, Null2D
    from nova.equilibrium.topology import Topology
    from nova.jax.config import configure_dtypes


def _raster_nulls(nx=41, nz=51, r_lim=(0.6, 1.4), z_lim=(-0.6, 0.6)):
    """Structured-grid ``Null2D`` plus a circular-wall ``Null1D``."""
    configure_dtypes()
    xg = np.linspace(*r_lim, nx)
    zg = np.linspace(*z_lim, nz)
    x2d, z2d = np.meshgrid(xg, zg, indexing="ij")
    coordinate = np.column_stack([x2d.ravel(), z2d.ravel()])
    patch = np.array([(0, 0), (-1, 0), (0, -1), (1, -1), (1, 0), (0, 1), (-1, 1)])
    stencil = np.ravel_multi_index(
        np.indices((nx - 2, nz - 2)).reshape(2, -1, 1) + 1 + patch.T[:, np.newaxis],
        (nx, nz),
    )
    grid = Null2D.from_coordinates(coordinate, stencil, maxsize=3)
    theta = np.linspace(0.0, 2.0 * np.pi, 128, endpoint=False)
    wall_coordinate = np.column_stack([1.0 + 0.3 * np.cos(theta), 0.45 * np.sin(theta)])
    wall = Null1D(jnp.asarray(wall_coordinate))
    return Topology(grid, wall), xg, zg, wall_coordinate


@pytest.fixture(scope="module")
def topology():
    return _raster_nulls()[0]


NAN_ROW = [np.nan, np.nan, np.nan]


def _boundary(topology, data_o, vmap_x, data_w, polarity=1.0):
    return np.asarray(
        topology.boundary(
            jnp.asarray(data_o),
            jnp.asarray(vmap_x),
            jnp.asarray(data_w),
            polarity,
        )
    )


def test_wall_binds_when_confined_most_despite_distant_x(topology):
    """A far-below-wall x flux must not steal the binding from the tangency."""
    data_o = [1.0, 0.0, 1.0]
    vmap_x = [[1.0, -0.4, 0.2], NAN_ROW]
    data_w = [1.05, 0.0, 0.6]
    out = _boundary(topology, data_o, vmap_x, data_w)
    np.testing.assert_allclose(out, data_w)


def test_x_point_binds_when_its_flux_is_confined_most(topology):
    data_o = [1.0, 0.0, 1.0]
    vmap_x = [[1.0, -0.4, 0.8], NAN_ROW]
    data_w = [1.05, 0.0, 0.6]
    out = _boundary(topology, data_o, vmap_x, data_w)
    np.testing.assert_allclose(out, vmap_x[0])


def test_wall_contact_in_private_flux_shadow_never_binds(topology):
    """A tangency below the lower null is shadowed even at higher flux."""
    data_o = [1.0, 0.0, 1.0]
    vmap_x = [[1.0, -0.4, 0.8], NAN_ROW]
    data_w = [1.0, -0.45, 0.9]
    out = _boundary(topology, data_o, vmap_x, data_w)
    np.testing.assert_allclose(out, vmap_x[0])


def test_upper_x_point_shadows_only_above_itself(topology):
    """An above-axis null shadows the wall above it, not the midplane."""
    data_o = [1.0, 0.0, 1.0]
    vmap_x = [[1.0, 0.4, 0.2], NAN_ROW]
    data_w_mid = [1.05, 0.0, 0.6]
    out_mid = _boundary(topology, data_o, vmap_x, data_w_mid)
    np.testing.assert_allclose(out_mid, data_w_mid)
    data_w_top = [1.0, 0.45, 0.6]
    out_top = _boundary(topology, data_o, vmap_x, data_w_top)
    np.testing.assert_allclose(out_top, vmap_x[0])


def test_wall_binds_when_no_x_point_exists(topology):
    data_o = [1.0, 0.0, 1.0]
    vmap_x = [NAN_ROW, NAN_ROW]
    data_w = [1.05, 0.0, 0.6]
    out = _boundary(topology, data_o, vmap_x, data_w)
    np.testing.assert_allclose(out, data_w)


def test_negative_polarity_selects_confined_most_by_minimum(topology):
    """With a flux minimum at the axis the smaller flux is confined-most."""
    data_o = [1.0, 0.0, -1.0]
    vmap_x = [[1.0, -0.4, -0.8], NAN_ROW]
    data_w = [1.05, 0.0, -0.6]
    out = _boundary(topology, data_o, vmap_x, data_w, polarity=-1.0)
    np.testing.assert_allclose(out, vmap_x[0])
    vmap_x_far = [[1.0, -0.4, -0.2], NAN_ROW]
    out_far = _boundary(topology, data_o, vmap_x_far, data_w, polarity=-1.0)
    np.testing.assert_allclose(out_far, data_w)


def test_x_mask_excludes_cells_beyond_null_heights(topology):
    """Cells below a lower null / above an upper null leave the plasma mask."""
    data_o = jnp.asarray([1.0, 0.0, 1.0])
    height = np.asarray(topology.grid.coordinate[:, 1])

    lower = jnp.asarray([[1.0, -0.3, 0.8], NAN_ROW])
    mask = np.asarray(topology.x_mask(data_o, lower))
    np.testing.assert_array_equal(mask, height > -0.3)

    both = jnp.asarray([[1.0, -0.3, 0.8], [1.0, 0.25, 0.75]])
    mask = np.asarray(topology.x_mask(data_o, both))
    np.testing.assert_array_equal(mask, (height > -0.3) & (height < 0.25))

    unmasked = np.asarray(topology.x_mask(data_o, jnp.asarray([NAN_ROW, NAN_ROW])))
    assert unmasked.all()


def test_native_compile_probe_preserves_child_exit_statuses(tmp_path):
    script = Path(__file__).parents[1] / "benchmarks" / "topology_boundary_compile.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--warmup-compilations",
            "4",
            "--memory-headroom-mib",
            "8",
            "--output-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    summary = json.loads((tmp_path / "summary.json").read_text())
    statuses = {
        result["condition"]: result["exit_status"] for result in summary["results"]
    }
    assert statuses == {"fresh": 0, "warm-cache": 0, "constrained-memory": 134}


def test_psi_lcfs_is_the_normalized_flux_interpolant(topology):
    psi_axis, psi_boundary = 1.0, 0.4
    lcfs = float(topology.psi_lcfs(psi_axis, psi_boundary))
    assert lcfs == pytest.approx(psi_axis + 0.999 * (psi_boundary - psi_axis))
    half = float(topology.psi_lcfs(psi_axis, psi_boundary, 0.5))
    assert half == pytest.approx(0.7)


def test_update_limited_gaussian_binds_at_the_wall():
    """End-to-end: a single O-point field ionizes the wall-bounded core."""
    topology, xg, zg, wall_coordinate = _raster_nulls()

    def psi_fn(r, z):
        return np.exp(-(((r - 1.0) ** 2 + z**2) / 0.35**2))

    coordinate = np.asarray(topology.grid.coordinate)
    psi_grid = psi_fn(coordinate[:, 0], coordinate[:, 1])
    psi_wall = psi_fn(wall_coordinate[:, 0], wall_coordinate[:, 1])
    psi = jnp.asarray(np.concatenate([psi_grid, psi_wall]))

    vmap_o, vmap_x = topology.grid(psi[: topology.grid.node_number])
    data_o = topology.o_point_data(vmap_o, 1.0)
    data_w = topology.wall(psi[topology.grid.node_number :], 1.0)
    data_b = topology.boundary(data_o, vmap_x, data_w, 1.0)
    # no saddle exists, so the boundary IS the wall tangency
    np.testing.assert_allclose(np.asarray(data_b), np.asarray(data_w))
    # the tangency rides the highest-flux wall node (sub-grid refined)
    assert float(data_b[2]) == pytest.approx(float(psi_wall.max()), rel=1e-3)

    psi_norm, ionize = topology.update(psi, 1.0)
    psi_norm = np.asarray(psi_norm)
    ionize = np.asarray(ionize)
    # normalization anchors: 0 at the axis, 1 at the boundary flux
    axis_index = int(np.argmax(psi_grid))
    assert psi_norm[axis_index] == pytest.approx(0.0, abs=1e-3)
    # every ionized cell lies inside the LCFS in normalized flux
    lcfs = float(topology.psi_lcfs(data_o[2], data_b[2]))
    np.testing.assert_array_equal(ionize, psi_grid >= lcfs)
    assert 0 < ionize.sum() < ionize.size


if __name__ == "__main__":
    pytest.main([__file__])
