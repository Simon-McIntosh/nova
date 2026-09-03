"""Tests for the wall line-of-sight veto on the reachable limiter selection.

A wall node touching the axis-connected pre-saddle flood on the connectivity
raster (:func:`_wall_nodes_touching_region`) can still sit behind a re-entrant
wall feature the raster is too coarse to resolve.  These tests pin the added
exact-geometry veto (:func:`_wall_nodes_in_line_of_sight`): a node only binds
when it BOTH touches the flood on the raster AND has an unobstructed straight
sightline from the axis to the exact wall polyline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nova.utilities.importmanager import skip_import

with skip_import("jax"):
    import jax
    import jax.numpy as jnp

    from nova.equilibrium import connectivity_boundary as cb
    from nova.jax.config import configure_dtypes

from nova.equilibrium.wall_mask import inside_polygon as _inside_polygon

#: a rectangular chamber with a single fin hanging down from the top wall,
#: opening a shadowed pocket beside the fin's tip (index 5) that is only
#: reachable by crossing the fin's vertical segment (index 3-4).
_FIN_WALL_R = np.asarray([0.2, 1.8, 1.8, 1.1, 1.1, 0.9, 0.9, 0.2])
_FIN_WALL_Z = np.asarray([-1.0, -1.0, 1.0, 1.0, 0.0, 0.0, -0.5, -0.5])
_FIN_TARGET_INDEX = 5
_FIN_AXIS = (1.4, 0.05)

_OPERANDS_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs/figures/topology-visual-corroboration/mast-topology-operands.npz"
)


def _fin_raster(nr, nz):
    """Build a raster over the fin wall at the given resolution."""
    rg = np.linspace(0.2, 1.8, nr)
    zg = np.linspace(-1.0, 1.0, nz)
    rr, zz = np.meshgrid(rg, zg)
    inside = _inside_polygon(rr.ravel(), zz.ravel(), _FIN_WALL_R, _FIN_WALL_Z)
    return rg, zg, inside.reshape(zz.shape)


def _fin_psi(rg, zg):
    """A single-O-point bowl centred at the axis, smooth over the whole grid."""
    rr, zz = np.meshgrid(rg, zg)
    axis_r, axis_z = _FIN_AXIS
    psi = -((rr - axis_r) ** 2 + (zz - axis_z) ** 2)
    psi_axis = float(
        psi[np.argmin(np.abs(zg - axis_z)), np.argmin(np.abs(rg - axis_r))]
    )
    edge = np.concatenate([psi[0], psi[-1], psi[:, 0], psi[:, -1]])
    psi_out = edge[np.argmax(np.abs(edge - psi_axis))]
    return psi, psi_axis, psi_out - psi_axis


def test_notch_back_wall_excluded_despite_flooded_nearest_cell():
    """A back-wall node loses even when its nearest raster cell is flooded.

    The pre-saddle region is set to the entire in-material raster (the most
    generous flood a connectivity read could ever produce), so
    ``_wall_nodes_touching_region`` alone reports the shadowed node
    reachable.  Its wall flux is also forced to the most extreme possible
    value (equal to the axis flux).  The combined reachable-and-visible
    selection must still refuse it.
    """
    configure_dtypes()
    rg, zg, inside = _fin_raster(17, 21)
    psi, psi_axis, span = _fin_psi(rg, zg)
    wall_r = jnp.asarray(_FIN_WALL_R)
    wall_z = jnp.asarray(_FIN_WALL_Z)
    axis_r, axis_z = (jnp.asarray(v) for v in _FIN_AXIS)

    region = jnp.asarray(inside)
    raster_only = cb._wall_nodes_touching_region(
        region, jnp.asarray(inside), jnp.asarray(rg), jnp.asarray(zg), wall_r, wall_z
    )
    assert bool(raster_only[_FIN_TARGET_INDEX]), (
        "fixture must reproduce the raster false positive before testing the veto"
    )

    wall_psi = jax.vmap(
        lambda r, z: cb._bilerp(
            jnp.asarray(psi), jnp.asarray(rg), jnp.asarray(zg), r, z
        )
    )(wall_r, wall_z)
    wall_psi = wall_psi.at[_FIN_TARGET_INDEX].set(psi_axis)

    selected = cb._select_reachable_wall_limiter(
        jnp.asarray(psi),
        jnp.asarray(rg),
        jnp.asarray(zg),
        jnp.asarray(inside),
        wall_r,
        wall_z,
        wall_psi,
        region,
        jnp.asarray(psi_axis),
        cb.fit_tensor_spline(jnp.asarray(rg), jnp.asarray(zg), jnp.asarray(psi)),
        axis_r,
        axis_z,
    )
    assert not bool(selected["reachable"][_FIN_TARGET_INDEX])
    assert int(selected["node_index"]) != _FIN_TARGET_INDEX


def test_notch_narrower_than_raster_cell_excluded_by_line_of_sight():
    """A notch opening narrower than the raster pitch still loses.

    The raster column pitch (0.32 m) is wider than the fin's 0.2 m opening,
    so a real flood-fill (not a hand-forced region) genuinely reaches the
    shadowed node's nearest cell -- no raster refinement at this pitch can
    tell the two sides of the fin apart.  Only the exact-geometry sightline
    test excludes it.
    """
    configure_dtypes()
    rg, zg, inside = _fin_raster(6, 11)
    r_pitch = float(rg[1] - rg[0])
    fin_opening = float(_FIN_WALL_R[3] - _FIN_WALL_R[5])
    assert r_pitch > fin_opening, "fixture must be coarser than the notch opening"

    psi, psi_axis, span = _fin_psi(rg, zg)
    u = (psi - psi_axis) / span
    wall_r = jnp.asarray(_FIN_WALL_R)
    wall_z = jnp.asarray(_FIN_WALL_Z)
    axis_r, axis_z = (jnp.asarray(v) for v in _FIN_AXIS)

    region = cb._axis_component_before_level(
        jnp.asarray(u),
        jnp.asarray(inside),
        jnp.asarray(rg),
        jnp.asarray(zg),
        axis_r,
        axis_z,
        jnp.asarray(0.5),
    )
    raster_only = cb._wall_nodes_touching_region(
        region, jnp.asarray(inside), jnp.asarray(rg), jnp.asarray(zg), wall_r, wall_z
    )
    assert bool(raster_only[_FIN_TARGET_INDEX]), (
        "the real flood must reach the shadowed node at this raster pitch"
    )

    line_of_sight = cb._wall_nodes_in_line_of_sight(
        axis_r, axis_z, wall_r, wall_z, wall_r, wall_z
    )
    assert not bool(line_of_sight[_FIN_TARGET_INDEX])
    assert not bool((raster_only & line_of_sight)[_FIN_TARGET_INDEX])


def test_notch_flux_forced_more_extreme_than_x_point_still_loses():
    """A notch flux more extreme than the X-point still never binds.

    The shadowed node's exact wall flux is forced to the axis flux itself --
    the most extreme value any wall candidate could ever take, guaranteed to
    beat any finite X-point level -- through the full classification path
    (:func:`traced_margin_candidate_diagnostics`).  The rule must still
    refuse it and bind a genuinely visible node instead, and the
    ``class_wall_shadowed`` diagnostic must report the excluded raw
    candidate as shadowed.
    """
    configure_dtypes()
    rg, zg, inside = _fin_raster(17, 21)
    psi, psi_axis, span = _fin_psi(rg, zg)
    wall_r = jnp.asarray(_FIN_WALL_R)
    wall_z = jnp.asarray(_FIN_WALL_Z)
    axis_r, axis_z = _FIN_AXIS

    wall_psi = jax.vmap(
        lambda r, z: cb._bilerp(
            jnp.asarray(psi), jnp.asarray(rg), jnp.asarray(zg), r, z
        )
    )(wall_r, wall_z)
    wall_psi = wall_psi.at[_FIN_TARGET_INDEX].set(psi_axis + 1.0e-6)

    x_level = 0.6
    x_flux = psi_axis + x_level * span
    classification_x = jnp.asarray([[1.0, -0.8, x_flux, 0.0]])
    classification_wall = jnp.asarray(
        [
            _FIN_WALL_R[_FIN_TARGET_INDEX],
            _FIN_WALL_Z[_FIN_TARGET_INDEX],
            float(wall_psi[_FIN_TARGET_INDEX]),
        ]
    )

    result = cb.traced_margin_candidate_diagnostics(
        jnp.asarray(psi),
        jnp.asarray(rg),
        jnp.asarray(zg),
        jnp.asarray(inside),
        jnp.asarray(axis_r),
        jnp.asarray(axis_z),
        48,
        12,
        wall_r,
        wall_z,
        wall_psi,
        classification_x,
        classification_wall,
    )

    assert bool(result["wall_shadowed"])
    assert not bool(result["reachable_wall_node_mask"][_FIN_TARGET_INDEX])
    assert int(result["limiter_wall_node_index"]) != _FIN_TARGET_INDEX
    np.testing.assert_array_less(
        np.abs(np.asarray(result["limiter_coordinate"]) - np.array([1.1, 0.0])).max(),
        1.0e-6,
    )


@pytest.mark.skipif(
    not _OPERANDS_PATH.exists(), reason="banked topology operands artifact absent"
)
@pytest.mark.parametrize(
    "row,identity",
    [(6, "21986/46 pure"), (8, "21989/55 pure"), (9, "21989/55 mixed")],
)
def test_bank_row_wall_no_longer_reaches_shadowed_node(row, identity):
    """The three captured bank rows lose their notch node on the real wall.

    ``per_cell_flux_values`` is empty for these rows in the banked artifact
    (the flux field is not yet republished), so the class-margin re-solve
    itself is not reproducible here; that gap is reported as a follow-on.
    But the veto this node adds is a purely geometric test on the exact
    governed wall polyline and the row's own EFIT axis -- neither needs a
    flux field -- so it is checked directly: on the real MAST wall these
    three rows share, wall-mesh node 10 (0.5649, 1.7281), the node the
    triage trace names as the captured limiter point, no longer has an
    unobstructed sightline from the row's own magnetic axis.
    """
    configure_dtypes()
    operands = np.load(_OPERANDS_PATH, allow_pickle=True)
    wall = operands[f"row_{row:02d}_wall"]
    axis = operands[f"row_{row:02d}_efit_axis"][0]
    wall_r = jnp.asarray(wall[:, 0])
    wall_z = jnp.asarray(wall[:, 1])

    node_index = 10
    np.testing.assert_allclose(
        [float(wall_r[node_index]), float(wall_z[node_index])],
        [0.56493068, 1.72808158],
        atol=1.0e-6,
    )

    line_of_sight = cb._wall_nodes_in_line_of_sight(
        jnp.asarray(float(axis[0])),
        jnp.asarray(float(axis[1])),
        wall_r,
        wall_z,
        wall_r,
        wall_z,
    )
    assert not bool(line_of_sight[node_index]), identity
