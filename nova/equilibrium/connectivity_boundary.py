"""Accelerator-native connectivity boundary read (JAX).

The last-closed-flux-surface resolved by CONNECTIVITY — the outermost closed
axis-enclosing flux contour that still lies inside the wall — computed with
fixed-shape, ``jit`` / ``vmap`` / ``grad``-safe device primitives.  This is the
device-native reimplementation of the host monotone flux-offset LCFS push
(the shipped monotone flux-offset push): SAME algorithm, no contourpy, no
``scipy.ndimage.label``, no ``argwhere`` / bisection-over-traced-contours.  ONE
code path handles limited and diverted plasmas alike, continuous through the
marginal limited↔diverted transition by construction (the boundary is never a
classify-first decision about *which* surface bounds).

Method (the connectivity read, re-expressed on device):

* **Normalised flux.**  ``u = (ψ − ψ_axis)/(ψ_out − ψ_axis)`` maps the axis to
  0 and the domain-edge extreme (the edge cell whose flux is furthest from the
  axis) to 1, so the confined side at a candidate level ``s`` is simply
  ``u ≤ s`` — sign-agnostic (MAST ψ_axis > ψ_bnd or the reverse).

* **Axis-connected region.**  At level ``s`` the confined-and-in-wall set
  ``(u ≤ s) ∧ inside_wall`` is propagated over the half-offset hexagonal
  cell graph.  Six-neighbour links remain open in the bulk only when their
  shared physical edge contains flux strictly on the axis side of the binding
  level, so the component cannot bridge a private lobe through a saddle cell.

* **The binding level.**  A level is *valid*
  while the axis region stays clear of the wall; each escape transition is the
  largest valid level (monotone → coarse sweep + bisection).  ψ_bnd is read at
  the MEAN of an inner escape test (region touches the innermost in-wall shell,
  ~one cell inside) and an outer test (region dilation reaches a still-confined
  out-of-wall/edge cell, ~one cell outside) — unbiased for an interior SADDLE
  binding, so a DIVERTED separatrix (ψ_N = 1) is reproduced by the mean.  A
  LIMITED wall tangency is refined SUB-GRID: the minimum interpolated ψ_N over
  the wall boundary points adjacent to the axis region (the cell mean carries the
  sub-cell wall-position error a tangency is sensitive to; the interpolated wall
  crossing removes it).  The wall and saddle candidates are reduced by their
  normalised flux ordering, so whichever obstruction is reached first closes
  the boundary.  The divertor legs are open branches the closed axis-region
  never floods, so the lobe — and the radii read off it — are unaffected.  The
  Boolean class and its signed margin use that same pair unless an exact
  comparator candidate table and wall extremum are supplied.  In that case only
  the class operands are replaced; the connectivity boundary remains unchanged.
  Typed saddles outside the wall polygon are masked before selection.  The wall
  operand is then selected from the wall touched by the axis component
  immediately inside that saddle, so private-region wall extrema cannot compete.

* **LCFS radii.**  Read at ψ_lcfs = ψ_axis + lcfs_norm·(ψ_bnd − ψ_axis) by a
  fixed outward ray-march from the axis at the evaluator's 8 poloidal angles —
  a differentiable interpolated crossing, the same fixed parameterisation
  the host fixed-angle ray read uses on the host.

Everything is a fixed-shape reduction over the full grid: no data-dependent
shapes, no host round-trip, no contour extraction — so a batch of slices sharing
one campaign grid is a single ``jax.vmap``.  The only machine input is the wall
as a raster boolean mask (``inside_limiter``), so a single loop (MAST), a union
of discrete limiters (AUG), or a per-pulse movable wall (WEST) is data, not a
separate code path.

Sub-grid note: the two-sided mean removes the ~one-cell systematic bias in the
scalar ψ_bnd, leaving an unbiased sub-cell residual; the LCFS radii are sub-grid
(interpolated ray crossing).  The sub-grid saddle/axis POSITION (as opposed to
the binding flux) is the nova stencil refinement — a separate rung, not folded
into the boundary sweep.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp

from nova.equilibrium.flux_surface_connectivity import (
    _dilate4,
    flood_fill_core,
    hex_edge_admissibility,
    label_saddle_aware_hex_connected_components,
)
from nova.equilibrium.stencil_nulls import (
    magnetic_axis_subgrid,
    xpoint_candidates,
)
from nova.equilibrium.labels import LCFS_ANGLES, N_XPOINT_SLOTS
from nova.geometry.select import (
    length_2d,
    traced_quadratic_wall,
    wall_coordinate,
    wall_length,
)
from nova.geometry.hexstencil import hex_stencil
from nova.jax.config import Precision, resolve_precision
from nova.linalg.tensor_spline import fit_tensor_spline


def _boundary_defaults(psi2d, rg, zg, angles, wall_r, wall_z, wall_psi):
    """Materialise optional values in geometry or solved-field dtype."""
    if angles is None:
        angles = jnp.asarray(LCFS_ANGLES, dtype=rg.dtype)
    if wall_r is None:
        wall_r = jnp.asarray([1.0e30], dtype=rg.dtype)
    if wall_z is None:
        wall_z = jnp.asarray([1.0e30], dtype=zg.dtype)
    if wall_psi is None:
        wall_psi = jnp.asarray([jnp.nan], dtype=psi2d.dtype)
    return angles, wall_r, wall_z, wall_psi


def _arg_extreme(values, *, maximize):
    """Return the first extreme index with a dtype-exact reduction seed."""
    values = jax.lax.stop_gradient(values)
    indices = jax.lax.broadcasted_iota(jnp.int32, values.shape, 0)
    initial_value = -jnp.inf if maximize else jnp.inf
    initial = (
        jnp.asarray(initial_value, dtype=values.dtype),
        jnp.asarray(values.size, dtype=jnp.int32),
    )

    def choose(left, right):
        left_value, left_index = left
        right_value, right_index = right
        better = right_value > left_value if maximize else right_value < left_value
        take_right = better | ((right_value == left_value) & (right_index < left_index))
        return (
            jnp.where(take_right, right_value, left_value),
            jnp.where(take_right, right_index, left_index),
        )

    return jax.lax.reduce((values, indices), initial, choose, dimensions=(0,))[1]


def _argmax_exact(values):
    """Return the first maximum index without a default-dtype seed."""
    return _arg_extreme(values, maximize=True)


def _argmin_exact(values):
    """Return the first minimum index without a default-dtype seed."""
    return _arg_extreme(values, maximize=False)


def _points_inside_polygon(point_r, point_z, polygon_r, polygon_z):
    """Test fixed-shape points against a polygon, including its boundary."""
    point_r = jnp.asarray(point_r)
    point_z = jnp.asarray(point_z)
    polygon_r = jnp.asarray(polygon_r, dtype=point_r.dtype)
    polygon_z = jnp.asarray(polygon_z, dtype=point_z.dtype)
    previous_r = jnp.roll(polygon_r, 1)
    previous_z = jnp.roll(polygon_z, 1)

    query_r = point_r[..., None]
    query_z = point_z[..., None]
    straddles = (polygon_z > query_z) != (previous_z > query_z)
    edge_height = previous_z - polygon_z
    safe_height = jnp.where(edge_height == 0.0, 1.0, edge_height)
    crossing_r = (previous_r - polygon_r) * (
        query_z - polygon_z
    ) / safe_height + polygon_r
    inside = jnp.sum(straddles & (query_r < crossing_r), axis=-1) % 2 == 1

    edge_r = previous_r - polygon_r
    edge_z = previous_z - polygon_z
    edge_length2 = edge_r**2 + edge_z**2
    safe_length2 = jnp.where(edge_length2 == 0.0, 1.0, edge_length2)
    projection = jnp.clip(
        ((query_r - polygon_r) * edge_r + (query_z - polygon_z) * edge_z)
        / safe_length2,
        0.0,
        1.0,
    )
    nearest_r = polygon_r + projection * edge_r
    nearest_z = polygon_z + projection * edge_z
    coordinate_scale = jnp.maximum(
        1.0,
        jnp.maximum(jnp.max(jnp.abs(polygon_r)), jnp.max(jnp.abs(polygon_z))),
    )
    tolerance = jnp.maximum(
        jnp.asarray(1.0e-12, dtype=point_r.dtype),
        16.0 * jnp.finfo(point_r.dtype).eps * coordinate_scale,
    )
    on_boundary = jnp.any(
        ((query_r - nearest_r) ** 2 + (query_z - nearest_z) ** 2) <= tolerance**2,
        axis=-1,
    )
    return inside | on_boundary


#: static count of X-point candidate slots the stencil classifier fills (a
#: double-null plus spares); the emergent set is then trimmed to N_XPOINT_SLOTS.
_K_XCAND = 8

#: half-width (in ψ_N span units) of the flux band around the flood binding level
#: within which an X-point saddle is accepted as the binding candidate.
_X_FLUX_BAND = 0.05

#: ψ_N tolerance for an X-point to be reported as sitting ON the boundary ring.
_X_ON_RING_U = 0.02

# Keep the connectivity partition strictly on the axis side of a selected
# saddle.  Scaling by the in-vessel flux range makes this a geometric
# separation from the crossing rather than a machine-gauge tolerance.
_PRE_SADDLE_OFFSET_FRACTION = 2.0e-4

# Fixed arc samples expose reachable sub-segment wall runs even when a machine's
# exact GEMM wall nodes are sparse, without data-dependent compaction.
_WALL_REACHABILITY_SAMPLES = 420

# Coarse-grid offsets probed around the preceding binding level.  The asymmetric
# upper reach covers both sides of the two-sided flood mean while retaining the
# exact grid points used by the cold sweep.
_WARM_BRACKET_OFFSETS = (-2, -1, 0, 1, 2, 3)

__all__ = [
    "ConnectivityBoundary",
    "traced_boundary_read",
    "traced_emit_boundary_read",
    "traced_iteration_boundary_read",
    "host_boundary_read",
    "host_boundary_read_batch",
    "traced_margin_candidate_diagnostics",
    "traced_smooth_boundary_read",
    "host_boundary_read_smooth",
]


# ---------------------------------------------------------------------------
# device primitives
# ---------------------------------------------------------------------------


def _bilerp(field: jnp.ndarray, rg: jnp.ndarray, zg: jnp.ndarray, r, z):
    """Bilinear-interpolate a ``(nz, nr)`` field at physical ``(r, z)`` (device)."""
    nr = rg.shape[0]
    nz = zg.shape[0]
    fr = jnp.clip(jnp.interp(r, rg, jnp.arange(nr, dtype=rg.dtype)), 0.0, nr - 1 - 1e-9)
    fz = jnp.clip(jnp.interp(z, zg, jnp.arange(nz, dtype=zg.dtype)), 0.0, nz - 1 - 1e-9)
    j0 = jnp.floor(fr).astype(jnp.int32)
    i0 = jnp.floor(fz).astype(jnp.int32)
    dj = fr - j0
    di = fz - i0
    f00 = field[i0, j0]
    f01 = field[i0, j0 + 1]
    f10 = field[i0 + 1, j0]
    f11 = field[i0 + 1, j0 + 1]
    return (
        f00 * (1 - di) * (1 - dj)
        + f01 * (1 - di) * dj
        + f10 * di * (1 - dj)
        + f11 * di * dj
    )


def _ray_radii(psi2d, rg, zg, ar, az, psi_axis, psi_lcfs, angles, n_ray):
    """LCFS radius at each poloidal angle by an outward ray-march from the axis.

    Marches ``n_ray`` fixed samples out to the grid diagonal, bilinear-interps ψ,
    and returns the first interpolated crossing of ψ_lcfs on each ray (NaN if a
    ray leaves the grid without crossing).  Fixed-shape, differentiable.
    """
    rmax = jnp.hypot(rg[-1] - rg[0], zg[-1] - zg[0])
    ss = jnp.linspace(0.0, rmax, n_ray)
    sign = jnp.sign(psi_lcfs - psi_axis)
    sign = jnp.where(sign == 0.0, 1.0, sign)
    target = (psi_lcfs - psi_axis) * sign
    idx = jnp.arange(n_ray)

    def one_angle(th):
        cr = jnp.cos(th)
        sr = jnp.sin(th)
        r = ar + ss * cr
        z = az + ss * sr
        vals = jax.vmap(lambda rr, zz: _bilerp(psi2d, rg, zg, rr, zz))(r, z)
        in_grid = (r >= rg[0]) & (r <= rg[-1]) & (z >= zg[0]) & (z <= zg[-1])
        # (ψ − ψ_axis)·sign grows from ~0 at the axis toward `target` at ψ_lcfs;
        # off-grid samples get −inf so they never register a crossing.
        f = jnp.where(in_grid, (vals - psi_axis) * sign, -jnp.inf)
        prev_f = jnp.concatenate([f[:1], f[:-1]])
        prev_s = jnp.concatenate([ss[:1], ss[:-1]])
        cross = (prev_f <= target) & (f >= target) & (idx > 0)
        has = jnp.any(cross)
        k = jnp.argmax(cross)  # first True
        fm, fp = f[k], prev_f[k]
        sm, sp = ss[k], prev_s[k]
        frac = jnp.where(fm == fp, 0.0, (target - fp) / (fm - fp))
        radius = sp + frac * (sm - sp)
        return jnp.where(has, radius, jnp.nan)

    return jax.vmap(one_angle)(angles)


def _linear_flood_fill_core(confined, seed, n_iter):
    """Reference one-cell-per-pass flood used to time the unaccelerated read."""

    def body(_index, core):
        return _dilate4(core > 0.5).astype(core.dtype) * confined

    start = seed.astype(jnp.float32) * confined
    return jax.lax.fori_loop(0, n_iter, body, start)


def _flood_fill(confined, seed, n_iter, use_doubling):
    """Select the production doubling fill or its exact linear reference."""
    if use_doubling:
        return flood_fill_core(confined, seed, n_iter)
    return _linear_flood_fill_core(confined, seed, n_iter)


def _raster_hex_partition_geometry(rg, zg):
    """Return centre-first rings and physical shared edges for a flux raster.

    The tensor field remains the flux authority.  Its static array indices are
    interpreted as the half-offset tiling used by the plasma component graph;
    each graph link receives a centred segment of its physical perpendicular
    bisector.  The segment stays inside the local cell pitch, which is enough
    to decide whether any finite portion of the shared edge lies on the axis
    side without requiring another mesh input.
    """
    shape = (zg.shape[0], rg.shape[0])
    rings = jnp.asarray(hex_stencil(shape), dtype=jnp.int32)
    radius, height = jnp.meshgrid(rg, zg)
    centres = jnp.stack((radius, height), axis=-1).reshape(-1, 2)
    centre = centres[rings[:, :1]]
    neighbour = centres[rings]
    midpoint = 0.5 * (centre + neighbour)
    separation = neighbour - centre
    norm = jnp.linalg.norm(separation, axis=-1, keepdims=True)
    norm = norm.at[:, 0].set(1.0)
    tangent = jnp.stack((-separation[..., 1], separation[..., 0]), axis=-1) / norm
    radial_pitch = jnp.abs(rg[1] - rg[0])
    vertical_pitch = jnp.abs(zg[1] - zg[0])
    half_edge = 0.45 * jnp.minimum(radial_pitch, vertical_pitch)
    endpoints = jnp.stack(
        (midpoint - half_edge * tangent, midpoint + half_edge * tangent), axis=-2
    )
    return rings, endpoints.at[:, 0].set(centre[:, 0, None, :])


def _axis_component_before_level(
    u,
    inside_limiter,
    rg,
    zg,
    axis_r,
    axis_z,
    level,
    n_iter,
    use_doubling,
):
    """Return the axis component immediately inside a flux obstruction."""
    inside_values = jnp.where(inside_limiter, u, jnp.nan)
    flux_range = jnp.nanmax(inside_values) - jnp.nanmin(inside_values)
    inward_offset = _PRE_SADDLE_OFFSET_FRACTION * flux_range
    confined = inside_limiter & (u <= level - inward_offset)

    distance2 = (rg[None, :] - axis_r) ** 2 + (zg[:, None] - axis_z) ** 2
    seed_index = _argmin_exact(
        jnp.where(confined.reshape(-1), distance2.reshape(-1), jnp.inf)
    )
    seed = (
        jnp.zeros_like(confined)
        .reshape(-1)
        .at[seed_index]
        .set(True)
        .reshape(confined.shape)
    )
    has_seed = jnp.any(confined)
    seed &= has_seed
    rings, shared_edges = _raster_hex_partition_geometry(rg, zg)
    link_admissible = hex_edge_admissibility(
        u,
        rg,
        zg,
        level - inward_offset,
        jnp.asarray(0.0, dtype=u.dtype),
        shared_edges,
    )
    labels = label_saddle_aware_hex_connected_components(
        confined,
        rings,
        link_admissible,
        n_iter,
    )
    sentinel = jnp.asarray(jnp.iinfo(labels.dtype).max, dtype=labels.dtype)
    axis_label = jnp.min(jnp.where(seed & (labels > 0), labels, sentinel))
    return has_seed & (labels == axis_label)


def _wall_nodes_touching_region(region, inside_limiter, rg, zg, wall_r, wall_z):
    """Mark wall nodes whose nearest in-material raster node is in ``region``."""
    distance2 = (wall_r[:, None, None] - rg[None, None, :]) ** 2 + (
        wall_z[:, None, None] - zg[None, :, None]
    ) ** 2
    nearest = jnp.argmin(
        jnp.where(inside_limiter[None, :, :], distance2, jnp.inf).reshape(
            wall_r.shape[0], -1
        ),
        axis=1,
    )
    return region.reshape(-1)[nearest]


def _sample_wall_polyline(wall_r, wall_z, sample_count):
    """Return fixed-count, equal-arc samples around a closed wall polyline."""
    segment_length = jnp.hypot(
        jnp.roll(wall_r, -1) - wall_r,
        jnp.roll(wall_z, -1) - wall_z,
    )
    segment_end = jnp.cumsum(segment_length)
    segment_start = segment_end - segment_length
    perimeter = segment_end[-1]
    arc = jnp.linspace(0.0, perimeter, sample_count, endpoint=False, dtype=wall_r.dtype)
    segment = jnp.clip(
        jnp.searchsorted(segment_end, arc, side="right"), 0, wall_r.size - 1
    )
    safe_length = jnp.where(segment_length[segment] > 0.0, segment_length[segment], 1.0)
    fraction = (arc - segment_start[segment]) / safe_length
    following = jnp.mod(segment + 1, wall_r.size)
    sample_r = wall_r[segment] + fraction * (wall_r[following] - wall_r[segment])
    sample_z = wall_z[segment] + fraction * (wall_z[following] - wall_z[segment])
    return arc, sample_r, sample_z


def _refine_wall_minimum_position(wall_r, wall_z, wall_psi, reachable, psi_axis):
    """Locate the reachable wall minimum with a cyclic three-node quadratic.

    The selected node and its two ring neighbours define the local
    least-squares quadratic, mirroring the fixed-ring stationary refinement
    used for grid nulls.  The stationary position is clipped to reachable
    adjacent segments, preserving a fixed-shape masked reduction.  This fit
    determines position only; it is not a field interpolant.
    """
    node_count = wall_r.shape[0]
    closed = (wall_r[0] == wall_r[-1]) & (wall_z[0] == wall_z[-1])
    unique_count = node_count - closed.astype(jnp.int32)
    node_indices = jnp.arange(node_count, dtype=jnp.int32)
    eligible = reachable & (node_indices < unique_count) & jnp.isfinite(wall_psi)
    distance = jnp.abs(wall_psi - psi_axis)
    selected = _argmin_exact(jnp.where(eligible, distance, jnp.inf))
    previous = jnp.mod(selected - 1, unique_count)
    following = jnp.mod(selected + 1, unique_count)

    left_length = jnp.hypot(
        wall_r[selected] - wall_r[previous], wall_z[selected] - wall_z[previous]
    )
    right_length = jnp.hypot(
        wall_r[following] - wall_r[selected], wall_z[following] - wall_z[selected]
    )

    cluster_r = jnp.stack((wall_r[previous], wall_r[selected], wall_r[following]))
    cluster_z = jnp.stack((wall_z[previous], wall_z[selected], wall_z[following]))
    cluster_distance = jnp.stack(
        (distance[previous], distance[selected], distance[following])
    )
    cluster_arc = length_2d(cluster_r, cluster_z, array_namespace=jnp)
    coefficients = traced_quadratic_wall(cluster_arc, cluster_distance)
    stationary_arc = wall_length(coefficients, array_namespace=jnp)
    lower_arc = jnp.where(eligible[previous], 0.0, left_length)
    upper_arc = jnp.where(eligible[following], left_length + right_length, left_length)
    stationary_arc = jnp.clip(stationary_arc, lower_arc, upper_arc)
    stationary_shift = stationary_arc - left_length
    refined_distance = (
        coefficients[0] * stationary_arc**2
        + coefficients[1] * stationary_arc
        + coefficients[2]
    )
    node_value = distance[selected]
    refinement_valid = (
        jnp.any(eligible)
        & eligible[previous]
        & eligible[following]
        & (coefficients[0] > 0.0)
        & (left_length > 0.0)
        & (right_length > 0.0)
        & (refined_distance <= node_value)
    )
    shift = jnp.where(refinement_valid, stationary_shift, 0.0)

    refined_cluster_arc = left_length + shift
    refined_r, refined_z = wall_coordinate(
        refined_cluster_arc,
        cluster_r,
        cluster_z,
        cluster_arc,
        array_namespace=jnp,
    )

    segment_lengths = jnp.hypot(
        jnp.roll(wall_r, -1) - wall_r,
        jnp.roll(wall_z, -1) - wall_z,
    )
    arc_nodes = jnp.concatenate(
        [jnp.zeros((1,), dtype=wall_r.dtype), jnp.cumsum(segment_lengths[:-1])]
    )
    perimeter = jnp.sum(segment_lengths)
    refined_arc = jnp.mod(arc_nodes[selected] + shift, perimeter)
    has_reachable = jnp.any(eligible)
    nan = jnp.asarray(jnp.nan, dtype=wall_r.dtype)
    return {
        "reachable": eligible,
        "node_index": selected,
        "node_arc": jnp.where(has_reachable, arc_nodes[selected], nan),
        "node_r": jnp.where(has_reachable, wall_r[selected], nan),
        "node_z": jnp.where(has_reachable, wall_z[selected], nan),
        "node_psi": jnp.where(has_reachable, wall_psi[selected], nan),
        "arc": jnp.where(has_reachable, refined_arc, nan),
        "r": jnp.where(has_reachable, refined_r, nan),
        "z": jnp.where(has_reachable, refined_z, nan),
        "shift": jnp.where(has_reachable, shift, nan),
        "valid": has_reachable,
    }


def _reachable_wall_limiter_point(
    psi2d,
    rg,
    zg,
    wall_r,
    wall_z,
    wall_psi,
    reachable,
    psi_axis,
    *,
    exact_nodes,
    global_surface=None,
):
    """Return a position-refined limiter point with globally sourced flux.

    Exact campaign wall flux selects the nearest node and locates the local
    along-wall stationary position.  When that position moves between nodes,
    its field value comes from the global C2 tensor spline fitted to the whole
    grid, never from the local position fit or a per-cell reconstruction.
    """
    point = _refine_wall_minimum_position(wall_r, wall_z, wall_psi, reachable, psi_axis)
    if global_surface is None:
        global_surface = fit_tensor_spline(rg, zg, psi2d)
    global_surface_flux = global_surface(point["r"], point["z"])
    between_nodes = point["valid"] & ((point["shift"] != 0.0) | (not exact_nodes))
    refined_flux = jnp.where(between_nodes, global_surface_flux, point["node_psi"])
    point = dict(point)
    point.update(
        {
            "psi": jnp.where(point["valid"], refined_flux, jnp.nan),
            "distance": jnp.where(
                point["valid"], jnp.abs(refined_flux - psi_axis), jnp.inf
            ),
            "flux_from_global_surface": between_nodes,
        }
    )
    return point


def _select_reachable_wall_limiter(
    psi2d,
    rg,
    zg,
    inside_limiter,
    wall_r,
    wall_z,
    wall_psi,
    pre_saddle_region,
    psi_axis,
    global_surface,
):
    """Select a limiter point over exact wall nodes and reachable sub-segments."""
    node_reachable = _wall_nodes_touching_region(
        pre_saddle_region, inside_limiter, rg, zg, wall_r, wall_z
    )
    node_point = _reachable_wall_limiter_point(
        psi2d,
        rg,
        zg,
        wall_r,
        wall_z,
        wall_psi,
        node_reachable,
        psi_axis,
        exact_nodes=True,
        global_surface=global_surface,
    )

    sample_arc, sample_r, sample_z = _sample_wall_polyline(
        wall_r, wall_z, _WALL_REACHABILITY_SAMPLES
    )
    sample_reachable = _wall_nodes_touching_region(
        pre_saddle_region, inside_limiter, rg, zg, sample_r, sample_z
    )
    sample_psi = global_surface(sample_r, sample_z)
    sample_point = _reachable_wall_limiter_point(
        psi2d,
        rg,
        zg,
        sample_r,
        sample_z,
        sample_psi,
        sample_reachable,
        psi_axis,
        exact_nodes=False,
        global_surface=global_surface,
    )
    # Exact GEMM nodes are the stronger field authority and win whenever the
    # reachable partition contains one.  Global-surface samples fill only a
    # reachable sub-segment run whose sparse exact endpoints are both absent.
    use_node = node_point["valid"]
    selected = {
        key: jnp.where(use_node, node_point[key], sample_point[key])
        for key in (
            "node_index",
            "node_arc",
            "node_r",
            "node_z",
            "node_psi",
            "arc",
            "r",
            "z",
            "psi",
            "distance",
            "shift",
            "valid",
            "flux_from_global_surface",
        )
    }
    selected.update(
        {
            "reachable": node_reachable,
            "reachable_samples": sample_reachable,
            "sample_arc": sample_arc,
            "selected_from_exact_nodes": use_node,
        }
    )
    return selected


# ---------------------------------------------------------------------------
# the connectivity boundary read (device kernel)
# ---------------------------------------------------------------------------


def _read_ingredients(
    psi2d,
    rg,
    zg,
    inside_limiter,
    axis_r,
    axis_z,
    n_levels,
    n_bisect,
    wall_r,
    wall_z,
    wall_psi,
    previous_flood_level,
    use_doubling,
    classification_x=None,
    classification_wall=None,
) -> dict:
    """Everything the binding needs, up to (but not including) the min/softmin.

    Shared by the HARD read (:func:`traced_boundary_read` — the reference, the
    binding is the exact ``min``) and the SMOOTH read
    (:func:`traced_smooth_boundary_read` — the differentiable path, the binding is
    a temperature-controlled softmin and the core mask a sigmoid).  Returns the
    normalised flux ``u``, the flood binding ``s_flood``, the two sub-grid
    binding candidates ``u_wall_c`` / ``u_x_c`` (``inf`` when absent), the class
    operands, and the X-candidate diagnostics. ``classification_x`` carries an
    exact candidate table and ``classification_wall`` its selected wall
    extremum; they must be supplied together and affect only the class read.
    The typed table stays fixed-shape while candidates outside the polygon
    described by ``wall_r`` / ``wall_z`` are made ineligible.
    """
    if (classification_x is None) != (classification_wall is None):
        raise ValueError("classification candidates and wall must be supplied together")

    nz = zg.shape[0]
    nr = rg.shape[0]
    n_iter = nr + nz  # flood-fill saturation count (≥ the region grid diameter)

    psi_axis = _bilerp(psi2d, rg, zg, axis_r, axis_z)
    edge = jnp.concatenate([psi2d[0, :], psi2d[-1, :], psi2d[:, 0], psi2d[:, -1]])
    psi_out = edge[_argmax_exact(jnp.abs(edge - psi_axis))]
    span = psi_out - psi_axis
    span_safe = jnp.where(jnp.abs(span) < 1e-30, 1e-30, span)
    u = (psi2d - psi_axis) / span_safe  # 0 at axis, 1 at the edge extreme

    ja = _argmin_exact(jnp.abs(rg - axis_r))
    ia = _argmin_exact(jnp.abs(zg - axis_z))
    seed = jnp.zeros((nz, nr), dtype=bool).at[ia, ja].set(True)
    seed_flat = ia * nr + ja

    # wall ring = in-wall cells adjacent to an out-of-wall cell (grid border
    # counts as out-of-wall).  The region "reaches the wall" when it touches this.
    border = (
        jnp.zeros((nz, nr), dtype=bool)
        .at[0, :]
        .set(True)
        .at[-1, :]
        .set(True)
        .at[:, 0]
        .set(True)
        .at[:, -1]
        .set(True)
    )
    outside = (~inside_limiter) | border
    wall_ring = _dilate4(outside) & inside_limiter

    # --- cell-level connectivity binding (the flood) --------------------------
    # A level is valid while the axis region stays clear of the wall; it turns
    # invalid the instant the region first reaches the wall.  Two escape tests
    # bracket the transition from opposite sides — an INNER test (region touches
    # the innermost in-wall shell, ~one cell inside) and an OUTER test (dilating
    # the region reaches a still-confined out-of-wall/edge cell, ~one cell outside).
    # Their mean is unbiased for an interior SADDLE binding (a diverted separatrix,
    # ψ_N=1), where the brackets straddle the saddle symmetrically; a LIMITED wall
    # tangency is then refined sub-grid below (the brackets straddle the wall, whose
    # sub-cell position is not the mean's implicit half-cell).
    def _alive_region(s):
        region = _flood_fill((u <= s) & inside_limiter, seed, n_iter, use_doubling)
        alive = region.reshape(-1)[seed_flat] > 0.5
        return region, alive

    def validity(s):
        region, alive = _alive_region(s)
        inner_touch = jnp.sum(region * wall_ring.astype(region.dtype)) > 0.5
        reach = _dilate4(region > 0.5) & outside & (u <= s)
        outer_touch = jnp.sum(reach.astype(region.dtype)) > 0.5
        return alive & (~inner_touch), alive & (~outer_touch)

    def valid_inner(s):
        return validity(s)[0]

    def valid_outer(s):
        return validity(s)[1]

    s_grid = jnp.linspace(0.0, 1.0, n_levels + 1)[1:]  # (n_levels,) in (0, 1]
    idxs = jnp.arange(n_levels, dtype=jnp.int32)

    def _bracket_from_grid(vk):
        last = jnp.max(jnp.where(vk, idxs, -1))
        lo0 = jnp.where(last >= 0, s_grid[jnp.clip(last, 0, n_levels - 1)], 0.0)
        hi0 = jnp.where(
            last < n_levels - 1, s_grid[jnp.clip(last + 1, 0, n_levels - 1)], 1.0
        )
        return last, lo0, hi0

    def _cold_brackets(_):
        valid_inner_grid, valid_outer_grid = jax.vmap(validity)(s_grid)
        inner = _bracket_from_grid(valid_inner_grid)
        outer = _bracket_from_grid(valid_outer_grid)
        return (
            *inner,
            *outer,
            jnp.asarray(False),
            jnp.asarray(n_levels, dtype=jnp.int32),
        )

    def _warm_brackets(previous_level):
        offsets = jnp.asarray(_WARM_BRACKET_OFFSETS, dtype=jnp.int32)
        centre = jnp.floor(previous_level * n_levels).astype(jnp.int32)
        level_numbers = centre + offsets
        active = (level_numbers >= 1) & (level_numbers <= n_levels)
        safe_numbers = jnp.clip(level_numbers, 1, n_levels)
        levels = s_grid[safe_numbers - 1]
        valid_inner_local, valid_outer_local = jax.vmap(validity)(levels)

        def local_bracket(valid_local):
            valid_local = valid_local & active
            last_number = jnp.max(jnp.where(valid_local, level_numbers, 0))
            last = last_number - 1
            lo0 = jnp.where(
                last_number > 0,
                s_grid[jnp.clip(last, 0, n_levels - 1)],
                0.0,
            )
            hi_number = jnp.where(last_number > 0, last_number + 1, 1)
            hi0 = jnp.where(
                last_number >= n_levels,
                1.0,
                s_grid[jnp.clip(hi_number - 1, 0, n_levels - 1)],
            )
            lower_known = (last_number > 0) | jnp.any(active & (level_numbers == 1))
            upper_known = (last_number >= n_levels) | jnp.any(
                active & (level_numbers == hi_number) & (~valid_local)
            )
            return last, lo0, hi0, lower_known & upper_known

        inner = local_bracket(valid_inner_local)
        outer = local_bracket(valid_outer_local)
        warm_hit = inner[3] & outer[3]

        def use_warm(_):
            return (
                inner[0],
                inner[1],
                inner[2],
                outer[0],
                outer[1],
                outer[2],
                jnp.asarray(True),
                jnp.asarray(len(_WARM_BRACKET_OFFSETS), dtype=jnp.int32),
            )

        def use_cold(_):
            cold = _cold_brackets(None)
            extra = jnp.asarray(len(_WARM_BRACKET_OFFSETS), dtype=jnp.int32)
            return (*cold[:-1], cold[-1] + extra)

        return jax.lax.cond(warm_hit, use_warm, use_cold, operand=None)

    def _refine(valid_fn, lo0, hi0):
        def body(_i, carry):
            lo, hi = carry
            mid = 0.5 * (lo + hi)
            v = valid_fn(mid)
            return (jnp.where(v, mid, lo), jnp.where(v, hi, mid))

        lo, _hi = jax.lax.fori_loop(0, n_bisect, body, (lo0, hi0))
        return lo

    (
        last_in,
        lo_in,
        hi_in,
        _last_out,
        lo_out,
        hi_out,
        binding_search_warm,
        binding_search_evaluations,
    ) = jax.lax.cond(
        jnp.isfinite(previous_flood_level),
        _warm_brackets,
        _cold_brackets,
        operand=previous_flood_level,
    )
    found = last_in >= 0
    s_in = _refine(valid_inner, lo_in, hi_in)
    s_out = _refine(valid_outer, lo_out, hi_out)
    s_flood = 0.5 * (s_in + s_out)  # unbiased for an interior saddle (diverted)

    # The connectivity flood localises the binding to s_flood; the two remaining
    # sub-grid refinements (the wall tangency and the X-point saddle) are read at
    # that level and the binding is the CONFINED-MOST (nearest ψ_N to the axis) of
    # the two — one rule, no limited/diverted branch.  A wall tangency binds a
    # limited plasma; an X-point saddle binds a diverted one; whichever is reached
    # first (smaller ψ_N) closes the boundary, exactly as the outermost-closed
    # connectivity read does.

    # region at the flood binding, and a few-cell dilation of it (the "flood
    # rejoin" band) — a candidate null must sit in this band to be ON the axis
    # region's edge, so a same-flux null elsewhere in (or out of) the vessel is
    # rejected.
    region_at = _flood_fill((u <= s_flood) & inside_limiter, seed, n_iter, use_doubling)
    reach = region_at > 0.5
    for _ in range(2):  # unrolled (static) — reach the wall polygon near the touch
        reach = _dilate4(reach)
    flood_adjacent = region_at > 0.5
    for _ in range(3):  # unrolled — the separatrix saddle sits at the region edge
        flood_adjacent = _dilate4(flood_adjacent)

    # --- sub-grid wall tangency ------------------------------------------------
    # The confined-most flux the closed surface reaches on the wall, read SUB-GRID
    # (the cell mean carries the sub-cell wall-position error a tangency is
    # sensitive to): the minimum interpolated ψ_N over the wall boundary points
    # adjacent to the axis region.  Computed for EVERY slice; on a diverted plasma
    # the reachable wall sits OUTSIDE the separatrix so this overshoots and loses
    # the confined-most min to the saddle below — no class switch needed.
    ar_idx = jnp.arange(nr, dtype=rg.dtype)
    az_idx = jnp.arange(nz, dtype=zg.dtype)
    wj = jnp.clip(jnp.round(jnp.interp(wall_r, rg, ar_idx)), 0, nr - 1)
    wi = jnp.clip(jnp.round(jnp.interp(wall_z, zg, az_idx)), 0, nz - 1)
    reachable = reach[wi.astype(jnp.int32), wj.astype(jnp.int32)]
    global_surface = fit_tensor_spline(rg, zg, psi2d)
    wall_global_flux = global_surface(wall_r, wall_z)
    u_wall_global = (wall_global_flux - psi_axis) / span_safe
    # Exact node flux (campaign g_wall) is authoritative where provided; the
    # global C2 tensor surface supplies the fallback.  A length-1 NaN sentinel
    # broadcasts to "all global-surface" while a per-node finite array reads each
    # tangency EXACTLY (no O(Δ²) floor at the lean point).  Sanitise the exact
    # branch to a finite value BEFORE the select so the NaN sentinel never
    # poisons the gradient (the jnp.where NaN-VJP trap).
    wall_psi_finite = jnp.isfinite(wall_psi)
    wall_psi_safe = jnp.where(wall_psi_finite, wall_psi, psi_axis)
    u_wall_exact = (wall_psi_safe - psi_axis) / span_safe
    u_wall_pts = jnp.where(wall_psi_finite, u_wall_exact, u_wall_global)
    u_wall = jnp.min(jnp.where(reachable, u_wall_pts, jnp.inf))
    u_wall_valid = jnp.any(reachable) & jnp.isfinite(u_wall)
    u_wall_c = jnp.where(u_wall_valid, u_wall, jnp.inf)

    # --- sub-grid X-point saddle at the binding (the diverted separatrix) -------
    # Classify saddles on the whole grid, then keep only those that (a) lie inside
    # the wall (xpoint_candidates ANDs inside_limiter), (b) sit in the flood-rejoin
    # band (spatially at the axis region's edge — rejects a same-flux X elsewhere),
    # (c) have flux within a band of the flood binding, and (d) are clear of the
    # axis (rejects a spurious near-axis null-space saddle — the device analogue of
    # the host min_axis_dist guard).  Refine sub-grid; the binding saddle is the
    # surviving candidate nearest the flood level.  Its flux closes the two-sided-
    # mean residual the diverted binding otherwise carries.
    d2_axis = (rg[None, :] - axis_r) ** 2 + (zg[:, None] - axis_z) ** 2
    min_axis_d = jnp.maximum(3.0 * (rg[1] - rg[0]), 0.05)
    x_mask = (
        (jnp.abs(u - s_flood) <= _X_FLUX_BAND)
        & (d2_axis >= min_axis_d**2)
        & flood_adjacent
    )
    xc = xpoint_candidates(psi2d, rg, zg, inside_limiter, _K_XCAND, extra_mask=x_mask)
    # Sanitise the masked-out (NaN) candidate flux to a finite value BEFORE any
    # arithmetic: a NaN numerator would poison the VJP of the /span_safe division
    # (0·NaN into span_safe, which depends on the axis), even though the value is
    # gated out downstream.  The x_valid flag still drops these slots.
    # The flood-adjacency and binding-flux band are independent connectivity
    # evidence.  They may consume an unresolved native-degree candidate while
    # preserving its state in the returned uncertainty metadata.
    xc_valid = xc["present"] & jnp.isfinite(xc["psi"])
    psi_x_safe = jnp.where(xc_valid, xc["psi"], psi_axis)
    u_x = (psi_x_safe - psi_axis) / span_safe  # (_K_XCAND,), finite everywhere
    x_valid = xc_valid
    x_key = jnp.where(x_valid, jnp.abs(u_x - s_flood), jnp.inf)
    kbind = _argmin_exact(x_key)
    x_bind_valid = x_valid[kbind] & jnp.isfinite(x_key[kbind])
    x_bind_state = jnp.where(x_bind_valid, xc["state"][kbind], 0)
    u_x_c = jnp.where(x_bind_valid, u_x[kbind], jnp.inf)

    # Exact candidate data can replace the class operands without changing the
    # connectivity binding.  Its wall operand is selected from the pre-saddle
    # axis component rather than trusting a whole-polygon extremum.
    if classification_x is None or classification_wall is None:
        class_u_wall = u_wall_c
        class_u_x = u_x_c
        class_x_valid = x_bind_valid
        class_x_state = x_bind_state
        class_wall_shadowed = jnp.asarray(False)
        class_wall = {
            "reachable": reachable,
            "node_index": _argmin_exact(jnp.where(reachable, u_wall_pts, jnp.inf)),
            "node_arc": jnp.asarray(jnp.nan, dtype=rg.dtype),
            "node_r": jnp.asarray(jnp.nan, dtype=rg.dtype),
            "node_z": jnp.asarray(jnp.nan, dtype=zg.dtype),
            "node_psi": jnp.asarray(jnp.nan, dtype=psi2d.dtype),
            "arc": jnp.asarray(jnp.nan, dtype=rg.dtype),
            "r": jnp.asarray(jnp.nan, dtype=rg.dtype),
            "z": jnp.asarray(jnp.nan, dtype=zg.dtype),
            "psi": jnp.asarray(jnp.nan, dtype=psi2d.dtype),
            "distance": jnp.asarray(jnp.inf, dtype=psi2d.dtype),
            "shift": jnp.asarray(jnp.nan, dtype=rg.dtype),
            "valid": u_wall_valid,
            "flux_from_global_surface": jnp.asarray(False),
            "reachable_samples": jnp.zeros((_WALL_REACHABILITY_SAMPLES,), dtype=bool),
            "selected_from_exact_nodes": jnp.asarray(False),
        }
    else:
        supplied_x = jnp.asarray(classification_x, dtype=psi2d.dtype)
        supplied_wall = jnp.asarray(classification_wall, dtype=psi2d.dtype)
        supplied_x_present = jnp.all(jnp.isfinite(supplied_x[:, :3]), axis=1)
        supplied_x_inside_wall = _points_inside_polygon(
            supplied_x[:, 0], supplied_x[:, 1], wall_r, wall_z
        )
        supplied_x_valid = supplied_x_present & supplied_x_inside_wall
        supplied_x_flux = jnp.where(supplied_x_valid, supplied_x[:, 2], psi_axis)
        supplied_x_level = (supplied_x_flux - psi_axis) / span_safe
        class_x_index = _argmin_exact(
            jnp.where(supplied_x_valid, supplied_x_level, jnp.inf)
        )
        class_x_valid = supplied_x_valid[class_x_index]
        class_u_x = jnp.where(class_x_valid, supplied_x_level[class_x_index], jnp.inf)
        class_x_state = jnp.where(class_x_valid, 2, 0)

        pre_saddle_region = _axis_component_before_level(
            u,
            inside_limiter,
            rg,
            zg,
            axis_r,
            axis_z,
            class_u_x,
            n_iter,
            use_doubling,
        )
        class_wall = _select_reachable_wall_limiter(
            psi2d,
            rg,
            zg,
            inside_limiter,
            wall_r,
            wall_z,
            wall_psi,
            pre_saddle_region,
            psi_axis,
            global_surface,
        )
        supplied_wall_valid = jnp.all(jnp.isfinite(supplied_wall[:3]))
        supplied_wall_index = _argmin_exact(
            (wall_r - supplied_wall[0]) ** 2 + (wall_z - supplied_wall[1]) ** 2
        )
        class_wall_shadowed = (
            supplied_wall_valid & ~class_wall["reachable"][supplied_wall_index]
        )
        refined_wall_level = (class_wall["psi"] - psi_axis) / span_safe
        class_u_wall = jnp.where(
            class_wall["valid"],
            refined_wall_level,
            jnp.inf,
        )

    return {
        "n_iter": n_iter,
        "seed": seed,
        "psi_axis": psi_axis,
        "psi_out": psi_out,
        "span": span,
        "span_safe": span_safe,
        "u": u,
        "found": found,
        "s_flood": s_flood,
        "binding_search_warm": binding_search_warm,
        "binding_search_evaluations": binding_search_evaluations,
        "u_wall_c": u_wall_c,
        "u_x_c": u_x_c,
        "class_u_wall": class_u_wall,
        "class_u_x": class_u_x,
        "class_x_valid": class_x_valid,
        "class_x_state": class_x_state,
        "class_wall_shadowed": class_wall_shadowed,
        "class_wall": class_wall,
        "class_x_inside_wall": supplied_x_inside_wall
        if classification_x is not None
        else jnp.ones((_K_XCAND,), dtype=bool),
        "x_bind_valid": x_bind_valid,
        "x_bind_state": x_bind_state,
        "u_x": u_x,
        "x_valid": x_valid,
        "xc": xc,
        "x_candidate_count": xc["candidate_count"],
        "x_overflow": xc["overflow"],
        "x_discarded_score_upper_bound": xc["discarded_score_upper_bound"],
        "x_unresolved_count": jnp.sum(xc["state"] == 1, dtype=jnp.int32),
    }


@partial(jax.jit, static_argnums=(6, 7, 8, 15))
def traced_boundary_read(
    psi2d: jnp.ndarray,
    rg: jnp.ndarray,
    zg: jnp.ndarray,
    inside_limiter: jnp.ndarray,
    axis_r,
    axis_z,
    n_levels: int = 96,
    n_bisect: int = 18,
    n_ray: int = 512,
    angles: jnp.ndarray | None = None,
    lcfs_norm=0.999,
    wall_r: jnp.ndarray | None = None,
    wall_z: jnp.ndarray | None = None,
    wall_psi: jnp.ndarray | None = None,
    previous_flood_level=jnp.nan,
    use_doubling: bool = True,
    classification_x: jnp.ndarray | None = None,
    classification_wall: jnp.ndarray | None = None,
) -> dict:
    """Connectivity LCFS read from ψ — the device-native ``lcfs_contour``.

    ``psi2d`` is ``(nz, nr)`` total poloidal flux; ``rg``/``zg`` the axis-ordered
    grid coordinates; ``inside_limiter`` the ``(nz, nr)`` boolean wall (raster)
    mask; ``(axis_r, axis_z)`` the read's axis (the current centroid, in metres).
    ``wall_r``/``wall_z`` are the wall boundary sample points (all wall units
    densified) — used for the SUB-GRID binding flux (see below); omit them to fall
    back to the cell-level flood binding.  ``wall_psi`` is the EXACT node flux
    (the campaign ``g_wall`` GEMM) aligned with those points; each finite entry
    reads its exact flux instead of a reconstructed grid value.  The sentinel
    NaN default evaluates the global C2 tensor surface at every wall node.
    ``classification_x`` and ``classification_wall`` optionally supply the
    exact saddle table and whole-polygon wall extremum used by the Boolean
    topology read. Typed saddles outside the ``wall_r`` / ``wall_z`` polygon are
    masked before selection. The wall operand is independently selected and
    refined on the wall reachable from the pre-saddle axis component. Their
    normalised flux difference defines ``class_margin``; they do not change
    ``s_star``, ``psi_bnd``, radii, or the connected core.

    Returns a dict of fixed-shape arrays: ``found`` (bool — a valid closed
    axis-enclosing level exists), ``psi_axis``, ``psi_out``, ``psi_bnd`` (the
    binding / separatrix / wall flux), ``psi_lcfs`` (the reported ring flux),
    ``s_star`` (the binding level in [0, 1]), ``radii`` ``(len(angles),)`` LCFS
    radii about the axis [m], and ``n_core_cells``.  ``jit``/``vmap``/``grad``-safe.
    """
    angles, wall_r, wall_z, wall_psi = _boundary_defaults(
        psi2d, rg, zg, angles, wall_r, wall_z, wall_psi
    )
    ing = _read_ingredients(
        psi2d,
        rg,
        zg,
        inside_limiter,
        axis_r,
        axis_z,
        n_levels,
        n_bisect,
        wall_r,
        wall_z,
        wall_psi,
        previous_flood_level,
        use_doubling,
        classification_x,
        classification_wall,
    )
    n_iter = ing["n_iter"]
    seed = ing["seed"]
    psi_axis = ing["psi_axis"]
    psi_out = ing["psi_out"]
    span = ing["span"]
    u = ing["u"]
    found = ing["found"]
    s_flood = ing["s_flood"]
    u_wall_c = ing["u_wall_c"]
    u_x_c = ing["u_x_c"]
    class_x_valid = ing["class_x_valid"]
    class_x_state = ing["class_x_state"]
    u_x = ing["u_x"]
    x_valid = ing["x_valid"]
    xc = ing["xc"]

    # --- unified binding: confined-most of {wall tangency, X-point saddle} ------
    u_min = jnp.minimum(u_wall_c, u_x_c)
    s_star = jnp.where(jnp.isfinite(u_min), u_min, s_flood)

    # diverted iff the X-point saddle is the confined-most obstruction; the margin
    # is a SOFT continuous quantity (>0 diverted, <0 limited, ~0 marginal) so the
    # class is never a code-path switch.
    class_u_wall = ing["class_u_wall"]
    class_u_x = ing["class_u_x"]
    is_diverted = class_x_valid & (class_u_x <= class_u_wall)
    boundary_resolved = (~is_diverted) | (class_x_state == 2)
    class_margin = class_u_wall - class_u_x

    psi_bnd = psi_axis + s_star * span
    # Radii are read on the surface the ray-cast sits on, ALWAYS a hair inside the
    # separatrix (≤0.999·span): a ray cast at exactly ψ_bnd runs down an open
    # divertor leg through the X-point cusp (the closed-lobe host read never does),
    # so lcfs_norm is clamped for the ray while ψ_bnd itself reports the true
    # separatrix / wall flux the caller (e.g. the disc pushout, clip_legs) wants.
    ray_norm = jnp.minimum(lcfs_norm, 0.999)
    psi_lcfs = psi_axis + ray_norm * (psi_bnd - psi_axis)
    radii = _ray_radii(psi2d, rg, zg, axis_r, axis_z, psi_axis, psi_lcfs, angles, n_ray)

    confined_star = (u <= s_star) & inside_limiter
    region_star = _flood_fill(confined_star, seed, n_iter, use_doubling)
    n_core = jnp.sum(region_star)

    # --- classify-after: sub-grid axis O-point + emergent X-set ----------------
    # The nulls are read AFTER the boundary, never as a prerequisite for ψ_N.  The
    # axis is the deepest O-point inside the confined region; the emergent X-set is
    # the saddles sitting ON the boundary (order-invariant, NaN-padded), the device
    # analogue of the host emergent_xpoints soft-margin rule.
    ax = magnetic_axis_subgrid(psi2d, rg, zg, inside_limiter, region=region_star)
    on_bound = x_valid & (jnp.abs(u_x - s_star) <= _X_ON_RING_U)
    ob_key = jnp.where(on_bound, jnp.abs(u_x - s_star), jnp.inf)
    order = jnp.argsort(ob_key)
    xr_s = xc["r"][order]
    xz_s = xc["z"][order]
    ob_s = on_bound[order]
    # Greedy flux-ordered fill with SPATIAL dedupe: on a coarse grid the
    # 4-sign-change stencil can fire on two adjacent vertices of ONE physical
    # saddle, and both duplicates rank ahead of a genuinely distinct null (a
    # balanced double-null loses its second X to the crowding).  A candidate
    # within ~one stencil footprint of an already-taken slot refines the SAME
    # saddle, so it is skipped instead of consuming a slot.  Fixed unroll over
    # the static candidate count — jit/vmap/grad-safe.
    dedupe_d2 = (1.5 * jnp.maximum(rg[1] - rg[0], zg[1] - zg[0])) ** 2
    sel_r = jnp.full((N_XPOINT_SLOTS,), jnp.nan, dtype=xr_s.dtype)
    sel_z = jnp.full((N_XPOINT_SLOTS,), jnp.nan, dtype=xz_s.dtype)
    sel_state = jnp.zeros((N_XPOINT_SLOTS,), dtype=xc["state"].dtype)
    n_taken = jnp.asarray(0)
    for m in range(_K_XCAND):
        d2 = (sel_r - xr_s[m]) ** 2 + (sel_z - xz_s[m]) ** 2  # NaN on empty slots
        dup = jnp.any(d2 < dedupe_d2)  # NaN compares False — empty slots pass
        take_m = ob_s[m] & ~dup & (n_taken < N_XPOINT_SLOTS)
        slot = jnp.clip(n_taken, 0, N_XPOINT_SLOTS - 1)
        sel_r = jnp.where(take_m, sel_r.at[slot].set(xr_s[m]), sel_r)
        sel_z = jnp.where(take_m, sel_z.at[slot].set(xz_s[m]), sel_z)
        sel_state = jnp.where(
            take_m, sel_state.at[slot].set(xc["state"][order[m]]), sel_state
        )
        n_taken = n_taken + take_m.astype(n_taken.dtype)
    xset = jnp.stack([sel_r, sel_z], axis=1)  # (N_XPOINT_SLOTS, 2)

    return {
        "found": found,
        "psi_axis": psi_axis,
        "psi_out": psi_out,
        "psi_bnd": jnp.where(found, psi_bnd, jnp.nan),
        "psi_lcfs": jnp.where(found, psi_lcfs, jnp.nan),
        "s_star": jnp.where(found, s_star, jnp.nan),
        "s_flood": jnp.where(found, s_flood, jnp.nan),
        "binding_search_warm": ing["binding_search_warm"],
        "binding_search_evaluations": ing["binding_search_evaluations"],
        "radii": jnp.where(found, radii, jnp.nan),
        "n_core_cells": n_core,
        # classify-after diagnostics (never feed ψ_N)
        "axis_r": jnp.where(ax["found"], ax["r"], jnp.nan),
        "axis_z": jnp.where(ax["found"], ax["z"], jnp.nan),
        "axis_psi_sub": jnp.where(ax["found"], ax["psi"], jnp.nan),
        "axis_state": ax["state"],
        "axis_confidence": ax["confidence"],
        "axis_candidate_count": ax["candidate_count"],
        "axis_overflow": ax["overflow"],
        "xset": xset,
        "xset_state": sel_state,
        "is_diverted": is_diverted,
        "boundary_resolved": boundary_resolved,
        "x_binding_state": class_x_state,
        "class_margin": class_margin,
        "u_wall": class_u_wall,
        "u_xpoint": class_u_x,
        "wall_shadowed": ing["class_wall_shadowed"],
        "reachable_wall_node_mask": ing["class_wall"]["reachable"],
        "reachable_wall_sample_mask": ing["class_wall"]["reachable_samples"],
        "limiter_wall_node_index": ing["class_wall"]["node_index"],
        "limiter_wall_node_arc": ing["class_wall"]["node_arc"],
        "limiter_wall_node_r": ing["class_wall"]["node_r"],
        "limiter_wall_node_z": ing["class_wall"]["node_z"],
        "limiter_wall_node_psi": ing["class_wall"]["node_psi"],
        "limiter_arc": ing["class_wall"]["arc"],
        "limiter_r": ing["class_wall"]["r"],
        "limiter_z": ing["class_wall"]["z"],
        "limiter_psi": ing["class_wall"]["psi"],
        "limiter_axis_flux_distance": ing["class_wall"]["distance"],
        "limiter_refinement_shift": ing["class_wall"]["shift"],
        "limiter_flux_from_global_surface": ing["class_wall"][
            "flux_from_global_surface"
        ],
        "limiter_selected_from_exact_nodes": ing["class_wall"][
            "selected_from_exact_nodes"
        ],
        "binding_u_wall": u_wall_c,
        "binding_u_xpoint": u_x_c,
        "x_candidate_count": ing["x_candidate_count"],
        "x_overflow": ing["x_overflow"],
        "x_discarded_score_upper_bound": ing["x_discarded_score_upper_bound"],
        "x_unresolved_count": ing["x_unresolved_count"],
    }


@partial(jax.jit, static_argnums=(6, 7, 13))
def traced_margin_candidate_diagnostics(
    psi2d: jnp.ndarray,
    rg: jnp.ndarray,
    zg: jnp.ndarray,
    inside_limiter: jnp.ndarray,
    axis_r,
    axis_z,
    n_levels: int,
    n_bisect: int,
    wall_r: jnp.ndarray,
    wall_z: jnp.ndarray,
    wall_psi: jnp.ndarray,
    classification_x: jnp.ndarray,
    classification_wall: jnp.ndarray,
    use_doubling: bool = True,
) -> dict:
    """Return fixed-shape evidence behind one exact topology-margin read.

    The exact comparator candidates determine the saddle operand.  The supplied
    whole-polygon wall extremum is retained for before/after evidence, while the
    class wall operand comes from the reachable pre-saddle wall.  The
    connectivity-local candidate table is retained beside them so a receipt can
    distinguish an absent typed saddle from a typed saddle that the flood-rejoin
    and binding-flux admission did not retain.
    This diagnostic adapter does not feed the boundary or class calculation.
    """
    supplied_x = jnp.asarray(classification_x, dtype=psi2d.dtype)
    supplied_wall = jnp.asarray(classification_wall, dtype=psi2d.dtype)
    ing = _read_ingredients(
        psi2d,
        rg,
        zg,
        inside_limiter,
        axis_r,
        axis_z,
        n_levels,
        n_bisect,
        wall_r,
        wall_z,
        wall_psi,
        jnp.asarray(jnp.nan, dtype=psi2d.dtype),
        use_doubling,
        supplied_x,
        supplied_wall,
    )

    typed_present = jnp.all(jnp.isfinite(supplied_x[:, :3]), axis=1)
    typed_inside_wall = _points_inside_polygon(
        supplied_x[:, 0], supplied_x[:, 1], wall_r, wall_z
    )
    typed_eligible = typed_present & typed_inside_wall
    typed_flux = jnp.where(typed_present, supplied_x[:, 2], ing["psi_axis"])
    typed_level = (typed_flux - ing["psi_axis"]) / ing["span_safe"]
    selected_index = _argmin_exact(jnp.where(typed_eligible, typed_level, jnp.inf))
    selected_present = typed_eligible[selected_index]
    selected_candidate = jnp.where(
        selected_present,
        supplied_x[selected_index],
        jnp.full(supplied_x.shape[1:], jnp.nan, dtype=supplied_x.dtype),
    )

    wall_present = jnp.all(jnp.isfinite(supplied_wall[:3]))
    wall_flux_safe = jnp.where(wall_present, supplied_wall[2], ing["psi_axis"])
    wall_level_before_shadow = (wall_flux_safe - ing["psi_axis"]) / ing["span_safe"]
    refined_wall = ing["class_wall"]
    connectivity = ing["xc"]
    connectivity_table = jnp.stack(
        (
            connectivity["r"],
            connectivity["z"],
            connectivity["psi"],
            connectivity["ntype"],
        ),
        axis=-1,
    )

    return {
        "class_margin": ing["class_u_wall"] - ing["class_u_x"],
        "axis_flux": ing["psi_axis"],
        "outward_flux_span": ing["span"],
        "typed_candidates": supplied_x,
        "typed_candidate_present": typed_present,
        "typed_candidate_inside_wall": typed_inside_wall,
        "typed_candidate_eligible": typed_eligible,
        "typed_candidate_count": jnp.sum(typed_present, dtype=jnp.int32),
        "typed_candidate_eligible_count": jnp.sum(typed_eligible, dtype=jnp.int32),
        "selected_typed_candidate_index": selected_index,
        "selected_typed_candidate_present": selected_present,
        "selected_typed_candidate": selected_candidate,
        "selected_x_normalized_flux_operand": ing["class_u_x"],
        "wall_candidate": supplied_wall,
        "wall_candidate_present": wall_present,
        "wall_normalized_flux_operand_before_shadow": wall_level_before_shadow,
        "wall_normalized_flux_operand": ing["class_u_wall"],
        "wall_shadowed": ing["class_wall_shadowed"],
        "reachable_wall_node_mask": refined_wall["reachable"],
        "reachable_wall_node_count": jnp.sum(
            refined_wall["reachable"], dtype=jnp.int32
        ),
        "reachable_wall_sample_mask": refined_wall["reachable_samples"],
        "reachable_wall_sample_count": jnp.sum(
            refined_wall["reachable_samples"], dtype=jnp.int32
        ),
        "limiter_wall_node_index": refined_wall["node_index"],
        "limiter_wall_node_arc": refined_wall["node_arc"],
        "limiter_wall_node_coordinate": jnp.stack(
            (refined_wall["node_r"], refined_wall["node_z"])
        ),
        "limiter_wall_node_flux": refined_wall["node_psi"],
        "limiter_arc": refined_wall["arc"],
        "limiter_coordinate": jnp.stack((refined_wall["r"], refined_wall["z"])),
        "limiter_flux": refined_wall["psi"],
        "limiter_axis_flux_distance": refined_wall["distance"],
        "limiter_refinement_shift": refined_wall["shift"],
        "limiter_flux_from_global_surface": refined_wall["flux_from_global_surface"],
        "limiter_selected_from_exact_nodes": refined_wall["selected_from_exact_nodes"],
        "connectivity_candidates": connectivity_table,
        "connectivity_candidate_present": connectivity["present"],
        "connectivity_candidate_admitted": ing["x_valid"],
        "connectivity_candidate_resolved": connectivity["resolved"],
        "connectivity_candidate_state": connectivity["state"],
        "connectivity_candidate_confidence": connectivity["confidence"],
        "connectivity_candidate_class_margin": connectivity["class_margin"],
        "connectivity_candidate_boundary_snr": connectivity["boundary_snr"],
        "connectivity_candidate_root_support_cell": connectivity["root_support_cell"],
        "connectivity_candidate_count_before_capacity": ing["x_candidate_count"],
        "connectivity_admitted_slot_count": jnp.sum(ing["x_valid"], dtype=jnp.int32),
        "connectivity_candidate_overflow": ing["x_overflow"],
        "connectivity_discarded_score_upper_bound": ing[
            "x_discarded_score_upper_bound"
        ],
    }


# ---------------------------------------------------------------------------
# the smooth (differentiable) read — softmin binding + sigmoid core mask
# ---------------------------------------------------------------------------

#: finite stand-in for an ABSENT binding candidate (``inf`` sentinel) inside the
#: softmin — far outside the confined range so its softmax weight vanishes, yet
#: finite so no ``0·inf`` poisons the reverse pass.
_ABSENT_U = 2.0


@partial(jax.jit, static_argnums=(6, 7, 8, 16))
def traced_smooth_boundary_read(
    psi2d: jnp.ndarray,
    rg: jnp.ndarray,
    zg: jnp.ndarray,
    inside_limiter: jnp.ndarray,
    axis_r,
    axis_z,
    n_levels: int = 96,
    n_bisect: int = 18,
    n_ray: int = 512,
    angles: jnp.ndarray | None = None,
    lcfs_norm=0.999,
    wall_r: jnp.ndarray | None = None,
    wall_z: jnp.ndarray | None = None,
    wall_psi: jnp.ndarray | None = None,
    temperature=0.01,
    previous_flood_level=jnp.nan,
    use_doubling: bool = True,
    classification_x: jnp.ndarray | None = None,
    classification_wall: jnp.ndarray | None = None,
) -> dict:
    """The SMOOTH connectivity boundary read — the end-to-end differentiable path.

    Same ingredients as :func:`traced_boundary_read` (the hard reference), with the
    two remaining hard thresholds replaced by temperature-controlled smooth
    surrogates so a gradient flows from any read scalar back to ψ (and through a
    linear Green's map, to the currents):

    * **Soft binding level.**  The hard read binds at the exact
      ``min(u_wall, u_x)``; here the binding is the softmin — the softmax-
      weighted mean of the two sub-grid candidates at temperature ``τ``
      (in ψ_N span units).  As τ→0 this reduces to the exact min; at finite τ
      the wall/saddle hand-off is a smooth blend instead of a kink.

    * **Smooth class score.** ``p_diverted`` is the saddle weight from a second
      softmax over the class operands. These are the connectivity candidates by
      default, or the exact comparator candidates when supplied. Consequently
      an exact class read does not change the boundary or core construction.

    * **Smooth core mask.**  The hard ``(u ≤ s*) ∧ flood`` cell mask becomes
      ``σ((s_soft − u)/τ) · gate`` — a sigmoid cutoff in normalised flux, gated
      by the axis-connected flood region (evaluated at ``s_soft + 3τ`` so the
      sigmoid tail is inside the gate).  The gate is a boolean connectivity
      SELECTION (no gradient path, exactly like an argmin index), so private-
      flux pockets stay excluded by connectivity while the mask edge moves
      smoothly with ψ.  As τ→0 the sigmoid → the step and the mask → the hard
      core mask.

    Everything else (flood localisation, sub-grid wall tangency / saddle flux,
    ray-marched radii) is shared with the hard read.  Returns ``found``,
    ``psi_axis``, ``psi_out``, ``psi_bnd``, ``psi_lcfs``, ``s_soft``, ``radii``,
    ``core_weight`` ``(nz, nr)``, ``n_core_soft``, ``p_diverted``, ``u_wall``,
    ``u_xpoint``.  ``jit``/``vmap``/``grad``-safe.
    """
    angles, wall_r, wall_z, wall_psi = _boundary_defaults(
        psi2d, rg, zg, angles, wall_r, wall_z, wall_psi
    )
    ing = _read_ingredients(
        psi2d,
        rg,
        zg,
        inside_limiter,
        axis_r,
        axis_z,
        n_levels,
        n_bisect,
        wall_r,
        wall_z,
        wall_psi,
        previous_flood_level,
        use_doubling,
        classification_x,
        classification_wall,
    )
    tau = temperature
    psi_axis = ing["psi_axis"]
    span = ing["span"]
    u = ing["u"]
    found = ing["found"]
    s_flood = ing["s_flood"]

    # --- soft binding: softmin over the two sub-grid candidates -----------------
    cands = jnp.stack([ing["u_wall_c"], ing["u_x_c"]])  # (2,), inf when absent
    valid = jnp.isfinite(cands)
    any_valid = jnp.any(valid)
    c_safe = jnp.where(valid, cands, _ABSENT_U)
    logits = jnp.where(valid, -c_safe / tau, -jnp.inf)
    # when NEITHER candidate exists the logits are all −inf and softmax would be
    # NaN (poisoning the grad through the select below even though the value is
    # discarded) — substitute uniform logits first, then discard via `where`.
    logits = jnp.where(any_valid, logits, jnp.zeros_like(logits))
    w = jax.nn.softmax(logits)
    s_soft = jnp.where(any_valid, jnp.sum(w * c_safe), s_flood)
    class_cands = jnp.stack([ing["class_u_wall"], ing["class_u_x"]])
    class_valid = jnp.isfinite(class_cands)
    class_any_valid = jnp.any(class_valid)
    class_safe = jnp.where(class_valid, class_cands, _ABSENT_U)
    class_logits = jnp.where(class_valid, -class_safe / tau, -jnp.inf)
    class_logits = jnp.where(
        class_any_valid, class_logits, jnp.zeros_like(class_logits)
    )
    class_weight = jax.nn.softmax(class_logits)
    p_diverted = jnp.where(class_any_valid, class_weight[1], 0.0)

    psi_bnd = psi_axis + s_soft * span
    ray_norm = jnp.minimum(lcfs_norm, 0.999)
    psi_lcfs = psi_axis + ray_norm * (psi_bnd - psi_axis)
    radii = _ray_radii(psi2d, rg, zg, axis_r, axis_z, psi_axis, psi_lcfs, angles, n_ray)

    # --- smooth core mask --------------------------------------------------------
    # sigmoid cutoff in u at the soft level, gated by the axis-connected flood ONE
    # TEMPERATURE INSIDE the binding (u ≤ s_soft − τ).  The retraction is what
    # seals the saddle pass: a grid sample AT a saddle vertex always sits a hair
    # below the sub-grid saddle flux (an O(Δ²) deficit), so a flood at exactly
    # s_soft walks through that cell and pours the mask into the private-flux /
    # opposing-null pocket beyond it; τ ≫ the vertex deficit, so the retracted
    # flood stops short of the pass while costing only an O(τ) shell that
    # vanishes with the temperature.  The boolean comparison carries no gradient
    # — the gate is a connectivity SELECTION, like an argmin index — while σ
    # (still centred on s_soft) moves the mask edge smoothly with ψ; a pocket is
    # never axis-connected, so its cells stay at zero weight for any τ.
    gate = _flood_fill(
        (u <= s_soft - tau) & inside_limiter,
        ing["seed"],
        ing["n_iter"],
        use_doubling,
    )
    core_weight = jax.nn.sigmoid((s_soft - u) / tau) * gate
    n_core_soft = jnp.sum(core_weight)

    return {
        "found": found,
        "psi_axis": psi_axis,
        "psi_out": ing["psi_out"],
        "psi_bnd": jnp.where(found, psi_bnd, jnp.nan),
        "psi_lcfs": jnp.where(found, psi_lcfs, jnp.nan),
        "s_soft": jnp.where(found, s_soft, jnp.nan),
        "s_flood": jnp.where(found, s_flood, jnp.nan),
        "binding_search_warm": ing["binding_search_warm"],
        "binding_search_evaluations": ing["binding_search_evaluations"],
        "radii": jnp.where(found, radii, jnp.nan),
        "core_weight": core_weight,
        "n_core_soft": n_core_soft,
        "p_diverted": p_diverted,
        "u_wall": ing["class_u_wall"],
        "u_xpoint": ing["class_u_x"],
        "wall_shadowed": ing["class_wall_shadowed"],
        "binding_u_wall": ing["u_wall_c"],
        "binding_u_xpoint": ing["u_x_c"],
        "x_candidate_count": ing["x_candidate_count"],
        "x_overflow": ing["x_overflow"],
        "x_discarded_score_upper_bound": ing["x_discarded_score_upper_bound"],
        "x_unresolved_count": ing["x_unresolved_count"],
    }


traced_emit_boundary_read = traced_smooth_boundary_read


def _coarse_indices(size: int, stride: int):
    """Return static subsampling indices while retaining both domain edges."""
    indices = jnp.arange(0, size, stride, dtype=jnp.int32)
    if (size - 1) % stride:
        indices = jnp.concatenate([indices, jnp.asarray([size - 1], dtype=jnp.int32)])
    return indices


def _interpolate_grid(field, coarse_r, coarse_z, full_r, full_z):
    """Bilinearly restore a coarse raster to the solve grid."""
    along_r = jax.vmap(lambda row: jnp.interp(full_r, coarse_r, row))(field)
    return jax.vmap(
        lambda column: jnp.interp(full_z, coarse_z, column),
        in_axes=1,
        out_axes=1,
    )(along_r)


@partial(jax.jit, static_argnums=(6, 7, 8, 16, 17))
def traced_iteration_boundary_read(
    psi2d: jnp.ndarray,
    rg: jnp.ndarray,
    zg: jnp.ndarray,
    inside_limiter: jnp.ndarray,
    axis_r,
    axis_z,
    n_levels: int = 96,
    n_bisect: int = 18,
    n_ray: int = 512,
    angles: jnp.ndarray | None = None,
    lcfs_norm=0.999,
    wall_r: jnp.ndarray | None = None,
    wall_z: jnp.ndarray | None = None,
    wall_psi: jnp.ndarray | None = None,
    temperature=0.01,
    previous_flood_level=jnp.nan,
    resolution_stride: int = 2,
    use_doubling: bool = True,
) -> dict:
    """Smooth topology approximation for nonlinear map evaluations.

    The solve field and limiter are sampled on a static, edge-preserving raster;
    scalar topology values are read there and the smooth core weight is restored
    to the solve grid by bilinear interpolation.  Final reported state must use
    :func:`traced_emit_boundary_read`, which is the calibrated full-resolution
    read.  A stride of one is exactly the full-resolution implementation.
    """
    r_index = _coarse_indices(rg.shape[0], resolution_stride)
    z_index = _coarse_indices(zg.shape[0], resolution_stride)
    coarse_r = rg[r_index]
    coarse_z = zg[z_index]
    coarse_psi = psi2d[z_index[:, None], r_index[None, :]]
    coarse_inside = inside_limiter[z_index[:, None], r_index[None, :]]
    result = traced_smooth_boundary_read(
        coarse_psi,
        coarse_r,
        coarse_z,
        coarse_inside,
        axis_r,
        axis_z,
        n_levels,
        n_bisect,
        n_ray,
        angles,
        lcfs_norm,
        wall_r,
        wall_z,
        wall_psi,
        temperature,
        previous_flood_level,
        use_doubling,
    )
    core_weight = _interpolate_grid(result["core_weight"], coarse_r, coarse_z, rg, zg)
    return {
        **result,
        "core_weight": core_weight,
        "n_core_soft": jnp.sum(core_weight),
        "iteration_resolution_stride": jnp.asarray(resolution_stride, dtype=jnp.int32),
    }


@partial(jax.jit, static_argnums=(6, 7, 8))
def _smooth_read_at_stencil_axis(
    psi2d,
    rg,
    zg,
    inside_limiter,
    seed_r,
    seed_z,
    n_levels: int,
    n_bisect: int,
    n_ray: int,
    angles,
    lcfs_norm,
    wall_r,
    wall_z,
    wall_psi,
    temperature,
):
    """Stencil O-point first, smooth read seeded at it — the solve-map wiring.

    The sub-grid stencil axis is read from the raw field (biquadratic refine,
    differentiable through the surface fit; falls back to the caller's seed
    when no in-wall O-point exists), the smooth boundary read floods from that
    axis, and the axis scalars ride along in the result dict
    (``axis_r``/``axis_z``/``axis_psi_sub``, NaN when absent).  One jitted
    graph per grid shape, so a per-sweep host caller pays no retrace.
    """
    ax = magnetic_axis_subgrid(psi2d, rg, zg, inside_limiter)
    ax_r = jnp.where(ax["found"], ax["r"], seed_r)
    ax_z = jnp.where(ax["found"], ax["z"], seed_z)
    out = traced_smooth_boundary_read(
        psi2d,
        rg,
        zg,
        inside_limiter,
        ax_r,
        ax_z,
        n_levels,
        n_bisect,
        n_ray,
        angles,
        lcfs_norm,
        wall_r,
        wall_z,
        wall_psi,
        temperature,
    )
    out = dict(out)
    out["axis_r"] = jnp.where(ax["found"], ax["r"], jnp.nan)
    out["axis_z"] = jnp.where(ax["found"], ax["z"], jnp.nan)
    out["axis_psi_sub"] = jnp.where(ax["found"], ax["psi"], jnp.nan)
    return out


# ---------------------------------------------------------------------------
# host adapters
# ---------------------------------------------------------------------------


def _densify_wall(grid, m: int = 720):
    """Wall surface nodes ``(wall_r, wall_z)`` for the SUB-GRID binding flux.

    Prefers the grid's precomputed multi-unit nodes (``wall_r``/``wall_z`` built
    at ~Δ/2 over every wall unit — vessel, tiles, movable limiters), so an
    arbitrary machine wall is data, not a code path.  Falls back to resampling a
    single ``limiter_r``/``limiter_z`` loop to ``m`` points (a bare test grid or
    an older grid), and to the single far-away no-wall point when neither exists
    (the read then uses the cell-level flood binding).
    """
    import numpy as np  # noqa: PLC0415

    gwr = getattr(grid, "wall_r", None)
    gwz = getattr(grid, "wall_z", None)
    if gwr is not None and gwz is not None and len(np.asarray(gwr)) >= 1:
        return np.asarray(gwr, dtype=np.float64), np.asarray(gwz, dtype=np.float64)

    lr = getattr(grid, "limiter_r", None)
    lz = getattr(grid, "limiter_z", None)
    if lr is None or lz is None or len(np.asarray(lr)) < 2:
        return np.array([1.0e30]), np.array([1.0e30])
    lr = np.asarray(lr, dtype=np.float64)
    lz = np.asarray(lz, dtype=np.float64)
    # close the loop, cumulative arc length, resample uniformly to m points
    rr = np.append(lr, lr[0])
    zz = np.append(lz, lz[0])
    seg = np.hypot(np.diff(rr), np.diff(zz))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    if total <= 0.0:
        return np.array([1.0e30]), np.array([1.0e30])
    q = np.linspace(0.0, total, m, endpoint=False)
    return np.interp(q, s, rr), np.interp(q, s, zz)


@dataclass
class ConnectivityBoundary:
    """Host-side result of :func:`host_boundary_read` (mirrors contour fields)."""

    found: bool
    psi_bnd: float
    psi_lcfs: float
    psi_axis: float
    radii: object  # np.ndarray (len(angles),) [m]
    s_star: float
    n_core_cells: int
    # classify-after diagnostics (read AFTER the boundary; never feed ψ_N)
    axis: tuple[float, float]  # sub-grid magnetic axis (O-point) [m]
    axis_psi: float  # sub-grid axis flux [Wb]
    xset: object  # np.ndarray (N_XPOINT_SLOTS, 2) NaN-padded emergent X-points [m]
    is_diverted: bool
    class_margin: float  # u_wall − u_xpoint (>0 diverted, <0 limited, ~0 marginal)
    axis_state: int
    axis_confidence: float
    axis_candidate_count: int
    axis_overflow: bool
    x_candidate_count: int
    x_overflow: bool
    x_discarded_score_upper_bound: float
    x_unresolved_count: int
    xset_state: object
    boundary_resolved: bool
    x_binding_state: int


def host_boundary_read(
    psi2d,
    grid,
    axis: tuple[float, float],
    *,
    n_levels: int = 96,
    n_bisect: int = 18,
    n_ray: int = 512,
    angles=LCFS_ANGLES,
    lcfs_norm: float = 0.999,
    wall_psi=None,
    precision: Precision | str = Precision.AUTOMATIC,
) -> ConnectivityBoundary:
    """Host adapter: run :func:`traced_boundary_read` on one slice, return numpy.

    ``grid`` is an any equilibrium grid (supplies
    ``rg``/``zg``/``inside_limiter`` and the multi-unit wall nodes).  ``wall_psi``
    is the exact node flux (``grid.wall_flux(i_pf, i_cell)``) aligned with the
    grid's wall nodes; pass it to read the tangency exactly (the campaign
    ``g_wall`` GEMM) instead of bilinear off the grid.  ``lcfs_norm=1.0`` reports
    the ring AT the separatrix (the ``lcfs_contour(clip_legs=True)`` convention
    used by the disc pushout); the 0.999 default reads a hair inside.
    """
    import numpy as np  # noqa: PLC0415

    resolved = resolve_precision(precision, Precision.DOUBLE)
    dtype = jnp.float32 if resolved is Precision.SINGLE else jnp.float64
    geometry_dtype = jnp.float64
    wall_r, wall_z = _densify_wall(grid)
    # ONE hard error per solve: the flood seed (the axis cell) must be occupiable
    # — if it lands in wall material (or outside the vessel) the connectivity read
    # has no plasma to grow.  (The read itself is fully differentiable; this is a
    # host-side precondition, not a branch inside the device kernel.)
    inside = np.asarray(grid.inside_limiter, dtype=bool)
    rg_np = np.asarray(grid.rg, dtype=np.float64)
    zg_np = np.asarray(grid.zg, dtype=np.float64)
    ia = int(np.argmin(np.abs(zg_np - float(axis[1]))))
    ja = int(np.argmin(np.abs(rg_np - float(axis[0]))))
    if not inside[ia, ja]:
        raise ValueError(
            f"flood seed (axis cell R={axis[0]:.4f}, Z={axis[1]:.4f} → grid "
            f"[{ia}, {ja}]) lies in wall material / outside the vessel — no "
            "axis-connected plasma to grow"
        )
    if wall_psi is None:
        wpsi = jnp.asarray([jnp.nan], dtype=dtype)
    else:
        wpsi = jnp.asarray(wall_psi, dtype=dtype)
    out = traced_boundary_read(
        jnp.asarray(psi2d, dtype=dtype),
        jnp.asarray(rg_np, dtype=geometry_dtype),
        jnp.asarray(zg_np, dtype=geometry_dtype),
        jnp.asarray(inside),
        jnp.asarray(axis[0], dtype=geometry_dtype),
        jnp.asarray(axis[1], dtype=geometry_dtype),
        int(n_levels),
        int(n_bisect),
        int(n_ray),
        jnp.asarray(angles, dtype=geometry_dtype),
        jnp.asarray(lcfs_norm, dtype=dtype),
        jnp.asarray(wall_r, dtype=geometry_dtype),
        jnp.asarray(wall_z, dtype=geometry_dtype),
        wpsi,
    )
    return ConnectivityBoundary(
        found=bool(out["found"]),
        psi_bnd=float(out["psi_bnd"]),
        psi_lcfs=float(out["psi_lcfs"]),
        psi_axis=float(out["psi_axis"]),
        radii=np.asarray(out["radii"], dtype=np.float64),
        s_star=float(out["s_star"]),
        n_core_cells=int(out["n_core_cells"]),
        axis=(float(out["axis_r"]), float(out["axis_z"])),
        axis_psi=float(out["axis_psi_sub"]),
        xset=np.asarray(out["xset"], dtype=np.float64),
        is_diverted=bool(out["is_diverted"]),
        class_margin=float(out["class_margin"]),
        axis_state=int(out["axis_state"]),
        axis_confidence=float(out["axis_confidence"]),
        axis_candidate_count=int(out["axis_candidate_count"]),
        axis_overflow=bool(out["axis_overflow"]),
        x_candidate_count=int(out["x_candidate_count"]),
        x_overflow=bool(out["x_overflow"]),
        x_discarded_score_upper_bound=float(out["x_discarded_score_upper_bound"]),
        x_unresolved_count=int(out["x_unresolved_count"]),
        xset_state=np.asarray(out["xset_state"], dtype=np.int8),
        boundary_resolved=bool(out["boundary_resolved"]),
        x_binding_state=int(out["x_binding_state"]),
    )


def host_boundary_read_smooth(
    psi2d,
    grid,
    axis: tuple[float, float],
    *,
    temperature: float = 0.001,
    n_levels: int = 96,
    n_bisect: int = 18,
    n_ray: int = 512,
    angles=LCFS_ANGLES,
    lcfs_norm: float = 0.999,
    wall_psi=None,
    precision: Precision | str = Precision.AUTOMATIC,
) -> dict:
    """Host adapter: the smooth read at the stencil axis, on one slice (numpy out).

    Same contract as :func:`host_boundary_read` with the softmin/sigmoid smooth read
    (:func:`_smooth_read_at_stencil_axis`): the sub-grid stencil O-point is read
    first (``axis`` is the flood seed / fallback only) and the smooth read is
    seeded at it, exactly the solve-map wiring the accelerator probe measured.
    ``temperature`` is the smoothing scale τ in normalised-flux span units (the
    read's gate-calibrated accuracy point is τ=10⁻³).  Returns the smooth read's
    dict with numpy values (``core_weight`` is the ``(nz, nr)`` smooth core
    mask; ``axis_r``/``axis_z``/``axis_psi_sub`` the stencil axis, NaN when no
    in-wall O-point exists).
    """
    import numpy as np  # noqa: PLC0415

    resolved = resolve_precision(precision, Precision.DOUBLE)
    dtype = jnp.float32 if resolved is Precision.SINGLE else jnp.float64
    geometry_dtype = jnp.float64
    wall_r, wall_z = _densify_wall(grid)
    if wall_psi is None:
        wpsi = jnp.asarray([jnp.nan], dtype=dtype)
    else:
        wpsi = jnp.asarray(wall_psi, dtype=dtype)
    out = _smooth_read_at_stencil_axis(
        jnp.asarray(psi2d, dtype=dtype),
        jnp.asarray(grid.rg, dtype=geometry_dtype),
        jnp.asarray(grid.zg, dtype=geometry_dtype),
        jnp.asarray(np.asarray(grid.inside_limiter, dtype=bool)),
        jnp.asarray(axis[0], dtype=geometry_dtype),
        jnp.asarray(axis[1], dtype=geometry_dtype),
        int(n_levels),
        int(n_bisect),
        int(n_ray),
        jnp.asarray(angles, dtype=geometry_dtype),
        jnp.asarray(lcfs_norm, dtype=dtype),
        jnp.asarray(wall_r, dtype=geometry_dtype),
        jnp.asarray(wall_z, dtype=geometry_dtype),
        wpsi,
        jnp.asarray(temperature, dtype=dtype),
    )
    return {k: np.asarray(v) for k, v in out.items()}


def host_boundary_read_batch(
    psi_stack,
    grid,
    axes,
    *,
    n_levels: int = 96,
    n_bisect: int = 18,
    n_ray: int = 512,
    angles=LCFS_ANGLES,
    lcfs_norm: float = 0.999,
    wall_psi=None,
    precision: Precision | str = Precision.AUTOMATIC,
) -> dict:
    """Batched read over ``(B, nz, nr)`` ψ fields sharing one grid — a single vmap.

    ``axes`` is ``(B, 2)`` (R, Z).  ``wall_psi`` is an optional ``(B, n_node)``
    exact node flux (per-slice ``grid.wall_flux``) aligned with the grid's wall
    nodes; omit it for the bilinear read.  Proves the fixed-shape / on-device
    batch the corpus labeller needs: one ``jax.vmap``, no host loop, no per-slice
    contour extraction.  Returns a dict of stacked device arrays.
    """
    import numpy as np  # noqa: PLC0415

    resolved = resolve_precision(precision, Precision.DOUBLE)
    dtype = jnp.float32 if resolved is Precision.SINGLE else jnp.float64
    geometry_dtype = jnp.float64
    rg = jnp.asarray(grid.rg, dtype=geometry_dtype)
    zg = jnp.asarray(grid.zg, dtype=geometry_dtype)
    inside = jnp.asarray(np.asarray(grid.inside_limiter, dtype=bool))
    ang = jnp.asarray(angles, dtype=geometry_dtype)
    ps = jnp.asarray(psi_stack, dtype=dtype)
    ax = jnp.asarray(axes, dtype=dtype)
    wall_r, wall_z = _densify_wall(grid)
    wr = jnp.asarray(wall_r, dtype=geometry_dtype)
    wz = jnp.asarray(wall_z, dtype=geometry_dtype)
    if wall_psi is None:
        wpsi = jnp.full((ps.shape[0], 1), jnp.nan, dtype=dtype)
    else:
        wpsi = jnp.asarray(wall_psi, dtype=dtype)

    def one(psi2d, axis, wp):
        return traced_boundary_read(
            psi2d,
            rg,
            zg,
            inside,
            axis[0],
            axis[1],
            int(n_levels),
            int(n_bisect),
            int(n_ray),
            ang,
            jnp.asarray(lcfs_norm, dtype=dtype),
            wr,
            wz,
            wp,
        )

    return jax.vmap(one)(ps, ax, wpsi)
