r"""Typed per-frame steering records and their netCDF session store.

Playback phase: a steering run re-solves the forward equilibrium as the
control parameters move and pushes each resulting state to the page.  The
recorded session is the sequence of those states with the action that produced
each one attached, so a downstream decoder trains on exactly what the run
emitted and never re-derives topology from ``psi``.  This module defines the
per-frame record, assembles it from the solve receipt the forward path already
returns, and writes/reads a session as a group-backed xarray store with time
as the last axis — no re-derivation: every channel is a typed read of a
carrier the solve published.

A frame is assembled, never computed: the raster channels come from the
rectangular current image (``ForwardRasterFlux``), the labelled points from
the consumer map (``ForwardLabelledFlux``), the request identity and keyframe
wall from the solve receipt (``ForwardSolveReceipt``), and the driving currents
and recorded action from the steering context that produced the frame.

The point fields use the imas-ambix corpus key vocabulary — ``magnetic_axis_r``,
``magnetic_axis_z``, ``x_point_r``, ``x_point_z``, ``lcfs_r``, ``lcfs_z``,
``n_boundary_coords``, ``finite_mask`` — so the decoder consumes the same names
it reads from the equilibrium store.  Masking is never imputation: an
absent component is NaN in the coordinate channels and False in
``finite_mask``; no fill value is substituted.

Coordinate convention
---------------------
All coordinates are cylindrical ``(R, phi, Z)`` in COCOS 17 with ``Z`` upward
and ``phi`` measured anticlockwise from ``+R`` viewed from above.  ``psi`` is
the poloidal flux per radian of toroidal angle; its sign follows the Nova
observation convention (NOVA_COCOS = 17).  ``psi_norm`` is dimensionless,
``(psi - psi_axis) / (psi_LCFS - psi_axis)`` with the boundary at unity.

X-point ordering is NOT order-invariant: slot 0 always holds the primary
X-point the topology read selected and slot 1 the secondary one, with NaN in
the absent slot.  The LCFS polyline is NaN-padded to a fixed per-session
capacity; ``n_boundary_coords`` is the valid vertex count.

Frame schema (session shapes carry time as the last axis; per-frame shapes
drop the trailing ``time`` entry; ``n_r``/``n_z`` are the raster axes,
``n_s`` the traced separatrix capacity, ``n_b`` the LCFS capacity,
``n_circuits`` the driven circuit count, ``n_rows`` the registered constraint
row count, ``n_cp`` the commanded control-point count, ``nt`` the frame
count).  All coordinates are COCOS 17 ``(R, phi, Z)`` metres; ``psi`` is Wb
per radian and ``psi_norm`` dimensionless.

+--------------------------+-----------------+---------+--------------------+
| field                    | session shape   | dtype   | units              |
+==========================+=================+=========+====================+
| time                     | (nt,)           | float64 | s                  |
| radius                   | (n_r,)          | float64 | m                  |
| height                   | (n_z,)          | float64 | m                  |
| shape                    | (2,)            | int32   | grid cells         |
| psi                      | (n_r, n_z, nt)  | float64 | Wb (per radian)    |
| psi_norm                 | (n_r, n_z, nt)  | float64 | dimensionless      |
| domain_label             | (n_r, n_z, nt)  | int8    | ForwardDomainLabel |
| separatrix               | (n_s, 2, nt)    | float64 | m (NaN-padded)     |
| separatrix_vertex_count  | (nt,)           | int32   | vertices           |
| magnetic_axis_r          | (nt,)           | float64 | m                  |
| magnetic_axis_z          | (nt,)           | float64 | m                  |
| x_point_r                | (2, nt)         | float64 | m (primary first)  |
| x_point_z                | (2, nt)         | float64 | m (primary first)  |
| strike_points_r          | (2, nt)         | float64 | m (in/outboard)    |
| strike_points_z          | (2, nt)         | float64 | m (in/outboard)    |
| lcfs_r                   | (n_b, nt)       | float64 | m (NaN-padded)     |
| lcfs_z                   | (n_b, nt)       | float64 | m (NaN-padded)     |
| n_boundary_coords        | (nt,)           | int32   | vertices           |
| finite_mask              | (6, nt)         | bool    | component present  |
| coil_current             | (n_circuits,nt) | float64 | A                  |
| compensating_current     | (n_rows, nt)    | float64 | A                  |
| action_delta             | (nt,)           | float64 | parameter units    |
| action_name              | (nt,)           | str     | parameter name     |
| commanded_control_points | (n_cp, 2, nt)   | float64 | m                  |
| wall_seconds             | (nt,)           | float64 | s                  |
| trip_count               | (nt,)           | int32   | trips              |
| carrier_identity         | (nt,)           | str     | response carrier   |
| nova_version             | (nt,)           | str     | package version    |
| policy_digest            | (nt,)           | str     | policy sha256      |
+--------------------------+-----------------+---------+--------------------+
"""

from __future__ import annotations

import hashlib
import json
from typing import NamedTuple, Sequence

import numpy as np
import xarray as xr

from nova.database.netcdf import netCDF
from nova.biot.contour import Contour
from nova.equilibrium.labels import N_XPOINT_SLOTS
from nova.equilibrium.flux_surface_geometry import PlasmaInternalGeometry
from nova.equilibrium.wall_mask import inside_polygon
from nova.equilibrium.solve_request import (
    ForwardSolvePolicy,
    ForwardSolveReceipt,
    ResolvedForwardSolveDefaults,
)

#: Toroidal coordinate system of every geometric field (Nova's observation
#: convention; see the module docstring).
COCOS = 17

#: Number of strike-point slots (inboard then outboard leg-to-wall crossings).
N_STRIKE_POINTS = 2

#: Default netCDF group a recorded steering session is written under.
SESSION_GROUP = "steering"

#: Ordered point components ``finite_mask`` labels, one flag per component.
#: A component is present (finite coordinate values and a True mask flag) or
#: absent (NaN coordinate values and a False flag); values are never imputed.
FINITE_MASK_COMPONENTS: tuple[str, ...] = (
    "magnetic_axis",
    "x_point_0",
    "x_point_1",
    "strike_0",
    "strike_1",
    "lcfs",
)

#: Session attribute naming the declared coordinate system.
COCOS_ATTR = "cocos"

# Fixed-shape topology blocks consumed by the decoder and transport reader.
N_SURFACE = 11
N_THETA = 64
N_RHO = 25
N_DIVERTOR_LEGS = 4
N_DIVERTOR_LEG_POINTS = 32

TORAX_PROFILE_FIELDS: tuple[str, ...] = (
    "rho_face_norm",
    "rho_tor",
    "Phi",
    "psi_face",
    "Ip_profile",
    "R_in",
    "R_out",
    "F",
    "int_dl_over_Bp",
    "inv_R",
    "inv_R2",
    "grad_psi",
    "grad_psi2",
    "grad_psi2_over_R2",
    "B2",
    "inv_B2",
    "delta_upper",
    "delta_lower",
    "elongation",
    "vpr",
    "volume",
    "area",
    "q",
    "g0",
    "g1",
    "g2",
    "g3",
    "psi_norm_face",
)


class SteeringAction(NamedTuple):
    """One control-parameter move that produced a frame."""

    name: str
    delta: float
    commanded_control_points: object


#: Field registry table for the docstring contract: (key, session shape with
#: ``nt`` for time, dtype, units).  Per-frame shapes drop the trailing ``nt``;
#: the module docstring expands this into the full schema table.
_FIELD_TABLE: tuple[tuple[str, tuple[str, ...], str, str], ...] = (
    ("time", ("nt",), "float64", "s"),
    ("radius", ("n_r",), "float64", "m"),
    ("height", ("n_z",), "float64", "m"),
    ("shape", ("2",), "int32", "grid cells"),
    ("psi", ("n_r", "n_z", "nt"), "float64", "Wb (per radian)"),
    ("psi_norm", ("n_r", "n_z", "nt"), "float64", "dimensionless"),
    ("domain_label", ("n_r", "n_z", "nt"), "int8", "ForwardDomainLabel"),
    ("separatrix", ("n_s", "2", "nt"), "float64", "m"),
    ("separatrix_vertex_count", ("nt",), "int32", "vertices"),
    ("magnetic_axis_r", ("nt",), "float64", "m"),
    ("magnetic_axis_z", ("nt",), "float64", "m"),
    ("x_point_r", ("2", "nt"), "float64", "m"),
    ("x_point_z", ("2", "nt"), "float64", "m"),
    ("strike_points_r", ("2", "nt"), "float64", "m"),
    ("strike_points_z", ("2", "nt"), "float64", "m"),
    ("lcfs_r", ("n_b", "nt"), "float64", "m"),
    ("lcfs_z", ("n_b", "nt"), "float64", "m"),
    ("n_boundary_coords", ("nt",), "int32", "vertices"),
    ("finite_mask", ("6", "nt"), "bool", "component present"),
    ("coil_current", ("n_circuits", "nt"), "float64", "A"),
    ("compensating_current", ("n_rows", "nt"), "float64", "A"),
    ("action_delta", ("nt",), "float64", "parameter units"),
    ("action_name", ("nt",), "str", "parameter name"),
    ("commanded_control_points", ("n_cp", "2", "nt"), "float64", "m"),
    ("wall_seconds", ("nt",), "float64", "s"),
    ("trip_count", ("nt",), "int32", "trips"),
    ("carrier_identity", ("nt",), "str", "response carrier"),
    ("nova_version", ("nt",), "str", "package version"),
    ("policy_digest", ("nt",), "str", "sha256 of resolved policy"),
)


class SteeringFrame(NamedTuple):
    """One solved steering state and the action that produced it.

    Shapes are per-frame: ``psi``/``psi_norm``/``domain_label`` span the
    ``(radius, height)`` raster grid, ``separatrix`` is the traced raster
    contour packed to a fixed ``(n_s, 2)`` polyline, ``lcfs_*`` the LCFS
    packed to ``(n_b, 2)``, ``coil_current`` one entry per driven circuit,
    ``compensating_current`` one entry per registered constraint row, and
    ``finite_mask`` one flag per component in :data:`FINITE_MASK_COMPONENTS`.
    """

    radius: object
    height: object
    shape: object
    psi: object
    psi_norm: object
    domain_label: object
    separatrix: object
    separatrix_vertex_count: object
    magnetic_axis_r: object
    magnetic_axis_z: object
    x_point_r: object
    x_point_z: object
    strike_points_r: object
    strike_points_z: object
    lcfs_r: object
    lcfs_z: object
    n_boundary_coords: object
    finite_mask: object
    coil_current: object
    compensating_current: object
    action: SteeringAction
    wall_seconds: float
    trip_count: int
    carrier_identity: str
    nova_version: str
    policy_digest: str
    flux_surface_psi_norm: object
    flux_surface_psi: object
    flux_surface_r: object
    flux_surface_z: object
    flux_surface_angle: object
    rho_face_norm: object
    rho_tor: object
    Phi: object
    psi_face: object
    Ip_profile: object
    R_in: object
    R_out: object
    F: object
    int_dl_over_Bp: object
    inv_R: object
    inv_R2: object
    grad_psi: object
    grad_psi2: object
    grad_psi2_over_R2: object
    B2: object
    inv_B2: object
    delta_upper: object
    delta_lower: object
    elongation: object
    vpr: object
    volume: object
    area: object
    q: object
    g0: object
    g1: object
    g2: object
    g3: object
    psi_norm_face: object
    R_major: float
    a_minor: float
    B_0: float
    boundary_toroidal_flux: float
    magnetic_axis_z_scalar: float
    diverted: bool
    divertor_leg_r: object
    divertor_leg_z: object
    divertor_leg_finite: object


def _as_numpy(value) -> np.ndarray:
    """Return the value as a plain numpy array."""
    return np.asarray(value)


def _point_absent(point) -> bool:
    """Return whether a coordinate pair is fully absent (NaN)."""
    return not bool(np.all(np.isfinite(point)))


def policy_digest(policy: ForwardSolvePolicy) -> str:
    """Return a stable hex digest naming one resolved forward solve policy.

    The digest hashes the JSON-native policy block with keys sorted, so two
    policies that differ only in insertion order or printed form collide only
    if their resolved choices are identical.
    """
    payload = json.dumps(
        policy.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _mask_components(
    axis,
    primary_x_point,
    secondary_x_point,
    strike_points,
    n_boundary_coords: int,
) -> np.ndarray:
    """Return the per-component presence mask for one frame's point slots."""
    strike = _as_numpy(strike_points)
    components = (
        not _point_absent(axis),
        not _point_absent(primary_x_point),
        not _point_absent(secondary_x_point),
        not _point_absent(strike[0]),
        not _point_absent(strike[1]),
        int(n_boundary_coords) > 0,
    )
    return np.asarray(components, dtype=bool)


def _compensating_current(eq) -> np.ndarray:
    """Return the per-row physical compensation a solve's records name.

    Rows appear in the same order the constraint pairs were registered, each
    record contributing one entry per residual row.  A solve with no
    constraint rows yields an empty channel.
    """
    rows = [
        np.ravel(_as_numpy(record.physical_unknown))
        for record in getattr(eq, "constraints", ())
    ]
    if not rows:
        return np.empty((0,), dtype=np.float64)
    return np.concatenate(rows)


def _empty_geometry_channels() -> dict[str, object]:
    """Return masked fixed-shape geometry for a receipt without a producer."""
    surface_psi_norm = np.linspace(0.0, 1.0, N_SURFACE)
    angle = np.linspace(0.0, 2.0 * np.pi, N_THETA, endpoint=False)
    return {
        "flux_surface_psi_norm": surface_psi_norm,
        "flux_surface_psi": np.full(N_SURFACE, np.nan),
        "flux_surface_r": np.full((N_SURFACE, N_THETA), np.nan),
        "flux_surface_z": np.full((N_SURFACE, N_THETA), np.nan),
        "flux_surface_angle": angle,
        **{
            name: np.full(N_RHO + 1, np.nan)
            for name in TORAX_PROFILE_FIELDS
            if name != "rho_face_norm"
        },
        "rho_face_norm": np.linspace(0.0, 1.0, N_RHO + 1),
        "R_major": np.nan,
        "a_minor": np.nan,
        "B_0": np.nan,
        "boundary_toroidal_flux": np.nan,
        "magnetic_axis_z_scalar": np.nan,
        "diverted": False,
        "divertor_leg_r": np.full((N_DIVERTOR_LEGS, N_DIVERTOR_LEG_POINTS), np.nan),
        "divertor_leg_z": np.full((N_DIVERTOR_LEGS, N_DIVERTOR_LEG_POINTS), np.nan),
        "divertor_leg_finite": np.zeros(N_DIVERTOR_LEGS, dtype=bool),
    }


def _geometry_channels(
    geometry: PlasmaInternalGeometry | None,
    *,
    divertor_leg_r=None,
    divertor_leg_z=None,
    divertor_leg_finite=None,
) -> dict[str, object]:
    """Flatten the ray-traced producer record into frame channel names."""
    if geometry is None:
        channels = _empty_geometry_channels()
    else:
        record = geometry.record
        dvolume_dpsi = np.abs(np.asarray(record.volume_flux_derivative))
        channels = {
            "flux_surface_psi_norm": np.asarray(geometry.surface_psi_norm),
            "flux_surface_psi": np.asarray(geometry.surface_psi),
            "flux_surface_r": np.asarray(geometry.surface_r),
            "flux_surface_z": np.asarray(geometry.surface_z),
            "flux_surface_angle": np.asarray(geometry.surface_angle),
            "rho_face_norm": np.asarray(record.rho_tor_norm),
            "rho_tor": np.asarray(record.rho_tor),
            "Phi": np.asarray(record.toroidal_flux),
            "psi_face": np.asarray(record.poloidal_flux),
            "Ip_profile": np.asarray(record.enclosed_toroidal_current),
            "R_in": np.asarray(record.r_in),
            "R_out": np.asarray(record.r_out),
            "F": np.asarray(record.field_function),
            "int_dl_over_Bp": np.asarray(record.int_dl_over_bp),
            "inv_R": np.asarray(record.inverse_radius),
            "inv_R2": np.asarray(record.inverse_square_radius),
            "grad_psi": np.asarray(record.gradient_psi),
            "grad_psi2": np.asarray(record.gradient_psi_squared),
            "grad_psi2_over_R2": np.asarray(
                record.gradient_psi_squared_over_radius_squared
            ),
            "B2": np.asarray(record.field_squared),
            "inv_B2": np.asarray(record.inverse_field_squared),
            "delta_upper": np.asarray(record.triangularity_upper),
            "delta_lower": np.asarray(record.triangularity_lower),
            "elongation": np.asarray(record.elongation),
            "vpr": np.asarray(record.volume_derivative),
            "volume": np.asarray(record.volume),
            "area": np.asarray(record.area),
            "q": np.asarray(record.safety_factor),
            "g0": np.asarray(record.gradient_psi) * dvolume_dpsi,
            "g1": np.asarray(record.gradient_psi_squared) * dvolume_dpsi**2,
            "g2": np.asarray(record.gradient_psi_squared_over_radius_squared)
            * dvolume_dpsi**2,
            "g3": np.asarray(record.inverse_square_radius),
            "psi_norm_face": np.asarray(record.psi_norm),
            "R_major": float(geometry.r_major),
            "a_minor": float(geometry.a_minor),
            "B_0": float(geometry.b0),
            "boundary_toroidal_flux": float(geometry.boundary_toroidal_flux),
            "magnetic_axis_z_scalar": np.nan,
            "diverted": bool(geometry.diverted),
            "divertor_leg_r": np.full((N_DIVERTOR_LEGS, N_DIVERTOR_LEG_POINTS), np.nan),
            "divertor_leg_z": np.full((N_DIVERTOR_LEGS, N_DIVERTOR_LEG_POINTS), np.nan),
            "divertor_leg_finite": np.zeros(N_DIVERTOR_LEGS, dtype=bool),
        }
    if divertor_leg_r is not None:
        channels["divertor_leg_r"] = np.asarray(divertor_leg_r, dtype=float)
    if divertor_leg_z is not None:
        channels["divertor_leg_z"] = np.asarray(divertor_leg_z, dtype=float)
    if divertor_leg_finite is not None:
        channels["divertor_leg_finite"] = np.asarray(divertor_leg_finite, dtype=bool)
    channels["magnetic_axis_z_scalar"] = channels.get("magnetic_axis_z_scalar", np.nan)
    return channels


def _resample_leg(points: np.ndarray, count: int) -> np.ndarray | None:
    """Resample one finite wall-clipped branch by arclength."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[0] < 2 or not np.all(np.isfinite(points)):
        return None
    distance = np.linalg.norm(np.diff(points, axis=0), axis=1)
    coordinate = np.concatenate(([0.0], np.cumsum(distance)))
    if coordinate[-1] <= 0.0:
        return None
    target = np.linspace(0.0, coordinate[-1], count)
    return np.column_stack(
        [np.interp(target, coordinate, points[:, axis]) for axis in range(2)]
    )


def _clip_leg(points: np.ndarray, wall: np.ndarray, x_point: np.ndarray):
    """Keep the branch from an X-point through its first wall crossing."""
    points = np.asarray(points, dtype=float)
    if points.shape[0] < 2:
        return None
    if np.linalg.norm(points[-1] - x_point) < np.linalg.norm(points[0] - x_point):
        points = points[::-1]
    inside = inside_polygon(points[:, 0], points[:, 1], wall[:, 0], wall[:, 1])
    if not bool(inside[0]):
        return None
    crossing = None
    for index in range(1, points.shape[0]):
        if bool(inside[index]):
            continue
        left, right = points[index - 1], points[index]
        for _ in range(24):
            middle = 0.5 * (left + right)
            if bool(inside_polygon(middle[0], middle[1], wall[:, 0], wall[:, 1])):
                left = middle
            else:
                right = middle
        crossing = left
        points = np.vstack((points[:index], crossing))
        break
    if crossing is None:
        return None
    return points


def _trace_divertor_legs(
    raster,
    boundary_flux: float,
    x_points: np.ndarray,
    wall,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trace separatrix branches and clip them at the supplied first wall."""
    result_r = np.full((N_DIVERTOR_LEGS, N_DIVERTOR_LEG_POINTS), np.nan)
    result_z = np.full((N_DIVERTOR_LEGS, N_DIVERTOR_LEG_POINTS), np.nan)
    finite = np.zeros(N_DIVERTOR_LEGS, dtype=bool)
    wall = np.asarray(wall, dtype=float)
    if wall.ndim != 2 or wall.shape[1] != 2 or wall.shape[0] < 3:
        return result_r, result_z, finite
    shape = tuple(np.asarray(raster.shape, dtype=int))
    values = np.asarray(raster.psi).reshape(shape).T
    contours = Contour(
        np.asarray(raster.radius),
        np.asarray(raster.height),
        values,
        levels=np.asarray([boundary_flux]),
    ).levelset(boundary_flux)
    open_lines = [surface.points for surface in contours if not surface.closed]
    for slot in range(N_XPOINT_SLOTS):
        point = np.asarray(x_points[slot], dtype=float)
        if not np.all(np.isfinite(point)):
            continue
        candidates = []
        for line in open_lines:
            nearest = np.min(np.linalg.norm(line - point, axis=1))
            if nearest <= 3.0 * max(
                np.diff(np.asarray(raster.radius)).min(),
                np.diff(np.asarray(raster.height)).min(),
            ):
                clipped = _clip_leg(line, wall, point)
                if clipped is not None:
                    candidates.append(clipped)
        radial_tests = (lambda value: value < point[0], lambda value: value >= point[0])
        for side, radial_test in enumerate(radial_tests):
            branches = [line for line in candidates if radial_test(line[-1, 0])]
            if not branches:
                continue
            branch = max(branches, key=lambda line: line.shape[0])
            sampled = _resample_leg(branch, N_DIVERTOR_LEG_POINTS)
            if sampled is None:
                continue
            index = 2 * slot + side
            result_r[index], result_z[index] = sampled.T
            finite[index] = True
    return result_r, result_z, finite


def assemble_frame(
    receipt: ForwardSolveReceipt,
    *,
    action: SteeringAction,
    carrier_identity: str,
    applied_current,
    compensating_current=None,
    internal_geometry: PlasmaInternalGeometry | None = None,
    wall=None,
    divertor_leg_r=None,
    divertor_leg_z=None,
    divertor_leg_finite=None,
) -> SteeringFrame:
    """Assemble one typed frame from one solved forward receipt.

    Every channel is a typed read of a carrier the receipt's terminal state
    publishes: the raster image for the grid channels, the labelled consumer
    map for the point fields and topology, the fixed-point history for the
    trip count, and the resolved defaults for the version and policy digest.
    ``applied_current`` names the coil currents the run drove this frame;
    ``compensating_current`` names the per-row physical compensation and, when
    omitted, is read from the terminal constraint records of the receipt.
    """
    if not isinstance(receipt, ForwardSolveReceipt):
        raise TypeError("a steering frame is assembled from a ForwardSolveReceipt")
    equilibrium = receipt.terminal_state
    raster = equilibrium.raster_flux
    labelled = equilibrium.labelled_flux
    if raster is None or labelled is None:
        raise ValueError("a steering frame needs raster and labelled flux receipts")

    radial_count, vertical_count = tuple(int(slot) for slot in np.asarray(raster.shape))
    grid_shape = (radial_count, vertical_count)
    axis = _as_numpy(labelled.o_point)
    primary = _as_numpy(labelled.primary_x_point)
    secondary = _as_numpy(labelled.secondary_x_point)
    expected_slots = (N_XPOINT_SLOTS,)
    if primary.shape != expected_slots or secondary.shape != expected_slots:
        raise ValueError("X-point slots must each be an R-Z coordinate pair")
    strike = _as_numpy(labelled.strike_points)
    lcfs = _as_numpy(labelled.lcfs)
    if lcfs.ndim != 2 or lcfs.shape[1] != 2:
        raise ValueError("the LCFS polyline must be packed as (vertex, R-Z)")
    boundary_count = int(np.asarray(labelled.lcfs_vertex_count))

    if compensating_current is None:
        compensating_current = _compensating_current(equilibrium)
    compensating = np.asarray(compensating_current, dtype=np.float64)

    resolved = receipt.resolved_defaults
    if not isinstance(resolved, ResolvedForwardSolveDefaults):
        raise TypeError("a steering frame needs resolved forward-solve defaults")
    trips = int(np.asarray(equilibrium.fixed_point.active_set_iterations))
    geometry = _geometry_channels(
        internal_geometry,
        divertor_leg_r=divertor_leg_r,
        divertor_leg_z=divertor_leg_z,
        divertor_leg_finite=divertor_leg_finite,
    )
    geometry["magnetic_axis_z_scalar"] = float(axis[1])
    if (
        wall is not None
        and internal_geometry is not None
        and internal_geometry.diverted
        and divertor_leg_finite is None
    ):
        leg_r, leg_z, leg_finite = _trace_divertor_legs(
            raster,
            float(np.asarray(equilibrium.topology.boundary_flux)),
            np.stack((primary, secondary)),
            wall,
        )
        geometry["divertor_leg_r"] = leg_r
        geometry["divertor_leg_z"] = leg_z
        geometry["divertor_leg_finite"] = leg_finite

    return SteeringFrame(
        radius=_as_numpy(raster.radius),
        height=_as_numpy(raster.height),
        shape=np.asarray(raster.shape, dtype=np.int32),
        psi=_as_numpy(raster.psi).reshape(grid_shape),
        psi_norm=_as_numpy(raster.psi_norm).reshape(grid_shape),
        domain_label=_as_numpy(raster.domain_label).reshape(grid_shape),
        separatrix=_as_numpy(raster.separatrix),
        separatrix_vertex_count=np.int32(raster.separatrix_vertex_count),
        magnetic_axis_r=axis[0],
        magnetic_axis_z=axis[1],
        x_point_r=np.stack((primary[0], secondary[0])),
        x_point_z=np.stack((primary[1], secondary[1])),
        strike_points_r=strike[:, 0],
        strike_points_z=strike[:, 1],
        lcfs_r=lcfs[:, 0],
        lcfs_z=lcfs[:, 1],
        n_boundary_coords=np.int32(boundary_count),
        finite_mask=_mask_components(axis, primary, secondary, strike, boundary_count),
        coil_current=np.asarray(applied_current, dtype=np.float64),
        compensating_current=compensating,
        action=action,
        wall_seconds=float(receipt.wall_seconds),
        trip_count=trips,
        carrier_identity=carrier_identity,
        nova_version=resolved.nova_version,
        policy_digest=policy_digest(resolved.policy),
        **geometry,
    )


def _frame_stack(frames: Sequence[SteeringFrame], name: str) -> np.ndarray:
    """Return one field stacked over frames with time as the last axis."""
    return np.stack([_as_numpy(getattr(frame, name)) for frame in frames], axis=-1)


def _assert_homogeneous(frames: Sequence[SteeringFrame]) -> None:
    """Require every frame to share the fixed per-session axis capacities."""
    first = frames[0]
    capacities = {
        "radius": _as_numpy(first.radius).size,
        "height": _as_numpy(first.height).size,
        "separatrix": int(_as_numpy(first.separatrix).shape[0]),
        "lcfs": int(_as_numpy(first.lcfs_r).size),
        "circuits": int(_as_numpy(first.coil_current).size),
        "rows": int(_as_numpy(first.compensating_current).size),
        "control_points": _as_numpy(first.action.commanded_control_points).shape[0],
        "surface": _as_numpy(first.flux_surface_psi_norm).size,
        "theta": _as_numpy(first.flux_surface_angle).size,
        "face": _as_numpy(first.rho_face_norm).size,
        "leg": _as_numpy(first.divertor_leg_r).shape[0],
        "leg_points": _as_numpy(first.divertor_leg_r).shape[1],
    }
    for frame in frames[1:]:
        for key, expected in capacities.items():
            if key == "radius":
                actual = _as_numpy(frame.radius).size
            elif key == "height":
                actual = _as_numpy(frame.height).size
            elif key == "separatrix":
                actual = int(_as_numpy(frame.separatrix).shape[0])
            elif key == "lcfs":
                actual = int(_as_numpy(frame.lcfs_r).size)
            elif key == "circuits":
                actual = int(_as_numpy(frame.coil_current).size)
            elif key == "rows":
                actual = int(_as_numpy(frame.compensating_current).size)
            else:
                actual = _as_numpy(frame.action.commanded_control_points).shape[0]
            if key == "surface":
                actual = _as_numpy(frame.flux_surface_psi_norm).size
            elif key == "theta":
                actual = _as_numpy(frame.flux_surface_angle).size
            elif key == "face":
                actual = _as_numpy(frame.rho_face_norm).size
            elif key == "leg":
                actual = _as_numpy(frame.divertor_leg_r).shape[0]
            elif key == "leg_points":
                actual = _as_numpy(frame.divertor_leg_r).shape[1]
            if actual != expected:
                raise ValueError(
                    f"steering frames must share {key} capacity, "
                    f"got {actual} and {expected}"
                )


def session_dataset(
    frames: Sequence[SteeringFrame],
    *,
    time=None,
    include_raster: bool = True,
) -> xr.Dataset:
    """Return one recording session as a time-last xarray dataset.

    All per-frame channels are stacked with time as the last axis.  The raster
    axes ``radius`` and ``height`` are shared per frame and become dimension
    coordinates; the fixed slots (X-points, strike points, LCFS vertices,
    circuits, constraint rows, control points, mask components) are named
    axes.  ``time`` defaults to the integer frame index in seconds.
    """
    frames = tuple(frames)
    if not frames:
        raise ValueError("a steering session needs at least one frame")
    _assert_homogeneous(frames)
    count = len(frames)
    first = frames[0]
    time_values: np.ndarray
    if time is None:
        time_values = np.arange(count, dtype=np.float64)
    else:
        time_values = np.asarray(time, dtype=np.float64)
    if time_values.shape != (count,):
        raise ValueError("time must carry one entry per frame")
    if not all(frame.action.name for frame in frames):
        raise ValueError("every frame needs a named action")

    variables: dict[str, object] = {
        "radius": ("radius", _as_numpy(first.radius)),
        "height": ("height", _as_numpy(first.height)),
        "shape": ("shape", np.asarray(first.shape, dtype=np.int32)),
        "psi": (("radius", "height", "time"), _frame_stack(frames, "psi")),
        "psi_norm": (
            ("radius", "height", "time"),
            _frame_stack(frames, "psi_norm"),
        ),
        "domain_label": (
            ("radius", "height", "time"),
            _frame_stack(frames, "domain_label").astype(np.int8),
        ),
        "separatrix": (
            ("separatrix_vertex", "coordinate", "time"),
            _frame_stack(frames, "separatrix"),
        ),
        "separatrix_vertex_count": (
            ("time",),
            _frame_stack(frames, "separatrix_vertex_count").astype(np.int32),
        ),
        "magnetic_axis_r": ("time", _frame_stack(frames, "magnetic_axis_r")),
        "magnetic_axis_z": ("time", _frame_stack(frames, "magnetic_axis_z")),
        "x_point_r": (("x_slot", "time"), _frame_stack(frames, "x_point_r")),
        "x_point_z": (("x_slot", "time"), _frame_stack(frames, "x_point_z")),
        "strike_points_r": (
            ("strike_slot", "time"),
            _frame_stack(frames, "strike_points_r"),
        ),
        "strike_points_z": (
            ("strike_slot", "time"),
            _frame_stack(frames, "strike_points_z"),
        ),
        "lcfs_r": (
            ("boundary_vertex", "time"),
            _frame_stack(frames, "lcfs_r"),
        ),
        "lcfs_z": (
            ("boundary_vertex", "time"),
            _frame_stack(frames, "lcfs_z"),
        ),
        "n_boundary_coords": (
            "time",
            _frame_stack(frames, "n_boundary_coords").astype(np.int32),
        ),
        "finite_mask": (
            ("component", "time"),
            _frame_stack(frames, "finite_mask").astype(bool),
        ),
        "coil_current": (
            ("circuit", "time"),
            _frame_stack(frames, "coil_current"),
        ),
        "compensating_current": (
            ("constraint_row", "time"),
            _frame_stack(frames, "compensating_current"),
        ),
        "action_delta": (
            "time",
            np.asarray([frame.action.delta for frame in frames], dtype=np.float64),
        ),
        "action_name": (
            "time",
            np.asarray([frame.action.name for frame in frames], dtype=str),
        ),
        "commanded_control_points": (
            ("control_point", "coordinate", "time"),
            np.stack(
                [
                    np.asarray(frame.action.commanded_control_points, dtype=np.float64)
                    for frame in frames
                ],
                axis=-1,
            ),
        ),
        "wall_seconds": ("time", _frame_stack(frames, "wall_seconds")),
        "trip_count": ("time", _frame_stack(frames, "trip_count").astype(np.int32)),
        "carrier_identity": (
            "time",
            np.asarray([frame.carrier_identity for frame in frames], dtype=str),
        ),
        "nova_version": (
            "time",
            np.asarray([frame.nova_version for frame in frames], dtype=str),
        ),
        "policy_digest": (
            "time",
            np.asarray([frame.policy_digest for frame in frames], dtype=str),
        ),
    }
    variables.update(
        {
            "flux_surface_psi_norm": (
                ("n_surface",),
                np.asarray(first.flux_surface_psi_norm),
            ),
            "flux_surface_psi": (
                (("n_surface", "time")),
                _frame_stack(frames, "flux_surface_psi"),
            ),
            "flux_surface_r": (
                (("n_surface", "n_theta", "time")),
                _frame_stack(frames, "flux_surface_r"),
            ),
            "flux_surface_z": (
                (("n_surface", "n_theta", "time")),
                _frame_stack(frames, "flux_surface_z"),
            ),
            "flux_surface_angle": (
                (("n_theta",)),
                np.asarray(first.flux_surface_angle),
            ),
            **{
                name: (("n_face", "time"), _frame_stack(frames, name))
                for name in TORAX_PROFILE_FIELDS
                if name != "rho_face_norm"
            },
            "rho_face_norm": (("n_face",), np.asarray(first.rho_face_norm)),
            "R_major": ("time", _frame_stack(frames, "R_major")),
            "a_minor": ("time", _frame_stack(frames, "a_minor")),
            "B_0": ("time", _frame_stack(frames, "B_0")),
            "boundary_toroidal_flux": (
                "time",
                _frame_stack(frames, "boundary_toroidal_flux"),
            ),
            "magnetic_axis_z_scalar": (
                "time",
                _frame_stack(frames, "magnetic_axis_z_scalar"),
            ),
            "diverted": (
                "time",
                _frame_stack(frames, "diverted").astype(bool),
            ),
            "divertor_leg_r": (
                ("n_leg", "n_leg_point", "time"),
                _frame_stack(frames, "divertor_leg_r"),
            ),
            "divertor_leg_z": (
                ("n_leg", "n_leg_point", "time"),
                _frame_stack(frames, "divertor_leg_z"),
            ),
            "divertor_leg_finite": (
                ("n_leg", "time"),
                _frame_stack(frames, "divertor_leg_finite").astype(bool),
            ),
        }
    )
    if not include_raster:
        for name in (
            "radius",
            "height",
            "shape",
            "psi",
            "psi_norm",
            "domain_label",
            "separatrix",
            "separatrix_vertex_count",
        ):
            variables.pop(name, None)
    training_inputs = ",".join(
        (
            "flux_surface_psi_norm",
            "flux_surface_psi",
            "flux_surface_r",
            "flux_surface_z",
            "flux_surface_angle",
            "divertor_leg_r",
            "divertor_leg_z",
            "divertor_leg_finite",
            "magnetic_axis_r",
            "magnetic_axis_z",
            "x_point_r",
            "x_point_z",
            "strike_points_r",
            "strike_points_z",
            "lcfs_r",
            "lcfs_z",
            "coil_current",
            "compensating_current",
            *TORAX_PROFILE_FIELDS,
        )
    )
    diagnostic_only = ",".join(
        (
            "psi",
            "psi_norm",
            "domain_label",
            "separatrix",
            "separatrix_vertex_count",
        )
    )
    return xr.Dataset(
        variables,
        coords={"time": ("time", time_values)},
        attrs={
            COCOS_ATTR: COCOS,
            "training_inputs": training_inputs,
            "diagnostic_only": diagnostic_only,
        },
    )


def write_session(
    frames: Sequence[SteeringFrame],
    *,
    filename: str,
    dirname: str,
    group: str = SESSION_GROUP,
    time=None,
    include_raster: bool = True,
) -> netCDF:
    """Record one steering session through the group-backed netCDF store."""
    dataset = session_dataset(frames, time=time, include_raster=include_raster)
    store = netCDF(
        filename=filename,
        dirname=dirname,
        group=group,
        data=dataset,
    )
    return store.store()


def read_session(
    *,
    filename: str,
    dirname: str,
    group: str = SESSION_GROUP,
) -> xr.Dataset:
    """Return the decoded dataset of one recorded steering session."""
    store = netCDF(filename=filename, dirname=dirname, group=group)
    store.load()
    return store.data


def frames_from_session(dataset: xr.Dataset) -> list[SteeringFrame]:
    """Return the per-frame records a session dataset holds.

    Reconstructs the typed frames by slicing the time-last channels back onto
    the per-frame shapes, preserving NaN padding and component masks exactly.
    """
    count = int(dataset.sizes["time"])
    frames = []
    for index in range(count):
        frame = dataset.isel(time=index)
        commanded = np.asarray(frame["commanded_control_points"].values)
        frames.append(
            SteeringFrame(
                radius=np.asarray(frame["radius"].values),
                height=np.asarray(frame["height"].values),
                shape=np.asarray(frame["shape"].values),
                psi=np.asarray(frame["psi"].values),
                psi_norm=np.asarray(frame["psi_norm"].values),
                domain_label=np.asarray(frame["domain_label"].values),
                separatrix=np.asarray(frame["separatrix"].values),
                separatrix_vertex_count=np.asarray(
                    frame["separatrix_vertex_count"].values
                ),
                magnetic_axis_r=np.asarray(frame["magnetic_axis_r"].values),
                magnetic_axis_z=np.asarray(frame["magnetic_axis_z"].values),
                x_point_r=np.asarray(frame["x_point_r"].values),
                x_point_z=np.asarray(frame["x_point_z"].values),
                strike_points_r=np.asarray(frame["strike_points_r"].values),
                strike_points_z=np.asarray(frame["strike_points_z"].values),
                lcfs_r=np.asarray(frame["lcfs_r"].values),
                lcfs_z=np.asarray(frame["lcfs_z"].values),
                n_boundary_coords=int(np.asarray(frame["n_boundary_coords"].values)),
                finite_mask=np.asarray(frame["finite_mask"].values),
                coil_current=np.asarray(frame["coil_current"].values),
                compensating_current=np.asarray(frame["compensating_current"].values),
                action=SteeringAction(
                    name=str(frame["action_name"].values),
                    delta=float(frame["action_delta"].values),
                    commanded_control_points=commanded,
                ),
                wall_seconds=float(frame["wall_seconds"].values),
                trip_count=int(np.asarray(frame["trip_count"].values)),
                carrier_identity=str(frame["carrier_identity"].values),
                nova_version=str(frame["nova_version"].values),
                policy_digest=str(frame["policy_digest"].values),
                flux_surface_psi_norm=np.asarray(
                    dataset["flux_surface_psi_norm"].values
                ),
                flux_surface_psi=np.asarray(frame["flux_surface_psi"].values),
                flux_surface_r=np.asarray(frame["flux_surface_r"].values),
                flux_surface_z=np.asarray(frame["flux_surface_z"].values),
                flux_surface_angle=np.asarray(dataset["flux_surface_angle"].values),
                **{
                    name: np.asarray(frame[name].values)
                    for name in TORAX_PROFILE_FIELDS
                    if name != "rho_face_norm"
                },
                rho_face_norm=np.asarray(dataset["rho_face_norm"].values),
                R_major=float(frame["R_major"].values),
                a_minor=float(frame["a_minor"].values),
                B_0=float(frame["B_0"].values),
                boundary_toroidal_flux=float(frame["boundary_toroidal_flux"].values),
                magnetic_axis_z_scalar=float(frame["magnetic_axis_z_scalar"].values),
                diverted=bool(frame["diverted"].values),
                divertor_leg_r=np.asarray(frame["divertor_leg_r"].values),
                divertor_leg_z=np.asarray(frame["divertor_leg_z"].values),
                divertor_leg_finite=np.asarray(frame["divertor_leg_finite"].values),
            )
        )
    return frames


__all__ = [
    "COCOS",
    "FINITE_MASK_COMPONENTS",
    "N_STRIKE_POINTS",
    "SESSION_GROUP",
    "SteeringAction",
    "SteeringFrame",
    "assemble_frame",
    "frames_from_session",
    "policy_digest",
    "read_session",
    "session_dataset",
    "write_session",
]
