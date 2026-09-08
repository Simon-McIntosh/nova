"""Steering-frame schema, assembly, and recorded-session round-trip measures.

The machine is the same bootstrapped Solov'ev free-boundary problem the
forward-solve contract uses (see ``test_reduced_newton``), with its reachable
production-route terminal state wrapped in a real forward solve receipt; the
frame assembled from it must carry every decoder channel the module docstring
tabulates.  A synthetic three-frame session exercises the store: every channel
written through the group-backed netCDF store is bit-identical on read.
"""

from __future__ import annotations

from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr
from scipy.constants import mu_0
from scipy.interpolate import RegularGridInterpolator

from nova.utilities.importmanager import skip_import

with skip_import("jax"):
    import jax.numpy as jnp

    from nova.biot.greens import hybrid_greens
    from nova.equilibrium.conservation import FluxLattice
    from nova.equilibrium.forward import ForwardProfile
    from nova.equilibrium.solve_request import (
        ForwardSolveReceipt,
        ResolvedForwardSolveDefaults,
        resolve_forward_solve_policy,
    )
    from nova.equilibrium.observation import MomentIntegralSupport
    from nova.equilibrium.source import DomainProfile, ForwardSource
    from nova.equilibrium.flux_surface_geometry import (
        FluxSurfaceGeometry,
        source_field_function,
    )
    from nova.equilibrium.steering_frames import (
        COCOS,
        FINITE_MASK_COMPONENTS,
        WALL_GROUP,
        SteeringAction,
        SteeringFrame,
        WallReference,
        _current_centroid,
        assemble_frame,
        count_labelled_outside_wall,
        frames_from_session,
        policy_digest,
        read_session,
        session_dataset,
        wall_reference_from_session,
        write_session,
    )
    from nova.equilibrium.wall_mask import WallUnit, inside_polygon
    from nova.jax.config import configure_dtypes


P_PRIME = -3.0e5
FF_PRIME = -0.25
AXIS_RADIUS = 1.0
SEED_SPAN = 0.35
DRIVE = 1.4
BOUNDARY_FIELD_FUNCTION = 5.0
CONDUCTORS = 16
SOLVE_TOLERANCE = 1.0e-8
PRODUCTION_NEWTON_STEPS = 12
GMRES_ITERATIONS = 12

#: Fixed per-run identity the recorded fixture frame is assembled under.
CARRIER_IDENTITY = "solovev-fixture"

RASTER_FIELDS = (
    "radius",
    "height",
    "shape",
    "psi",
    "psi_norm",
    "domain_label",
    "separatrix",
    "separatrix_vertex_count",
)

CORPUS_SESSION = Path(
    "/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29/29547.nc"
)


def _terms():
    """Return the Solov'ev quartic, offset and vertical coefficients."""
    alpha = np.pi**2 * mu_0 * P_PRIME / 2.0
    return alpha, -2.0 * alpha * AXIS_RADIUS**2, 2.0 * np.pi**2 * FF_PRIME


def _solovev(radius, height):
    """Return the analytic seed flux [Wb] the conductors are fitted to."""
    alpha, offset, beta = _terms()
    return alpha * radius**4 + offset * radius**2 + beta * height**2


def _wall_loop(points=61):
    """Return a material boundary lying on one seed flux surface."""
    alpha, offset, beta = _terms()
    wall_flux = _solovev(AXIS_RADIUS, 0.0) - SEED_SPAN
    inner, outer = np.sqrt(np.sort(np.roots([alpha, offset, -wall_flux])))
    centre, half = 0.5 * (inner + outer), 0.5 * (outer - inner)
    angle = 2 * np.pi * np.arange(points) / points
    radius = centre + half * np.cos(angle)
    argument = np.clip((wall_flux - _solovev(radius, 0.0)) / beta, 0.0, None)
    return np.c_[radius, np.sign(np.sin(angle)) * np.sqrt(argument)], wall_flux


def _wall_reference(wall) -> WallReference:
    """Return a stable reference for the Solov'ev fixture's vessel unit."""

    return WallReference.from_units(
        machine="solovev",
        source="/synthetic/solovev-machine-description",
        source_kind="ids",
        dd_version="test",
        units=(
            WallUnit(
                r=np.asarray(wall)[:, 0],
                z=np.asarray(wall)[:, 1],
                kind="vessel",
                closed=True,
                name="solovev vessel",
            ),
        ),
    )


def _green_block(target, source, section=0.05):
    """Return the total-flux coupling [Wb/A] of one source set on one target."""
    return np.stack(
        [
            hybrid_greens(target[:, 0], target[:, 1], a, z, section, section)[0]
            for a, z in source
        ],
        axis=1,
    )


def _flat_profile(amplitude):
    """Return a constant absolute gradient."""

    def gradient(psi_norm):
        """Return the constant value at every normalised flux."""
        return jnp.full_like(jnp.asarray(psi_norm, dtype=jnp.float64), amplitude)

    return gradient


def _edge_vanishing_profile(amplitude):
    """Return an absolute gradient that falls linearly to zero at the edge."""

    def gradient(psi_norm):
        """Return the tapered value at one normalised flux."""
        return amplitude * (1.0 - jnp.clip(jnp.asarray(psi_norm), 0.0, 1.0))

    return gradient


def _synthetic_geometry_fields(index: int) -> dict[str, object]:
    """Return deterministic internal geometry for the session contract."""
    surface = np.linspace(0.0, 1.0, 11)
    angle = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    radius = 1.0 + 0.25 * surface[:, None] * np.cos(angle)[None, :]
    height = 0.02 * index + 0.35 * surface[:, None] * np.sin(angle)[None, :]
    faces = np.linspace(0.0, 1.0, 26)
    scale = 1.0 + index * 0.01
    profiles = {
        name: scale * (0.1 + faces)
        for name in (
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
    }
    return {
        "flux_surface_psi_norm": surface,
        "flux_surface_psi": scale * surface,
        "flux_surface_r": radius,
        "flux_surface_z": height,
        "flux_surface_angle": angle,
        "rho_face_norm": faces,
        "p_prime_face": -2.0e5 * (1.0 - faces),
        "ff_prime_face": -0.2 * (1.0 - faces),
        **profiles,
        "R_major": 1.0,
        "a_minor": 0.35,
        "B_0": 2.0,
        "boundary_toroidal_flux": 0.5,
        "magnetic_axis_z_scalar": 0.02 * index,
        "diverted": False,
        "divertor_leg_r": np.full((4, 32), np.nan),
        "divertor_leg_z": np.full((4, 32), np.nan),
        "divertor_leg_finite": np.zeros(4, dtype=bool),
    }


def _circular_internal_geometry():
    """Return a small analytic geometry block for frame-assembly tests."""
    major_radius = 3.0
    minor_radius = 0.5
    radius = np.linspace(major_radius - 0.6, major_radius + 0.6, 65)
    height = np.linspace(-0.6, 0.6, 65)
    mesh_radius, mesh_height = np.meshgrid(radius, height, indexing="ij")
    psi = -0.2 * (
        ((mesh_radius - major_radius) / minor_radius) ** 2
        + (mesh_height / minor_radius) ** 2
    )

    def field_function(psi_norm):
        """Return the circular map's constant toroidal-field function."""
        return np.full_like(psi_norm, 2.0 * major_radius)

    geometry = FluxSurfaceGeometry.internal_geometry(
        FluxLattice(radius, height),
        psi,
        field_function,
        axis=(major_radius, 0.0),
        boundary_flux=-0.2,
        reference_radius=major_radius,
        n_surface=11,
        n_theta=64,
        n_rho=25,
    )
    return radius, height, psi, geometry


def _receipt_for_geometry_assembly(radius, height, psi, geometry, *, lcfs="surface"):
    """Return a typed receipt with lightweight solved-state carriers.

    ``lcfs`` selects the raster-derived boundary polyline: the string
    "surface" (default) uses the fixture's outer traced surface, an array
    supplies an explicit polyline, and ``None`` leaves the raster separatrix
    and its derived LCFS absent.
    """
    if lcfs is None:
        boundary = np.empty((0, 2))
    elif isinstance(lcfs, str):
        boundary = np.column_stack((geometry.surface_r[-1], geometry.surface_z[-1]))
    else:
        boundary = np.asarray(lcfs, dtype=float)
    raster = SimpleNamespace(
        radius=radius,
        height=height,
        shape=np.asarray([radius.size, height.size], dtype=np.int32),
        psi=psi,
        psi_norm=psi / -0.2,
        domain_label=np.zeros_like(psi, dtype=np.int8),
        separatrix=boundary,
        separatrix_vertex_count=np.int32(boundary.shape[0]),
    )
    labelled = SimpleNamespace(
        o_point=np.asarray([3.0, 0.0]),
        primary_x_point=np.full(2, np.nan),
        secondary_x_point=np.full(2, np.nan),
        strike_points=np.full((2, 2), np.nan),
        lcfs=boundary,
        lcfs_vertex_count=np.int32(boundary.shape[0]),
    )
    equilibrium = SimpleNamespace(
        raster_flux=raster,
        labelled_flux=labelled,
        fixed_point=SimpleNamespace(active_set_iterations=np.int32(1)),
        constraints=(),
        cell_current=np.ones(radius.size * height.size),
    )
    return ForwardSolveReceipt(
        terminal_state=equilibrium,
        qualified=True,
        termination_reason=np.int32(0),
        residual_history=np.empty(0),
        mask_history=np.empty(0),
        globalisation_decisions=(np.empty(0), np.empty(0)),
        amplitude_history=np.empty(0),
        topology_read=None,
        polish_receipt=None,
        compilation_cache_hit=False,
        wall_seconds=0.01,
        resolved_defaults=ResolvedForwardSolveDefaults.from_policy(
            resolve_forward_solve_policy()
        ),
    )


@pytest.fixture(scope="module")
def machine():
    """Return the bootstrapped free-boundary profile and its analytic seed."""
    configure_dtypes()
    lattice = FluxLattice(np.linspace(0.6, 1.42, 15), np.linspace(-0.42, 0.42, 15))
    coordinate = lattice.coordinate
    wall, wall_flux = _wall_loop()
    seed_flux = _solovev(coordinate[:, 0], coordinate[:, 1])
    wall_seed = _solovev(wall[:, 0], wall[:, 1])
    inside = seed_flux >= wall_flux

    angle = 2 * np.pi * np.arange(CONDUCTORS) / CONDUCTORS
    conductor = np.c_[1.0 + 0.62 * np.cos(angle), 0.62 * np.sin(angle)]
    coupling = {
        "plasma_to_grid": _green_block(coordinate, coordinate),
        "plasma_to_wall": _green_block(wall, coordinate),
        "source_to_grid": _green_block(coordinate, conductor),
        "source_to_wall": _green_block(wall, conductor),
    }

    def build(core, current):
        """Return the solve for one declared source and conductor state."""
        return ForwardProfile.from_lattice(
            lattice,
            ForwardSource(core=core, boundary_field_function=BOUNDARY_FIELD_FUNCTION),
            external_current=current,
            wall_coordinate=wall,
            polarity=1,
            inside_material=inside,
            **coupling,
        )

    seed = jnp.asarray(np.r_[seed_flux, wall_seed])
    flat = build(
        DomainProfile(p_prime=_flat_profile(P_PRIME), ff_prime=_flat_profile(FF_PRIME)),
        np.zeros(CONDUCTORS),
    )
    cell_current = np.asarray(flat.operator.cell_current(seed))
    target = np.r_[
        seed_flux - coupling["plasma_to_grid"] @ cell_current,
        wall_seed - coupling["plasma_to_wall"] @ cell_current,
    ]
    weight = np.r_[inside.astype(float), np.ones(len(wall))]
    matrix = np.r_[coupling["source_to_grid"], coupling["source_to_wall"]]
    current = np.linalg.lstsq(matrix * weight[:, None], target * weight, rcond=None)[0]

    profile = build(
        DomainProfile(
            p_prime=_edge_vanishing_profile(2.0 * DRIVE * P_PRIME),
            ff_prime=_edge_vanishing_profile(2.0 * DRIVE * FF_PRIME),
        ),
        current,
    )
    return profile, seed, np.asarray(current)


def _synthetic_frame(
    index: int,
    *,
    axis_radius: float = 1.5,
    rows: int = 2,
    circuits: int = 3,
    boundary_slots: int = 8,
    separatrix_slots: int = 12,
) -> SteeringFrame:
    """Return one deterministic synthetic frame with fixed slot capacities."""
    radial_count, vertical_count = 4, 3
    radius = np.linspace(0.6, 1.42, radial_count, dtype=np.float64)
    height = np.linspace(-0.42, 0.42, vertical_count, dtype=np.float64)
    scale = 1.0 + 0.1 * index
    primary = np.array([0.65, 0.15])
    secondary = np.full(2, np.nan) if index % 2 else np.array([0.86, -0.36])
    strike = np.array([[np.nan, np.nan], [1.2, 0.06]])
    lcfs = np.full((boundary_slots, 2), np.nan)
    valid_vertices = boundary_slots - index
    for vertex in range(valid_vertices):
        angle = 2.0 * np.pi * vertex / valid_vertices
        lcfs[vertex] = np.array(
            [axis_radius + 0.3 * np.cos(angle), 0.2 * np.sin(angle)]
        )
    for slot in range(2):
        lcfs[slot] = np.array([1.1 + 0.05 * index, -0.1 + 0.02 * index])
    finite = np.array(
        [
            True,
            True,
            not bool(np.isnan(secondary[0])),
            False,
            True,
            valid_vertices > 0,
        ]
    )
    action_names = ("minor_radius", "elongation", "x_point_r")
    return SteeringFrame(
        radius=radius,
        height=height,
        shape=np.array([radial_count, vertical_count], dtype=np.int32),
        psi=np.arange(radial_count * vertical_count, dtype=np.float64).reshape(
            radial_count, vertical_count
        )
        * scale,
        psi_norm=np.linspace(0.0, 1.0, radial_count * vertical_count).reshape(
            radial_count, vertical_count
        ),
        domain_label=np.full((radial_count, vertical_count), 0, dtype=np.int8),
        separatrix=np.full((separatrix_slots, 2), np.nan, dtype=np.float64),
        separatrix_vertex_count=np.int32(separatrix_slots - index),
        magnetic_axis_r=0.9 + 0.05 * index,
        magnetic_axis_z=0.02 * index,
        x_point_r=np.array([primary[0], secondary[0]]),
        x_point_z=np.array([primary[1], secondary[1]]),
        strike_points_r=strike[:, 0],
        strike_points_z=strike[:, 1],
        strike_segment=np.full(2, -1, dtype=np.int32),
        strike_parameter=np.full(2, np.nan),
        lcfs_r=lcfs[:, 0],
        lcfs_z=lcfs[:, 1],
        n_boundary_coords=np.int32(valid_vertices),
        finite_mask=finite,
        coil_current=np.arange(circuits, dtype=np.float64) * 1.0e4 + scale,
        compensating_current=np.array([scale, -2.0 * scale], dtype=np.float64)[:rows],
        action=SteeringAction(
            name=action_names[index % len(action_names)],
            delta=0.01 * (index + 1) * (-1) ** index,
            commanded_control_points=np.array([[0.8 + 0.01 * index, 0.0], [1.2, 0.1]]),
        ),
        wall_seconds=0.25 + 0.1 * index,
        trip_count=index + 1,
        carrier_identity="frozen-six",
        nova_version="9.9.9",
        policy_digest="0" * 64,
        p_prime_source="efm",
        current_centroid_r=axis_radius,
        current_centroid_z=0.02 * index,
        reference_centroid_z=0.02 * index + (0.01 if index != 2 else 0.06),
        branch_guard_ok=index != 2,
        **_synthetic_geometry_fields(index),
    )


def _synthetic_session() -> tuple[SteeringFrame, ...]:
    """Return three deterministic frames exercising every slot capacity."""
    return tuple(_synthetic_frame(index) for index in range(3))


def _assert_dataset_variables_bitwise(expected, actual) -> None:
    """Require the stored and the reloaded session to agree on every variable.

    NaN-padded channels are compared with NaN treated as equal; integer,
    boolean and string channels must match exactly.
    """
    names = set(expected.variables).union(set(actual.variables))
    missing = [name for name in names if name not in expected.variables]
    assert not missing, f"variables lost in the store: {missing}"
    for name in names:
        a = np.asarray(expected[name].values)
        b = np.asarray(actual[name].values)
        assert b.shape == a.shape, f"{name} shape {b.shape} != {a.shape}"
        if a.dtype.kind in "fc":
            assert np.array_equal(a, b, equal_nan=True), f"{name} differs (float)"
        else:
            assert np.array_equal(a, b), f"{name} differs ({a.dtype})"


def _assert_frame_fields_equal(
    actual: SteeringFrame,
    expected: SteeringFrame,
    *,
    excluded: tuple[str, ...] = (),
) -> None:
    """Require all selected fields of two typed frames to agree."""
    for name in SteeringFrame._fields:
        if name in excluded:
            continue
        if name == "action":
            assert actual.action.name == expected.action.name
            assert actual.action.delta == expected.action.delta
            np.testing.assert_array_equal(
                np.asarray(actual.action.commanded_control_points),
                np.asarray(expected.action.commanded_control_points),
            )
            continue
        np.testing.assert_array_equal(
            np.asarray(getattr(actual, name)),
            np.asarray(getattr(expected, name)),
        )


def test_synthetic_session_round_trips_bitwise(tmp_path) -> None:
    """Three synthetic frames survive the netCDF store bitwise on every field."""
    frames = _synthetic_session()
    expected = session_dataset(frames)
    write_session(frames, filename="session", dirname=str(tmp_path))
    actual = read_session(filename="session", dirname=str(tmp_path))

    assert int(actual.attrs.get("cocos")) == COCOS
    assert actual.attrs["p_prime_source"] == "efm"
    assert actual.attrs["flux_unit"] == "Wb"
    assert actual.sizes["time"] == 3
    _assert_dataset_variables_bitwise(expected, actual)


def test_session_uses_time_as_last_axis() -> None:
    """Every multi-dimensional channel stacks with time as its final axis."""
    dataset = session_dataset(_synthetic_session())
    time_last = {
        name
        for name, variable in dataset.variables.items()
        if variable.ndim >= 2 and variable.dims[-1] == "time"
    }
    multi = {name for name, variable in dataset.variables.items() if variable.ndim >= 2}
    assert time_last == multi


def test_session_frames_reconstruct_from_the_store(tmp_path) -> None:
    """A stored session decodes back into the typed per-frame records."""
    frames = _synthetic_session()
    write_session(frames, filename="session", dirname=str(tmp_path))
    actual = read_session(filename="session", dirname=str(tmp_path))
    restored = frames_from_session(actual)
    original = _synthetic_session()
    assert len(restored) == len(original)
    for got, want in zip(restored, original, strict=True):
        _assert_frame_fields_equal(got, want)


def test_rasterless_solovev_session_reconstructs_every_frame(machine, tmp_path) -> None:
    """A Solov'ev-grid session decodes with only its raster block absent."""
    profile, seed, _conductor_current = machine
    radial_count = profile.lattice.radius.size
    vertical_count = profile.lattice.height.size
    psi = np.asarray(seed[: radial_count * vertical_count]).reshape(
        radial_count, vertical_count
    )
    frames = tuple(
        frame._replace(
            radius=np.asarray(profile.lattice.radius),
            height=np.asarray(profile.lattice.height),
            shape=np.asarray([radial_count, vertical_count], dtype=np.int32),
            psi=psi * (index + 1),
            psi_norm=np.linspace(0.0, 1.0, psi.size).reshape(psi.shape),
            domain_label=np.zeros(psi.shape, dtype=np.int8),
        )
        for index, frame in enumerate(_synthetic_session())
    )
    expected = session_dataset(frames, include_raster=False)
    write_session(
        frames,
        filename="rasterless-session",
        dirname=str(tmp_path),
        include_raster=False,
    )
    actual = read_session(filename="rasterless-session", dirname=str(tmp_path))
    restored = frames_from_session(actual)

    assert set(RASTER_FIELDS).isdisjoint(actual.variables)
    _assert_dataset_variables_bitwise(expected, actual)
    assert len(restored) == len(frames)
    for got, want in zip(restored, frames, strict=True):
        assert all(getattr(got, name) is None for name in RASTER_FIELDS)
        _assert_frame_fields_equal(got, want, excluded=RASTER_FIELDS)


def test_real_rasterless_corpus_session_decodes() -> None:
    """One corpus session decodes all frames without diagnostic rasters."""
    if not CORPUS_SESSION.is_file():
        pytest.skip(f"corpus session is unreachable: {CORPUS_SESSION}")
    try:
        dataset = read_session(
            filename=CORPUS_SESSION.stem,
            dirname=str(CORPUS_SESSION.parent),
        )
    except OSError as error:
        pytest.skip(f"corpus session is unreachable: {error}")

    restored = frames_from_session(dataset)
    assert len(restored) == dataset.sizes["time"] == 96
    assert all(
        getattr(frame, name) is None
        for frame in (restored[0], restored[-1])
        for name in RASTER_FIELDS
    )
    np.testing.assert_array_equal(
        restored[0].flux_surface_r,
        dataset["flux_surface_r"].isel(time=0).values,
    )
    np.testing.assert_array_equal(
        restored[-1].coil_current,
        dataset["coil_current"].isel(time=-1).values,
    )


def test_rasterless_solovev_session_fills_lcfs_from_the_outer_surface(
    machine, tmp_path
) -> None:
    """A rasterless session carries a closed LCFS from the outer flux surface.

    The frame is assembled from a receipt with no raster separatrix and no
    derived LCFS: the polyline comes from the outermost traced flux surface,
    closed by repeating the first vertex, and the rasterless session
    round-trips that polyline.
    """
    profile, seed, _conductor_current = machine
    radial_count = profile.lattice.radius.size
    vertical_count = profile.lattice.height.size
    psi = np.asarray(seed[: radial_count * vertical_count]).reshape(
        radial_count, vertical_count
    )
    axis_flux = float(np.asarray(seed).max())
    boundary_flux = axis_flux - SEED_SPAN
    flux_span = axis_flux - boundary_flux
    geometry = FluxSurfaceGeometry.internal_geometry(
        profile.lattice,
        np.asarray(psi),
        source_field_function(profile.source, flux_span),
        axis=(AXIS_RADIUS, 0.0),
        boundary_flux=boundary_flux,
        n_surface=11,
        n_theta=64,
        n_rho=25,
    )
    receipt = _receipt_for_geometry_assembly(
        np.asarray(profile.lattice.radius),
        np.asarray(profile.lattice.height),
        psi,
        geometry,
        lcfs=None,
    )
    profile_psi_norm = np.linspace(0.0, 1.0, 65)
    frame = assemble_frame(
        receipt,
        action=SteeringAction(
            name="minor_radius",
            delta=0.01,
            commanded_control_points=np.asarray([[0.8, 0.0], [1.2, 0.0]]),
        ),
        carrier_identity="solovev-fixture",
        applied_current=np.zeros(CONDUCTORS),
        p_prime_psi_norm=profile_psi_norm,
        p_prime=np.asarray(profile.source.core.p_prime(profile_psi_norm)),
        ff_prime_psi_norm=profile_psi_norm,
        ff_prime=np.asarray(profile.source.core.ff_prime(profile_psi_norm)),
        p_prime_source="efm",
        reference_centroid_z=0.0,
        compensating_current=np.empty(0),
        internal_geometry=geometry,
    )

    outer_r = np.asarray(geometry.surface_r)[-1]
    outer_z = np.asarray(geometry.surface_z)[-1]
    assert np.all(np.isfinite(outer_r))
    assert frame.lcfs_r.size == outer_r.size + 1  # surface plus the repeat point
    assert frame.lcfs_r.size == frame.lcfs_z.size
    assert int(frame.n_boundary_coords) == frame.lcfs_r.size  # vertex count
    np.testing.assert_allclose(frame.lcfs_r[: outer_r.size], outer_r, atol=1e-12)
    np.testing.assert_allclose(frame.lcfs_z[: outer_z.size], outer_z, atol=1e-12)
    assert frame.lcfs_r[0] == frame.lcfs_r[-1]  # closed ring
    assert frame.lcfs_z[0] == frame.lcfs_z[-1]
    assert bool(frame.finite_mask[5])  # LCFS present

    # the rasterless session carries the filled closed polyline on read back
    write_session(
        (frame,),
        filename="rasterless-solovev-lcfs",
        dirname=str(tmp_path),
        include_raster=False,
    )
    restored = frames_from_session(
        read_session(filename="rasterless-solovev-lcfs", dirname=str(tmp_path))
    )[0]
    np.testing.assert_array_equal(restored.lcfs_r, frame.lcfs_r)
    np.testing.assert_array_equal(restored.lcfs_z, frame.lcfs_z)
    assert int(restored.n_boundary_coords) == frame.lcfs_r.size


def test_rasterless_solovev_session_wall_reference_round_trips(
    machine, tmp_path
) -> None:
    """The rasterless session round-trips its wall reference and unit table.

    The session stores no coordinates.  Its digest, source identity and one
    closed vessel unit bind the solve to the machine description it used.
    """
    profile, seed, _conductor_current = machine
    operator_wall = np.asarray(profile.operator.wall.coordinate, dtype=np.float64)
    radial_count = profile.lattice.radius.size
    vertical_count = profile.lattice.height.size
    psi = np.asarray(seed[: radial_count * vertical_count]).reshape(
        radial_count, vertical_count
    )
    frames = tuple(
        frame._replace(
            radius=np.asarray(profile.lattice.radius),
            height=np.asarray(profile.lattice.height),
            shape=np.asarray([radial_count, vertical_count], dtype=np.int32),
            psi=psi * (index + 1),
            psi_norm=np.linspace(0.0, 1.0, psi.size).reshape(psi.shape),
            domain_label=np.zeros(psi.shape, dtype=np.int8),
        )
        for index, frame in enumerate(_synthetic_session())
    )
    reference = _wall_reference(operator_wall)
    expected = session_dataset(frames, include_raster=False, wall_reference=reference)
    write_session(
        frames,
        filename="wall-session",
        dirname=str(tmp_path),
        include_raster=False,
        wall_reference=reference,
    )
    actual = read_session(filename="wall-session", dirname=str(tmp_path))

    assert "wall_r" not in actual.variables
    assert "wall_z" not in actual.variables
    assert actual.attrs["containment_reference"] == "wall"
    stored = wall_reference_from_session(actual)
    assert stored.machine == reference.machine
    assert stored.source == reference.source
    assert stored.content_digest == reference.content_digest
    assert stored.unit_vertex_count == (operator_wall.shape[0],)
    assert stored.unit_closed == (True,)
    assert stored.unit_kind == ("vessel",)
    with xr.open_dataset(
        tmp_path / "wall-session.nc", group=f"steering/{WALL_GROUP}"
    ) as group:
        assert set(group.variables).isdisjoint({"wall_r", "wall_z"})
        assert int(group.sizes["wall_unit"]) == 1
    _assert_dataset_variables_bitwise(expected, actual)


def test_labelled_points_outside_wall_receipt_counts_segment_membership() -> None:
    """The receipt counts an out-of-vessel X-point, not a recorded crossing.

    X-points and the magnetic axis are tested against the wall polygon by
    ray-cast point-in-polygon inclusion.  Each strike point reconstructs from
    the wall segment and along-segment parameter recorded with it, so a
    genuinely displaced point is outside even when it remains near the wall.
    """
    operator_wall, _ = _wall_loop()
    primary_x = np.array([1.7, 0.0])  # beyond the outboard wall limit
    assert not bool(
        inside_polygon(
            primary_x[0], primary_x[1], operator_wall[:, 0], operator_wall[:, 1]
        )
    )
    strike_segment = np.array([0, 1], dtype=np.int32)
    strike_parameter = np.array([0.25, 0.75])
    strike = np.array(
        [
            operator_wall[index]
            + parameter
            * (
                operator_wall[(index + 1) % operator_wall.shape[0]]
                - operator_wall[index]
            )
            for index, parameter in zip(strike_segment, strike_parameter, strict=True)
        ]
    )
    frame = _synthetic_frame(0)._replace(
        x_point_r=np.array([primary_x[0], np.nan]),
        x_point_z=np.array([primary_x[1], np.nan]),
        strike_points_r=strike[:, 0],
        strike_points_z=strike[:, 1],
        strike_segment=strike_segment,
        strike_parameter=strike_parameter,
        finite_mask=np.array([True, True, False, True, True, True]),
    )
    reference = _wall_reference(operator_wall)
    dataset = session_dataset((frame,), wall_reference=reference)
    assert count_labelled_outside_wall(dataset, reference) == 1

    # Positive control: moving one strike one millimetre off its segment makes
    # exactly that otherwise-valid slot count outside.
    displaced = frame._replace(
        strike_points_r=frame.strike_points_r + np.array([1.0e-3, 0.0]),
    )
    assert (
        count_labelled_outside_wall(
            session_dataset((displaced,), wall_reference=reference), reference
        )
        == 2
    )

    # Positive control: an invalid segment index cannot identify a wall
    # segment, even when the stored point itself has valid coordinates.
    invalid_segment = frame._replace(
        strike_segment=np.array([operator_wall.shape[0], 1], dtype=np.int32),
    )
    assert (
        count_labelled_outside_wall(
            session_dataset((invalid_segment,), wall_reference=reference), reference
        )
        == 2
    )


def test_legacy_copied_ring_reads_as_a_flagged_unknown_reference(tmp_path) -> None:
    """An inline historical ring is exposed as one flagged vessel reference."""

    wall, _ = _wall_loop()
    legacy = session_dataset((_synthetic_frame(0),))
    legacy.to_netcdf(tmp_path / "legacy.nc", mode="w", group="steering")
    xr.Dataset(
        {
            "wall_r": ("wall_vertex", wall[:, 0]),
            "wall_z": ("wall_vertex", wall[:, 1]),
        }
    ).to_netcdf(tmp_path / "legacy.nc", mode="a", group=f"steering/{WALL_GROUP}")

    restored = read_session(filename="legacy", dirname=str(tmp_path))
    reference = wall_reference_from_session(restored)

    assert "wall_r" not in restored.variables
    assert "wall_z" not in restored.variables
    assert reference.legacy_copied_wall
    assert reference.machine == reference.source == reference.dd_version == "unknown"
    assert reference.unit_vertex_count == (wall.shape[0],)
    assert reference.unit_closed == (True,)
    assert reference.unit_kind == ("vessel",)


def test_frame_psi_at_the_axis_is_the_solve_axis_flux_in_wb(machine) -> None:
    """The raster psi at the axis is the solve's total axis flux in Wb.

    COCOS 17 carries the poloidal flux as the total flux in Wb (``e_Bp = 1``,
    the 2 pi convention), so interpolating the frame's raster psi to the
    magnetic axis returns the axis flux in Wb — not 2 pi times or a fraction
    of it.  The axis is not a lattice node, so the interpolation carries its
    own sub-cell underestimate; the one-percent tolerance absorbs it, while a
    per-radian reading would sit a factor 2 pi away and fail.
    """
    profile, seed, _conductor_current = machine
    radial_count = profile.lattice.radius.size
    vertical_count = profile.lattice.height.size
    psi = np.asarray(seed[: radial_count * vertical_count]).reshape(
        radial_count, vertical_count
    )
    axis_flux = float(np.asarray(seed).max())
    boundary_flux = axis_flux - SEED_SPAN
    geometry = FluxSurfaceGeometry.internal_geometry(
        profile.lattice,
        np.asarray(psi),
        source_field_function(profile.source, SEED_SPAN),
        axis=(AXIS_RADIUS, 0.0),
        boundary_flux=boundary_flux,
        n_surface=11,
        n_theta=64,
        n_rho=25,
    )
    receipt = _receipt_for_geometry_assembly(
        np.asarray(profile.lattice.radius),
        np.asarray(profile.lattice.height),
        psi,
        geometry,
        lcfs=None,
    )
    profile_psi_norm = np.linspace(0.0, 1.0, 65)
    frame = assemble_frame(
        receipt,
        action=SteeringAction(
            name="minor_radius",
            delta=0.01,
            commanded_control_points=np.asarray([[0.8, 0.0], [1.2, 0.0]]),
        ),
        carrier_identity="solovev-fixture",
        applied_current=np.zeros(CONDUCTORS),
        p_prime_psi_norm=profile_psi_norm,
        p_prime=np.asarray(profile.source.core.p_prime(profile_psi_norm)),
        ff_prime_psi_norm=profile_psi_norm,
        ff_prime=np.asarray(profile.source.core.ff_prime(profile_psi_norm)),
        p_prime_source="efm",
        reference_centroid_z=0.0,
        compensating_current=np.empty(0),
        internal_geometry=geometry,
    )
    interpolant = RegularGridInterpolator(
        (np.asarray(profile.lattice.radius), np.asarray(profile.lattice.height)),
        np.asarray(frame.psi),
        bounds_error=False,
    )
    at_axis = float(interpolant((AXIS_RADIUS, 0.0)))
    assert at_axis == pytest.approx(axis_flux, rel=1e-2)


def test_rastered_frame_keeps_its_raster_derived_polyline() -> None:
    """A rastered session keeps the raster-derived LCFS, never the fallback."""
    radius, height, psi, geometry = _circular_internal_geometry()
    # a distinct raster-derived boundary: a smaller ring, not the outer surface
    custom = np.column_stack(
        (
            0.9 * np.asarray(geometry.surface_r)[-1],
            0.9 * np.asarray(geometry.surface_z)[-1],
        )
    )
    receipt = _receipt_for_geometry_assembly(radius, height, psi, geometry, lcfs=custom)
    profile_psi_norm = np.linspace(0.0, 1.0, 65)
    frame = assemble_frame(
        receipt,
        action=SteeringAction(
            name="minor_radius",
            delta=0.01,
            commanded_control_points=np.asarray([[2.5, 0.0], [3.5, 0.0]]),
        ),
        carrier_identity="circular-map",
        applied_current=np.asarray([0.0]),
        p_prime_psi_norm=profile_psi_norm,
        p_prime=profile_psi_norm,
        ff_prime_psi_norm=profile_psi_norm,
        ff_prime=-profile_psi_norm,
        p_prime_source="efm",
        reference_centroid_z=0.05,
        compensating_current=np.empty(0),
        internal_geometry=geometry,
    )
    np.testing.assert_array_equal(frame.lcfs_r, custom[:, 0])
    np.testing.assert_array_equal(frame.lcfs_z, custom[:, 1])
    assert int(frame.n_boundary_coords) == custom.shape[0]


def test_masked_components_stay_masked_never_imputed() -> None:
    """An absent X-point slot and a missing strike point carry NaN + False."""
    frames = _synthetic_session()
    second = frames[1]  # secondary X-point NaN, inboard strike NaN
    assert second.finite_mask.shape == (len(FINITE_MASK_COMPONENTS),)
    assert bool(second.finite_mask[0])  # magnetic axis present
    assert bool(second.finite_mask[1])  # primary X-point present
    assert not bool(second.finite_mask[2])  # secondary absent
    assert not bool(second.finite_mask[3])  # inboard strike absent
    assert bool(np.isnan(second.x_point_r[1]))
    assert bool(np.isnan(second.x_point_z[1]))
    assert bool(np.isnan(second.strike_points_r[0]))


def test_session_declares_cocos_17() -> None:
    """The recorded session marks its coordinate system as COCOS 17."""
    assert COCOS == 17
    dataset = session_dataset(_synthetic_session())
    assert int(dataset.attrs["cocos"]) == 17


def test_centroid_channels_are_diagnostic_only() -> None:
    """Centroid branch references never enter the decoder training inputs."""
    dataset = session_dataset(_synthetic_session())
    training = set(dataset.attrs["training_inputs"].split(","))
    diagnostic = set(dataset.attrs["diagnostic_only"].split(","))
    assert {"p_prime_face", "ff_prime_face"} <= training
    centroid_fields = {
        "current_centroid_r",
        "current_centroid_z",
        "reference_centroid_z",
        "branch_guard_ok",
    }
    assert centroid_fields <= diagnostic
    assert centroid_fields.isdisjoint(training)


def test_policy_digest_is_deterministic() -> None:
    """The same resolved policy always names the same digest."""
    policy = resolve_forward_solve_policy()
    first = policy_digest(policy)
    second = policy_digest(resolve_forward_solve_policy())
    assert first == second
    assert len(first) == 64
    assert all(char in "0123456789abcdef" for char in first)


def test_assemble_frame_carries_supplied_internal_geometry(tmp_path) -> None:
    """Assembly and storage preserve producer loops and face flux functions."""
    radius, height, psi, geometry = _circular_internal_geometry()
    receipt = _receipt_for_geometry_assembly(radius, height, psi, geometry)
    profile_psi_norm = np.linspace(0.0, 1.0, 65)
    frame = assemble_frame(
        receipt,
        action=SteeringAction(
            name="minor_radius",
            delta=0.01,
            commanded_control_points=np.asarray([[2.5, 0.0], [3.5, 0.0]]),
        ),
        carrier_identity="circular-map",
        applied_current=np.asarray([0.0]),
        p_prime_psi_norm=profile_psi_norm,
        p_prime=profile_psi_norm,
        ff_prime_psi_norm=profile_psi_norm,
        ff_prime=-profile_psi_norm,
        p_prime_source="efm",
        reference_centroid_z=0.05,
        compensating_current=np.empty(0),
        internal_geometry=geometry,
    )

    assert np.all(np.isfinite(frame.flux_surface_r))
    assert np.all(np.isfinite(frame.flux_surface_z))
    np.testing.assert_array_equal(frame.flux_surface_r, geometry.surface_r)
    np.testing.assert_array_equal(frame.flux_surface_z, geometry.surface_z)
    np.testing.assert_array_equal(
        frame.flux_surface_r[0],
        np.full_like(frame.flux_surface_r[0], frame.flux_surface_r[0, 0]),
    )
    assert frame.p_prime_face.dtype == np.float64
    assert frame.ff_prime_face.dtype == np.float64
    np.testing.assert_array_equal(frame.p_prime_face, frame.psi_norm_face)
    np.testing.assert_array_equal(frame.ff_prime_face, -frame.psi_norm_face)
    assert frame.p_prime_source == "efm"
    assert frame.current_centroid_z == pytest.approx(0.0, abs=1.0e-15)
    assert frame.reference_centroid_z == 0.05
    assert frame.branch_guard_ok
    write_session((frame,), filename="linear-profiles", dirname=str(tmp_path))
    restored = frames_from_session(
        read_session(filename="linear-profiles", dirname=str(tmp_path))
    )[0]
    np.testing.assert_array_equal(restored.p_prime_face, frame.psi_norm_face)
    np.testing.assert_array_equal(restored.ff_prime_face, -frame.psi_norm_face)


def test_current_centroid_matches_the_labeller_observation(machine) -> None:
    """The carried-current first moment matches the labeller driver definition."""
    profile, seed, _conductor_current = machine
    equilibrium = SimpleNamespace(cell_current=profile.operator.cell_current(seed))
    raster = SimpleNamespace(
        radius=profile.lattice.radius,
        height=profile.lattice.height,
    )
    achieved_r, achieved_z = _current_centroid(equilibrium, raster)
    expected = profile.current_moment_observation(
        seed,
        support=MomentIntegralSupport.ALL_DOMAIN,
    )
    assert achieved_r == pytest.approx(float(expected.centroid_r), abs=1.0e-12)
    assert achieved_z == pytest.approx(float(expected.centroid_z), abs=1.0e-12)


def test_centroid_branch_guard_flips_beyond_five_centimetres() -> None:
    """The branch guard includes 5 cm and rejects the next larger float."""
    radius, height, psi, geometry = _circular_internal_geometry()
    receipt = _receipt_for_geometry_assembly(radius, height, psi, geometry)
    profile_psi_norm = np.linspace(0.0, 1.0, 65)
    arguments = {
        "action": SteeringAction(
            name="minor_radius",
            delta=0.01,
            commanded_control_points=np.asarray([[2.5, 0.0], [3.5, 0.0]]),
        ),
        "carrier_identity": "circular-map",
        "applied_current": np.asarray([0.0]),
        "p_prime_psi_norm": profile_psi_norm,
        "p_prime": profile_psi_norm,
        "ff_prime_psi_norm": profile_psi_norm,
        "ff_prime": -profile_psi_norm,
        "p_prime_source": "efm",
        "compensating_current": np.empty(0),
        "internal_geometry": geometry,
    }
    at_limit = assemble_frame(receipt, reference_centroid_z=0.05, **arguments)
    beyond = assemble_frame(
        receipt,
        reference_centroid_z=np.nextafter(0.05, np.inf),
        **arguments,
    )
    absent = assemble_frame(receipt, reference_centroid_z=None, **arguments)

    assert at_limit.branch_guard_ok
    assert not beyond.branch_guard_ok
    assert np.isnan(absent.reference_centroid_z)
    assert not absent.branch_guard_ok


@pytest.mark.slow
def test_fixture_frame_carries_every_decoder_field(machine, tmp_path) -> None:
    """A frame from the solved Solov'ev fixture carries every decoder channel.

    The terminal state is the production route's reachable free-boundary
    result wrapped in a genuine forward solve receipt, so the raster channels,
    labelled points, coil currents, keyframe wall and trip count, and the
    request identity are the solve's actual outputs; the action and the
    recorded compensating rows are the steering context that produced the
    frame.  Convergence is reported by the receipt for consumers to filter and
    is not part of this decoder-field assembly contract.
    """
    profile, seed, conductor_current = machine
    started = time.perf_counter()
    equilibrium = profile.solve(
        seed,
        route="newton_krylov",
        convergence_tolerance=SOLVE_TOLERANCE,
        newton_steps=PRODUCTION_NEWTON_STEPS,
        gmres_iterations=GMRES_ITERATIONS,
    )
    wall_seconds = time.perf_counter() - started
    policy = resolve_forward_solve_policy(
        overrides={
            "newton_steps": PRODUCTION_NEWTON_STEPS,
            "gmres_iterations": GMRES_ITERATIONS,
            "kernel_tolerance": SOLVE_TOLERANCE,
        }
    )
    history = equilibrium.fixed_point
    receipt = ForwardSolveReceipt(
        terminal_state=equilibrium,
        qualified=bool(np.asarray(equilibrium.finite.passed)),
        termination_reason=history.termination_reason,
        residual_history=history.active_set_residuals,
        mask_history=history.active_set_mask_differences,
        globalisation_decisions=(
            history.inner_iteration_decisions,
            history.inner_iteration_applied_factors,
        ),
        amplitude_history=jnp.atleast_1d(equilibrium.normalisation.amplitude),
        topology_read=equilibrium.topology,
        polish_receipt=None,
        compilation_cache_hit=False,
        wall_seconds=wall_seconds,
        resolved_defaults=ResolvedForwardSolveDefaults.from_policy(policy),
    )

    action = SteeringAction(
        name="minor_radius",
        delta=-0.03,
        commanded_control_points=np.array(
            [[0.85, 0.0], [1.15, 0.1], [1.0, 0.2], [1.0, -0.2]]
        ),
    )
    axis = np.asarray(equilibrium.topology.axis, dtype=float)
    internal_geometry = FluxSurfaceGeometry.internal_geometry(
        profile.lattice,
        np.asarray(equilibrium.flux),
        source_field_function(profile.source, float(equilibrium.topology.flux_span)),
        axis=(float(axis[0]), float(axis[1])),
        boundary_flux=float(equilibrium.topology.boundary_flux),
        n_surface=11,
        n_theta=64,
        n_rho=25,
    )
    profile_psi_norm = np.linspace(0.0, 1.0, 65)
    p_prime = np.asarray(profile.source.core.p_prime(profile_psi_norm))
    ff_prime = np.asarray(profile.source.core.ff_prime(profile_psi_norm))
    centroid = profile.current_moment_observation(
        equilibrium.flux,
        support=MomentIntegralSupport.ALL_DOMAIN,
    )
    frame = assemble_frame(
        receipt,
        action=action,
        carrier_identity=CARRIER_IDENTITY,
        applied_current=conductor_current,
        p_prime_psi_norm=profile_psi_norm,
        p_prime=p_prime,
        ff_prime_psi_norm=profile_psi_norm,
        ff_prime=ff_prime,
        p_prime_source="efm",
        reference_centroid_z=float(centroid.centroid_z),
        compensating_current=np.array([-1.25e4, 3.1e3]),
        internal_geometry=internal_geometry,
    )

    radial_count, vertical_count = (
        int(np.asarray(equilibrium.raster_flux.shape)[0]),
        int(np.asarray(equilibrium.raster_flux.shape)[1]),
    )
    assert frame.psi.shape == (radial_count, vertical_count)
    assert frame.psi_norm.shape == (radial_count, vertical_count)
    assert frame.domain_label.shape == (radial_count, vertical_count)
    assert frame.psi.dtype == np.float64
    assert frame.psi_norm.dtype == np.float64
    assert frame.domain_label.dtype == np.int8
    assert frame.separatrix.ndim == 2 and frame.separatrix.shape[1] == 2
    assert frame.separatrix.shape[0] > 0
    assert int(frame.separatrix_vertex_count) > 0
    assert np.any(np.isfinite(frame.separatrix))

    assert np.isfinite(frame.magnetic_axis_r)
    assert np.isfinite(frame.magnetic_axis_z)
    assert frame.x_point_r.shape == (2,)
    assert frame.x_point_z.shape == (2,)
    assert np.all(np.isnan(frame.x_point_r))
    assert np.all(np.isnan(frame.x_point_z))
    assert frame.lcfs_r.shape == frame.lcfs_z.shape
    assert frame.lcfs_r.ndim == 1 and frame.lcfs_r.size > 0
    assert int(frame.n_boundary_coords) > 0
    assert frame.n_boundary_coords > 0
    assert int(frame.n_boundary_coords) <= frame.lcfs_r.size
    assert frame.finite_mask.shape == (len(FINITE_MASK_COMPONENTS),)
    assert frame.finite_mask.dtype == bool
    assert bool(frame.finite_mask[0])
    assert not bool(frame.finite_mask[1])  # this limited fixture has no X-point
    assert bool(frame.finite_mask[5])  # LCFS present
    assert frame.coil_current.shape == (CONDUCTORS,)
    assert frame.coil_current.dtype == np.float64
    assert np.all(np.isfinite(frame.coil_current))
    assert frame.compensating_current.shape == (2,)
    assert frame.compensating_current.dtype == np.float64
    assert np.all(np.isfinite(frame.compensating_current))
    assert frame.flux_surface_psi_norm.shape == (11,)
    assert frame.flux_surface_psi.shape == (11,)
    assert frame.flux_surface_r.shape == (11, 64)
    assert frame.flux_surface_z.shape == (11, 64)
    np.testing.assert_array_equal(
        frame.flux_surface_r[0],
        np.full_like(frame.flux_surface_r[0], frame.flux_surface_r[0, 0]),
    )
    assert frame.rho_face_norm.shape == (26,)
    assert frame.vpr.shape == (26,)
    assert frame.p_prime_face.shape == (26,)
    assert frame.ff_prime_face.shape == (26,)
    assert frame.p_prime_face.dtype == np.float64
    assert frame.ff_prime_face.dtype == np.float64
    assert frame.p_prime_source == "efm"
    assert frame.current_centroid_r == pytest.approx(
        float(centroid.centroid_r), abs=1.0e-12
    )
    assert frame.current_centroid_z == pytest.approx(
        float(centroid.centroid_z), abs=1.0e-12
    )
    assert frame.reference_centroid_z == pytest.approx(
        float(centroid.centroid_z), abs=0.0
    )
    assert frame.branch_guard_ok
    assert frame.divertor_leg_r.shape == (4, 32)
    assert not np.any(frame.divertor_leg_finite)

    assert frame.action.name == "minor_radius"
    assert frame.action.delta == -0.03
    assert frame.action.commanded_control_points.shape == (4, 2)
    assert frame.wall_seconds > 0.0
    assert frame.trip_count >= 1
    assert frame.carrier_identity == CARRIER_IDENTITY
    assert frame.nova_version
    assert len(frame.policy_digest) == 64

    # the fixture frame itself round-trips one recorded session
    write_session((frame,), filename="session", dirname=str(tmp_path))
    actual = read_session(filename="session", dirname=str(tmp_path))
    expected = session_dataset((frame,))
    assert actual.attrs["p_prime_source"] == "efm"
    assert "reference_centroid_z" in actual.attrs["diagnostic_only"]
    assert "branch_guard_ok" in actual.attrs["diagnostic_only"]
    _assert_dataset_variables_bitwise(expected, actual)
