"""Independent real-equilibrium checks for Nova's TORAX geometry handoff."""

import json
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
import zarr
from scipy.interpolate import RectBivariateSpline

from nova.catalog.mast_geometry import MachineGeometryRegistry
from nova.equilibrium.convention import TOTAL_FLUX_FACTOR
from nova.equilibrium.flux_surface_connectivity import flood_fill_core
from nova.equilibrium.flux_surface_extraction import extract_flux_surface_geometry
from nova.equilibrium.flux_surface_geometry import (
    BISECTION_STEPS,
    LADDER_PER_CELL,
    FluxSurfaceGeometry,
    _ray_reach,
    _refine_axis,
)
from nova.equilibrium.wall_mask import inside_polygon
from nova.imas.mast_efit_referee import read_efit_referee
from nova.imas.mast_vacuum_cohort import SHOT_STORE
from nova.io import geqdsk
from nova.jax.config import configure_dtypes
from nova.transport import torax_geometry_from_fsa


_TORAX_GEOMETRY_DIRECTORY = (
    Path(__import__("torax").__file__).parent / "data" / "third_party" / "geo"
)
_RADIAL_CELLS = 24
_DATA_MANIFEST = Path(__file__).with_name("data-manifest.json")
_MAST_CHARACTERIZATION_FRACTION = 0.10
_MAST_CHARACTERIZATION_BASELINES = {
    "vpr_face": 2.798051546e-2,
    "g1_face": 5.841465586e-2,
}
_CHARACTERIZATION_BASELINES = {
    "iterhybrid_cocos17.eqdsk": {
        "R_major": 6.734001081e-4,
        "a_minor": 7.272066613e-3,
        "B_0": 6.729469455e-4,
        "Phi_face": 2.208927741e-2,
        "volume_face": 1.752872235e-2,
        "area_face": 1.883413123e-2,
        "vpr_face": 2.305704950e-2,
        "spr_face": 2.249341340e-2,
        "g0_face": 8.915209369e-3,
        "g1_face": 2.024575120e-2,
        "g2_face": 1.886635685e-2,
        "g3_face": 9.619437286e-3,
        "g2g3_over_rhon_face": 2.625218535e-2,
        "gm4_face": 6.362461318e-3,
        "gm5_face": 1.026915592e-2,
        "F_face": 2.943625055e-4,
        "R_in_face": 2.825686162e-3,
        "R_out_face": 1.505225041e-3,
        "Ip_profile_face": 1.645214954e-2,
        "psi": 9.957194141e-3,
        "psi_from_Ip_face": 1.585097357e-2,
        "j_total_face": 1.310750373e-1,
        "elongation_face": 7.357881988e-3,
        "delta_face": 1.386278536e-1,
        "delta_upper_face": 2.022743777e-1,
        "delta_lower_face": 8.671390849e-2,
    },
    "STEP_SPP_001_ECHD_ftop.eqdsk": {
        "R_major": 3.532840794e-3,
        "a_minor": 5.802690055e-3,
        "B_0": 3.520403767e-3,
        "Phi_face": 2.560021345e-2,
        "volume_face": 1.837201820e-2,
        "area_face": 2.128043093e-2,
        "vpr_face": 1.841215442e-2,
        "spr_face": 1.646351688e-2,
        "g0_face": 3.647981067e-2,
        "g1_face": 8.096167065e-2,
        "g2_face": 3.229868815e-2,
        "g3_face": 2.166720937e-2,
        "g2g3_over_rhon_face": 5.041073643e-2,
        "gm4_face": 1.686781120e-2,
        "gm5_face": 2.179913664e-2,
        "F_face": 6.789583790e-4,
        "R_in_face": 6.163729571e-3,
        "R_out_face": 2.347521102e-3,
        "Ip_profile_face": 1.137149325e-1,
        "psi": 1.193773823e-2,
        "psi_from_Ip_face": 1.750045313e-2,
        "j_total_face": 7.873204022e-1,
        "elongation_face": 8.299687746e-3,
        "delta_face": 1.385644356e-1,
        "delta_upper_face": 1.400914555e-1,
        "delta_lower_face": 1.370374157e-1,
    },
}
_CHARACTERIZATION_FRACTION = 0.10
_CONTOUR_CHARACTERIZATION_BASELINES = {
    "iterhybrid_cocos17.eqdsk": {
        "Phi_face": 2.281283560e-2,
        "volume_face": 1.616994015e-2,
        "area_face": 1.815258146e-2,
        "vpr_face": 2.262633018e-2,
        "g0_face": 1.053972652e-2,
        "g1_face": 2.328957612e-2,
        "g2_face": 2.386247287e-2,
        "g3_face": 1.026661457e-2,
        "F_face": 2.887973334e-4,
        "psi": 1.003558834e-2,
    },
    "STEP_SPP_001_ECHD_ftop.eqdsk": {
        "Phi_face": 2.392377206e-2,
        "volume_face": 3.114705836e-2,
        "area_face": 4.610763352e-2,
        "vpr_face": 2.349368719e-2,
        "g0_face": 9.133828784e-3,
        "g1_face": 3.884813932e-2,
        "g2_face": 3.280621596e-2,
        "g3_face": 2.222157774e-2,
        "F_face": 5.178695663e-4,
        "psi": 1.129077246e-2,
    },
}
_STEP_CLIPPED_TO_CONTOUR_LIMITS = {
    "vpr_face": 3.92e-2,
    "g1_face": 6.20e-2,
}


def _torax_geometry(filename: str, cocos: int):
    from torax._src.geometry.eqdsk import EQDSKConfig

    return EQDSKConfig(
        cocos=cocos,
        geometry_file=filename,
        geometry_directory=str(_TORAX_GEOMETRY_DIRECTORY),
        Ip_from_parameters=False,
        n_rho=_RADIAL_CELLS,
        n_surfaces=100,
        last_surface_factor=0.99,
    ).build_geometry()


def _nova_input(filename: str, cocos: int) -> dict[str, object]:
    """Read with Nova, normalizing flux units and ITER-twin signs."""
    data = geqdsk.read(str(_TORAX_GEOMETRY_DIRECTORY / filename))
    if cocos == 17:
        return data

    data = dict(data)
    if cocos in (1, 2):
        flux_factor = 2.0 * np.pi
        data["psi"] = data["psi"] * flux_factor
        data["simagx"] *= flux_factor
        data["sibdry"] *= flux_factor
    elif cocos != 11:
        raise ValueError(f"unsupported EQDSK convention {cocos}")

    if cocos == 1:
        return data

    # These shipped twins preserve the poloidal-flux sense while reversing the
    # toroidal coordinate: current, vacuum field and F therefore reverse.
    data["Ip"] *= -1.0
    data["bcentr"] *= -1.0
    data["fpol"] = -data["fpol"]
    return data


def _nova_geometry(filename: str, cocos: int):
    configure_dtypes()
    data = _nova_input(filename, cocos)
    boundary_major_radius = 0.5 * (
        float(np.max(data["xbdry"])) + float(np.min(data["xbdry"]))
    )
    boundary_field = (
        float(data["bcentr"]) * float(data["xcentr"]) / boundary_major_radius
    )
    record = extract_flux_surface_geometry(
        jnp.asarray(np.asarray(data["psi"]).T),
        jnp.asarray(data["x"]),
        jnp.asarray(data["z"]),
        jnp.ones((int(data["nz"]), int(data["nx"])), dtype=bool),
        axis_psi=jnp.asarray(data["simagx"]),
        boundary_psi=jnp.asarray(data["sibdry"]),
        profile_coefficients=jnp.zeros(2),
        coefficient_scale=jnp.ones(2),
        ip_amperes=jnp.asarray(data["Ip"]),
        major_radius=jnp.asarray(boundary_major_radius),
        boundary_toroidal_field=jnp.asarray(boundary_field),
        field_function_psi_n=jnp.asarray(data["pnorm"]),
        field_function=jnp.asarray(data["fpol"]),
        n_pressure=1,
        n_diamagnetic=1,
        n_radial_cells=_RADIAL_CELLS,
        n_surface_bins=96,
        psi_n_min=jnp.asarray(0.01),
        psi_n_max=jnp.asarray(0.99),
    )
    assert bool(record["valid"])
    return torax_geometry_from_fsa(record), record


def _contour_geometry(filename: str, cocos: int, rho_face):
    """Read the same map through Nova's independent traced-contour route."""
    data = _nova_input(filename, cocos)
    boundary_major_radius = 0.5 * (
        float(np.max(data["xbdry"])) + float(np.min(data["xbdry"]))
    )
    lattice = SimpleNamespace(radius=data["x"], height=data["z"])

    def field_function(psi_n):
        return np.interp(psi_n, data["pnorm"], data["fpol"])

    return FluxSurfaceGeometry.from_flux_map(
        lattice,
        data["psi"],
        field_function,
        axis=(float(data["xmagx"]), float(data["zmagx"])),
        boundary_flux=float(data["sibdry"]),
        reference_radius=boundary_major_radius,
        rho_tor_norm=np.asarray(rho_face),
        surfaces=129,
        angles=256,
        edge_psi_norm=0.99,
    )


def _uniform_stored_axis(values, name: str) -> np.ndarray:
    """Verify stored precision before rebuilding a uniform float64 axis."""
    stored = np.asarray(values)
    if (
        stored.ndim != 1
        or stored.size < 2
        or not np.issubdtype(stored.dtype, np.floating)
    ):
        raise ValueError(f"{name} must be a one-dimensional floating coordinate")
    expected = np.linspace(stored[0], stored[-1], stored.size, dtype=stored.dtype)
    tolerance = 16.0 * np.finfo(stored.dtype).eps * max(1.0, abs(float(stored[-1])))
    deviation = float(np.max(np.abs(stored - expected)))
    if deviation > tolerance:
        raise ValueError(
            f"stored {name} is non-uniform at {stored.dtype} precision: "
            f"{deviation:.9g} exceeds {tolerance:.9g}"
        )
    return np.linspace(float(stored[0]), float(stored[-1]), stored.size)


def _mast_input() -> dict[str, object]:
    """Read one EFM map with its packaged MAST provenance and referee geometry."""
    metadata = json.loads(_DATA_MANIFEST.read_text())["mast_efm_geometry_reference"]
    registry = MachineGeometryRegistry.default()
    assert metadata["dd_version"] == registry.dd_version
    assert metadata["cocos"] == 3
    assert metadata["flux_unit"] == "Wb/rad"

    shot = int(metadata["shot"])
    index = int(metadata["slice_index"])
    source = Path(SHOT_STORE) / f"{shot}.zarr"
    if not source.is_dir():
        pytest.skip(f"MAST level-1 shot store not present at {source}")
    referee = read_efit_referee(shot, store=SHOT_STORE)
    group = zarr.open_group(str(source), mode="r")[str(metadata["group"])]
    stored_time = np.asarray(group["time"][:])
    time = float(stored_time[index])
    referee_index = int(np.argmin(np.abs(referee.time_s - time)))
    np.testing.assert_allclose(referee.time_s[referee_index], time, rtol=0.0, atol=0.0)

    stored_radius = np.asarray(group["gridr"][:])
    stored_height = np.asarray(group["gridz"][:])
    radius = _uniform_stored_axis(stored_radius, "efm/gridr")
    height = _uniform_stored_axis(stored_height, "efm/gridz")
    raw_flux = np.asarray(group["psirz"][index], dtype=np.float64)
    finite_columns = np.flatnonzero(np.all(np.isfinite(raw_flux), axis=0))
    if finite_columns.size != radius.size:
        raise ValueError(
            "MAST flux map does not carry one finite column per radial coordinate"
        )
    profile_radius = np.asarray(group["profile_r"][:], dtype=np.float64)
    np.testing.assert_allclose(
        profile_radius[finite_columns], stored_radius, rtol=2.0e-7, atol=1.0e-8
    )
    flux = TOTAL_FLUX_FACTOR * raw_flux[:, finite_columns]
    lcfs = referee.lcfs_m[referee_index]
    lcfs = lcfs[np.all(np.isfinite(lcfs), axis=1)]
    boundary_major_radius = 0.5 * (
        float(np.max(lcfs[:, 0])) + float(np.min(lcfs[:, 0]))
    )
    psi_norm = np.asarray(group["psi_norm"][:], dtype=np.float64)
    field_function = np.asarray(group["fpsi_c"][index], dtype=np.float64)
    limiter_radius = np.asarray(group["limiterr"][:], dtype=np.float64)
    limiter_height = np.asarray(group["limiterz"][:], dtype=np.float64)
    finite_limiter = np.isfinite(limiter_radius) & np.isfinite(limiter_height)
    mesh_radius, mesh_height = np.meshgrid(radius, height)
    inside_limiter = inside_polygon(
        mesh_radius.reshape(-1),
        mesh_height.reshape(-1),
        limiter_radius[finite_limiter],
        limiter_height[finite_limiter],
    ).reshape(flux.shape)
    return {
        "shot": shot,
        "slice_index": index,
        "time": time,
        "radius": radius,
        "height": height,
        "flux": flux,
        "inside_limiter": inside_limiter,
        "axis": tuple(referee.magnetic_axis_m[referee_index]),
        "axis_psi": TOTAL_FLUX_FACTOR * float(group["psi_axis"][index]),
        "boundary_psi": TOTAL_FLUX_FACTOR * float(group["psi_boundary"][index]),
        "ip_amperes": float(group["plasma_current_c"][index]),
        "major_radius": boundary_major_radius,
        "field_function_psi_n": psi_norm,
        "field_function": field_function,
        "boundary_toroidal_field": float(field_function[-1]) / boundary_major_radius,
    }


def _mast_geometries():
    configure_dtypes()
    data = _mast_input()
    record = extract_flux_surface_geometry(
        jnp.asarray(data["flux"]),
        jnp.asarray(data["radius"]),
        jnp.asarray(data["height"]),
        jnp.asarray(data["inside_limiter"]),
        axis_psi=jnp.asarray(data["axis_psi"]),
        boundary_psi=jnp.asarray(data["boundary_psi"]),
        profile_coefficients=jnp.zeros(2),
        coefficient_scale=jnp.ones(2),
        ip_amperes=jnp.asarray(data["ip_amperes"]),
        major_radius=jnp.asarray(data["major_radius"]),
        boundary_toroidal_field=jnp.asarray(data["boundary_toroidal_field"]),
        field_function_psi_n=jnp.asarray(data["field_function_psi_n"]),
        field_function=jnp.asarray(data["field_function"]),
        n_pressure=1,
        n_diamagnetic=1,
        n_radial_cells=_RADIAL_CELLS,
        n_surface_bins=96,
        psi_n_min=jnp.asarray(0.01),
        psi_n_max=jnp.asarray(0.99),
    )
    assert bool(record["valid"])

    lattice = SimpleNamespace(radius=data["radius"], height=data["height"])

    def field_function(psi_n):
        return np.interp(psi_n, data["field_function_psi_n"], data["field_function"])

    contour = FluxSurfaceGeometry.from_flux_map(
        lattice,
        np.asarray(data["flux"]).T,
        field_function,
        axis=data["axis"],
        boundary_flux=float(data["boundary_psi"]),
        reference_radius=float(data["major_radius"]),
        rho_tor_norm=np.asarray(record["rho_face"]),
        surfaces=129,
        angles=256,
        edge_psi_norm=0.99,
    )
    return data, record, contour


def _record_geometry_fields(record) -> dict[str, np.ndarray]:
    """Expose the service record's TORAX-named fields without its adapter."""
    rho_face = np.asarray(record["rho_face"])
    volume_derivative_per_flux = np.asarray(record["int_dl_over_bp_face"])
    gradient_scale = float(record["gradient_moment_scale"])
    g0 = gradient_scale * np.asarray(record["grad_psi_face"])
    g0 *= volume_derivative_per_flux
    g1 = gradient_scale**2 * np.asarray(record["grad_psi2_face"])
    g1 *= volume_derivative_per_flux**2
    g2 = gradient_scale**2 * np.asarray(record["grad_psi2_over_r2_face"])
    g2 *= volume_derivative_per_flux**2
    g2g3 = np.zeros_like(rho_face)
    g2g3[1:] = g2[1:] * np.asarray(record["g3_face"])[1:] / rho_face[1:]
    spr = (
        np.asarray(record["vpr_face"])
        * np.asarray(record["inv_r_face"])
        / (2.0 * np.pi)
    )
    area = np.concatenate(
        (
            np.zeros(1, dtype=rho_face.dtype),
            np.cumsum(0.5 * (spr[:-1] + spr[1:]) * np.diff(rho_face)),
        )
    )
    return {
        "vpr_face": np.asarray(record["vpr_face"]),
        "g0_face": g0,
        "g1_face": g1,
        "g2_face": g2,
        "g2g3_over_rhon_face": g2g3,
        "Phi_face": float(record["phi_b"]) * rho_face**2,
        "volume_face": np.asarray(record["volume_face"]),
        "area_face": area,
        "elongation_face": np.asarray(record["elongation_face"]),
        "delta_upper_face": np.asarray(record["delta_upper_face"]),
        "delta_lower_face": np.asarray(record["delta_lower_face"]),
    }


def _contour_shape(data, levels):
    """Evaluate extrema on the host contour route's spline surfaces."""
    radius = np.asarray(data["radius"])
    height = np.asarray(data["height"])
    state = np.asarray(data["flux"]).T
    interpolant = RectBivariateSpline(radius, height, state, kx=3, ky=3, s=0)
    centre = _refine_axis(interpolant, data["axis"], radius, height)
    axis_flux = float(interpolant.ev(*centre))
    span = float(data["boundary_psi"]) - axis_flux
    levels = np.asarray(levels, dtype=np.float64)
    positive = levels > 0.0
    traced_levels = levels[positive]
    angle = 2.0 * np.pi * np.arange(512) / 512
    cosine = np.cos(angle)
    sine = np.sin(angle)
    reach = _ray_reach(centre, cosine, sine, radius, height)
    cosine = cosine[:, None]
    sine = sine[:, None]

    def normalized(distance):
        return (
            interpolant.ev(
                centre[0] + distance * cosine,
                centre[1] + distance * sine,
            )
            - axis_flux
        ) / span

    samples = LADDER_PER_CELL * max(radius.size, height.size)
    ladder = reach[:, None] * np.linspace(0.0, 1.0, samples + 1)[None]
    crossed = normalized(ladder)[:, None, :] >= traced_levels[None, :, None]
    index = np.argmax(crossed, axis=-1)
    lower = np.take_along_axis(ladder, np.maximum(index - 1, 0), axis=1)
    upper = np.take_along_axis(ladder, index, axis=1)
    for _ in range(BISECTION_STEPS):
        middle = 0.5 * (lower + upper)
        below = normalized(middle) < traced_levels[None]
        lower = np.where(below, middle, lower)
        upper = np.where(below, upper, middle)
    distance = 0.5 * (lower + upper)
    surface_radius = centre[0] + distance * cosine
    surface_height = centre[1] + distance * sine
    r_in = surface_radius.min(axis=0)
    r_out = surface_radius.max(axis=0)
    local_major = 0.5 * (r_in + r_out)
    local_minor = 0.5 * (r_out - r_in)
    columns = np.arange(traced_levels.size)
    upper_slot = np.argmax(surface_height, axis=0)
    lower_slot = np.argmin(surface_height, axis=0)
    shape = {
        "elongation_face": (surface_height.max(axis=0) - surface_height.min(axis=0))
        / (2.0 * local_minor),
        "delta_upper_face": (local_major - surface_radius[upper_slot, columns])
        / local_minor,
        "delta_lower_face": (local_major - surface_radius[lower_slot, columns])
        / local_minor,
    }
    return {
        name: np.concatenate(([value[0]], value)) if not positive[0] else value
        for name, value in shape.items()
    }


def _relative_error(actual, expected) -> float:
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    if actual.ndim:
        actual = actual[2:]
        expected = expected[2:]
    scale = max(float(np.max(np.abs(expected))), 1.0e-12)
    return float(np.max(np.abs(actual - expected)) / scale)


@pytest.mark.parametrize(
    ("filename", "cocos"),
    [
        ("iterhybrid_cocos17.eqdsk", 17),
        ("STEP_SPP_001_ECHD_ftop.eqdsk", 1),
    ],
)
def test_nova_fsa_matches_torax_eqdsk_reader(filename, cocos):
    """Both independent readers agree field by field on real equilibria."""
    nova_geometry, record = _nova_geometry(filename, cocos)
    reference_geometry = _torax_geometry(filename, cocos)
    required = int(record["clipped_vertex_count_required"])
    used = int(record["clipped_vertex_count_max"])
    capacity = int(record["clipped_vertex_capacity"])
    print(
        f"{filename} clipped_vertices required={required} used={used} "
        f"capacity={capacity}"
    )
    assert required <= capacity
    assert used <= required
    errors = {}
    outside_characterization = {}
    baselines = _CHARACTERIZATION_BASELINES[filename]
    for field, baseline in baselines.items():
        error = _relative_error(
            getattr(nova_geometry, field), getattr(reference_geometry, field)
        )
        errors[field] = error
        print(f"{filename} {field} relative_error={error:.9e}")
        width = max(_CHARACTERIZATION_FRACTION * baseline, 1.0e-8)
        if not baseline - width < error < baseline + width:
            outside_characterization[field] = error
    if filename == "iterhybrid_cocos17.eqdsk":
        for field in ("g2g3_over_rhon_face", "g0_face", "g1_face", "g2_face"):
            assert errors[field] < 0.05, field
    assert not outside_characterization, outside_characterization


def test_clipped_cells_match_independent_contour_geometry():
    """Clipped supports and traced contours localize their shared metrics."""
    cases = (
        ("iterhybrid_cocos17.eqdsk", 17),
        ("STEP_SPP_001_ECHD_ftop.eqdsk", 1),
    )
    outside_characterization = {}
    for filename, cocos in cases:
        geometry, record = _nova_geometry(filename, cocos)
        contour = _contour_geometry(filename, cocos, record["rho_face"])
        flux_sign = float(record["flux_sign"])
        contour_fields = {
            "Phi_face": np.abs(contour.toroidal_flux),
            "volume_face": contour.volume,
            "area_face": contour.area,
            "vpr_face": contour.volume_derivative * contour.boundary_rho_tor,
            "g0_face": contour.gradient_rho * contour.volume_derivative,
            "g1_face": contour.gradient_rho_squared * contour.volume_derivative**2,
            "g2_face": contour.gradient_rho_squared_over_radius_squared
            * contour.volume_derivative**2,
            "g3_face": contour.inverse_square_radius,
            "F_face": np.abs(contour.field_function),
            "psi": flux_sign
            * 0.5
            * (contour.poloidal_flux[:-1] + contour.poloidal_flux[1:]),
        }
        for field, expected in contour_fields.items():
            error = _relative_error(getattr(geometry, field), expected)
            baseline = _CONTOUR_CHARACTERIZATION_BASELINES[filename][field]
            width = max(_CHARACTERIZATION_FRACTION * baseline, 1.0e-8)
            print(f"{filename} {field} contour_relative_error={error:.9e}")
            if not baseline - width < error < baseline + width:
                outside_characterization[(filename, field)] = error
            if filename == "STEP_SPP_001_ECHD_ftop.eqdsk":
                limit = _STEP_CLIPPED_TO_CONTOUR_LIMITS.get(field)
                if limit is not None:
                    assert error < limit, (field, error, limit)
    assert not outside_characterization, outside_characterization


def test_mast_efm_geometry_matches_host_contour_route():
    """The grid-in service is characterized against its sole MAST referee."""
    data, record, contour = _mast_geometries()
    rho_face = np.asarray(record["rho_face"])
    service_fields = _record_geometry_fields(record)
    contour_g2 = (
        contour.gradient_rho_squared_over_radius_squared * contour.volume_derivative**2
    )
    contour_g2g3 = np.zeros_like(rho_face)
    contour_g2g3[1:] = contour_g2[1:] * contour.inverse_square_radius[1:] / rho_face[1:]
    shape = _contour_shape(data, contour.psi_norm)
    contour_fields = {
        "vpr_face": contour.volume_derivative * contour.boundary_rho_tor,
        "g0_face": contour.gradient_rho * contour.volume_derivative,
        "g1_face": contour.gradient_rho_squared * contour.volume_derivative**2,
        "g2_face": contour_g2,
        "g2g3_over_rhon_face": contour_g2g3,
        "Phi_face": np.abs(contour.toroidal_flux),
        "volume_face": contour.volume,
        "area_face": contour.area,
        **shape,
    }
    errors = {
        field: _relative_error(service_fields[field], expected)
        for field, expected in contour_fields.items()
    }
    for field, error in errors.items():
        print(
            f"MAST shot={data['shot']} slice={data['slice_index']} "
            f"{field} contour_relative_error={error:.9e}"
        )

    print(
        "MAST_PROFILE\tface\tservice_rho\tcontour_rho\tservice_psi_n\t"
        "contour_psi_n\tservice_Phi\tcontour_Phi\tservice_elongation\t"
        "contour_elongation\tservice_delta_upper\tcontour_delta_upper\t"
        "service_delta_lower\tcontour_delta_lower\taxis_expansion\t"
        "limiting_half_boundary_cells"
    )
    for face in range(rho_face.size):
        print(
            "MAST_PROFILE\t"
            f"{face}\t{rho_face[face]:.17e}\t{contour.rho_tor_norm[face]:.17e}\t"
            f"{float(record['psi_n_face'][face]):.17e}\t"
            f"{contour.psi_norm[face]:.17e}\t"
            f"{service_fields['Phi_face'][face]:.17e}\t"
            f"{abs(contour.toroidal_flux[face]):.17e}\t"
            f"{service_fields['elongation_face'][face]:.17e}\t"
            f"{shape['elongation_face'][face]:.17e}\t"
            f"{service_fields['delta_upper_face'][face]:.17e}\t"
            f"{shape['delta_upper_face'][face]:.17e}\t"
            f"{service_fields['delta_lower_face'][face]:.17e}\t"
            f"{shape['delta_lower_face'][face]:.17e}\t"
            f"{int(record['shape_axis_expansion_face'][face])}\t"
            f"{int(record['shape_boundary_cell_count_face'][face])}"
        )

    psi_n_grid = (data["flux"] - data["axis_psi"]) / (
        data["boundary_psi"] - data["axis_psi"]
    )
    lobe_level = 0.675
    confined = (psi_n_grid < lobe_level) & data["inside_limiter"]
    seed_position = np.argmin(np.where(confined, psi_n_grid, np.inf))
    seed = np.zeros(confined.size, dtype=bool)
    seed[seed_position] = True
    level_core = np.asarray(
        flood_fill_core(
            jnp.asarray(confined),
            jnp.asarray(seed.reshape(confined.shape)),
            confined.shape[0] + confined.shape[1],
        )
    ).astype(bool)
    disconnected = np.argwhere(confined & ~level_core)
    assert disconnected.shape == (4, 2)
    np.testing.assert_allclose(
        np.asarray(data["radius"])[disconnected[:, 1]],
        0.5753124990151264,
        rtol=0.0,
        atol=2.0e-15,
    )
    np.testing.assert_allclose(
        np.sort(np.abs(np.asarray(data["height"])[disconnected[:, 0]])),
        (1.75, 1.75, 1.8125, 1.8125),
        rtol=0.0,
        atol=2.0e-15,
    )

    # CORE averages exclude the mirrored private-flux lobes that detach from
    # the axis component at their own level.  The affected face must therefore
    # follow the contour surface rather than the off-axis extremum.
    np.testing.assert_allclose(
        service_fields["elongation_face"][14],
        shape["elongation_face"][14],
        rtol=2.0e-2,
        atol=0.0,
    )

    assert bool(record["shape_axis_expansion_face"][3])
    assert int(record["shape_boundary_cell_count_face"][3]) == 8
    assert not bool(record["shape_axis_expansion_face"][4])
    np.testing.assert_allclose(
        service_fields["delta_lower_face"][3],
        shape["delta_lower_face"][3],
        rtol=1.5e-1,
        atol=0.0,
    )
    assert errors["delta_lower_face"] < 0.10
    assert errors["delta_upper_face"] <= 4.6204e-2

    # TORAX has no EFM reader, so MAST is intentionally a two-way comparison:
    # the traced grid-in service against Nova's independent host contour route.
    for field, expected in _MAST_CHARACTERIZATION_BASELINES.items():
        np.testing.assert_allclose(
            errors[field],
            expected,
            rtol=_MAST_CHARACTERIZATION_FRACTION,
            atol=2.0e-4,
        )
    assert errors["g1_face"] < 0.10


@pytest.mark.parametrize(
    ("filename", "cocos"),
    [
        ("iterhybrid_cocos02.eqdsk", 2),
        ("iterhybrid_cocos11.eqdsk", 11),
    ],
)
def test_iterhybrid_twins_recover_pinned_cocos_seventeen(filename, cocos):
    """The independently authored convention twins recover one Nova record."""
    geometry, _record = _nova_geometry(filename, cocos)
    pinned, _pinned_record = _nova_geometry("iterhybrid_cocos17.eqdsk", 17)
    outside_tolerance = {}
    for field in _CHARACTERIZATION_BASELINES["iterhybrid_cocos17.eqdsk"]:
        error = _relative_error(getattr(geometry, field), getattr(pinned, field))
        print(f"{filename} {field} convention_relative_error={error:.9e}")
        if error >= 1.0e-7:
            outside_tolerance[field] = error
    assert not outside_tolerance, outside_tolerance
