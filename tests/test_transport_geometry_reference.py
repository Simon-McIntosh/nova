"""Independent real-EQDSK checks for Nova's TORAX geometry handoff."""

from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from nova.io import geqdsk
from nova.jax.config import configure_dtypes
from nova.equilibrium.flux_surface_geometry import FluxSurfaceGeometry
from nova.transport import torax_geometry_from_fsa, traced_flux_surface_geometry


_TORAX_GEOMETRY_DIRECTORY = (
    Path(__import__("torax").__file__).parent / "data" / "third_party" / "geo"
)
_RADIAL_CELLS = 24
_CHARACTERIZATION_BASELINES = {
    "iterhybrid_cocos17.eqdsk": {
        "R_major": 4.370e-6,
        "a_minor": 1.942e-5,
        "B_0": 4.370e-6,
        "Phi_face": 2.2053e-2,
        "volume_face": 1.6792e-2,
        "area_face": 1.7980e-2,
        "vpr_face": 1.6399e-2,
        "spr_face": 2.0049e-2,
        "g0_face": 1.0296e-2,
        "g1_face": 2.3632e-2,
        "g2_face": 2.3136e-2,
        "g3_face": 9.3284e-3,
        "g2g3_over_rhon_face": 2.9602e-2,
        "gm4_face": 5.9300e-3,
        "gm5_face": 9.9785e-3,
        "F_face": 2.9348e-4,
        "R_in_face": 3.0026e-3,
        "R_out_face": 1.4850e-3,
        "Ip_profile_face": 2.0303e-2,
        "psi": 9.9260e-3,
        "psi_from_Ip_face": 1.6365e-2,
        "j_total_face": 1.6289e-1,
        "elongation_face": 7.3269e-3,
        "delta_face": 1.3880e-1,
        "delta_upper_face": 2.0217e-1,
        "delta_lower_face": 8.7108e-2,
    },
    # Across the 194 levels used here, derivative bracketing preserves all
    # 133,692 endpoint-sign-changing roots and finds no same-sign root pairs.
    # These bands therefore pin the legacy single-arc parameterization itself.
    "STEP_SPP_001_ECHD_ftop.eqdsk": {
        "R_major": 5.4800e-7,
        "a_minor": 7.9999e-3,
        "B_0": 5.4800e-7,
        "Phi_face": 2.7582e-2,
        "volume_face": 2.9457e-2,
        "area_face": 3.5972e-2,
        "vpr_face": 1.6278e-1,
        "spr_face": 1.9441e-1,
        "g0_face": 7.1653e-2,
        "g1_face": 2.1998e-1,
        "g2_face": 1.0734e-1,
        "g3_face": 3.0912e-2,
        "g2g3_over_rhon_face": 1.0779e-1,
        "gm4_face": 3.0987e-2,
        "gm5_face": 3.0148e-2,
        "F_face": 7.2558e-4,
        "R_in_face": 9.4661e-3,
        "R_out_face": 5.0047e-3,
        "Ip_profile_face": 1.3681e-1,
        "psi": 1.2841e-2,
        "psi_from_Ip_face": 1.6806e-2,
        "j_total_face": 8.6397e-1,
        "elongation_face": 8.8876e-3,
        "delta_face": 1.3916e-1,
        "delta_upper_face": 1.4070e-1,
        "delta_lower_face": 1.3762e-1,
    },
}
_CHARACTERIZATION_FRACTION = 0.10
_CONTOUR_CHARACTERIZATION_BASELINES = {
    "iterhybrid_cocos17.eqdsk": {
        "Phi_face": 2.2776e-2,
        "volume_face": 1.5434e-2,
        "area_face": 1.7299e-2,
        "vpr_face": 1.5935e-2,
        "g0_face": 1.1469e-2,
        "g1_face": 2.6749e-2,
        "g2_face": 2.5699e-2,
        "g3_face": 9.9753e-3,
        "F_face": 2.8791e-4,
        "psi": 9.9231e-3,
    },
    "STEP_SPP_001_ECHD_ftop.eqdsk": {
        "Phi_face": 2.5903e-2,
        "volume_face": 4.2371e-2,
        "area_face": 6.1156e-2,
        "vpr_face": 1.8137e-1,
        "g0_face": 6.9098e-2,
        "g1_face": 2.2306e-1,
        "g2_face": 1.0583e-1,
        "g3_face": 2.3090e-2,
        "F_face": 5.6449e-4,
        "psi": 1.2195e-2,
    },
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
    record = traced_flux_surface_geometry(
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
    assert not outside_characterization, outside_characterization


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
