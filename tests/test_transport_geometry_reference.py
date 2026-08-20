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
        "R_major": 3.37e-6,
        "a_minor": 1.42e-6,
        "B_0": 3.37e-6,
        "Phi_face": 2.204e-2,
        "volume_face": 1.676e-2,
        "area_face": 1.796e-2,
        "vpr_face": 1.639e-2,
        "spr_face": 2.007e-2,
        "g0_face": 9.047e-3,
        "g1_face": 2.082e-2,
        "g2_face": 1.984e-2,
        "g3_face": 9.326e-3,
        "g2g3_over_rhon_face": 2.628e-2,
        "gm4_face": 5.935e-3,
        "gm5_face": 9.962e-3,
        "F_face": 2.934e-4,
        "R_in_face": 3.00e-3,
        "R_out_face": 1.46e-3,
        "Ip_profile_face": 1.685e-2,
        "psi": 9.925e-3,
        "psi_from_Ip_face": 1.632e-2,
        "j_total_face": 1.420e-1,
        "elongation_face": 7.3e-3,
        "delta_face": 6.5e-2,
        "delta_upper_face": 1.45e-1,
        "delta_lower_face": 9.0e-2,
    },
    "STEP_SPP_001_ECHD_ftop.eqdsk": {
        "R_major": 2.75e-7,
        "a_minor": 8.00e-3,
        "B_0": 2.75e-7,
        "Phi_face": 2.746e-2,
        "volume_face": 2.853e-2,
        "area_face": 3.506e-2,
        "vpr_face": 1.460e-1,
        "spr_face": 1.837e-1,
        "g0_face": 2.293e-2,
        "g1_face": 1.603e-1,
        "g2_face": 4.476e-2,
        "g3_face": 4.078e-2,
        "g2g3_over_rhon_face": 8.185e-2,
        "gm4_face": 3.597e-2,
        "gm5_face": 4.285e-2,
        "F_face": 7.24e-4,
        "R_in_face": 1.04e-2,
        "R_out_face": 5.00e-3,
        "Ip_profile_face": 7.879e-2,
        "psi": 1.279e-2,
        "psi_from_Ip_face": 1.678e-2,
        "j_total_face": 8.90e-1,
        "elongation_face": 1.2e-2,
        "delta_face": 1.72e-1,
        "delta_upper_face": 1.72e-1,
        "delta_lower_face": 1.72e-1,
    },
}
_CHARACTERIZATION_FRACTION = 0.35
_CONTOUR_CHARACTERIZATION_BASELINES = {
    "iterhybrid_cocos17.eqdsk": {
        "Phi_face": 2.276e-2,
        "volume_face": 1.541e-2,
        "area_face": 1.728e-2,
        "vpr_face": 1.595e-2,
        "g0_face": 1.067e-2,
        "g1_face": 2.393e-2,
        "g2_face": 2.392e-2,
        "g3_face": 9.973e-3,
        "F_face": 2.878e-4,
        "psi": 9.922e-3,
    },
    "STEP_SPP_001_ECHD_ftop.eqdsk": {
        "Phi_face": 2.578e-2,
        "volume_face": 4.143e-2,
        "area_face": 6.022e-2,
        "vpr_face": 1.646e-1,
        "g0_face": 1.037e-2,
        "g1_face": 9.776e-2,
        "g2_face": 3.078e-2,
        "g3_face": 2.168e-2,
        "F_face": 5.625e-4,
        "psi": 1.215e-2,
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
    baselines = _CHARACTERIZATION_BASELINES[filename]
    for field, baseline in baselines.items():
        error = _relative_error(
            getattr(nova_geometry, field), getattr(reference_geometry, field)
        )
        errors[field] = error
        print(f"{filename} {field} relative_error={error:.9e}")
        width = max(_CHARACTERIZATION_FRACTION * baseline, 1.0e-8)
        assert baseline - width < error < baseline + width, field
    if filename == "iterhybrid_cocos17.eqdsk":
        for field in ("g2g3_over_rhon_face", "g0_face", "g1_face", "g2_face"):
            assert errors[field] < 0.05, field


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
    for field in _CHARACTERIZATION_BASELINES["iterhybrid_cocos17.eqdsk"]:
        error = _relative_error(getattr(geometry, field), getattr(pinned, field))
        print(f"{filename} {field} convention_relative_error={error:.9e}")
        assert error < 1.0e-8, field
