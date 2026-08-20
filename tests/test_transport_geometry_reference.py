"""Independent real-EQDSK checks for Nova's TORAX geometry handoff."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from nova.io import geqdsk
from nova.jax.config import configure_dtypes
from nova.transport import torax_geometry_from_fsa, traced_flux_surface_geometry


_TORAX_GEOMETRY_DIRECTORY = (
    Path(__import__("torax").__file__).parent / "data" / "third_party" / "geo"
)
_RADIAL_CELLS = 24
_FIELD_TOLERANCES = {
    "R_major": 0.01,
    "a_minor": 0.01,
    "B_0": 0.01,
    "Phi_face": 0.025,
    "volume_face": 0.020,
    "area_face": 0.025,
    "vpr_face": 0.070,
    "spr_face": 0.070,
    "g0_face": 0.26,
    "g1_face": 0.30,
    "g2_face": 0.40,
    "g3_face": 0.14,
    "g2g3_over_rhon_face": 0.45,
    "gm4_face": 0.10,
    "gm5_face": 0.14,
    "F_face": 0.002,
    "R_in_face": 0.035,
    "R_out_face": 0.025,
    "Ip_profile_face": 0.18,
    "psi": 0.015,
    "psi_from_Ip_face": 0.50,
    "j_total_face": 1.00,
    "elongation_face": 0.05,
    "delta_face": 0.24,
    "delta_upper_face": 0.23,
    "delta_lower_face": 0.24,
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
        bandwidth_factor=jnp.asarray(1.0),
    )
    assert bool(record["valid"])
    return torax_geometry_from_fsa(record), record


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
    nova_geometry, _record = _nova_geometry(filename, cocos)
    reference_geometry = _torax_geometry(filename, cocos)
    for field, tolerance in _FIELD_TOLERANCES.items():
        error = _relative_error(
            getattr(nova_geometry, field), getattr(reference_geometry, field)
        )
        print(f"{filename} {field} relative_error={error:.9e}")
        assert error < tolerance, field


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
    for field in _FIELD_TOLERANCES:
        error = _relative_error(getattr(geometry, field), getattr(pinned, field))
        print(f"{filename} {field} convention_relative_error={error:.9e}")
        assert error < 1.0e-8, field
