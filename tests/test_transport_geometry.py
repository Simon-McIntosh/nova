"""Analytic and TORAX-contract checks for the direct geometry handoff."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nova.jax.config import configure_dtypes
from nova.transport import traced_flux_surface_geometry, torax_geometry_from_fsa


def _shaped_record(grid_points=101, radial_cells=24, flux_scale=1.0):
    configure_dtypes()
    major_radius = 3.0
    minor_radius = 0.55
    elongation = 1.55
    flux_span = -0.24 * flux_scale
    radius = jnp.linspace(2.3, 3.7, grid_points)
    height = jnp.linspace(-1.05, 1.05, grid_points)
    mesh_radius, mesh_height = jnp.meshgrid(radius, height)
    psi_n = ((mesh_radius - major_radius) / minor_radius) ** 2 + (
        mesh_height / (elongation * minor_radius)
    ) ** 2
    psi = flux_span * psi_n
    inside = psi_n <= 1.08**2
    record = traced_flux_surface_geometry(
        psi,
        radius,
        height,
        inside,
        axis_psi=jnp.asarray(0.0),
        boundary_psi=jnp.asarray(flux_span),
        profile_coefficients=jnp.asarray([1.0, 0.0]),
        coefficient_scale=jnp.asarray([8.0e5, 8.0e5]),
        ip_amperes=jnp.asarray(5.0e5),
        major_radius=jnp.asarray(major_radius),
        boundary_toroidal_field=jnp.asarray(2.0),
        n_pressure=1,
        n_diamagnetic=1,
        n_radial_cells=radial_cells,
        n_surface_bins=28,
    )
    return record, (major_radius, minor_radius, elongation, flux_span)


def _contour_reference(level, major_radius, minor_radius, elongation, flux_span):
    angle = np.linspace(0.0, 2.0 * np.pi, 20001)
    root = np.sqrt(level)
    radius = major_radius + minor_radius * root * np.cos(angle)
    height = elongation * minor_radius * root * np.sin(angle)
    dr = -minor_radius * root * np.sin(angle)
    dz = elongation * minor_radius * root * np.cos(angle)
    dl = np.sqrt(dr**2 + dz**2)
    grad_r = 2.0 * flux_span * (radius - major_radius) / minor_radius**2
    grad_z = 2.0 * flux_span * height / (elongation * minor_radius) ** 2
    grad_flux = np.sqrt(grad_r**2 + grad_z**2)
    weight = radius * dl / grad_flux
    gradient_psi = grad_flux / (2.0 * np.pi)
    field_function = major_radius * 2.0
    b2 = (gradient_psi**2 + field_function**2) / radius**2

    def average(values):
        return np.trapezoid(values * weight, angle) / np.trapezoid(weight, angle)

    return {
        "int_dl_over_bp": 2.0 * np.pi * np.trapezoid(weight, angle),
        "grad_psi": average(gradient_psi),
        "grad_psi2": average(gradient_psi**2),
        "inv_b2": average(1.0 / b2),
    }


def _torax_reader_geometry(record):
    """Build the comparison object through TORAX's standard reader pipeline."""
    from torax._src.geometry.geometry import GeometryType
    from torax._src.geometry.standard_geometry import (
        StandardGeometryIntermediates,
        build_standard_geometry,
    )

    def host(name):
        return np.asarray(record[name]).copy()

    r_in = host("r_in_face")
    r_out = host("r_out_face")
    f_face = np.abs(host("f_face"))
    rho_face = host("rho_face")
    phi_face = float(record["phi_b"]) * rho_face**2
    intermediates = StandardGeometryIntermediates(
        geometry_type=GeometryType.EQDSK,
        Ip_from_parameters=False,
        R_major=0.5 * (r_in[-1] + r_out[-1]),
        a_minor=0.5 * (r_out[-1] - r_in[-1]),
        B_0=f_face[-1] / (0.5 * (r_in[-1] + r_out[-1])),
        psi=float(record["flux_sign"]) * host("psi_face"),
        Ip_profile=np.abs(host("ip_profile_face")),
        Phi=phi_face,
        R_in=r_in,
        R_out=r_out,
        F=f_face,
        int_dl_over_Bp=host("int_dl_over_bp_face"),
        flux_surf_avg_1_over_R=host("inv_r_face"),
        flux_surf_avg_1_over_R2=host("g3_face"),
        flux_surf_avg_grad_psi=host("grad_psi_face"),
        flux_surf_avg_grad_psi2=host("grad_psi2_face"),
        flux_surf_avg_grad_psi2_over_R2=host("grad_psi2_over_r2_face"),
        flux_surf_avg_B2=host("b2_face"),
        flux_surf_avg_1_over_B2=host("inv_b2_face"),
        delta_upper_face=host("delta_upper_face"),
        delta_lower_face=host("delta_lower_face"),
        elongation=host("elongation_face"),
        vpr=host("vpr_face"),
        face_centers=rho_face,
        hires_factor=4,
        z_magnetic_axis=None,
        diverted=False,
        connection_length_target=None,
        connection_length_divertor=None,
        angle_of_incidence_target=None,
        R_OMP=None,
        R_target=None,
        B_pol_OMP=None,
    )
    return build_standard_geometry(intermediates)


def _run_torax_step(geometry, monkeypatch):
    """Run one transport step with a supplied geometry object."""
    from torax._src.geometry import circular_geometry
    from torax._src.orchestration.run_simulation import run_simulation
    from torax._src.test_utils.default_configs import get_default_config_dict
    from torax._src.torax_pydantic.model_config import ToraxConfig

    monkeypatch.setattr(
        circular_geometry.CircularConfig,
        "build_geometry",
        lambda _config: geometry,
    )
    config_data = get_default_config_dict()
    config_data["geometry"]["n_rho"] = geometry.torax_mesh.nx
    config = ToraxConfig.from_dict(config_data)
    output, history = run_simulation(config, progress_bar=False, max_steps=1)
    assert history.sim_error.name in {"NO_ERROR", "DID_NOT_REACH_T_FINAL"}
    return np.asarray(output.children["profiles"].dataset["psi"])


def test_extended_columns_match_shaped_analytic_references_and_batch():
    """The fixed coarea record matches independent contour quadrature."""
    record, parameters = _shaped_record()
    assert bool(record["valid"])
    levels = np.asarray(record["psi_n_face"])
    indices = np.array([8, 12, 16, 20])
    references = [
        _contour_reference(float(levels[index]), *parameters) for index in indices
    ]
    comparisons = {
        "int_dl_over_bp_face": ("int_dl_over_bp", 0.006),
        "grad_psi_face": ("grad_psi", 0.020),
        "grad_psi2_face": ("grad_psi2", 0.006),
        "inv_b2_face": ("inv_b2", 5e-5),
    }
    for column, (reference_name, relative_tolerance) in comparisons.items():
        actual = np.asarray(record[column])[indices]
        expected = np.array([item[reference_name] for item in references])
        np.testing.assert_allclose(actual, expected, rtol=relative_tolerance, atol=1e-9)

    rho = np.asarray(record["rho_face"])[indices]
    np.testing.assert_allclose(
        np.asarray(record["r_in_face"])[indices],
        parameters[0] - parameters[1] * rho,
        rtol=0.04,
        atol=0.02,
    )
    np.testing.assert_allclose(
        np.asarray(record["r_out_face"])[indices],
        parameters[0] + parameters[1] * rho,
        rtol=0.04,
        atol=0.02,
    )
    np.testing.assert_allclose(
        np.asarray(record["elongation_face"])[indices], parameters[2], rtol=0.08
    )
    assert np.max(np.abs(np.asarray(record["delta_upper_face"])[indices])) < 0.12
    assert np.max(np.abs(np.asarray(record["delta_lower_face"])[indices])) < 0.12

    batched = jax.vmap(
        lambda scale: _shaped_record(flux_scale=scale)[0]["grad_psi_face"]
    )(jnp.asarray([0.98, 1.02]))
    assert batched.shape == (2, 25)
    assert not np.allclose(np.asarray(batched[0]), np.asarray(batched[1]))


def test_direct_adapter_matches_torax_reader_run(monkeypatch):
    """A TORAX step agrees for direct and standard-reader geometry objects."""
    record, parameters = _shaped_record()
    geometry = torax_geometry_from_fsa(record)
    reader_geometry = _torax_reader_geometry(record)

    assert geometry.geometry_type.name == "EQDSK"
    np.testing.assert_allclose(
        geometry.torax_mesh.face_centers,
        np.asarray(record["rho_face"]),
        rtol=0.0,
        atol=8.0e-15,
    )
    assert geometry.Phi_face.shape == record["rho_face"].shape
    assert geometry.elongation_face[-1] > 1.35
    assert abs(float(geometry.R_major) - parameters[0]) < 0.04
    assert abs(float(geometry.a_minor) - parameters[1]) < 0.05
    assert np.all(np.diff(np.asarray(geometry.psi_from_Ip_face)) >= 0.0)
    np.testing.assert_allclose(
        np.asarray(geometry.gm4_face), np.asarray(record["inv_b2_face"]), rtol=1e-12
    )
    parity_fields = (
        "Phi_face",
        "volume_face",
        "area_face",
        "vpr_face",
        "spr_face",
        "g0_face",
        "g1_face",
        "g2_face",
        "g3_face",
        "g2g3_over_rhon_face",
        "gm4_face",
        "gm5_face",
        "F_face",
        "R_in_face",
        "R_out_face",
        "Ip_profile_face",
        "psi",
        "psi_from_Ip_face",
        "j_total_face",
        "elongation_face",
        "delta_upper_face",
        "delta_lower_face",
    )
    for field in parity_fields:
        np.testing.assert_allclose(
            np.asarray(getattr(geometry, field)),
            np.asarray(getattr(reader_geometry, field)),
            rtol=1e-12,
            atol=1e-12,
            err_msg=field,
        )
    direct_psi = _run_torax_step(geometry, monkeypatch)
    reader_psi = _run_torax_step(reader_geometry, monkeypatch)
    np.testing.assert_allclose(direct_psi, reader_psi, rtol=1e-10, atol=1e-12)


def test_direct_adapter_refuses_nonuniform_radial_grid_under_jit():
    """A traced grid cannot be silently replaced by TORAX's uniform mesh."""
    record, _ = _shaped_record(radial_cells=8)
    nonuniform = jnp.asarray(record["rho_face"]).at[3].add(0.02)

    def build_phi_face(rho_face):
        traced_record = dict(record)
        traced_record["rho_face"] = rho_face
        return torax_geometry_from_fsa(traced_record).Phi_face

    with pytest.raises(
        jax.errors.JaxRuntimeError,
        match="rho_face must be the uniform normalized grid",
    ):
        jax.jit(build_phi_face)(nonuniform).block_until_ready()
