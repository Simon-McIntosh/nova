"""Direct JAX handoff from equilibrium-owned FSA records to TORAX."""

from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp
import numpy as np


def _cumulative_trapezoid(values, coordinates):
    increments = 0.5 * (values[1:] + values[:-1]) * jnp.diff(coordinates)
    return jnp.concatenate((jnp.zeros(1, dtype=values.dtype), jnp.cumsum(increments)))


def _smooth_near_axis(values, rho_face, *, polynomial_order):
    """Apply TORAX's nearest-edge Savitzky-Golay normalization near axis."""
    if polynomial_order == 1:
        coefficients = jnp.full(5, 0.2, dtype=values.dtype)
    elif polynomial_order == 2:
        coefficients = jnp.asarray([-3, 12, 17, 12, -3], dtype=values.dtype) / 35
    else:
        raise ValueError(f"unsupported polynomial order: {polynomial_order}")

    padded = jnp.pad(values, (2, 2), mode="edge")
    smoothed = sum(
        coefficient * padded[offset : offset + values.size]
        for offset, coefficient in enumerate(coefficients)
    )
    smoothing_limit = jnp.argmin(jnp.abs(rho_face - 0.1))
    index = jnp.arange(values.size)
    return jnp.where((index > 0) & (index < smoothing_limit), smoothed, values)


def _validate_uniform_normalized_grid(rho_face):
    """Validate a concrete grid received at the JAX runtime boundary."""
    rho_host = np.asarray(rho_face)
    expected = np.linspace(0.0, 1.0, rho_host.size, dtype=rho_host.dtype)
    tolerance = 32.0 * np.finfo(rho_host.dtype).eps
    if not np.all(np.isfinite(rho_host)) or not np.all(
        np.abs(rho_host - expected) <= tolerance
    ):
        raise ValueError(
            "rho_face must be the uniform normalized grid implied by its length"
        )
    return rho_host


def _validated_grid_callback(rho_face):
    result = jax.ShapeDtypeStruct(rho_face.shape, rho_face.dtype)
    return jax.pure_callback(
        _validate_uniform_normalized_grid,
        result,
        rho_face,
        vmap_method="sequential",
    )


@jax.custom_jvp
def _require_uniform_normalized_grid(rho_face):
    """Reject a radial grid that the static TORAX mesh would replace."""
    if not jnp.issubdtype(rho_face.dtype, jnp.floating):
        raise TypeError("rho_face must have a floating-point dtype")
    return _validated_grid_callback(rho_face)


@_require_uniform_normalized_grid.defjvp
def _require_uniform_normalized_grid_jvp(primals, tangents):
    (rho_face,), (rho_tangent,) = primals, tangents
    return _validated_grid_callback(rho_face), rho_tangent


def torax_geometry_from_fsa(record: Mapping[str, object], *, hires_factor: int = 4):
    """Construct a TORAX ``StandardGeometry`` without a host geometry reader.

    Nova carries total poloidal flux. TORAX requires outward-increasing flux,
    positive plasma current and a positive field-function branch, so those
    sign normalisations are applied explicitly at this boundary. All profile
    calculations remain in ``jax.numpy``; only the final typed TORAX mesh and
    dataclass construction are host object operations.
    """
    from torax._src.geometry.geometry import GeometryType
    from torax._src.geometry.standard_geometry import StandardGeometry
    from torax._src.torax_pydantic.torax_pydantic import Grid1D

    rho_face = _require_uniform_normalized_grid(jnp.asarray(record["rho_face"]))
    vpr_face = _smooth_near_axis(
        jnp.asarray(record["vpr_face"]), rho_face, polynomial_order=1
    )
    inv_r_face = jnp.asarray(record["inv_r_face"])
    g3_face = jnp.asarray(record["g3_face"])
    dvolume_dpsi = jnp.asarray(record["int_dl_over_bp_face"])
    gradient_moment_scale = jnp.asarray(record.get("gradient_moment_scale", 1.0))
    grad_psi_face = _smooth_near_axis(
        gradient_moment_scale * jnp.asarray(record["grad_psi_face"]),
        rho_face,
        polynomial_order=1,
    )
    grad_psi2_face = _smooth_near_axis(
        gradient_moment_scale**2 * jnp.asarray(record["grad_psi2_face"]),
        rho_face,
        polynomial_order=2,
    )
    grad_psi2_over_r2_face = _smooth_near_axis(
        gradient_moment_scale**2 * jnp.asarray(record["grad_psi2_over_r2_face"]),
        rho_face,
        polynomial_order=2,
    )
    f_nova = jnp.asarray(record["f_face"])
    flux_sign = jnp.asarray(record["flux_sign"])
    psi_nova = jnp.asarray(record["psi_face"])
    psi_face = flux_sign * psi_nova
    f_face = jnp.abs(f_nova)
    ip_profile_face = jnp.abs(jnp.asarray(record["ip_profile_face"]))
    phi_boundary = jnp.asarray(record["phi_b"])
    phi_face = phi_boundary * rho_face**2

    r_in_face = jnp.asarray(record["r_in_face"])
    r_out_face = jnp.asarray(record["r_out_face"])
    major_radius = 0.5 * (r_in_face[-1] + r_out_face[-1])
    minor_radius = 0.5 * (r_out_face[-1] - r_in_face[-1])
    vacuum_field = f_face[-1] / major_radius

    g0_face = grad_psi_face * dvolume_dpsi
    g1_face = grad_psi2_face * dvolume_dpsi**2
    g2_face = grad_psi2_over_r2_face * dvolume_dpsi**2
    g2g3_face = (
        jnp.zeros_like(rho_face).at[1:].set(g2_face[1:] * g3_face[1:] / rho_face[1:])
    )
    spr_face = vpr_face * inv_r_face / (2.0 * jnp.pi)
    volume_face = _cumulative_trapezoid(vpr_face, rho_face)
    area_face = _cumulative_trapezoid(spr_face, rho_face)

    dpsi_drho = (
        jnp.zeros_like(rho_face)
        .at[1:]
        .set(
            ip_profile_face[1:]
            * (16.0 * jnp.pi**3 * 4.0e-7 * jnp.pi * phi_boundary)
            / jnp.maximum(g2g3_face[1:] * f_face[1:], 1e-30)
        )
    )
    psi_from_ip_face = psi_face[0] + _cumulative_trapezoid(dpsi_drho, rho_face)
    psi_from_ip_face = psi_from_ip_face.at[-1].set(
        psi_from_ip_face[-2] + dpsi_drho[-1] * (rho_face[-1] - rho_face[-2])
    )
    j_total_face = jnp.gradient(ip_profile_face, rho_face) / jnp.maximum(
        spr_face, 1e-30
    )
    j_total_face = j_total_face.at[0].set(j_total_face[1])

    def cell(values):
        return 0.5 * (values[:-1] + values[1:])

    rho_hires_norm = jnp.linspace(
        0.0, 1.0, (rho_face.size - 1) * hires_factor + 1, dtype=rho_face.dtype
    )

    def hires(values):
        return jnp.interp(rho_hires_norm, rho_face, values)

    delta_upper_face = jnp.asarray(record["delta_upper_face"])
    delta_lower_face = jnp.asarray(record["delta_lower_face"])
    elongation_face = jnp.asarray(record["elongation_face"])
    b2_face = jnp.asarray(record["b2_face"])
    inv_b2_face = jnp.asarray(record["inv_b2_face"])
    mesh = Grid1D(nx=rho_face.size - 1)
    return StandardGeometry(
        geometry_type=GeometryType.EQDSK,
        torax_mesh=mesh,
        Phi=cell(phi_face),
        Phi_face=phi_face,
        R_major=major_radius,
        a_minor=minor_radius,
        B_0=vacuum_field,
        volume=cell(volume_face),
        volume_face=volume_face,
        area=cell(area_face),
        area_face=area_face,
        vpr=cell(vpr_face),
        vpr_face=vpr_face,
        spr=cell(spr_face),
        spr_face=spr_face,
        delta_face=0.5 * (delta_upper_face + delta_lower_face),
        elongation=cell(elongation_face),
        elongation_face=elongation_face,
        g0=cell(g0_face),
        g0_face=g0_face,
        g1=cell(g1_face),
        g1_face=g1_face,
        g2=cell(g2_face),
        g2_face=g2_face,
        g3=cell(g3_face),
        g3_face=g3_face,
        gm4=cell(inv_b2_face),
        gm4_face=inv_b2_face,
        gm5=cell(b2_face),
        gm5_face=b2_face,
        g2g3_over_rhon=cell(g2g3_face),
        g2g3_over_rhon_face=g2g3_face,
        g2g3_over_rhon_hires=hires(g2g3_face),
        F=cell(f_face),
        F_face=f_face,
        F_hires=hires(f_face),
        R_in=cell(r_in_face),
        R_in_face=r_in_face,
        R_out=cell(r_out_face),
        R_out_face=r_out_face,
        Ip_from_parameters=False,
        Ip_profile_face=ip_profile_face,
        psi=cell(psi_face),
        psi_from_Ip=cell(psi_from_ip_face),
        psi_from_Ip_face=psi_from_ip_face,
        j_total=cell(j_total_face),
        j_total_face=j_total_face,
        delta_upper_face=delta_upper_face,
        delta_lower_face=delta_lower_face,
        spr_hires=hires(spr_face),
        rho_hires_norm=rho_hires_norm,
        rho_hires=rho_hires_norm * jnp.sqrt(phi_boundary / (jnp.pi * vacuum_field)),
        Phi_b_dot=jnp.zeros((), dtype=rho_face.dtype),
        _z_magnetic_axis=None,
        diverted=False,
        connection_length_target=None,
        connection_length_divertor=None,
        angle_of_incidence_target=None,
        R_OMP=None,
        R_target=None,
        B_pol_OMP=None,
    )
