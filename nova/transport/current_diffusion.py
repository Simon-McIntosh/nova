"""1D flux-surface-averaged resistive current diffusion -- the temporal prior.

The poloidal flux on a flux-surface-averaged radial coordinate obeys

    toc dpsi/dt = d/drho [ (g2 g3 / rho) dpsi/drho ],
    toc = sigma_parallel mu0 16 pi^2 Phi_b^2 rho / F^2,

with ``psi`` the TOTAL poloidal flux [Wb] (the spine's ``Phi = 2 pi R A_phi``
convention), ``rho`` the normalised toroidal-flux coordinate
``sqrt(Phi_tor / Phi_tor,boundary)``, a regularity condition on axis, and the
measured plasma current as the edge gradient condition

    dpsi/drho|_1 = Ip 16 pi^3 mu0 Phi_b / (D_1 F_1).

The formulation follows the TORAX finite-volume psi equation, which is a
reference implementation this module is verified against in form and never a
runtime dependency.

What the module provides:

* :class:`EtaProfile` -- the bounded low-degree-of-freedom parallel resistivity
  profile, the equation's only genuine unknown, fitted ACROSS shots and never
  per-slice;
* :class:`FluxSurfaceGeometry` -- the 1D metrics of one equilibrium on a uniform
  ``rho`` face grid, plus the Ampere identity that reads an enclosed-current
  profile back out of a flux profile;
* :func:`flux_surface_geometry` -- the profile-ladder and equilibrium-grid
  assembly of those metrics from Nova's fixed-shape connectivity bins;
* :func:`diffuse_psi` -- the implicit theta-scheme flux-diffusion step, a jitted
  fp64 Thomas solve per sub-step;
* :func:`predicted_current` -- the evolved state's flux-surface-averaged toroidal
  current density ``dI/dS`` and its ohmic parallel current
  ``<J.B> = sigma_parallel <E.B>``;
* :func:`basis_projection_images` / :func:`project_coefficients` -- the predicted
  profiles projected back onto a profile-coefficient ladder as a non-negative
  least squares.  The two current targets weight the pressure-gradient and
  diamagnetic families DIFFERENTLY, so the parallel Ohm's law -- pinned by the
  measured flux evolution -- carries split information no single-slice magnetics
  fit has;
* :func:`flux_budget` / :func:`ejima_coefficient` -- the consumption ledger.  The
  surface flux swing decomposes structurally into RESISTIVE consumption (the axis
  swing: Ohm's law at the axis, since no flux is stored inside a zero-volume
  surface) plus INDUCTIVE internal storage, and the Ejima coefficient falls out.

All inputs are raw measurements or the spine's own fits.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import numpy as np

from nova.biot.greens import MU0
from nova.jax.config import Precision, resolve_precision
from nova.utilities.importmanager import skip_import

with skip_import("jax"):
    import jax
    import jax.numpy as jnp


_TWO_PI = 2.0 * np.pi
_16PI3 = 16.0 * np.pi**3
_16PI2 = 16.0 * np.pi**2


@dataclass(frozen=True)
class EtaProfile:
    """Bounded monotone parallel-resistivity profile ``eta(psi_n)`` [Ohm m].

    ``eta(psi_n) = eta0 exp(contrast psi_n**shape)`` -- an axis value, a
    non-negative edge/axis log-contrast, and a shape exponent.  The family
    brackets the Spitzer ``eta ~ T_e**-1.5`` form for peaked-to-broad temperature
    profiles while staying finite at the separatrix, where the
    temperature-derived form diverges.  Cross-shot parameters; never per-slice.
    """

    eta0: float = 5.0e-8
    contrast: float = 2.0
    shape: float = 2.0

    BOUNDS = ((1.0e-9, 3.0e-6), (0.0, 8.0), (0.3, 4.0))

    def __call__(self, psi_n: np.ndarray) -> np.ndarray:
        """Return the resistivity at the given normalised poloidal flux."""
        normalised = np.clip(np.asarray(psi_n, dtype=np.float64), 0.0, 1.0)
        return self.eta0 * np.exp(self.contrast * normalised**self.shape)

    def as_vector(self) -> np.ndarray:
        """Return the parameters as an optimiser vector (log axis value)."""
        return np.array(
            [np.log10(self.eta0), self.contrast, self.shape], dtype=np.float64
        )

    @classmethod
    def from_vector(cls, vector: np.ndarray) -> EtaProfile:
        """Build a profile from an optimiser vector, clipped into the bounds."""
        low, high = zip(*cls.BOUNDS, strict=True)
        return cls(
            eta0=float(np.clip(10.0 ** float(vector[0]), low[0], high[0])),
            contrast=float(np.clip(vector[1], low[1], high[1])),
            shape=float(np.clip(vector[2], low[2], high[2])),
        )


@dataclass(frozen=True)
class FluxSurfaceGeometry:
    """1D metrics of one equilibrium on a uniform ``rho`` grid.

    Face arrays sit on ``rho`` in [0, 1] inclusive (``n_rho + 1`` of them); cell
    arrays midway between them.  ``psi_face`` is the TOTAL poloidal flux [Wb] on
    the faces -- the initial condition of a diffusion step -- and ``phi_b`` the
    boundary toroidal flux [Wb].  ``psi_n_face`` maps ``rho`` back to the
    equilibrium's normalised poloidal flux, which is what the resistivity profile
    and the coefficient basis shapes are keyed to.

    ``flux_sign`` is +1 when psi increases outward and -1 on the opposite
    convention (axis flux above boundary flux with a positive toroidal current).
    It normalises the Ampere identity and the current boundary condition, so a
    caller always passes and reads a POSITIVE plasma current whichever convention
    the equilibrium carries.
    """

    rho_face: np.ndarray
    rho_cell: np.ndarray
    psi_face: np.ndarray
    psi_n_face: np.ndarray
    psi_n_cell: np.ndarray
    vpr_face: np.ndarray
    vpr_cell: np.ndarray
    g2_face: np.ndarray
    g3_face: np.ndarray
    g3_cell: np.ndarray
    f_face: np.ndarray
    f_cell: np.ndarray
    b2_cell: np.ndarray
    inv_r_cell: np.ndarray
    phi_b: float
    r0: float
    ip_amperes: float
    axis_psi: float
    boundary_psi: float
    volume: float
    q_face: np.ndarray
    flux_sign: float = 1.0

    @property
    def d_face(self) -> np.ndarray:
        """Diffusion face coefficient ``g2 g3 / rho`` (regular limit 0 on axis)."""
        coefficient = np.zeros_like(self.rho_face)
        coefficient[1:] = self.g2_face[1:] * self.g3_face[1:] / self.rho_face[1:]
        return coefficient

    def ip_edge_gradient(self, ip_amperes: float) -> float:
        """``dpsi/drho`` at the edge carrying the given total current.

        ``ip_amperes`` is the measured (positive-normalised) current; the stored
        ``flux_sign`` orients the gradient to the equilibrium's own convention.
        """
        return (
            self.flux_sign
            * ip_amperes
            * (_16PI3 * MU0 * self.phi_b)
            / (self.d_face[-1] * self.f_face[-1])
        )

    def enclosed_current(self, psi_face: np.ndarray) -> np.ndarray:
        """Enclosed current on the faces from the flux gradient (Ampere).

        Built from the SCHEME'S OWN stable discrete fluxes: the midpoint gradients
        with midpoint-averaged diffusion and F metrics give the enclosed current at
        cell centres, averaged back onto interior faces (axis exactly zero, edge
        linearly extrapolated).  A pointwise gradient inversion is exact for a
        constructed initial state but amplifies odd-even components of an evolved
        state into current oscillations; the midpoint read is second-order and
        unconditionally stable.
        """
        psi = np.asarray(psi_face, dtype=np.float64)
        drho = float(self.rho_face[1] - self.rho_face[0])
        d_mid = 0.5 * (self.d_face[:-1] + self.d_face[1:])
        f_mid = 0.5 * (self.f_face[:-1] + self.f_face[1:])
        i_mid = (
            self.flux_sign
            * d_mid
            * (np.diff(psi) / drho)
            * f_mid
            / (self.phi_b * _16PI3 * MU0)
        )
        i_face = np.empty_like(psi)
        i_face[0] = 0.0
        i_face[1:-1] = 0.5 * (i_mid[:-1] + i_mid[1:])
        i_face[-1] = 1.5 * i_mid[-1] - 0.5 * i_mid[-2]
        return i_face


def _traced_profile_shapes(psi_n, n_terms: int, *, nonnegative: bool):
    """Profile-ladder shapes on a normalised-flux sample, as JAX arrays."""
    normalised = jnp.clip(psi_n, 0.0, 1.0)
    if n_terms == 0:
        return jnp.empty((*normalised.shape, 0), dtype=normalised.dtype)
    if nonnegative:
        exponents = jnp.asarray(NONNEGATIVE_EXPONENTS[:n_terms], dtype=normalised.dtype)
        return (1.0 - normalised)[..., jnp.newaxis] ** exponents

    coordinate = 2.0 * normalised - 1.0
    terms = [jnp.ones_like(normalised)]
    if n_terms > 1:
        terms.append(coordinate)
    for degree in range(2, n_terms):
        terms.append(
            ((2 * degree - 1) * coordinate * terms[-1] - (degree - 1) * terms[-2])
            / degree
        )
    return jnp.stack(terms, axis=-1) * (1.0 - normalised)[..., jnp.newaxis]


def _cumulative_trapezoid_from_axis(coordinate, values):
    """Cumulative trapezoid with the finite axis value extended to zero."""
    increments = 0.5 * (values[1:] + values[:-1]) * (coordinate[1:] - coordinate[:-1])
    return jnp.concatenate(
        [
            values[:1] * coordinate[:1],
            values[:1] * coordinate[:1] + jnp.cumsum(increments),
        ]
    )


def _integrate_diamagnetic_drive(
    psi_n, diamagnetic_drive, *, boundary_f, poloidal_flux_span
):
    """Integrate ``F F'`` inward from the boundary and return ``F(psi_n)``."""
    increments = (
        0.5 * (diamagnetic_drive[1:] + diamagnetic_drive[:-1]) * jnp.diff(psi_n)
    )
    integral_from_boundary = jnp.concatenate(
        [-jnp.cumsum(increments[::-1])[::-1], jnp.zeros(1, psi_n.dtype)]
    )
    f_squared = (
        boundary_f**2 + 2.0 * poloidal_flux_span / _TWO_PI * integral_from_boundary
    )
    return (
        jnp.sign(boundary_f) * jnp.sqrt(jnp.clip(f_squared, min=0.0)),
        jnp.all(f_squared > 0.0),
    )


def _connectivity_core(psi_n, inside_limiter):
    """Return the axis-connected confined cells using Nova's flood-fill kernel."""
    from nova.equilibrium.flux_surface_connectivity import flood_fill_core

    confined = (psi_n < 1.0) & inside_limiter
    seed_position = jnp.argmin(jnp.where(confined, psi_n, jnp.inf).reshape(-1))
    seed = (
        jnp.zeros(psi_n.size, dtype=bool)
        .at[seed_position]
        .set(True)
        .reshape(psi_n.shape)
    )
    return flood_fill_core(confined, seed, psi_n.shape[0] + psi_n.shape[1])


def _surface_interpolation(rho, rho_samples, values, axis_value, edge_value):
    """Interpolate surface samples onto ``rho`` with explicit axis and edge limits."""
    interpolated = jnp.interp(rho, rho_samples, values, right=edge_value)
    inner = rho < rho_samples[0]
    inner_value = axis_value + (values[0] - axis_value) * (
        rho / jnp.maximum(rho_samples[0], 1e-12)
    )
    return jnp.where(inner, inner_value, interpolated)


def _integrate_bilinear(supports, corner_values, dr, dz):
    """Integrate cellwise bilinear values over fixed clipped polygons."""
    lower_left, lower_right, upper_right, upper_left = jnp.moveaxis(
        corner_values, -1, 0
    )
    constant = 0.25 * (lower_left + lower_right + upper_right + upper_left)
    radial = 0.5 * (-lower_left + lower_right + upper_right - upper_left)
    vertical = 0.5 * (-lower_left - lower_right + upper_right + upper_left)
    cross = lower_left - lower_right + upper_right - upper_left
    return (
        constant * supports.area
        + radial * supports.first_area_moment[:, 0] / dr
        + vertical * supports.first_area_moment[:, 1] / dz
        + cross * supports.second_area_moment[:, 0, 1] / (dr * dz)
    )


def _clipped_surface_geometry(
    psi2d,
    psi_n_grid,
    core,
    radius,
    height,
    f_grid,
    psi_n_min,
    psi_n_max,
    n_surface_bins,
):
    """Return exact clipped-cell coarea bins and TORAX surface columns."""
    from nova.equilibrium.separatrix_clip import _traced_clip

    dtype = psi2d.dtype
    nz, nr = psi2d.shape
    dr = radius[1] - radius[0]
    dz = height[1] - height[0]
    node_index = jnp.arange(nz * nr).reshape(nz, nr)
    cell_nodes = jnp.stack(
        (
            node_index[:-1, :-1],
            node_index[:-1, 1:],
            node_index[1:, 1:],
            node_index[1:, :-1],
        ),
        axis=-1,
    ).reshape(-1, 4)
    mesh_radius, mesh_height = jnp.meshgrid(radius, height)
    node_coordinates = jnp.stack((mesh_radius, mesh_height), axis=-1).reshape(-1, 2)
    centroids = jnp.mean(node_coordinates[cell_nodes], axis=1)
    cell_count = cell_nodes.shape[0]
    vertex_count = jnp.full(cell_count, 4, dtype=jnp.int32)

    psi_cells = psi2d.reshape(-1)[cell_nodes]
    lower_left, lower_right, upper_right, upper_left = jnp.moveaxis(psi_cells, -1, 0)
    gradient_r_lower = (lower_right - lower_left) / dr
    gradient_r_upper = (upper_right - upper_left) / dr
    gradient_z_left = (upper_left - lower_left) / dz
    gradient_z_right = (upper_right - lower_right) / dz
    gradient_flux = jnp.sqrt(
        jnp.stack(
            (
                gradient_r_lower**2 + gradient_z_left**2,
                gradient_r_lower**2 + gradient_z_right**2,
                gradient_r_upper**2 + gradient_z_right**2,
                gradient_r_upper**2 + gradient_z_left**2,
            ),
            axis=-1,
        )
    )
    radius_cells = node_coordinates[cell_nodes, 0]
    field_cells = f_grid.reshape(-1)[cell_nodes]
    gradient_psi = gradient_flux / _TWO_PI
    magnetic_field_squared = (gradient_psi**2 + field_cells**2) / radius_cells**2
    volume_weighted_values = jnp.stack(
        (
            _TWO_PI * radius_cells,
            _TWO_PI / radius_cells,
            jnp.full_like(radius_cells, _TWO_PI),
            _TWO_PI * gradient_flux**2 / radius_cells,
            _TWO_PI * radius_cells * gradient_psi,
            _TWO_PI * radius_cells * gradient_psi**2,
            _TWO_PI * gradient_psi**2 / radius_cells,
            _TWO_PI * radius_cells * magnetic_field_squared,
            _TWO_PI * radius_cells / jnp.maximum(magnetic_field_squared, 1e-30),
        ),
        axis=0,
    )
    eligible = (core > 0.0) | (psi_n_grid >= 1.0)
    flat_flux = psi_n_grid.reshape(-1)
    flat_eligible = eligible.reshape(-1)
    support_capacity = 8

    def signed_flux(level):
        return jnp.where(flat_eligible, level - flat_flux, -1.0)

    def required_vertex_count(level):
        inside = signed_flux(level)[cell_nodes] > 0.0
        crossing = inside != jnp.roll(inside, -1, axis=1)
        return jnp.max(jnp.sum(inside, axis=1) + jnp.sum(crossing, axis=1))

    def clip(level):
        return _traced_clip(
            node_coordinates,
            cell_nodes,
            vertex_count,
            centroids,
            support_capacity,
            signed_flux(level),
        )

    def cumulative(level):
        supports = clip(level)
        integrals = jax.vmap(
            lambda values: jnp.sum(_integrate_bilinear(supports, values, dr, dz))
        )(volume_weighted_values)
        return (
            integrals,
            jnp.max(supports.vertex_count),
            required_vertex_count(level),
        )

    levels = jnp.linspace(psi_n_min, psi_n_max, n_surface_bins + 1, dtype=dtype)
    psi_n_surface = 0.5 * (levels[:-1] + levels[1:])
    cumulative_values, cumulative_used, cumulative_required = jax.lax.map(
        cumulative, levels
    )
    shell_values = jnp.diff(cumulative_values, axis=0)
    shell_volume = jnp.maximum(shell_values[:, 0], 1e-30)
    surface_values = shell_values[:, 1:] / shell_volume[:, None]

    def extrema(level):
        supports = clip(level)
        slots = jnp.arange(supports.support_vertices.shape[1])
        live = slots[None, :] < supports.vertex_count[:, None]
        radial = supports.support_vertices[..., 0]
        vertical = supports.support_vertices[..., 1]
        r_in = jnp.min(jnp.where(live, radial, jnp.inf))
        r_out = jnp.max(jnp.where(live, radial, -jnp.inf))
        lower_slot = jnp.argmin(jnp.where(live, vertical, jnp.inf).reshape(-1))
        upper_slot = jnp.argmax(jnp.where(live, vertical, -jnp.inf).reshape(-1))
        flat_radial = radial.reshape(-1)
        flat_vertical = vertical.reshape(-1)
        return jnp.asarray(
            (
                r_in,
                r_out,
                flat_vertical[lower_slot],
                flat_vertical[upper_slot],
                flat_radial[lower_slot],
                flat_radial[upper_slot],
                jnp.max(supports.vertex_count),
                required_vertex_count(level),
            )
        )

    surface_extrema = jax.lax.map(extrema, psi_n_surface)
    (
        r_in,
        r_out,
        z_lower,
        z_upper,
        r_lower,
        r_upper,
        surface_used,
        surface_required,
    ) = surface_extrema.T
    edge_extrema = extrema(jnp.asarray(1.0, dtype=dtype))
    (
        edge_r_in,
        edge_r_out,
        edge_z_lower,
        edge_z_upper,
        edge_r_lower,
        edge_r_upper,
        edge_used,
        edge_required,
    ) = edge_extrema
    local_major = 0.5 * (r_in + r_out)
    local_minor = jnp.maximum(0.5 * (r_out - r_in), 1e-12)
    edge_major = 0.5 * (edge_r_in + edge_r_out)
    edge_minor = jnp.maximum(0.5 * (edge_r_out - edge_r_in), 1e-12)
    total_values, total_used, total_required = cumulative(jnp.asarray(1.0, dtype=dtype))
    total_volume = total_values[0]
    maximum_used = jnp.max(
        jnp.concatenate(
            (
                cumulative_used,
                surface_used,
                jnp.atleast_1d(edge_used),
                jnp.atleast_1d(total_used),
            )
        )
    )
    maximum_required = jnp.max(
        jnp.concatenate(
            (
                cumulative_required,
                surface_required,
                jnp.atleast_1d(edge_required),
                jnp.atleast_1d(total_required),
            )
        )
    )
    axis_position = jnp.argmin(jnp.where(core > 0, psi_n_grid, jnp.inf))
    axis_radius = mesh_radius.reshape(-1)[axis_position]
    dlevel = (psi_n_max - psi_n_min) / n_surface_bins
    return (
        {
            "pn_s": psi_n_surface,
            "dv_dpn": shell_values[:, 0] / dlevel,
            "inv_r2": surface_values[:, 0],
            "inv_r": surface_values[:, 1],
            "grad2_r2": surface_values[:, 2],
            "v_cum": 0.5 * (cumulative_values[:-1, 0] + cumulative_values[1:, 0]),
            "v_total": total_volume,
        },
        {
            "grad_psi_surface": surface_values[:, 3],
            "grad_psi2_surface": surface_values[:, 4],
            "grad_psi2_over_r2_surface": surface_values[:, 5],
            "b2_surface": surface_values[:, 6],
            "inv_b2_surface": surface_values[:, 7],
            "r_in_surface": r_in,
            "r_in_edge": edge_r_in,
            "r_out_surface": r_out,
            "r_out_edge": edge_r_out,
            "elongation_surface": (z_upper - z_lower) / (2.0 * local_minor),
            "elongation_edge": (edge_z_upper - edge_z_lower) / (2.0 * edge_minor),
            "delta_upper_surface": (local_major - r_upper) / local_minor,
            "delta_upper_edge": (edge_major - edge_r_upper) / edge_minor,
            "delta_lower_surface": (local_major - r_lower) / local_minor,
            "delta_lower_edge": (edge_major - edge_r_lower) / edge_minor,
            "axis_radius": jnp.asarray(axis_radius, dtype=dtype),
            "clipped_vertex_count_max": maximum_used,
            "clipped_vertex_count_required": maximum_required,
            "clipped_vertex_capacity": jnp.asarray(support_capacity),
        },
    )


@partial(
    jax.jit,
    static_argnames=(
        "n_pressure",
        "n_diamagnetic",
        "n_radial_cells",
        "nonnegative",
    ),
)
def traced_assemble_flux_surface_geometry(
    surface_bins,
    psi2d,
    radius,
    height,
    inside_limiter,
    *,
    axis_psi,
    boundary_psi,
    profile_coefficients,
    coefficient_scale,
    ip_amperes,
    major_radius,
    boundary_toroidal_field,
    field_function_psi_n=None,
    field_function=None,
    n_pressure: int,
    n_diamagnetic: int,
    n_radial_cells: int = 24,
    nonnegative: bool = True,
    torax_columns=None,
):
    """Assemble fixed-shape transport metrics from connectivity surface bins.

    The input grid is ``psi2d[height, radius]`` in total poloidal flux [Wb].
    ``surface_bins`` is the output of
    :func:`nova.equilibrium.flux_surface_connectivity.traced_flux_surface_bins`:
    its
    ``n_surface_bins`` arrays live on increasing normalised-poloidal-flux
    mid-levels and may be nonuniform. ``profile_coefficients`` and
    ``coefficient_scale`` are the pressure-gradient columns followed by the
    diamagnetic columns; their product has toroidal-current-density units
    [A m-2]. A reader may instead supply ``field_function`` on
    ``field_function_psi_n``; that preserves the equilibrium's measured F and
    poloidal-flux profiles while the coarea moments supply its current profile.

    The returned dictionary is a JAX PyTree. Face arrays have
    ``n_radial_cells + 1`` entries, cell arrays have ``n_radial_cells`` entries,
    and all shapes are independent of the number of confined grid cells. The
    scalar ``valid`` is false for empty or ill-posed surfaces; callers can
    discard that slice without fabricating geometry.
    """
    dtype = jnp.asarray(psi2d).dtype
    psi2d = jnp.asarray(psi2d, dtype=dtype)
    radius = jnp.asarray(radius, dtype=dtype)
    height = jnp.asarray(height, dtype=dtype)
    inside_limiter = jnp.asarray(inside_limiter, dtype=bool)
    profile_coefficients = jnp.asarray(profile_coefficients, dtype=dtype)
    coefficient_scale = jnp.asarray(coefficient_scale, dtype=dtype)
    if (field_function is None) != (field_function_psi_n is None):
        raise ValueError(
            "field_function and field_function_psi_n must be supplied together"
        )

    poloidal_flux_span = boundary_psi - axis_psi
    safe_span = jnp.where(
        jnp.abs(poloidal_flux_span) > 1e-12, poloidal_flux_span, 1e-12
    )
    psi_n_grid = (psi2d - axis_psi) / safe_span
    core = _connectivity_core(psi_n_grid, inside_limiter)
    scaled_coefficients = profile_coefficients * coefficient_scale
    psi_n_profile = jnp.linspace(0.0, 1.0, 101, dtype=dtype)
    if field_function is None:
        diamagnetic_profile = (
            _traced_profile_shapes(
                psi_n_profile, n_diamagnetic, nonnegative=nonnegative
            )
            @ scaled_coefficients[n_pressure:]
        )
        f_profile, f_well_posed = _integrate_diamagnetic_drive(
            psi_n_profile,
            MU0 * major_radius * diamagnetic_profile,
            boundary_f=major_radius * boundary_toroidal_field,
            poloidal_flux_span=safe_span,
        )
    else:
        supplied_psi_n = jnp.asarray(field_function_psi_n, dtype=dtype)
        supplied_field_function = jnp.asarray(field_function, dtype=dtype)
        f_profile = jnp.interp(psi_n_profile, supplied_psi_n, supplied_field_function)
        f_well_posed = (
            (supplied_psi_n.ndim == 1)
            & (supplied_field_function.shape == supplied_psi_n.shape)
            & jnp.all(jnp.diff(supplied_psi_n) > 0.0)
            & jnp.all(jnp.isfinite(supplied_field_function))
            & jnp.all(jnp.abs(supplied_field_function) > 0.0)
        )

    psi_n_surface = jnp.asarray(surface_bins["pn_s"], dtype=dtype)
    volume_derivative = jnp.asarray(surface_bins["dv_dpn"], dtype=dtype)
    inverse_radius_squared = jnp.asarray(surface_bins["inv_r2"], dtype=dtype)
    inverse_radius = jnp.asarray(surface_bins["inv_r"], dtype=dtype)
    gradient_squared_over_radius_squared = jnp.asarray(
        surface_bins["grad2_r2"], dtype=dtype
    )
    cumulative_volume = jnp.asarray(surface_bins["v_cum"], dtype=dtype)
    volume = jnp.asarray(surface_bins["v_total"], dtype=dtype)

    f_surface = jnp.interp(psi_n_surface, psi_n_profile, f_profile)
    f_grid = jnp.interp(psi_n_grid, psi_n_profile, f_profile)
    if torax_columns is None:
        _, torax_columns = _clipped_surface_geometry(
            psi2d,
            psi_n_grid,
            core,
            radius,
            height,
            f_grid,
            psi_n_surface[0] - 0.5 * (psi_n_surface[1] - psi_n_surface[0]),
            psi_n_surface[-1] + 0.5 * (psi_n_surface[-1] - psi_n_surface[-2]),
            psi_n_surface.size,
        )
    volume_derivative_per_flux = volume_derivative / jnp.abs(safe_span)
    safety_factor = (
        jnp.abs(f_surface)
        * inverse_radius_squared
        * volume_derivative_per_flux
        / _TWO_PI
    )
    toroidal_flux_surface = _cumulative_trapezoid_from_axis(
        psi_n_surface, safety_factor * jnp.abs(safe_span)
    )
    boundary_toroidal_flux = toroidal_flux_surface[-1] + (
        1.0 - psi_n_surface[-1]
    ) * safety_factor[-1] * jnp.abs(safe_span)
    safe_boundary_toroidal_flux = jnp.maximum(boundary_toroidal_flux, 1e-30)
    rho_surface = jnp.sqrt(
        jnp.clip(toroidal_flux_surface / safe_boundary_toroidal_flux, 0.0, 1.0)
    )
    rho_surface = jax.lax.associative_scan(jnp.maximum, rho_surface)

    magnetic_field_squared = (
        gradient_squared_over_radius_squared / (4.0 * jnp.pi**2)
        + f_surface**2 * inverse_radius_squared
    )
    rho_face = jnp.linspace(0.0, 1.0, n_radial_cells + 1, dtype=dtype)
    rho_cell = 0.5 * (rho_face[:-1] + rho_face[1:])
    psi_n_face = _surface_interpolation(rho_face, rho_surface, psi_n_surface, 0.0, 1.0)
    psi_n_cell = _surface_interpolation(rho_cell, rho_surface, psi_n_surface, 0.0, 1.0)
    f_face = _surface_interpolation(
        rho_face,
        rho_surface,
        f_surface,
        f_profile[0],
        major_radius * boundary_toroidal_field,
    )
    f_cell = 0.5 * (f_face[:-1] + f_face[1:])
    g3_face = _surface_interpolation(
        rho_face,
        rho_surface,
        inverse_radius_squared,
        inverse_radius_squared[0],
        inverse_radius_squared[-1],
    )
    g3_cell = 0.5 * (g3_face[:-1] + g3_face[1:])
    inverse_radius_cell = jnp.interp(psi_n_cell, psi_n_surface, inverse_radius)
    inverse_radius_face = _surface_interpolation(
        rho_face,
        rho_surface,
        inverse_radius,
        inverse_radius[0],
        inverse_radius[-1],
    )
    magnetic_field_squared_cell = jnp.interp(
        psi_n_cell, psi_n_surface, magnetic_field_squared
    )
    safety_factor_face = _surface_interpolation(
        rho_face,
        rho_surface,
        safety_factor,
        safety_factor[0],
        safety_factor[-1],
    )

    volume_face = _surface_interpolation(
        rho_face, rho_surface, cumulative_volume, 0.0, volume
    )
    volume_face = jax.lax.associative_scan(jnp.maximum, jnp.nan_to_num(volume_face))
    radial_spacing = 1.0 / n_radial_cells
    volume_derivative_face = jnp.gradient(volume_face, radial_spacing)
    volume_derivative_face = volume_derivative_face.at[0].set(0.0)
    volume_derivative_cell = jnp.diff(volume_face) / radial_spacing

    volume_derivative_per_flux_face = jnp.interp(
        psi_n_face, psi_n_surface, volume_derivative_per_flux
    )
    gradient_squared_face = jnp.interp(
        psi_n_face, psi_n_surface, gradient_squared_over_radius_squared
    )
    g2_face = (
        (volume_derivative_per_flux_face**2 * gradient_squared_face).at[0].set(0.0)
    )

    # Ampere's law supplies the enclosed current from the same coarea moments:
    # I = <Bp^2> int(dl/Bp) / mu0.  This keeps the geometry record tied to the
    # supplied equilibrium rather than to the profile ladder used by the solver.
    enclosed_current_surface = (
        gradient_squared_over_radius_squared
        / (4.0 * jnp.pi**2)
        * volume_derivative_per_flux
        / MU0
    )
    enclosed_current = _surface_interpolation(
        rho_face,
        rho_surface,
        enclosed_current_surface,
        0.0,
        enclosed_current_surface[-1],
    )
    current_edge = enclosed_current[-1]
    if field_function is None:
        enclosed_current = (
            enclosed_current
            * jnp.abs(ip_amperes)
            / jnp.maximum(jnp.abs(current_edge), 1e-30)
            * jnp.sign(current_edge)
        )
    enclosed_current = enclosed_current.at[0].set(0.0)

    diffusion_face = jnp.zeros_like(rho_face)
    diffusion_face = diffusion_face.at[1:].set(g2_face[1:] * g3_face[1:] / rho_face[1:])
    flux_sign = jnp.where(safe_span >= 0.0, 1.0, -1.0)
    safe_denominator = jnp.where(
        jnp.abs(diffusion_face[1:] * f_face[1:]) > 1e-30,
        diffusion_face[1:] * f_face[1:],
        1e-30,
    )
    poloidal_flux_gradient = (
        jnp.zeros_like(rho_face)
        .at[1:]
        .set(
            flux_sign
            * enclosed_current[1:]
            * (_16PI3 * MU0 * safe_boundary_toroidal_flux)
            / safe_denominator
        )
    )
    diffusion_mid = 0.5 * (diffusion_face[:-1] + diffusion_face[1:])
    f_mid = 0.5 * (f_face[:-1] + f_face[1:])
    current_mid = (
        flux_sign
        * diffusion_mid
        * 0.5
        * (poloidal_flux_gradient[:-1] + poloidal_flux_gradient[1:])
        * f_mid
        / (safe_boundary_toroidal_flux * _16PI3 * MU0)
    )
    readback_edge_current = 1.5 * current_mid[-1] - 0.5 * current_mid[-2]
    poloidal_flux_gradient = (
        poloidal_flux_gradient
        * jnp.abs(ip_amperes)
        / jnp.maximum(jnp.abs(readback_edge_current), 1e-30)
        * jnp.sign(readback_edge_current)
    )
    if field_function is None:
        psi_face = axis_psi + jnp.concatenate(
            [
                jnp.zeros(1, dtype=dtype),
                jnp.cumsum(
                    0.5
                    * (poloidal_flux_gradient[1:] + poloidal_flux_gradient[:-1])
                    * radial_spacing
                ),
            ]
        )
    else:
        psi_face = axis_psi + safe_span * psi_n_face

    axis_radius = torax_columns["axis_radius"]

    def torax_face(name, axis_value):
        values = torax_columns[name]
        edge_value = torax_columns.get(
            name.removesuffix("_surface") + "_edge", values[-1]
        )
        return _surface_interpolation(
            rho_face, rho_surface, values, axis_value, edge_value
        )

    grad_psi_face = torax_face("grad_psi_surface", 0.0)
    grad_psi2_face = torax_face("grad_psi2_surface", 0.0)
    grad_psi2_over_r2_face = torax_face("grad_psi2_over_r2_surface", 0.0)
    b2_face = torax_face("b2_surface", torax_columns["b2_surface"][0])
    inv_b2_face = torax_face("inv_b2_surface", torax_columns["inv_b2_surface"][0])
    r_in_face = torax_face("r_in_surface", axis_radius)
    r_out_face = torax_face("r_out_surface", axis_radius)
    elongation_face = torax_face(
        "elongation_surface", torax_columns["elongation_surface"][0]
    )
    delta_upper_face = torax_face(
        "delta_upper_surface", torax_columns["delta_upper_surface"][0]
    )
    delta_lower_face = torax_face(
        "delta_lower_surface", torax_columns["delta_lower_surface"][0]
    )
    int_dl_over_bp_face = volume_derivative_per_flux_face

    finite_arrays = (
        jnp.all(jnp.isfinite(psi_n_surface))
        & jnp.all(jnp.isfinite(volume_derivative))
        & jnp.all(jnp.isfinite(inverse_radius_squared))
        & jnp.all(jnp.isfinite(psi_face))
    )
    valid = (
        (jnp.abs(poloidal_flux_span) > 1e-12)
        & (jnp.sum(core) >= 200)
        & jnp.all(jnp.diff(psi_n_surface) > 0.0)
        & jnp.all(volume_derivative > 0.0)
        & (boundary_toroidal_flux > 0.0)
        & (jnp.abs(current_edge) > 1e-6 * jnp.maximum(jnp.abs(ip_amperes), 1.0))
        & jnp.all(diffusion_face[1:] > 0.0)
        & f_well_posed
        & finite_arrays
    )
    return {
        "rho_face": rho_face,
        "rho_cell": rho_cell,
        "psi_face": psi_face,
        "psi_n_face": psi_n_face,
        "psi_n_cell": psi_n_cell,
        "vpr_face": volume_derivative_face,
        "vpr_cell": volume_derivative_cell,
        "g2_face": g2_face,
        "g3_face": g3_face,
        "g3_cell": g3_cell,
        "f_face": f_face,
        "f_cell": f_cell,
        "b2_cell": magnetic_field_squared_cell,
        "inv_r_cell": inverse_radius_cell,
        "inv_r_face": inverse_radius_face,
        "phi_b": safe_boundary_toroidal_flux,
        "r0": major_radius,
        "ip_amperes": jnp.abs(ip_amperes),
        "axis_psi": axis_psi,
        "boundary_psi": boundary_psi,
        "volume": volume,
        "q_face": safety_factor_face,
        "volume_face": volume_face,
        "ip_profile_face": enclosed_current,
        "int_dl_over_bp_face": int_dl_over_bp_face,
        "grad_psi_face": grad_psi_face,
        "grad_psi2_face": grad_psi2_face,
        "grad_psi2_over_r2_face": grad_psi2_over_r2_face,
        "b2_face": b2_face,
        "inv_b2_face": inv_b2_face,
        "r_in_face": r_in_face,
        "r_out_face": r_out_face,
        "elongation_face": elongation_face,
        "delta_upper_face": delta_upper_face,
        "delta_lower_face": delta_lower_face,
        "clipped_vertex_count_max": torax_columns["clipped_vertex_count_max"],
        "clipped_vertex_count_required": torax_columns["clipped_vertex_count_required"],
        "clipped_vertex_capacity": torax_columns["clipped_vertex_capacity"],
        "gradient_moment_scale": jnp.asarray(
            _TWO_PI if field_function is not None else 1.0, dtype=dtype
        ),
        "flux_sign": flux_sign,
        "valid": valid,
    }


@partial(
    jax.jit,
    static_argnames=(
        "n_pressure",
        "n_diamagnetic",
        "n_radial_cells",
        "n_surface_bins",
        "nonnegative",
    ),
)
def traced_flux_surface_geometry(
    psi2d,
    radius,
    height,
    inside_limiter,
    *,
    axis_psi,
    boundary_psi,
    profile_coefficients,
    coefficient_scale,
    ip_amperes,
    major_radius,
    boundary_toroidal_field,
    field_function_psi_n=None,
    field_function=None,
    n_pressure: int,
    n_diamagnetic: int,
    n_radial_cells: int = 24,
    n_surface_bins: int = 28,
    psi_n_min=0.04,
    psi_n_max=0.985,
    nonnegative: bool = True,
):
    """Build fixed-shape flux-surface geometry directly from an equilibrium grid.

    This device entry point composes Nova's connectivity bins with
    :func:`traced_assemble_flux_surface_geometry`; it is safe under ``jit`` and
    ``vmap`` when slices share a machine grid and the static shape parameters.
    EQDSK-like callers may pass the paired ``field_function`` inputs to preserve
    the reader's measured F and poloidal-flux profiles.
    """
    poloidal_flux_span = boundary_psi - axis_psi
    safe_span = jnp.where(
        jnp.abs(poloidal_flux_span) > 1e-12, poloidal_flux_span, 1e-12
    )
    psi_n_grid = (psi2d - axis_psi) / safe_span
    core = _connectivity_core(psi_n_grid, inside_limiter)
    psi_n_profile = jnp.linspace(0.0, 1.0, 101, dtype=psi2d.dtype)
    if field_function is None:
        scaled_coefficients = profile_coefficients * coefficient_scale
        diamagnetic_profile = (
            _traced_profile_shapes(
                psi_n_profile, n_diamagnetic, nonnegative=nonnegative
            )
            @ scaled_coefficients[n_pressure:]
        )
        f_profile, _ = _integrate_diamagnetic_drive(
            psi_n_profile,
            MU0 * major_radius * diamagnetic_profile,
            boundary_f=major_radius * boundary_toroidal_field,
            poloidal_flux_span=safe_span,
        )
    else:
        f_profile = jnp.interp(
            psi_n_profile,
            jnp.asarray(field_function_psi_n, dtype=psi2d.dtype),
            jnp.asarray(field_function, dtype=psi2d.dtype),
        )
    f_grid = jnp.interp(psi_n_grid, psi_n_profile, f_profile)
    surface_bins, torax_columns = _clipped_surface_geometry(
        psi2d,
        psi_n_grid,
        core,
        radius,
        height,
        f_grid,
        psi_n_min,
        psi_n_max,
        n_surface_bins,
    )
    return traced_assemble_flux_surface_geometry(
        surface_bins,
        psi2d,
        radius,
        height,
        inside_limiter,
        axis_psi=axis_psi,
        boundary_psi=boundary_psi,
        profile_coefficients=profile_coefficients,
        coefficient_scale=coefficient_scale,
        ip_amperes=ip_amperes,
        major_radius=major_radius,
        boundary_toroidal_field=boundary_toroidal_field,
        field_function_psi_n=field_function_psi_n,
        field_function=field_function,
        n_pressure=n_pressure,
        n_diamagnetic=n_diamagnetic,
        n_radial_cells=n_radial_cells,
        nonnegative=nonnegative,
        torax_columns=torax_columns,
    )


def flux_surface_geometry(
    psi2d,
    grid,
    *,
    axis_psi: float,
    boundary_psi: float,
    profile_coefficients: np.ndarray,
    coefficient_scale: np.ndarray,
    ip_amperes: float,
    n_pressure: int,
    n_diamagnetic: int,
    boundary_toroidal_field: float,
    n_radial_cells: int = 24,
    n_surface_bins: int = 28,
    psi_n_min: float = 0.04,
    psi_n_max: float = 0.985,
    nonnegative: bool = True,
    precision: Precision | str = Precision.AUTOMATIC,
) -> FluxSurfaceGeometry | None:
    """Return transport geometry for one equilibrium, or ``None`` if invalid.

    ``grid`` supplies increasing ``rg``/``zg`` coordinates, an
    ``inside_limiter`` mask, and the machine major radius ``r0``. Inputs and
    outputs use raw SI: total poloidal flux [Wb], plasma current [A], toroidal
    field [T], distances [m], and profile-column scales [A m-2].
    """
    resolved = resolve_precision(precision, Precision.DOUBLE)
    np_dtype = np.float32 if resolved is Precision.SINGLE else np.float64
    jax_dtype = jnp.float32 if resolved is Precision.SINGLE else jnp.float64
    expected_coefficients = n_pressure + n_diamagnetic
    coefficients = np.asarray(profile_coefficients, dtype=np_dtype)
    scales = np.asarray(coefficient_scale, dtype=np_dtype)
    if coefficients.shape != (expected_coefficients,):
        raise ValueError(
            "profile_coefficients must have shape "
            f"({expected_coefficients},), got {coefficients.shape}"
        )
    if scales.shape != (expected_coefficients,):
        raise ValueError(
            f"coefficient_scale must have shape ({expected_coefficients},), "
            f"got {scales.shape}"
        )
    if nonnegative and max(n_pressure, n_diamagnetic) > len(NONNEGATIVE_EXPONENTS):
        raise ValueError(
            "the nonnegative ladder supports at most "
            f"{len(NONNEGATIVE_EXPONENTS)} terms per profile family"
        )
    if n_radial_cells < 2:
        raise ValueError("n_radial_cells must be at least 2")
    if n_surface_bins < 2:
        raise ValueError("n_surface_bins must be at least 2")

    assembled = traced_flux_surface_geometry(
        jnp.asarray(np.asarray(psi2d, dtype=np_dtype), dtype=jax_dtype),
        jnp.asarray(np.asarray(grid.rg, dtype=np_dtype), dtype=jax_dtype),
        jnp.asarray(np.asarray(grid.zg, dtype=np_dtype), dtype=jax_dtype),
        jnp.asarray(np.asarray(grid.inside_limiter, dtype=bool)),
        axis_psi=jnp.asarray(axis_psi, dtype=jax_dtype),
        boundary_psi=jnp.asarray(boundary_psi, dtype=jax_dtype),
        profile_coefficients=jnp.asarray(coefficients, dtype=jax_dtype),
        coefficient_scale=jnp.asarray(scales, dtype=jax_dtype),
        ip_amperes=jnp.asarray(ip_amperes, dtype=jax_dtype),
        major_radius=jnp.asarray(grid.r0, dtype=jax_dtype),
        boundary_toroidal_field=jnp.asarray(boundary_toroidal_field, dtype=jax_dtype),
        n_pressure=int(n_pressure),
        n_diamagnetic=int(n_diamagnetic),
        n_radial_cells=int(n_radial_cells),
        n_surface_bins=int(n_surface_bins),
        psi_n_min=jnp.asarray(psi_n_min, dtype=jax_dtype),
        psi_n_max=jnp.asarray(psi_n_max, dtype=jax_dtype),
        nonnegative=bool(nonnegative),
    )
    if not bool(assembled["valid"]):
        return None

    array_fields = {
        field: np.asarray(assembled[field])
        for field in (
            "rho_face",
            "rho_cell",
            "psi_face",
            "psi_n_face",
            "psi_n_cell",
            "vpr_face",
            "vpr_cell",
            "g2_face",
            "g3_face",
            "g3_cell",
            "f_face",
            "f_cell",
            "b2_cell",
            "inv_r_cell",
            "q_face",
        )
    }
    scalar_fields = {
        field: float(assembled[field])
        for field in (
            "phi_b",
            "r0",
            "ip_amperes",
            "axis_psi",
            "boundary_psi",
            "volume",
            "flux_sign",
        )
    }
    return FluxSurfaceGeometry(**array_fields, **scalar_fields)


def poloidal_field_energy_li(geometry: FluxSurfaceGeometry) -> float:
    """Internal inductance ``li3 = 4 Wpol / (mu0 Ip^2 R0)`` from the 1D state."""
    dpsi = np.gradient(geometry.psi_face, geometry.rho_face, edge_order=2)
    vpr = np.clip(geometry.vpr_face, 1e-30, None)
    b_poloidal2 = (dpsi / _TWO_PI) ** 2 * geometry.g2_face / vpr**2
    b_poloidal2[0] = 0.0
    energy = float(np.trapezoid(b_poloidal2 * geometry.vpr_face, geometry.rho_face)) / (
        2.0 * MU0
    )
    return 4.0 * energy / (MU0 * geometry.ip_amperes**2 * geometry.r0)


# ---------------------------------------------------------------------------
# the diffusion step
# ---------------------------------------------------------------------------
def _thomas_solve(sub, diag, sup, rhs):
    """Solve a tridiagonal system by the Thomas algorithm (fixed-shape scans).

    ``sub[i]`` multiplies ``x[i-1]`` and ``sup[i]`` multiplies ``x[i+1]``, so
    ``sub[0]`` and ``sup[-1]`` are unused.  Two ``lax.scan`` sweeps replace the
    dense factorisation: O(n) instead of O(n^3), and jit / vmap / grad-safe.
    """

    def eliminate(carry, row):
        diag_prev, rhs_prev, sup_prev = carry
        sub_i, diag_i, sup_i, rhs_i = row
        factor = sub_i / diag_prev
        diag_i = diag_i - factor * sup_prev
        rhs_i = rhs_i - factor * rhs_prev
        return (diag_i, rhs_i, sup_i), (diag_i, rhs_i)

    _carry, swept = jax.lax.scan(
        eliminate,
        (diag[0], rhs[0], sup[0]),
        (sub[1:], diag[1:], sup[1:], rhs[1:]),
    )
    diag_swept = jnp.concatenate([diag[:1], swept[0]])
    rhs_swept = jnp.concatenate([rhs[:1], swept[1]])

    def substitute(x_next, row):
        diag_i, sup_i, rhs_i = row
        x_i = (rhs_i - sup_i * x_next) / diag_i
        return x_i, x_i

    x_edge = rhs_swept[-1] / diag_swept[-1]
    _carry, x_rest = jax.lax.scan(
        substitute,
        x_edge,
        (diag_swept[:-1], sup[:-1], rhs_swept[:-1]),
        reverse=True,
    )
    return jnp.concatenate([x_rest, x_edge[jnp.newaxis]])


def _diffusion_step(psi_in, d_face, d_mid, toc_face, drho, dt, grad_edge, theta):
    """One implicit theta-scheme sub-step of the flux diffusion.

    Vertex-centred finite volume on the face grid: an interior point owns a cell
    of width ``drho`` with fluxes ``D dpsi/drho`` at the midpoints, while the END
    points own HALF cells so the physical boundary flux sits exactly at ``rho = 0``
    (regularity: zero) and ``rho = 1`` (the prescribed enclosed current AT the
    boundary, not half a cell outside it).
    """
    lam = dt / (toc_face * drho * drho)
    zero = jnp.zeros(1, dtype=psi_in.dtype)

    sub = jnp.concatenate(
        [zero, -theta * lam[1:-1] * d_mid[:-1], -theta * 2.0 * lam[-1:] * d_mid[-1:]]
    )
    sup = jnp.concatenate(
        [-theta * 2.0 * lam[:1] * d_mid[:1], -theta * lam[1:-1] * d_mid[1:], zero]
    )
    diag = jnp.concatenate(
        [
            1.0 + theta * 2.0 * lam[:1] * d_mid[:1],
            1.0 + theta * lam[1:-1] * (d_mid[:-1] + d_mid[1:]),
            1.0 + theta * 2.0 * lam[-1:] * d_mid[-1:],
        ]
    )
    rhs = psi_in.at[-1].add(2.0 * lam[-1] * d_face[-1] * grad_edge * drho)
    explicit = jnp.concatenate(
        [
            2.0 * d_mid[:1] * (psi_in[1:2] - psi_in[:1]),
            d_mid[1:] * (psi_in[2:] - psi_in[1:-1])
            - d_mid[:-1] * (psi_in[1:-1] - psi_in[:-2]),
            -2.0 * d_mid[-1:] * (psi_in[-1:] - psi_in[-2:-1]),
        ]
    )
    rhs = rhs + (1.0 - theta) * lam * explicit
    return _thomas_solve(sub, diag, sup, rhs)


def _diffuse_scan(psi0, d_face, d_mid, toc_face, drho, intervals, grad_edge, theta):
    """Scan the diffusion step over the sub-step intervals, recording voltages."""

    def step(psi, row):
        dt, gradient = row
        advanced = _diffusion_step(
            psi, d_face, d_mid, toc_face, drho, dt, gradient, theta
        )
        # a repeated sample carries no evolution and no loop voltage
        moving = dt > 0.0
        psi_next = jnp.where(moving, advanced, psi)
        safe_dt = jnp.where(moving, dt, 1.0)
        voltage = jnp.where(moving, (psi_next - psi) / safe_dt, 0.0)
        return psi_next, (psi_next, voltage[0], voltage[-1])

    _final, history = jax.lax.scan(step, psi0, (intervals, grad_edge))
    return history


def diffuse_psi(
    geometry: FluxSurfaceGeometry,
    eta,
    *,
    t_grid: np.ndarray,
    ip_of_t: np.ndarray,
    psi0_face: np.ndarray | None = None,
    theta: float = 1.0,
    precision: Precision | str = Precision.AUTOMATIC,
) -> dict:
    """Integrate the flux diffusion over ``t_grid`` with the current condition.

    ``t_grid`` ``(n_t,)`` are the sub-step times [s] and ``ip_of_t`` the measured
    plasma current at those times [A].  ``eta`` is any callable mapping
    normalised poloidal flux to resistivity [Ohm m] -- an :class:`EtaProfile`, or
    a table when an external conductivity model supplies one.

    The geometry is FROZEN over the interval (metrics from the starting slice); a
    recorded approximation the soft prior downstream absorbs.  Implicit
    theta-scheme (``theta = 1`` backward Euler), one jitted tridiagonal solve per
    sub-step. ``precision="auto"`` resolves to float64; explicit float32 selects
    a separate trace.

    Returns ``psi_face`` ``(n_t, n_rho + 1)`` [Wb] plus the flux-budget traces:
    ``v_axis`` and ``v_bdry``, the axis and boundary ``dpsi/dt`` [V], and
    ``psidot_face`` at the final step for the ohmic parallel current.
    """
    resolved = resolve_precision(precision, Precision.DOUBLE)
    np_dtype = np.float32 if resolved is Precision.SINGLE else np.float64
    jax_dtype = jnp.float32 if resolved is Precision.SINGLE else jnp.float64
    times = np.asarray(t_grid, dtype=np_dtype)
    ip = np.asarray(ip_of_t, dtype=np_dtype)
    n_face = geometry.rho_face.size
    drho = float(geometry.rho_face[1] - geometry.rho_face[0])

    sigma_cell = 1.0 / np.asarray(eta(geometry.psi_n_cell), dtype=np_dtype)
    toc_cell = (
        sigma_cell
        * MU0
        * _16PI2
        * geometry.phi_b**2
        * geometry.rho_cell
        / geometry.f_cell**2
    )
    # face-centred state: interpolate the time coefficient onto the faces
    toc_face = np.empty(n_face, dtype=np_dtype)
    toc_face[1:-1] = 0.5 * (toc_cell[:-1] + toc_cell[1:])
    toc_face[0] = toc_cell[0]
    toc_face[-1] = toc_cell[-1]

    d_face = geometry.d_face
    d_mid = 0.5 * (d_face[:-1] + d_face[1:])
    psi0 = np.asarray(
        geometry.psi_face if psi0_face is None else psi0_face, dtype=np_dtype
    )
    intervals = np.diff(times)
    gradients = np.array([geometry.ip_edge_gradient(float(value)) for value in ip[1:]])

    if intervals.size:
        psi_history, v_axis_history, v_bdry_history = _COMPILED_SCAN(
            jnp.asarray(psi0, dtype=jax_dtype),
            jnp.asarray(d_face, dtype=jax_dtype),
            jnp.asarray(d_mid, dtype=jax_dtype),
            jnp.asarray(toc_face, dtype=jax_dtype),
            drho,
            jnp.asarray(intervals, dtype=jax_dtype),
            jnp.asarray(gradients, dtype=jax_dtype),
            float(theta),
        )
        psi_face = np.vstack([psi0, np.asarray(psi_history)])
        v_axis = np.concatenate([[0.0], np.asarray(v_axis_history)])
        v_bdry = np.concatenate([[0.0], np.asarray(v_bdry_history)])
    else:
        psi_face = psi0[np.newaxis, :]
        v_axis = np.zeros(1)
        v_bdry = np.zeros(1)

    if times.size > 1 and (times[-1] - times[-2]) > 0:
        psidot_face = (psi_face[-1] - psi_face[-2]) / (times[-1] - times[-2])
    else:
        psidot_face = np.zeros(n_face)
    return {
        "t": times,
        "psi_face": psi_face,
        "v_axis": v_axis,
        "v_bdry": v_bdry,
        "psidot_face": psidot_face,
    }


with skip_import("jax"):
    #: the traced/compiled diffusion scan; ``drho`` and ``theta`` are static so a
    #: change of scheme retraces rather than silently reusing a stale kernel
    _COMPILED_SCAN = jax.jit(_diffuse_scan, static_argnums=(4, 7))


# ---------------------------------------------------------------------------
# predicted profiles + basis projection
# ---------------------------------------------------------------------------
def predicted_current(
    geometry: FluxSurfaceGeometry,
    psi_face: np.ndarray,
    psidot_face: np.ndarray,
    eta,
) -> dict:
    """Evolved-state current profiles on the cell grid.

    ``j_tor`` is ``dI/dS`` with ``dS = vpr <1/R> drho / 2pi``, the
    flux-surface-averaged toroidal current density.  ``j_par_b`` is the OHMIC
    ``<J.B> = sigma_parallel <E.B>`` with
    ``<E.B> = flux_sign psidot F <1/R^2> / 2pi``.  The ``flux_sign`` factor keeps
    the dissipative channel POSITIVE for a positive normalised current in either
    flux convention -- without it the ohmic target flips sign on one convention and
    a non-negative projection can only reach it by annihilating the profile.
    """
    i_face = geometry.enclosed_current(psi_face)
    surface_per_rho = geometry.vpr_cell * geometry.inv_r_cell / _TWO_PI
    j_tor = np.diff(i_face) / (
        np.diff(geometry.rho_face) * np.clip(surface_per_rho, 1e-30, None)
    )
    psidot_cell = 0.5 * (psidot_face[:-1] + psidot_face[1:])
    sigma_cell = 1.0 / np.asarray(eta(geometry.psi_n_cell), dtype=np.float64)
    e_dot_b = (
        geometry.flux_sign * psidot_cell * geometry.f_cell * geometry.g3_cell / _TWO_PI
    )
    return {"i_face": i_face, "j_tor": j_tor, "j_par_b": sigma_cell * e_dot_b}


#: exponents of the non-negative monomial profile ladder ``(1 - psi_n)**e``
NONNEGATIVE_EXPONENTS = (0.5, 1.0, 1.5, 2.0, 3.0)


def profile_shapes(psi_n: np.ndarray, n_terms: int, *, nonneg: bool) -> np.ndarray:
    """Profile-ladder shape functions on a normalised-flux sample.

    The non-negative arm is the monomial family ``(1 - psi_n)**e``, each term
    edge-vanishing and sign-definite; the general arm is the Legendre family on
    the unit interval times the edge-vanishing factor.  Returns
    ``(psi_n.size, n_terms)``.
    """
    normalised = np.clip(np.asarray(psi_n, dtype=np.float64), 0.0, 1.0)
    if n_terms == 0:
        # an empty ladder is a well-formed request -- a channel may carry no
        # free coefficients -- so it returns the empty basis the projection
        # expects rather than failing to stack nothing
        return np.empty((*normalised.shape, 0), dtype=np.float64)
    if nonneg:
        return np.column_stack(
            [
                (1.0 - normalised) ** NONNEGATIVE_EXPONENTS[term]
                for term in range(n_terms)
            ]
        )
    from numpy.polynomial import legendre

    scaled = 2.0 * normalised - 1.0
    edge = 1.0 - normalised
    return np.column_stack(
        [
            legendre.legval(scaled, [0.0] * term + [1.0]) * edge
            for term in range(n_terms)
        ]
    )


def basis_projection_images(
    geometry: FluxSurfaceGeometry,
    coefficient_scale: np.ndarray,
    *,
    n_pressure: int,
    n_diamagnetic: int,
    nonneg: bool = True,
) -> dict:
    """Per-unit-coefficient current images on the geometry's cell grid.

    ``coefficient_scale`` holds the source fit's per-column amplitude scales, so
    a stored normalised coefficient vector maps back to a physical current.  For
    the pressure-gradient family (drive ``R/R0``) the images are
    ``j_tor = s phi / (R0 <1/R>)`` and ``<J.B> = F s phi / R0``; for the
    diamagnetic family (drive ``R0/R``) they are
    ``j_tor = s phi R0 <1/R^2> / <1/R>`` and ``<J.B> = <B^2> R0 s phi / F``.  The
    differing F weights are the split leverage the parallel Ohm's law provides.
    Both matrices are ``(n_rho, n_pressure + n_diamagnetic)``.
    """
    psi_n = geometry.psi_n_cell
    phi_pressure = profile_shapes(psi_n, n_pressure, nonneg=nonneg)
    phi_diamagnetic = profile_shapes(psi_n, n_diamagnetic, nonneg=nonneg)
    scale = np.asarray(coefficient_scale, dtype=np.float64)
    scale_pressure = scale[:n_pressure]
    scale_diamagnetic = scale[n_pressure:]
    inv_r = np.clip(geometry.inv_r_cell, 1e-9, None)
    f_cell = geometry.f_cell
    a_tor = np.hstack(
        [
            phi_pressure
            * scale_pressure[np.newaxis, :]
            / (geometry.r0 * inv_r[:, np.newaxis]),
            phi_diamagnetic
            * scale_diamagnetic[np.newaxis, :]
            * geometry.r0
            * (geometry.g3_cell / inv_r)[:, np.newaxis],
        ]
    )
    a_par = np.hstack(
        [
            phi_pressure
            * scale_pressure[np.newaxis, :]
            * (f_cell / geometry.r0)[:, np.newaxis],
            phi_diamagnetic
            * scale_diamagnetic[np.newaxis, :]
            * (geometry.b2_cell * geometry.r0 / f_cell)[:, np.newaxis],
        ]
    )
    return {"a_tor": a_tor, "a_par": a_par}


def project_coefficients(
    geometry: FluxSurfaceGeometry,
    images: dict,
    j_tor_predicted: np.ndarray,
    j_par_b_predicted: np.ndarray,
    *,
    nonneg: bool = True,
    parallel_weight: float = 1.0,
    ridge: float = 1e-6,
) -> np.ndarray | None:
    """Predicted profile-ladder coefficients from the evolved 1D profiles.

    Volume-weighted least squares over the cell grid with both current targets
    stacked, each block normalised to a unit root-mean-square target so
    ``parallel_weight`` is a dimensionless balance, and non-negativity per the
    source fit's arm.  Returns ``None`` on a degenerate solve -- the caller then
    skips the prior for that slice rather than fabricating one.
    """
    from scipy import optimize

    weight = np.sqrt(np.clip(geometry.vpr_cell, 0.0, None))
    weight = weight / max(np.linalg.norm(weight), 1e-30)

    def block(matrix, target, block_weight):
        scale = max(float(np.sqrt(np.mean((weight * target) ** 2))), 1e-30)
        gain = np.sqrt(block_weight) / scale
        return gain * (matrix * weight[:, np.newaxis]), gain * (target * weight)

    a_tor, y_tor = block(
        images["a_tor"], np.asarray(j_tor_predicted, dtype=np.float64), 1.0
    )
    a_par, y_par = block(
        images["a_par"],
        np.asarray(j_par_b_predicted, dtype=np.float64),
        parallel_weight,
    )
    n_terms = a_tor.shape[1]
    matrix = np.vstack([a_tor, a_par, np.sqrt(ridge) * np.eye(n_terms)])
    target = np.concatenate([y_tor, y_par, np.zeros(n_terms)])
    try:
        if nonneg:
            coefficients = optimize.lsq_linear(
                matrix,
                target,
                bounds=(np.zeros(n_terms), np.full(n_terms, np.inf)),
            ).x
        else:
            coefficients, *_ = np.linalg.lstsq(matrix, target, rcond=None)
    except ValueError, np.linalg.LinAlgError:
        return None
    return coefficients if np.isfinite(coefficients).all() else None


# ---------------------------------------------------------------------------
# flux-consumption ledger
# ---------------------------------------------------------------------------
def flux_budget(step: dict, geometry: FluxSurfaceGeometry) -> dict:
    """Inductive/resistive decomposition of an interval's flux consumption.

    Working in the equilibrium's own sign, over the integration window:

    * ``d_psi_bdry`` -- the total surface flux swing [Wb];
    * ``d_psi_axis`` -- the RESISTIVE consumption [Wb].  The axis loop voltage is
      purely resistive: no flux is stored inside a zero-volume surface;
    * ``d_psi_internal`` -- the INDUCTIVE storage change [Wb].

    The identity ``d_psi_bdry = d_psi_axis + d_psi_internal`` holds by
    construction; reporting all three keeps both channels explicitly accounted.
    """
    psi = step["psi_face"]
    d_axis = float(psi[-1, 0] - psi[0, 0])
    d_bdry = float(psi[-1, -1] - psi[0, -1])
    multi_step = psi.shape[0] > 1
    return {
        "d_psi_bdry": d_bdry,
        "d_psi_axis": d_axis,
        "d_psi_internal": d_bdry - d_axis,
        "v_axis_mean": float(np.mean(step["v_axis"][1:])) if multi_step else 0.0,
        "v_bdry_mean": float(np.mean(step["v_bdry"][1:])) if multi_step else 0.0,
        "r0": geometry.r0,
    }


def ejima_coefficient(d_psi_resistive: float, d_ip: float, r0: float) -> float:
    """Windowed Ejima coefficient ``|dPsi_res| / (mu0 R0 |dIp|)``.

    The resistive poloidal-flux consumption normalised by ``mu0 R0 dIp`` over the
    same window.  The classical definition runs from breakdown, so a chained
    window starting at the first reconstructed slice gives the INCREMENTAL
    coefficient over the covered ramp -- comparable across shots, and comparable
    to literature values when the window covers most of the ramp.
    """
    return float(abs(d_psi_resistive) / (MU0 * abs(r0) * max(abs(d_ip), 1e-30)))


@dataclass
class CurrentDiffusion:
    """Resistive current-diffusion solver over one frozen-geometry interval.

    The stateful face over this module: hold the interval's 1D metrics and the
    resistivity profile, then :meth:`evolve` the flux, :meth:`predict` the current
    profiles, and :meth:`project` them back onto a coefficient ladder.
    """

    geometry: FluxSurfaceGeometry
    eta: EtaProfile
    theta: float = 1.0
    precision: Precision | str = Precision.AUTOMATIC

    def __post_init__(self):
        self.precision = resolve_precision(self.precision, Precision.DOUBLE)

    def evolve(
        self,
        t_grid: np.ndarray,
        ip_of_t: np.ndarray,
        psi0_face: np.ndarray | None = None,
    ) -> dict:
        """Diffuse the flux across the interval."""
        return diffuse_psi(
            self.geometry,
            self.eta,
            t_grid=t_grid,
            ip_of_t=ip_of_t,
            psi0_face=psi0_face,
            theta=self.theta,
            precision=self.precision,
        )

    def predict(self, step: dict) -> dict:
        """Current profiles of the evolved state at the end of the interval."""
        return predicted_current(
            self.geometry, step["psi_face"][-1], step["psidot_face"], self.eta
        )

    def project(
        self,
        prediction: dict,
        coefficient_scale: np.ndarray,
        *,
        n_pressure: int,
        n_diamagnetic: int,
        nonneg: bool = True,
        parallel_weight: float = 1.0,
    ) -> np.ndarray | None:
        """Project the predicted profiles onto the profile-ladder coefficients."""
        images = basis_projection_images(
            self.geometry,
            coefficient_scale,
            n_pressure=n_pressure,
            n_diamagnetic=n_diamagnetic,
            nonneg=nonneg,
        )
        return project_coefficients(
            self.geometry,
            images,
            prediction["j_tor"],
            prediction["j_par_b"],
            nonneg=nonneg,
            parallel_weight=parallel_weight,
        )

    def budget(self, step: dict) -> dict:
        """Flux-consumption ledger of the interval."""
        return flux_budget(step, self.geometry)


__all__ = [
    "NONNEGATIVE_EXPONENTS",
    "CurrentDiffusion",
    "EtaProfile",
    "FluxSurfaceGeometry",
    "traced_assemble_flux_surface_geometry",
    "basis_projection_images",
    "diffuse_psi",
    "ejima_coefficient",
    "flux_budget",
    "flux_surface_geometry",
    "traced_flux_surface_geometry",
    "poloidal_field_energy_li",
    "predicted_current",
    "profile_shapes",
    "project_coefficients",
]
