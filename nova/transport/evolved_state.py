r"""Turn an evolved one-dimensional transport state into equilibrium sources.

The transport receipt carries temperatures, electron density and poloidal flux,
while its flux-surface geometry carries the enclosed-current shape and the
surface averages needed to separate pressure and diamagnetic drive.  Together
they determine the static Grad-Shafranov source without passing a sampled
current image into the equilibrium solver.

All flux derivatives returned here use Nova's negated-total-flux convention.
For an enclosed toroidal current :math:`I_p(V)`, flux-surface averaging gives

.. math::

    \frac{d I_p}{dV}
      = -\left(p' + \frac{FF'}{\mu_0}\left\langle R^{-2}\right\rangle\right),

so the evolved pressure and current profile uniquely recover ``FFprime`` on
the declared FSA geometry.  The resulting interpolants are callables of
normalised flux and therefore cross :class:`nova.equilibrium.DomainProfile`'s
physical-source boundary; the raw samples never masquerade as a cell image.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from scipy.constants import electron_volt, mu_0

from nova.equilibrium.source import DomainProfile, ForwardSource
from nova.transport.forward import (
    ForwardTransportReceipt,
    TransportGeometry,
    TransportRung,
)

__all__ = ["EvolvedFluxFunction", "forward_source_from_receipt"]

_KEV_JOULES = 1.0e3 * electron_volt
_REQUIRED_FACE_FIELDS = (
    "rho_face",
    "ip_profile_face",
    "volume_face",
    "g3_face",
    "f_face",
    "flux_sign",
)


def _is_traced(value: object) -> bool:
    """Return whether validation would force a JAX tracer onto the host."""
    return isinstance(value, jax.core.Tracer)


def _host_validate_finite(name: str, value) -> None:
    """Validate a host value while leaving differentiated calls traceable."""
    if _is_traced(value):
        return
    array = np.asarray(value)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")


def _host_validate_increasing(name: str, value) -> None:
    """Validate a host coordinate while leaving differentiated calls traceable."""
    if _is_traced(value):
        return
    array = np.asarray(value)
    if not np.all(np.diff(array) > 0.0):
        raise ValueError(f"{name} must be strictly increasing")


def _host_validate_close(name: str, left, right, *, scale: float) -> None:
    """Reject a contradictory host receipt without constraining tracers."""
    if _is_traced(left) or _is_traced(right):
        return
    residual = abs(float(np.asarray(left)) - float(np.asarray(right)))
    tolerance = 1.0e-10 * max(float(scale), 1.0)
    if residual > tolerance:
        raise ValueError(
            f"{name} is inconsistent by {residual:.6g}, above {tolerance:.6g}"
        )


def _host_scale(value) -> float:
    """Return a host magnitude for a consistency tolerance."""
    if _is_traced(value):
        return 1.0
    return abs(float(np.asarray(value)))


def _quadratic_gradient(value, coordinate):
    """Return a second-order derivative on one nonuniform radial coordinate."""
    value = jnp.asarray(value)
    coordinate = jnp.asarray(coordinate)
    left_step = coordinate[1:-1] - coordinate[:-2]
    right_step = coordinate[2:] - coordinate[1:-1]
    interior = (
        -right_step / (left_step * (left_step + right_step)) * value[:-2]
        + (right_step - left_step) / (left_step * right_step) * value[1:-1]
        + left_step / (right_step * (left_step + right_step)) * value[2:]
    )

    first_step = coordinate[1] - coordinate[0]
    second_step = coordinate[2] - coordinate[1]
    first = (
        -(2.0 * first_step + second_step)
        / (first_step * (first_step + second_step))
        * value[0]
        + (first_step + second_step) / (first_step * second_step) * value[1]
        - first_step / (second_step * (first_step + second_step)) * value[2]
    )

    penultimate_step = coordinate[-2] - coordinate[-3]
    final_step = coordinate[-1] - coordinate[-2]
    final = (
        final_step / (penultimate_step * (penultimate_step + final_step)) * value[-3]
        - (penultimate_step + final_step) / (penultimate_step * final_step) * value[-2]
        + (penultimate_step + 2.0 * final_step)
        / (final_step * (penultimate_step + final_step))
        * value[-1]
    )
    return jnp.concatenate((jnp.atleast_1d(first), interior, jnp.atleast_1d(final)))


@dataclass(frozen=True)
class EvolvedFluxFunction:
    """JAX-traceable interpolation of one physical flux-function gradient."""

    normalised_flux: jax.Array
    value: jax.Array

    def __post_init__(self) -> None:
        coordinate = jnp.asarray(self.normalised_flux)
        value = jnp.asarray(self.value)
        if coordinate.ndim != 1 or coordinate.size < 3:
            raise ValueError("an evolved flux function needs at least three faces")
        if value.shape != coordinate.shape:
            raise ValueError("flux-function values must match the radial coordinate")
        _host_validate_finite("normalised flux", coordinate)
        _host_validate_finite("flux-function values", value)
        _host_validate_increasing("normalised flux", coordinate)
        object.__setattr__(self, "normalised_flux", coordinate)
        object.__setattr__(self, "value", value)

    def __call__(self, normalised_flux):
        """Evaluate the evolved gradient on normalised poloidal flux."""
        return jnp.interp(
            jnp.asarray(normalised_flux), self.normalised_flux, self.value
        )


def _face_field(record, name: str, geometry_rho, state_rho):
    """Interpolate one finite FSA face field onto the receipt's radial grid."""
    value = jnp.asarray(record[name])
    if value.shape != geometry_rho.shape:
        raise ValueError(
            f"transport geometry {name} must match its rho_face coordinate"
        )
    _host_validate_finite(f"transport geometry {name}", value)
    return jnp.interp(state_rho, geometry_rho, value)


def _physical_flux(receipt: ForwardTransportReceipt, geometry: TransportGeometry):
    """Restore the evolved flux to Nova's signed total-flux convention."""
    flux = jnp.asarray(receipt.state.psi)
    if receipt.provenance.rung is TransportRung.TORAX_MULTI_CHANNEL:
        flux = jnp.asarray(geometry.record["flux_sign"]) * flux
    return flux


def forward_source_from_receipt(
    receipt: ForwardTransportReceipt,
    geometry: TransportGeometry,
    *,
    ion_density_per_electron: float,
) -> ForwardSource:
    """Map one evolved transport receipt into an absolute equilibrium source.

    Parameters
    ----------
    receipt:
        The evolved state and its conservation/current ledgers.
    geometry:
        The flux-surface geometry used for the transport interval.  Its
        enclosed-current profile supplies the evolved radial current shape;
        the receipt's achieved edge current fixes that shape's amplitude.
    ion_density_per_electron:
        The explicitly declared ratio :math:`n_i/n_e` used to recover thermal
        pressure from the transported electron density and the two
        temperatures.  ``1.0`` is the singly charged quasineutral closure.

    Returns
    -------
    ForwardSource
        A static, absolute source whose core gradients are JAX-callable
        functions of normalised flux.  Boundary pressure is in Pa and the
        boundary field function is in T m.
    """
    if not isinstance(receipt, ForwardTransportReceipt):
        raise TypeError("receipt must be a ForwardTransportReceipt")
    if not isinstance(geometry, TransportGeometry):
        raise TypeError("geometry must be a TransportGeometry")
    if ion_density_per_electron <= 0.0 or not np.isfinite(ion_density_per_electron):
        raise ValueError("ion_density_per_electron must be finite and positive")

    missing = [name for name in _REQUIRED_FACE_FIELDS if name not in geometry.record]
    if missing:
        raise ValueError(
            "transport geometry is missing return-channel fields: " + ", ".join(missing)
        )

    state = receipt.state
    size = state.rho.size
    if size < 3:
        raise ValueError("an evolved transport state needs at least three faces")
    state_rho = jnp.asarray(state.rho)
    geometry_rho = jnp.asarray(geometry.record["rho_face"])
    if geometry_rho.ndim != 1 or geometry_rho.size < 3:
        raise ValueError("transport geometry rho_face needs at least three faces")
    _host_validate_finite("transport geometry rho_face", geometry_rho)
    _host_validate_increasing("transport geometry rho_face", geometry_rho)
    _host_validate_close(
        "state/geometry radial axis", state_rho[0], geometry_rho[0], scale=1.0
    )
    _host_validate_close(
        "state/geometry radial boundary",
        state_rho[-1],
        geometry_rho[-1],
        scale=1.0,
    )
    _host_validate_close(
        "receipt boundary flux",
        receipt.boundary.psi,
        state.psi[-1],
        scale=_host_scale(state.psi[-1]),
    )
    _host_validate_close(
        "receipt boundary current",
        receipt.boundary.plasma_current,
        receipt.plasma_current.achieved_final,
        scale=_host_scale(receipt.plasma_current.achieved_final),
    )
    _host_validate_close(
        "flux-consumption ledger",
        receipt.flux_consumption.boundary,
        receipt.flux_consumption.resistive + receipt.flux_consumption.internal,
        scale=_host_scale(receipt.flux_consumption.boundary),
    )

    physical_flux = _physical_flux(receipt, geometry)
    flux_span = physical_flux[-1] - physical_flux[0]
    if not _is_traced(flux_span) and abs(float(np.asarray(flux_span))) <= 0.0:
        raise ValueError("evolved axis and boundary flux must differ")
    profile_slice = (
        slice(1, None)
        if receipt.provenance.rung is TransportRung.TORAX_MULTI_CHANNEL
        else slice(None)
    )
    profile_rho = state_rho[profile_slice]
    profile_flux = physical_flux[profile_slice]
    normalised_flux = (profile_flux - physical_flux[0]) / flux_span
    _host_validate_increasing("evolved normalised flux", normalised_flux)

    electron_density = jnp.asarray(state.electron_density)
    pressure = (
        electron_density
        * _KEV_JOULES
        * (
            jnp.asarray(state.electron_temperature)
            + ion_density_per_electron * jnp.asarray(state.ion_temperature)
        )
    )
    _host_validate_finite("evolved pressure", pressure)
    if not _is_traced(pressure) and np.any(np.asarray(pressure) < 0.0):
        raise ValueError("evolved thermal pressure must be nonnegative")
    pressure = pressure[profile_slice]
    pressure_gradient = -_quadratic_gradient(pressure, profile_flux)

    reference_current = _face_field(
        geometry.record, "ip_profile_face", geometry_rho, profile_rho
    )
    reference_edge = reference_current[-1]
    if not _is_traced(reference_edge) and abs(float(np.asarray(reference_edge))) <= 0.0:
        raise ValueError("transport geometry needs a nonzero edge current")
    evolved_current = (
        reference_current
        * jnp.asarray(receipt.plasma_current.achieved_final)
        / reference_edge
    )
    volume = _face_field(geometry.record, "volume_face", geometry_rho, profile_rho)
    _host_validate_increasing("transport enclosed volume", volume)
    current_per_volume = _quadratic_gradient(evolved_current, volume)
    inverse_radius_squared = _face_field(
        geometry.record, "g3_face", geometry_rho, profile_rho
    )
    if not _is_traced(inverse_radius_squared) and np.any(
        np.asarray(inverse_radius_squared) <= 0.0
    ):
        raise ValueError("transport geometry g3_face must be positive")
    diamagnetic_gradient = (
        -mu_0 * (current_per_volume + pressure_gradient) / inverse_radius_squared
    )

    field_function = _face_field(geometry.record, "f_face", geometry_rho, profile_rho)
    return ForwardSource(
        core=DomainProfile(
            p_prime=EvolvedFluxFunction(normalised_flux, pressure_gradient),
            ff_prime=EvolvedFluxFunction(normalised_flux, diamagnetic_gradient),
        ),
        boundary_pressure=pressure[-1],
        boundary_field_function=field_function[-1],
    )
