"""COCOS 17 sign receipts at the Nova-to-TORAX geometry boundary.

COCOS identifies coordinate senses, not the polarity of one discharge.  The
single-sign perturbations below therefore isolate the boundary normalisation
for each transported quantity while the pinned convention remains COCOS 17.
"""

from dataclasses import dataclass, replace

import numpy as np
import pytest

from nova.io.cocos import convention
from nova.transport import torax_geometry_from_fsa


@dataclass(frozen=True)
class _CocosState:
    psi_face: np.ndarray
    ip_profile_face: np.ndarray
    f_face: np.ndarray
    phi_boundary: float
    pprime_face: np.ndarray
    ffprime_face: np.ndarray


@dataclass(frozen=True)
class _SignProvenance:
    cocos: int
    flux_sign: float
    current_sign: float
    field_sign: float
    toroidal_flux_sign: float


@dataclass(frozen=True)
class _ToraxState:
    geometry: object
    pprime_face: np.ndarray
    ffprime_face: np.ndarray


def _cell(values):
    values = np.asarray(values)
    return 0.5 * (values[:-1] + values[1:])


def _base_state() -> _CocosState:
    return _CocosState(
        psi_face=np.array([0.60, 0.49, 0.36, 0.21, 0.05]),
        ip_profile_face=np.array([0.0, 7.0e4, 1.9e5, 3.3e5, 5.0e5]),
        f_face=np.array([6.15, 6.13, 6.10, 6.06, 6.02]),
        phi_boundary=2.8,
        pprime_face=np.array([8.0e4, 6.8e4, 5.0e4, 2.8e4, 1.0e4]),
        ffprime_face=np.array([-1.8, -1.55, -1.2, -0.75, -0.3]),
    )


def _flip_one(state: _CocosState, quantity: str) -> _CocosState:
    if quantity == "psi":
        return replace(state, psi_face=-state.psi_face)
    if quantity == "ip":
        return replace(state, ip_profile_face=-state.ip_profile_face)
    if quantity == "f":
        return replace(state, f_face=-state.f_face)
    if quantity == "phi":
        return replace(state, phi_boundary=-state.phi_boundary)
    if quantity == "pprime":
        return replace(state, pprime_face=-state.pprime_face)
    if quantity == "ffprime":
        return replace(state, ffprime_face=-state.ffprime_face)
    raise AssertionError(f"unsupported quantity {quantity!r}")


def _provenance(state: _CocosState) -> _SignProvenance:
    return _SignProvenance(
        cocos=17,
        flux_sign=float(np.sign(state.psi_face[-1] - state.psi_face[0])),
        current_sign=float(np.sign(state.ip_profile_face[-1])),
        field_sign=float(np.sign(state.f_face[-1])),
        toroidal_flux_sign=float(np.sign(state.phi_boundary)),
    )


def _fsa_record(state: _CocosState, provenance: _SignProvenance):
    """Return the positive-toroidal-flux FSA magnitude plus signed profiles."""
    rho_face = np.linspace(0.0, 1.0, state.psi_face.size)
    return {
        "cocos": provenance.cocos,
        "rho_face": rho_face,
        "vpr_face": 36.0 * rho_face,
        "inv_r_face": np.full_like(rho_face, 1 / 3.0),
        "g3_face": np.full_like(rho_face, 1 / 9.0),
        "int_dl_over_bp_face": np.array([0.0, 6.0, 7.0, 8.0, 9.0]),
        "grad_psi_face": np.array([0.0, 0.09, 0.14, 0.19, 0.23]),
        "grad_psi2_face": np.array([0.0, 0.010, 0.022, 0.040, 0.061]),
        "grad_psi2_over_r2_face": np.array([0.0, 0.001, 0.003, 0.005, 0.007]),
        "f_face": state.f_face,
        "flux_sign": provenance.flux_sign,
        "psi_face": state.psi_face,
        "ip_profile_face": state.ip_profile_face,
        "phi_b": abs(state.phi_boundary),
        "r_in_face": 3.0 - 0.6 * rho_face,
        "r_out_face": 3.0 + 0.6 * rho_face,
        "volume_face": 28.0 * rho_face**2,
        "delta_upper_face": 0.18 * rho_face,
        "delta_lower_face": 0.16 * rho_face,
        "elongation_face": 1.0 + 0.5 * rho_face,
        "b2_face": 4.0 + 0.3 * rho_face,
        "inv_b2_face": 1.0 / (4.0 + 0.3 * rho_face),
    }


def _to_torax(state: _CocosState) -> tuple[_ToraxState, _SignProvenance]:
    provenance = _provenance(state)
    record = _fsa_record(state, provenance)
    geometry = torax_geometry_from_fsa(record)
    return (
        _ToraxState(
            geometry=geometry,
            pprime_face=provenance.flux_sign * state.pprime_face,
            ffprime_face=provenance.flux_sign * state.ffprime_face,
        ),
        provenance,
    )


def _to_cocos(torax_state: _ToraxState, provenance: _SignProvenance) -> _CocosState:
    geometry = torax_state.geometry
    return _CocosState(
        psi_face=provenance.flux_sign * np.asarray(geometry.psi),
        ip_profile_face=(
            provenance.current_sign * np.asarray(geometry.Ip_profile_face)
        ),
        f_face=provenance.field_sign * np.asarray(geometry.F_face),
        phi_boundary=(
            provenance.toroidal_flux_sign * float(np.asarray(geometry.Phi_face)[-1])
        ),
        pprime_face=provenance.flux_sign * torax_state.pprime_face,
        ffprime_face=provenance.flux_sign * torax_state.ffprime_face,
    )


@pytest.mark.parametrize(
    "flipped_quantity", ["psi", "ip", "f", "phi", "pprime", "ffprime"]
)
def test_each_independent_sign_round_trips_through_torax(flipped_quantity):
    """Each polarity survives positive TORAX normalisation via provenance."""
    state = _flip_one(_base_state(), flipped_quantity)
    torax_state, provenance = _to_torax(state)
    geometry = torax_state.geometry

    cocos_seventeen = convention(provenance.cocos)
    assert cocos_seventeen.digits == (-1, 1, 1, 1)
    assert cocos_seventeen.e_bp == 1

    np.testing.assert_allclose(
        np.asarray(geometry.psi), provenance.flux_sign * _cell(state.psi_face)
    )
    assert np.all(np.diff(np.asarray(geometry.psi)) > 0.0)
    np.testing.assert_allclose(
        np.asarray(geometry.Ip_profile_face), np.abs(state.ip_profile_face)
    )
    np.testing.assert_allclose(np.asarray(geometry.F_face), np.abs(state.f_face))
    np.testing.assert_allclose(
        np.asarray(geometry.Phi_face),
        abs(state.phi_boundary) * np.linspace(0.0, 1.0, state.psi_face.size) ** 2,
    )
    np.testing.assert_allclose(
        torax_state.pprime_face, provenance.flux_sign * state.pprime_face
    )
    np.testing.assert_allclose(
        torax_state.ffprime_face, provenance.flux_sign * state.ffprime_face
    )

    restored = _to_cocos(torax_state, provenance)
    np.testing.assert_allclose(restored.psi_face, _cell(state.psi_face))
    np.testing.assert_allclose(restored.ip_profile_face, state.ip_profile_face)
    np.testing.assert_allclose(restored.f_face, state.f_face)
    assert restored.phi_boundary == pytest.approx(state.phi_boundary)
    np.testing.assert_allclose(restored.pprime_face, state.pprime_face)
    np.testing.assert_allclose(restored.ffprime_face, state.ffprime_face)
