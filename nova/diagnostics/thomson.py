r"""Predict a calibrated Thomson chord from a solved flux map.

The equilibrium is context produced by the Grad--Shafranov forward: its flux
map and axis/boundary normalisation are consumed, never re-supplied as
diagnostic inputs.  Everything else is explicit and listed in
``THOMSON_FORWARD_INPUTS``.  In particular, the model does not locate an edge,
extract a crossing, or compare against a target while predicting.

The spectral model is the non-relativistic, incoherent Thomson approximation.
It uses the unpolarised differential cross section and a Gaussian Doppler
width.  This is deliberately a forward detector model: spatial quadrature maps
the supplied electron profiles through the solved flux map, then the declared
optical bands turn the scattered spectrum into photoelectrons.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike, NDArray

from nova.equilibrium.observation_kernels import NOVA_COCOS, synthesize_thomson

__all__ = [
    "THOMSON_FORWARD_INPUTS",
    "ForwardEquilibrium",
    "IndependentElectronProfiles",
    "InputDeclaration",
    "ThomsonChordGeometry",
    "ThomsonComparison",
    "ThomsonInstrumentResponse",
    "ThomsonPrediction",
    "ThomsonScatteringPhysics",
    "compare_thomson_prediction",
    "predict_thomson",
]

_PLANCK_J_S = 6.62607015e-34
_LIGHT_SPEED_M_S = 299792458.0
_ELECTRON_REST_ENERGY_EV = 510998.95
_CLASSICAL_ELECTRON_RADIUS_M = 2.8179403262e-15
_QUADRATURE_NODES, _QUADRATURE_WEIGHTS = np.polynomial.hermite.hermgauss(5)
_QUADRATURE_WEIGHTS = _QUADRATURE_WEIGHTS / np.sqrt(pi)


@dataclass(frozen=True)
class InputDeclaration:
    """One diagnostic input and why the equilibrium cannot provide it."""

    name: str
    category: str
    absent_from_gs_forward: str


THOMSON_FORWARD_INPUTS = (
    InputDeclaration(
        "scattering_positions_m",
        "chord geometry",
        "the equilibrium has no diagnostic location or channel identity",
    ),
    InputDeclaration(
        "scattering_volume_sigma_m",
        "instrument geometry",
        "the equilibrium has no laser or collection-volume extent",
    ),
    InputDeclaration(
        "laser_wavelength_m, laser_energy_j, scattering_angle_rad",
        "scattering physics",
        "the equilibrium carries fields and force balance, not a laser",
    ),
    InputDeclaration(
        "spectral_band_edges_m, throughput, solid_angle_sr, scattering_length_m",
        "instrument response",
        "the equilibrium has no polychromator, collection optics, or calibration",
    ),
    InputDeclaration(
        "electron_temperature_ev, electron_density_m3",
        "independent electron profiles",
        "Grad--Shafranov pressure and current profiles do not determine "
        "electron temperature and density separately",
    ),
)


def _finite_array(value: ArrayLike, name: str, ndim: int) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != ndim or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite {ndim}-dimensional array")
    return array


@dataclass(frozen=True)
class ForwardEquilibrium:
    """The solved Grad--Shafranov context consumed by the diagnostic."""

    radius_m: ArrayLike
    height_m: ArrayLike
    flux_wb: ArrayLike
    axis_flux_wb: float
    boundary_flux_wb: float
    identity: str
    cocos: int = NOVA_COCOS

    def __post_init__(self) -> None:
        radius = _finite_array(self.radius_m, "radius_m", 1)
        height = _finite_array(self.height_m, "height_m", 1)
        flux = _finite_array(self.flux_wb, "flux_wb", 2)
        if radius.size < 2 or height.size < 2:
            raise ValueError("the equilibrium grid needs at least two nodes per axis")
        if flux.shape != (radius.size, height.size):
            raise ValueError("flux_wb shape must match the radius and height axes")
        if np.any(np.diff(radius) <= 0.0) or np.any(np.diff(height) <= 0.0):
            raise ValueError("equilibrium grid axes must be strictly increasing")
        if int(self.cocos) != NOVA_COCOS:
            raise ValueError(f"Thomson prediction requires Nova COCOS {NOVA_COCOS}")
        if not self.identity:
            raise ValueError("equilibrium identity must be nonempty")
        span = float(self.boundary_flux_wb) - float(self.axis_flux_wb)
        if not np.isfinite(span) or span == 0.0:
            raise ValueError("axis and boundary flux must define a finite nonzero span")


@dataclass(frozen=True)
class ThomsonChordGeometry:
    """Scattering-volume centres and their extent along one chord."""

    name: str
    channel_names: tuple[str, ...]
    scattering_positions_m: ArrayLike
    chord_direction_rz: tuple[float, float]
    scattering_volume_sigma_m: ArrayLike

    def __post_init__(self) -> None:
        positions = _finite_array(
            self.scattering_positions_m, "scattering_positions_m", 2
        )
        sigma = _finite_array(
            self.scattering_volume_sigma_m, "scattering_volume_sigma_m", 1
        )
        direction = _finite_array(self.chord_direction_rz, "chord_direction_rz", 1)
        if positions.shape[1:] != (2,) or positions.shape[0] == 0:
            raise ValueError("scattering_positions_m must have shape (channel, 2)")
        names_match = len(self.channel_names) == positions.shape[0]
        names_unique = len(set(self.channel_names)) == len(self.channel_names)
        if not names_match or not names_unique:
            raise ValueError("channel names must be unique and match the positions")
        if sigma.shape != (positions.shape[0],) or np.any(sigma <= 0.0):
            raise ValueError("each channel needs a strictly positive spatial sigma")
        if direction.shape != (2,) or not np.isclose(np.linalg.norm(direction), 1.0):
            raise ValueError("chord_direction_rz must be a unit R-Z vector")
        if not self.name:
            raise ValueError("chord name must be nonempty")


@dataclass(frozen=True)
class ThomsonScatteringPhysics:
    """Laser and scattering parameters outside the equilibrium state."""

    laser_wavelength_m: float
    laser_energy_j: float
    scattering_angle_rad: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.laser_wavelength_m) or self.laser_wavelength_m <= 0.0:
            raise ValueError("laser wavelength must be finite and positive")
        if not np.isfinite(self.laser_energy_j) or self.laser_energy_j <= 0.0:
            raise ValueError("laser energy must be finite and positive")
        if not 0.0 < float(self.scattering_angle_rad) < pi:
            raise ValueError("scattering angle must lie strictly between zero and pi")

    @property
    def differential_cross_section_m2_sr(self) -> float:
        """Return the unpolarised classical differential cross section."""
        cosine = np.cos(float(self.scattering_angle_rad))
        return 0.5 * _CLASSICAL_ELECTRON_RADIUS_M**2 * (1.0 + cosine**2)


@dataclass(frozen=True)
class ThomsonInstrumentResponse:
    """Collection geometry and spectral response of one polychromator."""

    spectral_band_edges_m: ArrayLike
    throughput: ArrayLike
    solid_angle_sr: float
    scattering_length_m: float
    quantum_efficiency: float
    background_photoelectrons: ArrayLike

    def __post_init__(self) -> None:
        edges = _finite_array(self.spectral_band_edges_m, "spectral_band_edges_m", 2)
        throughput = _finite_array(self.throughput, "throughput", 1)
        background = _finite_array(
            self.background_photoelectrons, "background_photoelectrons", 1
        )
        if edges.shape[1:] != (2,) or np.any(edges[:, 1] <= edges[:, 0]):
            raise ValueError("spectral bands must have finite increasing edge pairs")
        if throughput.shape != (edges.shape[0],) or np.any(
            (throughput < 0.0) | (throughput > 1.0)
        ):
            raise ValueError("throughput must give one fraction in [0, 1] per band")
        if background.shape != (edges.shape[0],) or np.any(background < 0.0):
            raise ValueError("background must give one nonnegative count per band")
        for name in ("solid_angle_sr", "scattering_length_m", "quantum_efficiency"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.quantum_efficiency > 1.0:
            raise ValueError("quantum_efficiency cannot exceed one")


@dataclass(frozen=True)
class IndependentElectronProfiles:
    """Electron profiles not determined separately by Grad--Shafranov balance."""

    psi_norm: ArrayLike
    electron_temperature_ev: ArrayLike
    electron_density_m3: ArrayLike
    provenance: str

    def __post_init__(self) -> None:
        support = _finite_array(self.psi_norm, "psi_norm", 1)
        temperature = _finite_array(
            self.electron_temperature_ev, "electron_temperature_ev", 1
        )
        density = _finite_array(self.electron_density_m3, "electron_density_m3", 1)
        if support.size < 2 or np.any(np.diff(support) <= 0.0):
            raise ValueError("profile psi_norm must be strictly increasing")
        if temperature.shape != support.shape or np.any(temperature <= 0.0):
            raise ValueError(
                "electron temperature must be positive on the profile support"
            )
        if density.shape != support.shape or np.any(density <= 0.0):
            raise ValueError("electron density must be positive on the profile support")
        if not self.provenance:
            raise ValueError("independent profiles require provenance")


@dataclass(frozen=True)
class ThomsonPrediction:
    """One chord prediction in calibrated and raw detector coordinates."""

    equilibrium_identity: str
    chord_name: str
    channel_names: tuple[str, ...]
    positions_m: NDArray[np.float64]
    psi_norm: jax.Array
    supported: jax.Array
    electron_temperature_ev: jax.Array
    electron_density_m3: jax.Array
    spectral_photoelectrons: jax.Array
    spectral_width_m: jax.Array
    input_declarations: tuple[InputDeclaration, ...]


@dataclass(frozen=True)
class ThomsonComparison:
    """Numerical disagreement between a prediction and one measured trace."""

    compared_channels: int
    temperature_rmse_ev: float
    temperature_median_absolute_error_ev: float
    temperature_normalised_rmse: float
    density_rmse_m3: float
    density_median_absolute_error_m3: float
    density_normalised_rmse: float


def _spectral_band_fraction(
    centre_m: float, sigma_m: jax.Array, band_edges_m: ArrayLike
) -> jax.Array:
    edges = jnp.asarray(band_edges_m, dtype=sigma_m.dtype)
    scale = jnp.sqrt(2.0) * sigma_m[..., None]
    upper = (edges[:, 1] - centre_m) / scale
    lower = (edges[:, 0] - centre_m) / scale
    return 0.5 * (jax.scipy.special.erf(upper) - jax.scipy.special.erf(lower))


def predict_thomson(
    equilibrium: ForwardEquilibrium,
    geometry: ThomsonChordGeometry,
    scattering: ThomsonScatteringPhysics,
    instrument: ThomsonInstrumentResponse,
    profiles: IndependentElectronProfiles,
) -> ThomsonPrediction:
    """Predict calibrated profiles and detector counts without inversion."""

    positions = np.asarray(geometry.scattering_positions_m, dtype=np.float64)
    sigma = np.asarray(geometry.scattering_volume_sigma_m, dtype=np.float64)
    direction = np.asarray(geometry.chord_direction_rz, dtype=np.float64)
    samples = positions[:, None, :] + (
        np.sqrt(2.0)
        * sigma[:, None, None]
        * _QUADRATURE_NODES[None, :, None]
        * direction[None, None, :]
    )
    sampled = synthesize_thomson(
        equilibrium.radius_m,
        equilibrium.height_m,
        equilibrium.flux_wb,
        profiles.psi_norm,
        profiles.electron_temperature_ev,
        profiles.electron_density_m3,
        samples.reshape(-1, 2),
        axis_flux=equilibrium.axis_flux_wb,
        boundary_flux=equilibrium.boundary_flux_wb,
        cocos=equilibrium.cocos,
    )
    channel_count = positions.shape[0]
    weights = jnp.asarray(_QUADRATURE_WEIGHTS, dtype=jnp.float64)[None, :]
    temperature_samples = sampled.electron_temperature.reshape(channel_count, -1)
    density_samples = sampled.electron_density.reshape(channel_count, -1)
    psi_samples = sampled.psi_norm.reshape(channel_count, -1)
    supported_samples = sampled.receipt.interpolation_support.supported.reshape(
        channel_count, -1
    )
    supported = jnp.all(supported_samples, axis=1)
    temperature = jnp.sum(temperature_samples * weights, axis=1)
    density = jnp.sum(density_samples * weights, axis=1)
    psi_norm = jnp.sum(psi_samples * weights, axis=1)

    thermal_ratio = jnp.maximum(temperature, 0.0) / _ELECTRON_REST_ENERGY_EV
    spectral_width = (
        scattering.laser_wavelength_m
        * 2.0
        * np.sin(0.5 * scattering.scattering_angle_rad)
        * jnp.sqrt(thermal_ratio)
    )
    fractions = _spectral_band_fraction(
        scattering.laser_wavelength_m,
        spectral_width,
        instrument.spectral_band_edges_m,
    )
    incident_photons = scattering.laser_energy_j / (
        _PLANCK_J_S * _LIGHT_SPEED_M_S / scattering.laser_wavelength_m
    )
    total_scattered = (
        incident_photons
        * density
        * instrument.scattering_length_m
        * scattering.differential_cross_section_m2_sr
        * instrument.solid_angle_sr
    )
    photoelectrons = (
        total_scattered[:, None]
        * fractions
        * jnp.asarray(instrument.throughput)[None, :]
        * instrument.quantum_efficiency
        + jnp.asarray(instrument.background_photoelectrons)[None, :]
    )
    photoelectrons = jnp.where(supported[:, None], photoelectrons, jnp.nan)
    return ThomsonPrediction(
        equilibrium_identity=equilibrium.identity,
        chord_name=geometry.name,
        channel_names=geometry.channel_names,
        positions_m=positions,
        psi_norm=psi_norm,
        supported=supported,
        electron_temperature_ev=temperature,
        electron_density_m3=density,
        spectral_photoelectrons=photoelectrons,
        spectral_width_m=spectral_width,
        input_declarations=THOMSON_FORWARD_INPUTS,
    )


def _comparison_metrics(
    predicted: np.ndarray, measured: np.ndarray
) -> tuple[float, float, float]:
    difference = predicted - measured
    rmse = float(np.sqrt(np.mean(difference**2)))
    median = float(np.median(np.abs(difference)))
    scale = float(np.sqrt(np.mean(measured**2)))
    return rmse, median, rmse / max(scale, np.finfo(float).tiny)


def compare_thomson_prediction(
    prediction: ThomsonPrediction,
    measured_temperature_ev: ArrayLike,
    measured_density_m3: ArrayLike,
) -> ThomsonComparison:
    """State disagreement with a measured trace without changing the model."""

    measured_temperature = np.asarray(measured_temperature_ev, dtype=np.float64)
    measured_density = np.asarray(measured_density_m3, dtype=np.float64)
    predicted_temperature = np.asarray(prediction.electron_temperature_ev)
    predicted_density = np.asarray(prediction.electron_density_m3)
    expected_shape = (len(prediction.channel_names),)
    if (
        measured_temperature.shape != expected_shape
        or measured_density.shape != expected_shape
    ):
        raise ValueError("measured traces must give one value per predicted channel")
    selected = (
        np.asarray(prediction.supported, dtype=bool)
        & np.isfinite(predicted_temperature)
        & np.isfinite(predicted_density)
        & np.isfinite(measured_temperature)
        & np.isfinite(measured_density)
        & (measured_temperature > 0.0)
        & (measured_density > 0.0)
    )
    if not np.any(selected):
        raise ValueError("prediction and measurement have no comparable channels")
    temperature = _comparison_metrics(
        predicted_temperature[selected], measured_temperature[selected]
    )
    density = _comparison_metrics(
        predicted_density[selected], measured_density[selected]
    )
    return ThomsonComparison(
        compared_channels=int(np.count_nonzero(selected)),
        temperature_rmse_ev=temperature[0],
        temperature_median_absolute_error_ev=temperature[1],
        temperature_normalised_rmse=temperature[2],
        density_rmse_m3=density[0],
        density_median_absolute_error_m3=density[1],
        density_normalised_rmse=density[2],
    )
