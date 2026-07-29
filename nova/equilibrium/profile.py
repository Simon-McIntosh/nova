r"""JAX-native profile reconstruction from external magnetic measurements.

The plasma current is carried by two named flux-function families: pressure
drive and diamagnetic drive.  Each Picard sweep reads the axis-connected core
with Nova's fixed-shape flood-fill and smooth boundary-push topology, evaluates
the profile basis there, and maps its cell currents through precomputed Green
operators.  The same map supports prescribed-coefficient Picard iteration and
an equality-constrained least-squares reconstruction.

All Green operators are assembled by :func:`nova.biot.greens.hybrid_greens`.
They carry total poloidal flux :math:`\Phi = 2\pi R A_\phi` in Wb per ampere
and poloidal field in T per ampere.  The iterative path is pure JAX, with fixed
shapes throughout, so callers may ``jit`` one solve or ``vmap`` a leading
shot/time axis.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy.constants import mu_0

from nova.biot.greens import hybrid_greens
from nova.equilibrium.measurement import Magnetics
from nova.jax.connectivity_boundary import boundary_read_smooth_jax

jax.config.update("jax_enable_x64", True)


class ProfileResult(NamedTuple):
    """Fixed-shape result of a profile reconstruction."""

    flux: jax.Array
    cell_current: jax.Array
    coefficients: jax.Array
    residual: jax.Array
    axis: jax.Array
    boundary_flux: jax.Array
    core_weight: jax.Array


@dataclass(frozen=True)
class ProfileDegrees:
    """Named sizes of the pressure and diamagnetic profile families."""

    n_pressure: int
    n_diamagnetic: int

    def __post_init__(self):
        """Validate that both physical families are represented."""
        if self.n_pressure < 1:
            raise ValueError("n_pressure must be at least one")
        if self.n_diamagnetic < 1:
            raise ValueError("n_diamagnetic must be at least one")

    @property
    def names(self) -> tuple[str, ...]:
        """Return stable physical names in coefficient-column order."""
        pressure = tuple(f"pressure_{order}" for order in range(self.n_pressure))
        diamagnetic = tuple(
            f"diamagnetic_{order}" for order in range(self.n_diamagnetic)
        )
        return pressure + diamagnetic

    @property
    def number(self) -> int:
        """Return the total profile coefficient count."""
        return self.n_pressure + self.n_diamagnetic


@dataclass(frozen=True)
class ProfilePrior:
    """One covariance-weighted linear prior on named profile coefficients.

    ``sensitivity`` maps physical coefficient names to one moment or profile
    observable.  This represents the donor moment-prior contract directly:
    total-current, beta-plus-inductance, centroid, pressure-gradient, annulus,
    or temporal sensitivities are ordinary rows with their own targets and
    covariance.  Priors compose by passing any sequence of rows to a solve.
    """

    name: str
    sensitivity: Mapping[str, float]
    target: float
    sigma: float

    def row(self, coefficient_names: Sequence[str]) -> np.ndarray:
        """Return the whitened row in the solver's coefficient order."""
        if not self.name:
            raise ValueError("prior name must not be empty")
        if not np.isfinite(self.sigma) or self.sigma <= 0.0:
            raise ValueError(f"prior {self.name!r} sigma must be positive")
        unknown = set(self.sensitivity).difference(coefficient_names)
        if unknown:
            joined = ", ".join(sorted(unknown))
            raise ValueError(f"prior {self.name!r} has unknown coefficients: {joined}")
        return np.asarray(
            [
                self.sensitivity.get(name, 0.0) / self.sigma
                for name in coefficient_names
            ],
            dtype=np.float64,
        )

    @property
    def whitened_target(self) -> float:
        """Return the target divided by its standard deviation."""
        return float(self.target) / float(self.sigma)


def _prior_rows(
    priors: Sequence[ProfilePrior], coefficient_names: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Stack composable prior rows, preserving an empty fixed-width matrix."""
    if not priors:
        return np.zeros((0, len(coefficient_names))), np.zeros(0)
    return (
        np.stack([prior.row(coefficient_names) for prior in priors]),
        np.asarray([prior.whitened_target for prior in priors]),
    )


def _green_columns(
    target_r: np.ndarray,
    target_z: np.ndarray,
    source_r: np.ndarray,
    source_z: np.ndarray,
    source_width: np.ndarray,
    source_height: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return target-by-source flux, radial-field and vertical-field operators."""
    with np.errstate(divide="ignore", invalid="ignore"):
        columns = [
            hybrid_greens(
                target_r,
                target_z,
                float(radius),
                float(height),
                float(width),
                float(thickness),
            )
            for radius, height, width, thickness in zip(
                source_r, source_z, source_width, source_height, strict=True
            )
        ]
    if not columns:
        shape = (target_r.size, 0)
        empty = np.zeros(shape, dtype=np.float64)
        return empty, empty.copy(), empty.copy()
    flux, radial, vertical = zip(*columns, strict=True)
    return np.stack(flux, axis=1), np.stack(radial, axis=1), np.stack(vertical, axis=1)


@dataclass
class ReconstructProfile:
    """Reconstruct a force-balanced profile on one fixed machine geometry.

    Geometry-dependent Green operators are immutable campaign data.  Construct
    them with :meth:`from_geometry`, then reuse this object for every slice
    sharing the same physical machine configuration.

    ``source_names`` label the externally supplied conductor controls.
    :meth:`pack_source_currents` and :meth:`pack_coefficients` provide strict
    named host adapters; the numerical methods receive their packed arrays so
    their traced signatures remain fixed-shape.
    """

    grid_r: np.ndarray
    grid_z: np.ndarray
    inside_limiter: np.ndarray
    cell_area: np.ndarray
    source_to_grid: np.ndarray
    plasma_to_grid: np.ndarray
    source_to_sensor: np.ndarray
    plasma_to_sensor: np.ndarray
    source_names: tuple[str, ...]
    degrees: ProfileDegrees
    axis_seed: tuple[float, float]
    wall_r: np.ndarray
    wall_z: np.ndarray
    priors: tuple[ProfilePrior, ...] = ()
    iterations: int = 8
    relaxation: float = 0.6
    ridge: float = 1.0e-10
    topology_temperature: float = 1.0e-3
    topology_levels: int = 48
    topology_bisections: int = 12
    topology_rays: int = 128

    def __post_init__(self):
        """Validate fixed shapes and place numerical state on the JAX device."""
        grid_r = np.asarray(self.grid_r, dtype=np.float64)
        grid_z = np.asarray(self.grid_z, dtype=np.float64)
        inside = np.asarray(self.inside_limiter, dtype=bool)
        n_grid = grid_r.size * grid_z.size
        n_source = len(self.source_names)
        if grid_r.ndim != 1 or grid_z.ndim != 1:
            raise ValueError("grid_r and grid_z must be one-dimensional")
        if inside.shape != (grid_z.size, grid_r.size):
            raise ValueError("inside_limiter shape must be (len(grid_z), len(grid_r))")
        shapes = {
            "cell_area": (n_grid,),
            "source_to_grid": (n_grid, n_source),
            "plasma_to_grid": (n_grid, n_grid),
        }
        for name, shape in shapes.items():
            if np.asarray(getattr(self, name)).shape != shape:
                raise ValueError(f"{name} shape must be {shape}")
        n_sensor = np.asarray(self.plasma_to_sensor).shape[0]
        if np.asarray(self.plasma_to_sensor).shape != (n_sensor, n_grid):
            raise ValueError("plasma_to_sensor shape must be (n_sensor, n_grid)")
        if np.asarray(self.source_to_sensor).shape != (n_sensor, n_source):
            raise ValueError("source_to_sensor shape must be (n_sensor, n_source)")
        if len(set(self.source_names)) != n_source:
            raise ValueError("source_names must be unique")
        if self.iterations < 1:
            raise ValueError("iterations must be at least one")
        if not 0.0 <= self.relaxation <= 1.0:
            raise ValueError("relaxation must be in [0, 1]")
        if self.ridge < 0.0:
            raise ValueError("ridge must be non-negative")
        if self.topology_temperature <= 0.0:
            raise ValueError("topology_temperature must be positive")
        prior_matrix, prior_target = _prior_rows(self.priors, self.coefficient_names)
        for name in (
            "grid_r",
            "grid_z",
            "inside_limiter",
            "cell_area",
            "source_to_grid",
            "plasma_to_grid",
            "source_to_sensor",
            "plasma_to_sensor",
            "wall_r",
            "wall_z",
        ):
            setattr(self, name, jnp.asarray(getattr(self, name)))
        self._prior_matrix = jnp.asarray(prior_matrix)
        self._prior_target = jnp.asarray(prior_target)
        radius, height = jnp.meshgrid(self.grid_r, self.grid_z)
        self._cell_r = radius.reshape(-1)
        self._cell_z = height.reshape(-1)

    @classmethod
    def from_geometry(
        cls,
        *,
        grid_r: np.ndarray,
        grid_z: np.ndarray,
        inside_limiter: np.ndarray,
        cell_width: np.ndarray,
        cell_height: np.ndarray,
        source_r: np.ndarray,
        source_z: np.ndarray,
        source_width: np.ndarray,
        source_height: np.ndarray,
        source_names: Sequence[str],
        magnetics: Magnetics,
        degrees: ProfileDegrees,
        axis_seed: tuple[float, float],
        wall_r: np.ndarray,
        wall_z: np.ndarray,
        **kwargs,
    ) -> ReconstructProfile:
        """Build all campaign Green operators through Nova's canonical kernels."""
        grid_r = np.asarray(grid_r, dtype=np.float64)
        grid_z = np.asarray(grid_z, dtype=np.float64)
        radius, height = np.meshgrid(grid_r, grid_z)
        cell_r = radius.ravel()
        cell_z = height.ravel()
        cell_width = np.broadcast_to(
            np.asarray(cell_width, dtype=np.float64), cell_r.shape
        )
        cell_height = np.broadcast_to(
            np.asarray(cell_height, dtype=np.float64), cell_r.shape
        )
        source_r = np.asarray(source_r, dtype=np.float64)
        source_z = np.asarray(source_z, dtype=np.float64)
        source_width = np.broadcast_to(
            np.asarray(source_width, dtype=np.float64), source_r.shape
        )
        source_height = np.broadcast_to(
            np.asarray(source_height, dtype=np.float64), source_r.shape
        )
        source_grid = _green_columns(
            cell_r,
            cell_z,
            source_r,
            source_z,
            source_width,
            source_height,
        )[0]
        plasma_grid = _green_columns(
            cell_r,
            cell_z,
            cell_r,
            cell_z,
            cell_width,
            cell_height,
        )[0]
        source_sensor_fields = _green_columns(
            np.asarray(magnetics.r, dtype=np.float64),
            np.asarray(magnetics.z, dtype=np.float64),
            source_r,
            source_z,
            source_width,
            source_height,
        )
        plasma_sensor_fields = _green_columns(
            np.asarray(magnetics.r, dtype=np.float64),
            np.asarray(magnetics.z, dtype=np.float64),
            cell_r,
            cell_z,
            cell_width,
            cell_height,
        )
        source_sensor = magnetics.project(*source_sensor_fields)
        plasma_sensor = magnetics.project(*plasma_sensor_fields)
        return cls(
            grid_r=grid_r,
            grid_z=grid_z,
            inside_limiter=inside_limiter,
            cell_area=cell_width * cell_height,
            source_to_grid=source_grid,
            plasma_to_grid=plasma_grid,
            source_to_sensor=source_sensor,
            plasma_to_sensor=plasma_sensor,
            source_names=tuple(source_names),
            degrees=degrees,
            axis_seed=axis_seed,
            wall_r=wall_r,
            wall_z=wall_z,
            **kwargs,
        )

    @property
    def coefficient_names(self) -> tuple[str, ...]:
        """Return profile coefficient names in traced column order."""
        return self.degrees.names

    def _pack_named(
        self, values: Mapping[str, float], names: Sequence[str], kind: str
    ) -> jax.Array:
        """Validate and pack one named physical control mapping."""
        missing = set(names).difference(values)
        unknown = set(values).difference(names)
        if missing or unknown:
            details = []
            if missing:
                details.append("missing " + ", ".join(sorted(missing)))
            if unknown:
                details.append("unknown " + ", ".join(sorted(unknown)))
            raise ValueError(f"{kind} names do not match: {'; '.join(details)}")
        return jnp.asarray([values[name] for name in names], dtype=jnp.float64)

    def pack_source_currents(self, values: Mapping[str, float]) -> jax.Array:
        """Pack named conductor currents [A] into the traced source order."""
        return self._pack_named(values, self.source_names, "source current")

    def pack_coefficients(self, values: Mapping[str, float]) -> jax.Array:
        """Pack named pressure/diamagnetic coefficients into column order."""
        return self._pack_named(values, self.coefficient_names, "profile coefficient")

    def initial_flux(
        self, source_current: jax.Array, plasma_current: jax.Array
    ) -> jax.Array:
        """Return a compact current seed added to the known-conductor field."""
        scale_r = jnp.maximum(0.2 * (self.grid_r[-1] - self.grid_r[0]), 1.0e-3)
        scale_z = jnp.maximum(0.2 * (self.grid_z[-1] - self.grid_z[0]), 1.0e-3)
        seed = jnp.exp(
            -(
                ((self._cell_r - self.axis_seed[0]) / scale_r) ** 2
                + ((self._cell_z - self.axis_seed[1]) / scale_z) ** 2
            )
        )
        seed = seed * self.inside_limiter.reshape(-1)
        seed = seed * plasma_current / jnp.maximum(jnp.sum(seed), 1.0e-30)
        return self.source_to_grid @ source_current + self.plasma_to_grid @ seed

    def _profile_basis(self, flux: jax.Array) -> tuple[jax.Array, dict[str, jax.Array]]:
        """Return per-coefficient cell-current columns and the topology read."""
        flux2d = flux.reshape(self.grid_z.size, self.grid_r.size)
        topology = boundary_read_smooth_jax(
            flux2d,
            self.grid_r,
            self.grid_z,
            self.inside_limiter,
            jnp.asarray(self.axis_seed[0]),
            jnp.asarray(self.axis_seed[1]),
            self.topology_levels,
            self.topology_bisections,
            self.topology_rays,
            wall_r=self.wall_r,
            wall_z=self.wall_z,
            temperature=jnp.asarray(self.topology_temperature),
        )
        span = topology["psi_bnd"] - topology["psi_axis"]
        span = jnp.where(jnp.abs(span) > 1.0e-30, span, 1.0e-30)
        flux_norm = (flux - topology["psi_axis"]) / span
        edge = jnp.clip(1.0 - flux_norm, 0.0, 1.0)
        weight = topology["core_weight"].reshape(-1) * self.cell_area
        radius = jnp.maximum(self._cell_r, 1.0e-6)
        pressure = [
            -2.0 * jnp.pi * radius * edge ** (order + 1) * weight
            for order in range(self.degrees.n_pressure)
        ]
        diamagnetic = [
            -2.0 * jnp.pi / (mu_0 * radius) * edge ** (order + 1) * weight
            for order in range(self.degrees.n_diamagnetic)
        ]
        return jnp.stack(pressure + diamagnetic, axis=1), topology

    def _least_squares_coefficients(
        self,
        basis: jax.Array,
        source_current: jax.Array,
        plasma_current: jax.Array,
        measured: jax.Array,
        scale: jax.Array,
        mask: jax.Array,
    ) -> jax.Array:
        """Solve one whitened sweep with a hard measured-current equality."""
        response = self.plasma_to_sensor @ basis
        target = measured - self.source_to_sensor @ source_current
        safe_scale = jnp.maximum(jnp.abs(scale), 1.0e-30)
        weight = jnp.where(mask & jnp.isfinite(measured), 1.0 / safe_scale, 0.0)
        data_matrix = response * weight[:, None]
        data_target = jnp.nan_to_num(target) * weight
        matrix = jnp.concatenate([data_matrix, self._prior_matrix], axis=0)
        rhs = jnp.concatenate([data_target, self._prior_target], axis=0)
        gram = matrix.T @ matrix + self.ridge * jnp.eye(self.degrees.number)
        vector = matrix.T @ rhs
        current_row = jnp.sum(basis, axis=0)
        kkt, constrained_rhs, coefficient_scale = self._scaled_kkt(
            gram, vector, current_row, plasma_current
        )
        solution = jnp.linalg.solve(kkt, constrained_rhs)
        return solution[:-1] / coefficient_scale

    def _scaled_kkt(
        self,
        gram: jax.Array,
        vector: jax.Array,
        current_row: jax.Array,
        plasma_current: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Equilibrate the constrained normal equations by congruence.

        Profile families have different physical units, so their unscaled KKT
        matrix can be poorly conditioned even when the constrained fit is
        identifiable.  Congruence scaling preserves the exact minimizer and
        measured-current equality while making scalar and batched
        factorizations comparably stable.
        """
        coefficient_scale = jnp.sqrt(jnp.diag(gram))
        coefficient_scale = jnp.maximum(
            coefficient_scale, jnp.sqrt(jnp.finfo(gram.dtype).tiny)
        )
        scaled_gram = gram / (coefficient_scale[:, None] * coefficient_scale[None, :])
        scaled_vector = vector / coefficient_scale
        scaled_current = current_row / coefficient_scale
        equality_scale = jnp.linalg.norm(scaled_current)
        equality_scale = jnp.maximum(
            equality_scale, jnp.sqrt(jnp.finfo(gram.dtype).tiny)
        )
        scaled_current = scaled_current / equality_scale
        kkt = jnp.block(
            [
                [scaled_gram, scaled_current[:, None]],
                [scaled_current[None, :], jnp.zeros((1, 1))],
            ]
        )
        rhs = jnp.concatenate([scaled_vector, plasma_current[None] / equality_scale])
        return kkt, rhs, coefficient_scale

    def _result(
        self,
        flux: jax.Array,
        previous_flux: jax.Array,
        basis: jax.Array,
        coefficients: jax.Array,
        topology: dict[str, jax.Array],
    ) -> ProfileResult:
        cell_current = basis @ coefficients
        residual = jnp.max(jnp.abs(flux - previous_flux)) / jnp.maximum(
            jnp.max(jnp.abs(flux)), 1.0e-30
        )
        return ProfileResult(
            flux=flux,
            cell_current=cell_current,
            coefficients=coefficients,
            residual=residual,
            axis=jnp.asarray(self.axis_seed),
            boundary_flux=topology["psi_bnd"],
            core_weight=topology["core_weight"],
        )

    def picard(
        self,
        source_current: jax.Array,
        coefficients: jax.Array,
        initial_flux: jax.Array,
    ) -> ProfileResult:
        """Iterate the force-balanced map for prescribed named coefficients."""

        def sweep(flux, _):
            basis, _topology = self._profile_basis(flux)
            mapped = self.source_to_grid @ source_current + self.plasma_to_grid @ (
                basis @ coefficients
            )
            updated = self.relaxation * mapped + (1.0 - self.relaxation) * flux
            return updated, flux

        flux, history = jax.lax.scan(sweep, initial_flux, None, length=self.iterations)
        basis, topology = self._profile_basis(flux)
        return self._result(flux, history[-1], basis, coefficients, topology)

    def least_squares(
        self,
        source_current: jax.Array,
        plasma_current: jax.Array,
        measured: jax.Array,
        scale: jax.Array,
        mask: jax.Array,
        initial_flux: jax.Array,
    ) -> ProfileResult:
        """Alternate named-profile least squares with the boundary-push map."""

        def sweep(flux, _):
            basis, _topology = self._profile_basis(flux)
            coefficients = self._least_squares_coefficients(
                basis, source_current, plasma_current, measured, scale, mask
            )
            mapped = self.source_to_grid @ source_current + self.plasma_to_grid @ (
                basis @ coefficients
            )
            updated = self.relaxation * mapped + (1.0 - self.relaxation) * flux
            return updated, (flux, coefficients)

        flux, history = jax.lax.scan(sweep, initial_flux, None, length=self.iterations)
        previous_flux, _previous_coefficients = history
        basis, topology = self._profile_basis(flux)
        coefficients = self._least_squares_coefficients(
            basis, source_current, plasma_current, measured, scale, mask
        )
        return self._result(flux, previous_flux[-1], basis, coefficients, topology)

    def least_squares_batch(
        self,
        source_current: jax.Array,
        plasma_current: jax.Array,
        measured: jax.Array,
        scale: jax.Array,
        mask: jax.Array,
        initial_flux: jax.Array,
    ) -> ProfileResult:
        """Map :meth:`least_squares` over a leading shot/time axis."""
        return jax.vmap(self.least_squares)(
            source_current, plasma_current, measured, scale, mask, initial_flux
        )

    def solve(
        self,
        source_current: jax.Array,
        plasma_current: jax.Array,
        measured: jax.Array,
        scale: jax.Array,
        mask: jax.Array,
        initial_flux: jax.Array,
    ) -> ProfileResult:
        """Run the default least-squares profile reconstruction."""
        return self.least_squares(
            source_current,
            plasma_current,
            measured,
            scale,
            mask,
            initial_flux,
        )

    def solve_batch(
        self,
        source_current: jax.Array,
        plasma_current: jax.Array,
        measured: jax.Array,
        scale: jax.Array,
        mask: jax.Array,
        initial_flux: jax.Array,
    ) -> ProfileResult:
        """Run :meth:`solve` over a leading shot/time axis."""
        return self.least_squares_batch(
            source_current,
            plasma_current,
            measured,
            scale,
            mask,
            initial_flux,
        )


__all__ = [
    "ProfileDegrees",
    "ProfilePrior",
    "ProfileResult",
    "ReconstructProfile",
]
