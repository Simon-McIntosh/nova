"""Audit the DINA flux-function mapping independently and replay its solve.

The stored equilibrium is DDv3, whose documented convention is COCOS 11.
Nova's forward-equilibrium convention is COCOS 17 and carries total poloidal
flux.  This benchmark composes the documented convention digits locally,
reads the stored IDS directly with imas-python, and derives the mapped flux,
pressure gradient and diamagnetic gradient before importing the reproduction
reader.  Comparing with that reader is therefore an audit result, not an input
to the derivation.

The mapped profiles then drive the same passive-inclusive 1,587-cell hex-mesh
solve.  Results are compared with the passive-inclusive current-tree baseline;
the active-only demonstration is a different physical configuration and is not
used here.

Usage::

    uv run python benchmarks/dina_reference_mapping.py \
      --output /tmp/dina-reference-mapping.json
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import getpass
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from scipy.constants import mu_0
from scipy.interpolate import RectBivariateSpline

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

PULSE = 135011
RUN = 7
TIME_SLICE = 353
DD_VERSION = "3.39.0"
MACHINE = "iter"
REFERENCE_CELLS = -1500
SOURCE_COCOS = 11
TARGET_COCOS = 17
RASTER_RADIUS_NODES = 401
RASTER_HEIGHT_NODES = 601
BASELINE_DEVIATION = {
    "plasma current percent": -1.124056611,
    "poloidal beta percent": 2.236992628,
    "internal inductance percent": -0.135265926,
    "axis position mm": 41.237233984,
}


@dataclass(frozen=True)
class ConventionDigits:
    """The four Sauter convention digits in their documented order."""

    sigma_bp: int
    flux_exponent: int
    sigma_r_phi_z: int
    sigma_rho_theta_phi: int


@dataclass(frozen=True)
class StoredProfiles:
    """Quantities read directly from one stored equilibrium slice."""

    user: str
    time: float
    plasma_current: float
    axis: np.ndarray
    reference_radius: float
    psi_axis: float
    psi_boundary: float
    psi_1d: np.ndarray
    pressure: np.ndarray
    p_prime: np.ndarray
    field_function: np.ndarray
    ff_prime: np.ndarray
    boundary: np.ndarray
    grid_radius: np.ndarray
    grid_height: np.ndarray
    grid_flux: np.ndarray


@dataclass(frozen=True)
class DerivedMapping:
    """Stored profiles expressed in Nova's forward-equilibrium convention."""

    psi_factor: float
    plasma_current_factor: float
    field_function_factor: float
    conventional_derivative_factor: float
    nova_gradient_factor: float
    flux_axis: float
    flux_boundary: float
    psi_1d: np.ndarray
    psi_norm: np.ndarray
    pressure: np.ndarray
    p_prime: np.ndarray
    field_function: np.ndarray
    ff_prime: np.ndarray
    grid_flux: np.ndarray


@dataclass(frozen=True)
class IndependentlyMappedCase:
    """Reference interface backed by the independently derived IDS mapping."""

    geometry: object
    stored: StoredProfiles
    mapped: DerivedMapping

    @property
    def time(self) -> float:
        return self.stored.time

    @property
    def plasma_current(self) -> float:
        return self.stored.plasma_current * self.mapped.plasma_current_factor

    @property
    def axis(self) -> np.ndarray:
        return self.stored.axis

    @property
    def reference_radius(self) -> float:
        return self.stored.reference_radius

    @property
    def flux_axis(self) -> float:
        return self.mapped.flux_axis

    @property
    def flux_boundary(self) -> float:
        return self.mapped.flux_boundary

    @property
    def flux_span(self) -> float:
        return self.flux_boundary - self.flux_axis

    @property
    def psi_norm(self) -> np.ndarray:
        return self.mapped.psi_norm

    @property
    def p_prime(self) -> np.ndarray:
        return self.mapped.p_prime

    @property
    def ff_prime(self) -> np.ndarray:
        return self.mapped.ff_prime

    @property
    def pressure(self) -> np.ndarray:
        return self.mapped.pressure

    @property
    def field_function(self) -> np.ndarray:
        return self.mapped.field_function

    @property
    def boundary(self) -> np.ndarray:
        return self.stored.boundary

    @property
    def wall(self) -> np.ndarray:
        return self.geometry.wall

    @property
    def active(self):
        return self.geometry.active

    @property
    def passive(self):
        return self.geometry.passive

    def drive(self, passive: bool = True):
        """Return the unchanged conductor inventory for one machine model."""
        return self.geometry.drive(passive)

    @property
    def spline(self) -> RectBivariateSpline:
        """Return the independently mapped total-flux interpolant."""
        return RectBivariateSpline(
            self.stored.grid_radius,
            self.stored.grid_height,
            self.mapped.grid_flux,
        )

    def flux(self, radius, height) -> np.ndarray:
        """Return mapped total poloidal flux at arbitrary coordinates."""
        return self.spline.ev(np.asarray(radius), np.asarray(height))

    def map_moments(self) -> dict[str, float]:
        """Integrate the mapped reference without calling the current reader."""
        from matplotlib.path import Path as PolygonPath

        radius = np.linspace(
            self.boundary[:, 0].min() - 0.05,
            self.boundary[:, 0].max() + 0.05,
            RASTER_RADIUS_NODES,
        )
        height = np.linspace(
            self.boundary[:, 1].min() - 0.05,
            self.boundary[:, 1].max() + 0.05,
            int(1.8 * RASTER_RADIUS_NODES),
        )
        grid_r, grid_z = np.meshgrid(radius, height, indexing="ij")
        cell_area = (radius[1] - radius[0]) * (height[1] - height[0])
        inside = (
            PolygonPath(self.boundary)
            .contains_points(np.c_[grid_r.ravel(), grid_z.ravel()])
            .reshape(grid_r.shape)
        )
        volume_element = np.where(inside, 2.0 * np.pi * grid_r * cell_area, 0.0)
        psi_norm = (self.flux(grid_r, grid_z) - self.flux_axis) / self.flux_span
        pressure = np.interp(np.clip(psi_norm, 0.0, 1.0), self.psi_norm, self.pressure)
        field_squared = (
            self.spline.ev(grid_r, grid_z, dx=1) ** 2
            + self.spline.ev(grid_r, grid_z, dy=1) ** 2
        ) / (2.0 * np.pi * grid_r) ** 2
        pressure_integral = float(np.sum(pressure * volume_element))
        field_integral = float(np.sum(field_squared * volume_element))
        reference = mu_0 * self.reference_radius * self.plasma_current**2
        return {
            "volume": float(volume_element.sum()),
            "pressure_integral": pressure_integral,
            "field_integral": field_integral,
            "poloidal_beta": 4.0 * pressure_integral / reference,
            "internal_inductance": 2.0 * field_integral / (mu_0 * reference),
        }


def _entry_uri(user: str) -> str:
    """Return the direct IMAS locator for one candidate owner."""
    return (
        f"imas:hdf5?user={user};pulse={PULSE};run={RUN};"
        f"database={MACHINE};version={DD_VERSION.split('.')[0]}"
    )


def read_stored_profiles() -> StoredProfiles:
    """Read the stored equilibrium directly through imas-python."""
    import imas

    failures = []
    for user in ("public", getpass.getuser()):
        try:
            entry = imas.DBEntry(_entry_uri(user), "r", dd_version=DD_VERSION)
            try:
                equilibrium = entry.get("equilibrium")
            finally:
                entry.close()
            break
        except Exception as error:  # noqa: BLE001 - availability audit
            failures.append(f"{user}: {type(error).__name__}")
    else:
        detail = "; ".join(failures)
        raise RuntimeError(f"DINA {PULSE}/{RUN} is unreachable ({detail})")

    slice_ = equilibrium.time_slice[TIME_SLICE]
    globals_ = slice_.global_quantities
    profiles = slice_.profiles_1d
    surface = slice_.profiles_2d[0]
    return StoredProfiles(
        user=user,
        time=float(np.asarray(equilibrium.time)[TIME_SLICE]),
        plasma_current=float(globals_.ip),
        axis=np.array(
            [float(globals_.magnetic_axis.r), float(globals_.magnetic_axis.z)]
        ),
        reference_radius=float(equilibrium.vacuum_toroidal_field.r0),
        psi_axis=float(globals_.psi_axis),
        psi_boundary=float(globals_.psi_boundary),
        psi_1d=np.asarray(profiles.psi),
        pressure=np.asarray(profiles.pressure),
        p_prime=np.asarray(profiles.dpressure_dpsi),
        field_function=np.asarray(profiles.f),
        ff_prime=np.asarray(profiles.f_df_dpsi),
        boundary=np.c_[
            np.asarray(slice_.boundary.outline.r),
            np.asarray(slice_.boundary.outline.z),
        ],
        grid_radius=np.asarray(surface.grid.dim1),
        grid_height=np.asarray(surface.grid.dim2),
        grid_flux=np.asarray(surface.psi),
    )


def derive_mapping(stored: StoredProfiles) -> DerivedMapping:
    """Compose documented COCOS digits and Nova's gradient definition."""
    documented = {
        11: ConventionDigits(+1, 1, +1, +1),
        17: ConventionDigits(-1, 1, +1, +1),
    }
    source = documented[SOURCE_COCOS]
    target = documented[TARGET_COCOS]
    psi_sign = source.sigma_bp * target.sigma_bp
    cylindrical_sign = source.sigma_r_phi_z * target.sigma_r_phi_z
    psi_factor = float(psi_sign * cylindrical_sign) * (2.0 * np.pi) ** (
        target.flux_exponent - source.flux_exponent
    )
    plasma_current_factor = float(cylindrical_sign)
    field_function_factor = float(cylindrical_sign)
    conventional_derivative_factor = field_function_factor**2 / psi_factor
    # Nova stores the NEGATIVE derivative with respect to its total flux.
    nova_gradient_factor = -conventional_derivative_factor
    psi_1d = psi_factor * stored.psi_1d
    flux_axis = psi_factor * stored.psi_axis
    flux_boundary = psi_factor * stored.psi_boundary
    return DerivedMapping(
        psi_factor=psi_factor,
        plasma_current_factor=plasma_current_factor,
        field_function_factor=field_function_factor,
        conventional_derivative_factor=conventional_derivative_factor,
        nova_gradient_factor=nova_gradient_factor,
        flux_axis=flux_axis,
        flux_boundary=flux_boundary,
        psi_1d=psi_1d,
        psi_norm=(psi_1d - flux_axis) / (flux_boundary - flux_axis),
        pressure=np.array(stored.pressure, copy=True),
        p_prime=nova_gradient_factor * stored.p_prime,
        field_function=field_function_factor * stored.field_function,
        ff_prime=nova_gradient_factor * stored.ff_prime,
        grid_flux=psi_factor * stored.grid_flux,
    )


def _difference(derived, current) -> dict[str, float | bool]:
    """Return signed and absolute differences for one mapped quantity."""
    delta = np.asarray(derived) - np.asarray(current)
    return {
        "bit_identical": bool(
            np.array_equal(np.asarray(derived), np.asarray(current), equal_nan=True)
        ),
        "signed_minimum": float(np.min(delta)),
        "signed_maximum": float(np.max(delta)),
        "signed_mean": float(np.mean(delta)),
        "maximum_absolute": float(np.max(np.abs(delta))),
    }


def mapping_comparison(
    current, mapped: DerivedMapping
) -> dict[str, dict[str, float | bool]]:
    """Compare the independent derivation with the current reader afterward."""
    return {
        "axis total flux Wb": _difference(mapped.flux_axis, current.flux_axis),
        "boundary total flux Wb": _difference(
            mapped.flux_boundary, current.flux_boundary
        ),
        "normalised flux": _difference(mapped.psi_norm, current.psi_norm),
        "pressure gradient Pa per Wb": _difference(mapped.p_prime, current.p_prime),
        "FF prime tesla squared metre squared per Wb": _difference(
            mapped.ff_prime, current.ff_prime
        ),
        "total flux map Wb": _difference(mapped.grid_flux, -current.grid_flux),
    }


def current_convention_check(case: IndependentlyMappedCase) -> dict[str, float]:
    """Distinguish total flux from flux per radian by integrated current."""
    from matplotlib.path import Path as PolygonPath

    radius = np.linspace(
        case.boundary[:, 0].min(),
        case.boundary[:, 0].max(),
        RASTER_RADIUS_NODES,
    )
    height = np.linspace(
        case.boundary[:, 1].min(),
        case.boundary[:, 1].max(),
        RASTER_HEIGHT_NODES,
    )
    grid_r, grid_z = np.meshgrid(radius, height, indexing="ij")
    cell_area = (radius[1] - radius[0]) * (height[1] - height[0])
    inside = (
        PolygonPath(case.boundary)
        .contains_points(np.c_[grid_r.ravel(), grid_z.ravel()])
        .reshape(grid_r.shape)
    )
    psi_norm = np.clip(
        (case.flux(grid_r, grid_z) - case.flux_axis) / case.flux_span, 0.0, 1.0
    )
    p_prime = np.interp(psi_norm, case.psi_norm, case.p_prime)
    ff_prime = np.interp(psi_norm, case.psi_norm, case.ff_prime)
    density = -2.0 * np.pi * (grid_r * p_prime + ff_prime / (mu_0 * grid_r))
    total_flux_current = float(np.sum(np.where(inside, density, 0.0)) * cell_area)
    per_radian_current = total_flux_current / (2.0 * np.pi)
    expected = case.plasma_current
    return {
        "stored plasma current A": expected,
        "total flux candidate current A": total_flux_current,
        "total flux candidate relative difference": total_flux_current / expected - 1.0,
        "per radian candidate current A": per_radian_current,
        "per radian candidate relative difference": per_radian_current / expected - 1.0,
    }


def _axis_distance(deviation: dict[str, float]) -> float:
    """Return magnetic-axis displacement magnitude in millimetres."""
    return float(math.hypot(deviation["axis radius"], deviation["axis height"]))


def metric_rows(deviation: dict[str, float]) -> dict[str, dict[str, float]]:
    """Compare the independently mapped solve with the passive baseline."""
    solved = {
        "plasma current percent": deviation["plasma current"],
        "poloidal beta percent": deviation["poloidal beta"],
        "internal inductance percent": deviation["internal inductance"],
        "axis position mm": _axis_distance(deviation),
    }
    return {
        name: {
            "passive inclusive baseline": baseline,
            "independently mapped": solved[name],
            "signed change": solved[name] - baseline,
            "response relative to remaining deviation percent": (
                100.0 * abs(solved[name] - baseline) / abs(baseline)
            ),
        }
        for name, baseline in BASELINE_DEVIATION.items()
    }


def run(cells: int = REFERENCE_CELLS) -> dict:
    """Derive, audit and solve the independently mapped reference."""
    stored = read_stored_profiles()
    mapped = derive_mapping(stored)

    from nova.jax.config import configure_dtypes

    configure_dtypes()
    from tests import test_equilibrium_forward_reference as reference

    current = reference.reference_case()
    if isinstance(current, str):
        raise RuntimeError(f"the current reference reader is unavailable: {current}")
    case = IndependentlyMappedCase(current, stored, mapped)
    comparison = mapping_comparison(current, mapped)
    mapping_bit_identical = all(row["bit_identical"] for row in comparison.values())

    start = time.perf_counter()
    machine = reference.build_machine(case, cells, passive=True)
    assembly_seconds = time.perf_counter() - start
    if len(machine.node) != 1587:
        raise RuntimeError(f"the fixed production mesh has {len(machine.node)} cells")

    start = time.perf_counter()
    solved = reference.solve(case, machine)
    solve_seconds = time.perf_counter() - start
    deviation = solved.deviations()
    metrics = metric_rows(deviation)
    largest_response = max(
        row["response relative to remaining deviation percent"]
        for row in metrics.values()
    )
    if mapping_bit_identical:
        verdict = (
            "The independently derived profile and total-flux mapping is "
            "bit-identical to the current reader and is excluded as a cause. "
            "No named cause remains for the passive-inclusive reproduction "
            f"deviation; the largest replay response is {largest_response:.6g}% "
            "of a baseline quantity."
        )
    else:
        verdict = (
            "The independent mapping differs from the current reader; its largest "
            f"measured solve response is {largest_response:.6g}% of a passive-"
            "inclusive baseline quantity."
        )
    return {
        "reference": {
            "pulse": PULSE,
            "run": RUN,
            "time slice": TIME_SLICE,
            "time s": stored.time,
            "dd version": DD_VERSION,
            "source convention": SOURCE_COCOS,
            "target convention": TARGET_COCOS,
            "mesh cells": len(machine.node),
            "passive conductors included": True,
            "solve route": "fixed-budget Newton-Krylov",
        },
        "derived convention factors": {
            "total flux": mapped.psi_factor,
            "plasma current": mapped.plasma_current_factor,
            "field function": mapped.field_function_factor,
            "conventional derivative": mapped.conventional_derivative_factor,
            "Nova negative gradient": mapped.nova_gradient_factor,
        },
        "current convention check": current_convention_check(case),
        "current reader comparison": comparison,
        "mapping bit identical": mapping_bit_identical,
        "metrics": metrics,
        "axis components mm": {
            "radius": deviation["axis radius"],
            "height": deviation["axis height"],
        },
        "timing seconds": {
            "machine assembly": assembly_seconds,
            "solve": solve_seconds,
        },
        "verdict": verdict,
    }


def main() -> None:
    """Run the fixed mapping audit and write its JSON receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, default=REFERENCE_CELLS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.cells)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["metrics"], indent=2))
    print(result["verdict"])


if __name__ == "__main__":
    main()
